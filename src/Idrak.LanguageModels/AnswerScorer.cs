// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Generation;
using Idrak.Layers;

namespace Idrak.LanguageModels;

/// <summary>How <see cref="AnswerScorer.Choose"/> rated the candidate answers to one prompt.</summary>
/// <param name="Best">Index of the most likely answer.</param>
/// <param name="Probabilities">Each answer's probability, the softmax of <paramref name="LogLikelihoods"/> over the answers (even when the prompt could not be scored).</param>
/// <param name="LogLikelihoods">log p(answer | prompt) of each answer (NaN when the prompt could not be scored).</param>
public sealed record AnswerChoice(int Best, IReadOnlyList<double> Probabilities, IReadOnlyList<double> LogLikelihoods);

/// <summary>
/// Scores given answers to chat prompts with a (tuned) chat model: log p(answer | prompt), the probability of the answer's
/// tokens (and of the turn's end) after the prompt, rendered with the model's own chat template and tokenizer (any model
/// the library loads). For classifiers and multiple-choice questions whose answer is one of a known list: the answer
/// always comes from the list, with every candidate's probability, and nothing is generated.
/// </summary>
/// <remarks>
/// A pass takes up to <see cref="PromptsPerPass"/> prompts of similar length (and <see cref="PositionsPerPass"/> positions). Each prompt runs once for all its answers (a
/// key/value cache, then only the answers' tokens; see <see cref="SharePrompts"/> and <see cref="Precision"/>), and the
/// output layer and the log-softmax run on the device on the answer positions only (<see cref="Losses.TokenLogProbabilities"/>).
/// A prompt too long for <see cref="MaxLength"/> has its last user message shortened (its start kept) until it fits with
/// the longest answer, so every answer is scored on the same text.
/// </remarks>
/// <example>
/// <code>
/// var scorer = new AnswerScorer(model, maxLength: 256);
/// var choice = scorer.Choose([[new ChatMessage("system", instruction), new ChatMessage("user", message)]], ["yes", "no"])[0];
/// string answer = choice.Best == 0 ? "yes" : "no";
/// </code>
/// </example>
public sealed class AnswerScorer
{
    private readonly PretrainedModel _model;
    private readonly List<Module> _modules;
    private readonly TextGenerator _batches;

    /// <summary>Creates a scorer for <paramref name="model"/> (which needs a chat template and a tokenizer).</summary>
    /// <param name="model">The chat model (in evaluation mode, e.g. from <see cref="TuningManifest.LoadModel"/>).</param>
    /// <param name="maxLength">Longest prompt plus answer in tokens (the tuning's maximum length).</param>
    public AnswerScorer(PretrainedModel model, int maxLength = 2048)
    {
        ArgumentNullException.ThrowIfNull(model);
        _model = model;
        Encoder = new ChatTranscriptEncoder(model.ChatTemplate ?? throw new InvalidOperationException("The model has no chat template."),
            model.Tokenizer ?? throw new InvalidOperationException("The model has no tokenizer."));
        MaxLength = Math.Max(2, Math.Min(maxLength, model.MaxPositions));
        _modules = [.. model.Network];
        _batches = new TextGenerator(model.Network, model.Tokenizer, model.MaxPositions);
    }

    /// <summary>The encoder rendering prompts and answers with the model's chat template.</summary>
    public ChatTranscriptEncoder Encoder { get; }

    /// <summary>Longest prompt plus answer in tokens.</summary>
    public int MaxLength { get; }

    /// <summary>Most prompts per device pass (each with a row per answer). 32 by default.</summary>
    public int PromptsPerPass { get; init; } = 32;

    /// <summary>
    /// Most token positions a pass runs through the network at once (rows times their padded length; a pass has at least
    /// one prompt): bounds the memory of a pass of long prompts. 8192 by default.
    /// </summary>
    public int PositionsPerPass { get; init; } = 8192;

    /// <summary>
    /// Run each prompt once for all its answers (from a key/value cache), where the model's attention can batch rows of
    /// different lengths (<see cref="TextGenerator.SupportsBatches"/>); otherwise, or when false, every (prompt, answer)
    /// row runs the whole prompt. Both give the same scores. On by default.
    /// </summary>
    public bool SharePrompts { get; init; } = true;

    /// <summary>
    /// The precision of the matrix products and attention while scoring. Null (the default): bfloat16 tensor cores where
    /// the model's GPU has them (the precision bfloat16 weights are multiplied in anyway, and what attention over rows of
    /// different lengths needs to run each prompt once), else the thread's <see cref="MixedPrecision.Current"/>.
    /// </summary>
    public MatMulPrecision? Precision { get; init; }

    /// <summary>The precision scoring uses (see <see cref="Precision"/>).</summary>
    public MatMulPrecision EffectivePrecision => Precision
        ?? (MixedPrecision.Current == MatMulPrecision.Float32 && MixedPrecision.TensorCoresUnavailable(_model.Device) is null
            ? MatMulPrecision.BFloat16
            : MixedPrecision.Current);

    /// <summary>
    /// Whether each prompt runs once for all its answers (<see cref="SharePrompts"/> and a model whose attention batches
    /// rows of different lengths from a cache at <see cref="EffectivePrecision"/>).
    /// </summary>
    public bool SharesPrompts
    {
        get
        {
            using var precision = MixedPrecision.Use(EffectivePrecision);
            return SharePrompts && _batches.SupportsBatches;
        }
    }

    /// <summary>
    /// <paramref name="prompt"/> as it is scored: its last user message shortened (its start kept) so that the prompt with
    /// the longest of <paramref name="answers"/> fits in <see cref="MaxLength"/>; null when even an empty message does not.
    /// </summary>
    public IReadOnlyList<ChatMessage>? Fit(IReadOnlyList<ChatMessage> prompt, IReadOnlyList<string> answers)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        string longest = Longest(answers);
        return Fitted(prompt, longest);
    }

    /// <summary>
    /// log p(answer | prompt) for each prompt (rows) and answer (columns): the sum of the log-probabilities of the answer's
    /// tokens and of the end of the turn. NaN for a prompt that cannot be fitted (see <see cref="Fit"/>).
    /// <paramref name="progress"/> gets the number of prompts scored so far.
    /// </summary>
    public double[][] LogLikelihoods(IReadOnlyList<IReadOnlyList<ChatMessage>> prompts, IReadOnlyList<string> answers, Action<int>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(prompts);
        ArgumentNullException.ThrowIfNull(answers);
        if (answers.Count == 0)
        {
            throw new ArgumentException("There must be at least one answer to score.", nameof(answers));
        }

        string longest = Longest(answers);
        bool share = SharesPrompts;
        using var precision = MixedPrecision.Use(EffectivePrecision);
        var results = new double[prompts.Count][];
        var order = Enumerable.Range(0, prompts.Count).OrderBy(i => prompts[i].Sum(m => m.Content.Length)).ToArray();
        int perPass = Math.Max(1, PromptsPerPass), budget = Math.Max(1, PositionsPerPass), done = 0;
        for (int first = 0; first < order.Length;)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // A row per (prompt, answer): the fitted prompt with the answer as the assistant's turn. Prompts join the pass
            // while it holds fewer than PromptsPerPass of them and PositionsPerPass positions (at least one prompt).
            var indices = new List<int>();
            var sequences = new List<TrainingSequence?>();
            int longestRow = 0;
            while (first < order.Length && indices.Count < perPass)
            {
                int i = order[first];
                var fitted = Fitted(prompts[i], longest);
                var rows = answers.Select(a => fitted is null ? null : Encoder.Encode(Transcript(fitted, a), MaxLength)).ToList();
                int length = Math.Max(longestRow, rows.Max(r => r?.Tokens.Length ?? 1));
                int needed = (indices.Count + 1) * length * (share ? 1 : answers.Count);   // rows are padded to the longest
                if (indices.Count > 0 && needed > budget)
                {
                    break;
                }

                longestRow = length;
                indices.Add(i);
                sequences.AddRange(rows);
                first++;
            }

            var scores = share ? SharedPass(sequences, answers.Count) : RowPass(sequences);

            for (int p = 0; p < indices.Count; p++)
            {
                var own = new double[answers.Count];
                for (int a = 0; a < answers.Count; a++)
                {
                    own[a] = sequences[p * answers.Count + a] is null ? double.NaN : scores[p * answers.Count + a];
                }

                results[indices[p]] = own;
            }

            done += indices.Count;
            progress?.Invoke(done);
        }

        return results;
    }

    /// <summary>
    /// The most likely of <paramref name="answers"/> for each prompt, with every answer's probability (the softmax of
    /// <see cref="LogLikelihoods"/> over the answers; even for a prompt that cannot be scored, whose best is the first).
    /// </summary>
    public AnswerChoice[] Choose(IReadOnlyList<IReadOnlyList<ChatMessage>> prompts, IReadOnlyList<string> answers, Action<int>? progress = null,
        CancellationToken cancellationToken = default) =>
        [.. LogLikelihoods(prompts, answers, progress, cancellationToken).Select(Choice)];

    // One row per (prompt, answer), each running the whole prompt: for models whose attention cannot batch rows of
    // different lengths from a cache.
    private double[] RowPass(List<TrainingSequence?> sequences)
    {
        int rows = sequences.Count, length = Math.Max(1, sequences.Max(s => s?.Tokens.Length ?? 2) - 1);
        var input = new float[rows * length];
        var positions = new List<int>();
        var targets = new List<int>();
        var owners = new List<int>();
        for (int r = 0; r < rows; r++)
        {
            if (sequences[r] is not { } s)
            {
                continue;
            }

            for (int t = 0; t + 1 < s.Tokens.Length; t++)
            {
                input[r * length + t] = s.Tokens[t];
                if (s.Trained[t + 1])
                {
                    positions.Add(r * length + t);
                    targets.Add(s.Tokens[t + 1]);
                    owners.Add(r);
                }
            }
        }

        var scores = new double[rows];
        using (Autograd.NoGrad())
        using (new TensorScope())
        {
            var hidden = _model.Network.ForwardFirst(Tensor.From(input, [rows, length], _model.Device), _modules.Count - 1);
            Add(scores, hidden, positions, targets, owners);
        }

        return scores;
    }

    // Each prompt runs once: the tokens its rows share (the prompt and the assistant turn's start) go through the network
    // with a key/value cache, prompts padded on the left to end together; then, per answer, the cache goes back to that
    // end (DecodingContext.Truncate) and only the answer's tokens run, for every prompt at once. The answer's first token
    // is scored from the prompt's last position, the others from the answer pass.
    private double[] SharedPass(List<TrainingSequence?> sequences, int answerCount)
    {
        var scores = new double[sequences.Count];
        int prompts = sequences.Count / answerCount;
        var live = Enumerable.Range(0, prompts).Where(p => sequences[p * answerCount] is not null).ToArray();
        if (live.Length == 0)
        {
            return scores;
        }

        // The shared length of each prompt's rows: their common first tokens, leaving each row at least one more.
        var shared = new int[live.Length];
        for (int i = 0; i < live.Length; i++)
        {
            var rows = Enumerable.Range(0, answerCount).Select(a => sequences[live[i] * answerCount + a]!.Tokens).ToArray();
            int common = rows.Min(r => r.Length) - 1;
            for (int a = 1; a < rows.Length; a++)
            {
                int same = 0;
                while (same < common && rows[a][same] == rows[0][same])
                {
                    same++;
                }

                common = same;
            }

            shared[i] = common;
        }

        if (shared.Any(c => c < 1))
        {
            return RowPass(sequences);                               // nothing shared to run once
        }

        int end = shared.Max();
        int longestAnswer = 0;
        for (int i = 0; i < live.Length; i++)
        {
            for (int a = 0; a < answerCount; a++)
            {
                longestAnswer = Math.Max(longestAnswer, sequences[live[i] * answerCount + a]!.Tokens.Length - 1 - shared[i]);
            }
        }

        var device = _model.Device;
        int layers = _modules.Count - 1;
        using var context = new DecodingContext(device, live.Length, end + Math.Max(1, longestAnswer));
        context.SetRowStarts([.. shared.Select(c => end - c)]);
        using (Autograd.NoGrad())
        {
            // The prompts, once. Trained tokens inside the shared part (a template's fixed start of the assistant turn)
            // count for every answer; the last shared position predicts each answer's own next token.
            using (new TensorScope())
            {
                var input = new float[live.Length * end];
                var (positions, targets, owners) = (new List<int>(), new List<int>(), new List<int>());
                for (int i = 0; i < live.Length; i++)
                {
                    int offset = end - shared[i];
                    var first = sequences[live[i] * answerCount]!;
                    for (int t = 0; t < shared[i]; t++)
                    {
                        input[i * end + offset + t] = first.Tokens[t];
                    }

                    for (int a = 0; a < answerCount; a++)
                    {
                        int row = live[i] * answerCount + a;
                        var s = sequences[row]!;
                        for (int t = 0; t < shared[i]; t++)
                        {
                            if (s.Trained[t + 1])
                            {
                                positions.Add(i * end + offset + t);
                                targets.Add(s.Tokens[t + 1]);
                                owners.Add(row);
                            }
                        }
                    }
                }

                var hidden = _model.Network.ForwardCached(Tensor.From(input, [live.Length, end], device), context, layers);
                Add(scores, hidden, positions, targets, owners);
            }

            // Each answer's remaining tokens, all prompts together, from the cached prompts.
            for (int a = 0; a < answerCount; a++)
            {
                int steps = 0;
                for (int i = 0; i < live.Length; i++)
                {
                    steps = Math.Max(steps, sequences[live[i] * answerCount + a]!.Tokens.Length - 1 - shared[i]);
                }

                if (steps == 0)
                {
                    continue;
                }

                using var scope = new TensorScope();
                context.Truncate(end);
                var input = new float[live.Length * steps];
                var (positions, targets, owners) = (new List<int>(), new List<int>(), new List<int>());
                for (int i = 0; i < live.Length; i++)
                {
                    int row = live[i] * answerCount + a;
                    var s = sequences[row]!;
                    for (int t = shared[i]; t + 1 < s.Tokens.Length; t++)
                    {
                        input[i * steps + t - shared[i]] = s.Tokens[t];
                        if (s.Trained[t + 1])
                        {
                            positions.Add(i * steps + t - shared[i]);
                            targets.Add(s.Tokens[t + 1]);
                            owners.Add(row);
                        }
                    }
                }

                var hidden = _model.Network.ForwardCached(Tensor.From(input, [live.Length, steps], device), context, layers);
                Add(scores, hidden, positions, targets, owners);
            }
        }

        return scores;
    }

    // scores[owners[i]] += log p(targets[i] | hidden position positions[i]), on the device (only these values come back).
    private void Add(double[] scores, Tensor hidden, List<int> positions, List<int> targets, List<int> owners)
    {
        var values = Losses.TokenLogProbabilities(hidden.Reshape(-1, hidden.Shape[^1]), _modules[^1].Forward, [.. positions], [.. targets]);
        for (int i = 0; i < values.Length; i++)
        {
            scores[owners[i]] += values[i];
        }
    }

    private static AnswerChoice Choice(double[] logLikelihoods)
    {
        if (logLikelihoods.Any(double.IsNaN))
        {
            double even = 1.0 / logLikelihoods.Length;
            return new AnswerChoice(0, [.. logLikelihoods.Select(_ => even)], logLikelihoods);
        }

        double best = logLikelihoods.Max();
        var weights = logLikelihoods.Select(x => Math.Exp(x - best)).ToArray();
        double total = weights.Sum();
        return new AnswerChoice(Array.IndexOf(logLikelihoods, best), [.. weights.Select(w => w / total)], logLikelihoods);
    }

    // The answer with the most tokens: prompts are fitted to it, so every answer is scored on the same text.
    private string Longest(IReadOnlyList<string> answers)
    {
        ArgumentNullException.ThrowIfNull(answers);
        return answers.Count == 0 ? "" : answers.MaxBy(a => Encoder.Tokenizer.Encode(a).Count)!;
    }

    private IReadOnlyList<ChatMessage>? Fitted(IReadOnlyList<ChatMessage> prompt, string answer)
    {
        // A message far longer than any fit is cut first (the fit then shortens it by tokens).
        int limit = 8 * MaxLength;
        var trimmed = prompt.Select(m => m.Content.Length > limit ? m with { Content = m.Content[..limit] } : m).ToList();
        return Encoder.Fit(Transcript(trimmed, answer), MaxLength)?.Messages.SkipLast(1).ToList();
    }

    private static ChatTranscript Transcript(IReadOnlyList<ChatMessage> prompt, string answer) =>
        new([.. prompt, new ChatMessage("assistant", answer)], [], Think: false);
}
