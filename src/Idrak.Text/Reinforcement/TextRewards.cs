// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Globalization;
using System.Text.RegularExpressions;
using Idrak.Abstraction.Reinforcement;
using Idrak.Models;
using Idrak.Text.Abstractions;

namespace Idrak.Text.Reinforcement;

/// <summary>
/// The text rewards Idrak.Text registers in <see cref="RewardFunctions"/> (Plan 27, R3, contract R-C4), each scoring an
/// answer alone:
/// <list type="bullet">
/// <item><see cref="Metric"/> ("metric"): a <see cref="TuningMetrics"/> metric against the reference (option
/// <c>metric</c>, "exact" by default, and the metric's own options): its score for a metric where higher is better, 1
/// less it for an error rate.</item>
/// <item><see cref="Pattern"/> ("regex"): 1 when the answer matches <c>pattern</c> (<c>whole</c>: the whole answer;
/// <c>ignore_case</c>), else 0: a final-answer tag, a format.</item>
/// <item><see cref="Json"/> ("json"): 1 when the answer holds JSON as written, <c>repaired</c> (0.5 by default) when it
/// parses only after <see cref="JsonRepairs"/>' repair <c>repair</c> ("lenient"), 0 when none.</item>
/// <item><see cref="Length"/> ("length"): a penalty past a budget: −<c>penalty</c> per <c>unit</c> beyond <c>budget</c>
/// (tokens, read from the episode's info <see cref="EpisodeInfoKeys.Tokens"/>; words; characters).</item>
/// <item><see cref="Judge"/> ("judge"): a judge model's probability of its "yes" word against its "no" word
/// (<see cref="AnswerScorer"/>, any model the library loads: options <c>model</c>, <c>device</c>, <c>question</c>,
/// <c>yes</c>, <c>no</c>, <c>max_length</c>; the question and the two words are the caller's, in the judge's language).</item>
/// <item><see cref="Sum"/> ("sum"): a weighted sum, <c>terms</c> = "exact:1,length:0.1" (each a reward's name, or a
/// metric's through "metric"), each term's options as "term.key".</item>
/// </list>
/// </summary>
public static class TextRewards
{
    /// <summary>A tuning metric against the reference.</summary>
    public const string Metric = "metric";

    /// <summary>A regular expression the answer matches.</summary>
    public const string Pattern = "regex";

    /// <summary>The answer holds JSON.</summary>
    public const string Json = "json";

    /// <summary>A penalty past a length budget.</summary>
    public const string Length = "length";

    /// <summary>A judge model's yes.</summary>
    public const string Judge = "judge";

    /// <summary>A weighted sum of rewards.</summary>
    public const string Sum = "sum";

    /// <summary>The library's text rewards, as registered.</summary>
    internal static IReadOnlyList<IRewardFunction> Library() =>
        [new MetricReward(), new RegexReward(), new JsonReward(), new LengthReward(), new JudgeReward(), new SumReward()];

    /// <summary>
    /// Scores <paramref name="inputs"/> with a reward expression: one reward or metric name ("exact"), or terms with
    /// weights ("exact:1,length:0.1"), each term's options as "term.key" in <paramref name="options"/>.
    /// </summary>
    public static ValueTask<float[]> ScoreAsync(string expression, IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null,
        CancellationToken cancellationToken = default) =>
        RewardExpression.Parse(expression, options).ScoreAsync(inputs, cancellationToken);

    // The options of a reward, as messages name what takes them: "reward length".
    internal static OptionSchema Schema(string reward, params ReadOnlySpan<OptionSpec> options) => new($"reward {reward}", options);

    private sealed class MetricReward : IRewardFunction, IOptionSchema
    {
        private static readonly OptionSpec<string?> MetricName = OptionSpec.Text("metric", TuningMetrics.ExactMatch, summary: "the tuning metric");

        public string Name => Metric;

        public string Summary => "a tuning metric against the reference (metric=exact by default): its score, or 1 less an error rate";

        public OptionSchema Options { get; } = Schema(Metric, MetricName);

        public IReadOnlySet<string> Keys => Options.Keys;

        public ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null, CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(inputs);
            var metric = TuningMetrics.Get(options?.GetValueOrDefault("metric") ?? TuningMetrics.ExactMatch);
            var own = Options.Parse(options, metric.Keys.Contains).Rest;           // the metric's own options, as given
            cancellationToken.ThrowIfCancellationRequested();
            var scores = new float[inputs.Count];
            for (int i = 0; i < inputs.Count; i++)
            {
                var input = inputs[i];
                string reference = input.Reference ?? throw new ArgumentException($"The reward {Name} ({metric.Name}) scores against a reference: input {i} has none.", nameof(inputs));
                double value = metric.Score(input.Answer, reference, own.Count == 0 ? null : own).Value;
                scores[i] = (float)(metric.LowerIsBetter ? 1 - value : value);
            }

            return ValueTask.FromResult(scores);
        }
    }

    private sealed class RegexReward : IRewardFunction, IOptionSchema
    {
        private static readonly OptionSpec<string?> PatternText = OptionSpec.Text("pattern", takes: "a regular expression");
        private static readonly OptionSpec<bool> Whole = OptionSpec.Flag("whole", summary: "the whole answer matches");
        private static readonly OptionSpec<bool> IgnoreCase = OptionSpec.Flag("ignore_case");

        // One matcher per pattern and thread (one Regex shared by threads builds a matcher per match on the others).
        [ThreadStatic]
        private static Dictionary<(string, RegexOptions), Regex>? _patterns;

        public string Name => Pattern;

        public string Summary => "1 when the answer matches the pattern (whole: the whole answer; ignore_case), else 0";

        public OptionSchema Options { get; } = Schema(Pattern, PatternText, Whole, IgnoreCase);

        public IReadOnlySet<string> Keys => Options.Keys;

        public ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null, CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(inputs);
            var own = Options.Parse(options);
            string pattern = own.Get(PatternText) ?? throw new ArgumentException($"The reward {Name} needs its option pattern (a regular expression).", nameof(options));
            if (own.Get(Whole))
            {
                pattern = $"^(?:{pattern})$";
            }

            var flags = RegexOptions.CultureInvariant | (own.Get(IgnoreCase) ? RegexOptions.IgnoreCase : RegexOptions.None);
            _patterns ??= [];
            if (!_patterns.TryGetValue((pattern, flags), out var regex))
            {
                _patterns[(pattern, flags)] = regex = new Regex(pattern, flags, TimeSpan.FromSeconds(1));
            }

            cancellationToken.ThrowIfCancellationRequested();
            var scores = new float[inputs.Count];
            for (int i = 0; i < inputs.Count; i++)
            {
                try
                {
                    scores[i] = regex.IsMatch(inputs[i].Answer) ? 1f : 0f;
                }
                catch (RegexMatchTimeoutException)
                {
                    scores[i] = 0f;                                         // a pattern that does not finish matching does not match
                }
            }

            return ValueTask.FromResult(scores);
        }
    }

    private sealed class JsonReward : IRewardFunction, IOptionSchema
    {
        private static readonly OptionSpec<string?> Repair = OptionSpec.Text("repair", JsonRepairs.Lenient, takes: "a JSON repair's name");
        private static readonly OptionSpec<float> Repaired = OptionSpec.Number("repaired", 0.5f, takes: "a number", summary: "the score of JSON a repair parses");

        public string Name => Json;

        public string Summary => "1 when the answer holds JSON as written, `repaired` (0.5) when only a repair (repair=lenient) parses it, else 0";

        public OptionSchema Options { get; } = Schema(Json, Repair, Repaired);

        public IReadOnlySet<string> Keys => Options.Keys;

        public ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null, CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(inputs);
            var own = Options.Parse(options);
            var repair = JsonRepairs.Get(own.Get(Repair)!);
            float repaired = own.Get(Repaired);
            cancellationToken.ThrowIfCancellationRequested();
            var scores = new float[inputs.Count];
            for (int i = 0; i < inputs.Count; i++)
            {
                var result = repair.Repair(inputs[i].Answer);
                scores[i] = result is null ? 0f : result.Repaired ? repaired : 1f;
            }

            return ValueTask.FromResult(scores);
        }
    }

    private sealed class LengthReward : IRewardFunction, IOptionSchema
    {
        private static readonly OptionSpec<float> Budget = OptionSpec.Number("budget", 256f, takes: "a number");
        private static readonly OptionSpec<string> Unit = OptionSpec.Choice("unit", "tokens", "tokens", "words", "characters");
        private static readonly OptionSpec<float> Penalty = OptionSpec.Number("penalty", 1f / 256f, takes: "a number", summary: "per unit past the budget (1 / budget unless given)");

        public string Name => Length;

        public string Summary => "−penalty per unit past budget (unit: tokens from the episode's info, words or characters; budget 256, penalty 1 / budget)";

        public OptionSchema Options { get; } = Schema(Length, Budget, Unit, Penalty);

        public IReadOnlySet<string> Keys => Options.Keys;

        public ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null, CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(inputs);
            var own = Options.Parse(options);
            float budget = own.Get(Budget);
            if (budget < 0f)
            {
                throw new ArgumentException($"The reward {Name}'s budget is 0 or more.", nameof(options));
            }

            float penalty = own.IsGiven(Penalty) ? own.Get(Penalty) : 1f / Math.Max(1f, budget);
            string unit = own.Get(Unit);
            cancellationToken.ThrowIfCancellationRequested();
            var scores = new float[inputs.Count];
            for (int i = 0; i < inputs.Count; i++)
            {
                var input = inputs[i];
                double count = unit switch
                {
                    "tokens" => input.Info is { } info && info.TryGet(EpisodeInfoKeys.Tokens, out int n) ? n
                        : throw new ArgumentException($"The reward {Name} in tokens reads the episode's info \"tokens\" (an environment made with a tokenizer gives it); "
                                                      + $"input {i} has none: give unit=words or unit=characters.", nameof(inputs)),
                    "words" => input.Answer.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Length,
                    _ => input.Answer.Length,
                };
                scores[i] = (float)(-penalty * Math.Max(0, count - budget));
            }

            return ValueTask.FromResult(scores);
        }
    }

    private sealed class SumReward : IRewardFunction, IOptionSchema
    {
        private static readonly OptionSpec<string?> Terms = OptionSpec.Text("terms", takes: "weighted terms (\"exact:1,length:0.1\")");
        private static readonly Func<string, bool> TermOption = key => key.Contains('.', StringComparison.Ordinal);

        public string Name => Sum;

        public string Summary => "a weighted sum: terms=exact:1,length:0.1 (rewards, or metrics through metric), each term's options as term.key";

        public OptionSchema Options { get; } = Schema(Sum, Terms);

        public IReadOnlySet<string> Keys => Options.Keys;

        public ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null, CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(inputs);
            var own = Options.Parse(options, TermOption);
            string terms = own.Get(Terms) ?? throw new ArgumentException($"The reward {Name} needs its option terms (\"exact:1,length:0.1\").", nameof(options));
            return RewardExpression.Parse(terms, own.Rest).ScoreAsync(inputs, cancellationToken);
        }
    }
}

/// <summary>
/// The judge reward ("judge", <see cref="TextRewards.Judge"/>): a judge model's probability of its "yes" word over its
/// "no" word after a question about the answer (<see cref="AnswerScorer.Choose"/>, nothing generated). Made with a
/// scorer it uses that judge (an app's own model, already loaded); the library's loads the model of the option
/// <c>model</c> on <c>device</c> on first use and keeps it for the process (<see cref="Release"/> frees them). The
/// question and the two words are options, in the judge's language; none is assumed.
/// </summary>
public sealed class JudgeReward : IRewardFunction, IOptionSchema
{
    private static readonly OptionSpec<string?> Model = OptionSpec.Text("model", takes: "the judge's folder");
    private static readonly OptionSpec<string?> OnDevice = OptionSpec.Text("device", "cpu", takes: "a device");
    private static readonly OptionSpec<string?> Question = OptionSpec.Text("question", takes: "what the judge is asked, in its language");
    private static readonly OptionSpec<string?> Yes = OptionSpec.Text("yes", takes: "the judge's word for yes");
    private static readonly OptionSpec<string?> No = OptionSpec.Text("no", takes: "the judge's word for no");
    private static readonly OptionSpec<float> MaxLength = OptionSpec.Number("max_length", 2048f, takes: "a number");

    private readonly AnswerScorer? _scorer;
    private readonly ConcurrentDictionary<(string Model, string Device, int MaxLength), Lazy<(PretrainedModel Model, AnswerScorer Scorer)>> _loaded = new();
    private readonly Lock _gate = new();

    /// <summary>The library's judge: the model comes from the options.</summary>
    public JudgeReward()
    {
        Name = TextRewards.Judge;
        Options = TextRewards.Schema(Name, Model, OnDevice, Question, Yes, No, MaxLength);
    }

    /// <summary>A judge on <paramref name="scorer"/>'s model, registered as <paramref name="name"/>.</summary>
    public JudgeReward(AnswerScorer scorer, string name = TextRewards.Judge)
    {
        ArgumentNullException.ThrowIfNull(scorer);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        _scorer = scorer;
        Name = name;
        Options = TextRewards.Schema(Name, Question, Yes, No);
    }

    /// <inheritdoc />
    public string Name { get; }

    /// <inheritdoc />
    public string Summary => "a judge model's probability of yes (options question, yes, no; model and device unless made with a scorer)";

    /// <inheritdoc />
    public OptionSchema Options { get; }

    /// <inheritdoc />
    public IReadOnlySet<string> Keys => Options.Keys;

    /// <inheritdoc />
    public ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        var own = Options.Parse(options);
        string question = own.Get(Question) ?? throw new ArgumentException($"The reward {Name} needs its option question (what the judge is asked, in its language).", nameof(options));
        string yes = own.Get(Yes) ?? throw new ArgumentException($"The reward {Name} needs its option yes (the judge's word for yes).", nameof(options));
        string no = own.Get(No) ?? throw new ArgumentException($"The reward {Name} needs its option no (the judge's word for no).", nameof(options));
        cancellationToken.ThrowIfCancellationRequested();
        if (inputs.Count == 0)
        {
            return ValueTask.FromResult(Array.Empty<float>());
        }

        var scorer = _scorer ?? Load(own);
        var prompts = inputs.Select(i => (IReadOnlyList<ChatMessage>)
            [new ChatMessage("system", question), new ChatMessage("user", i.Reference is null ? $"{i.Prompt}\n\n{i.Answer}" : $"{i.Prompt}\n\n{i.Answer}\n\n{i.Reference}")]).ToList();
        AnswerChoice[] choices;
        lock (_gate)
        {
            choices = scorer.Choose(prompts, [yes, no]);
        }

        return ValueTask.FromResult(choices.Select(c => float.IsFinite((float)c.Probabilities[0]) ? (float)c.Probabilities[0] : 0f).ToArray());
    }

    private AnswerScorer Load(OptionValues options)
    {
        string model = options.Get(Model) ?? throw new ArgumentException($"The reward {Name} needs its option model (the judge's folder).");
        string device = options.Get(OnDevice)!;
        int maxLength = (int)options.Get(MaxLength);
        var loaded = _loaded.GetOrAdd((model, device, maxLength), key => new Lazy<(PretrainedModel, AnswerScorer)>(() =>
        {
            var judge = PretrainedModel.Load(key.Model, new PretrainedOptions { Device = Device.Parse(key.Device) });
            judge.Network.Eval();
            return (judge, new AnswerScorer(judge, key.MaxLength));
        }));
        return loaded.Value.Scorer;
    }

    /// <summary>Disposes the judge models this reward loaded.</summary>
    public void Release()
    {
        foreach (var entry in _loaded.Values.Where(l => l.IsValueCreated))
        {
            entry.Value.Model.Dispose();
        }

        _loaded.Clear();
    }
}

// A reward expression: one reward or metric name, or weighted terms "exact:1,length:0.1"; each term's options come as
// "term.key" (a term that is a metric is scored by the metric reward with metric=name). Scores add in the terms' order.
internal sealed class RewardExpression
{
    private readonly List<(IRewardFunction Reward, float Weight, Dictionary<string, string>? Options)> _terms = [];

    public static RewardExpression Parse(string expression, IReadOnlyDictionary<string, string>? options)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(expression);
        var result = new RewardExpression();
        var names = new List<string>();
        foreach (var part in expression.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            int colon = part.LastIndexOf(':');
            string name = (colon < 0 ? part : part[..colon]).Trim();
            float weight = colon < 0 ? 1f
                : float.TryParse(part[(colon + 1)..], NumberStyles.Float, CultureInfo.InvariantCulture, out float w) && float.IsFinite(w) ? w
                : throw new ArgumentException($"The reward term '{part}' has no number for its weight (\"exact:1,length:0.1\").");
            if (name.Equals(TextRewards.Sum, StringComparison.OrdinalIgnoreCase))
            {
                throw new ArgumentException("A reward expression's terms are rewards or metrics, not sum itself.");
            }

            var own = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var (key, value) in options ?? new Dictionary<string, string>())
            {
                if (key.StartsWith(name + ".", StringComparison.OrdinalIgnoreCase))
                {
                    own[key[(name.Length + 1)..]] = value;
                }
            }

            IRewardFunction reward;
            if (RewardFunctions.Find(name) is { } registered)
            {
                reward = registered;
            }
            else if (TuningMetrics.Find(name) is { } metric)
            {
                reward = RewardFunctions.Get(TextRewards.Metric);
                own["metric"] = metric.Name;
            }
            else
            {
                throw new NotSupportedException($"'{name}' is neither a reward ({string.Join(", ", RewardFunctions.Names)}) nor a metric ({string.Join(", ", TuningMetrics.Names)}).");
            }

            names.Add(name);
            result._terms.Add((reward, weight, own.Count == 0 ? null : own));
        }

        if (result._terms.Count == 0)
        {
            throw new ArgumentException($"The reward expression '{expression}' names no reward.");
        }

        foreach (var key in options?.Keys ?? [])
        {
            int dot = key.IndexOf('.');
            if (dot <= 0 || !names.Contains(key[..dot], StringComparer.OrdinalIgnoreCase))
            {
                throw new ArgumentException($"The reward option '{key}' names no term of '{expression}' (a term's options are written term.key).");
            }
        }

        return result;
    }

    public async ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, CancellationToken cancellationToken)
    {
        var totals = new double[inputs.Count];
        foreach (var (reward, weight, options) in _terms)
        {
            var scores = await reward.ScoreAsync(inputs, options, cancellationToken).ConfigureAwait(false);
            for (int i = 0; i < totals.Length; i++)
            {
                totals[i] += (double)weight * scores[i];
            }
        }

        return [.. totals.Select(t => (float)t)];
    }
}
