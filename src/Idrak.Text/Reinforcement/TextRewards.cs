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
/// The text rewards of Idrak.Text (Plan 27, R3, contract R-C4), each scoring an answer alone, their options their
/// constructors':
/// <list type="bullet">
/// <item><see cref="Metric"/> ("metric"): a tuning metric against the reference: its score for a metric where higher is
/// better, 1 less it for an error rate.</item>
/// <item><see cref="Pattern"/> ("regex"): 1 when the answer matches a pattern (whole: the whole answer; ignoring case or
/// not), else 0: a final-answer tag, a format.</item>
/// <item><see cref="Json"/> ("json"): 1 when the answer holds JSON as written, a repaired score (0.5 by default) when it
/// parses only after a repair (<see cref="JsonRepairs"/>' lenient one unless given), 0 when none.</item>
/// <item><see cref="Length"/> ("length"): a penalty past a budget: −penalty per unit beyond the budget (tokens, read from
/// the episode's info <see cref="EpisodeInfoKeys.Tokens"/>; words; characters).</item>
/// <item><see cref="Judge"/> ("judge"): a judge model's probability of its "yes" word against its "no" word
/// (<see cref="AnswerScorer"/>, any model the library loads; the question and the two words are the caller's, in the
/// judge's language).</item>
/// <item><see cref="Sum"/> ("sum"): a weighted sum, terms = "exact:1,length:0.1" (each a reward's name, or a metric's
/// through "metric"), each term's options as "term.key".</item>
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

    /// <summary>
    /// Scores <paramref name="inputs"/> with a reward expression: one reward or metric name ("exact"), or terms with
    /// weights ("exact:1,length:0.1"), each term's options as "term.key" in <paramref name="options"/>.
    /// </summary>
    /// <exception cref="NotSupportedException">Always for now: a term's reward or metric is picked by its name, which is not on the new design yet.</exception>
    public static ValueTask<float[]> ScoreAsync(string expression, IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null,
        CancellationToken cancellationToken = default) =>
        RewardExpression.Parse(expression, options).ScoreAsync(inputs, cancellationToken);

    // The options by key the contract still passes: a reward's options are its constructor's, so every key is one it does
    // not take.
    internal static void NoOptions(string reward, IReadOnlyDictionary<string, string>? options)
    {
        if (options is { Count: > 0 })
        {
            throw new ArgumentException($"The reward {reward} takes its options as constructor parameters, not by key (given '{string.Join("', '", options.Keys)}').", nameof(options));
        }
    }
}

// "metric": a tuning metric against the reference: its score for a metric where higher is better, 1 less it for an
// error rate.
internal sealed class MetricReward(ITuningMetric metric, string metricName = TuningMetrics.ExactMatch) : IRewardFunction
{
    private readonly ITuningMetric _metric = metric ?? throw new ArgumentNullException(nameof(metric));

    public string Name => TextRewards.Metric;

    public ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        TextRewards.NoOptions(Name, options);
        cancellationToken.ThrowIfCancellationRequested();
        var scores = new float[inputs.Count];
        for (int i = 0; i < inputs.Count; i++)
        {
            var input = inputs[i];
            string reference = input.Reference ?? throw new ArgumentException($"The reward {Name} ({metricName}) scores against a reference: input {i} has none.", nameof(inputs));
            double value = _metric.Score(input.Answer, reference).Value;
            scores[i] = (float)(_metric.LowerIsBetter ? 1 - value : value);
        }

        return ValueTask.FromResult(scores);
    }
}

// "regex": 1 when the answer matches the pattern (whole: the whole answer; ignoreCase), else 0.
internal sealed class RegexReward : IRewardFunction
{
    // One matcher per pattern and thread (one Regex shared by threads builds a matcher per match on the others).
    [ThreadStatic]
    private static Dictionary<(string, RegexOptions), Regex>? _patterns;

    private readonly string _pattern;
    private readonly RegexOptions _flags;

    public RegexReward(string pattern, bool whole = false, bool ignoreCase = false)
    {
        ArgumentException.ThrowIfNullOrEmpty(pattern);
        _pattern = whole ? $"^(?:{pattern})$" : pattern;
        _flags = RegexOptions.CultureInvariant | (ignoreCase ? RegexOptions.IgnoreCase : RegexOptions.None);
    }

    public string Name => TextRewards.Pattern;

    public ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        TextRewards.NoOptions(Name, options);
        string pattern = _pattern;
        var flags = _flags;
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

// "json": 1 when the answer holds JSON as written, `repaired` (0.5) when only the repair (the lenient one unless given)
// parses it, else 0.
internal sealed class JsonReward(float repaired = 0.5f, IJsonRepair? repair = null) : IRewardFunction
{
    public string Name => TextRewards.Json;

    public ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        TextRewards.NoOptions(Name, options);
        cancellationToken.ThrowIfCancellationRequested();
        var scores = new float[inputs.Count];
        for (int i = 0; i < inputs.Count; i++)
        {
            var result = repair is null ? JsonRepairs.Parse(inputs[i].Answer) : repair.Repair(inputs[i].Answer);
            scores[i] = result is null ? 0f : result.Repaired ? repaired : 1f;
        }

        return ValueTask.FromResult(scores);
    }
}

// "length": −penalty per unit past budget (unit: tokens from the episode's info, words or characters; budget 256,
// penalty 1 / budget unless given).
internal sealed class LengthReward : IRewardFunction
{
    private readonly float _budget, _penalty;
    private readonly string _unit;

    public LengthReward(float budget = 256f, string unit = "tokens", float? penalty = null)
    {
        if (!(budget >= 0f))
        {
            throw new ArgumentException($"The reward {TextRewards.Length}'s budget is 0 or more.", nameof(budget));
        }

        _budget = budget;
        _unit = unit is "tokens" or "words" or "characters" ? unit : throw new ArgumentException($"The reward {TextRewards.Length}'s unit is tokens, words or characters, not '{unit}'.", nameof(unit));
        _penalty = penalty ?? 1f / Math.Max(1f, budget);
    }

    public string Name => TextRewards.Length;

    public ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        TextRewards.NoOptions(Name, options);
        float budget = _budget;
        float penalty = _penalty;
        string unit = _unit;
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

// "sum": a weighted sum, terms = "exact:1,length:0.1" (rewards, or metrics through metric), each term's options as
// term.key in termOptions.
internal sealed class SumReward(string terms, IReadOnlyDictionary<string, string>? termOptions = null) : IRewardFunction
{
    public string Name => TextRewards.Sum;

    public ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        TextRewards.NoOptions(Name, options);
        ArgumentException.ThrowIfNullOrWhiteSpace(terms);
        return RewardExpression.Parse(terms, termOptions).ScoreAsync(inputs, cancellationToken);
    }
}

/// <summary>
/// The judge reward ("judge", <see cref="TextRewards.Judge"/>): a judge model's probability of its "yes" word over its
/// "no" word after a question about the answer (<see cref="AnswerScorer.Choose"/>, nothing generated). Made with a
/// scorer it uses that judge (an app's own model, already loaded); made with a model's folder it loads that model on its
/// device on first use and keeps it for the process (<see cref="Release"/> frees it). The question and the two words are
/// the caller's, in the judge's language; none is assumed.
/// </summary>
public sealed class JudgeReward : IRewardFunction
{
    private readonly AnswerScorer? _scorer;
    private readonly string? _model;
    private readonly string _device = "cpu";
    private readonly int _maxLength = 2048;
    private readonly string _question, _yes, _no;
    private readonly ConcurrentDictionary<(string Model, string Device, int MaxLength), Lazy<(PretrainedModel Model, AnswerScorer Scorer)>> _loaded = new();
    private readonly Lock _gate = new();

    /// <summary>A judge on the model in <paramref name="model"/>, loaded on <paramref name="device"/> on first use.</summary>
    /// <param name="model">The judge's folder.</param>
    /// <param name="question">What the judge is asked, in its language.</param>
    /// <param name="yes">The judge's word for yes.</param>
    /// <param name="no">The judge's word for no.</param>
    /// <param name="device">The device the judge is loaded on.</param>
    /// <param name="maxLength">The longest prompt the judge reads, in tokens.</param>
    public JudgeReward(string model, string question, string yes, string no, string device = "cpu", int maxLength = 2048)
        : this(question, yes, no, TextRewards.Judge)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(device);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxLength);
        _model = model;
        _device = device;
        _maxLength = maxLength;
    }

    /// <summary>A judge on <paramref name="scorer"/>'s model, named <paramref name="name"/>.</summary>
    /// <param name="scorer">The judge's scorer, its model already loaded.</param>
    /// <param name="question">What the judge is asked, in its language.</param>
    /// <param name="yes">The judge's word for yes.</param>
    /// <param name="no">The judge's word for no.</param>
    /// <param name="name">What messages call the reward.</param>
    public JudgeReward(AnswerScorer scorer, string question, string yes, string no, string name = TextRewards.Judge)
        : this(question, yes, no, name)
    {
        ArgumentNullException.ThrowIfNull(scorer);
        _scorer = scorer;
    }

    private JudgeReward(string question, string yes, string no, string name)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(question);
        ArgumentException.ThrowIfNullOrWhiteSpace(yes);
        ArgumentException.ThrowIfNullOrWhiteSpace(no);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        (_question, _yes, _no, Name) = (question, yes, no, name);
    }

    /// <summary>What messages call the reward.</summary>
    public string Name { get; }

    /// <inheritdoc />
    public ValueTask<float[]> ScoreAsync(IReadOnlyList<RewardInput> inputs, IReadOnlyDictionary<string, string>? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        TextRewards.NoOptions(Name, options);
        string question = _question, yes = _yes, no = _no;
        cancellationToken.ThrowIfCancellationRequested();
        if (inputs.Count == 0)
        {
            return ValueTask.FromResult(Array.Empty<float>());
        }

        var scorer = _scorer ?? Load();
        var prompts = inputs.Select(i => (IReadOnlyList<ChatMessage>)
            [new ChatMessage("system", question), new ChatMessage("user", i.Reference is null ? $"{i.Prompt}\n\n{i.Answer}" : $"{i.Prompt}\n\n{i.Answer}\n\n{i.Reference}")]).ToList();
        AnswerChoice[] choices;
        lock (_gate)
        {
            choices = scorer.Choose(prompts, [yes, no]);
        }

        return ValueTask.FromResult(choices.Select(c => float.IsFinite((float)c.Probabilities[0]) ? (float)c.Probabilities[0] : 0f).ToArray());
    }

    private AnswerScorer Load()
    {
        string model = _model!;
        string device = _device;
        int maxLength = _maxLength;
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
    private readonly List<(IRewardFunction Reward, float Weight)> _terms = [];

    // Picking a term's reward or metric by its name, made with the term's own options, is not on the new design yet.
    private static IRewardFunction RewardNamed(string name, IReadOnlyDictionary<string, string> options) =>
        throw new NotSupportedException($"The reward or metric '{name}' by name is not on the new design yet.");

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

            var reward = RewardNamed(name, own);
            names.Add(name);
            result._terms.Add((reward, weight));
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
        foreach (var (reward, weight) in _terms)
        {
            var scores = await reward.ScoreAsync(inputs, cancellationToken: cancellationToken).ConfigureAwait(false);
            for (int i = 0; i < totals.Length; i++)
            {
                totals[i] += (double)weight * scores[i];
            }
        }

        return [.. totals.Select(t => (float)t)];
    }
}
