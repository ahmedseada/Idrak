// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Optimizers;

/// <summary>
/// Several optimizers acting as one, so different parameter groups can use different settings (for fine-tuning:
/// a small learning rate for the pretrained body, a larger one for the new head). <see cref="Optimizer.LearningRate"/>
/// starts at the first group's rate; when a schedule changes it, every group's rate is scaled by the same factor, so
/// the groups keep their ratio. Its state (<see cref="IStatefulOptimizer"/>) is every group's, under "group{g}."; saving
/// or loading it needs every group to keep a state of its own.
/// </summary>
/// <example>
/// <code>
/// using var trainer = new Trainer(model, Losses.MeanSquaredError,
///     _ => new GroupedOptimizer(new AdamW(body.Parameters(), 1e-4f), new AdamW(head.Parameters(), 1e-3f)));
/// </code>
/// </example>
public sealed class GroupedOptimizer : Optimizer, IStatefulOptimizer, IGroupedOptimizer
{
    private readonly Optimizer[] _groups;
    private readonly float[] _initialRates;
    private readonly int[] _before;                                     // the parameters of the groups before each
    private readonly float _initialRate;

    /// <summary>Combines <paramref name="groups"/>; each must have its own, non-overlapping parameters.</summary>
    public GroupedOptimizer(params Optimizer[] groups)
        : base(groups.SelectMany(g => g.Parameters), groups.Length > 0 ? groups[0].LearningRate : 0f)
    {
        if (groups.Length == 0)
        {
            throw new ArgumentException("At least one optimizer is required.", nameof(groups));
        }

        if (groups.SelectMany(g => g.Parameters).Distinct().Count() != Parameters.Count)
        {
            throw new ArgumentException("A parameter appears in more than one group.", nameof(groups));
        }

        _groups = groups;
        _initialRates = [.. groups.Select(g => g.LearningRate)];
        _before = new int[groups.Length];
        for (int g = 1; g < groups.Length; g++)
        {
            _before[g] = _before[g - 1] + groups[g - 1].Parameters.Count;
        }

        _initialRate = groups[0].LearningRate;
    }

    /// <inheritdoc />
    /// <exception cref="NotSupportedException">A group keeps no state it can save.</exception>
    public IReadOnlyDictionary<string, OptimizerStateValue> SaveState()
    {
        var state = new Dictionary<string, OptimizerStateValue>(StringComparer.Ordinal);
        for (int g = 0; g < _groups.Length; g++)
        {
            foreach (var (name, value) in _groups[g].RequireState().SaveState())
            {
                state[$"group{g}.{name}"] = value;
            }
        }

        return state;
    }

    /// <inheritdoc />
    /// <exception cref="NotSupportedException">A group keeps no state it can restore.</exception>
    public void LoadState(IReadOnlyDictionary<string, OptimizerStateValue> state)
    {
        ArgumentNullException.ThrowIfNull(state);
        for (int g = 0; g < _groups.Length; g++)
        {
            string prefix = $"group{g}.";
            _groups[g].RequireState().LoadState(state.Where(p => p.Key.StartsWith(prefix, StringComparison.Ordinal)).ToDictionary(p => p.Key[prefix.Length..], p => p.Value, StringComparer.Ordinal));
        }
    }

    /// <inheritdoc />
    /// <remarks>
    /// An entry "group{g}.{name}" is group g's entry: its parameter is that group's, counted after the groups before it.
    /// The group's number is read from the name (no string made).
    /// </remarks>
    public int? StateParameter(ReadOnlySpan<char> name)
    {
        if (!name.StartsWith("group", StringComparison.Ordinal))
        {
            return null;
        }

        var rest = name["group".Length..];
        int dot = rest.IndexOf('.');
        if (dot <= 0 || (dot > 1 && rest[0] == '0')                     // "group{g}." as the group's number is written
            || !int.TryParse(rest[..dot], System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out int g) || g >= _groups.Length)
        {
            return null;
        }

        return _groups[g] is IStatefulOptimizer group && group.StateParameter(rest[(dot + 1)..]) is { } local ? _before[g] + local : null;
    }

    /// <summary>The combined optimizers, in order.</summary>
    public IReadOnlyList<Optimizer> Groups => _groups;

    /// <inheritdoc />
    IReadOnlyList<IOptimizer> IGroupedOptimizer.Groups => _groups;

    /// <inheritdoc />
    public override void Step()
    {
        float factor = _initialRate == 0f ? 1f : LearningRate / _initialRate;
        for (int i = 0; i < _groups.Length; i++)
        {
            _groups[i].LearningRate = _initialRates[i] * factor;
            _groups[i].Step();
        }
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        foreach (var g in _groups)
        {
            g.Dispose();
        }

        base.Dispose();
    }
}
