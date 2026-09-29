namespace Idrak.Optimizers;

/// <summary>
/// Several optimizers acting as one, so different parameter groups can use different settings (for fine-tuning:
/// a small learning rate for the pretrained body, a larger one for the new head). <see cref="Optimizer.LearningRate"/>
/// starts at the first group's rate; when a schedule changes it, every group's rate is scaled by the same factor, so
/// the groups keep their ratio.
/// </summary>
/// <example>
/// <code>
/// using var trainer = new Trainer(model, Losses.MeanSquaredError,
///     _ => new GroupedOptimizer(new AdamW(body.Parameters(), 1e-4f), new AdamW(head.Parameters(), 1e-3f)));
/// </code>
/// </example>
public sealed class GroupedOptimizer : Optimizer
{
    private readonly Optimizer[] _groups;
    private readonly float[] _initialRates;
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
        _initialRate = groups[0].LearningRate;
    }

    /// <summary>The combined optimizers, in order.</summary>
    public IReadOnlyList<Optimizer> Groups => _groups;

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
