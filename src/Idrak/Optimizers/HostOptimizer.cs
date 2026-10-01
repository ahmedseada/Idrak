using Idrak.Backends.Cpu;

namespace Idrak.Optimizers;

/// <summary>
/// Runs another optimizer's step on the CPU, keeping its state (Adam's two moments: twice the parameters' size) in system
/// memory instead of on the GPU. Each step downloads the gradients, updates CPU copies of the parameters with the CPU's
/// SIMD kernels, and uploads the new values: only gradients and weights cross PCIe, never the state. Use it when the
/// optimizer state does not fit on the GPU (full fine-tuning of a model that just fits); it costs two copies of the
/// trained parameters per step. Works with any optimizer: <c>new HostOptimizer(model.Parameters(), ps =&gt; new AdamW(ps, 1e-4f))</c>.
/// </summary>
public sealed class HostOptimizer : Optimizer
{
    private readonly Tensor[] _copies;                                  // the parameters on the CPU, updated by Inner

    /// <summary>Creates the CPU copies of <paramref name="parameters"/> and the optimizer that updates them.</summary>
    /// <param name="parameters">The parameters to train (on any device).</param>
    /// <param name="create">Creates the optimizer from the CPU copies it is given, e.g. <c>ps =&gt; new AdamW(ps, 1e-4f)</c>.</param>
    public HostOptimizer(IEnumerable<Tensor> parameters, Func<IReadOnlyList<Tensor>, Optimizer> create) : base(parameters, 0f)
    {
        ArgumentNullException.ThrowIfNull(create);
        _copies = [.. Parameters.Select(p => Tensor.Persistent(p.ToArray(), p.Shape, Device.Cpu, requiresGrad: true))];
        Inner = create(_copies);
        LearningRate = Inner.LearningRate;
    }

    /// <summary>The optimizer that runs on the CPU (its learning rate follows <see cref="Optimizer.LearningRate"/>).</summary>
    public Optimizer Inner { get; }

    /// <inheritdoc />
    public override void Step()
    {
        bool[] changed = DownloadGradients(release: false);
        Inner.LearningRate = LearningRate;
        Inner.Step();
        Upload(changed);
    }

    /// <summary>
    /// Clips and steps on the CPU (see <see cref="Optimizer.ClipAndStep"/>). The parameters' gradients are released on
    /// their device after the download: the next backward pass writes fresh ones without zero-filling them first.
    /// </summary>
    public override void ClipAndStep(float maxNorm)
    {
        bool[] changed = DownloadGradients(release: true);
        Inner.LearningRate = LearningRate;
        Inner.ClipAndStep(maxNorm);
        Upload(changed);
        GradientsZeroed = true;
    }

    // Copies each gradient into its CPU copy's gradient (written in place, no zero fill); returns which parameters had one.
    private bool[] DownloadGradients(bool release)
    {
        var changed = new bool[_copies.Length];
        for (int i = 0; i < _copies.Length; i++)
        {
            var (p, copy) = (Parameters[i], _copies[i]);
            if (p.Grad is not { } g)
            {
                copy.ReleaseGrad();                                    // no gradient: the CPU optimizer skips it too
                continue;
            }

            copy.GradientTarget(out _);
            g.Backend.Download(g.Storage, CpuBackend.D(copy.Grad!.Storage).AsSpan(0, copy.Size));
            changed[i] = true;
            if (release)
            {
                p.ReleaseGrad();
            }
        }

        return changed;
    }

    private void Upload(bool[] changed)
    {
        for (int i = 0; i < _copies.Length; i++)
        {
            if (changed[i])
            {
                Parameters[i].Backend.Upload(CpuBackend.D(_copies[i].Storage).AsSpan(0, _copies[i].Size), Parameters[i].Storage);
            }
        }
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        Inner.Dispose();
        foreach (var copy in _copies)
        {
            copy.Dispose();
        }

        base.Dispose();
    }
}
