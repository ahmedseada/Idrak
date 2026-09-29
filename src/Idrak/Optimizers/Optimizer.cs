namespace Idrak.Optimizers;

/// <summary>Updates parameters from their gradients. Call <see cref="ZeroGrad"/>, then <c>loss.Backward()</c>, then <see cref="Step"/>.</summary>
public abstract class Optimizer : IDisposable
{
    /// <summary>Creates an optimizer for the given parameters (usually <c>model.Parameters()</c>).</summary>
    protected Optimizer(IEnumerable<Tensor> parameters, float learningRate)
    {
        Parameters = [.. parameters];
        if (Parameters.Count == 0)
        {
            throw new ArgumentException("The optimizer was given no parameters.", nameof(parameters));
        }

        LearningRate = learningRate;
    }

    /// <summary>The tensors this optimizer updates.</summary>
    public IReadOnlyList<Tensor> Parameters { get; }

    /// <summary>Step size; can be changed between steps (e.g. by a schedule).</summary>
    public float LearningRate { get; set; }

    /// <summary>Resets every parameter's gradient to zero (nothing to do right after a <see cref="ClipAndStep"/> that zeroed them).</summary>
    public void ZeroGrad()
    {
        if (GradientsZeroed)
        {
            GradientsZeroed = false;
            return;
        }

        foreach (var p in Parameters)
        {
            p.ZeroGrad();
        }
    }

    /// <summary>Applies one update using the current gradients.</summary>
    public abstract void Step();

    /// <summary>
    /// Clips the gradients' global norm to <paramref name="maxNorm"/> on the device (0: no clipping), applies one update,
    /// and may leave the gradients zeroed for the next step (then <see cref="ZeroGrad"/> has nothing to do): optimizers
    /// with a fused version (Adam, AdamW) do all of it in a few device passes over every parameter at once, instead of
    /// several per parameter. The gradients are not to be read after it.
    /// </summary>
    public virtual void ClipAndStep(float maxNorm)
    {
        if (maxNorm > 0f)
        {
            ClipGradientNormOnDevice(maxNorm);
        }

        Step();
    }

    /// <summary>Set by a <see cref="ClipAndStep"/> that zeroed the gradients: the next <see cref="ZeroGrad"/> skips its pass.</summary>
    protected bool GradientsZeroed { get; set; }

    /// <summary>Global L2 norm of all gradients (one device synchronization).</summary>
    public double GradientNorm()
    {
        var withGrad = Parameters.Where(p => p.Grad is not null).ToList();
        if (withGrad.Count == 0)
        {
            return 0;
        }

        using var total = Tensor.Zeros([1], withGrad[0].Device);
        foreach (var p in withGrad)
        {
            total.Backend.SumSquares(p.Grad!.Storage, total.Storage, p.Size);
        }

        return Math.Sqrt(total.Item());
    }

    /// <summary>
    /// Scales all gradients down so their global L2 norm is at most <paramref name="maxNorm"/> (no change when
    /// already smaller). Prevents exploding gradients, especially in recurrent networks. Returns the norm before clipping.
    /// </summary>
    public double ClipGradientNorm(float maxNorm)
    {
        double norm = GradientNorm();
        if (norm > maxNorm && norm > 0)
        {
            float factor = (float)(maxNorm / norm);
            if (ScalesGradientsInStep)
            {
                GradientScale = factor;                             // applied as the next Step reads the gradients
                return norm;
            }

            foreach (var p in Parameters)
            {
                if (p.Grad is { } g)
                {
                    g.Backend.Affine(g.Storage, g.Storage, g.Size, factor, 0f);
                }
            }
        }

        return norm;
    }

    /// <summary>
    /// <see cref="ClipGradientNorm"/> without reading the norm on the host: the norm and the clipping factor are computed
    /// on the device and the gradients scaled there (by 1 when the norm is within <paramref name="maxNorm"/>), so a training
    /// step need not wait for the device. Returns nothing: reading the norm would be that wait.
    /// </summary>
    public void ClipGradientNormOnDevice(float maxNorm)
    {
        var withGrad = Parameters.Where(p => p.Grad is not null).ToList();
        if (withGrad.Count == 0)
        {
            return;
        }

        var backend = withGrad[0].Backend;
        using var factor = Tensor.PersistentZeros([1], withGrad[0].Device);
        foreach (var p in withGrad)
        {
            backend.SumSquares(p.Grad!.Storage, factor.Storage, p.Size);
        }

        backend.ClipFactor(factor.Storage, factor.Storage, maxNorm);
        foreach (var p in withGrad)
        {
            backend.GroupScaleShift(p.Grad!.Storage, factor.Storage, null, p.Grad.Storage, p.Size, 1, p.Size, false);
        }
    }

    /// <summary>
    /// True when <see cref="Step"/> multiplies the gradients by <see cref="GradientScale"/> as it reads them (so
    /// <see cref="ClipGradientNorm"/> needs no separate pass over them); the step then resets it to 1.
    /// </summary>
    protected virtual bool ScalesGradientsInStep => false;

    /// <summary>Factor for the gradients of the next <see cref="Step"/> (see <see cref="ScalesGradientsInStep"/>).</summary>
    protected float GradientScale { get; set; } = 1f;

    /// <summary>Adds L2 weight decay to the gradients: g += decay · p.</summary>
    protected void ApplyCoupledWeightDecay(float decay)
    {
        if (decay == 0f)
        {
            return;
        }

        foreach (var p in Parameters)
        {
            if (p.Grad is { } g)
            {
                p.Backend.Axpy(p.Storage, g.Storage, p.Size, decay);
            }
        }
    }

    /// <summary>Allocates a zeroed state buffer shaped like <paramref name="parameter"/>, outside any <see cref="TensorScope"/>.</summary>
    protected static Tensor CreateState(Tensor parameter) => Tensor.PersistentZeros(parameter.Shape, parameter.Device);

    /// <summary>Releases optimizer state (moment buffers).</summary>
    public virtual void Dispose() => GC.SuppressFinalize(this);
}
