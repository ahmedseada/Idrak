// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;

namespace Idrak.Abstraction.Training;

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
        // Between steps: with offloading on, cold data makes room for the next step, or offloaded tensors come back.
        TensorOffloading.StepBoundary(Parameters[0].Device);
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

    /// <summary>
    /// The optimizer's state (its moments, step count, …) by name, copied to the host, for a checkpoint that resumes
    /// training where it stopped (<see cref="LoadState"/>). Names are the optimizer's own; a parameter's state is named
    /// by its index in <see cref="Parameters"/>.
    /// </summary>
    /// <exception cref="NotSupportedException">The optimizer cannot save its state.</exception>
    public virtual IReadOnlyDictionary<string, OptimizerStateValue> SaveState() =>
        throw new NotSupportedException($"{GetType().Name} cannot save its state; a resumed run would start its moments again.");

    /// <summary>
    /// Restores a state <see cref="SaveState"/> gave (by an optimizer of the same kind over parameters of the same shapes,
    /// in the same order).
    /// </summary>
    /// <exception cref="NotSupportedException">The optimizer cannot restore a state.</exception>
    /// <exception cref="ArgumentException">A value's shape is not the one this optimizer holds, or a required value is missing.</exception>
    public virtual void LoadState(IReadOnlyDictionary<string, OptimizerStateValue> state) =>
        throw new NotSupportedException($"{GetType().Name} cannot restore a saved state.");

    /// <summary>
    /// The index in <see cref="Parameters"/> of the parameter a saved state entry (<see cref="SaveState"/>) belongs to, or
    /// null for an entry of the whole optimizer (its step count): a sharded run (plan 23, phase C6) keeps each parameter's
    /// entries on the rank that owns it. By default the convention the library's optimizers follow: the parameter's index
    /// after the last dot ("m.3"); an optimizer that names its entries otherwise overrides it.
    /// </summary>
    public virtual int? StateParameter(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        int dot = name.LastIndexOf('.');
        return dot >= 0 && int.TryParse(name.AsSpan(dot + 1), System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out int index)
               && index < Parameters.Count
            ? index
            : null;
    }

    /// <summary>A state tensor's values as a saved value.</summary>
    protected static OptimizerStateValue StateOf(Tensor tensor) => new([.. tensor.Shape], tensor.ToArray());

    /// <summary>The step count as a saved value.</summary>
    protected static OptimizerStateValue StateOf(int step) => new([1], [step]);

    /// <summary>
    /// The state tensor <paramref name="name"/> of <paramref name="state"/> loaded into <paramref name="target"/> (made by
    /// <paramref name="create"/> when null), or <paramref name="target"/> disposed and null when the state has none.
    /// </summary>
    protected static Tensor? LoadStateTensor(IReadOnlyDictionary<string, OptimizerStateValue> state, string name, Tensor? target, Func<Tensor> create)
    {
        if (!state.TryGetValue(name, out var value))
        {
            target?.Dispose();
            return null;
        }

        target ??= create();
        if (!target.Shape.SequenceEqual(value.Shape) || value.Values.Length != target.Size)
        {
            throw new ArgumentException($"The saved optimizer state '{name}' has shape [{string.Join(", ", value.Shape)}], not [{string.Join(", ", target.Shape.ToArray())}].", nameof(state));
        }

        target.Load(value.Values);
        return target;
    }

    /// <summary>The saved step count <paramref name="name"/>.</summary>
    protected static int LoadStep(IReadOnlyDictionary<string, OptimizerStateValue> state, string name = "step") =>
        state.TryGetValue(name, out var value) && value.Values.Length == 1
            ? (int)value.Values[0]
            : throw new ArgumentException($"The saved optimizer state has no '{name}'.", nameof(state));

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

    /// <summary>
    /// Decoupled weight decay (AdamW, Lion), in place: p -= <see cref="LearningRate"/> · decay · p for every parameter
    /// with a gradient. Call it at the start of <see cref="Step"/>, before the update.
    /// </summary>
    protected void ApplyDecoupledWeightDecay(float decay)
    {
        if (decay == 0f)
        {
            return;
        }

        foreach (var p in Parameters)
        {
            if (p.Grad is not null)
            {
                p.Backend.Affine(p.Storage, p.Storage, p.Size, 1f - LearningRate * decay, 0f);
            }
        }
    }

    /// <summary>
    /// Allocates a zeroed state buffer shaped like <paramref name="parameter"/>, outside any <see cref="TensorScope"/>.
    /// It is marked as optimizer state: when offloading, the first data to move to system memory.
    /// </summary>
    protected static Tensor CreateState(Tensor parameter) => CreateState(parameter.Shape, parameter.Device);

    /// <summary>Allocates a zeroed optimizer-state buffer of <paramref name="shape"/> (see <see cref="CreateState(Tensor)"/>).</summary>
    protected static Tensor CreateState(ReadOnlySpan<int> shape, Device device)
    {
        var state = Tensor.PersistentZeros(shape, device);
        TensorOffloading.MarkCold(state.Storage, OffloadPriority.OptimizerState);
        return state;
    }

    /// <summary>Releases optimizer state (moment buffers).</summary>
    public virtual void Dispose() => GC.SuppressFinalize(this);
}

/// <summary>One value of an optimizer's saved state (<see cref="Optimizer.SaveState"/>): a tensor's shape and values.</summary>
/// <param name="Shape">The tensor's shape.</param>
/// <param name="Values">Its values (packed state, such as 8-bit moments, as the float32 words that hold it).</param>
public sealed record OptimizerStateValue(int[] Shape, float[] Values);

