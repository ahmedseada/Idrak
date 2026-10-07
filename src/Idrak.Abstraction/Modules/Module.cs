// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text;
using Idrak.Abstraction.Diagnostics;
using Idrak.Abstraction.Modules;

namespace Idrak.Abstraction;

/// <summary>
/// Base class for network building blocks. A module maps an input tensor to an output tensor
/// and owns the trainable parameters it uses.
/// </summary>
public abstract class Module : IDisposable
{
    /// <summary>An optional name shown in summaries and telemetry.</summary>
    public string? Name { get; set; }

    /// <summary>The name, or a description of the module when no name was set.</summary>
    public string DisplayName => Name ?? ToString() ?? GetType().Name;

    /// <summary>
    /// True in training mode (the default), false in evaluation mode. Layers such as
    /// <c>Dropout</c> behave differently in each. Change it with <see cref="Train"/> and <see cref="Eval"/>.
    /// </summary>
    public bool IsTraining { get; private set; } = true;

    /// <summary>
    /// Computes the module's output for a batch of inputs, recording gradients when autograd is on.
    /// Publishes a <see cref="LayerForward"/> event when <see cref="TelemetryLevel.Layers"/> is enabled.
    /// </summary>
    public Tensor Forward(Tensor input)
    {
        // Weights offloaded to system memory are staged on the device around the forward (Idrak's layers do it; only
        // devices that offload, and only while something is offloaded or offloading is on, return a scope).
        using var offload = input.Device.Backend.Offload is not null ? ModuleHooks.EnterForward?.Invoke(this, input) : null;
        if (!Telemetry.IsEnabled(TelemetryLevel.Layers))
        {
            return ForwardCore(input);
        }

        int depth = Telemetry.EnterLayer();
        long start = Stopwatch.GetTimestamp();
        Tensor output;
        try
        {
            output = ForwardCore(input);
        }
        catch
        {
            Telemetry.LeaveLayer(depth);
            throw;
        }

        Telemetry.LayerForward(this, input, output, start, depth);
        return output;
    }

    /// <summary>Implements the forward computation. Override this in custom layers.</summary>
    protected abstract Tensor ForwardCore(Tensor input);

    /// <summary>
    /// Runs inference: evaluation mode, no gradient recording. Publishes an <see cref="InferenceCompleted"/>
    /// event when <see cref="TelemetryLevel.Inference"/> is enabled.
    /// </summary>
    public Tensor Predict(Tensor input)
    {
        bool wasTraining = IsTraining;
        long start = Telemetry.Start(TelemetryLevel.Inference);
        Eval();
        try
        {
            Tensor output;
            using (Autograd.NoGrad())
            using (var scope = new TensorScope())
            {
                // Intermediate activations are released immediately; only the result survives.
                output = Forward(input);
                if (!ReferenceEquals(output, input))
                {
                    scope.Keep(output);
                }
            }

            if (start != 0)
            {
                output.Device.Synchronize();
                Telemetry.Inference(new InferenceCompleted(DisplayName, input.Rank > 0 ? input.Shape[0] : 1,
                    input.Shape.ToArray(), output.Shape.ToArray(), output.Device, Stopwatch.GetElapsedTime(start)));
            }

            return output;
        }
        finally
        {
            Train(wasTraining);
        }
    }

    /// <summary>
    /// The device of the module's first parameter or, for a module without parameters (a fixed normalization), of its
    /// first buffer; null when it has neither. Where inputs for it must go.
    /// </summary>
    public Device? WeightsDevice => Parameters().FirstOrDefault()?.Device ?? Buffers().FirstOrDefault()?.Device;

    /// <summary>Predicts a batch given as a [rows, features] array and returns [rows, outputs].</summary>
    public float[,] Predict(float[,] input)
    {
        var device = WeightsDevice ?? Device.Default;
        using var x = Tensor.From(input, device);
        using var y = Predict(x);
        return y.ToArray2D();
    }

    /// <summary>Switches this module and its children to training (true) or evaluation (false) mode.</summary>
    public void Train(bool training = true)
    {
        IsTraining = training;
        foreach (var child in Children())
        {
            child.Train(training);
        }
    }

    /// <summary>Switches to evaluation mode (same as <c>Train(false)</c>).</summary>
    public void Eval() => Train(false);

    /// <summary>Direct sub-modules, for containers. Leaf layers return none.</summary>
    public virtual IEnumerable<Module> Children() => [];

    /// <summary>The trainable tensors of this module and its children, in a stable order.</summary>
    public virtual IEnumerable<Tensor> Parameters() => Children().SelectMany(c => c.Parameters());

    /// <summary>
    /// Non-trainable state of this module and its children (e.g. BatchNorm running statistics).
    /// Buffers are saved with <c>ModuleFiles.Save</c> (Idrak) and moved by <see cref="To"/>, but not optimized.
    /// </summary>
    public virtual IEnumerable<Tensor> Buffers() => Children().SelectMany(c => c.Buffers());

    /// <summary>Creates a trainable parameter (outside any <see cref="TensorScope"/>).</summary>
    protected static Tensor CreateParameter(float[] values, int[] shape, Device device) =>
        Tensor.Persistent(values, shape, device, requiresGrad: true);

    /// <summary>Creates a non-trainable buffer (outside any <see cref="TensorScope"/>).</summary>
    protected static Tensor CreateBuffer(float[] values, int[] shape, Device device) =>
        Tensor.Persistent(values, shape, device, requiresGrad: false);

    /// <summary>Returns <paramref name="tensor"/> on <paramref name="device"/>, disposing the original when it had to be copied.</summary>
    protected static Tensor MoveTensor(Tensor tensor, Device device)
    {
        if (tensor.Device == device)
        {
            return tensor;
        }

        var moved = Tensor.Persistent(tensor.ToArray(), tensor.Shape, device, tensor.RequiresGrad);
        tensor.Dispose();
        return moved;
    }

    /// <summary>Uniform values in [-bound, bound).</summary>
    protected static float[] UniformValues(int count, float bound, Random random)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = (random.NextSingle() * 2f - 1f) * bound;
        }

        return values;
    }

    /// <summary>A table of layers and parameter counts.</summary>
    public string Summary()
    {
        var sb = new StringBuilder();
        void Walk(Module m, int depth)
        {
            long own = m.Children().Any() ? 0 : m.ParameterCount;
            sb.Append(new string(' ', depth * 2)).Append(m.DisplayName);
            sb.Append(own > 0 ? $"  [{own:N0} params]" : "").AppendLine();
            foreach (var child in m.Children())
            {
                Walk(child, depth + 1);
            }
        }

        Walk(this, 0);
        sb.Append($"Total trainable parameters: {ParameterCount:N0}");
        return sb.ToString();
    }

    /// <summary>The total number of trainable values.</summary>
    public long ParameterCount => Parameters().Sum(p => (long)p.Size);

    /// <summary>Moves every parameter to <paramref name="device"/> and returns this module.</summary>
    public Module To(Device device)
    {
        MoveTo(device);
        return this;
    }

    /// <summary>Moves this module's own parameters; by default forwards the call to the children.</summary>
    protected internal virtual void MoveTo(Device device)
    {
        foreach (var child in Children())
        {
            child.MoveTo(device);
        }
    }

    /// <summary>Releases the device memory held by the parameters.</summary>
    public virtual void Dispose()
    {
        foreach (var p in Parameters().Concat(Buffers()))
        {
            p.Dispose();
        }

        GC.SuppressFinalize(this);
    }
}
