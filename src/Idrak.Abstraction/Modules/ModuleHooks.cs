// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Modules;

/// <summary>
/// What <see cref="Module.Forward"/> and <see cref="Module.Predict(Tensor)"/> report or prepare, set by Idrak: weight
/// offloading (its devices), layer and inference telemetry (its <c>Telemetry</c>, while a hook listens). Null hooks cost
/// a field read.
/// </summary>
internal static class ModuleHooks
{
    /// <summary>Stages a layer's offloaded weights before its forward; the result (if any) is disposed after it.</summary>
    public static Func<Module, Tensor, IDisposable?>? EnterForward;

    /// <summary>Layer telemetry, while a hook listens for layers.</summary>
    public static volatile ILayerTelemetry? Layers;

    /// <summary>Receives each <see cref="Module.Predict(Tensor)"/> (module, input, output, start timestamp), while a hook listens for inference.</summary>
    public static volatile Action<Module, Tensor, Tensor, long>? Inference;
}

/// <summary>Layer telemetry as <see cref="Module.Forward"/> reports it.</summary>
internal interface ILayerTelemetry
{
    /// <summary>Enters a layer; returns the depth to restore.</summary>
    int EnterLayer();

    /// <summary>Leaves a layer that threw.</summary>
    void LeaveLayer(int depth);

    /// <summary>A layer's forward finished (it also leaves the layer).</summary>
    void LayerForward(Module module, Tensor input, Tensor output, long start, int depth);
}
