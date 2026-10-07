// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Modules;

/// <summary>
/// What <see cref="Module.Forward"/> prepares through Idrak: the staging of offloaded weights, which Idrak's devices do
/// (set when they register). Null costs a field read.
/// </summary>
internal static class ModuleHooks
{
    /// <summary>Stages a layer's offloaded weights before its forward; the result (if any) is disposed after it.</summary>
    public static Func<Module, Tensor, IDisposable?>? EnterForward;
}
