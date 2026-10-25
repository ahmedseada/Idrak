// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

[assembly: LibraryRegistrations(typeof(Idrak.Onnx.Runtime.LibraryRegistrations))]

namespace Idrak.Onnx.Runtime;

/// <summary>
/// What this assembly adds to the registries of Idrak.Abstraction, which cannot name it: ONNX Runtime ("onnxruntime" in
/// <see cref="OnnxRuntimeLoaders"/>, an .onnx file's bytes to an <see cref="OnnxModule"/>), which the idrak tool's
/// <c>onnx check</c> compares Idrak's import with. <c>LibraryDefaults.Ensure</c> (in Idrak.Abstraction) calls
/// <see cref="RegisterFor"/> once per registry, before that registry is first used, when the application ships this
/// assembly or a host loads it later (<c>-P</c>).
/// </summary>
internal sealed class LibraryRegistrations : ILibraryRegistrations
{
    public void RegisterFor(Type registry)
    {
        if (registry == typeof(OnnxRuntimeLoaders))
        {
            OnnxRuntimeLoaders.Register(new OnnxRuntimeLoader());
        }
    }

    // ONNX Runtime on its default execution provider (the CPU), as OnnxModule.Load with no options runs it.
    private sealed class OnnxRuntimeLoader : IOnnxRuntimeLoader
    {
        public string Name => "onnxruntime";

        public string Summary => "ONNX Runtime (Microsoft.ML.OnnxRuntime) on its default execution provider";

        public Module Load(byte[] model) => OnnxModule.Load(model);
    }
}
