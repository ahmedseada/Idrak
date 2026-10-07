// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak;

/// <summary>
/// What this assembly adds to the registries of Idrak.Abstraction, which cannot name it: the offloading hooks of its layers
/// (with the devices, which Idrak.Gpu registers right after), its sample sources, its layer types, graph operations and
/// network-builder steps, its ONNX import operators and export translators, the hooks tensors and modules reach its layers
/// through, and what loading a pretrained model asks (checkpoint formats, model sources, GGUF types and families,
/// pretrained families). <c>LibraryDefaults.Ensure</c> (in Idrak.Abstraction) calls
/// <see cref="RegisterFor"/> once per registry, before that registry is first used, so an application pays only for the
/// registries it uses.
/// </summary>
internal static class LibraryRegistrations
{
    private static readonly Dictionary<Type, Action> ByRegistry = new()
    {
        [typeof(DeviceProviders)] = Offloading.ConnectTensors,   // the GPU devices (Idrak.Gpu) are the ones that offload weights
        [typeof(SampleSources)] = Data.LibrarySampleSources.RegisterAll,
        [typeof(LayerTypes)] = Layers.LibraryLayerTypes.RegisterAll,
        [typeof(GraphOps)] = Layers.LibraryGraphOps.RegisterAll,
        [typeof(NetworkOps)] = Layers.LibraryNetworkOps.RegisterAll,
        [typeof(OnnxImportOps)] = Onnx.OnnxBuiltIns.RegisterImports,
        [typeof(OnnxExportOps)] = Onnx.OnnxBuiltIns.RegisterExports,
        [typeof(CheckpointFormats)] = Models.LibraryModelFormats.RegisterCheckpointFormats,
        [typeof(ModelSources)] = Models.LibraryModelFormats.RegisterModelSources,
        [typeof(GgufTypes)] = Models.LibraryModelFormats.RegisterGgufTypes,
        [typeof(GgufArchitectures)] = Models.LibraryModelFormats.RegisterGgufArchitectures,
        [typeof(GgufPreTokenizers)] = Models.LibraryModelFormats.RegisterGgufPreTokenizers,
        [typeof(PretrainedArchitectures)] = Models.LibraryModelFormats.RegisterPretrainedArchitectures,
    };

    private static void RegisterFor(Type registry)
    {
        if (ByRegistry.TryGetValue(registry, out var register))
        {
            register();
        }
    }
}
