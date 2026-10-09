// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak;

/// <summary>
/// What this assembly adds to the registries of Idrak.Abstraction, which cannot name it: the offloading hooks of its layers
/// (with the devices, which Idrak.Gpu registers right after) and the model sources loading a pretrained model asks
/// (folder, store, .gguf). <c>LibraryDefaults.Ensure</c> (in Idrak.Abstraction) calls <see cref="RegisterFor"/> once per
/// registry, before that registry is first used, so an application pays only for the registries it uses. The registries
/// that live here (layer types, graph operations, network-builder steps, sample sources, ONNX translators, checkpoint
/// formats, GGUF tables, pretrained families) register their built-ins themselves, on first use.
/// </summary>
internal static class LibraryRegistrations
{
    private static readonly Dictionary<Type, Action> ByRegistry = new()
    {
        [typeof(DeviceProviders)] = Offloading.ConnectTensors,   // the GPU devices (Idrak.Gpu) are the ones that offload weights
        [typeof(ModelSources)] = Models.LibraryModelFormats.RegisterModelSources,
        [typeof(ImageTransforms)] = Data.LibraryImageTransforms.RegisterDefaults,   // Pillow's operations, byte for byte
        [typeof(Augmentations)] = Data.LibraryAugmentations.RegisterDefaults,       // flip and shift (Idrak.Vision adds the others)
        [typeof(ImageCodecs)] = RegisterImageCodecs,                                 // png, jpeg, bmp, netpbm (asked newest first)
    };

    private static void RegisterImageCodecs()
    {
        foreach (var codec in (IImageCodec[])[new Data.NetpbmCodec(), new Data.BmpCodec(), new Data.JpegCodec(), new Data.PngCodec()])
        {
            ImageCodecs.Register(codec);
        }
    }

    private static void RegisterFor(Type registry)
    {
        if (ByRegistry.TryGetValue(registry, out var register))
        {
            register();
        }
    }
}
