// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Backends;

namespace Idrak;

/// <summary>
/// What this assembly adds to the registries of Idrak.Abstraction, which cannot name it: its GPU devices and the model
/// sources loading a pretrained model asks (folder, store, .gguf). <c>LibraryDefaults.Ensure</c> (in Idrak.Abstraction)
/// calls <see cref="RegisterFor"/> once per registry, before that registry is first used, so an application pays only for
/// the registries it uses. The registries that live here (layer types, graph operations, network-builder steps, sample
/// sources, ONNX translators, checkpoint formats, GGUF tables, pretrained families) register their built-ins themselves.
/// </summary>
internal static class LibraryRegistrations
{
    private static readonly Dictionary<Type, Action> ByRegistry = new()
    {
        [typeof(DeviceProviders)] = RegisterDevices,
        [typeof(ModelSources)] = Models.LibraryModelFormats.RegisterModelSources,
    };

    private static void RegisterFor(Type registry)
    {
        if (ByRegistry.TryGetValue(registry, out var register))
        {
            register();
        }
    }

    private static void RegisterDevices()
    {
        Offloading.ConnectTensors();   // the devices below are the ones that offload weights
        foreach (var provider in LibraryDevices.Providers())
        {
            DeviceProviders.Register(provider);
        }
    }
}
