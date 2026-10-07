// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Backends;

namespace Idrak;

/// <summary>
/// What this assembly adds to the registries of Idrak.Abstraction, which cannot name it: its GPU devices, its sample sources,
/// its layer types, graph operations and network-builder steps, its ONNX import operators and export translators, and the hooks tensors and modules reach its layers through. <c>LibraryDefaults.Ensure</c> (in Idrak.Abstraction) calls
/// <see cref="RegisterAll"/> once, before any of those registries is first used.
/// </summary>
internal static class LibraryRegistrations
{
    private static void RegisterAll()
    {
        Offloading.ConnectTensors();   // the devices below are the ones that offload weights
        foreach (var provider in LibraryDevices.Providers())
        {
            DeviceProviders.Register(provider);
        }

        Data.LibrarySampleSources.RegisterAll();
        Layers.LibraryLayerTypes.RegisterAll();
        Layers.LibraryGraphOps.RegisterAll();
        Layers.LibraryNetworkOps.RegisterAll();
        Onnx.OnnxBuiltIns.RegisterAll();
    }
}
