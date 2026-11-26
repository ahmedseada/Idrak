// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices;

/// <summary>What a separate backend of a device (<see cref="Device.CreateSeparateBackend"/>) is given.</summary>
/// <param name="Name">Its name in messages and listings ("sim:2 (cuda:0, 8 GB cap)").</param>
/// <param name="MemoryLimit">Its hard cap in bytes, or null for the limit every device of its kind has.</param>
/// <param name="Offload">
/// Whether allocations past the cap go to pinned host memory where the kind can (CUDA), or null for
/// <see cref="ComputeResources.OffloadToHostMemory"/>. Where the kind cannot, the cap is hard: an allocation past it throws
/// <see cref="ResourceLimitExceededException"/>.
/// </param>
public sealed record SeparateBackendOptions(string Name, long? MemoryLimit, bool? Offload = null);
