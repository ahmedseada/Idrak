// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction;

/// <summary>
/// Work that only queues device work (no reads back to the host, no uploads), so a step that runs it can be recorded once
/// as a <see cref="ComputeGraph"/> and replayed: a packed weight's product (<c>IPackedWeight.MatMul</c>), a key/value
/// cache format's writes and attention (<c>IRecordableLayout</c>). What does not implement it is run as it is, every step;
/// the test is made once, when the model or the generator is made, never per step.
/// </summary>
public interface IRecordable;
