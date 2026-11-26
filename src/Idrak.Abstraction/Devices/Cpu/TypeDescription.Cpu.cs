// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices.Cpu;

namespace Idrak.Abstraction;

// The CPU's described lane's own plans of a description, held by the description itself beside its reader and writer:
// each made the first time the lane needs it and kept by this object, never by the description's value (a description
// made apart, equal or not, makes its own). A copy (`with`) makes its own again (the copy constructor copies none of
// them). Not readonly: each is made in place (Once).
public sealed partial record TypeDescription
{
    private Once<CpuBackend.DescribedPlan> cpuPlan;
    private Once<TypeDescription> cpuRowCodes;
    private Once<string> cpuFingerprint;

    /// <summary>The CPU's described lane's plan of the description, or null where the lane does not read it.</summary>
    internal CpuBackend.DescribedPlan? CpuPlan => cpuPlan.Get(this, static d => CpuBackend.DescribedPlan.Make(d));

    /// <summary>
    /// The codes alone of the description's rows (one plane of whole bytes a value under one scale a row), which the CPU's
    /// lane multiplies before each row's scale; asked only of a description of such rows.
    /// </summary>
    internal TypeDescription CpuRowCodes => cpuRowCodes.Get(this, static d => CpuBackend.CodesAlone(d))!;

    /// <summary>The description's fingerprint (from its fields alone), which the CPU's measured lanes are kept under on the machine.</summary>
    internal string CpuFingerprint => cpuFingerprint.Get(this, static d => CpuBackend.FingerprintOf(d))!;
}
