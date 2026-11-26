// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Text.Abstractions;

/// <summary>
/// The names of the preference losses: "dpo", "orpo" and "simpo" are the library's, made by
/// <see cref="FineTuningLosses.Dpo"/>, <see cref="FineTuningLosses.Orpo"/> and <see cref="FineTuningLosses.SimPo"/>.
/// </summary>
public static class PreferenceLosses
{
    /// <summary>Direct preference optimization against the reference: "dpo".</summary>
    public const string Dpo = "dpo";

    /// <summary>Odds ratio preference optimization (no reference): "orpo".</summary>
    public const string Orpo = "orpo";

    /// <summary>Simple preference optimization (no reference): "simpo".</summary>
    public const string SimPo = "simpo";
}
