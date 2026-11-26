// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Text.Plugins.Harmony;

/// <summary>
/// The harmony message format of GPT-OSS models: its tool-call format "harmony"
/// (reasoning in the analysis channel, calls as messages to <c>functions.name</c>; <see cref="HarmonyToolCallParser"/>).
/// </summary>
public static class HarmonyPlugin
{
    /// <summary>The tool-call format's name: "harmony".</summary>
    public const string ToolCallFormat = "harmony";
}

// The family's tool-call format.
internal sealed class HarmonyToolCallFormat : IToolCallFormat
{
    public string Name => HarmonyPlugin.ToolCallFormat;

    public bool Detect(ToolCallProbe probe) => probe.Shows("to=functions.");

    public IToolCallParser Create(ToolCallContext context) => new HarmonyToolCallParser();
}
