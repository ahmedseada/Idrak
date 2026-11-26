// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Inference;
using Idrak.Models.Abstractions;
using Idrak.Vision;
using Idrak.Vision.Abstractions;
using Idrak.Vision.Plugins.Crnn;
using Idrak.Vision.Plugins.Gemma;
using Idrak.Vision.Plugins.IdrakVision;
using Idrak.Vision.Plugins.Llava;
using Idrak.Vision.Plugins.Qwen;

// Plan 26, phase P5: the vision families in Idrak.Vision.Plugins (Qwen-VL, LLaVA, IdrakVision, the CRNN line recognizer).
// Each family's behaviour keeps its own groups (qwen vl:, vision contracts, vlm scratch:, ...); these check the plug-ins
// and the CRNN family, which became an image model family of the line-recognition task here.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] VisionPluginGroup =
    [
    ];
}
