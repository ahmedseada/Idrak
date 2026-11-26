// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Vision.Abstractions;

namespace Idrak.Vision.Plugins.Crnn;

/// <summary>
/// The CRNN line recognizer (convolutions, a bidirectional LSTM, CTC) as a plug-in: <see cref="CrnnFamily"/> registered in
/// <c>ImageModelFamilies</c> under <see cref="Architecture"/>, so a recognizer's model folder loads through
/// <c>ImageModels.Load</c> and reads lines through the library's line-recognition task (<c>model.LineReader()</c>,
/// <c>idrak predict</c>, <c>idrak train --task line-recognition</c>). <see cref="CrnnRecognizer"/> makes, trains, saves
/// and reads one. <c>VisionPlugins.RegisterAll</c> registers it with the package's other families (the idrak tool
/// does at startup); an app may call <c>Register</c> alone.
/// </summary>
public static class CrnnPlugin
{
    /// <summary>The architecture name a recognizer's config.json gives: "CrnnForLineRecognition".</summary>
    public const string Architecture = "CrnnForLineRecognition";

}
