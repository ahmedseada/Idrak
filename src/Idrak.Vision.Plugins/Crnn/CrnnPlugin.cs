// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Vision.Abstractions;

namespace Idrak.Vision.Plugins.Crnn;

/// <summary>
/// The CRNN line recognizer (convolutions, a bidirectional LSTM, CTC) as a plug-in: <see cref="CrnnFamily"/>, for the
/// architecture name <see cref="Architecture"/>. <see cref="CrnnRecognizer"/> makes, trains, saves and reads one.
/// Finding the family of a model folder by its architecture name is not on the new design yet.
/// </summary>
public static class CrnnPlugin
{
    /// <summary>The architecture name a recognizer's config.json gives: "CrnnForLineRecognition".</summary>
    public const string Architecture = "CrnnForLineRecognition";

}
