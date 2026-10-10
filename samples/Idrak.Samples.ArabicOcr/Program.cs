// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Reading document images into text, Arabic first (any script): an application on Idrak's public API (plan 13, "The OCR
// sample"). Two readers, chosen per run: a vision-language model (a Gemma 3 fine-tune through the Gemma 3 plug-in, which
// this app registers itself) and a line recognizer this app trains (convolutional features, a bidirectional LSTM, CTC).
// See README.md for the commands, the data and the owner's path from page scans to a trained recognizer.
//
//   dotnet run -c Release --project samples/Idrak.Samples.ArabicOcr -- --help

using System.Text;
using Idrak.Samples.ArabicOcr;

Console.OutputEncoding = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);   // Arabic on any console, no BOM
return OcrApp.Run(args, Console.Out, Console.Error);
