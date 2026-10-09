// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// An app that reads a scan with a Gemma 3 vision-language model (bakrianoo/arabic-legal-documents-ocr-1.0, or any
// Gemma3ForConditionalGeneration folder or Hugging Face id) through Idrak's public API. The library knows no vision family: the app
// registers Gemma 3's (samples/Gemma3Vision/Idrak.Gemma3Vision), then loads the model, encodes the image with the
// family's encoder and streams the answer.
//
//   dotnet run -c Release --project samples/Gemma3Vision/Idrak.Samples.Gemma3Ocr -- MODEL IMAGE [PROMPT] [options]
//
//   MODEL is a folder or a Hugging Face id (downloaded once into Idrak's cache, as idrak pull does; HF_TOKEN for gated ones).
//
//   -d, --device NAME     cpu (default), cuda:0, vulkan:0, ...
//   -w, --weights FORMAT  bf16, int8, int4 or a registered packed format (default: as stored); the encoder is float32
//   --grayscale           turn the image grey first (the OCR fine-tune's card asks for it)
//   --system TEXT         a system message
//   --prompt-file FILE    the prompt from a UTF-8 file
//   --max-tokens N        stop after N tokens (default 2048)
//   -o, --output FILE     also write the answer to FILE as UTF-8 without a BOM
//
// Greedy decoding (temperature 0, no repetition penalty), as transformers' generate does by default.

using System.Diagnostics;
using System.Text;
using Idrak.Data;
using Idrak.Gemma3Vision;
using Idrak.Generation;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Nlp;

var positional = new List<string>();
string device = "cpu", weights = "", system = "", output = "";
string? promptFile = null;
bool grayscale = false;
int maxTokens = 2048;
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "-d" or "--device": device = args[++i]; break;
        case "-w" or "--weights": weights = args[++i]; break;
        case "--grayscale": grayscale = true; break;
        case "--system": system = args[++i]; break;
        case "--prompt-file": promptFile = args[++i]; break;
        case "--max-tokens": maxTokens = int.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); break;
        case "-o" or "--output": output = args[++i]; break;
        default: positional.Add(args[i]); break;
    }
}

if (positional.Count < 2)
{
    Console.Error.WriteLine("usage: Idrak.Samples.Gemma3Ocr MODEL IMAGE [PROMPT] [-d DEVICE] [-w FORMAT] [--grayscale] [--system TEXT] [--prompt-file FILE] [--max-tokens N] [-o FILE]");
    return 2;
}

string model = positional[0], imagePath = positional[1];
string prompt = promptFile is not null ? File.ReadAllText(promptFile, Encoding.UTF8)
    : positional.Count > 2 ? string.Join(' ', positional.Skip(2))
    : "Extract the contents of this document.";
Console.OutputEncoding = Encoding.UTF8;

// 1. The family: the library registers none, so loading this model without it fails (VisionFamilies names itself).
Gemma3VisionPlugin.Register();

// 2. The model: the text decoder with the chosen weights, and its vision part as the family read it.
var watch = Stopwatch.StartNew();
var options = new PretrainedOptions
{
    Device = Device.Parse(device),
    BFloat16 = weights is "bf16" or "bfloat16",
    Int8 = weights == "int8",
    Int4 = weights == "int4",
    PackedFormatName = weights is "" or "bf16" or "bfloat16" or "int8" or "int4" ? null : weights,
};
var folder = Directory.Exists(model) ? model
    : await HuggingFaceModels.DownloadAsync(model, token: Environment.GetEnvironmentVariable("HF_TOKEN"));
using var pretrained = PretrainedModel.Load(folder, options);
var vision = pretrained.Vision ?? throw new InvalidOperationException($"{model} has no vision part.");
Console.Error.WriteLine($"loaded {model} on {pretrained.Device} in {watch.Elapsed.TotalSeconds:F1} s; vision: {vision.Describe()}");

// 3. The image side: the family's encoder (its preprocessing, grey when asked), prompt format and attention rule.
using var encoder = vision.CreateEncoder(new VisionEncoderOptions { Device = pretrained.Device, Grayscale = grayscale });
var chat = pretrained.CreateChat(KeyValueFormat.Float32);
var reader = new ChatGenerator(chat.Generator, chat.Template) { Images = new ChatImages(encoder, vision.PromptFormat, vision.Attention) };

// 4. The request: the image, then the prompt (as transformers' examples write it), greedy.
var messages = new List<ChatMessage>();
if (system.Length > 0)
{
    messages.Add(new ChatMessage("system", system));
}

messages.Add(new ChatMessage("user", [ChatImage.FromFile(imagePath), new ChatText(prompt)]));
var request = new ChatRequest(messages) { Options = new GenerationOptions { Temperature = 0, RepeatPenalty = 1, NumPredict = maxTokens } };

// 5. The answer, streamed; then the file and the speed.
var answer = new StringBuilder();
GenerationStats? stats = null;
foreach (var chunk in reader.Stream(request))
{
    Console.Write(chunk.Delta.Content);
    answer.Append(chunk.Delta.Content);
    stats = chunk.Stats ?? stats;
}

Console.WriteLine();
if (output.Length > 0)
{
    File.WriteAllText(output, answer.ToString().Trim() + "\n", new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
    Console.Error.WriteLine($"written to {output}");
}

if (stats is not null)
{
    Console.Error.WriteLine($"{stats.PromptTokens} prompt tokens in {stats.PromptDuration.TotalSeconds:F1} s, {stats.GeneratedTokens} generated at {stats.TokensPerSecond:F1} tokens/s");
}

return 0;
