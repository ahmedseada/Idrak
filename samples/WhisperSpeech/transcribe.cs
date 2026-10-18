#:project Idrak.WhisperSpeech/Idrak.WhisperSpeech.csproj
#:project ../../src/Idrak.Gpu/Idrak.Gpu.csproj
#:property PublishAot=false

// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.
//
// transcribe.cs: speech to text through the Whisper plug-in (plan 21, S4). The command line does the same now and is the
// way to use it: `idrak transcribe -P Idrak.WhisperSpeech.dll MODEL AUDIO... [-o out.srt]` (idrak help transcribe). This
// script stays as the prototype it matches.
//   dotnet run samples/WhisperSpeech/transcribe.cs -- MODEL AUDIO [--language en|ar] [--task transcribe|translate] [--timestamps] [--beams 1] [--device cuda] [-o out.txt]
// MODEL: a Whisper folder (idrak pull openai/whisper-small). AUDIO: WAV, FLAC or MP3, any rate and channels.
using System.Diagnostics;
using System.Globalization;
using Idrak.Abstraction;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.WhisperSpeech;

WhisperSpeechPlugin.Register();
var positional = new List<string>();
var options = new Dictionary<string, string>(StringComparer.Ordinal);
for (int i = 0; i < args.Length; i++)
{
    if (args[i] == "--timestamps")
    {
        options[args[i]] = "true";
    }
    else if (args[i].StartsWith('-') && i + 1 < args.Length)
    {
        options[args[i]] = args[++i];
    }
    else
    {
        positional.Add(args[i]);
    }
}

if (positional.Count < 2 || positional.Take(2).Any(string.IsNullOrWhiteSpace))
{
    Console.Error.WriteLine("transcribe.cs MODEL AUDIO [--language en|ar] [--task transcribe|translate] [--timestamps] [--beams 1] [--device cuda] [-o out.txt]"
        + " (a PowerShell variable that is empty, such as a model folder that was not found, is dropped from the command line)");
    return 2;
}

string model = positional[0], audioPath = positional[1];
foreach (string path in new[] { model, audioPath })
{
    if (!Directory.Exists(path) && !File.Exists(path))
    {
        Console.Error.WriteLine($"transcribe.cs: {path} is not there");
        return 2;
    }
}

var device = options.TryGetValue("--device", out var d) ? Device.Parse(d) : Device.Default;
var clock = Stopwatch.StartNew();
void Step(string what) => Console.WriteLine($"[{clock.Elapsed.TotalSeconds,6:0.0} s] {what}");

Step($"loading {model} on {device}");
using var recognizer = SpeechModels.Load(model, new SpeechModelOptions { Device = device });
var audio = AudioCodecs.Decode(audioPath);
Step($"{Path.GetFileName(audioPath)}: {audio.Duration.TotalSeconds:0.00} s, {audio.Channels} channel(s) at {audio.SampleRate} Hz");
var transcription = new TranscriptionOptions
{
    Language = options.GetValueOrDefault("--language"),
    Task = options.GetValueOrDefault("--task") ?? "transcribe",
    Timestamps = options.ContainsKey("--timestamps"),
    Beams = int.Parse(options.GetValueOrDefault("--beams") ?? "1", CultureInfo.InvariantCulture),
};
var progress = new Progress<TranscriptionProgress>(p => Console.Write($"\r  {p.Stage,-8} segment {p.Segment}/{p.Segments}, {p.Elapsed.TotalSeconds:0.0} s   "));
var result = recognizer.Transcribe(audio, transcription, progress);
Console.WriteLine();
var s = result.Stats;
Step($"cold: {s.Total.TotalMilliseconds:0} ms (features {s.Features.TotalMilliseconds:0}, encoder {s.Encoder.TotalMilliseconds:0}, decoder {s.Decoder.TotalMilliseconds:0}), {s.Segments} segment(s)");
result = recognizer.Transcribe(audio, transcription);
s = result.Stats;
Step($"warm: {s.Total.TotalMilliseconds:0} ms (features {s.Features.TotalMilliseconds:0}, encoder {s.Encoder.TotalMilliseconds:0}, decoder {s.Decoder.TotalMilliseconds:0}), "
    + $"x{s.Total.TotalSeconds / Math.Max(1e-9, s.AudioDuration.TotalSeconds):0.000} real time, peak device memory {s.PeakDeviceBytes / 1048576.0:0} MB");
Console.WriteLine($"language: {result.Language ?? "(none)"}");
foreach (var segment in result.Segments)
{
    Console.WriteLine($"  [{segment.Start,7:0.00} - {segment.End,7:0.00}] {segment.Text.Trim()}");
}

Console.WriteLine(result.Text.Trim());
if (options.TryGetValue("-o", out var output))
{
    File.WriteAllText(output, result.Text.Trim() + Environment.NewLine);
    Step($"wrote {Path.GetFullPath(output)}");
}

return 0;
