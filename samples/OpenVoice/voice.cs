#:project Idrak.OpenVoice/Idrak.OpenVoice.csproj
#:project ../VitsSpeech/Idrak.VitsSpeech/Idrak.VitsSpeech.csproj
#:project ../../src/Idrak.Gpu/Idrak.Gpu.csproj
#:property PublishAot=false

// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.
//
// voice.cs: install a person's voice and speak with it (plan 21, S9 route B). The command line does the same now and is
// the way to use it: `idrak voice add|list|show|remove|export|import` and `idrak speak MODEL TEXT --voice NAME` (with
// -P Idrak.OpenVoice.dll, -P Idrak.VitsSpeech.dll; idrak help audio). This script stays as the prototype they match.
//   dotnet run samples/OpenVoice/voice.cs -- add NAME --converter DIR --consent TEXT [--speaker WHO] [--language ar]... [--keep-clips] [--device cuda] CLIP...
//   dotnet run samples/OpenVoice/voice.cs -- list | show NAME | remove NAME | export NAME ZIP | import ZIP|DIR
//   dotnet run samples/OpenVoice/voice.cs -- speak MODEL TEXT --voice NAME [--tau 0.3] [--conversion auto|off|NAME] [--decoder-seconds 10] [--device cuda] [-o out.wav]
// Voices live in IDRAK_VOICES, else voices/ under IDRAK_CACHE or ~/.cache/idrak. Clips: WAV, FLAC or MP3, 1 to 3 minutes in all.
using System.Diagnostics;
using Idrak.Abstraction;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.OpenVoice;
using Idrak.VitsSpeech;

OpenVoicePlugin.Register();
VitsSpeechPlugin.Register();
var store = VoiceStores.Get();
if (args.Length == 0)
{
    Console.Error.WriteLine("voice.cs add|list|show|remove|export|import|speak ... (see the top of samples/OpenVoice/voice.cs)");
    return 2;
}

var positional = new List<string>();
var options = new Dictionary<string, List<string>>(StringComparer.Ordinal);
for (int i = 1; i < args.Length; i++)
{
    if (args[i] is "--keep-clips")
    {
        options[args[i]] = [];
    }
    else if (args[i].StartsWith('-') && i + 1 < args.Length)
    {
        (options.TryGetValue(args[i], out var list) ? list : options[args[i]] = []).Add(args[++i]);
    }
    else
    {
        positional.Add(args[i]);
    }
}

string? One(string name) => options.TryGetValue(name, out var v) ? v[^1] : null;
var device = One("--device") is { } d ? Device.Parse(d) : Device.Default;
var clock = Stopwatch.StartNew();
void Step(string what) => Console.WriteLine($"[{clock.Elapsed.TotalSeconds,6:0.0} s] {what}");

switch (args[0])
{
    case "add":
    {
        string name = positional[0];
        var clips = positional.Skip(1).SelectMany(p => Directory.Exists(p) ? Directory.EnumerateFiles(p).Where(AudioCodecs.CanDecode).Order() : (IEnumerable<string>)[p]).ToList();
        string converterFolder = One("--converter") ?? throw new ArgumentException("--converter DIR: the OpenVoice folder (config.json and model.safetensors)");
        string consent = One("--consent") ?? throw new ArgumentException("--consent TEXT: the speaker's agreement, in their words");
        Step($"loading the converter from {converterFolder} on {device}");
        using var converter = VoiceConverters.Load(converterFolder, new VoiceConverterLoadOptions { Device = device });
        Step($"embedding {clips.Count} clip(s) for '{name}'");
        var voice = store.Add(name, clips, converter, new VoiceAddOptions
        {
            Consent = new VoiceConsent(One("--speaker") ?? name, consent, DateTimeOffset.UtcNow),
            Languages = options.TryGetValue("--language", out var languages) ? languages : [],
            KeepClips = options.ContainsKey("--keep-clips"),
        });
        Step($"installed '{voice.Name}' at {voice.Folder}");
        Show(voice);
        break;
    }

    case "list":
        foreach (var v in store.List())
        {
            Console.WriteLine($"{v.Name,-20} {v.Kind,-10} [{string.Join(", ", v.Languages)}] {v.Clips.Sum(c => c.Seconds):0.0} s of clips, {v.Encoder}");
        }

        Console.WriteLine($"({store.Location})");
        break;

    case "show":
        Show(store.Find(positional[0]) ?? throw new ArgumentException($"'{positional[0]}' is not installed in {store.Location}"));
        break;

    case "remove":
        Console.WriteLine(store.Remove(positional[0]) ? $"removed '{positional[0]}'" : $"'{positional[0]}' is not installed");
        break;

    case "export":
        store.Export(positional[0], positional[1]);
        Console.WriteLine($"wrote {Path.GetFullPath(positional[1])}");
        break;

    case "import":
        Show(store.Import(positional[0]));
        break;

    case "speak":
    {
        string model = positional[0], text = positional[1], output = One("-o") ?? "voice.wav";
        Step($"loading {model} on {device}");
        using var voiced = new VoicedSynthesizer(SpeechSynthesisModels.Load(model, new SpeechSynthesisLoadOptions { Device = device }), store);
        if (voiced.Synthesizer is VitsSynthesizer vits && One("--decoder-seconds") is { } pieceSeconds)
        {
            vits.DecoderChunkSeconds = double.Parse(pieceSeconds, System.Globalization.CultureInfo.InvariantCulture);   // 0: one pass
        }
        var speech = new SpeechSynthesisOptions
        {
            Voice = One("--voice"),
            VoiceConversion = One("--conversion") ?? "auto",
            Conversion = new VoiceConversionOptions { Tau = float.Parse(One("--tau") ?? "0.3", System.Globalization.CultureInfo.InvariantCulture) },
        };
        Step($"speaking {text.Length} characters{(speech.Voice is null ? "" : $" as '{speech.Voice}'")} (cold: the first call loads the converter and compiles)");
        var (audio, stats) = voiced.Speak(text, speech);
        Step($"cold: {audio.Duration.TotalSeconds:0.00} s of sound; model {stats.Model.TotalMilliseconds:0} ms, decoder {stats.Vocoder.TotalMilliseconds:0} ms, conversion {stats.Conversion.TotalMilliseconds:0} ms, x{stats.RealTimeFactor:0.00} real time");
        (audio, stats) = voiced.Speak(text, speech);
        Step($"warm: model {stats.Model.TotalMilliseconds:0} ms, decoder {stats.Vocoder.TotalMilliseconds:0} ms, conversion {stats.Conversion.TotalMilliseconds:0} ms, x{stats.RealTimeFactor:0.00} real time, peak device memory {stats.PeakDeviceBytes / 1048576.0:0} MB");
        AudioEncoders.Save(output, audio);
        Step($"wrote {Path.GetFullPath(output)} ({audio.SampleRate} Hz)");
        break;
    }

    default:
        Console.Error.WriteLine($"unknown command '{args[0]}'");
        return 2;
}

return 0;

static void Show(VoiceProfile v)
{
    Console.WriteLine($"{v.Name}: {v.Kind}, languages [{string.Join(", ", v.Languages)}], encoder {v.Encoder} ({v.EncoderFolder}), embedding {v.EmbeddingSize} wide");
    Console.WriteLine($"  consent: {v.Consent?.Speaker} agreed {v.Consent?.Agreed:u}: \"{v.Consent?.Statement}\"");
    foreach (var c in v.Clips)
    {
        Console.WriteLine($"  clip {c.File}: {c.Seconds:0.0} s, agreement {c.Agreement:0.000}{(c.Agreement < 0.8f ? "  <- far from the others: another speaker or noise?" : "")}");
    }

    Console.WriteLine($"  folder: {v.Folder}");
}
