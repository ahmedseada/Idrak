// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Cli.Commands.Train;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands.Run;

/// <summary>
/// <c>idrak speak MODEL TEXT -o out.wav</c> (plan 21, S7): text to speech through the model's registered family
/// (<see cref="SpeechSynthesisModels"/>), in a checkpoint's speaker or an installed voice (<see cref="VoicedSynthesizer"/>:
/// the checkpoint's speakers first, then the voice store), written by the encoder of the output's extension.
/// </summary>
internal sealed class SpeakCommand : Command
{
    public override string Name => "speak";

    public override string Summary => "Text to speech with a text-to-speech model, in its own voice, a speaker or an installed voice";

    public override string Usage => """
        MODEL TEXT|-|--input FILE -o OUT [options]

        Speaks the text with the model's family (the tool registers Idrak.Audio.Plugins' families, VITS among them;
        another plug-in with -P) and writes the sound with the encoder of OUT's extension: .wav and .mp3 are built in
        (IDRAK_MP3_* set the MP3 encoder's bit rate, rate, channels and quality). --voice names a speaker of the
        checkpoint or a voice installed with idrak voice add; an embedding voice speaks in the model's voice and is
        converted to the voice's timbre by the converter it was made with (OpenVoice is built in too). -v adds the stage
        times, cold (the first call compiles and loads) and warm (a second call), and the device's peak memory. The
        library plays no sound: open the file in any player.

        Arguments:
          MODEL                  a model folder (config.json and its weights), a hub id in the cache or an alias
          TEXT                   the text (the words after MODEL); - reads it from standard input

        Options:
          -i, --input FILE       read the text from FILE (UTF-8)
          -o, --out FILE         the sound to write (.wav; .mp3 and others with the plug-in that registers their encoder)
              --voice NAME       a speaker of the checkpoint (--list-voices) or an installed voice (idrak voice list)
              --voice-conversion M  auto (convert an embedding voice with its converter), off (the model's own voice,
                                 to compare) or a converter family's name (refused when the voice was made by another)
              --tau T            how loosely the conversion copies the speech's detail (0: deterministic; default 0.3)
              --rate R           the speaking rate: 2 is twice as fast, 0.5 half as fast (default 1)
              --noise X          the noise scale of the acoustic latents (0: the same samples every time; default: the
                                 model's)
              --duration-noise X the noise scale of the durations (0: deterministic; default: the model's)
              --vocoder NAME     a registered vocoder for a model that makes a spectrogram (default: the model's own)
              --sample-rate R    resample the sound to R Hz (default: the model's, or the converter's for a converted voice)
              --split N          speak sentences in pieces of at most N characters (memory follows the piece; 0: one
                                 pass, the default)
              --decoder-seconds S  the seconds of sound the decoder makes at a time (memory follows the piece, the
                                 samples are the same; 0: one pass; default: the model's)
              --bits B           the samples written: 16 (default), 24, 32, 8 or float
              --voices DIR       the installed voices' folder (default IDRAK_VOICES, else voices/ under the cache)
              --list-voices      list the checkpoint's speakers and the installed voices, then exit
              --no-warm          with -v, report the cold call only (no second, warm call)
              --seed N           the seed of the noise (the same text and seed give the same sound)

        Examples:
          idrak speak facebook/mms-tts-eng "Hello, world." -o hello.wav
          idrak speak ./mms-tts-ara "مرحبا" -o hello.mp3 -d cuda
          idrak speak ./mms-tts-eng "Hi" --voice ahmed -o ahmed.wav -v
          cat chapter.txt | idrak speak ./mms-tts-eng - --split 400 -o chapter.wav
          idrak speak ./vits-vctk --list-voices
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } =
    [
        "--input", "--out", "--voice", "--voice-conversion", "--tau", "--rate", "--noise", "--duration-noise", "--vocoder", "--sample-rate", "--split",
        "--decoder-seconds", "--bits", "--voices",
    ];

    public override IReadOnlyCollection<string> Flags { get; } = ["--list-voices", "--no-warm"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-i"] = "--input", ["-o"] = "--out" };

    public override int Run(CommandContext context)
    {
        string name = context.Argument(0, "MODEL (a text-to-speech model folder, a hub id in the cache or an alias)");
        bool list = context.Flag("--list-voices");
        string? output = context.Option("--out");
        if (!list && output is null)
        {
            throw new UsageException("Missing -o, --out FILE (the sound to write: .wav, or .mp3 with its plug-in).");
        }

        // Everything the command line gives is checked before the model loads: a missing encoder fails at once.
        var encoder = output is null ? null : AudioFiles.EncoderFor(output);
        var bits = AudioFiles.Bits(context.Option("--bits"));
        string? text = list ? null : Text(context);
        var speech = Options(context);
        var store = AudioFiles.VoiceStore(context);
        var choice = ModelChoices.Choose(context, name);
        string folder = ModelChoices.Resolve(context, choice.Model);
        var device = context.Device;
        try
        {
            ISpeechSynthesizer synthesizer;
            using (var loading = new ProgressLine(context, $"loading {choice.Model} on {device}", unit: ProgressUnit.Elapsed))
            {
                synthesizer = SpeechSynthesisModels.Load(folder, new SpeechSynthesisLoadOptions { Device = device });
                loading.Clear();
            }

            using var voiced = new VoicedSynthesizer(synthesizer, store);
            if (list)
            {
                return ListVoices(context, choice.Model, synthesizer, store);
            }

            context.Detail($"loaded {choice.Model} on {device}: {synthesizer.SampleRate} Hz, {synthesizer.Voices.Count} speaker(s); voices in {store.Location}");
            ComputeResources.ResetPeakMemoryUsage(device);
            var (audio, cold) = Speak(context, voiced, text!, speech, "speaking");
            SynthesisStats? warm = null;
            if (context.Verbose && !context.Flag("--no-warm"))
            {
                ComputeResources.ResetPeakMemoryUsage(device);
                (_, warm) = Speak(context, voiced, text!, speech, "speaking again (warm)");
            }

            AudioFiles.Save(output!, audio, encoder!, bits);
            if (context.Verbose && !context.Json)
            {
                context.ErrorOutput.WriteLine(StageLine("cold", cold, audio));
                if (warm is not null)
                {
                    context.ErrorOutput.WriteLine(StageLine("warm", warm, audio));
                }
            }

            context.Write(string.Create(CultureInfo.InvariantCulture,
                $"wrote {output} ({AudioFiles.Seconds(audio.Duration.TotalSeconds)} at {audio.SampleRate} Hz, {AudioFiles.BitsName(bits)}-bit{(speech.Voice is null ? "" : $", voice {speech.Voice}")}; "
                + $"{AudioFiles.Pace(cold.RealTimeFactor)})"));
            var json = new JsonObject
            {
                ["model"] = choice.Model,
                ["device"] = device.ToString(),
                ["out"] = output,
                ["voice"] = speech.Voice,
                ["seconds"] = Math.Round(audio.Duration.TotalSeconds, 4),
                ["sample_rate"] = audio.SampleRate,
                ["channels"] = audio.Channels,
                ["bits"] = AudioFiles.BitsName(bits),
                ["characters"] = text!.Length,
                ["cold"] = StatsJson(cold),
            };
            if (warm is not null)
            {
                json["warm"] = StatsJson(warm);
            }

            context.WriteJson(json);
            return ExitCodes.Ok;
        }
        catch (NotSupportedException e) when (ImageModelFiles.NotRegistered(e))
        {
            return ImageModelFiles.Refuse(context, e);
        }
        catch (ArgumentException e) when (e.ParamName is "options" or "voice")
        {
            context.Error(e.Message.Split(" (Parameter", 2)[0]);                 // an unknown voice: its speakers and the installed voices
            return ExitCodes.Failed;
        }
    }

    // The text: the words after MODEL, - for standard input, or --input files; one of them.
    private static string Text(CommandContext context)
    {
        var words = context.Positional.Skip(1).ToList();
        var files = context.Options("--input");
        string text;
        if (words is ["-"])
        {
            text = StandardInput.Reader.ReadToEnd();
        }
        else if (words.Count > 0)
        {
            text = string.Join(' ', words);
        }
        else if (files.Count > 0)
        {
            text = string.Join("\n", files.Select(f => File.Exists(f) ? File.ReadAllText(f) : throw new UsageException($"Input file not found: {f}")));
        }
        else
        {
            throw new UsageException("Missing the text to speak: give it after MODEL, with --input FILE, or - to read standard input.");
        }

        return text.Trim() is { Length: > 0 } trimmed ? trimmed : throw new UsageException("The text to speak is empty.");
    }

    private static SpeechSynthesisOptions Options(CommandContext context)
    {
        int seed = context.Seed ?? 0;
        string conversion = context.Option("--voice-conversion") ?? "auto";
        return new SpeechSynthesisOptions
        {
            Voice = context.Option("--voice"),
            VoiceConversion = conversion,
            Conversion = new VoiceConversionOptions
            {
                Tau = (float)AudioFiles.Number(context, "--tau", new VoiceConversionOptions().Tau, 0),
                Seed = seed,
                DecoderSeconds = AudioFiles.OptionalNumber(context, "--decoder-seconds", 0),
            },
            Rate = (float)AudioFiles.Number(context, "--rate", 1, double.Epsilon),
            NoiseScale = (float?)AudioFiles.OptionalNumber(context, "--noise", 0),
            DurationNoiseScale = (float?)AudioFiles.OptionalNumber(context, "--duration-noise", 0),
            Seed = seed,
            Vocoder = context.Option("--vocoder"),
            SampleRate = context.Option("--sample-rate") is null ? null : Positive(context, "--sample-rate"),
            MaxCharacters = context.Option("--split") is null ? 0 : Math.Max(0, context.IntOption("--split", 0)),
            DecoderSeconds = AudioFiles.OptionalNumber(context, "--decoder-seconds", 0),
        };
    }

    private static int Positive(CommandContext context, string option) =>
        context.IntOption(option, 0) is > 0 and var value ? value : throw new UsageException($"{option} needs a whole number above 0.");

    // One call, with a progress line (the time so far) while it runs.
    private static (AudioData Audio, SynthesisStats Stats) Speak(CommandContext context, VoicedSynthesizer voiced, string text, SpeechSynthesisOptions options, string what)
    {
        string voice = options.Voice is null ? "" : $" as {options.Voice}";
        using var progress = new ProgressLine(context, $"{what} {text.Length:N0} characters{voice} on {voiced.Device}", unit: ProgressUnit.Elapsed);
        var result = voiced.Speak(text, options, context.TimeoutToken);
        progress.Clear();
        return result;
    }

    private static string StageLine(string label, SynthesisStats stats, AudioData audio) => string.Create(CultureInfo.InvariantCulture,
        $"{label}: text {stats.Text.TotalMilliseconds:0} ms, model {stats.Model.TotalMilliseconds:0} ms, decoder {stats.Vocoder.TotalMilliseconds:0} ms, "
        + $"conversion {stats.Conversion.TotalMilliseconds:0} ms; {AudioFiles.Seconds(audio.Duration.TotalSeconds)} of sound, {AudioFiles.Pace(stats.RealTimeFactor)}, "
        + $"peak device memory {Units.Bytes(stats.PeakDeviceBytes)}");

    private static JsonObject StatsJson(SynthesisStats stats) => new()
    {
        ["text_ms"] = Math.Round(stats.Text.TotalMilliseconds, 3),
        ["model_ms"] = Math.Round(stats.Model.TotalMilliseconds, 3),
        ["decoder_ms"] = Math.Round(stats.Vocoder.TotalMilliseconds, 3),
        ["conversion_ms"] = Math.Round(stats.Conversion.TotalMilliseconds, 3),
        ["real_time_factor"] = Math.Round(stats.RealTimeFactor, 5),
        ["peak_device_bytes"] = stats.PeakDeviceBytes,
    };

    private static int ListVoices(CommandContext context, string model, ISpeechSynthesizer synthesizer, IVoiceStore store)
    {
        var installed = store.List();
        context.Write(synthesizer.Voices.Count == 0
            ? $"{model}: one voice (no speakers to choose)"
            : $"{model}: {synthesizer.Voices.Count} speaker(s): {string.Join(", ", synthesizer.Voices)}");
        context.Write(installed.Count == 0 ? $"installed voices: none ({store.Location})" : $"installed voices ({store.Location}):");
        if (installed.Count > 0)
        {
            context.Table(["Voice", "Kind", "Languages", "Encoder"],
                installed.Select(v => (IReadOnlyList<string>)[v.Name, v.Kind, string.Join(",", v.Languages), v.Encoder ?? "-"]));
        }

        context.WriteJson(new JsonObject
        {
            ["model"] = model,
            ["speakers"] = new JsonArray([.. synthesizer.Voices.Select(v => (JsonNode)v)]),
            ["voices"] = new JsonArray([.. installed.Select(v => (JsonNode)new JsonObject { ["name"] = v.Name, ["kind"] = v.Kind, ["encoder"] = v.Encoder })]),
            ["store"] = store.Location,
        });
        return ExitCodes.Ok;
    }
}
