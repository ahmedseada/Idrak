// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Audio.Plugins.Vits;
using Idrak.Cli;
using Idrak.Cli.Commands.Run;
using Idrak.Cli.Shared;

// Plan 31, P4: the audio pipeline on the command line, run in-process on the fixture's sound as the audio CLI tests do:
// idrak audio processors and audio process, transcribe --audio-clean (files and --live), --clean taking a pipeline in
// data audio (the old stage names keeping their files), and SpeechTrainingOptions.Processing through JSON, checkpoints
// and the feature cache.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] AudioProcessingCliGroup =
    [
        ("audio processing: cli: audio processors lists every registered processor with live or whole recording (normalize's live alternative agc), its summary and each parameter (key, kind, range and unit, default, help), -j the same as JSON; a name lists one, an unknown name is refused naming the registered ones; the help of both commands, their Arabic summaries, 'idrak audio' naming its subcommands and IDRAK_AUDIO_CLEAN in transcribe's environment", d => { if (d == Device.Cpu) AudioProcessingCliList(); }),
        ("audio processing: cli: audio process on a stereo fixture WAV writes the samples of AudioPipeline.Parse(P).Apply(decoded) within the WAV's 16-bit rounding, the same file in blocks (--block-seconds), from a JSON array and from @FILE; -j names the steps and the real-time factor; an unknown processor is refused naming the registered ones, a missing --pipeline and a missing file naming them", d => { if (d == Device.Cpu) AudioProcessingCliProcess(); }),
        ("audio processing: cli: transcribe FILE --audio-clean gives the text of the cleaned sound (IDRAK_AUDIO_CLEAN the same, none turning it off), --live FILE --audio-clean the text of --live on the file cleaned first; --live refuses normalize naming agc (from the option and from the variable)", AudioProcessingCliTranscribe),
        ("audio processing: cli: data audio --clean with the old stage names and --clean-settings writes the files AudioCleaners wrote before, with pipeline text the pipeline's (its text in -j), and refuses --clean-settings beside a pipeline and an unknown processor naming the registered ones; which values are the old stages", d => { if (d == Device.Cpu) AudioProcessingCliDataAudio(); }),
        ("audio processing: cli: SpeechTrainingOptions carry Processing through JSON (text, an array of steps; none by default; an unknown processor refused naming it) and a checkpoint's state, and the feature cache keeps processed rows apart from unprocessed ones (a second processed pass adds none)", AudioProcessingCliTraining),
    ];

    // A WAV's samples against the floats they were written from: within half a 16-bit step (and the float's rounding).
    private static void ProcessingCliSameWithin16Bits(AudioData written, AudioData expected, string what)
    {
        Check(written.Channels == expected.Channels && written.Length == expected.Length && written.SampleRate == expected.SampleRate,
            $"{what}: {written.Channels} channel(s), {written.Length} samples at {written.SampleRate} Hz; expected {expected.Channels}, {expected.Length} at {expected.SampleRate}");
        double worst = 0;
        for (int i = 0; i < expected.Samples.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(written.Samples[i] - Math.Clamp(expected.Samples[i], -1f, 1f)));
        }

        Check(worst <= 1.0 / 32767 + 1e-6, $"{what}: the samples differ by up to {worst:E3} (16-bit rounding is {1.0 / 32767:E3})");
    }

    private static void AudioProcessingCliList()
    {
        var (code, output, error) = AudioCli("audio", "processors");
        Check(code == 0, $"audio processors: exit {code}\n{error}");
        foreach (string name in AudioProcessors.Names)
        {
            Check(output.Contains($"{name} (", StringComparison.Ordinal), $"audio processors lists {name}: {output}");
        }

        Check(output.Contains("normalize (", StringComparison.Ordinal) && output.Contains("whole recording; live: agc", StringComparison.Ordinal)
              && output.Contains("quantile", StringComparison.Ordinal) && output.Contains("=VALUE", StringComparison.Ordinal) && output.Contains("Hz", StringComparison.Ordinal),
            $"the live alternative, an option, the value and a unit: {output}");

        var json = AudioCliJson("audio", "processors");
        var rows = json["processors"]!.AsArray();
        Check(rows.Count == AudioProcessors.Names.Count && rows.Select(r => (string?)r!["name"]).SequenceEqual(AudioProcessors.Names), $"-j lists every processor in order: {rows.Count}");
        var normalize = rows.First(r => (string?)r!["name"] == "normalize")!;
        Check((bool?)normalize["live"] == false && (string?)normalize["live_alternative"] == "agc" && (string?)normalize["origin"] == Overrides.Library, $"normalize: {normalize}");
        var denoise = rows.First(r => (string?)r!["name"] == "denoise")!["parameters"]!.AsArray();
        Check(denoise.Select(p => (string?)p!["key"]).SequenceEqual(AudioProcessors.Get("denoise").Parameters.Select(p => p.Key)), $"denoise's parameters: {denoise}");
        foreach (var processor in AudioProcessors.Names.Select(AudioProcessors.Get))
        {
            var row = rows.First(r => (string?)r!["name"] == processor.Name)!;
            var parameters = row["parameters"]!.AsArray();
            Check((bool?)row["live"] == processor.Live && (string?)row["summary"] == processor.Summary && parameters.Count == processor.Parameters.Count, $"{processor.Name}: {row}");
            for (int i = 0; i < parameters.Count; i++)
            {
                var p = processor.Parameters[i];
                var given = parameters[i]!;
                Check((string?)given["key"] == p.Key && (string?)given["kind"] == p.Input.ToString().ToLowerInvariant() && (string?)given["range"] == p.Range
                      && (string?)given["unit"] == p.Unit && (string?)given["default"] == p.Default && (string?)given["help"] == p.Help && (double?)given["min"] == p.Min && (double?)given["max"] == p.Max,
                    $"{processor.Name}'s parameter {i}: {given}");
            }
        }

        var one = AudioCliJson("audio", "processors", "agc");
        Check(one["processors"]!.AsArray().Count == 1 && (string?)one["processors"]![0]!["name"] == "agc", $"one processor: {one}");
        var unknown = AudioCli("audio", "processors", "vacuum");
        Check(unknown.Code == 2 && unknown.Err.Contains("vacuum", StringComparison.Ordinal) && unknown.Err.Contains("highpass", StringComparison.Ordinal), $"an unknown name: {unknown.Err}");

        foreach (string name in new[] { "audio process", "audio processors" })
        {
            var (helpCode, help, _) = AudioCli([.. name.Split(' '), "--help"]);
            Check(helpCode == 0 && help.Contains("\nExamples:\n", StringComparison.Ordinal) && help.Contains($"idrak {name}", StringComparison.Ordinal)
                  && help.TrimEnd().Split('\n').All(l => l.Length <= 120), $"idrak {name} --help: {help}");
            Check(Messages.Arabic.ContainsKey(CommandTable.All.First(c => c.Name == name).Summary), $"idrak {name}: no Arabic summary");
        }

        var (group, _, groupError) = AudioCli("audio");
        Check(group == 2 && groupError.Contains("idrak audio process", StringComparison.Ordinal) && groupError.Contains("idrak audio processors", StringComparison.Ordinal), $"'idrak audio': {groupError}");
        Check(Help.For(CommandTable.All.First(c => c.Name == "transcribe")).Contains(AudioFiles.AudioCleanVariable, StringComparison.Ordinal)
              && EnvironmentVariables.All.Any(v => v.Name == AudioFiles.AudioCleanVariable), "IDRAK_AUDIO_CLEAN in the table and transcribe's help");
        foreach (string name in new[] { "transcribe", "speak", "data audio", "voice train" })
        {
            var (helpCode, help, _) = AudioCli([.. name.Split(' '), "--help"]);
            Check(helpCode == 0 && help.Contains(name == "transcribe" || name == "speak" ? "--audio-clean" : "PIPELINE", StringComparison.Ordinal), $"idrak {name} --help names the pipeline");
        }
    }

    private static void AudioProcessingCliProcess()
    {
        string root = TempFolder("processing-cli"), wav = AudioData("wav/stereo16.wav");
        const string Text = "highpass=80, gain=-3, normalize=-20 peak=0.9";
        try
        {
            var decoded = AudioCodecs.Decode(wav);
            var expected = AudioPipeline.Parse(Text).Apply(decoded);
            string whole = Path.Combine(root, "whole.wav");
            var (code, output, error) = AudioCli("audio", "process", wav, "-o", whole, "--pipeline", Text);
            Check(code == 0 && output.Contains("real-time factor", StringComparison.Ordinal) && output.Contains("normalize=-20 peak=0.9", StringComparison.Ordinal)
                  && output.Contains($"wrote {whole}", StringComparison.Ordinal), $"audio process: exit {code}\n{output}\n{error}");
            ProcessingCliSameWithin16Bits(AudioCodecs.Decode(whole), expected, "audio process");

            // Block by block (the live steps' state kept between blocks, normalize held until the end): the same file.
            string blocks = Path.Combine(root, "blocks.wav");
            Check(AudioCli("audio", "process", wav, "-o", blocks, "--pipeline", Text, "--block-seconds", "0.05").Code == 0
                  && File.ReadAllBytes(blocks).AsSpan().SequenceEqual(File.ReadAllBytes(whole)), "--block-seconds 0.05 writes the same file");

            // The pipeline as a JSON array, inline and from a file: the same file.
            string array = "[\"highpass=80\", {\"name\": \"gain\", \"value\": -3}, {\"name\": \"normalize\", \"value\": -20, \"peak\": 0.9}]";
            string fromArray = Path.Combine(root, "array.wav"), fromFile = Path.Combine(root, "file.wav"), file = Path.Combine(root, "clean.json");
            File.WriteAllText(file, array);
            Check(AudioCli("audio", "process", wav, "-o", fromArray, "--pipeline", array).Code == 0 && File.ReadAllBytes(fromArray).AsSpan().SequenceEqual(File.ReadAllBytes(whole)),
                "a JSON array gives the same file");
            Check(AudioCli("audio", "process", wav, "-o", fromFile, "--pipeline", "@" + file).Code == 0 && File.ReadAllBytes(fromFile).AsSpan().SequenceEqual(File.ReadAllBytes(whole)),
                "@FILE gives the same file");

            var json = AudioCliJson("audio", "process", wav, "-o", Path.Combine(root, "json.wav"), "--pipeline", Text);
            Check((string?)json["pipeline"] == Text && json["steps"]!.AsArray().Count == 3 && (bool?)json["steps"]![2]!["live"] == false
                  && (double)json["seconds_in"]! == Math.Round(decoded.Duration.TotalSeconds, 4) && (double)json["real_time_factor"]! > 0 && (int)json["channels_out"]! == decoded.Channels,
                $"-j: {json}");

            var unknown = AudioCli("audio", "process", wav, "-o", Path.Combine(root, "x.wav"), "--pipeline", "highpass=80, vacuum");
            Check(unknown.Code == 2 && unknown.Err.Contains("vacuum", StringComparison.Ordinal) && unknown.Err.Contains("registered", StringComparison.Ordinal)
                  && unknown.Err.Contains("normalize", StringComparison.Ordinal) && !File.Exists(Path.Combine(root, "x.wav")), $"an unknown processor: {unknown.Err}");
            var option = AudioCli("audio", "process", wav, "-o", Path.Combine(root, "x.wav"), "--pipeline", "denoise=1.5 strength=2");
            Check(option.Code == 2 && option.Err.Contains("strength", StringComparison.Ordinal) && option.Err.Contains("quantile", StringComparison.Ordinal), $"an unknown option: {option.Err}");
            var missing = AudioCli("audio", "process", wav, "-o", Path.Combine(root, "x.wav"));
            Check(missing.Code == 2 && missing.Err.Contains("--pipeline", StringComparison.Ordinal), $"no pipeline: {missing.Err}");
            var noFile = AudioCli("audio", "process", Path.Combine(root, "nothing.wav"), "-o", Path.Combine(root, "x.wav"), "--pipeline", "gain=1");
            Check(noFile.Code == 2 && noFile.Err.Contains("not found", StringComparison.Ordinal), $"a missing file: {noFile.Err}");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void AudioProcessingCliTranscribe(Device device)
    {
        string model = AudioData("tiny-whisper"), wav = AudioData("wav/pcm16.wav"), d = device.ToString(), root = TempFolder("processing-cli-transcribe");
        const string FilePipeline = "highpass=80, normalize=-20", LivePipeline = "highpass=80, gain=-3";
        try
        {
            var audio = AudioCodecs.Decode(wav);
            string expected;
            using (var recognizer = TinyWhisper(device))
            {
                expected = recognizer.Transcribe(AudioPipeline.Parse(FilePipeline).Apply(audio), TranscribeCommand.Defaults).Text.Trim();
            }

            var document = AudioCliJson("transcribe", model, wav, "--audio-clean", FilePipeline, "-d", d);
            Check((string?)document["text"] == expected && (string?)document["audio_clean"] == FilePipeline, $"transcribe --audio-clean: {document["text"]} for '{expected}'");
            JsonObject? fromVariable = null, turnedOff = null;
            WithEnvironment(new Dictionary<string, string?> { [AudioFiles.AudioCleanVariable] = FilePipeline }, () =>
            {
                fromVariable = AudioCliJson("transcribe", model, wav, "-d", d);
                turnedOff = AudioCliJson("transcribe", model, wav, "--audio-clean", "none", "-d", d);
            });
            Check((string?)fromVariable!["text"] == expected && (string?)fromVariable["audio_clean"] == FilePipeline, $"IDRAK_AUDIO_CLEAN: {fromVariable}");
            Check(turnedOff!["audio_clean"] is null, $"--audio-clean none turns the variable off: {turnedOff}");

            // --live FILE --audio-clean against --live on the file cleaned first (one channel, float samples: nothing rounded).
            var mono = new float[audio.Length];
            audio.ToMono(mono);
            string cleaned = Path.Combine(root, "cleaned.wav");
            AudioEncoders.Save(cleaned, AudioPipeline.Parse(LivePipeline).Apply(new AudioData(mono, 1, audio.SampleRate)), new AudioEncodeOptions { BitsPerSample = 32, Float = true });
            var live = AudioCliJson("transcribe", model, "--live", wav, "--audio-clean", LivePipeline, "-l", "ar", "--step", "0.3", "-d", d);
            var first = AudioCliJson("transcribe", model, "--live", cleaned, "-l", "ar", "--step", "0.3", "-d", d);
            Check((string?)live["text"] == (string?)first["text"] && (string?)live["files"]![0]!["audio_clean"] == LivePipeline,
                $"--live --audio-clean: '{live["text"]}'; the file cleaned first: '{first["text"]}'");

            var refused = AudioCli("transcribe", model, "--live", wav, "--audio-clean", "highpass=80, normalize=-20", "-d", d);
            Check(refused.Code == 2 && refused.Err.Contains("normalize", StringComparison.Ordinal) && refused.Err.Contains("agc", StringComparison.Ordinal)
                  && refused.Err.Contains("--audio-clean", StringComparison.Ordinal), $"--live refuses normalize: {refused.Err}");
            (int Code, string Out, string Err) byVariable = (-1, "", "");
            WithEnvironment(new Dictionary<string, string?> { [AudioFiles.AudioCleanVariable] = "normalize" }, () => byVariable = AudioCli("transcribe", model, "--live", wav, "-d", d));
            Check(byVariable.Code == 2 && byVariable.Err.Contains(AudioFiles.AudioCleanVariable, StringComparison.Ordinal) && byVariable.Err.Contains("agc", StringComparison.Ordinal),
                $"--live refuses the variable's normalize: {byVariable.Err}");
            var unknown = AudioCli("transcribe", model, wav, "--audio-clean", "vacuum", "-d", d);
            Check(unknown.Code == 2 && unknown.Err.Contains("vacuum", StringComparison.Ordinal) && unknown.Err.Contains("highpass", StringComparison.Ordinal), $"an unknown processor: {unknown.Err}");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void AudioProcessingCliDataAudio()
    {
        string root = TempFolder("processing-cli-data"), wav = AudioData("wav/pcm16.wav");
        try
        {
            var decoded = AudioCodecs.Decode(wav);
            var sixteen = new AudioEncodeOptions { BitsPerSample = 16 };

            // The old stage names with their settings: the files the cleaners write, byte for byte, and no pipeline.
            const string Settings = "{\"target_decibels\": -18, \"high_pass_hertz\": 100}";
            var settings = AudioCleaningOptions.FromJson(JsonNode.Parse(Settings)!.AsObject());
            string before = Path.Combine(root, "before.wav"), stages = Path.Combine(root, "stages");
            AudioEncoders.Save(before, AudioCleaners.Apply(decoded, ["highpass", "trim", "normalize"], settings), sixteen);
            var old = AudioCliJson("data", "audio", wav, "-o", stages, "--clean", "highpass, trim,normalize", "--clean-settings", Settings);
            Check(File.ReadAllBytes(Path.Combine(stages, "pcm16.wav")).AsSpan().SequenceEqual(File.ReadAllBytes(before)) && old["processing"] is null,
                $"--clean highpass,trim,normalize writes the cleaners' file: {old}");

            // Pipeline text: the pipeline's samples, its text in -j.
            const string Text = "highpass=80, normalize=-20 peak=0.9";
            string expected = Path.Combine(root, "expected.wav"), processed = Path.Combine(root, "processed");
            AudioEncoders.Save(expected, AudioPipeline.Parse(Text).Apply(decoded), sixteen);
            var json = AudioCliJson("data", "audio", wav, "-o", processed, "--clean", Text);
            Check(File.ReadAllBytes(Path.Combine(processed, "pcm16.wav")).AsSpan().SequenceEqual(File.ReadAllBytes(expected)) && (string?)json["processing"] == Text,
                $"--clean \"{Text}\" writes the pipeline's file: {json}");

            var both = AudioCli("data", "audio", wav, "-o", Path.Combine(root, "x"), "--clean", Text, "--clean-settings", Settings);
            Check(both.Code == 2 && both.Err.Contains("--clean-settings", StringComparison.Ordinal), $"--clean-settings beside a pipeline: {both.Err}");
            var unknown = AudioCli("data", "audio", wav, "-o", Path.Combine(root, "y"), "--clean", "highpass=80, vacuum");
            Check(unknown.Code == 2 && unknown.Err.Contains("vacuum", StringComparison.Ordinal) && unknown.Err.Contains("registered", StringComparison.Ordinal), $"an unknown processor: {unknown.Err}");

            Check(AudioFiles.IsStageList("highpass,denoise,trim,normalize") && AudioFiles.IsStageList(" trim , normalize "), "bare stage names are the old stages");
            Check(!AudioFiles.IsStageList("highpass=80") && !AudioFiles.IsStageList("trim, normalize peak=0.9") && !AudioFiles.IsStageList("highpass, agc")
                  && !AudioFiles.IsStageList("none") && !AudioFiles.IsStageList(""), "a value, an option or another processor is a pipeline");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void AudioProcessingCliTraining(Device device)
    {
        var pipeline = AudioPipeline.Parse("highpass=80, normalize=-20 peak=0.9");
        var options = new SpeechTrainingOptions { Cleaning = ["trim"], Processing = pipeline };
        var json = options.ToJson();
        var back = SpeechTrainingOptions.FromJson(json);
        Check((string?)json["processing"] == pipeline.ToString() && back.Processing.Equals(pipeline) && back.Cleaning.SequenceEqual(["trim"]), $"Processing round-trips: {json["processing"]}");
        Check(SpeechTrainingOptions.FromJson(new JsonObject()).Processing.IsEmpty && new SpeechTrainingOptions().ToJson()["processing"] is null, "no processing by default");
        var array = SpeechTrainingOptions.FromJson(new JsonObject { ["processing"] = new JsonArray("highpass=80", new JsonObject { ["name"] = "gain", ["value"] = -3 }) });
        Check(array.Processing.ToString() == "highpass=80, gain=-3", $"an array of steps: {array.Processing}");
        ThrowsNaming<ArgumentException>(() => SpeechTrainingOptions.FromJson(new JsonObject { ["processing"] = "highpass=80, vacuum" }), "an unknown processor", "vacuum");

        string root = TempFolder("processing-cli-training");
        VitsPlugin.Register();
        try
        {
            var (transcripts, _) = ToneSpeech(Path.Combine(root, "data"), 6, 43, maxLetters: 4);
            var train = new SpeechSource(SpeechSources.Transcripts, [transcripts]);
            string cache = Path.Combine(root, "cache");
            var tone = VitsToneTraining(cache) with { Steps = 1 };
            var processing = AudioPipeline.Parse("highpass=80, gain=-3");
            int Rows() => Directory.Exists(cache) ? Directory.GetFiles(cache, "*", SearchOption.AllDirectories).Length : 0;
            using (var model = LoadVitsTrain(device))
            {
                var result = AdversarialSynthesisTrainer.Train(model, train, null, tone with { Processing = processing }, Path.Combine(root, "processed"));
                Check(result.Steps == 1 && result.Losses.All(float.IsFinite), "a processed run trains");
            }

            int processed = Rows();
            using (var model = LoadVitsTrain(device))
            {
                AdversarialSynthesisTrainer.Train(model, train, null, tone, Path.Combine(root, "plain"));
            }

            int both = Rows();
            using (var model = LoadVitsTrain(device))
            {
                AdversarialSynthesisTrainer.Train(model, train, null, tone with { Processing = processing }, Path.Combine(root, "processed-again"));
            }

            Check(processed > 0 && both > processed && Rows() == both, $"cache rows: {processed} after the processed run, {both} after the plain one too, {Rows()} after the processed run again");
            string state = JsonNode.Parse(File.ReadAllText(Path.Combine(root, "processed", "checkpoint-1", SpeechTrainer.StateFile)))!.ToJsonString();
            Check(state.Contains("\"processing\":\"highpass=80, gain=-3\"", StringComparison.Ordinal), "the checkpoint's state holds the pipeline");
            var restored = SpeechTrainer.ReadCheckpoint(Path.Combine(root, "processed", "checkpoint-1")).Options.Processing;
            Check(restored.Equals(processing), $"the checkpoint's options give the pipeline back: {restored}");
        }
        finally
        {
            VitsPlugin.Unregister();
            Directory.Delete(root, recursive: true);
        }
    }
}
