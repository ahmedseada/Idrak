// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Cli;
using Idrak.Cli.Commands.Data;
using Idrak.Cli.Commands.Run;
using Idrak.Cli.Shared;
using Idrak.Models;
using Idrak.Audio.Plugins.OpenVoice;
using Idrak.Audio.Plugins.Vits;
using Idrak.Audio.Plugins.Whisper;

// Plan 21, S7: the audio commands run in-process through CommandLine.Run with captured output, their families given
// with -P as a user gives them (the plug-in projects the tests reference), against the fixtures of the audio tests:
// transcribe, speak, voice, data audio, data mel and --audio, each against the library's own call on the same input.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] AudioCliGroup =
    [
        ("audio cli: transcribe -P the Whisper plug-in on tiny-whisper writes the library's text, JSON, SRT and VTT (from the segments' timestamps) to files and a folder, -j prints one document; without the plug-in the family exits 1 with the -P remedy", AudioCliTranscribe),
        ("audio cli: speak -P the VITS plug-in on tiny-vits writes a float WAV within 1e-3 of reference/waveform.npy's largest sample (noise 0, not silent) and an MP3 with the library's encoder; a missing text and an unregistered family are refused naming the remedy; --list-voices lists the store", AudioCliSpeak),
        ("audio cli: voice add (consent required, the clips' agreement printed and flagged), list, show, export, remove, import (again refused, --overwrite replaces) with tiny-openvoice and the fixture's WAVs; speak --voice the installed embedding voice gives VoicedSynthesizer's samples", AudioCliVoice),
        ("audio cli: data audio turns a 44.1 kHz stereo file into 16 kHz mono (the resampled mean within 1e-5), writes manifest.jsonl with --transcripts' text, splits a two-tone file at its silence and by seconds, and --stats describes the inputs alone", d => { if (d == Device.Cpu) AudioCliDataAudio(); }),
        ("audio cli: data mel writes the fixture's log-mel features within 1e-4 ([frames, mels], and [mels, frames] with --transpose) and features.json from tiny-whisper's preprocessor_config.json", AudioCliDataMel),
        ("audio cli: run --audio on an audio-language model (an audio_config and a family registered from the test) answers with the sound in the message; on a text model it is a usage error naming audio; input_audio parts of the OpenAI-style API become audio parts", AudioCliRunAudio),
        ("audio cli: every audio command's help has its examples and options, idrak help audio lists the registries, 'idrak voice' names its subcommands, and --format srt is transcribe's own", d => { if (d == Device.Cpu) AudioCliHelp(); }),
    ];

    private static string WhisperPluginPath => typeof(WhisperPlugin).Assembly.Location;

    private static string VitsPluginPath => typeof(VitsPlugin).Assembly.Location;

    private static string OpenVoicePluginPath => typeof(OpenVoicePlugin).Assembly.Location;

    private static (int Code, string Out, string Err) AudioCli(params string[] args)
    {
        var output = new StringWriter();
        var error = new StringWriter();
        int code = StandardInput.With(new StringReader(""), () =>
            CommandLine.Run([.. args, "-C", Path.Combine(Path.GetTempPath(), "idrak-cli-audio-no-config.json")], output, error));
        return (code, output.ToString(), error.ToString());
    }

    // The tool with none of its plug-in packages registered at startup (IDRAK_PLUGIN_PACKAGES=none): a family the test
    // unregistered stays so, as for a host without Idrak.Audio.Plugins.
    private static (int Code, string Out, string Err) AudioCliWithoutPackages(params string[] args)
    {
        (int Code, string Out, string Err) result = default;
        WithEnvironment(new Dictionary<string, string?> { [CommandLine.PluginPackagesVariable] = "none" }, () => result = AudioCli(args));
        return result;
    }

    private static JsonObject AudioCliJson(params string[] args)
    {
        var (code, output, error) = AudioCli([.. args, "-j"]);
        Check(code == 0, $"idrak {string.Join(' ', args)} -j: exit {code}\n{output}\n{error}");
        return JsonNode.Parse(output) as JsonObject ?? throw new Exception($"not one JSON object: {output}");
    }

    private static void AudioCliTranscribe(Device device)
    {
        string model = AudioData("tiny-whisper"), wav = AudioData("wav/pcm16.wav"), d = device.ToString();
        string root = TempFolder("cli-transcribe");
        WhisperPlugin.Unregister();
        try
        {
            var refused = AudioCliWithoutPackages("transcribe", model, wav, "-d", d);
            Check(refused.Code == 1 && refused.Err.Contains("is not registered", StringComparison.Ordinal) && refused.Err.Contains("-P", StringComparison.Ordinal),
                $"without the plug-in: {refused.Code} {refused.Err}");

            WhisperPlugin.Register();
            Transcription plain, timed;
            using (var recognizer = SpeechModels.Load(model, new SpeechModelOptions { Device = device }))
            {
                var audio = AudioCodecs.Decode(wav);
                plain = recognizer.Transcribe(audio, TranscribeCommand.Defaults);   // the command's defaults: OpenAI's guards on
                timed = recognizer.Transcribe(audio, TranscribeCommand.Defaults with { Timestamps = true });
            }

            WhisperPlugin.Unregister();                                    // the commands register it through -P
            string text = Path.Combine(root, "talk.txt");
            var (code, _, error) = AudioCli("transcribe", "-P", WhisperPluginPath, model, wav, "-o", text, "-d", d);
            Check(code == 0 && File.ReadAllText(text) == plain.Text.Trim() + "\n", $"-o talk.txt: {code} {error} '{(File.Exists(text) ? File.ReadAllText(text) : "")}' for '{plain.Text.Trim()}'");

            var printed = AudioCli("transcribe", "-P", WhisperPluginPath, model, wav, "-d", d);
            Check(printed.Code == 0 && printed.Out.TrimEnd() == plain.Text.Trim(), $"printed: '{printed.Out}' {printed.Err}");

            var document = AudioCliJson("transcribe", "-P", WhisperPluginPath, model, wav, "-d", d);
            var file = document["files"]!.AsArray()[0]!.AsObject();
            Check((string?)document["text"] == plain.Text.Trim() && (string?)file["text"] == plain.Text.Trim() && (string?)file["language"] == plain.Language
                  && file["segments"]!.AsArray().Count == plain.Segments.Count && (double)file["stats"]!["audio_seconds"]! == 1.5, $"-j: {document}");

            string json = Path.Combine(root, "talk.json");
            Check(AudioCli("transcribe", "-P", WhisperPluginPath, model, wav, "-o", json, "-d", d).Code == 0
                  && (string?)JsonNode.Parse(File.ReadAllText(json))!["text"] == plain.Text.Trim(), "-o talk.json: the JSON from the extension");

            string srt = Path.Combine(root, "talk.srt"), vtt = Path.Combine(root, "talk.vtt");
            Check(AudioCli("transcribe", "-P", WhisperPluginPath, model, wav, "-o", srt, "-d", d).Code == 0, "-o talk.srt");
            Check(File.ReadAllText(srt) == Subtitles.Srt(timed), $"the SRT from the timestamped segments: {File.ReadAllText(srt)}");
            var vttRun = AudioCli("transcribe", "-P", WhisperPluginPath, model, wav, "--format", "vtt", "-o", vtt, "-d", d);
            string cues = File.ReadAllText(vtt);
            Check(vttRun.Code == 0 && cues.StartsWith("WEBVTT\n\n", StringComparison.Ordinal) && cues == Subtitles.Vtt(timed), $"--format vtt: {vttRun.Err} {cues}");
            foreach (var segment in timed.Segments.Where(s => s.Text.Trim().Length > 0))
            {
                long ms = (long)Math.Round(segment.Start * 1000);
                string at = $"{ms / 3_600_000:00}:{ms / 60_000 % 60:00}:{ms / 1000 % 60:00}";
                Check(File.ReadAllText(srt).Contains($"{at},{ms % 1000:000} --> ", StringComparison.Ordinal) && cues.Contains($"{at}.{ms % 1000:000} --> ", StringComparison.Ordinal),
                    $"the segment at {segment.Start} s is a cue of both");
            }

            // Several files into a folder: NAME.EXT each, in the order given.
            string folder = Path.Combine(root, "texts");
            var many = AudioCliJson("transcribe", "-P", WhisperPluginPath, model, wav, AudioData("wav/stereo16.wav"), "-o", folder, "--format", "srt", "-d", d);
            Check(File.Exists(Path.Combine(folder, "pcm16.srt")) && File.Exists(Path.Combine(folder, "stereo16.srt")) && many["files"]!.AsArray().Count == 2
                  && (string?)many["files"]![0]!["file"] == wav, $"a folder of subtitles: {many}");

            var bad = AudioCli("transcribe", "-P", WhisperPluginPath, model, wav, "--format", "csv");
            Check(bad.Code == 2 && bad.Err.Contains("text, json, srt or vtt", StringComparison.Ordinal), $"--format csv: {bad.Err}");
            var missing = AudioCli("transcribe", "-P", WhisperPluginPath, model, Path.Combine(root, "nothing.wav"));
            Check(missing.Code == 2 && missing.Err.Contains("not found", StringComparison.Ordinal), $"a missing file: {missing.Err}");
        }
        finally
        {
            WhisperPlugin.Unregister();
            Directory.Delete(root, recursive: true);
        }
    }

    private static void AudioCliSpeak(Device device)
    {
        string model = AudioData("tiny-vits"), d = device.ToString(), root = TempFolder("cli-speak");
        string text = (string)JsonNode.Parse(File.ReadAllText(Path.Combine(model, "reference", "input.json")))!["text"]!;
        var expected = ReadNpyFloat32(Path.Combine(model, "reference", "waveform.npy"));
        VitsPlugin.Unregister();
        try
        {
            string wav = Path.Combine(root, "abc.wav");
            var refused = AudioCliWithoutPackages("speak", model, text, "-o", wav, "-d", d);
            Check(refused.Code == 1 && refused.Err.Contains("is not registered", StringComparison.Ordinal) && refused.Err.Contains("-P", StringComparison.Ordinal),
                $"without the plug-in: {refused.Code} {refused.Err}");

            var spoken = AudioCliJson("speak", "-P", VitsPluginPath, model, text, "--noise", "0", "--duration-noise", "0", "--bits", "float", "-o", wav, "-d", d);
            var audio = AudioCodecs.Decode(wav);
            Check(audio.SampleRate == 16000 && audio.Channels == 1 && (int)spoken["sample_rate"]! == 16000 && (string?)spoken["bits"] == "float", $"the WAV: {spoken}");
            float largest = expected.Max(MathF.Abs);
            Check(largest > 0 && audio.Samples.Max(MathF.Abs) >= 0.5f * largest, $"speak's WAV is not silent: largest {audio.Samples.Max(MathF.Abs):G3}, the reference's {largest:G3}");
            AssertClose(expected, audio.Samples, 1e-3f * largest, $"speak's WAV against reference/waveform.npy (largest sample {largest:G3})");
            Check(spoken["cold"]!["real_time_factor"] is not null && spoken["warm"] is null, "-j: the cold call's figures, no warm call without -v");

            var verbose = AudioCli("speak", "-P", VitsPluginPath, model, text, "--noise", "0", "--duration-noise", "0", "-o", Path.Combine(root, "v.wav"), "-v", "-d", d);
            Check(verbose.Code == 0 && verbose.Err.Contains("cold: ", StringComparison.Ordinal) && verbose.Err.Contains("warm: ", StringComparison.Ordinal)
                  && verbose.Err.Contains("peak device memory", StringComparison.Ordinal), $"-v: cold and warm stage times: {verbose.Err}");

            string mp3 = Path.Combine(root, "abc.mp3");
            var asMp3 = AudioCli("speak", "-P", VitsPluginPath, model, text, "--noise", "0", "--duration-noise", "0", "-o", mp3, "-d", d);
            Check(asMp3.Code == 0 && File.Exists(mp3), $"-o abc.mp3 with the library's encoder: {asMp3.Err}");
            var decoded = AudioCodecs.Decode(mp3);
            Check(decoded.Length >= expected.Length && decoded.Channels == 1 && decoded.Samples.Any(v => v != 0), $"the MP3 decodes to {decoded.Length} samples at {decoded.SampleRate} Hz");

            var noText = AudioCli("speak", "-P", VitsPluginPath, model, "-o", wav);
            Check(noText.Code == 2 && noText.Err.Contains("text to speak", StringComparison.Ordinal), $"no text: {noText.Err}");
            var noOut = AudioCli("speak", "-P", VitsPluginPath, model, text);
            Check(noOut.Code == 2 && noOut.Err.Contains("--out", StringComparison.Ordinal), $"no -o: {noOut.Err}");

            string store = Path.Combine(root, "voices");
            var listed = AudioCliJson("speak", "-P", VitsPluginPath, model, "--list-voices", "--voices", store);
            Check(listed["speakers"]!.AsArray().Count == 0 && listed["voices"]!.AsArray().Count == 0 && (string?)listed["store"] == Path.GetFullPath(store), $"--list-voices: {listed}");
            var unknown = AudioCli("speak", "-P", VitsPluginPath, model, text, "--voice", "stranger", "--voices", store, "-o", wav, "-d", d);
            Check(unknown.Code == 1 && unknown.Err.Contains("'stranger'", StringComparison.Ordinal) && unknown.Err.Contains("installed", StringComparison.Ordinal), $"an unknown voice: {unknown.Err}");
        }
        finally
        {
            VitsPlugin.Unregister();
                Directory.Delete(root, recursive: true);
        }
    }

    private static void AudioCliVoice(Device device)
    {
        string converter = AudioData("tiny-openvoice"), model = AudioData("tiny-vits"), d = device.ToString(), root = TempFolder("cli-voice");
        string store = Path.Combine(root, "voices"), stereo = AudioData("wav/stereo16.wav"), pcm = AudioData("wav/pcm16.wav");
        const string Consent = "I agree that this test voice is installed.";
        string[] add = ["voice", "add", "tester", "--from", stereo, "--from", pcm, "--converter", converter, "-P", OpenVoicePluginPath, "--voices", store, "-d", d];
        OpenVoicePlugin.Unregister();
        VitsPlugin.Unregister();
        try
        {
            var noConsent = AudioCli(add);
            Check(noConsent.Code == 2 && noConsent.Err.Contains("--consent", StringComparison.Ordinal) && !Directory.Exists(Path.Combine(store, "tester")), $"no consent: {noConsent.Err}");

            var installed = AudioCliJson([.. add, "--consent", Consent, "-l", "ar", "-l", "en", "--speaker", "A tester", "--warn-agreement", "1"]);
            Check((string?)installed["name"] == "tester" && (string?)installed["kind"] == VoiceKinds.Embedding && (string?)installed["encoder"] == OpenVoiceFamily.FamilyName
                  && installed["languages"]!.AsArray().Select(l => (string?)l).SequenceEqual(["ar", "en"]) && (string?)installed["consent"]!["statement"] == Consent
                  && (string?)installed["consent"]!["speaker"] == "A tester", $"voice add -j: {installed}");
            var clips = installed["clips"]!.AsArray();
            Check(clips.Count == 2 && clips.All(c => (bool)c!["flagged"]!) && (string?)clips[0]!["file"] == "stereo16.wav", $"the clips, all flagged under 1: {clips}");

            // A converter pulled as a repository's subfolder (owner/name/FOLDER, OpenVoiceV2/converter's layout: no
            // config.json at the revision's top, a .pth checkpoint) is listed and found by that id.
            string cache = Path.Combine(root, "cache"), pulled = Path.Combine(cache, "downloads", "huggingface", "models", "test", "openvoice", "0123456789ab", "converter");
            Directory.CreateDirectory(pulled);
            foreach (string file in new[] { "config.json", "checkpoint.pth" })
            {
                File.Copy(Path.Combine(converter, file), Path.Combine(pulled, file));
            }

            var cached = AudioCliJson("list", "--cache", cache);
            Check(cached["models"]!.AsArray().Any(m => (string?)m!["name"] == "test/openvoice/converter" && (string?)m["kind"] == "huggingface"), $"the pulled subfolder listed by its id: {cached}");
            var fromCache = AudioCli("voice", "add", "cached", "--from", pcm, "--converter", "test/openvoice/converter", "--cache", cache, "-P", OpenVoicePluginPath, "--voices", store,
                "--consent", Consent, "-d", d);
            Check(fromCache.Code == 0 && fromCache.Out.Contains("installed 'cached'", StringComparison.Ordinal), $"voice add --converter owner/name/FOLDER from the cache: {fromCache.Out}\n{fromCache.Err}");
            Check(AudioCli("voice", "remove", "cached", "--voices", store).Code == 0, "the cached converter's voice removed");

            var text = AudioCli("voice", "show", "tester", "--voices", store);
            Check(text.Code == 0 && text.Out.Contains(Consent, StringComparison.Ordinal) && text.Out.Contains("agreement", StringComparison.Ordinal) && !text.Out.Contains("<- far", StringComparison.Ordinal),
                $"voice show (agreements under the default 0.8 flagged only): {text.Out}");
            var again = AudioCli([.. add, "--consent", Consent]);
            Check(again.Code == 1 && again.Err.Contains("--overwrite", StringComparison.Ordinal), $"a second voice of the name: {again.Err}");

            var list = AudioCliJson("voice", "list", "--voices", store);
            Check(list["voices"]!.AsArray().Select(v => (string?)v!["name"]).SequenceEqual(["tester"]), $"voice list: {list}");
            Check(AudioCli("voice", "ls", "--voices", store).Out.Contains("tester", StringComparison.Ordinal), "voice ls");

            // speak --voice: the embedding voice converted by the converter voice.json names, as VoicedSynthesizer does it.
            string spokenPath = Path.Combine(root, "tester.wav");
            string sentence = (string)JsonNode.Parse(File.ReadAllText(Path.Combine(model, "reference", "input.json")))!["text"]!;
            var spoken = AudioCli("speak", "-P", VitsPluginPath, "-P", OpenVoicePluginPath, model, sentence, "--voice", "tester", "--voices", store, "--noise", "0", "--duration-noise", "0",
                "--bits", "float", "--seed", "3", "-o", spokenPath, "-d", d);
            Check(spoken.Code == 0, $"speak --voice tester: {spoken.Err}");
            using (var voiced = new VoicedSynthesizer(SpeechSynthesisModels.Load(model, new SpeechSynthesisLoadOptions { Device = device }), new FolderVoiceStore(store)))
            {
                var options = new SpeechSynthesisOptions
                {
                    Voice = "tester", NoiseScale = 0, DurationNoiseScale = 0, Seed = 3, Conversion = new VoiceConversionOptions { Seed = 3 },
                };
                var (expected, _) = voiced.Speak(sentence, options);
                var written = AudioCodecs.Decode(spokenPath);
                Check(written.SampleRate == expected.SampleRate && written.Samples.AsSpan().SequenceEqual(expected.Samples), $"the converted voice: {written.Length} samples at {written.SampleRate} Hz");
            }

            var voices = AudioCliJson("speak", "-P", VitsPluginPath, model, "--list-voices", "--voices", store);
            Check(voices["voices"]!.AsArray().Select(v => (string?)v!["name"]).SequenceEqual(["tester"]), $"--list-voices: {voices}");

            string zip = Path.Combine(root, "tester.zip");
            Check(AudioCli("voice", "export", "tester", "-o", zip, "--voices", store).Code == 0 && File.Exists(zip), "voice export");
            Check(AudioCli("voice", "remove", "tester", "--voices", store).Code == 0 && AudioCli("voice", "show", "tester", "--voices", store).Code == 1, "voice remove, then show fails");
            Check(AudioCli("voice", "rm", "tester", "--voices", store).Code == 1, "voice rm of a voice not there: exit 1");
            var imported = AudioCliJson("voice", "import", zip, "--voices", store);
            Check((string?)imported["name"] == "tester" && (string?)imported["consent"]!["statement"] == Consent, $"voice import: {imported}");
            var twice = AudioCli("voice", "import", zip, "--voices", store);
            Check(twice.Code == 1 && twice.Err.Contains("installed already", StringComparison.Ordinal), $"importing over an installed voice: {twice.Err}");
            Check(AudioCli("voice", "import", zip, "--overwrite", "--voices", store).Code == 0, "voice import --overwrite");
        }
        finally
        {
            OpenVoicePlugin.Unregister();
            VitsPlugin.Unregister();
            Directory.Delete(root, recursive: true);
        }
    }

    private static void AudioCliDataAudio()
    {
        string root = TempFolder("cli-data-audio");
        try
        {
            // A 44.1 kHz stereo file: the fixture's 44.1 kHz tone on the left, half of it on the right.
            var tone = AudioCodecs.Decode(AudioData("wav/rate44k.wav"));
            var left = tone.Channel(0).ToArray();
            string stereo = Path.Combine(root, "in", "stereo44k.wav");
            AudioEncoders.Save(stereo, new AudioData([.. left, .. left.Select(v => v / 2)], 2, 44100), new AudioEncodeOptions { BitsPerSample = 32, Float = true });
            File.WriteAllText(Path.Combine(root, "texts.tsv"), "stereo44k.wav\tthe tone\npcm16|another\n");
            string output = Path.Combine(root, "out");
            var converted = AudioCliJson("data", "audio", stereo, AudioData("wav/pcm16.wav"), "-o", output, "--rate", "16000", "--mono", "--bits", "float", "--transcripts", Path.Combine(root, "texts.tsv"));
            Check((int)converted["files"]! == 2 && (int)converted["transcripts_attached"]! == 2, $"data audio -j: {converted}");
            var mono = AudioCodecs.Decode(Path.Combine(output, "stereo44k.wav"));
            var mean = left.Select(v => (v + v / 2) / 2).ToArray();
            var expected = Resampler.Resample(new AudioData(mean, 1, 44100), 16000).Samples;
            Check(mono.SampleRate == 16000 && mono.Channels == 1, $"{mono.Channels} channel(s) at {mono.SampleRate} Hz");
            AssertClose(expected, mono.Samples, 1e-5f, "the 16 kHz mono file against the resampled mean of the channels");
            AssertClose([.. ReadNpyFloat32(AudioData("resample/rate44k-16k.npy")).Select(v => v * 0.75f)], mono.Samples, 1e-5f, "against the fixture's 16 kHz resampling (scaled by the mean, 0.75)");
            var rows = File.ReadAllLines(Path.Combine(output, DataAudioCommand.ManifestFile)).Select(l => JsonNode.Parse(l)!.AsObject()).ToList();
            Check(rows.Count == 2 && (string?)rows[0]["name"] == "stereo44k.wav" && (int)rows[0]["rate"]! == 16000 && (int)rows[0]["channels"]! == 1 && (string?)rows[0]["text"] == "the tone"
                  && (string?)rows[1]["text"] == "another" && Math.Abs((double)rows[0]["seconds"]! - 1.5) < 1e-3, $"the manifest: {string.Join("\n", rows)}");

            // A two-tone file split at its silence: 0.5 s of 440 Hz, 0.6 s of silence, 0.4 s of 660 Hz.
            const int Rate = 16000;
            var samples = new float[(int)(1.5 * Rate)];
            for (int i = 0; i < samples.Length; i++)
            {
                double t = (double)i / Rate;
                samples[i] = t < 0.5 ? (float)(0.5 * Math.Sin(2 * Math.PI * 440 * t)) : t >= 1.1 ? (float)(0.5 * Math.Sin(2 * Math.PI * 660 * t)) : 0f;
            }

            string twoTone = Path.Combine(root, "in", "tones.wav");
            AudioEncoders.Save(twoTone, new AudioData(samples, 1, Rate), new AudioEncodeOptions { BitsPerSample = 32, Float = true });
            string pieces = Path.Combine(root, "pieces");
            var split = AudioCliJson("data", "audio", twoTone, "-o", pieces, "--split", "silence", "--bits", "float");
            var cut = File.ReadAllLines(Path.Combine(pieces, DataAudioCommand.ManifestFile)).Select(l => JsonNode.Parse(l)!.AsObject()).ToList();
            Check((int)split["files"]! == 2 && cut.Select(r => (string?)r["name"]).SequenceEqual(["tones-0001.wav", "tones-0002.wav"]), $"split at the silence: {string.Join("\n", cut)}");
            Check((double)cut[0]["start"]! == 0 && Math.Abs((double)cut[0]["end"]! - 0.6) < 0.011 && Math.Abs((double)cut[1]["start"]! - 1.0) < 0.011 && (double)cut[1]["end"]! == 1.5,
                $"the pieces keep 0.1 s of silence on each side: {cut[0]["start"]}-{cut[0]["end"]}, {cut[1]["start"]}-{cut[1]["end"]}");
            var first = AudioCodecs.Decode(Path.Combine(pieces, "tones-0001.wav"));
            Check(first.Samples.AsSpan().SequenceEqual(samples.AsSpan(0, first.Length)), "a piece holds the source's samples");
            var library = AudioSplitters.Get("silence").Split(new AudioData(samples, 1, Rate), new AudioSplitOptions());
            Check(library.Count == 2 && library[0].Length == first.Length, "the pieces are the library splitter's");

            string seconds = Path.Combine(root, "seconds");
            Check((int)AudioCliJson("data", "audio", twoTone, "-o", seconds, "--split", "seconds", "--max-seconds", "0.5")["files"]! == 3, "--split seconds --max-seconds 0.5: three pieces");
            var noLength = AudioCli("data", "audio", twoTone, "-o", seconds, "--split", "seconds");
            Check(noLength.Code == 2 && noLength.Err.Contains("--max-seconds", StringComparison.Ordinal), $"--split seconds without a length: {noLength.Err}");
            var unknown = AudioCli("data", "audio", twoTone, "-o", seconds, "--split", "nobody");
            Check(unknown.Code == 2 && unknown.Err.Contains("not registered", StringComparison.Ordinal), $"an unknown splitter: {unknown.Err}");

            var stats = AudioCliJson("data", "audio", Path.Combine(root, "in"), "--stats");
            Check(stats["out"] is null && (int)stats["stats"]!["files"]! == 2 && stats["stats"]!["rates"]!["16000"] is not null && stats["stats"]!["rates"]!["44100"] is not null
                  && !Directory.Exists(Path.Combine(root, "in", "out")), $"--stats alone: {stats}");
            var statsText = AudioCli("data", "audio", Path.Combine(root, "in"), "--stats", "--plain");
            Check(statsText.Code == 0 && statsText.Out.Contains("44100 Hz", StringComparison.Ordinal) && statsText.Out.Contains('#'), $"--stats text: {statsText.Out}");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void AudioCliDataMel(Device device)
    {
        string root = TempFolder("cli-data-mel");
        try
        {
            var reference = ReadNpyFloat32(AudioData("features/log-mel.npy"));         // torch's [80, 3000]
            float largest = reference.Max(MathF.Abs);
            var mel = AudioCliJson("data", "mel", AudioData("wav/pcm16.wav"), "-o", root, "--preprocessor", AudioData("tiny-whisper/preprocessor_config.json"), "-d", device.ToString());
            var (values, shape) = NpyFile.ReadFloat32(Path.Combine(root, "pcm16.npy"));
            Check(shape.SequenceEqual([3000, 80]) && (int)mel["files"]![0]!["frames"]! == 3000, $"[frames, mels]: [{string.Join(", ", shape)}]");
            var transposed = new float[values.Length];
            for (int f = 0; f < 3000; f++)
            {
                for (int m = 0; m < 80; m++)
                {
                    transposed[m * 3000 + f] = values[f * 80 + m];
                }
            }

            AssertClose(reference, transposed, 1e-4f * largest, "data mel's features against transformers' log-mel");
            var options = JsonNode.Parse(File.ReadAllText(Path.Combine(root, DataMelCommand.OptionsFile)))!.AsObject();
            Check((int)options["mels"]! == 80 && (int)options["hop_size"]! == 160 && (string?)options["extractor"] == "log-mel", $"features.json: {options}");

            string other = Path.Combine(root, "t");
            Check(AudioCli("data", "mel", AudioData("wav/pcm16.wav"), "-o", other, "--preprocessor", AudioData("tiny-whisper"), "--transpose", "-d", device.ToString()).Code == 0, "--transpose");
            var (direct, directShape) = NpyFile.ReadFloat32(Path.Combine(other, "pcm16.npy"));
            Check(directShape.SequenceEqual([80, 3000]), $"[mels, frames]: [{string.Join(", ", directShape)}]");
            AssertClose(reference, direct, 1e-4f * largest, "data mel --transpose against transformers' input_features");
            var bad = AudioCli("data", "mel", AudioData("wav/pcm16.wav"), "-o", other, "--mel-scale", "bark");
            Check(bad.Code == 2 && bad.Err.Contains("MelScale", StringComparison.Ordinal), $"a bad option: {bad.Err}");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void AudioCliRunAudio(Device device)
    {
        string d = device.ToString(), wav = AudioData("wav/pcm16.wav");
        var text = AudioCli("run", PretrainData("tiny-llama"), "Transcribe.", "--audio", wav, "--max-tokens", "2", "-d", d);
        Check(text.Code == 2 && text.Err.Contains("--audio", StringComparison.Ordinal) && text.Err.Contains("reads no sound", StringComparison.Ordinal), $"--audio on a text model: {text.Err}");
        var missing = AudioCli("run", PretrainData("tiny-llama"), "Transcribe.", "--audio", "nowhere.wav");
        Check(missing.Code == 2 && missing.Err.Contains("Sound file not found", StringComparison.Ordinal), $"a missing sound: {missing.Err}");

        // The tiny Llama with an audio_config, the audio template and a family registered from the test.
        string folder = TempFolder("cli-run-audio");
        try
        {
            foreach (string file in Directory.GetFiles(PretrainData("tiny-llama")))
            {
                File.Copy(file, Path.Combine(folder, Path.GetFileName(file)));
            }

            var config = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "config.json")))!.AsObject();
            config["audio_config"] = new JsonObject { ["mels"] = 80 };
            File.WriteAllText(Path.Combine(folder, "config.json"), config.ToJsonString());
            var tokenizer = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "tokenizer_config.json")))!.AsObject();
            tokenizer["chat_template"] = AudioTemplate;
            File.WriteAllText(Path.Combine(folder, "tokenizer_config.json"), tokenizer.ToJsonString());

            var unregistered = AudioCli("run", folder, "Transcribe.", "--audio", wav, "--max-tokens", "2", "-d", d);
            Check(unregistered.Code == 2 && unregistered.Err.Contains("AudioFamilies.Register", StringComparison.Ordinal) && unregistered.Err.Contains("-P", StringComparison.Ordinal),
                $"an audio part without its family: {unregistered.Err}");
            int width;
            using (var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
            {
                width = model.Spec.Dim;
            }

            AudioFamilies.Register(new FakeAudioFamily("LlamaForCausalLM", width));
            try
            {
                var answer = AudioCliJson("run", folder, "Transcribe.", "--audio", wav, "--max-tokens", "2", "--temperature", "0", "-d", d);
                Check(answer["text"] is not null && answer["audios"]!.AsArray().Select(a => (string?)a).SequenceEqual([wav]), $"run --audio answers: {answer}");
            }
            finally
            {
                AudioFamilies.Unregister("LlamaForCausalLM");
            }
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }

        // The OpenAI-style API's input_audio part: the registered "audio" kind's part, its media type from "format".
        var bytes = File.ReadAllBytes(wav);
        var body = new JsonObject
        {
            ["role"] = "user",
            ["content"] = new JsonArray(
                new JsonObject { ["type"] = "text", ["text"] = "What is said?" },
                new JsonObject { ["type"] = "input_audio", ["input_audio"] = new JsonObject { ["data"] = Convert.ToBase64String(bytes), ["format"] = "wav" } }),
        };
        var parts = OpenAiStyleParts(body);
        Check(parts.Count == 2 && parts[1] is ChatAudio { MediaType: "audio/wav" } sound && sound.Data.Span.SequenceEqual(bytes), $"input_audio: {string.Join(", ", parts.Select(p => p.Kind))}");
    }

    // The parts of the first message as the OpenAI-style endpoint reads a request's body.
    private static IReadOnlyList<ChatPart> OpenAiStyleParts(JsonObject message) =>
        Idrak.AspNetCore.CompletionsTranslation.Chat(new JsonObject { ["model"] = "m", ["messages"] = new JsonArray(message) }).Messages[0].Parts;

    private static void AudioCliHelp()
    {
        string[] commands = ["transcribe", "speak", "voice add", "voice list", "voice show", "voice remove", "voice export", "voice import", "data audio", "data mel"];
        foreach (string name in commands)
        {
            var (code, help, _) = AudioCli([.. name.Split(' '), "--help"]);
            Check(code == 0 && help.Contains("\nExamples:\n", StringComparison.Ordinal) && help.Contains($"idrak {name}", StringComparison.Ordinal)
                  && help.TrimEnd().Split('\n').All(l => l.Length <= 120), $"idrak {name} --help: {help}");
            Check(Idrak.Cli.Shared.Messages.Arabic.ContainsKey(CommandTable.All.First(c => c.Name == name).Summary), $"idrak {name}: no Arabic summary");
        }

        var pageOutput = new StringWriter();
        int topic = CommandLine.Run(["help", "audio"], pageOutput, new StringWriter());       // a topic takes no options after it
        string page = pageOutput.ToString();
        Check(topic == 0 && page.Contains("idrak transcribe", StringComparison.Ordinal) && page.Contains("wav", StringComparison.Ordinal) && page.Contains("silence", StringComparison.Ordinal),
            $"idrak help audio: {page}");
        var (group, _, groupError) = AudioCli("voice");
        Check(group == 2 && groupError.Contains("idrak voice add", StringComparison.Ordinal), $"'idrak voice': {groupError}");
        Check(Help.For(CommandTable.All.First(c => c.Name == "speak")).Contains("IDRAK_VOICES", StringComparison.Ordinal)
              && Help.For(CommandTable.All.First(c => c.Name == "transcribe")).Contains("IDRAK_STFT_PATH", StringComparison.Ordinal), "the environment sections");
        var (format, _, formatError) = AudioCli("list", "--format", "srt");
        Check(format == 2 && formatError.Contains("--format", StringComparison.Ordinal), $"--format srt is transcribe's own: {formatError}");
    }
}
