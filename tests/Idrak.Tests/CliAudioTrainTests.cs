// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Generation;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Cli.Shared;
using Idrak.Data;
using Idrak.IdrakAudio;
using Idrak.Models;

// Plan 21, S7's training commands: create, pretrain, resume and runs show of speech models, data audio --align, tune and
// run with an audio-language model's audio part, chat's /audio, each against the library's own call on the same data
// (the IdrakAudio plug-in given with -P as a user gives it). And S6's frame-level CTC distillation in the library.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] AudioCliTrainGroup =
    [
        ("audio cli: create init --audio ctc, seq2seq, tts and audio-language writes commented specs of their task; create makes a ctc recognizer SpeechModels.Load reads and a tts synthesizer SpeechSynthesisModels.Load reads, the same files as the library's Create with the seed; without -P the family is not registered (exit 1 naming -P)", d => { if (d == Device.Cpu) AudioCliCreate(); }),
        ("audio cli: pretrain trains a ctc model through SpeechTrainer (run.json, log.jsonl, a checkpoint; the library's losses), resume --steps goes on to the losses of one longer run, runs show names the trainer, a text option is refused; data audio --align writes the library's alignments; a tts model pretrains on them with --durations; --teacher distils a CTC student as SpeechTrainingOptions.Teacher", AudioCliPretrain),
        ("audio cli: tune --audio --parts audio-projector tunes the audio-language model on transcripts (the projector saved beside the adapters), run --audio --adapter reads the adapter's projector through the model's audio part, tune evaluate --audio scores the base model and the adapter, and chat's /audio adds a sound (refused on a text model)", d => { if (d == Device.Cpu) AudioCliTune(); }),
        ("audio train: CtcDistillation: a teacher equal to the student adds nothing (the loss is (1 - alpha) times CTC), alpha 0 is CTC alone, another teacher adds its divergence; a teacher of other classes and one for a non-CTC objective are refused; the options' JSON keeps alpha and the temperature", AudioTrainDistillation),
    ];

    private static string IdrakAudioPluginPath => typeof(IdrakAudioPlugin).Assembly.Location;

    private static readonly JsonDocumentOptions CommentedJson = new() { CommentHandling = JsonCommentHandling.Skip, AllowTrailingCommas = true };

    // A speech spec file for idrak create: the family, the tokenizer and the family's keys.
    private static string SpeechSpecFile(string folder, string name, JsonObject spec, string tokenizer)
    {
        var speech = (JsonObject)spec.DeepClone();
        string architecture = (string)speech["architecture"]!;
        speech.Remove("architecture");
        string path = Path.Combine(folder, name);
        File.WriteAllText(path, new JsonObject { ["format"] = "idrak-create/1", ["architecture"] = architecture, ["tokenizer"] = tokenizer, ["speech"] = speech, ["seed"] = 7 }.ToJsonString());
        return path;
    }

    private static void AudioCliCreate()
    {
        string root = TempFolder("cli-audio-create");
        IdrakAudioPlugin.Unregister();
        try
        {
            foreach (string task in (string[])["ctc", "seq2seq", "tts", "audio-language"])
            {
                string file = Path.Combine(root, $"{task}-spec.json");
                var (code, output, error) = AudioCli("create", "init", "--audio", task, "-o", file);
                Check(code == 0 && File.Exists(file), $"create init --audio {task}: {code} {output} {error}");
                var spec = JsonNode.Parse(File.ReadAllText(file), documentOptions: CommentedJson)!.AsObject();
                Check(task == "audio-language"
                        ? (string?)spec["architecture"] == IdrakAudioLanguageFamily.Architecture && spec["audio"]?["audio_config"] is JsonObject && spec["audio"]?["audio_projector"] is JsonObject
                        : (string?)spec["architecture"] == IdrakAudioFamily.Architecture && (string?)spec["speech"]?["task"] == task && (task != "tts" || spec["speech"]!["subsample_layers"] is null),
                    $"create init --audio {task} writes its task's spec: {spec}");
            }

            var bad = AudioCli("create", "init", "--audio", "music", "-o", Path.Combine(root, "x.json"));
            Check(bad.Code == 2 && bad.Err.Contains("audio-language, ctc, seq2seq or tts", StringComparison.Ordinal), $"an unknown --audio task: {bad.Err}");

            var (_, texts) = ToneSpeech(Path.Combine(root, "data"), 4, 5);
            var tokenizer = ToneTokenizer(root, texts);
            string tokenizerFile = Path.Combine(root, "tokenizer.json");
            tokenizer.Save(tokenizerFile);
            string ctcSpec = SpeechSpecFile(root, "ctc.json", ToneSpec(), tokenizerFile), ttsSpec = SpeechSpecFile(root, "tts.json", ToneTtsSpec(), tokenizerFile);

            var unregistered = AudioCli("create", ctcSpec, "-o", Path.Combine(root, "none"));
            Check(unregistered.Code == 1 && unregistered.Err.Contains("not registered", StringComparison.Ordinal) && unregistered.Err.Contains("-P", StringComparison.Ordinal),
                $"create without the plug-in: {unregistered.Code} {unregistered.Err}");

            string ctc = Path.Combine(root, "ctc"), tts = Path.Combine(root, "tts");
            var made = AudioCliJson("create", ctcSpec, "-o", ctc, "-P", IdrakAudioPluginPath);
            Check((string?)made["task"] == "ctc" && (long)made["parameters"]! > 0, $"create of the ctc spec: {made}");
            var spoken = AudioCliJson("create", ttsSpec, "-o", tts, "-P", IdrakAudioPluginPath);
            Check((string?)spoken["task"] == "tts" && (long)spoken["parameters"]! > 0, $"create of the tts spec: {spoken}");
            using (var recognizer = SpeechModels.Load(ctc))
            using (var synthesizer = SpeechSynthesisModels.Load(tts))
            {
                Check(recognizer is IdrakAudioCtcModel && synthesizer is IdrakAudioTtsModel, $"the folders read back: {recognizer.GetType().Name}, {synthesizer.GetType().Name}");
            }

            // The library's Create with the same spec, tokenizer and seed writes the same weights.
            CreateTone(Path.Combine(root, "ctc-library"), tokenizer, Device.Cpu).Dispose();
            CreateToneTts(Path.Combine(root, "tts-library"), tokenizer, Device.Cpu).Dispose();
            Check(File.ReadAllBytes(Path.Combine(ctc, "model.safetensors")).AsSpan().SequenceEqual(File.ReadAllBytes(Path.Combine(root, "ctc-library", "model.safetensors")))
                  && File.ReadAllBytes(Path.Combine(tts, "model.safetensors")).AsSpan().SequenceEqual(File.ReadAllBytes(Path.Combine(root, "tts-library", "model.safetensors"))),
                "create writes the library's weights for the seed");
        }
        finally
        {
            IdrakAudioPlugin.Unregister();
            Directory.Delete(root, recursive: true);
        }
    }

    // The trainer's options the pretrain command line below gives (constant rate, no warm-up, no feature cache).
    private static SpeechTrainingOptions CliToneTraining(int steps) => new()
    {
        Steps = steps, BatchSeconds = 3, LearningRate = 3e-3f, WarmupFraction = 0, Schedule = "constant", Seed = 3, FeatureCache = null,
    };

    private static string[] CliToneTrainingArgs(int steps) =>
        ["--steps", steps.ToString(System.Globalization.CultureInfo.InvariantCulture), "--batch-seconds", "3", "--lr", "3e-3", "--warmup", "0", "--schedule", "constant",
         "--seed", "3", "--feature-cache", "none"];

    private static void AudioCliPretrain(Device device)
    {
        string root = TempFolder("cli-audio-pretrain"), d = device.ToString();
        IdrakAudioPlugin.Register();
        try
        {
            var (transcripts, texts) = ToneSpeech(Path.Combine(root, "data"), 8, 17);
            var tokenizer = ToneTokenizer(root, texts);
            string model = Path.Combine(root, "ctc");
            CreateTone(model, tokenizer, Device.Cpu).Dispose();
            var source = new SpeechSource(SpeechSources.Transcripts, [Path.GetFullPath(transcripts)]);

            // The library's run of four steps.
            SpeechTrainingResult reference;
            using (var start = (IdrakAudioCtcModel)SpeechModels.Load(model, new SpeechModelOptions { Device = device }))
            {
                reference = SpeechTrainer.Train(start, source, null, CliToneTraining(4), Path.Combine(root, "reference"));
            }

            string run = Path.Combine(root, "run");
            var first = TrainCliJson(["pretrain", model, "--data", transcripts, "-o", run, "-P", IdrakAudioPluginPath, "-d", d, .. CliToneTrainingArgs(2)]);
            Check((float)first["final_loss"]! == reference.Losses[1] && (int)first["steps"]! == 2 && first["checkpoint"] is not null,
                $"pretrain's two steps end at the library's loss {reference.Losses[1]}: {first}");
            var settings = JsonNode.Parse(File.ReadAllText(Path.Combine(run, "run.json")))!.AsObject();
            Check((string?)settings["format"] == "idrak-speech-run/1" && (string?)settings["trainer"] == "recognizer" && (string?)settings["objective"] == "ctc"
                  && File.Exists(Path.Combine(run, "log.jsonl")), $"run.json and log.jsonl: {settings}");

            var resumed = TrainCliJson("resume", run, "--steps", "2", "-P", IdrakAudioPluginPath, "-d", d);
            Check((float)resumed["final_loss"]! == reference.Losses[3] && (int)resumed["steps"]! == 4, $"resume --steps 2 ends at the four-step run's loss {reference.Losses[3]}: {resumed}");
            var shown = TrainCli("runs", "show", run);
            Check(shown.Code == 0 && shown.Out.Contains("recognizer (ctc)", StringComparison.Ordinal), $"runs show: {shown.Out} {shown.Err}");
            var refused = TrainCli("pretrain", model, "--data", transcripts, "--tokens", "tokens.bin", "-o", Path.Combine(root, "x"), "-P", IdrakAudioPluginPath);
            Check(refused.Code == 2 && refused.Err.Contains("--tokens", StringComparison.Ordinal) && refused.Err.Contains("speech model", StringComparison.Ordinal),
                $"a text option on a speech model: {refused.Code} {refused.Err}");

            // data audio --align: the library's alignments of the same rows.
            string aligned = Path.Combine(root, "aligned");
            var alignment = AudioCliJson("data", "audio", transcripts, "--align", model, "-o", aligned, "-P", IdrakAudioPluginPath, "-d", d);
            string alignments = Path.Combine(aligned, "alignments.jsonl");
            Check((string?)alignment["alignments"] == alignments && File.Exists(alignments), $"data audio --align: {alignment}");
            using (var ctc = (IdrakAudioCtcModel)SpeechModels.Load(model, new SpeechModelOptions { Device = device }))
            {
                var expected = SpeechAlignments.Align(ctc, source).Alignments;
                var written = SpeechAlignments.Read(alignments);
                Check(written.Count == expected.Count && written.Zip(expected).All(p => p.First.Audio == p.Second.Audio && p.First.Labels.AsSpan().SequenceEqual(p.Second.Labels)
                        && p.First.Starts.AsSpan().SequenceEqual(p.Second.Starts) && p.First.Durations.AsSpan().SequenceEqual(p.Second.Durations)),
                    $"--align writes the library's {expected.Count} alignments ({written.Count})");
            }

            // The same utterances as a folder of name.wav + name.txt: --align on the folder (no -o: alignments.jsonl inside it),
            // and data tokenizer --from the transcripts and the folder, each as the library reads them.
            string pairs = Path.Combine(root, "pairs");
            Directory.CreateDirectory(pairs);
            foreach (var row in File.ReadAllLines(transcripts).Where(l => l.Length > 0).Select(l => JsonNode.Parse(l)!.AsObject()))
            {
                string name = (string)row["audio"]!;
                File.Copy(Path.Combine(Path.GetDirectoryName(transcripts)!, name), Path.Combine(pairs, name));
                File.WriteAllText(Path.Combine(pairs, Path.ChangeExtension(name, ".txt")), (string)row["text"]!);
            }

            var folderAlignment = AudioCliJson("data", "audio", pairs, "--align", model, "-P", IdrakAudioPluginPath, "-d", d);
            string folderAlignments = Path.Combine(pairs, "alignments.jsonl");
            Check((string?)folderAlignment["source"] == SpeechSources.AudioFolder && File.Exists(folderAlignments), $"data audio FOLDER --align: {folderAlignment}");
            using (var ctc = (IdrakAudioCtcModel)SpeechModels.Load(model, new SpeechModelOptions { Device = device }))
            {
                var expected = SpeechAlignments.Align(ctc, new SpeechSource(SpeechSources.AudioFolder, [Path.GetFullPath(pairs)])).Alignments;
                var written = SpeechAlignments.Read(folderAlignments);
                Check(written.Count == expected.Count && written.Zip(expected).All(p => p.First.Audio == p.Second.Audio && p.First.Starts.AsSpan().SequenceEqual(p.Second.Starts)),
                    $"--align on a folder writes the library's {expected.Count} alignments of it ({written.Count})");
            }

            int alphabet = string.Concat(texts).Distinct().Count() + 2;
            foreach (string data in (string[])[transcripts, pairs])
            {
                string tokenizerFile = Path.Combine(root, $"from-{Path.GetFileName(data)}.json");
                var learned = AudioCliJson("data", "tokenizer", "--from", data, "--vocab", alphabet.ToString(System.Globalization.CultureInfo.InvariantCulture), "--min-frequency", "1",
                    "--pre-tokenizer", "metaspace", "--no-special", "--workers", "1", "-o", tokenizerFile);
                Check(BpeTokenizer.Load(tokenizerFile).ToJson().ToJsonString() == tokenizer.ToJson().ToJsonString(), $"data tokenizer --from {Path.GetFileName(data)} learns the library's tokenizer of the texts: {learned}");
            }

            // A synthesizer trained on them.
            string tts = Path.Combine(root, "tts");
            CreateToneTts(tts, tokenizer, Device.Cpu).Dispose();
            SpeechTrainingResult spoken;
            using (var start = (IdrakAudioTtsModel)SpeechSynthesisModels.Load(tts, new SpeechSynthesisLoadOptions { Device = device }))
            {
                spoken = SpeechSynthesisTrainer.Train(start, source, null, CliToneTraining(2) with { Durations = Path.GetFullPath(alignments) }, Path.Combine(root, "tts-reference"));
            }

            var synthesis = TrainCliJson(["pretrain", tts, "--data", transcripts, "--durations", alignments, "-o", Path.Combine(root, "tts-run"), "-P", IdrakAudioPluginPath, "-d", d,
                .. CliToneTrainingArgs(2)]);
            Check((float)synthesis["final_loss"]! == spoken.Losses[1], $"pretrain of the tts model ends at the library's loss {spoken.Losses[1]}: {synthesis}");
            var noDurations = TrainCli(["pretrain", tts, "--data", transcripts, "-o", Path.Combine(root, "tts-none"), "-P", IdrakAudioPluginPath, .. CliToneTrainingArgs(1)]);
            Check(noDurations.Code == 2 && noDurations.Err.Contains("SpeechTrainingOptions.Durations", StringComparison.Ordinal), $"a synthesizer without --durations: {noDurations.Err}");

            // A CTC student of the trained model: --teacher as SpeechTrainingOptions.Teacher.
            string student = Path.Combine(root, "student");
            CreateTone(student, tokenizer, Device.Cpu, seed: 8).Dispose();
            SpeechTrainingResult taught;
            using (var teacher = (IdrakAudioCtcModel)SpeechModels.Load(first["checkpoint"]!.GetValue<string>(), new SpeechModelOptions { Device = device }))
            using (var start = (IdrakAudioCtcModel)SpeechModels.Load(student, new SpeechModelOptions { Device = device }))
            {
                taught = SpeechTrainer.Train(start, source, null, CliToneTraining(1) with { Teacher = teacher, DistillationAlpha = 0.25f, DistillationTemperature = 2f },
                    Path.Combine(root, "student-reference"));
            }

            var distilled = TrainCliJson(["pretrain", student, "--data", transcripts, "--teacher", first["checkpoint"]!.GetValue<string>(), "--alpha", "0.25", "--temperature", "2",
                "-o", Path.Combine(root, "student-run"), "-P", IdrakAudioPluginPath, "-d", d, .. CliToneTrainingArgs(1)]);
            Check((float)distilled["final_loss"]! == taught.Losses[0], $"--teacher gives the library's distilled loss {taught.Losses[0]}: {distilled}");
        }
        finally
        {
            IdrakAudioPlugin.Unregister();
            Directory.Delete(root, recursive: true);
        }
    }

    // A chat on standard input.
    private static (int Code, string Out, string Err) AudioCliChat(string input, params string[] args)
    {
        var output = new StringWriter();
        var error = new StringWriter();
        int code = StandardInput.With(new StringReader(input), () =>
            Idrak.Cli.CommandLine.Run([.. args, "-C", Path.Combine(Path.GetTempPath(), "idrak-cli-audio-no-config.json")], output, error));
        return (code, output.ToString(), error.ToString());
    }

    private static void AudioCliTune()
    {
        string root = TempFolder("cli-audio-tune"), model = AudioLanguageData("model"), wav = AudioData("wav/pcm16.wav");
        IdrakAudioPlugin.Unregister();
        try
        {
            string train = Path.Combine(root, "train.jsonl");
            File.WriteAllLines(train, [.. new[] { ("wav/pcm16.wav", "the quick brown fox"), ("wav/pcm8.wav", "hello there"), ("wav/list-chunk16.wav", "a short tone") }
                .Select(r => new JsonObject { ["audio"] = r.Item1, ["text"] = r.Item2 }.ToJsonString())]);
            string adapter = Path.Combine(root, "adapter");
            var tuned = TrainCli(["tune", "train", model, train, "--audio", AudioData(""), "--data-format", "transcripts", "--parts", "audio-projector", "--epochs", "1",
                "-o", adapter, "-P", IdrakAudioPluginPath, .. TinyTuning]);
            Check(tuned.Code == 0 && File.Exists(Path.Combine(adapter, "adapter_model.safetensors")), $"tune --audio: {tuned.Code}\n{tuned.Out}\n{tuned.Err}");
            Dictionary<string, float[]> saved;
            using (var reader = SafeTensorsReader.Open(Path.Combine(adapter, "adapter_model.safetensors")))
            {
                ITensorStore store = reader;
                saved = store.Names.Where(n => n.Contains("audio_projector.", StringComparison.Ordinal)).ToDictionary(n => n, n => store.Read(n));
                Check(saved.Count > 0 && store.Names.Any(n => n.Contains("lora_A", StringComparison.Ordinal)), $"adapters and the audio projector saved: {string.Join(", ", store.Names)}");
            }

            Check(JsonNode.Parse(File.ReadAllText(Path.Combine(adapter, "adapter_config.json")))!["modules_to_save"] is JsonArray { Count: > 0 }, "adapter_config.json lists modules_to_save");

            // The run's audio source with the adapter: the model's audio part with the adapter's projector.
            using (var loaded = PretrainedModel.Load(model, new PretrainedOptions { Device = Device.Cpu }))
            {
                loaded.LoadAdapter(adapter);
                var source = AudioInputs.For(loaded, out string? reason) as ITrainableSoftParts ?? throw new Exception($"no audio source: {reason}");
                using (source)
                {
                    var exported = source.Export(source.TrainableParts).ToDictionary(t => t.Name, t => t.Values);
                    Check(saved.All(p => exported.Keys.Any(k => p.Key.EndsWith(k, StringComparison.Ordinal) && exported[k].AsSpan().SequenceEqual(p.Value))),
                        $"the run reads the adapter's projector ({string.Join(", ", saved.Keys)} in {string.Join(", ", exported.Keys)})");
                }
            }

            var answer = AudioCliJson("run", model, "Transcribe.", "--audio", wav, "--adapter", adapter, "--max-tokens", "3", "--temperature", "0", "-P", IdrakAudioPluginPath, "-d", "cpu");
            Check(answer["text"] is not null, $"run --audio --adapter answers: {answer}");

            var evaluated = TrainCli("tune", "evaluate", model, train, "--audio", AudioData(""), "--data-format", "transcripts", "--adapter", adapter, "--metric", "cer",
                "--max-new", "4", "-P", IdrakAudioPluginPath, "-d", "cpu");
            Check(evaluated.Code == 0 && System.Text.RegularExpressions.Regex.Matches(evaluated.Out.ReplaceLineEndings("\n"), @"loss [0-9.]+, cer [0-9.]+ on 3 answers").Count == 2,
                $"tune evaluate --audio: {evaluated.Code} {evaluated.Out} {evaluated.Err}");

            var chat = AudioCliChat($"/audio {wav}\n/audio nowhere.wav\n/exit\n", "chat", model, "-P", IdrakAudioPluginPath, "-d", "cpu");
            Check(chat.Code == 0 && chat.Out.Contains($"{wav} will be sent with the next message.", StringComparison.Ordinal) && chat.Out.Contains("No file nowhere.wav.", StringComparison.Ordinal),
                $"chat /audio: {chat.Code} {chat.Out} {chat.Err}");
            var text = AudioCliChat($"/audio {wav}\n/exit\n", "chat", PretrainData("tiny-llama"), "-d", "cpu");
            Check(text.Code == 0 && text.Out.Contains("reads no sound", StringComparison.Ordinal), $"chat /audio on a text model: {text.Out} {text.Err}");
        }
        finally
        {
            IdrakAudioPlugin.Unregister();
            Directory.Delete(root, recursive: true);
        }
    }

    private static void AudioTrainDistillation(Device device)
    {
        string root = TempFolder("ctc-distillation");
        IdrakAudioPlugin.Register();
        try
        {
            var (transcripts, texts) = ToneSpeech(Path.Combine(root, "data"), 6, 23);
            var tokenizer = ToneTokenizer(root, texts);
            var source = new SpeechSource(SpeechSources.Transcripts, [transcripts]);
            var options = CliToneTraining(1);
            float Loss(string name, int seed, Func<SpeechTrainingOptions, SpeechTrainingOptions> with, int teacherSeed = -1)
            {
                using var student = CreateTone(Path.Combine(root, name), tokenizer, device, seed);
                using var teacher = teacherSeed < 0 ? null : teacherSeed == seed ? null : CreateTone(Path.Combine(root, name + "-teacher"), tokenizer, device, teacherSeed);
                var given = with(options);
                if (teacherSeed == seed)
                {
                    given = given with { Teacher = student };                       // the student teaching itself
                }
                else if (teacher is not null)
                {
                    given = given with { Teacher = teacher };
                }

                return SpeechTrainer.Train(student, source, null, given, Path.Combine(root, name + "-run")).Losses[0];
            }

            float ctc = Loss("plain", 7, o => o);
            float self = Loss("self", 7, o => o with { DistillationAlpha = 0.5f }, teacherSeed: 7);
            float none = Loss("alpha0", 7, o => o with { DistillationAlpha = 0f }, teacherSeed: 9);
            float other = Loss("other", 7, o => o with { DistillationAlpha = 0.5f }, teacherSeed: 9);
            Check(Math.Abs(self - 0.5f * ctc) <= 1e-6f * Math.Abs(ctc), $"a teacher equal to the student adds nothing: {self} against half of {ctc}");
            Check(none == ctc, $"alpha 0 is the CTC loss: {none} against {ctc}");
            Check(other > 0.5f * ctc, $"another teacher adds its divergence: {other} against half of {ctc}");

            // A teacher of other classes, and one for a seq2seq student.
            var wider = ToneTokenizer(Directory.CreateDirectory(Path.Combine(root, "wide")).FullName, [.. texts, "xyz"]);
            using (var student = CreateTone(Path.Combine(root, "refused"), tokenizer, device))
            using (var teacher = CreateTone(Path.Combine(root, "refused-teacher"), wider, device))
            {
                ThrowsNaming<ArgumentException>(() => SpeechTrainer.Train(student, source, null, options with { Teacher = teacher }, Path.Combine(root, "refused-run")),
                    "a teacher of other classes", "classes", "the same labels");
            }

            using (var sequence = CreateToneSeq(Path.Combine(root, "seq"), tokenizer, device))
            using (var teacher = CreateTone(Path.Combine(root, "seq-teacher"), tokenizer, device))
            {
                ThrowsNaming<ArgumentException>(() => SpeechTrainer.Train(sequence, source, null, options with { Teacher = teacher }, Path.Combine(root, "seq-run")),
                    "a teacher for a seq2seq student", "teaches a CTC student", "pseudo-labels");
            }

            var json = SpeechTrainingOptions.FromJson((options with { DistillationAlpha = 0.3f, DistillationTemperature = 3f }).ToJson());
            Check(json.DistillationAlpha == 0.3f && json.DistillationTemperature == 3f && json.Teacher is null, "the options' JSON keeps alpha and the temperature, not the teacher");
            using (var student = CreateTone(Path.Combine(root, "bad"), tokenizer, device))
            {
                ThrowsNaming<ArgumentException>(() => SpeechTrainer.Train(student, source, null, options with { DistillationAlpha = 2f }, Path.Combine(root, "bad-run")),
                    "alpha out of range", "DistillationAlpha 0 to 1");
            }
        }
        finally
        {
            IdrakAudioPlugin.Unregister();
            Directory.Delete(root, recursive: true);
        }
    }
}
