// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Models;
using Idrak.Models.Abstractions;

namespace Idrak.Cli.Commands.Train;

/// <summary>
/// <c>idrak create SPEC.json -o FOLDER</c>: a model that did not exist, its weights drawn from a seed, written as a checkpoint
/// of its family (<see cref="PretrainedModel.Create"/>), ready for <c>idrak pretrain</c> (plan 19).
/// </summary>
internal sealed class CreateCommand : Command
{
    public const string Format = "idrak-create/1";

    public override string Name => "create";

    public override string Summary => "Make a new model from a spec, its weights drawn from a seed, as a checkpoint to pretrain";

    public override string Usage => """
        SPEC.json -o FOLDER [options]

        Makes a model that did not exist: the network SPEC describes, every weight drawn from --seed, written to FOLDER
        as a checkpoint of its family (config.json, model.safetensors, the tokenizer and a chat template that takes
        images), so run, chat, serve, tune and pretrain read it like any model. The family must be one that writes a
        spec as config.json (LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM,
        GemmaForCausalLM, or one from a plug-in, -P). idrak create init writes a SPEC to start from.

        Arguments:
          SPEC.json  {"format": "idrak-create/1", "architecture": ..., "spec": {...}, "tokenizer": ..., ...}, or the
                     decoder spec alone (then --architecture and --tokenizer). The spec's keys are dim, layers, heads,
                     ffDim and maxPositions; kvHeads (heads), headDim (dim / heads), vocabulary (the tokenizer's, rounded
                     up to --vocab-multiple), rope ({"theta": 10000}), normEpsilon, activation (Silu, Gelu, Relu),
                     tieEmbeddings (true) and the rest of DecoderSpec's are optional. Paths are read from SPEC's folder.

        Options:
              --architecture NAME  the family (LlamaForCausalLM)
              --tokenizer FILE     the tokenizer.json (idrak data tokenizer learns one), or a model folder holding one
              --chat-template FILE a Jinja chat template (default: ChatML that takes image parts)
              --bos TOKEN          the token that begins a text (<s> when the tokenizer has it)
              --eos TOKEN          the token that ends a turn: generation stops at it (<|im_end|>)
              --pad TOKEN          the padding token (<pad> when the tokenizer has it)
              --stop TOKEN         another token generation stops at (repeat; </s> when the tokenizer has it)
              --init-std X         the normal draw's deviation of every weight matrix and embedding (0.02); 0 for
                                   Xavier-uniform matrices
              --dtype T            how the weights are stored: f32 (the default, for training), bf16 or f16
              --vocab-multiple N   a spec without a vocabulary takes the tokenizer's, rounded up to a multiple of N (64)
              --decoder MODEL      start the decoder from this model's weights (a text model pretrained first; its spec
                                   must be SPEC's), the rest drawn
              --overwrite          write into a folder that already holds files
              --seed N             the seed the weights are drawn from (0)
          -o, --out FOLDER         the model folder to write

        Examples:
          idrak create init -o spec.json
          idrak create spec.json -o my-model --seed 1
          idrak create spec.json -o my-model --tokenizer tokenizer.json --architecture Qwen3ForCausalLM
          idrak create init --vision -o vision-spec.json
          idrak create vision-spec.json -o my-vlm --decoder my-model -P Idrak.IdrakVision.dll
          idrak create vision-spec.json -o my-vlm -P Idrak.IdrakVision.dll
          idrak pretrain my-model -o run --tokens tokens.bin --steps 2000
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } =
        ["--architecture", "--tokenizer", "--chat-template", "--bos", "--eos", "--pad", "--stop", "--init-std", "--dtype", "--vocab-multiple", "--decoder", "--out"];

    public override IReadOnlyCollection<string> Flags { get; } = ["--overwrite"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-o"] = "--out" };

    public override int Run(CommandContext context)
    {
        string specPath = context.Argument(0, "SPEC.json (idrak create init writes one)");
        string output = context.Option("--out") ?? throw new UsageException("Give -o, --out FOLDER for the model to write.");
        if (!File.Exists(specPath))
        {
            throw new UsageException($"No spec file {specPath} (idrak create init writes one).");
        }

        JsonObject file;
        try
        {
            file = JsonNode.Parse(File.ReadAllText(specPath), documentOptions: CliConfig.ReadOptions) as JsonObject
                ?? throw new UsageException($"{specPath} is not a JSON object.");
        }
        catch (System.Text.Json.JsonException e)
        {
            throw new UsageException($"{specPath}: {e.Message}");
        }

        if (file["speech"] is JsonObject speech)
        {
            return CreateSpeech(context, specPath, file, speech, output);
        }

        bool wrapped = file["spec"] is JsonObject;
        if (wrapped && (string?)file["format"] is { } format && format != Format)
        {
            throw new UsageException($"{specPath} is in format '{format}', not {Format}.");
        }

        var vision = wrapped ? file["vision"] as JsonObject : null;
        var audio = wrapped ? file["audio"] as JsonObject : null;
        string folder = Path.GetDirectoryName(Path.GetFullPath(specPath))!;
        string? Text(string option, string key) => context.Option(option) ?? (wrapped ? (string?)file[key] : null);
        string? PathOf(string option, string key) => context.Option(option) ?? (wrapped && (string?)file[key] is { } p ? Path.Combine(folder, p) : null);

        string architecture = Text("--architecture", "architecture") ?? "LlamaForCausalLM";
        string tokenizerPath = PathOf("--tokenizer", "tokenizer") ?? throw new UsageException("Give --tokenizer FILE (idrak data tokenizer learns one), or \"tokenizer\" in the spec file.");
        if (!File.Exists(tokenizerPath) && !File.Exists(Path.Combine(tokenizerPath, "tokenizer.json")))
        {
            throw new UsageException($"No tokenizer.json at {tokenizerPath}.");
        }

        var tokenizer = BpeTokenizer.Load(tokenizerPath);
        int multiple = context.IntOption("--vocab-multiple", wrapped && (int?)file["vocab_multiple"] is int m ? m : 64);
        DecoderSpec spec;
        try
        {
            spec = SpecOf(wrapped ? file["spec"]!.AsObject() : file, tokenizer.VocabularySize, multiple);
        }
        catch (Exception e) when (e is InvalidDataException or InvalidOperationException or ArgumentException or FormatException or KeyNotFoundException)
        {
            throw new UsageException($"{specPath}: {e.Message}");
        }

        string? Token(string option, string key, string fallback) =>
            Text(option, key) is { } given ? (given == "none" ? null : given) : tokenizer.IdOf(fallback) is null ? null : fallback;
        var stops = context.Options("--stop") is { Count: > 0 } s ? [.. s]
            : wrapped && file["stop"] is JsonArray listed ? [.. listed.Select(t => (string)t!)]
            : tokenizer.IdOf("</s>") is null ? new List<string>() : ["</s>"];
        string? templatePath = PathOf("--chat-template", "chat_template");
        var files = new ModelFiles
        {
            Tokenizer = tokenizer,
            ChatTemplate = templatePath is null ? null : File.ReadAllText(templatePath),
            BosToken = Token("--bos", "bos", "<s>"),
            EosToken = Text("--eos", "eos") ?? "<|im_end|>",
            PadToken = Token("--pad", "pad", "<pad>"),
            StopTokens = stops,
            AddBosToken = wrapped && (bool?)file["add_bos"] == true,
        };
        float initStd = context.Option("--init-std") is { } std
            ? float.TryParse(std, NumberStyles.Float, CultureInfo.InvariantCulture, out float v) ? v : throw new UsageException($"--init-std is a number, not '{std}'.")
            : wrapped && (float?)file["init_std"] is float f ? f : 0.02f;
        var type = (Text("--dtype", "dtype") ?? "f32").ToLowerInvariant() switch
        {
            "f32" or "float32" => SafeTensorType.F32,
            "bf16" or "bfloat16" => SafeTensorType.BF16,
            "f16" or "float16" => SafeTensorType.F16,
            var t => throw new UsageException($"--dtype is f32, bf16 or f16, not '{t}'."),
        };
        int seed = context.Seed ?? (wrapped && (int?)file["seed"] is int fs ? fs : 0);

        PretrainedModel model;
        try
        {
            model = PretrainedModel.Create(output, spec, architecture, files, new CreateOptions
            {
                Device = Idrak.Abstraction.Device.Cpu, Seed = seed, InitStd = initStd > 0 ? initStd : null, Type = type, Overwrite = context.Flag("--overwrite"),
                Vision = (JsonObject?)vision?.DeepClone(),
                Parts = audio is null ? null : new Dictionary<string, JsonObject> { [Idrak.Abstraction.Generation.ChatParts.Audio] = (JsonObject)audio.DeepClone() },
                Decoder = PathOf("--decoder", "decoder"),
            });
        }
        catch (NotSupportedException e) when (ImageModelFiles.NotRegistered(e))
        {
            return ImageModelFiles.Refuse(context, e);
        }
        catch (Exception e) when (e is NotSupportedException or ArgumentException or IOException or InvalidDataException)
        {
            context.Error(e.Message);
            return ExitCodes.Failed;
        }

        using (model)
        {
            long parameters = model.Network.ParameterCount;
            context.Write(string.Create(CultureInfo.InvariantCulture,
                $"{architecture}: {parameters:N0} parameters ({spec.Layers} layers of width {spec.Dim}, {spec.Heads} heads, vocabulary {spec.Vocabulary:N0}, {spec.MaxPositions:N0} positions), seed {seed}, written to {model.Folder}"));
            context.Write($"next: idrak pretrain \"{output}\" -o run --tokens tokens.bin --steps 1000");
            context.WriteJson(new JsonObject
            {
                ["path"] = Path.GetFullPath(output), ["architecture"] = architecture, ["parameters"] = parameters, ["seed"] = seed,
                ["spec"] = spec.ToJson(), ["files"] = new JsonArray([.. Directory.GetFiles(output).Order(StringComparer.Ordinal).Select(f => (JsonNode)Path.GetFileName(f))]),
            });
        }

        return ExitCodes.Ok;
    }

    // A speech model made from its family's spec ("speech": the family's keys): a recognizer through SpeechModels.Create
    // into a folder SpeechModels.Load (idrak transcribe, idrak pretrain) reads, or, when the architecture names a
    // synthesizer family and the recognizer family refuses the spec (or there is none), a synthesizer through
    // SpeechSynthesisModels.Create (idrak speak, idrak pretrain --durations).
    private static int CreateSpeech(CommandContext context, string specPath, JsonObject file, JsonObject speech, string output)
    {
        string folder = Path.GetDirectoryName(Path.GetFullPath(specPath))!;
        string architecture = context.Option("--architecture") ?? (string?)file["architecture"] ?? (string?)speech["architecture"]
            ?? throw new UsageException($"{specPath}: give the speech family's \"architecture\" (IdrakAudio, from -P Idrak.IdrakAudio.dll).");
        string tokenizerPath = context.Option("--tokenizer") ?? ((string?)file["tokenizer"] is { } t ? Path.Combine(folder, t) : null)
            ?? throw new UsageException("Give --tokenizer FILE (idrak data tokenizer learns one from the transcripts), or \"tokenizer\" in the spec file.");
        if (!File.Exists(tokenizerPath) && !File.Exists(Path.Combine(tokenizerPath, "tokenizer.json")))
        {
            throw new UsageException($"No tokenizer.json at {tokenizerPath}.");
        }

        var tokenizer = BpeTokenizer.Load(tokenizerPath);
        var spec = (JsonObject)speech.DeepClone();
        spec["architecture"] = architecture;
        int seed = context.Seed ?? ((int?)file["seed"] ?? 0);
        IDisposable model;
        var options = new Idrak.Audio.SpeechCreateOptions { Device = Idrak.Abstraction.Device.Cpu, Seed = seed, Overwrite = context.Flag("--overwrite") };
        bool synthesizer = Idrak.Audio.Abstractions.SpeechSynthesisFamilies.Find(architecture) is not null;
        try
        {
            if (Idrak.Audio.Abstractions.SpeechFamilies.Find(architecture) is null && synthesizer)
            {
                model = Idrak.Audio.Abstractions.SpeechSynthesisModels.Create(output, spec, tokenizer, options);
            }
            else
            {
                try
                {
                    model = Idrak.Audio.SpeechModels.Create(output, spec, tokenizer, options);
                    synthesizer = false;
                }
                catch (NotSupportedException) when (synthesizer)
                {
                    model = Idrak.Audio.Abstractions.SpeechSynthesisModels.Create(output, spec, tokenizer, options);   // a synthesizer's task
                }
            }
        }
        catch (NotSupportedException e) when (ImageModelFiles.NotRegistered(e))
        {
            return ImageModelFiles.Refuse(context, e);
        }
        catch (Exception e) when (e is NotSupportedException or ArgumentException or IOException or InvalidDataException)
        {
            context.Error(e.Message);
            return ExitCodes.Failed;
        }

        using (model)
        {
            long parameters = model switch
            {
                Idrak.Audio.Abstractions.ITrainableSpeechModel trainable => trainable.Network.ParameterCount,
                Idrak.Audio.Abstractions.ITrainableSpeechSynthesizer trainable => trainable.Network.ParameterCount,
                _ => 0,
            };
            string task = (string?)spec["task"] ?? (synthesizer ? "synthesizer" : "recognizer");
            context.Write(string.Create(CultureInfo.InvariantCulture, $"{architecture} {task}: {parameters:N0} parameters, seed {seed}, written to {Path.GetFullPath(output)}"));
            context.Write(synthesizer
                ? $"next: idrak data audio transcripts.jsonl --align MY-CTC -o aligned, then idrak pretrain \"{output}\" --data transcripts.jsonl --durations aligned/alignments.jsonl -o run"
                : $"next: idrak pretrain \"{output}\" --data transcripts.jsonl --steps 2000 -o run");
            context.WriteJson(new JsonObject
            {
                ["path"] = Path.GetFullPath(output), ["architecture"] = architecture, ["task"] = task, ["parameters"] = parameters, ["seed"] = seed,
                ["files"] = new JsonArray([.. Directory.GetFiles(output).Order(StringComparer.Ordinal).Select(f => (JsonNode)Path.GetFileName(f))]),
            });
        }

        return ExitCodes.Ok;
    }

    // A decoder spec from a short one: what is left out is a Llama-style default, the vocabulary the tokenizer's.
    internal static DecoderSpec SpecOf(JsonObject given, int tokenizerSize, int multiple)
    {
        int Required(string key) => (int?)given[key] ?? throw new InvalidDataException($"The spec has no \"{key}\".");
        int dim = Required("dim"), heads = Required("heads");
        var basis = new DecoderSpec
        {
            Vocabulary = (tokenizerSize + Math.Max(1, multiple) - 1) / Math.Max(1, multiple) * Math.Max(1, multiple),
            Dim = dim, Layers = Required("layers"), Heads = heads, KvHeads = heads, HeadDim = dim / Math.Max(1, heads),
            FfDim = Required("ffDim"), MaxPositions = Required("maxPositions"), TieEmbeddings = true,
        }.ToJson();
        foreach (var (key, value) in given)
        {
            if (key == "rope" && value is JsonObject rope && basis["rope"] is JsonObject baseRope)
            {
                foreach (var (k, v) in rope)
                {
                    baseRope[k] = v?.DeepClone();
                }

                continue;
            }

            basis[key] = value?.DeepClone();
        }

        if (given["kvHeads"] is null)
        {
            basis["kvHeads"] = (int)basis["heads"]!;
        }

        if (given["headDim"] is null)
        {
            basis["headDim"] = (int)basis["dim"]! / Math.Max(1, (int)basis["heads"]!);
        }

        return DecoderSpec.FromJson(JsonNode.Parse(basis.ToJsonString())!.AsObject());
    }
}

/// <summary><c>idrak create init</c>: a commented spec file to start a new model from.</summary>
internal sealed class CreateInitCommand : Command
{
    // The vision part of create init --vision's file (IdrakVision's description; every key optional).
    private const string VisionPart = """
          "seed": 0,
          // The vision part: the description the vision family reads (IdrakVision here). Every key is optional.
          "vision": {
            "vision_config": {
              "hidden_size": 192, "intermediate_size": 768, "num_hidden_layers": 4, "num_attention_heads": 3,
              "image_size": 224, "patch_size": 14,
              "pool_size": 2           // the patch grid pooled 2 x 2: (224 / 14 / 2)² = 64 image tokens
            },
            "projector_config": { "depth": 2, "hidden_act": "gelu_pytorch_tanh" },
            "image_attention": "image-blocks",   // or "causal"
            "image_token": "<image>", "begin_image": "<start_of_image>", "end_image": "<end_of_image>",
            "init_std": 0.02
          }
        }
        """;

    // The audio part of create init --audio audio-language's file (IdrakAudio's audio-language description; every key optional).
    private const string AudioPart = """
          "seed": 0,
          // The audio part: the description the audio family reads (IdrakAudio's audio-language part here). Every key is optional.
          "audio": {
            "audio_config": {
              "mels": 80, "dim": 256, "layers": 6, "heads": 4, "ffn_dim": 1024,
              "subsample_layers": 2,   // each halves the frames: 100 frames a second become 25 steps
              "convolution_module": false
            },
            "audio_projector": { "stack": 4, "type": "mlp" },   // 4 steps a soft token: about 6 tokens a second
            "audio_token": "<audio>",          // the marker the chat template writes for a sound (one token of the tokenizer)
            "begin_audio": null, "end_audio": null,
            "audio_attention": "causal"
          }
        }
        """;

    // create init --audio ctc|seq2seq's file: a recognizer made from the IdrakAudio speech family's spec.
    private const string SpeechFile = """
        {
          // idrak create's spec of a speech recognizer (plan 21, S6). Paths are read from this file's folder.
          "format": "idrak-create/1",
          // The speech family (load its plug-in: idrak create ... -P Idrak.IdrakAudio.dll).
          "architecture": "IdrakAudio",
          // The tokenizer (idrak data tokenizer transcripts.txt --vocab N -o tokenizer.json learns one from the texts).
          "tokenizer": "tokenizer.json",
          "speech": {
            "task": "TASK",            // "ctc" (a CTC head), "seq2seq" (a text decoder attending to the encoder) or "tts" (speech from text)
            "mels": 80,                // the log-mel features' bands
            "dim": 256, "layers": 6, "heads": 4, "ffn_dim": 1024,
            "subsample_layers": 2, "subsample_kernel": 3,
            "convolution_module": false,   // a Conformer-style convolution module in each block
            "dropout": 0.0SEQ2SEQ
          },
          "seed": 0
        }

        """;

    public override string Name => "create init";

    public override string Summary => "Write a spec file to make a new model from (idrak create), every setting explained";

    public override string Usage => """
        [-o SPEC.json] [--vision | --audio TASK]

        Writes a spec file for idrak create: the family, the network's sizes (small ones to change: nothing in Idrak
        prefers a size), the tokenizer and the special tokens, every key with a comment. With --vision, a
        vision-language model of the IdrakVision family (its plug-in, samples/IdrakVision, with -P). With --audio, an
        IdrakAudio model (samples/IdrakAudio, with -P): audio-language (a chat model that hears), ctc or seq2seq (a
        speech recognizer) or tts (a synthesizer). Comments and trailing commas are allowed in the file.

        Options:
          -o, --out FILE  the file to write (spec.json)
              --vision    a vision-language model (IdrakVision): the tower, projector and image settings too
              --audio T   an IdrakAudio model: audio-language, ctc, seq2seq or tts

        Examples:
          idrak create init -o spec.json
          idrak create spec.json -o my-model
          idrak create init --vision -o vision-spec.json
          idrak create vision-spec.json -o my-vlm --decoder my-model -P Idrak.IdrakVision.dll
          idrak create init --audio ctc -o asr-spec.json
          idrak create init --audio audio-language -o audio-spec.json
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } = ["--out", "--audio"];

    public override IReadOnlyCollection<string> Flags { get; } = ["--vision"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-o"] = "--out" };

    public override int Run(CommandContext context)
    {
        string output = context.Option("--out") ?? "spec.json";
        if (File.Exists(output))
        {
            context.Error($"{output} exists; give another -o.");
            return ExitCodes.Failed;
        }

        string? audio = context.Option("--audio");
        if (audio is not (null or "audio-language" or "ctc" or "seq2seq" or "tts"))
        {
            throw new UsageException($"--audio is audio-language, ctc, seq2seq or tts, not '{audio}'.");
        }

        if (audio is not null && context.Flag("--vision"))
        {
            throw new UsageException("Give --vision or --audio, not both.");
        }

        if (audio is "ctc" or "seq2seq" or "tts")
        {
            string decoder = audio switch
            {
                "seq2seq" => ",\n    \"decoder_layers\": 4,       // the text decoder's blocks\n    \"max_target_positions\": 448   // its longest text in tokens",
                "tts" => ",\n    \"decoder_layers\": 4, \"decoder_dim\": 256, \"decoder_heads\": 4, \"decoder_ffn_dim\": 1024,   // the mel decoder\n"
                         + "    \"duration_filters\": 256   // the duration predictor's convolutions",
                _ => "",
            };
            string text = SpeechFile.ReplaceLineEndings("\n").Replace("TASK", audio, StringComparison.Ordinal).Replace("SEQ2SEQ", decoder, StringComparison.Ordinal);
            if (audio == "tts")
            {
                // A synthesizer reads text, not frames: no subsampling and no convolution module.
                text = string.Join('\n', text.Split('\n').Where(l => !l.Contains("\"subsample_layers\"", StringComparison.Ordinal) && !l.Contains("\"convolution_module\"", StringComparison.Ordinal)));
            }

            File.WriteAllText(output, text);
            context.Write($"wrote {output}; next: idrak create {output} -o {(audio == "tts" ? "my-tts" : "my-asr")} -P Idrak.IdrakAudio.dll");
            context.WriteJson(new JsonObject { ["path"] = Path.GetFullPath(output) });
            return ExitCodes.Ok;
        }

        File.WriteAllText(output, """
            {
              // idrak create's spec (plan 19). Paths are read from this file's folder.
              "format": "idrak-create/1",
              // The family: one that writes a spec as config.json (LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM,
              // Qwen3ForCausalLM, GemmaForCausalLM, or one from a plug-in).
              "architecture": "LlamaForCausalLM",
              // The tokenizer (idrak data tokenizer corpus.txt --vocab N -o tokenizer.json learns one).
              "tokenizer": "tokenizer.json",
              // The network. Change the sizes to the model you want; the device's memory decides what trains, not Idrak.
              "spec": {
                "dim": 256,            // the width
                "layers": 4,           // decoder blocks
                "heads": 4,            // query heads
                "kvHeads": 2,          // key/value heads (fewer than heads: grouped-query attention)
                "ffDim": 768,          // the feed-forward width
                "maxPositions": 1024,  // the longest sequence
                "rope": { "theta": 10000 },
                "normEpsilon": 1e-6,
                "activation": "Silu",
                "tieEmbeddings": true
                // "vocabulary" is the tokenizer's, rounded up to "vocab_multiple"; headDim is dim / heads.
              },
              "vocab_multiple": 64,
              // The tokens that begin a text, end a turn (generation stops there), pad, and also stop generation.
              "bos": "<s>",
              "eos": "<|im_end|>",
              "pad": "<pad>",
              "stop": ["</s>"],
              // null: ChatML that takes image parts. Or a .jinja file.
              "chat_template": null,
              "init_std": 0.02,
              "dtype": "f32",
              "seed": 0
            }

            """.ReplaceLineEndings("\n").Replace("\"architecture\": \"LlamaForCausalLM\",", context.Flag("--vision")
                ? "\"architecture\": \"IdrakVisionForConditionalGeneration\",   // load its plug-in: idrak create ... -P Idrak.IdrakVision.dll"
                : audio is not null ? "\"architecture\": \"IdrakAudioForConditionalGeneration\",   // load its plug-in: idrak create ... -P Idrak.IdrakAudio.dll"
                : "\"architecture\": \"LlamaForCausalLM\",", StringComparison.Ordinal)
            .Replace("  \"seed\": 0\n}", context.Flag("--vision") ? VisionPart.ReplaceLineEndings("\n")
                : audio is not null ? AudioPart.ReplaceLineEndings("\n") : "  \"seed\": 0\n}", StringComparison.Ordinal));
        context.Write($"wrote {output}; next: idrak create {output} -o my-model");
        context.WriteJson(new JsonObject { ["path"] = Path.GetFullPath(output) });
        return ExitCodes.Ok;
    }
}
