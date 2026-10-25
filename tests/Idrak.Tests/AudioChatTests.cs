// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.IO.Compression;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Text;
using Idrak.Text.Abstractions;
using Idrak.Text.Generation;

// Plan 21, phase S3: the "audio" chat part, the soft-part contract in generation and tuning, audio in the tuning data
// formats, and the audio-language contracts. The files come from tools/audio/make_tiny_audio.py (wav/, transcripts/).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] AudioChatGroup =
    [
        ("audio chat: ChatAudio from bytes, files, streams and data URLs (media types sniffed for WAV, FLAC, MP3, Ogg and MP4 or given; SHA-256 identity; equality by bytes; bad data URLs refused); the 'audio' kind is a library default an app's kind shadows and Unregister restores; the kit's ChatPartKindSuite passes on its own samples and the fixture's files", d => { if (d == Device.Cpu) AudioChatPart(); }),
        ("audio chat: a message with text and audio round-trips through the chat JSON (ChatJson, ChatTranscript) and ContentToJson; text-only models and templates refuse an audio part naming 'audio'; a Jinja template that writes a marker for audio parts takes them", AudioChatJsonAndRefusal),
        ("audio chat: sharegpt reads <audio> markers with the record's \"audios\" list (from the file's folder, an audios root and a zip) and messages reads audio parts by path, bare with the list, and <audio> markers, each part carrying the fixture's WAV bytes; a count that disagrees or a missing file names the record", d => { if (d == Device.Cpu) AudioChatTuningData(); }),
        ("audio chat: the transcript encoder with a fake soft-part source (two tokens per sound) expands each marker to its soft tokens, records the parts' blocks, trains the answer alone, keeps whole blocks when cut, and refuses a sound in an assistant turn or without a source; the training step reads the parts through FineTuningOptions.Parts and refuses them without it", AudioChatTranscriptEncoder),
        ("audio chat: ChatGenerator with the same source on the tiny Llama renders the soft tokens in place, encodes each sound once per request, answers, and the features land at the soft tokens' positions; ChatBatch refuses soft parts and StreamBatch answers them one by one", AudioChatGenerator),
        ("audio chat: AudioLanguage.Attach is null for a model without audio_config, names AudioFamilies.Register for one with an unregistered family, and reads the part through a family registered from the test; ChatAudioParts caches features by the sound's hash", AudioChatAttach),
    ];

    private static readonly int[] TinyLlamaAudioMarker = [5, 6, 7];   // the fixture's <start_of_image>, <image>, <end_of_image>: reused as the fake family's audio marker, soft token and end

    // The tiny Llama's ChatML template with an audio branch (its own writes <image> for image parts only): an audio part
    // is written as the marker the fake family expands.
    private const string AudioTemplate = """
        {%- for message in messages -%}
        {{- '<|im_start|>' + message['role'] + '\n' -}}
        {%- if message['content'] is string -%}
        {{- message['content'] -}}
        {%- else -%}
        {%- for item in message['content'] -%}
        {%- if item['type'] == 'image' -%}
        {{- '<image>' -}}
        {%- elif item['type'] == 'audio' -%}
        {{- '<start_of_image>' -}}
        {%- elif item['type'] == 'text' -%}
        {{- item['text'] -}}
        {%- endif -%}
        {%- endfor -%}
        {%- endif -%}
        {{- '<|im_end|>\n' -}}
        {%- endfor -%}
        {%- if add_generation_prompt -%}
        {{- '<|im_start|>assistant\n' -}}
        {%- endif -%}
        """;

    // A soft-part source for audio parts: every sound takes `tokens` soft tokens, its features a seeded pattern of the sound's hash.
    private sealed class FakeAudioSource(int tokens, int width, Device device) : ISoftPartSource
    {
        public List<IReadOnlyList<ChatPart>> EncodeCalls { get; } = [];

        public string Kind => ChatParts.Audio;

        public int Width => width;

        public IImagePromptFormat Format { get; } = new ImageTokenFormat("fake-audio", TinyLlamaAudioMarker[0], TinyLlamaAudioMarker[1], end: TinyLlamaAudioMarker[2]);

        public IImageAttentionRule Attention => ImageAttentionRules.Get(ImageAttentionRules.Causal);

        public IReadOnlyList<ImageTokenLayout> Blocks(ChatPart part) => part is ChatAudio ? [new ImageTokenLayout(tokens)] : throw new ArgumentException($"not audio: {part.Kind}");

        public IReadOnlyList<ImageFeatures> Encode(IReadOnlyList<ChatPart> parts)
        {
            EncodeCalls.Add(parts);
            return [.. parts.Select(p =>
            {
                var random = new Random(((ChatAudio)p).Hash.GetHashCode(StringComparison.Ordinal));
                var values = new float[tokens * width];
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = (float)(random.NextDouble() - 0.5) * 0.1f;
                }

                return new ImageFeatures(Tensor.From(values, [tokens, width], device), new ImageTokenLayout(tokens));
            })];
        }
    }

    private static void AudioChatPart()
    {
        byte[] wav = File.ReadAllBytes(AudioData("wav/pcm16.wav"));
        var fromFile = ChatAudio.FromFile(AudioData("wav/pcm16.wav"));
        var fromBytes = ChatAudio.FromBytes(wav);
        Check(fromFile.MediaType == "audio/wav" && fromFile.Hash == Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(wav)) && fromFile.Hash.Length == 64
              && fromFile == fromBytes && fromFile.Data.Span.SequenceEqual(wav) && fromFile.Kind == "audio" && fromFile.ToString().StartsWith("audio/wav, 48044 bytes", StringComparison.Ordinal),
            $"from a file: {fromFile}");
        using (var stream = new MemoryStream(wav))
        {
            Check(ChatAudio.FromStream(stream, wav.Length) == fromFile, "from a stream");
        }

        Check(ChatAudio.FromBytes(wav, "Audio/X-Wave").MediaType == "audio/x-wave" && ChatAudio.FromBytes(wav, "audio/x-wave") == fromFile, "a given media type, equality by bytes");
        var sniffed = new (byte[] Bytes, string? Type)[]
        {
            ([(byte)'f', (byte)'L', (byte)'a', (byte)'C', 0, 0, 0, 34], "audio/flac"),
            ([(byte)'I', (byte)'D', (byte)'3', 4, 0, 0], "audio/mpeg"),
            ([0xFF, 0xFB, 0x90, 0x00], "audio/mpeg"),
            ([0xFF, 0xF8, 0x90, 0x00], null),                                 // a sync with layer 00 (reserved): not a frame
            ([(byte)'O', (byte)'g', (byte)'g', (byte)'S', 0, 2], "audio/ogg"),
            ([0, 0, 0, 24, (byte)'f', (byte)'t', (byte)'y', (byte)'p', (byte)'M', (byte)'4', (byte)'A', (byte)' '], "audio/mp4"),
            ([1, 2, 3], null),
        };
        foreach (var (bytes, type) in sniffed)
        {
            Check(ChatAudio.FromBytes(bytes).MediaType == type, $"sniffed {ChatAudio.FromBytes(bytes).MediaType ?? "null"} for {Convert.ToHexString(bytes)}, expected {type ?? "null"}");
        }

        string url = fromFile.ToDataUrl();
        Check(url.StartsWith("data:audio/wav;base64,", StringComparison.Ordinal) && ChatAudio.FromDataUrl(url) == fromFile
              && ChatAudio.FromDataUrl($"data:;base64,{Convert.ToBase64String(wav)}").MediaType == "audio/wav", "data URLs round-trip; a type-less one is sniffed");
        Throws<FormatException>(() => ChatAudio.FromDataUrl("data:image/png;base64,AAAA"), "an image data URL");
        Throws<FormatException>(() => ChatAudio.FromDataUrl("data:audio/wav,AAAA"), "not base64");
        Throws<FormatException>(() => ChatAudio.FromDataUrl("data:audio/wav;base64,***"), "bad base64");
        Throws<ArgumentException>(() => ChatAudio.FromBytes([]), "empty bytes");
        Throws<FileNotFoundException>(() => ChatAudio.FromFile(AudioData("wav/none.wav")), "a missing file");
        Check(fromFile.Decode().Length == AudioCodecs.Decode(AudioData("wav/pcm16.wav")).Length, "Decode goes through the codecs");

        // The registry: a library default from Idrak.Audio, behind an app's kind of the same name.
        Check(ChatParts.Origin("audio") == Overrides.Library && ChatParts.Default("audio") is AudioPartKind && ChatParts.Get("audio") is AudioPartKind
              && ChatParts.Names.Contains("audio"), "the library's audio kind");
        var json = ChatParts.ToJson(fromFile);
        var expectedJson = new JsonObject { ["type"] = "audio", ["media_type"] = "audio/wav", ["data"] = Convert.ToBase64String(wav) };   // the serializer's own escaping of base64's '+'
        Check(json.ToJsonString() == expectedJson.ToJsonString() && ChatParts.FromJson(json) == fromFile, $"the part's JSON: {json.ToJsonString()[..60]}…");
        Throws<FormatException>(() => ChatParts.FromJson(new JsonObject { ["type"] = "audio" }), "audio with no data");
        Throws<FormatException>(() => ChatParts.FromJson(new JsonObject { ["type"] = "audio", ["data"] = "***" }), "bad base64");
        var library = ChatParts.Default("audio")!;
        ChatParts.Register(new ShadowingAudioKind());
        try
        {
            Check(ChatParts.Origin("audio") == typeof(Tests).Assembly.GetName().Name && ReferenceEquals(ChatParts.Default("audio"), library) && ChatParts.FromJson(json) == fromFile,
                "the app's kind shadows the default, which stays behind it");
        }
        finally
        {
            ChatParts.Unregister("audio");
        }

        Check(ReferenceEquals(ChatParts.Get("audio"), library) && !ChatParts.Unregister("audio"), "Unregister brings the default back");

        // The kit: the suite's own samples (no samples given), and the fixture's files.
        Conformance.CheckChatPartKind("audio").ThrowIfFailed();
        var samples = WavFixtureFiles().Select(f => (JsonObject)ChatParts.ToJson(ChatAudio.FromFile(AudioData($"wav/{f.Name}.wav")))).ToList();
        Conformance.CheckChatPartKind("audio", samples: samples, options: new ContractCheckOptions { RandomCases = 4 }).ThrowIfFailed();
    }

    private sealed class ShadowingAudioKind : IChatPartKind
    {
        public string Name => "audio";

        public string Summary => "the library's audio, passed through";

        public void Write(ChatPart part, JsonObject json) => ChatParts.Default("audio")!.Write(part, json);

        public ChatPart Read(JsonObject json) => ChatParts.Default("audio")!.Read(json);
    }

    private static void AudioChatJsonAndRefusal(Device device)
    {
        var audio = ChatAudio.FromFile(AudioData("wav/pcm16.wav"));
        var message = new ChatMessage("user", [audio, new ChatText("Transcribe.")]);
        var json = ChatJson.Messages([message]);
        var back = ChatTranscript.FromJson(new JsonObject { ["messages"] = json }).Messages.Single();
        Check(back == message && back.Content == "Transcribe." && back.Parts[0] is ChatAudio a && a == audio, $"through the chat JSON: {json.ToJsonString()[..80]}");
        Check(ChatParts.ContentToJson(message.Parts) is JsonArray { Count: 2 } content && (string?)content[0]!["type"] == "audio"
              && ChatParts.ContentFromJson(content) is [ChatAudio, ChatText], "content with an audio part is an array");

        ChatMessage[] withAudio = [new("system", "Be brief."), message];
        void Refused(Action action, string who)
        {
            try
            {
                action();
                Check(false, $"{who} refuses a sound");
            }
            catch (Exception e) when (e is NotSupportedException || e.InnerException is NotSupportedException)
            {
                string text = (e as NotSupportedException ?? e.InnerException!).Message;
                Check(text.Contains(who, StringComparison.Ordinal) && text.Contains("message 2 (user)", StringComparison.Ordinal) && text.Contains("kind 'audio'", StringComparison.Ordinal), text);
            }
        }

        Refused(() => new ChatMLTemplate().Render(withAudio, [], null), "ChatMLTemplate");
        var (model, tokenizer) = TinyLanguageModel(device);
        using var _ = model;
        var chat = new ChatGenerator(new TextGenerator(model, tokenizer, 32));
        var options = new GenerationOptions { Seed = 3, NumPredict = 4 };
        Check(((IChatModel)chat).PartKinds.SetEquals(["text"]), "text only without Parts");
        Refused(() => chat.Chat(new ChatRequest(withAudio, Options: options)), "ChatGenerator");
        Refused(() => chat.RenderPrompt(new ChatRequest(withAudio)), "ChatGenerator");
        Refused(() => chat.ChatBatch([new ChatRequest(withAudio)]), "ChatGenerator");
        var fake = FakeChatModel.Script(FakeChatModel.Answer("ok"));
        Refused(() => fake.ChatAsync(new ChatRequest(withAudio)).GetAwaiter().GetResult(), "FakeChatModel");

        // A Jinja template that writes something for an audio part takes it (the probe finds every registered kind).
        var template = new JinjaChatTemplate(AudioTemplate, ["<|im_end|>"]);
        Check(template.PartKinds.SetEquals(["text", "image", "audio"]), $"the template's kinds: {string.Join(", ", template.PartKinds)}");
        string rendered = template.Render(withAudio, [], null, addGenerationPrompt: true);
        Check(rendered.Contains("<start_of_image>Transcribe.", StringComparison.Ordinal), rendered);
        // The source's line endings are the checkout's (CRLF on Windows): the branch is cut from the text with LF endings.
        string withoutAudio = AudioTemplate.ReplaceLineEndings("\n").Replace("{%- elif item['type'] == 'audio' -%}\n{{- '<start_of_image>' -}}\n", "", StringComparison.Ordinal);
        Check(!withoutAudio.Contains("'audio'", StringComparison.Ordinal), "the audio branch is cut from the template");
        var imagesOnly = new JinjaChatTemplate(withoutAudio, ["<|im_end|>"]);
        Check(imagesOnly.PartKinds.SetEquals(["text", "image"]), $"a template without an audio branch keeps text and image: {string.Join(", ", imagesOnly.PartKinds)}");
    }

    private static void AudioChatTuningData()
    {
        string file = AudioData("transcripts/train-sharegpt.jsonl");
        var expected = File.ReadAllLines(file).Select(l => JsonNode.Parse(l)!.AsObject()).Select(r => ((string)r["audios"]![0]!, (string)r["conversations"]![1]!["value"]!)).ToList();
        void CheckRead(IEnumerable<ChatTranscript> transcripts, string what)
        {
            var read = transcripts.ToList();
            Check(read.Count == expected.Count, $"{what}: {read.Count} records");
            for (int i = 0; i < read.Count; i++)
            {
                var (path, answer) = expected[i];
                var messages = read[i].Messages;
                Check(messages is [{ Role: "user", Parts: [ChatAudio audio, ChatText { Text: "Transcribe." }] }, { Role: "assistant" }] && messages[1].Content == answer
                      && audio.Data.Span.SequenceEqual(File.ReadAllBytes(AudioData(path))) && audio.MediaType == "audio/wav",
                    $"{what}: record {i + 1}: {string.Join(" | ", messages.Select(m => $"{m.Role}: {string.Join(", ", m.Parts.Select(p => p.ToString()))}"))}");
            }
        }

        var format = TuningDataFormats.Get("sharegpt");
        Check(TuningDataFormats.Detect(file).Name == "sharegpt", "detected");
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-audio-chat-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        try
        {
            // The fixture's paths are relative to the audio folder: a copy with its sounds beside it reads from its own folder.
            string beside = Path.Combine(folder, "beside", "train.jsonl");
            foreach (var (path, _) in expected.DistinctBy(e => e.Item1))
            {
                Directory.CreateDirectory(Path.GetDirectoryName(Path.Combine(folder, "beside", path))!);
                File.Copy(AudioData(path), Path.Combine(folder, "beside", path));
            }

            File.Copy(file, beside);
            CheckRead(format.Read(beside), "from the file's folder");
            using (var pending = format.ReadPending(beside))
            {
                Check(pending.Count == expected.Count && pending.Images == 0, "pending: counted");
                CheckRead(Enumerable.Range(0, pending.Count).Select(pending.Read), "pending");
            }

            // The same records with no sounds beside them: the audios root a folder, then a zip; the images root alone serves audio too.
            CheckRead(format.Read(file, new TuningDataOptions { Audios = AudioData("") }), "the fixture with its audios root");
            string copy = Path.Combine(folder, "train.jsonl");
            File.Copy(file, copy);
            CheckRead(format.Read(copy, new TuningDataOptions { Audios = AudioData("") }), "an audios root");
            CheckRead(format.Read(copy, new TuningDataOptions { Images = AudioData("") }), "the images root serves sounds");
            string zip = Path.Combine(folder, "audios.zip");
            using (var archive = ZipFile.Open(zip, ZipArchiveMode.Create))
            {
                foreach (var (path, _) in expected.DistinctBy(e => e.Item1))
                {
                    archive.CreateEntryFromFile(AudioData(path), "sounds/" + path);
                }
            }

            CheckRead(format.Read(copy, new TuningDataOptions { Audios = zip }), "a zip");
            ThrowsNaming<DirectoryNotFoundException>(() => format.Read(copy, new TuningDataOptions { Audios = Path.Combine(folder, "none") }).ToList(), "a root that is not there", "audios root", "neither a folder nor a .zip");

            // The messages format: a path, a bare part with the list, a marker in a text part, and a data URL.
            string wav = AudioData("wav/pcm16.wav");
            string messages = Path.Combine(folder, "messages.jsonl");
            File.WriteAllLines(messages,
            [
                $$"""{"messages": [{"role": "user", "content": [{"type": "audio", "path": "{{wav.Replace("\\", "/")}}"}, {"type": "text", "text": "Transcribe."}]}, {"role": "assistant", "content": "a"}]}""",
                """{"messages": [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": "Transcribe."}]}, {"role": "assistant", "content": "b"}], "audios": ["wav/pcm16.wav"]}""",
                """{"messages": [{"role": "user", "content": "<audio>Transcribe."}, {"role": "assistant", "content": "c"}], "audios": ["wav/pcm16.wav"]}""",
                $$"""{"messages": [{"role": "user", "content": [{"type": "audio", "audio": "{{ChatAudio.FromFile(wav).ToDataUrl()}}"}, {"type": "text", "text": "Transcribe."}]}, {"role": "assistant", "content": "d"}]}""",
                """{"messages": [{"role": "user", "content": "<audio> is kept as text without an audios list"}, {"role": "assistant", "content": "e"}]}""",
            ]);
            var read = TuningDataFormats.Get("messages").Read(messages, new TuningDataOptions { Audios = AudioData("") }).ToList();
            Check(read.Count == 5 && read.Take(4).All(t => t.Messages[0].Parts is [ChatAudio a, ChatText { Text: "Transcribe." }] && a.Data.Span.SequenceEqual(File.ReadAllBytes(wav)))
                  && read[4].Messages[0].Parts is [ChatText { Text: "<audio> is kept as text without an audios list" }],
                $"messages: {string.Join(" | ", read.Select(t => string.Join(", ", t.Messages[0].Parts.Select(p => p.ToString()))))}");

            // Refusals name the record.
            string bad = Path.Combine(folder, "bad.jsonl");
            File.WriteAllText(bad, """{"conversations": [{"from": "human", "value": "<audio><audio>Transcribe."}, {"from": "gpt", "value": "x"}], "audios": ["wav/pcm16.wav"]}""" + "\n");
            ThrowsNaming<InvalidDataException>(() => format.Read(bad, new TuningDataOptions { Audios = AudioData("") }).ToList(), "two markers for one entry", "2 audio placeholders", "bad.jsonl");
            File.WriteAllText(bad, """{"conversations": [{"from": "human", "value": "<audio>Transcribe."}, {"from": "gpt", "value": "x"}], "audios": ["wav/none.wav"]}""" + "\n");
            ThrowsNaming<FileNotFoundException>(() => format.Read(bad, new TuningDataOptions { Audios = AudioData("") }).ToList(), "a missing sound", "none.wav", "bad.jsonl");
            File.WriteAllText(bad, """{"conversations": [{"from": "human", "value": "<audio>Transcribe."}, {"from": "gpt", "value": "x"}], "audios": ["wav/bad-no-fmt.wav"]}""" + "\n");
            Check(format.Read(bad, new TuningDataOptions { Audios = AudioData("") }).Single().Messages[0].Parts[0] is ChatAudio, "a damaged file is read as bytes (decoding is the model's)");

            // The kit's data-format suite on the sharegpt file.
            Conformance.CheckTuningDataFormat(path => format.Read(path, new TuningDataOptions { Audios = AudioData("") }).Select(t => t.Messages), [new TuningDataSample(copy)]).ThrowIfFailed();
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void ThrowsNaming<T>(Action action, string what, params string[] contains) where T : Exception
    {
        try
        {
            action();
            Check(false, $"{what}: expected {typeof(T).Name}");
        }
        catch (T e)
        {
            foreach (string text in contains)
            {
                Check(e.Message.Contains(text, StringComparison.Ordinal), $"{what}: the message names '{text}': {e.Message}");
            }
        }
    }

    private static (PretrainedModel Model, JinjaChatTemplate Template) TinyLlamaWithAudio(Device device)
    {
        var model = PretrainedModel.Load(PretrainData("tiny-llama"), new PretrainedOptions { Device = device });
        return (model, new JinjaChatTemplate(AudioTemplate, ["<|im_end|>"]));
    }

    private static void AudioChatTranscriptEncoder(Device device)
    {
        var (model, template) = TinyLlamaWithAudio(device);
        using var _ = model;
        var tokenizer = model.Tokenizer!;
        var source = new FakeAudioSource(tokens: 2, width: model.Spec.Dim, device);
        var encoder = new ChatTranscriptEncoder(template, tokenizer) { Parts = [source] };
        var audio = ChatAudio.FromFile(AudioData("wav/pcm16.wav"));
        var other = ChatAudio.FromFile(AudioData("wav/rate8k.wav"));
        var transcript = new ChatTranscript([new ChatMessage("user", [audio, new ChatText("Transcribe.")]), new ChatMessage("assistant", "hello"), new ChatMessage("user", [new ChatText("And "), other]), new ChatMessage("assistant", "again")], []);
        var sequence = encoder.Encode(transcript, 128)!;

        // The expected ids: the rendered text with each marker expanded to two soft tokens and the end token.
        string rendered = template.Render(transcript.Messages, [], null, addGenerationPrompt: false);
        string expanded = rendered.Replace("<start_of_image>", "<image><image><end_of_image>", StringComparison.Ordinal);
        var ids = tokenizer.Encode(expanded);
        Check(sequence.Tokens.SequenceEqual(ids), $"ids [{string.Join(" ", sequence.Tokens)}] against [{string.Join(" ", ids)}]");
        var positions = Enumerable.Range(0, ids.Count).Where(i => ids[i] == 6 && (i == 0 || ids[i - 1] != 6)).ToList();
        Check(sequence.Parts.Count == 2 && sequence.Parts[0].Part == audio && sequence.Parts[1].Part == other && sequence.Parts[0].Kind == "audio"
              && sequence.Parts.SelectMany(p => p.Blocks).SequenceEqual([(positions[0], 2), (positions[1], 2)]) && sequence.Parts.Sum(p => p.Tokens) == 4 && sequence.Images.Count == 0,
            $"parts' blocks [{string.Join(", ", sequence.Parts.SelectMany(p => p.Blocks))}] at the runs of soft tokens [{string.Join(", ", positions)}]");
        int trained = sequence.Trained.Count(t => t);
        var answers = tokenizer.Encode("hello<|im_end|>").Count + tokenizer.Encode("again<|im_end|>").Count;   // a trained turn ends at its end marker
        Check(trained == answers && !sequence.Trained.Take(positions[0] + 3).Any(t => t), $"trained {trained} tokens, the answers take {answers}; nothing before the first sound");
        Check(source.EncodeCalls.Count == 0, "the encoder encodes nothing");

        // Cut inside the second sound's block: the cut moves back to the block's start, keeping the first part whole.
        var cut = encoder.Encode(transcript, positions[1])!;
        Check(cut.Tokens.Length <= positions[1] + 1 && cut.Parts.Count == 1 && cut.Parts[0].Blocks.SequenceEqual([(positions[0], 2)]) && cut.TrainedTokens > 0,
            $"cut to {cut.Tokens.Length} tokens with {cut.Parts.Count} part(s)");

        // Refusals.
        var inAnswer = new ChatTranscript([new ChatMessage("user", "Say it."), new ChatMessage("assistant", [audio, new ChatText("hello")])], []);
        ThrowsNaming<InvalidDataException>(() => encoder.Encode(inAnswer, 128), "a sound in an assistant turn", "assistant turn");
        var textOnly = new ChatTranscriptEncoder(template, tokenizer);
        ThrowsNaming<InvalidOperationException>(() => textOnly.Encode(transcript, 128), "no source", "ChatTranscriptEncoder.Parts", "'audio'");
        ThrowsNaming<ArgumentException>(() => new ChatTranscriptEncoder(template, tokenizer) { Parts = [source, source] }, "a kind twice", "given twice");

        // The training step reads the parts through their source (FineTuningOptions.Parts); without one it refuses, naming it.
        string tuned = Path.Combine(Path.GetTempPath(), $"idrak-audio-chat-{Guid.NewGuid():N}");
        try
        {
            ThrowsNaming<InvalidOperationException>(() => FineTuner.Train(model, [sequence], null, new FineTuningOptions { Epochs = 1, BatchTokens = 256, MaxLength = 128 }, tuned),
                "training on parts without a source", "'audio' part", "FineTuningOptions.Parts");
            var evaluations = FineTuner.Train(model, [sequence], null, new FineTuningOptions { Epochs = 1, BatchTokens = 256, MaxLength = 128, Parts = [source] }, tuned);
            Check(source.EncodeCalls.Count > 0 && evaluations.Count == 0, "a step with parts encodes them through the source");
        }
        finally
        {
            if (Directory.Exists(tuned))
            {
                Directory.Delete(tuned, recursive: true);
            }
        }
    }

    private static void AudioChatGenerator(Device device)
    {
        var (model, template) = TinyLlamaWithAudio(device);
        using var _ = model;
        var source = new FakeAudioSource(tokens: 3, width: model.Spec.Dim, device);
        var chat = new ChatGenerator(model.CreateChat(KeyValueFormat.Float32, 64).Generator, template) { Parts = [source] };
        Check(((IChatModel)chat).PartKinds.SetEquals(["text", "audio"]), $"kinds: {string.Join(", ", chat.PartKinds)}");
        var audio = ChatAudio.FromFile(AudioData("wav/pcm16.wav"));
        var request = new ChatRequest([new ChatMessage("user", [audio, new ChatText("Transcribe.")])], Options: new GenerationOptions { Temperature = 0, NumPredict = 3, Seed = 1 });
        string prompt = chat.RenderPrompt(request);
        var ids = model.Tokenizer!.Encode(prompt);
        var runs = Enumerable.Range(0, ids.Count).Where(i => ids[i] == 6 && (i == 0 || ids[i - 1] != 6)).ToList();
        Check(prompt.Contains("<image><image><image><end_of_image>Transcribe.", StringComparison.Ordinal) && runs.Count == 1 && ids.Count(i => i == 6) == 3, prompt);

        var answer = chat.Chat(request);
        Check(answer.Done && answer.Message is not null, "answers");
        Check(source.EncodeCalls is [[ChatAudio one]] && one == audio, $"encoded once per request: {source.EncodeCalls.Count} call(s)");
        using var features = source.Encode([audio])[0];
        var located = ImagePrefill.Locate(ids, 6, [features]);
        Check(located is [{ Position: var at, Tokens: 3 }] && at == runs[0], $"the features land at {string.Join(", ", located.Select(l => l.Position))}, the soft tokens at {runs[0]}");

        // Two sounds in one request: both encoded in one call, in order; a text-only request is unchanged.
        var two = new ChatRequest([new ChatMessage("user", [audio, new ChatText(" then "), ChatAudio.FromFile(AudioData("wav/rate8k.wav")), new ChatText(" Transcribe both.")])],
            Options: request.Options);
        Check(chat.Chat(two).Done && source.EncodeCalls[^1].Count == 2 && model.Tokenizer.Encode(chat.RenderPrompt(two)).Count(i => i == 6) == 6, "two sounds");
        int calls = source.EncodeCalls.Count;                                    // the requests' calls and the test's own one above
        Check(chat.Chat(new ChatRequest([new ChatMessage("user", "hi")], Options: request.Options)).Done && source.EncodeCalls.Count == calls, $"text alone encodes nothing: {source.EncodeCalls.Count - calls} call(s)");

        // Batches: ChatBatch takes text only; StreamBatch answers requests with sounds one by one.
        ThrowsNaming<NotSupportedException>(() => chat.ChatBatch([request]), "ChatBatch", "soft parts");
        var streamed = chat.StreamBatch([request, new ChatRequest([new ChatMessage("user", "hi")], Options: request.Options)]).Where(c => c.Chunk.Done).ToList();
        Check(streamed.Count == 2 && streamed.Select(s => s.Index).Order().SequenceEqual([0, 1]), "StreamBatch answers both");
        ThrowsNaming<ArgumentException>(() => new ChatGenerator(chat.Generator, template) { Parts = [source, source] }, "a kind twice", "given twice");
    }

    private sealed class FakeAudioFamily(string name, int width) : IAudioFamily
    {
        public string Name => name;

        public IPretrainedAudio? Read(AudioCheckpoint checkpoint) => new FakePretrainedAudio(name, width, checkpoint);
    }

    private sealed class FakePretrainedAudio(string family, int width, AudioCheckpoint checkpoint) : PretrainedAudio
    {
        public AudioCheckpoint Checkpoint => checkpoint;

        public override string Family => family;

        public override int Width => width;

        public override IReadOnlySet<string> StoredTensors => new HashSet<string>();

        public override IAudioFeatureExtractor Features { get; } = AudioFeatureExtractors.Create("log-mel", new AudioFeatureOptions { PadSeconds = null });

        public override IImagePromptFormat PromptFormat { get; } = new ImageTokenFormat("fake-audio", TinyLlamaAudioMarker[0], TinyLlamaAudioMarker[1], end: TinyLlamaAudioMarker[2]);

        public override IImageAttentionRule Attention => ImageAttentionRules.Get(ImageAttentionRules.Causal);

        public override IAudioLanguageEncoder CreateEncoder(AudioEncoderOptions? options = null) => new FakeAudioEncoder(width, options?.Device ?? Device.Default);
    }

    // Two soft tokens per sound, the features its first samples' mean repeated (so a cache hit is checked value for value).
    private sealed class FakeAudioEncoder(int width, Device device) : IAudioLanguageEncoder
    {
        public int Encoded { get; private set; }

        public int Width => width;

        public Device Device => device;

        public IReadOnlyList<ImageTokenLayout> Blocks(AudioData audio) => [new ImageTokenLayout(2)];

        public IReadOnlyList<ImageFeatures> Encode(IReadOnlyList<AudioData> audios)
        {
            Encoded += audios.Count;
            return [.. audios.Select(a =>
            {
                float mean = a.Samples.Take(1000).Average();
                var values = new float[2 * width];
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = mean + i * 1e-3f;
                }

                return new ImageFeatures(Tensor.From(values, [2, width], device), new ImageTokenLayout(2));
            })];
        }

        public void Dispose()
        {
        }
    }

    private static void AudioChatAttach(Device device)
    {
        using (var text = PretrainedModel.Load(PretrainData("tiny-llama"), new PretrainedOptions { Device = device }))
        {
            Check(AudioLanguage.Attach(text) is null, "no audio_config: null");
        }

        // The same checkpoint with an audio_config: refused until its family is registered, then read through it.
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-audio-attach-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        try
        {
            foreach (string file in Directory.GetFiles(PretrainData("tiny-llama")))
            {
                File.Copy(file, Path.Combine(folder, Path.GetFileName(file)));
            }

            var config = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "config.json")))!.AsObject();
            config["audio_config"] = new JsonObject { ["mels"] = 80 };
            File.WriteAllText(Path.Combine(folder, "config.json"), config.ToJsonString());
            using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
            Check(AudioFamilies.HasAudio(model.Config) && !AudioFamilies.HasAudio(JsonNode.Parse("{}")!.AsObject()), "HasAudio");
            ThrowsNaming<NotSupportedException>(() => AudioLanguage.Attach(model), "no family", "AudioFamilies.Register", "'LlamaForCausalLM'");
            ThrowsNaming<NotSupportedException>(() => AudioFamilies.Get("LlamaForCausalLM"), "Get", "not registered");
            Check(AudioFamilies.For("LlamaForCausalLM", JsonNode.Parse("{}")!.AsObject()) is null, "For: no audio part, no family: null");

            AudioFamilies.Register(new FakeAudioFamily("LlamaForCausalLM", model.Spec.Dim));
            try
            {
                Check(AudioFamilies.Names.Contains("LlamaForCausalLM") && AudioFamilies.Origin("LlamaForCausalLM") == typeof(Tests).Assembly.GetName().Name && AudioFamilies.Default("LlamaForCausalLM") is null, "registered from the test");
                var audio = AudioLanguage.Attach(model) as FakePretrainedAudio;
                Check(audio is not null && audio.Family == "LlamaForCausalLM" && audio.Checkpoint.Folder == folder && audio.Checkpoint.Architecture == "LlamaForCausalLM"
                      && audio.Checkpoint.Config["audio_config"] is JsonObject && audio.Describe().Contains("log-mel", StringComparison.Ordinal), $"attached: {audio}");
                using (var opened = audio!.Checkpoint.Open())
                {
                    Check(opened.Names.Contains("model.embed_tokens.weight"), "Open reads the checkpoint again");
                }

                // ChatAudioParts over it: blocks and features through the encoder, the second call a cache hit with the same values.
                var encoder = new FakeAudioEncoder(model.Spec.Dim, device);
                using var parts = new ChatAudioParts(audio, encoder);
                var sound = ChatAudio.FromFile(AudioData("wav/pcm16.wav"));
                Check(parts.Kind == "audio" && parts.Width == model.Spec.Dim && parts.Blocks(sound) is [{ Tokens: 2 }] && ReferenceEquals(parts.Format, audio.PromptFormat), "blocks through the encoder");
                var first = parts.Encode([sound, ChatAudio.FromFile(AudioData("wav/rate8k.wav"))]);
                var again = parts.Encode([sound]);
                try
                {
                    Check(first.Count == 2 && again.Count == 1 && encoder.Encoded == 2 && parts.CacheHits == 1 && parts.Encoded == 2
                          && first[0].Features.ToArray().SequenceEqual(again[0].Features.ToArray()) && again[0].Layout.Tokens == 2, $"cached: encoded {encoder.Encoded}, hits {parts.CacheHits}");
                    ThrowsNaming<ArgumentException>(() => parts.Blocks(new ChatText("x")), "a text part", "ChatAudio");
                }
                finally
                {
                    foreach (var f in first.Concat(again))
                    {
                        f.Dispose();
                    }
                }
            }
            finally
            {
                AudioFamilies.Unregister("LlamaForCausalLM");
            }

            Check(!AudioFamilies.Names.Contains("LlamaForCausalLM"), "unregistered");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }
}
