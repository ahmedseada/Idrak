// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Text;
using Idrak.Text.Abstractions;
using Idrak.Text.Generation;

// Plan 21, phase S3: the "audio" chat part and the soft-part contract in generation and tuning. The files come from
// tools/audio/make_tiny_audio.py (wav/).
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] AudioChatGroup =
    [
        ("audio chat: ChatAudio from bytes, files, streams and data URLs (media types sniffed for WAV, FLAC, MP3, Ogg and MP4 or given; SHA-256 identity; equality by bytes; bad data URLs refused); the audio kind writes a part to its JSON and reads it back, refusing one without data or with bad base64", d => { CpuOnly(d); AudioChatPart(); }),
        ("audio chat: text-only models and templates refuse an audio part naming 'audio'; a Jinja template that writes a marker for audio parts takes them", AudioChatJsonAndRefusal),
        ("audio chat: the transcript encoder with a fake soft-part source (two tokens per sound) expands each marker to its soft tokens, records the parts' blocks, trains the answer alone, keeps whole blocks when cut, and refuses a sound in an assistant turn or without a source; the training step reads the parts through FineTuningOptions.Parts and refuses them without it", AudioChatTranscriptEncoder),
        ("audio chat: ChatGenerator with the same source on the tiny Llama renders the soft tokens in place, encodes each sound once per request, answers, and the features land at the soft tokens' positions; ChatBatch refuses soft parts and StreamBatch answers them one by one", AudioChatGenerator),
    ];

    internal static readonly int[] TinyLlamaAudioMarker = [5, 6, 7];   // the fixture's <start_of_image>, <image>, <end_of_image>: reused as the fake family's audio marker, soft token and end

    // The tiny Llama's ChatML template with an audio branch (its own writes <image> for image parts only): an audio part
    // is written as the marker the fake family expands.
    internal const string AudioTemplate = """
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
    internal sealed class FakeAudioSource(int tokens, int width, Device device) : ISoftPartSource
    {
        public List<IReadOnlyList<ChatPart>> EncodeCalls { get; } = [];

        public string Kind => ChatParts.Audio;

        public int Width => width;

        public IImagePromptFormat Format { get; } = new ImageTokenFormat("fake-audio", TinyLlamaAudioMarker[0], TinyLlamaAudioMarker[1], end: TinyLlamaAudioMarker[2]);

        public IImageAttentionRule Attention => new CausalRule();

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

    internal static void AudioChatPart()
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

        // The audio kind: a part to its JSON (the "type" given first, as a writer finds it) and back.
        var kind = new AudioPartKind();
        var json = new JsonObject { ["type"] = "audio" };
        kind.Write(fromFile, json);
        var expectedJson = new JsonObject { ["type"] = "audio", ["media_type"] = "audio/wav", ["data"] = Convert.ToBase64String(wav) };   // the serializer's own escaping of base64's '+'
        Check(json.ToJsonString() == expectedJson.ToJsonString() && kind.Read(json) == fromFile, $"the part's JSON: {json.ToJsonString()[..60]}…");
        Throws<FormatException>(() => kind.Read(new JsonObject { ["type"] = "audio" }), "audio with no data");
        Throws<FormatException>(() => kind.Read(new JsonObject { ["type"] = "audio", ["data"] = "***" }), "bad base64");
    }

    internal static void AudioChatJsonAndRefusal(Device device)
    {
        var audio = ChatAudio.FromFile(AudioData("wav/pcm16.wav"));
        var message = new ChatMessage("user", [audio, new ChatText("Transcribe.")]);

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

        // A Jinja template that writes something for an audio part takes it (the probe finds every kind).
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

    internal static void ThrowsNaming<T>(Action action, string what, params string[] contains) where T : Exception
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

    internal static (PretrainedModel Model, JinjaChatTemplate Template) TinyLlamaWithAudio(Device device)
    {
        var model = PretrainedModel.Load(PretrainData("tiny-llama"), new PretrainedOptions { Device = device });
        return (model, new JinjaChatTemplate(AudioTemplate, ["<|im_end|>"]));
    }

    internal static void AudioChatTranscriptEncoder(Device device)
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

    internal static void AudioChatGenerator(Device device)
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
}
