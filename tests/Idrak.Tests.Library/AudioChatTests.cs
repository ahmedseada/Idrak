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
        var model = PretrainedModel.Load(PretrainData("tiny-llama"), new PretrainedOptions { Device = device, Families = Idrak.Text.Plugins.BuiltInFamilies.All });
        return (model, new JinjaChatTemplate(AudioTemplate, ["<|im_end|>"]));
    }
}
