// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Security.Cryptography;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Cli.Commands.Run;
using Idrak.Generation;
using Idrak.Inference;
using Idrak.Nlp;

// Plan 11, phase 1: message content as parts (ChatPart: text, images, kinds of one's own) registered in ChatParts, and
// what a chat model or template takes (PartKinds), checked by one shared function before anything answers.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] ChatPartGroup =
    [
        ("chat parts: ChatImage from bytes, files and data URLs (media types sniffed or given, SHA-256 hash, equality by bytes), bad data URLs refused", d => { if (d == Device.Cpu) ChatImageFactories(); }),
        ("chat parts: ChatMessage holds ordered parts; text-only messages, Content, with { Content } and equality are as before", d => { if (d == Device.Cpu) ChatMessageParts(); }),
        ("chat parts: the registry (text and image as library defaults, Register/Unregister, Origin, versions, policies) and the JSON of parts and content", d => { if (d == Device.Cpu) ChatPartRegistry(); }),
        ("chat parts: the testing kit's ChatPartKindSuite passes the built-in kinds and catches a kind that loses data", d => { if (d == Device.Cpu) ChatPartKit(); }),
        ("chat parts: text-only models (ChatGenerator, the engine's chat model, FakeChatModel, WithTools) and templates refuse an image with a clear message; text is unchanged", ChatPartModelsRefuse),
        ("chat parts: chat JSON (ChatJson, fine-tuning transcripts, the CLI's --history) round-trips images and keeps text-only content a string; /v1 refuses image_url parts", ChatPartJson),
    ];

    private static readonly byte[] PngBytes = [0x89, (byte)'P', (byte)'N', (byte)'G', 13, 10, 26, 10, 0, 0, 0, 13, 1, 2, 3];

    private static void ChatImageFactories()
    {
        var png = ChatImage.FromBytes(PngBytes);
        Check(png.MediaType == "image/png" && png.Kind == "image" && png.Data.Span.SequenceEqual(PngBytes), "PNG sniffed");
        Check(png.Hash == Convert.ToHexStringLower(SHA256.HashData(PngBytes)) && png.Hash.Length == 64, "SHA-256 hash");
        Check(ChatImage.FromBytes([0xFF, 0xD8, 0xFF, 0xE0]).MediaType == "image/jpeg" && ChatImage.FromBytes([1, 2]).MediaType is null
              && ChatImage.FromBytes([1, 2], "Image/X-Own").MediaType == "image/x-own", "JPEG sniffed, unknown null, given media type kept (lowercase)");

        var copy = ChatImage.FromBytes(PngBytes, "image/whatever");
        Check(copy == png && copy.GetHashCode() == png.GetHashCode() && !ChatImage.FromBytes([.. PngBytes, 0]).Equals(png), "equal by bytes, whatever the media type");
        byte[] source = [.. PngBytes];
        var fromArray = ChatImage.FromBytes(source);
        source[^1] = 99;
        Check(fromArray == png, "the bytes are copied");

        string url = png.ToDataUrl();
        Check(url.StartsWith("data:image/png;base64,", StringComparison.Ordinal) && ChatImage.FromDataUrl(url) == png
              && ChatImage.FromDataUrl(" data:;base64," + Convert.ToBase64String(PngBytes) + " ").MediaType == "image/png"
              && ChatImage.FromDataUrl("DATA:image/jpeg;charset=x;base64," + Convert.ToBase64String(PngBytes)).MediaType == "image/jpeg", "data URLs");
        foreach (var (bad, why) in new[]
        {
            ("http://example.com/a.png", "Not a data URL"), ("data:image/png;base64", "Not a data URL"), ("data:image/png,abc", "must be base64"),
            ("data:text/plain;base64,aGk=", "not an image"), ("data:image/png;base64,***", "does not decode"), ("data:image/png;base64,", "no bytes"),
        })
        {
            try
            {
                ChatImage.FromDataUrl(bad);
                Check(false, $"{bad} refused");
            }
            catch (FormatException e)
            {
                Check(e.Message.Contains(why, StringComparison.Ordinal), $"{bad}: {e.Message}");
            }
        }

        Throws<ArgumentException>(() => ChatImage.FromBytes([]), "empty bytes");
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-chat-image-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        try
        {
            string file = Path.Combine(folder, "scan.png");
            File.WriteAllBytes(file, PngBytes);
            Check(ChatImage.FromFile(file) == png && ChatImage.FromFile(file).MediaType == "image/png", "from a file");
            Throws<FileNotFoundException>(() => ChatImage.FromFile(Path.Combine(folder, "missing.png")), "a missing file");
            File.WriteAllBytes(file, []);
            Throws<ArgumentException>(() => ChatImage.FromFile(file), "an empty file");
        }
        finally
        {
            Directory.Delete(folder, true);
        }

        Check(png.ToString().StartsWith("image/png, 15 bytes, sha256 " + png.Hash[..8], StringComparison.Ordinal), png.ToString());
    }

    private static void ChatMessageParts()
    {
        var text = new ChatMessage("user", "hi");
        Check(text.Parts is [ChatText { Text: "hi" }] && text.Content == "hi" && text == new ChatMessage("user", "hi")
              && text.GetHashCode() == new ChatMessage("user", "hi").GetHashCode() && text != new ChatMessage("user", "ho"), "a text message, as before");
        Check(new ChatMessage("assistant").Parts.Count == 0 && new ChatMessage("assistant", "").Content == ""
              && new ChatMessage("assistant", (IReadOnlyList<ChatPart>)null!).Parts.Count == 0, "no content: no parts");
        var calls = new ChatMessage("assistant", ToolCalls: [new ToolCall("f", [])]);
        Check(calls.Content == "" && calls.ToolCalls!.Count == 1 && calls == calls with { }, "named arguments as before");

        var image = ChatImage.FromBytes(PngBytes);
        var mixed = new ChatMessage("user", [image, new ChatText("Read "), new ChatText("this.")]);
        Check(mixed.Content == "Read this." && mixed.Parts.Count == 3 && mixed == new ChatMessage("user", [ChatImage.FromBytes(PngBytes), new ChatText("Read "), new ChatText("this.")])
              && mixed != new ChatMessage("user", [new ChatText("Read "), new ChatText("this."), image]), "ordered parts; Content joins the text; equality by parts in order");
        var edited = mixed with { Content = "Shorter." };
        Check(edited.Parts is [ChatImage, ChatText { Text: "Shorter." }] && (mixed with { Content = "" }).Parts is [ChatImage], "with { Content } replaces the text, keeps the image");
        Check((text with { Parts = null! }).Parts.Count == 0, "null parts are none");
    }

    private static void ChatPartRegistry()
    {
        Check(ChatParts.Names.SequenceEqual(["text", "image"]) && ChatParts.Origin("text") == Overrides.Library && ChatParts.DefaultVersion("image") == 1
              && ChatParts.Default("image", 1) == ChatParts.Default("image") && ChatParts.Default("image", 2) is null && ChatParts.Default("audio") is null
              && ChatParts.TextOnly.SetEquals(["text"]), "text and image are library defaults, version 1");
        Throws<NotSupportedException>(() => ChatParts.Get("audio"), "an unknown kind");
        Check(ChatParts.Find("audio") is null && ChatParts.Find("text") is not null, "Find");

        var image = ChatImage.FromBytes(PngBytes);
        var json = ChatParts.ToJson(image);
        Check(json.ToJsonString() == $"{{\"type\":\"image\",\"media_type\":\"image/png\",\"data\":\"{Convert.ToBase64String(PngBytes)}\"}}", json.ToJsonString());
        Check(ChatParts.FromJson(json) == image && ChatParts.FromJson(ChatParts.ToJson(new ChatText("x"))) == new ChatText("x")
              && ChatParts.FromJson(JsonValue.Create("bare")) == new ChatText("bare"), "parts round-trip; a bare string is text");
        Throws<FormatException>(() => ChatParts.FromJson(new JsonObject { ["text"] = "no type" }), "a part with no type");
        Throws<FormatException>(() => ChatParts.FromJson(new JsonObject { ["type"] = "image", ["data"] = "***" }), "bad base64");
        Throws<FormatException>(() => ChatParts.FromJson(new JsonObject { ["type"] = "text" }), "text with no text");
        try
        {
            ChatParts.FromJson(new JsonObject { ["type"] = "image_url", ["image_url"] = new JsonObject() });
            Check(false, "an unregistered type is refused");
        }
        catch (NotSupportedException e)
        {
            Check(e.Message.Contains("'image_url'", StringComparison.Ordinal) && e.Message.Contains("ChatParts.Register", StringComparison.Ordinal), e.Message);
        }

        Check(ChatParts.ContentToJson([]).ToJsonString() == "\"\"" && ChatParts.ContentToJson([new ChatText("hi")]).ToJsonString() == "\"hi\""
              && ChatParts.ContentToJson([image, new ChatText("hi")]) is JsonArray { Count: 2 } && ChatParts.ContentFromJson(null).Count == 0
              && ChatParts.ContentFromJson(JsonValue.Create("")).Count == 0 && ChatParts.ContentFromJson(ChatParts.ContentToJson([image, new ChatText("hi")])) is [ChatImage, ChatText],
            "content: a string when text alone, else parts");

        // An app's image kind over the library's: it shadows the default, falls back under FallBack, and unregisters.
        var library = ChatParts.Default("image")!;
        int calls = 0;
        ChatParts.Register(new BrokenImageKind(() => calls++));
        try
        {
            Check(ChatParts.Origin("image") == typeof(Tests).Assembly.GetName().Name && ReferenceEquals(ChatParts.Default("image"), library)
                  && Overrides.Report().Any(o => o.Registry == "ChatParts" && o.Name == "image"), "the app's kind shadows the default and shows in the report");
            Throws<InvalidDataException>(() => ChatParts.FromJson(json), "Throw: the app's failure reaches the caller");
            ChatParts.SetPolicy("image", SlotPolicy.FallBack);
            Check(ChatParts.FromJson(json) == image && ChatParts.ToJson(image).ToJsonString() == json.ToJsonString() && calls == 3, "FallBack: the library's kind answers");
        }
        finally
        {
            ChatParts.SetPolicy("image", SlotPolicy.Throw);
            ChatParts.Unregister("image");
        }

        Check(ReferenceEquals(ChatParts.Get("image"), library) && ChatParts.Origin("image") == Overrides.Library && !ChatParts.Unregister("image"), "Unregister brings the default back");
    }

    private sealed class BrokenImageKind(Action called) : IChatPartKind
    {
        public string Name => "image";

        public void Write(ChatPart part, JsonObject json)
        {
            called();
            throw new InvalidDataException("the app's image kind broke");
        }

        public ChatPart Read(JsonObject json)
        {
            called();
            throw new InvalidDataException("the app's image kind broke");
        }
    }

    private static void ChatPartKit()
    {
        Conformance.CheckChatPartKind("text").ThrowIfFailed();
        Conformance.CheckChatPartKind("image", options: new ContractCheckOptions { RandomCases = 10 }).ThrowIfFailed();
        var report = Conformance.CheckChatPartKind("image", new ForgetfulImageKind());
        Check(!report.Passed && report.Failures.Any(f => f.Check == "writes the JSON the library's kind writes"), report.ToString());
        Throws<ArgumentException>(() => new ChatPartKindSuite("audio"), "a kind of one's own needs samples");
    }

    // Drops the media type: still reads back equal (images compare by bytes), but not the library's JSON.
    private sealed class ForgetfulImageKind : IChatPartKind
    {
        public string Name => "image";

        public void Write(ChatPart part, JsonObject json) => json["data"] = Convert.ToBase64String(((ChatImage)part).Data.Span);

        public ChatPart Read(JsonObject json) => ChatParts.Default("image")!.Read(json);
    }

    private static void ChatPartModelsRefuse(Device device)
    {
        var image = ChatImage.FromBytes(PngBytes);
        ChatMessage[] withImage = [new("system", "Be brief."), new("user", [image, new ChatText("Read this.")])];

        void Refused(Action action, string who)
        {
            try
            {
                action();
                Check(false, $"{who} refuses an image");
            }
            catch (Exception e) when (e is NotSupportedException || e.InnerException is NotSupportedException)
            {
                string message = (e as NotSupportedException ?? e.InnerException!).Message;
                Check(message.Contains(who, StringComparison.Ordinal) && message.Contains("message 2 (user)", StringComparison.Ordinal)
                      && message.Contains("kind 'image'", StringComparison.Ordinal) && message.Contains("only 'text'", StringComparison.Ordinal), message);
            }
        }

        // Templates.
        Refused(() => new ChatMLTemplate().Render(withImage, [], null), "ChatMLTemplate");
        var jinja = new JinjaChatTemplate("{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}", ["</s>"]);
        Refused(() => jinja.Render(withImage, [], null), "JinjaChatTemplate");
        Check(jinja.Render([new ChatMessage("user", "hi")], [], null) == "user: hi\n" && new ChatMLTemplate().PartKinds.SetEquals(["text"]), "text renders as before");

        // Models: ChatGenerator, a fake, a tool wrapper, the engine's chat model (named in the message).
        var (model, tokenizer) = TinyLanguageModel(device);
        var chat = new ChatGenerator(new TextGenerator(model, tokenizer, 32));
        var options = new GenerationOptions { Seed = 3, NumPredict = 4 };
        Check(((IChatModel)chat).PartKinds.SetEquals(["text"]) && chat.Chat(new ChatRequest([new ChatMessage("user", "hi")], Options: options)).Done, "text answers");
        Refused(() => chat.Chat(new ChatRequest(withImage, Options: options)), "ChatGenerator");
        Refused(() => chat.RenderPrompt(new ChatRequest(withImage)), "ChatGenerator");
        Refused(() => chat.ChatBatch([new ChatRequest([new ChatMessage("user", "hi")]), new ChatRequest(withImage)]), "ChatGenerator");
        Refused(() => chat.StreamBatch([new ChatRequest(withImage)]).ToList(), "ChatGenerator");

        var fake = FakeChatModel.Script(FakeChatModel.Answer("ok"));
        Refused(() => fake.ChatAsync(new ChatRequest(withImage)).GetAwaiter().GetResult(), "FakeChatModel");
        Check(fake.Requests.Count == 0, "nothing reached the fake");
        var tools = ToolRegistry.Create().Add("noop", "Does nothing.", () => "done").Build();
        Refused(() => fake.WithTools(tools, 1).ChatAsync(new ChatRequest(withImage)).GetAwaiter().GetResult(), "FakeChatModel");
        fake.PartKinds = new HashSet<string> { "text", "image" };
        Check(fake.WithTools(tools, 1).PartKinds.Contains("image") && fake.ChatAsync(new ChatRequest(withImage)).GetAwaiter().GetResult().Message!.Content == "ok"
              && fake.Requests[^1].Messages[1].Parts[0] == image, "a model that takes images gets them, through a wrapper too");
        var conversation = Conversation.For(FakeChatModel.Script(FakeChatModel.Answer("no"))).Build();
        conversation.Messages.AddRange(withImage);
        Refused(() => conversation.ContinueAsync().GetAwaiter().GetResult(), "FakeChatModel");

        if (device != Device.Cpu)
        {
            return;
        }

        var engine = InferenceEngine.Create()
            .ChatModel("tiny-chat", () => { var (m, t) = TinyLanguageModel(device); return new TextGenerator(m, t, 32); })
            .BuildAsync().GetAwaiter().GetResult();
        try
        {
            var hosted = engine.Model<IChatModel>("tiny-chat");
            Check(hosted.PartKinds.SetEquals(["text"]) && hosted.ChatAsync(new ChatRequest([new ChatMessage("user", "hi")], Options: options)).Result.Done, "the engine answers text");
            Refused(() => hosted.ChatAsync(new ChatRequest(withImage, Options: options)).GetAwaiter().GetResult(), "'tiny-chat'");
        }
        finally
        {
            engine.DisposeAsync().AsTask().GetAwaiter().GetResult();
        }
    }

    private static void ChatPartJson(Device device)
    {
        var image = ChatImage.FromBytes(PngBytes);
        List<ChatMessage> messages =
        [
            new("system", "Be brief."),
            new("user", [image, new ChatText("Read this.")]),
            new("assistant", "It says hi.", "Looked at it.", [new ToolCall("noop", [])]),
            new("tool", "done", ToolName: "noop"),
            new("assistant", ""),
        ];
        var transcript = ChatJson.Transcript(messages);
        var lines = transcript["messages"]!.AsArray();
        Check(lines[0]!["content"]!.GetValueKind() == System.Text.Json.JsonValueKind.String && lines[4]!["content"]!.ToJsonString() == "\"\""
              && lines[1]!["content"] is JsonArray { Count: 2 } parts && (string?)parts[0]!["type"] == "image" && (string?)parts[1]!["text"] == "Read this.",
            $"text stays a string, images make a list: {transcript.ToJsonString()}");
        var back = ChatTranscript.FromJson(JsonNode.Parse(transcript.ToJsonString())!.AsObject()).Messages;
        Check(back.Count == 5 && back[1] == messages[1] && back[0] == messages[0] && back[3] == messages[3] && back[2].Content == "It says hi." && back[4].Parts.Count == 0,
            "fine-tuning transcripts read the images back");
        var openAi = ChatTranscript.FromJson(JsonNode.Parse("""{"messages": [{"role": "user", "content": [{"type": "text", "text": "a"}, "b", {"type": "text", "text": "c"}]}]}""")!.AsObject());
        Check(openAi.Messages[0].Content == "abc", "text parts join as before");
        Throws<InvalidDataException>(() => ChatTranscript.FromJson(JsonNode.Parse("""{"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}""")!.AsObject()),
            "an unknown part type is refused, not dropped");

        if (device != Device.Cpu)
        {
            return;
        }

        string folder = Path.Combine(Path.GetTempPath(), $"idrak-chat-parts-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        try
        {
            string history = Path.Combine(folder, "history.json");
            ChatHistory.Save(history, messages, null, null);
            var loaded = ChatHistory.Load(history);
            Check(loaded.Count == 5 && loaded[1] == messages[1] && loaded[0] == messages[0] && loaded[3] == messages[3], "--history round-trips the image");
            File.WriteAllText(history, """{"messages": [{"role": "user", "content": [{"type": "video", "data": "x"}]}]}""");
            try
            {
                ChatHistory.Load(history);
                Check(false, "an unknown part in a history file is refused");
            }
            catch (InvalidOperationException e)
            {
                Check(e.Message.Contains("'video'", StringComparison.Ordinal), e.Message);
            }

            // A text model resumed from a history with an image says why it cannot answer.
            ChatHistory.Save(history, messages[..2], null, null);
            var (code, text, error) = RunIdrakOn(device, "go on\n", "chat", CliModel, "--max-tokens", "2", "--history", history);
            Check(code != 0 && (text + error).Contains("kind 'image'", StringComparison.Ordinal), $"chat with an image in its history: exit {code}, {text}{error}");
        }
        finally
        {
            Directory.Delete(folder, true);
        }

        try
        {
            Idrak.AspNetCore.CompletionsTranslation.Chat(JsonNode.Parse("""{"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}]}""")!.AsObject());
            Check(false, "/v1 refuses image_url parts until it takes them");
        }
        catch (ArgumentException e)
        {
            Check(e.Message.Contains("'image_url'", StringComparison.Ordinal), e.Message);
        }

        var request = Idrak.AspNetCore.CompletionsTranslation.Chat(JsonNode.Parse("""{"messages": [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}, {"role": "assistant", "content": null}]}""")!.AsObject());
        Check(request.Messages[0].Content == "ab" && request.Messages[1].Content == "", "/v1 text parts as before");
    }
}
