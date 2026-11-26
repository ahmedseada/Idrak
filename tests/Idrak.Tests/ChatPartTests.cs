// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Security.Cryptography;
using System.Text.Json.Nodes;
using Idrak.Cli.Commands.Run;
using Idrak.Generation;
using Idrak.Inference;
using Idrak.Text;
using Idrak.Text.Generation;
using Idrak.Text.Inference;

// Plan 11, phase 1: message content as parts (ChatPart: text, images, kinds of one's own), and what a chat model or
// template takes (PartKinds), checked by one shared function before anything answers.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] ChatPartGroup =
    [
        ("chat parts: ChatImage from bytes, files and data URLs (media types sniffed or given, SHA-256 hash, equality by bytes), bad data URLs refused", d => { CpuOnly(d); ChatImageFactories(); }),
        ("chat parts: ChatMessage holds ordered parts; text-only messages, Content, with { Content } and equality are as before", d => { CpuOnly(d); ChatMessageParts(); }),
        ("chat parts: text-only models (ChatGenerator, the engine's chat model, FakeChatModel, WithTools) and templates refuse an image with a clear message; text is unchanged", ChatPartModelsRefuse),
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
}
