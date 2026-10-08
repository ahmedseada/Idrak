// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Generation;
using Idrak.Nlp;

namespace Idrak.PluginTests;

/// <summary>A message part of a kind the library does not have: a short sound, as its samples and their rate.</summary>
/// <param name="Rate">Samples per second.</param>
/// <param name="Samples">The samples, comma-separated (kept as text so the record compares by value).</param>
public sealed record AudioClip(int Rate, string Samples) : ChatPart
{
    /// <summary>The kind's name.</summary>
    public const string KindName = "outside-audio";

    /// <inheritdoc />
    public override string Kind => KindName;
}

/// <summary>How <see cref="AudioClip"/> parts are written to and read from the chat JSON: <c>{"type": "outside-audio", "rate": 16000, "samples": "0,1,2"}</c>.</summary>
public sealed class AudioClipKind : IChatPartKind
{
    /// <inheritdoc />
    public string Name => AudioClip.KindName;

    /// <inheritdoc />
    public void Write(ChatPart part, JsonObject json)
    {
        var clip = (AudioClip)part;
        json["rate"] = clip.Rate;
        json["samples"] = clip.Samples;
    }

    /// <inheritdoc />
    public ChatPart Read(JsonObject json) =>
        new AudioClip((int?)json["rate"] ?? throw new FormatException("An audio clip needs its rate."), (string?)json["samples"] ?? "");
}

/// <summary>The part kind registered from outside the library, checked by the testing kit and carried through a conversation.</summary>
public static class ChatPartPluginTests
{
    /// <summary>Registers <see cref="AudioClipKind"/>, checks it, round-trips a message through the chat JSON and a model that takes it, then unregisters it.</summary>
    public static void AudioKind(Device device)
    {
        _ = device;
        ChatParts.Register(new AudioClipKind());
        try
        {
            Check(ChatParts.Names.Contains(AudioClip.KindName) && ChatParts.Origin(AudioClip.KindName) == typeof(AudioClipKind).Assembly.GetName().Name
                  && ChatParts.Default(AudioClip.KindName) is null, "registered, from this assembly, with no library default");
            Conformance.CheckChatPartKind(AudioClip.KindName, samples:
            [
                new JsonObject { ["type"] = AudioClip.KindName, ["rate"] = 16_000, ["samples"] = "0,1,-1" },
                new JsonObject { ["type"] = AudioClip.KindName, ["rate"] = 8_000, ["samples"] = "" },
            ]).ThrowIfFailed();

            var message = new ChatMessage("user", [new AudioClip(16_000, "3,4"), new ChatText("What is said?")]);
            var json = ChatJson.Messages([message]);
            var back = ChatTranscript.FromJson(new JsonObject { ["messages"] = json }).Messages.Single();
            Check(back == message && back.Content == "What is said?", $"through the chat JSON: {json.ToJsonString()}");

            var textOnly = FakeChatModel.Script(FakeChatModel.Answer("no"));
            try
            {
                textOnly.ChatAsync(new ChatRequest([message])).GetAwaiter().GetResult();
                Check(false, "a text-only model refuses the clip");
            }
            catch (NotSupportedException e)
            {
                Check(e.Message.Contains($"'{AudioClip.KindName}'", StringComparison.Ordinal), e.Message);
            }

            var listening = FakeChatModel.Script(FakeChatModel.Answer("hello"));
            listening.PartKinds = new HashSet<string> { ChatParts.Text, AudioClip.KindName };
            Check(listening.ChatAsync(new ChatRequest([message])).GetAwaiter().GetResult().Message!.Content == "hello", "a model that takes clips answers");
        }
        finally
        {
            ChatParts.Unregister(AudioClip.KindName);
        }

        Check(!ChatParts.Names.Contains(AudioClip.KindName), "unregistered");
        try
        {
            ChatParts.FromJson(new JsonObject { ["type"] = AudioClip.KindName, ["rate"] = 1 });
            Check(false, "an unregistered kind is not read");
        }
        catch (NotSupportedException e)
        {
            Check(e.Message.Contains("ChatParts.Register", StringComparison.Ordinal), e.Message);
        }
    }

    private static void Check(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }
}
