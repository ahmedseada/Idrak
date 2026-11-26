// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Generation;
using Idrak.Text;
using Idrak.Text.Generation;

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

/// <summary>A part of a kind of its own carried through a conversation.</summary>
public static class ChatPartPluginTests
{
    /// <summary>A message with an <see cref="AudioClip"/>: refused by a text-only model, answered by a model that takes the kind.</summary>
    public static void AudioKind(Device device)
    {
        _ = device;
        var message = new ChatMessage("user", [new AudioClip(16_000, "3,4"), new ChatText("What is said?")]);
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

    private static void Check(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }
}
