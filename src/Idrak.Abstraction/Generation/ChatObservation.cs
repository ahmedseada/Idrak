// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Generation;

/// <summary>
/// What a chat agent sees at a turn of an episode (the observation of a conversation environment,
/// <c>Idrak.Abstraction.Reinforcement.Space.Conversation</c>): the conversation so far, the tools it may call, and the
/// reasoning mode. Its action is the assistant's next <see cref="ChatMessage"/>. Environments that run several turns
/// give the next observation as this one's messages, the reply, and what followed (tool results, a user's turn).
/// </summary>
/// <param name="Messages">The conversation so far (system, user, assistant and tool messages).</param>
/// <param name="Tools">The tools offered, or null for none.</param>
/// <param name="Think">The reasoning mode the template renders with (null: the model's default).</param>
public sealed record ChatObservation(IReadOnlyList<ChatMessage> Messages, IReadOnlyList<ToolDefinition>? Tools = null, bool? Think = null)
{
    /// <summary>Equal when both hold equal messages and tools, in order, and the same reasoning mode (an episode replayed from its seed observes equal conversations).</summary>
    public bool Equals(ChatObservation? other) =>
        other is not null && Think == other.Think && Messages.SequenceEqual(other.Messages)
        && (Tools is null || Tools.Count == 0 ? other.Tools is null || other.Tools.Count == 0 : other.Tools is not null && Tools.SequenceEqual(other.Tools));

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add(Think);
        foreach (var message in Messages)
        {
            hash.Add(message);
        }

        return hash.ToHashCode();
    }
}
