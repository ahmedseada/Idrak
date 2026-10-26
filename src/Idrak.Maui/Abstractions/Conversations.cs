// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Maui.Abstractions;

/// <summary>A chat kept between runs: its messages (user and assistant, their text and reasoning) and when it last changed.</summary>
/// <param name="Id">Its id (a file name's worth of letters and digits).</param>
/// <param name="Title">Its title (its first question, shortened).</param>
/// <param name="Updated">When it last changed.</param>
/// <param name="Model">The model it was last answered by ("Qwen/Qwen3-0.6B"), or null.</param>
/// <param name="Messages">Its messages, oldest first (the system prompt is not kept: the settings give it).</param>
public sealed record ChatConversation(string Id, string Title, DateTimeOffset Updated, string? Model, IReadOnlyList<ChatMessage> Messages);

/// <summary>A kept chat as a list shows it.</summary>
/// <param name="Id">Its id.</param>
/// <param name="Title">Its title.</param>
/// <param name="Updated">When it last changed.</param>
/// <param name="Preview">The start of its last answer (or question).</param>
/// <param name="Model">The model it was last answered by, or null.</param>
public sealed record ConversationSummary(string Id, string Title, DateTimeOffset Updated, string Preview, string? Model);

/// <summary>
/// Where the Chat lab keeps its conversations between runs: files in the app's own storage by default
/// (<c>Idrak.Maui.FileConversationStore</c>), in memory in tests; an app registers its own in its services to keep them
/// elsewhere (or nowhere).
/// </summary>
public interface IConversationStore
{
    /// <summary>The kept chats, the latest first.</summary>
    IReadOnlyList<ConversationSummary> List();

    /// <summary>The chat <paramref name="id"/>, or null when none is kept under it.</summary>
    ChatConversation? Load(string id);

    /// <summary>Keeps <paramref name="conversation"/> (over the one of its id).</summary>
    void Save(ChatConversation conversation);

    /// <summary>Forgets the chat <paramref name="id"/>; false when none was kept.</summary>
    bool Delete(string id);
}
