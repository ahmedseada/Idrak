// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using Idrak.Maui.Abstractions;

namespace Idrak.Maui;

/// <summary>
/// The Chat lab's conversations kept as files, one JSON file a chat in <see cref="Folder"/> (the app's own storage by
/// default: <c>conversations</c> beside the cache), written whole to a temporary file and moved over the old one, so a
/// chat is never half written. Only the text of the messages is kept (user and assistant, and the reasoning).
/// </summary>
public sealed class FileConversationStore : IConversationStore
{
    private readonly Lock _gate = new();

    /// <summary>A store in <paramref name="folder"/> (made when the first chat is kept).</summary>
    public FileConversationStore(string folder)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(folder);
        Folder = Path.GetFullPath(folder);
    }

    /// <summary>The folder the chats are kept in.</summary>
    public string Folder { get; }

    /// <inheritdoc />
    public IReadOnlyList<ConversationSummary> List()
    {
        if (!Directory.Exists(Folder))
        {
            return [];
        }

        var list = new List<ConversationSummary>();
        foreach (string file in Directory.EnumerateFiles(Folder, "*.json"))
        {
            if (Read(file) is { } conversation)
            {
                list.Add(ConversationText.Summary(conversation));
            }
        }

        return [.. list.OrderByDescending(c => c.Updated)];
    }

    /// <inheritdoc />
    public ChatConversation? Load(string id) => Read(PathOf(id));

    /// <inheritdoc />
    public void Save(ChatConversation conversation)
    {
        ArgumentNullException.ThrowIfNull(conversation);
        string path = PathOf(conversation.Id);
        var stored = new StoredConversation
        {
            Id = conversation.Id,
            Title = conversation.Title,
            Updated = conversation.Updated,
            Model = conversation.Model,
            Messages = [.. conversation.Messages.Select(m => new StoredMessage { Role = m.Role, Content = m.Content, Thinking = m.Thinking })],
        };
        lock (_gate)
        {
            Directory.CreateDirectory(Folder);
            string temporary = path + ".tmp";
            using (var stream = File.Create(temporary))
            {
                JsonSerializer.Serialize(stream, stored, ConversationJson.Default.StoredConversation);
            }

            File.Move(temporary, path, overwrite: true);
        }
    }

    /// <inheritdoc />
    public bool Delete(string id)
    {
        string path = PathOf(id);
        lock (_gate)
        {
            if (!File.Exists(path))
            {
                return false;
            }

            File.Delete(path);
            return true;
        }
    }

    private string PathOf(string id) => Path.Combine(Folder, ConversationText.CheckId(id) + ".json");

    private static ChatConversation? Read(string path)
    {
        try
        {
            if (!File.Exists(path))
            {
                return null;
            }

            using var stream = File.OpenRead(path);
            var stored = JsonSerializer.Deserialize(stream, ConversationJson.Default.StoredConversation);
            return stored is null || string.IsNullOrEmpty(stored.Id) ? null
                : new ChatConversation(stored.Id, stored.Title ?? "", stored.Updated, stored.Model,
                    [.. (stored.Messages ?? []).Select(m => new ChatMessage(m.Role ?? "user", m.Content ?? "", string.IsNullOrEmpty(m.Thinking) ? null : m.Thinking))]);
        }
        catch (Exception e) when (e is JsonException or IOException or UnauthorizedAccessException)
        {
            return null;                                                         // a file that does not read is not listed
        }
    }
}

/// <summary>Conversations kept in memory only (tests, or an app that keeps none between runs).</summary>
public sealed class MemoryConversationStore : IConversationStore
{
    private readonly ConcurrentDictionary<string, ChatConversation> _chats = new(StringComparer.Ordinal);

    /// <inheritdoc />
    public IReadOnlyList<ConversationSummary> List() => [.. _chats.Values.Select(ConversationText.Summary).OrderByDescending(c => c.Updated)];

    /// <inheritdoc />
    public ChatConversation? Load(string id) => _chats.TryGetValue(ConversationText.CheckId(id), out var chat) ? chat : null;

    /// <inheritdoc />
    public void Save(ChatConversation conversation)
    {
        ArgumentNullException.ThrowIfNull(conversation);
        _chats[ConversationText.CheckId(conversation.Id)] = conversation;
    }

    /// <inheritdoc />
    public bool Delete(string id) => _chats.TryRemove(ConversationText.CheckId(id), out _);
}

/// <summary>Ids, titles and previews of kept chats.</summary>
public static class ConversationText
{
    /// <summary>A new id: the time (UTC) and a random part, letters and digits only.</summary>
    public static string NewId(DateTimeOffset now) =>
        now.UtcDateTime.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture) + "-" + Guid.NewGuid().ToString("N")[..8];

    /// <summary>The title a chat takes from its first question: its first line, at most 48 characters.</summary>
    public static string Title(string question)
    {
        ArgumentNullException.ThrowIfNull(question);
        string line = question.Trim().Split('\n')[0].Trim();
        return line.Length <= 48 ? line : line[..47].TrimEnd() + "…";
    }

    /// <summary>The summary of <paramref name="chat"/>: the start of its last message's first line.</summary>
    public static ConversationSummary Summary(ChatConversation chat)
    {
        ArgumentNullException.ThrowIfNull(chat);
        string last = chat.Messages.Count == 0 ? "" : chat.Messages[^1].Content.Trim().Split('\n')[0].Trim();
        return new ConversationSummary(chat.Id, chat.Title, chat.Updated, last.Length <= 80 ? last : last[..79] + "…", chat.Model);
    }

    /// <summary>
    /// When a chat last changed, as its list shows it: the time today, the day's name this week, else the date
    /// (<paramref name="now"/> and <paramref name="updated"/> in the phone's time).
    /// </summary>
    public static string When(DateTimeOffset updated, DateTimeOffset now)
    {
        var local = updated.ToLocalTime();
        var today = now.ToLocalTime().Date;
        return local.Date == today ? local.ToString("H:mm", CultureInfo.CurrentCulture)
            : local.Date > today.AddDays(-7) ? local.ToString("ddd", CultureInfo.CurrentCulture)
            : local.ToString("d MMM", CultureInfo.CurrentCulture);
    }

    /// <summary>The group a chat is listed under: "Today", "Yesterday", "This week", "Earlier".</summary>
    public static string Group(DateTimeOffset updated, DateTimeOffset now)
    {
        var day = updated.ToLocalTime().Date;
        var today = now.ToLocalTime().Date;
        return day == today ? "Today" : day == today.AddDays(-1) ? "Yesterday" : day > today.AddDays(-7) ? "This week" : "Earlier";
    }

    // An id is letters, digits and dashes only, so it never names a path outside the store's folder.
    internal static string CheckId(string id)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        return id.Length <= 64 && id.All(c => char.IsAsciiLetterOrDigit(c) || c == '-') ? id
            : throw new ArgumentException($"'{id}' is not a conversation id (letters, digits and dashes).", nameof(id));
    }
}

/// <summary>A kept chat as written.</summary>
internal sealed class StoredConversation
{
    public string Id { get; set; } = "";

    public string? Title { get; set; }

    public DateTimeOffset Updated { get; set; }

    public string? Model { get; set; }

    public List<StoredMessage>? Messages { get; set; }
}

/// <summary>A kept message as written.</summary>
internal sealed class StoredMessage
{
    public string? Role { get; set; }

    public string? Content { get; set; }

    public string? Thinking { get; set; }
}

[JsonSourceGenerationOptions(WriteIndented = false, DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull, PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
[JsonSerializable(typeof(StoredConversation))]
internal sealed partial class ConversationJson : JsonSerializerContext;
