// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Frozen;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Generation;

/// <summary>
/// One part of a message's content (<see cref="ChatMessage.Parts"/>): text (<see cref="ChatText"/>), an image
/// (<see cref="ChatImage"/>), or a kind of one's own (audio, a video, a document). Every part has a
/// <see cref="Kind"/>, the name its <see cref="IChatPartKind"/> is registered under in <see cref="ChatParts"/>, which
/// writes and reads it as JSON. A chat model says which kinds it takes (<see cref="IChatModel.PartKinds"/>) and a chat
/// template which it renders (<see cref="ChatTemplate.PartKinds"/>); either fails on any other kind, never drops it.
/// Parts are immutable and compare by value.
/// </summary>
public abstract record ChatPart
{
    /// <summary>The part's kind: "text", "image", or the name of a kind registered with <see cref="ChatParts.Register"/>.</summary>
    public abstract string Kind { get; }
}

/// <summary>Text in a message.</summary>
/// <param name="Text">The text.</param>
public sealed record ChatText(string Text) : ChatPart
{
    /// <summary>The text (never null).</summary>
    public string Text { get; init; } = Text ?? throw new ArgumentNullException(nameof(Text));

    /// <inheritdoc />
    public override string Kind => ChatParts.Text;
}

/// <summary>
/// An image in a message, as it was received: its encoded bytes (PNG, JPEG, ...; never decoded here, decoding is core's
/// <c>ImageCodecs</c>), its media type when known, and the SHA-256 of the bytes (<see cref="Hash"/>), which identifies it:
/// two images are equal when their bytes are, so a model can encode an image once per conversation. Make one with
/// <see cref="FromFile"/>, <see cref="FromBytes"/> or <see cref="FromDataUrl"/>.
/// </summary>
public sealed record ChatImage : ChatPart
{
    private readonly byte[] _data;

    private ChatImage(byte[] data, string? mediaType)
    {
        if (data.Length == 0)
        {
            throw new ArgumentException("An image needs its bytes; these are empty.", nameof(data));
        }

        _data = data;
        MediaType = string.IsNullOrWhiteSpace(mediaType) ? Sniff(data) : mediaType.Trim().ToLowerInvariant();
        Hash = Convert.ToHexStringLower(SHA256.HashData(data));
    }

    /// <inheritdoc />
    public override string Kind => ChatParts.Image;

    /// <summary>The encoded bytes, as received.</summary>
    public ReadOnlyMemory<byte> Data => _data;

    /// <summary>
    /// The media type ("image/png"): as given, else read from the bytes' signature (PNG, JPEG, GIF, BMP, WebP, TIFF); null
    /// when neither says.
    /// </summary>
    public string? MediaType { get; }

    /// <summary>The SHA-256 of <see cref="Data"/>, 64 lowercase hex digits: the image's identity.</summary>
    public string Hash { get; }

    /// <summary>The image in the file at <paramref name="path"/> (its bytes as they are; the media type from their signature).</summary>
    public static ChatImage FromFile(string path, string? mediaType = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"Image file not found: {path}", path);
        }

        byte[] data = File.ReadAllBytes(path);
        return data.Length > 0 ? new ChatImage(data, mediaType) : throw new ArgumentException($"The image file {path} is empty.", nameof(path));
    }

    /// <summary>The image encoded in <paramref name="data"/> (copied), of <paramref name="mediaType"/> when given.</summary>
    public static ChatImage FromBytes(ReadOnlySpan<byte> data, string? mediaType = null) => new(data.ToArray(), mediaType);

    /// <summary>
    /// The image in a base64 data URL, <c>data:image/png;base64,iVBOR…</c> (as OpenAI's <c>image_url</c> parts carry it).
    /// The media type must be an image type (or left out); the data must be base64.
    /// </summary>
    /// <exception cref="FormatException">The text is not a base64 data URL of an image.</exception>
    public static ChatImage FromDataUrl(string url)
    {
        ArgumentNullException.ThrowIfNull(url);
        string text = url.Trim();
        int comma = text.IndexOf(',', StringComparison.Ordinal);
        if (!text.StartsWith("data:", StringComparison.OrdinalIgnoreCase) || comma < 0)
        {
            throw new FormatException($"Not a data URL (data:image/png;base64,...): {Shorten(text)}");
        }

        string[] header = text[5..comma].Split(';', StringSplitOptions.TrimEntries);
        if (!header.Skip(1).Contains("base64", StringComparer.OrdinalIgnoreCase))
        {
            throw new FormatException($"An image data URL must be base64 (data:image/png;base64,...): {Shorten(text)}");
        }

        string? mediaType = header[0].Length == 0 ? null : header[0];
        if (mediaType is not null && !mediaType.StartsWith("image/", StringComparison.OrdinalIgnoreCase))
        {
            throw new FormatException($"The data URL holds '{mediaType}', not an image: {Shorten(text)}");
        }

        byte[] data;
        try
        {
            data = Convert.FromBase64String(text[(comma + 1)..]);
        }
        catch (FormatException)
        {
            throw new FormatException($"The data URL's base64 does not decode: {Shorten(text)}");
        }

        return data.Length > 0 ? new ChatImage(data, mediaType) : throw new FormatException("The data URL holds no bytes.");
    }

    /// <summary>The image as a base64 data URL (<c>data:image/png;base64,…</c>; application/octet-stream when the media type is unknown).</summary>
    public string ToDataUrl() => $"data:{MediaType ?? "application/octet-stream"};base64,{Convert.ToBase64String(_data)}";

    /// <summary>Two images are equal when their bytes are (their <see cref="Hash"/>es match), whatever media type they were given.</summary>
    public bool Equals(ChatImage? other) => other is not null && string.Equals(Hash, other.Hash, StringComparison.Ordinal);

    /// <inheritdoc />
    public override int GetHashCode() => Hash.GetHashCode(StringComparison.Ordinal);

    /// <summary>The media type, size and the start of the hash: "image/png, 12345 bytes, sha256 3f2a9c1b…".</summary>
    public override string ToString() => $"{MediaType ?? "image"}, {_data.Length} bytes, sha256 {Hash[..8]}…";

    private static string Shorten(string text) => text.Length <= 48 ? text : text[..48] + "…";

    // The media type from the bytes' signature; null when it is none of the common image formats.
    private static string? Sniff(ReadOnlySpan<byte> data) => data switch
    {
        [0x89, (byte)'P', (byte)'N', (byte)'G', ..] => "image/png",
        [0xFF, 0xD8, 0xFF, ..] => "image/jpeg",
        [(byte)'G', (byte)'I', (byte)'F', (byte)'8', ..] => "image/gif",
        [(byte)'B', (byte)'M', ..] => "image/bmp",
        [(byte)'R', (byte)'I', (byte)'F', (byte)'F', _, _, _, _, (byte)'W', (byte)'E', (byte)'B', (byte)'P', ..] => "image/webp",
        [(byte)'I', (byte)'I', 42, 0, ..] or [(byte)'M', (byte)'M', 0, 42, ..] => "image/tiff",
        _ => null,
    };
}

/// <summary>
/// A kind of message part, registered in <see cref="ChatParts"/> under <see cref="Name"/>: how parts of this kind are
/// written to and read from the chat JSON (the content-part objects of saved conversations, <c>{"type": name, …}</c>).
/// The library's are "text" and "image"; a plug-in adds its own (audio, video, documents) with
/// <see cref="ChatParts.Register"/> and makes its own <see cref="ChatPart"/> type for them.
/// </summary>
public interface IChatPartKind
{
    /// <summary>The kind's name: the <see cref="ChatPart.Kind"/> of its parts and the "type" of their JSON.</summary>
    string Name { get; }

    /// <summary>Writes the fields of <paramref name="part"/> (one of this kind) into <paramref name="json"/>, which already holds its "type".</summary>
    void Write(ChatPart part, JsonObject json);

    /// <summary>The part <paramref name="json"/> (a <c>{"type": name, …}</c> object) describes; a clear exception when it is not valid.</summary>
    ChatPart Read(JsonObject json);
}

/// <summary>
/// The kinds of message parts, by name, and what every chat model and template shares about them: the JSON they are
/// saved in (<see cref="ToJson"/>, <see cref="FromJson"/>) and the check that a model or template takes every part it is
/// given (<see cref="ThrowIfUnsupported(IChatModel, ChatRequest, string?)"/>). Library defaults:
/// <list type="bullet">
/// <item>"text" (<see cref="ChatText"/>): <c>{"type": "text", "text": "…"}</c>.</item>
/// <item>"image" (<see cref="ChatImage"/>): <c>{"type": "image", "media_type": "image/png", "data": "&lt;base64&gt;"}</c>.</item>
/// </list>
/// Register another with <see cref="Register"/>; registering a built-in name overrides the built-in, which stays behind it
/// until <see cref="Unregister"/>. Writing and reading a part is one call, so an app's kind can fall back to the
/// library's per call (<see cref="SetPolicy"/>).
/// </summary>
public static class ChatParts
{
    /// <summary>The kind of text parts.</summary>
    public const string Text = "text";

    /// <summary>The kind of image parts.</summary>
    public const string Image = "image";

    /// <summary>The kinds a text-only model or template takes: "text". The default of <see cref="IChatModel.PartKinds"/>.</summary>
    public static IReadOnlySet<string> TextOnly { get; } = FrozenSet.Create(StringComparer.Ordinal, Text);

    private static readonly SlotTable<string, IChatPartKind> Table = BuiltIn();

    private static SlotTable<string, IChatPartKind> BuiltIn()
    {
        var table = new SlotTable<string, IChatPartKind>(nameof(ChatParts), Guard, StringComparer.Ordinal);
        table.RegisterDefault(Text, new TextKind());
        table.RegisterDefault(Image, new ImageKind());
        return table;
    }

    // Writing and reading are single calls: an app's kind falls back (or is shadowed) per call.
    private static IChatPartKind Guard(Slot slot, IChatPartKind app, IChatPartKind library) => new GuardedKind(slot, app, library);

    /// <summary>
    /// Registers <paramref name="kind"/> under its <see cref="IChatPartKind.Name"/>: under a built-in name it takes that
    /// kind's place (the built-in stays behind it, see <see cref="SetPolicy"/>). Names are matched exactly.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IChatPartKind kind)
    {
        ArgumentNullException.ThrowIfNull(kind);
        ArgumentException.ThrowIfNullOrEmpty(kind.Name);
        Table.Register(kind.Name, kind, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's kind <paramref name="name"/> (a built-in name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered kind names, in registration order.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The kind registered as <paramref name="name"/>; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    public static IChatPartKind Get(string name) => Table.Find(name)
        ?? throw new NotSupportedException($"No chat part kind '{name}' is registered ({string.Join(", ", Table.Keys)}); add it with ChatParts.Register.");

    /// <summary>The kind registered as <paramref name="name"/>, or null.</summary>
    public static IChatPartKind? Find(string name) => Table.Find(name);

    /// <summary>The library's kind <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IChatPartKind? Default(string name) => Table.Default(name);

    /// <summary>Version <paramref name="version"/> of the library's kind <paramref name="name"/>; null when there is none.</summary>
    public static IChatPartKind? Default(string name, int version) => Table.Default(name, version);

    /// <summary>The version of the library's kind <paramref name="name"/> (0 when the library has none).</summary>
    public static int DefaultVersion(string name) => Table.DefaultVersion(name);

    /// <summary>Who registered the kind <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>
    /// What happens when the app's kind <paramref name="name"/> fails to write or read a part (<see cref="SlotPolicy.Throw"/>
    /// unless set: the error reaches the caller; <see cref="SlotPolicy.FallBack"/> retries on the library's kind;
    /// <see cref="SlotPolicy.Shadow"/> lets the library's answer and compares the app's on a sample of calls).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary><paramref name="part"/> as a content-part object, <c>{"type": kind, …}</c>, written by its registered kind.</summary>
    public static JsonObject ToJson(ChatPart part)
    {
        ArgumentNullException.ThrowIfNull(part);
        var json = new JsonObject { ["type"] = part.Kind };
        Get(part.Kind).Write(part, json);
        return json;
    }

    /// <summary>
    /// The part a content-part object describes, read by the kind its "type" names (a bare string is text). Throws
    /// <see cref="NotSupportedException"/> for a type no kind is registered for, and <see cref="FormatException"/> for an
    /// object that is not a part.
    /// </summary>
    public static ChatPart FromJson(JsonNode? json) => json switch
    {
        JsonValue v when v.TryGetValue(out string? text) => new ChatText(text),
        JsonObject o when (string?)o["type"] is { Length: > 0 } type => Get(type).Read(o),
        _ => throw new FormatException($"A content part needs a \"type\": {json?.ToJsonString() ?? "null"}"),
    };

    /// <summary>
    /// A message's content as chat JSON (its "content"): a string when it is text alone (one text part; "" for none), so
    /// text-only conversations are written as before; otherwise an array of part objects (<see cref="ToJson"/>), in order.
    /// </summary>
    public static JsonNode ContentToJson(IReadOnlyList<ChatPart> parts)
    {
        ArgumentNullException.ThrowIfNull(parts);
        return parts switch
        {
            [] => JsonValue.Create(""),
            [ChatText only] => JsonValue.Create(only.Text),
            _ => new JsonArray([.. parts.Select(p => (JsonNode)ToJson(p))]),
        };
    }

    /// <summary>
    /// A message's parts from its chat JSON "content" (see <see cref="ContentToJson"/>): none for null or "", one text part
    /// for a string, the parts of an array in order (each read by <see cref="FromJson"/>, so an unknown part type throws).
    /// </summary>
    public static IReadOnlyList<ChatPart> ContentFromJson(JsonNode? content) => content switch
    {
        null => [],
        JsonValue v when v.TryGetValue(out string? text) => text.Length == 0 ? [] : [new ChatText(text)],
        JsonArray parts => [.. parts.Select(FromJson)],
        _ => throw new FormatException($"A message's content must be a string or an array of parts: {content.ToJsonString()}"),
    };

    /// <summary>
    /// Throws when <paramref name="request"/> has a part of a kind <paramref name="model"/> does not take
    /// (<see cref="IChatModel.PartKinds"/>): every chat model of the library calls it before it answers, so a text-only
    /// model given an image fails with a message saying which part, and does not answer as if it were not there.
    /// </summary>
    /// <param name="model">The model asked.</param>
    /// <param name="request">The request.</param>
    /// <param name="name">The model's name in the message (its type's name when null).</param>
    /// <exception cref="NotSupportedException">A part of a kind the model does not take.</exception>
    public static void ThrowIfUnsupported(IChatModel model, ChatRequest request, string? name = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(request);
        ThrowIfUnsupported(request.Messages, model.PartKinds, $"The chat model {(name is null ? model.GetType().Name : $"'{name}'")}");
    }

    /// <summary>
    /// Throws when a message of <paramref name="messages"/> has a part whose kind is not in <paramref name="kinds"/>, naming
    /// <paramref name="who"/> ("The chat template ChatMLTemplate"), the message and the part's kind.
    /// </summary>
    /// <exception cref="NotSupportedException">A part of a kind not in <paramref name="kinds"/>.</exception>
    public static void ThrowIfUnsupported(IReadOnlyList<ChatMessage> messages, IReadOnlySet<string> kinds, string who)
    {
        ArgumentNullException.ThrowIfNull(messages);
        ArgumentNullException.ThrowIfNull(kinds);
        for (int i = 0; i < messages.Count; i++)
        {
            foreach (var part in messages[i].Parts)
            {
                if (!kinds.Contains(part.Kind))
                {
                    throw new NotSupportedException(
                        $"{who} takes only {string.Join(", ", kinds.Order(StringComparer.Ordinal).Select(k => $"'{k}'"))} parts, but message {i + 1} " +
                        $"({messages[i].Role}) has a part of kind '{part.Kind}' ({part}). Use a model that takes '{part.Kind}' parts, or remove them from the conversation.");
                }
            }
        }
    }

    private sealed class TextKind : IChatPartKind
    {
        public string Name => Text;

        public void Write(ChatPart part, JsonObject json) => json["text"] = Of<ChatText>(part, Name).Text;

        public ChatPart Read(JsonObject json) =>
            new ChatText(json["text"] is JsonValue v && v.TryGetValue(out string? text) ? text : throw new FormatException($"A text part needs a \"text\" string: {json.ToJsonString()}"));
    }

    private sealed class ImageKind : IChatPartKind
    {
        public string Name => Image;

        public void Write(ChatPart part, JsonObject json)
        {
            var image = Of<ChatImage>(part, Name);
            if (image.MediaType is { } mediaType)
            {
                json["media_type"] = mediaType;
            }

            json["data"] = Convert.ToBase64String(image.Data.Span);
        }

        public ChatPart Read(JsonObject json)
        {
            string? mediaType = (string?)json["media_type"];
            if ((string?)json["data"] is not { Length: > 0 } data)
            {
                throw new FormatException("An image part needs its bytes as base64 in \"data\".");
            }

            try
            {
                return ChatImage.FromBytes(Convert.FromBase64String(data), mediaType);
            }
            catch (FormatException)
            {
                throw new FormatException("The \"data\" of an image part is not valid base64.");
            }
        }
    }

    private static T Of<T>(ChatPart part, string kind) where T : ChatPart =>
        part as T ?? throw new ArgumentException($"The '{kind}' kind writes {typeof(T).Name} parts, not {part.GetType().Name}.", nameof(part));

    private sealed class GuardedKind(Slot slot, IChatPartKind app, IChatPartKind library) : IChatPartKind
    {
        public string Name => app.Name;

        public void Write(ChatPart part, JsonObject json)
        {
            var written = slot.Call(() => Fill(app, part, json), () => Fill(library, part, json),
                (expected, actual) => Comparisons.Difference(expected.ToJsonString(), actual.ToJsonString()));
            foreach (var (key, value) in written)
            {
                if (key != "type")
                {
                    json[key] = value?.DeepClone();
                }
            }
        }

        public ChatPart Read(JsonObject json) => slot.Call(() => app.Read(json), () => library.Read(json), Comparisons.Exact);

        // The kind's fields written into a copy of `json`, so a failed or shadowed call leaves it untouched.
        private static JsonObject Fill(IChatPartKind kind, ChatPart part, JsonObject json)
        {
            var copy = (JsonObject)json.DeepClone();
            kind.Write(part, copy);
            return copy;
        }
    }
}
