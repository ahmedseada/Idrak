// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of a kind of message part (<see cref="IChatPartKind"/>, registered in <see cref="ChatParts"/>) for one kind
/// name: it reads a part's JSON (<c>{"type": name, …}</c>) into a part of its own kind, writes it back into JSON it reads
/// again to an equal part, the same every time, writing leaves the "type" as it was, and it agrees with the library's kind
/// of the same name (<see cref="ChatParts.Default(string)"/>): the same parts read and the same JSON written.
/// </summary>
/// <param name="kind">The kind's name ("image", or a kind of your own).</param>
/// <param name="samples">
/// Parts of the kind as JSON objects, each with its "type"; the built-in kinds have their own, a kind of your own needs
/// some.
/// </param>
public sealed class ChatPartKindSuite(string kind, IReadOnlyList<JsonObject>? samples = null) : ContractSuite<IChatPartKind>
{
    private readonly IReadOnlyList<JsonObject> _samples = samples ?? BuiltInSamples(kind)
        ?? throw new ArgumentException($"Chat part kind '{kind}' is not built in: give sample parts (as JSON) to check it with.", nameof(samples));

    /// <summary>The kind checked.</summary>
    public string Kind { get; } = kind;

    /// <inheritdoc />
    public override string Name => "chat-part-kind";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases() => _samples.Select((s, i) => Case($"sample {i + 1}", s));

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        switch (Kind)
        {
            case ChatParts.Text:
                var text = new StringBuilder();
                int length = random.Next(0, large ? 20_000 : 200);
                while (text.Length < length)
                {
                    int c = random.Next(6) switch { 0 => random.Next(0x20, 0x7F), 1 => random.Next(0x600, 0x6FF), 2 => random.Next(0x4E00, 0x9FFF), 3 => '\n', 4 => '"', _ => 0x1F600 + random.Next(80) };
                    text.Append(char.ConvertFromUtf32(c));
                }

                return Case($"random text of {text.Length} chars", new JsonObject { ["type"] = Kind, ["text"] = text.ToString() });
            case ChatParts.Image:
                var bytes = new byte[random.Next(1, large ? 1 << 20 : 4096)];
                random.NextBytes(bytes);
                string? mediaType = random.Next(3) switch { 0 => "image/png", 1 => "image/jpeg", _ => null };
                var json = new JsonObject { ["type"] = Kind };
                if (mediaType is not null)
                {
                    json["media_type"] = mediaType;
                }

                json["data"] = Convert.ToBase64String(bytes);
                return Case($"random image of {bytes.Length} bytes", json);
            default:
                return Case("a sample", _samples[random.Next(_samples.Count)]);
        }
    }

    /// <inheritdoc />
    public override void Run(IChatPartKind implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var json = @case.Data["part"]!.AsObject();
        checks.Check("named as the kind", implementation.Name == Kind, $"named '{implementation.Name}', checked as '{Kind}'");

        var part = implementation.Read((JsonObject)json.DeepClone());
        checks.Check("reads a part of its kind", part.Kind == Kind, $"read a '{part.Kind}' part");
        var written = Write(implementation, part);
        checks.Check("writing keeps the type", (string?)written["type"] == Kind, $"wrote {written.ToJsonString()}");
        var again = implementation.Read((JsonObject)written.DeepClone());
        checks.Check("round trip: the part written reads back equal", Equals(part, again), $"{part} became {again}");
        checks.Compare("the same JSON every time", Comparisons.Difference(written.ToJsonString(), Write(implementation, again).ToJsonString()));

        var library = ChatParts.Default(Kind);
        if (library is null)
        {
            checks.Skip("agrees with the library's kind", $"the library has no '{Kind}' kind");
            return;
        }

        var expected = library.Read((JsonObject)json.DeepClone());
        checks.Check("reads the part the library's kind reads", Equals(expected, part), $"read {part}, the library reads {expected}");
        checks.Compare("writes the JSON the library's kind writes", Comparisons.Difference(Write(library, expected).ToJsonString(), written.ToJsonString()));
    }

    private static JsonObject Write(IChatPartKind kind, ChatPart part)
    {
        var json = new JsonObject { ["type"] = part.Kind };
        kind.Write(part, json);
        return json;
    }

    private static ContractCase Case(string name, JsonObject part)
    {
        string shown = part.ToJsonString();
        return new($"{name}: {(shown.Length > 60 ? shown[..60] + "…" : shown)}", new JsonObject { ["part"] = part.DeepClone() });
    }

    private static List<JsonObject>? BuiltInSamples(string kind) => kind switch
    {
        ChatParts.Text =>
        [
            new() { ["type"] = kind, ["text"] = "" },
            new() { ["type"] = kind, ["text"] = "Read this scan." },
            new() { ["type"] = kind, ["text"] = "اقرأ هذه الوثيقة\n\"quoted\" \u0000 😀" },
        ],
        ChatParts.Image =>
        [
            new() { ["type"] = kind, ["media_type"] = "image/png", ["data"] = Convert.ToBase64String([0x89, (byte)'P', (byte)'N', (byte)'G', 13, 10, 26, 10, 0, 0]) },
            new() { ["type"] = kind, ["data"] = Convert.ToBase64String([0xFF, 0xD8, 0xFF, 0xE0, 1, 2, 3]) },
            new() { ["type"] = kind, ["media_type"] = "image/x-own", ["data"] = Convert.ToBase64String([1]) },
        ],
        _ => null,
    };
}
