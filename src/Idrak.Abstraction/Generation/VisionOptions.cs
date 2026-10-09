// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Generation;

/// <summary>
/// Options for a vision-language model's vision family, by name, as text: what its processor takes beyond the decoded
/// pixels (how many views of an image it makes, a resolution, a crop). The library gives them meaning nowhere: the
/// family's encoder reads its own keys (<see cref="IVisionEncoder.Blocks"/>, <see cref="IVisionEncoder.Encode"/>) and
/// refuses any other (<see cref="ThrowIfUnknown"/>, naming the keys it takes). They come from the command line
/// (<c>--vision-option KEY=VALUE</c>), a request (<see cref="ChatRequest.VisionOptions"/>, <c>"vision_options"</c> in
/// the server's JSON), or the app; the family's own defaults (its configuration files) apply to every key not given.
/// Keys compare ordinally; values are kept as given (JSON booleans and numbers as their invariant text). Immutable.
/// </summary>
public sealed class VisionOptions : IReadOnlyDictionary<string, string>
{
    private readonly SortedDictionary<string, string> _values;

    /// <summary>The options <paramref name="values"/> (a later duplicate key wins).</summary>
    /// <exception cref="ArgumentException">A key is empty or has spaces around it, or a value is null.</exception>
    public VisionOptions(IEnumerable<KeyValuePair<string, string>> values)
    {
        ArgumentNullException.ThrowIfNull(values);
        _values = new SortedDictionary<string, string>(StringComparer.Ordinal);
        foreach (var (key, value) in values)
        {
            if (string.IsNullOrWhiteSpace(key) || key.Trim() != key)
            {
                throw new ArgumentException($"A vision option needs a name without spaces around it, not '{key}'.", nameof(values));
            }

            _values[key] = value ?? throw new ArgumentException($"The vision option '{key}' has no value.", nameof(values));
        }
    }

    /// <summary>No options: the family's defaults throughout.</summary>
    public static VisionOptions Empty { get; } = new([]);

    /// <summary>
    /// The options written as <c>KEY=VALUE</c>, one per item (the command line's <c>--vision-option</c>); spaces around
    /// the key and the value are dropped; a later duplicate key wins.
    /// </summary>
    /// <exception cref="FormatException">An item is not KEY=VALUE.</exception>
    public static VisionOptions Parse(IEnumerable<string> pairs)
    {
        ArgumentNullException.ThrowIfNull(pairs);
        var values = new List<KeyValuePair<string, string>>();
        foreach (string pair in pairs)
        {
            int at = pair?.IndexOf('=', StringComparison.Ordinal) ?? -1;
            if (at <= 0 || pair![..at].Trim().Length == 0)
            {
                throw new FormatException($"A vision option is KEY=VALUE (such as crops=4), not '{pair}'.");
            }

            values.Add(KeyValuePair.Create(pair[..at].Trim(), pair[(at + 1)..].Trim()));
        }

        return new VisionOptions(values);
    }

    /// <summary>
    /// The options in a JSON object (a request's <c>"vision_options"</c>): strings as they are, booleans and numbers as
    /// their invariant text, null as not given (the family's default).
    /// </summary>
    /// <exception cref="FormatException">A value is an object or an array.</exception>
    public static VisionOptions FromJson(JsonObject json)
    {
        ArgumentNullException.ThrowIfNull(json);
        var values = new List<KeyValuePair<string, string>>();
        foreach (var (key, node) in json)
        {
            if (node is null)
            {
                continue;
            }

            string text = node.GetValueKind() switch
            {
                JsonValueKind.String => node.GetValue<string>(),
                JsonValueKind.True => "true",
                JsonValueKind.False => "false",
                JsonValueKind.Number => node.ToJsonString(),
                var kind => throw new FormatException($"The vision option '{key}' is {kind.ToString().ToLowerInvariant()}: give a string, a number or true/false."),
            };
            values.Add(KeyValuePair.Create(key, text));
        }

        return new VisionOptions(values);
    }

    /// <summary>These options with <paramref name="over"/>'s on top (its keys win); this when it is null or empty.</summary>
    public VisionOptions With(VisionOptions? over) =>
        over is null || over.Count == 0 ? this : Count == 0 ? over : new VisionOptions(_values.Concat(over._values));

    /// <summary>
    /// Throws unless every key is one <paramref name="family"/> takes (<paramref name="accepted"/>), naming them; a family
    /// that takes none says so.
    /// </summary>
    /// <exception cref="ArgumentException">A key is not in <paramref name="accepted"/>.</exception>
    public void ThrowIfUnknown(string family, IReadOnlyCollection<string> accepted)
    {
        ArgumentNullException.ThrowIfNull(accepted);
        var unknown = _values.Keys.Where(k => !accepted.Contains(k, StringComparer.Ordinal)).ToList();
        if (unknown.Count == 0)
        {
            return;
        }

        string which = unknown.Count == 1 ? $"option '{unknown[0]}'" : $"options {string.Join(", ", unknown.Select(k => $"'{k}'"))}";
        throw new ArgumentException(accepted.Count == 0
            ? $"The vision family {family} takes no vision options (given {which})."
            : $"The vision family {family} does not take the vision {which}; it takes: {string.Join(", ", accepted)}.");
    }

    /// <summary>The option <paramref name="key"/> as true or false (true, false, 1, 0, yes, no, on, off), or <paramref name="otherwise"/> when not given.</summary>
    /// <exception cref="ArgumentException">The value is none of those.</exception>
    public bool Flag(string key, bool otherwise) => !_values.TryGetValue(key, out var text) ? otherwise : text.Trim().ToLowerInvariant() switch
    {
        "true" or "1" or "yes" or "on" => true,
        "false" or "0" or "no" or "off" => false,
        _ => throw new ArgumentException($"The vision option {key} is true or false, not '{text}'."),
    };

    /// <summary>The option <paramref name="key"/> as a whole number, or <paramref name="otherwise"/> when not given.</summary>
    /// <exception cref="ArgumentException">The value is not a whole number.</exception>
    public int Integer(string key, int otherwise) => !_values.TryGetValue(key, out var text) ? otherwise
        : int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) ? value
        : throw new ArgumentException($"The vision option {key} is a whole number, not '{text}'.");

    /// <summary>The option <paramref name="key"/> as a number, or <paramref name="otherwise"/> when not given.</summary>
    /// <exception cref="ArgumentException">The value is not a finite number.</exception>
    public double Number(string key, double otherwise) => !_values.TryGetValue(key, out var text) ? otherwise
        : double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double value) && double.IsFinite(value) ? value
        : throw new ArgumentException($"The vision option {key} is a number, not '{text}'.");

    /// <summary>The options as a JSON object of strings.</summary>
    public JsonObject ToJson() => new([.. _values.Select(p => KeyValuePair.Create(p.Key, (JsonNode?)p.Value))]);

    /// <inheritdoc />
    public string this[string key] => _values[key];

    /// <inheritdoc />
    public IEnumerable<string> Keys => _values.Keys;

    /// <inheritdoc />
    public IEnumerable<string> Values => _values.Values;

    /// <inheritdoc />
    public int Count => _values.Count;

    /// <inheritdoc />
    public bool ContainsKey(string key) => _values.ContainsKey(key);

    /// <inheritdoc />
    public bool TryGetValue(string key, out string value) => _values.TryGetValue(key, out value!);

    /// <inheritdoc />
    public IEnumerator<KeyValuePair<string, string>> GetEnumerator() => _values.GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    /// <summary>"KEY=VALUE, KEY=VALUE" in key order; "none" when empty.</summary>
    public override string ToString() => Count == 0 ? "none" : string.Join(", ", _values.Select(p => $"{p.Key}={p.Value}"));
}
