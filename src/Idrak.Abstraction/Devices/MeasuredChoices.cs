// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

namespace Idrak.Abstraction.Devices;

/// <summary>
/// The choices a backend measured on its device at run time (which kernel variant of a product, how many splits of a
/// reduction, the workgroup width, which attention path), as one rank of a cluster hands them to another
/// (<c>IMeasuredChoicesProvider</c>). Some choices pick different algorithms whose results differ in the last bits
/// (an online against a two-pass softmax, a cooperative-matrix product's order), so ranks that each measure their own
/// can
/// run different kernels; a cluster run gives every rank rank 0's (<see cref="Cluster.ClusterOptions.ShareTuning"/>).
/// </summary>
/// <param name="Kind">The device kind they were measured on ("vulkan", "cuda").</param>
/// <param name="Identity">
/// What they apply to, as the backend names it: the device (its UUID), its driver and the version of its kernels. A
/// backend takes choices of its own identity only.
/// </param>
/// <param name="Device">The device's name, for messages.</param>
/// <param name="Entries">Each choice under its key, both as the backend writes them (no tab or line break in
/// either).</param>
public sealed record MeasuredChoices(string Kind, string Identity, string Device, IReadOnlyDictionary<string, string> Entries)
{
    private const string Magic = "idrak-measured-choices 1";

    /// <summary>The choices as bytes to send to another rank (UTF-8 lines: a header, then a key and its value a line).</summary>
    /// <exception cref="ArgumentException">A field, key or value holds a tab or a line break.</exception>
    public byte[] ToBytes()
    {
        var text = new StringBuilder();
        text.Append(Magic).Append('\n');
        foreach (string field in (string[])[Kind, Identity, Device])
        {
            text.Append(Plain(field, "field")).Append('\n');
        }

        foreach (var (key, value) in Entries)
        {
            text.Append(Plain(key, "key")).Append('\t').Append(Plain(value, "value")).Append('\n');
        }

        return Encoding.UTF8.GetBytes(text.ToString());
    }

    /// <summary>The choices <see cref="ToBytes"/> wrote.</summary>
    /// <exception cref="InvalidDataException">The bytes are not measured choices.</exception>
    public static MeasuredChoices FromBytes(ReadOnlySpan<byte> bytes)
    {
        string[] lines = Encoding.UTF8.GetString(bytes).Split('\n');
        if (lines.Length < 4 || lines[0] != Magic)
        {
            throw new InvalidDataException("not a rank's measured choices (no header)");
        }

        var entries = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int i = 4; i < lines.Length; i++)
        {
            if (lines[i].Length == 0)
            {
                continue;
            }

            int tab = lines[i].IndexOf('\t', StringComparison.Ordinal);
            if (tab <= 0)
            {
                throw new InvalidDataException($"line {i + 1} of a rank's measured choices has no key and value: {lines[i]}");
            }

            entries[lines[i][..tab]] = lines[i][(tab + 1)..];
        }

        return new MeasuredChoices(lines[1], lines[2], lines[3], entries);
    }

    private static string Plain(string text, string what) =>
        text.AsSpan().IndexOfAny('\t', '\n', '\r') < 0 ? text : throw new ArgumentException($"A measured choice's {what} holds a tab or a line break: {text}");
}

/// <summary>What a backend did with another rank's measured choices (<c>IMeasuredChoicesProvider</c>).</summary>
/// <param name="Taken">Whether it runs them now; false: it keeps its own (another device, driver or kind), and
/// <paramref name="Detail"/> says why.</param>
/// <param name="Choices">How many choices it took (those known when it took them; a live share adds rank 0's later
/// ones).</param>
/// <param name="Detail">What happened, or why not, for the run's report.</param>
public sealed record MeasuredChoicesResult(bool Taken, int Choices, string Detail);

