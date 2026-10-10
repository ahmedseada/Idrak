// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text;

namespace Idrak.Nlp.Abstractions;

/// <summary>
/// What of a model's answer a metric compares with the reference: the answer as it is, the values of its JSON, or
/// another reading. Register one with <see cref="AnswerTexts.Register"/>.
/// </summary>
public interface IAnswerText
{
    /// <summary>The name it is registered under.</summary>
    string Name { get; }

    /// <summary>One line on what it keeps of an answer.</summary>
    string Summary { get; }

    /// <summary>The text of <paramref name="answer"/> to compare (not normalized: <see cref="AnswerTexts.Normalize"/> does that).</summary>
    string Text(string answer);
}

/// <summary>
/// The answer texts, by name (ignoring case). The library's: <see cref="Raw"/> (the answer as it is) and
/// <see cref="Values"/> (its JSON's string and number values in document order, one a line, the JSON read through the
/// lenient <see cref="JsonRepairs"/>; an answer without JSON as it is). Register another with <see cref="Register"/>;
/// one under a library name shadows it, which <see cref="Unregister"/> brings back.
/// </summary>
public static class AnswerTexts
{
    /// <summary>The answer as it is.</summary>
    public const string Raw = "raw";

    /// <summary>The values of the answer's JSON in document order, one a line (an answer without JSON as it is).</summary>
    public const string Values = "values";

    private static readonly SlotTable<string, IAnswerText> Table = BuiltIn();

    private static SlotTable<string, IAnswerText> BuiltIn()
    {
        var table = new SlotTable<string, IAnswerText>(nameof(AnswerTexts), (slot, app, library) => new GuardedText(slot, app, library), StringComparer.OrdinalIgnoreCase);
        table.RegisterDefault(Raw, new RawAnswerText());
        table.RegisterDefault(Values, new JsonValuesAnswerText());
        return table;
    }

    /// <summary>Registers <paramref name="text"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IAnswerText text)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentException.ThrowIfNullOrWhiteSpace(text.Name);
        Table.Register(text.Name, text, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's answer text of that name (the library's, if any, answers again).</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The answer text of that name, or null.</summary>
    public static IAnswerText? Find(string name) => Table.Find(name);

    /// <summary>The answer text of that name; throws when none is registered.</summary>
    public static IAnswerText Get(string name) => Table.Find(name)
        ?? throw new NotSupportedException($"Answer text '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with AnswerTexts.Register.");

    /// <summary>The library's answer text of that name, or null.</summary>
    public static IAnswerText? Default(string name) => Table.Default(name);

    /// <summary>The assembly that registered the answer text of that name.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>How an app's answer text of that name is run against the library's.</summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary>One line an answer text.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(t => $"{t.Name}: {t.Summary}"));

    /// <summary>The text of <paramref name="answer"/> by the answer text <paramref name="name"/>, normalized (<see cref="Normalize"/>).</summary>
    public static string Of(string answer, string name = Raw) => Normalize(Get(name).Text(answer ?? ""));

    /// <summary>
    /// Text as metrics compare it: Unicode composed (NFC), byte-order marks dropped, every whitespace run one space,
    /// trimmed (so line breaks and spacing do not count as errors).
    /// </summary>
    public static string Normalize(string? text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return "";
        }

        var builder = new StringBuilder(text.Length);
        bool space = false;
        foreach (char c in text.Normalize(NormalizationForm.FormC))
        {
            if (c == '﻿')
            {
                continue;
            }

            if (char.IsWhiteSpace(c))
            {
                space = builder.Length > 0;
                continue;
            }

            if (space)
            {
                builder.Append(' ');
                space = false;
            }

            builder.Append(c);
        }

        return builder.ToString();
    }

    private sealed class GuardedText(Slot slot, IAnswerText app, IAnswerText library) : IAnswerText
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public string Text(string answer) =>
            slot.Call(() => app.Text(answer), () => library.Text(answer), (a, b) => a == b ? null : $"\"{a}\" against the library's \"{b}\"");
    }
}
