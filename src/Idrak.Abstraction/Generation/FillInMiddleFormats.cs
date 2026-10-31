// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text;

namespace Idrak.Abstraction.Generation;

/// <summary>A file of the repository given as context to a fill-in-the-middle prompt: its path and the part of its text given.</summary>
/// <param name="Path">The file's path as the model is to read it (relative to the repository, '/' separated).</param>
/// <param name="Text">The text given (the whole file or a part of it).</param>
public sealed record FillInMiddleSnippet(string Path, string Text);

/// <summary>
/// What a fill-in-the-middle prompt is made for (<see cref="IFillInMiddleFormat"/>): the text before the caret, the text
/// after it, and optional repository context, with the model's tokenizer, whose special tokens the format writes.
/// </summary>
/// <param name="Tokenizer">The model's tokenizer: the format's special tokens are looked up in it (<see cref="IPretrainedTokenizer.IdOf"/>).</param>
/// <param name="Prefix">The text before the caret.</param>
/// <param name="Suffix">The text after the caret.</param>
public sealed record FillInMiddleContext(IPretrainedTokenizer Tokenizer, string Prefix, string Suffix)
{
    /// <summary>Other files of the repository, in the order they are written (empty: none).</summary>
    public IReadOnlyList<FillInMiddleSnippet> Snippets { get; init; } = [];

    /// <summary>The path of the file being completed, written after the snippets when the format writes paths; null: none.</summary>
    public string? Path { get; init; }

    /// <summary>The repository's name, for formats with a repository token (Qwen's); null: none written.</summary>
    public string? Repository { get; init; }
}

/// <summary>
/// A fill-in-the-middle prompt (<see cref="IKind{TRequest, T}.Create"/> of an <see cref="IFillInMiddleFormat"/>): what to feed the model and the tokens that end
/// the middle. A text model (<see cref="ITextModel"/>) takes <see cref="Text"/> with <see cref="Stop"/> as
/// <see cref="GenerationOptions.Stop"/> (<see cref="FillInMiddleFormats.Options"/>); a caller that feeds ids takes
/// <see cref="Ids"/>, the same prompt.
/// </summary>
public interface IFillInMiddlePrompt
{
    /// <summary>The prompt as text: the format's special tokens written as the tokenizer's own entries, which it reads back as those tokens.</summary>
    string Text { get; }

    /// <summary>The same prompt as token ids: each special token by its id, each piece of text between them encoded by the tokenizer.</summary>
    IReadOnlyList<int> Ids { get; }

    /// <summary>The tokens that end the middle, as text (the ones the tokenizer has), in the format's order.</summary>
    IReadOnlyList<string> Stop { get; }

    /// <summary>The ids of <see cref="Stop"/>, in the same order.</summary>
    IReadOnlyList<int> StopIds { get; }

    /// <summary>
    /// Where the context's prefix ends in <see cref="Text"/>: a prompt for the same context with a longer prefix starts
    /// with <c>Text[..PrefixEnd]</c>, which is what a kept cache (<c>TextGenerator.KeepCache</c>) reuses while the caret
    /// moves on.
    /// </summary>
    int PrefixEnd { get; }

    /// <summary>How many of the context's snippets the prompt holds (0 when the format writes none): a caller says so when it is fewer than it gave.</summary>
    int Snippets { get; }
}

/// <summary>A fill-in-the-middle prompt as data (<see cref="IFillInMiddlePrompt"/>), for formats of one's own.</summary>
public sealed record FillInMiddlePrompt(string Text, IReadOnlyList<int> Ids, IReadOnlyList<string> Stop, IReadOnlyList<int> StopIds, int PrefixEnd, int Snippets)
    : IFillInMiddlePrompt;

/// <summary>
/// A fill-in-the-middle format, registered by its <see cref="IKind.Name"/> in <see cref="FillInMiddleFormats"/>:
/// <see cref="Detect"/> says from a model's tokenizer (its special tokens) whether the model was trained to fill in this
/// way, and <see cref="IKind{TRequest, T}.Create"/> lays a context out as a prompt. The library's two layouts
/// (<see cref="FillInMiddleLayout"/>) are the building blocks; a family's plug-in names its tokens.
/// </summary>
public interface IFillInMiddleFormat : IKind<FillInMiddleContext, IFillInMiddlePrompt>
{
    /// <summary>Whether <paramref name="tokenizer"/> has the format's special tokens (the model can be prompted this way).</summary>
    bool Detect(IPretrainedTokenizer tokenizer);
}

/// <summary>The order a <see cref="FillInMiddleLayout"/> writes the prefix and the suffix in.</summary>
public enum FillInMiddleOrder
{
    /// <summary>Prefix, suffix, then the middle (PSM): <c>PRE prefix SUF suffix MID</c>.</summary>
    PrefixSuffixMiddle,

    /// <summary>
    /// Suffix first (SPM): with a middle token, <c>PRE SUF suffix MID prefix</c> (the FIM paper's, CodeLlama's); without
    /// one, <c>SUF suffix PRE prefix</c> (Mistral's). The prompt ends with the prefix, so the model continues it.
    /// </summary>
    SuffixPrefixMiddle,
}

/// <summary>
/// The special tokens of a <see cref="FillInMiddleLayout"/>, as the tokenizer's entries (<see cref="IPretrainedTokenizer.IdOf"/>):
/// a family's plug-in names them.
/// </summary>
/// <param name="Prefix">The token before the prefix.</param>
/// <param name="Suffix">The token before the suffix.</param>
/// <param name="Middle">The token that asks for the middle; null for a suffix-first layout without one (Mistral's).</param>
public sealed record FillInMiddleTokens(string Prefix, string Suffix, string? Middle)
{
    /// <summary>Written first when the tokenizer has it (a beginning-of-text token the model expects and the tokenizer does not add); null: none.</summary>
    public string? Begin { get; init; }

    /// <summary>The token before the repository's name (Qwen's <c>&lt;|repo_name|&gt;</c>); null: the name is not written.</summary>
    public string? Repository { get; init; }

    /// <summary>
    /// The token before each file's path (<c>&lt;|file_sep|&gt;</c>): each snippet is written as <c>SEP path\ntext\n</c>, then
    /// <c>SEP path\n</c> of the file being completed, before the prefix and suffix. Null: <see cref="SnippetMarker"/>.
    /// </summary>
    public string? FileSeparator { get; init; }

    /// <summary>
    /// For formats without a <see cref="FileSeparator"/>: the text written before a snippet's path at the start of the
    /// prefix (DeepSeek-Coder's repository files are <c>#path\ntext</c>): <c>marker path\ntext\n</c> per snippet, then
    /// <c>marker path\n</c> of the file being completed. Null (and no separator): snippets are not written
    /// (<see cref="IFillInMiddlePrompt.Snippets"/> is 0).
    /// </summary>
    public string? SnippetMarker { get; init; }

    /// <summary>The tokens that end the middle, in order; those the tokenizer lacks are left out of a prompt's stops.</summary>
    public IReadOnlyList<string> Stop { get; init; } = [];

    /// <summary>The tokens <see cref="FillInMiddleLayout.Detect"/> needs: the prefix, suffix and middle tokens, the repository token and the file separator when given.</summary>
    public IEnumerable<string> Required()
    {
        yield return Prefix;
        yield return Suffix;
        if (Middle is not null)
        {
            yield return Middle;
        }

        if (Repository is not null)
        {
            yield return Repository;
        }

        if (FileSeparator is not null)
        {
            yield return FileSeparator;
        }
    }
}

/// <summary>
/// The library's fill-in-the-middle layouts, the building blocks of the families' formats: prefix-suffix-middle and
/// suffix-prefix-middle (<see cref="FillInMiddleOrder"/>) over a family's tokens (<see cref="FillInMiddleTokens"/>).
/// It detects a tokenizer that has every required token, and lays a context out as
/// <c>[Begin] [repository and snippets] PRE prefix SUF suffix MID</c> (or suffix first), each special token written as
/// the tokenizer's entry and fed by its id, the texts between them encoded by the tokenizer.
/// </summary>
public sealed class FillInMiddleLayout : IFillInMiddleFormat
{
    /// <summary>A layout named <paramref name="name"/> in <paramref name="order"/> over <paramref name="tokens"/>.</summary>
    /// <exception cref="ArgumentException">A prefix-first layout without a middle token, or an empty token.</exception>
    public FillInMiddleLayout(string name, string summary, FillInMiddleOrder order, FillInMiddleTokens tokens)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentException.ThrowIfNullOrWhiteSpace(summary);
        ArgumentNullException.ThrowIfNull(tokens);
        if (tokens.Required().Any(string.IsNullOrEmpty) || tokens.Stop.Any(string.IsNullOrEmpty) || tokens.Begin is "")
        {
            throw new ArgumentException($"Fill-in-the-middle format '{name}': a token is empty.", nameof(tokens));
        }

        if (order == FillInMiddleOrder.PrefixSuffixMiddle && tokens.Middle is null)
        {
            throw new ArgumentException($"Fill-in-the-middle format '{name}': a prefix-suffix-middle layout needs a middle token.", nameof(tokens));
        }

        (Name, Summary, Order, Tokens) = (name, summary, order, tokens);
    }

    /// <summary>A prefix-suffix-middle layout (<see cref="FillInMiddleOrder.PrefixSuffixMiddle"/>).</summary>
    public static FillInMiddleLayout PrefixSuffixMiddle(string name, string summary, FillInMiddleTokens tokens) => new(name, summary, FillInMiddleOrder.PrefixSuffixMiddle, tokens);

    /// <summary>A suffix-prefix-middle layout (<see cref="FillInMiddleOrder.SuffixPrefixMiddle"/>).</summary>
    public static FillInMiddleLayout SuffixPrefixMiddle(string name, string summary, FillInMiddleTokens tokens) => new(name, summary, FillInMiddleOrder.SuffixPrefixMiddle, tokens);

    /// <inheritdoc />
    public string Name { get; }

    /// <inheritdoc />
    public string Summary { get; }

    /// <inheritdoc />
    public IReadOnlySet<string> Keys => OptionKeys.None;

    /// <summary>The order the prefix and suffix are written in.</summary>
    public FillInMiddleOrder Order { get; }

    /// <summary>The family's tokens.</summary>
    public FillInMiddleTokens Tokens { get; }

    /// <inheritdoc />
    public bool Detect(IPretrainedTokenizer tokenizer)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        foreach (string token in Tokens.Required())
        {
            if (tokenizer.IdOf(token) is null)
            {
                return false;
            }
        }

        return true;
    }

    /// <inheritdoc />
    /// <exception cref="NotSupportedException">The tokenizer lacks a required token (the message names the missing ones).</exception>
    public IFillInMiddlePrompt Create(FillInMiddleContext request)
    {
        ArgumentNullException.ThrowIfNull(request);
        var tokenizer = request.Tokenizer;
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(request.Prefix);
        ArgumentNullException.ThrowIfNull(request.Suffix);
        var missing = Tokens.Required().Where(t => tokenizer.IdOf(t) is null).ToList();
        if (missing.Count > 0)
        {
            throw new NotSupportedException($"Fill-in-the-middle format '{Name}' needs the tokens {string.Join(", ", Tokens.Required())}; the tokenizer has no "
                                            + $"{string.Join(", ", missing)}: detect the model's format with FillInMiddleFormats.Detect.");
        }

        var builder = new Builder(tokenizer, request.Prefix.Length + request.Suffix.Length + 64);
        if (Tokens.Begin is { } begin && tokenizer.IdOf(begin) is int beginId)
        {
            builder.Special(begin, beginId);
        }

        int snippets = 0;
        bool separated = Tokens.FileSeparator is not null;
        if (separated)
        {
            snippets = Repository(builder, request, Tokens.FileSeparator!);
        }

        string prefix = request.Prefix, suffix = request.Suffix;
        int prefixEnd;
        if (Order == FillInMiddleOrder.PrefixSuffixMiddle)
        {
            builder.Special(Tokens.Prefix);
            snippets += separated ? 0 : Marked(builder, request);
            builder.Plain(prefix);
            prefixEnd = builder.Length;
            builder.Special(Tokens.Suffix);
            builder.Plain(suffix);
            builder.Special(Tokens.Middle!);
        }
        else
        {
            if (Tokens.Middle is { } middle)
            {
                builder.Special(Tokens.Prefix);
                builder.Special(Tokens.Suffix);
                builder.Plain(suffix);
                builder.Special(middle);
            }
            else
            {
                builder.Special(Tokens.Suffix);
                builder.Plain(suffix);
                builder.Special(Tokens.Prefix);
            }

            snippets += separated ? 0 : Marked(builder, request);
            builder.Plain(prefix);
            prefixEnd = builder.Length;
        }

        var stop = new List<string>(Tokens.Stop.Count);
        var stopIds = new List<int>(Tokens.Stop.Count);
        foreach (string token in Tokens.Stop)
        {
            if (tokenizer.IdOf(token) is int id && !stopIds.Contains(id))
            {
                stop.Add(token);
                stopIds.Add(id);
            }
        }

        return new FillInMiddlePrompt(builder.Text(), builder.Ids, stop, stopIds, prefixEnd, snippets);
    }

    // The repository's name, each snippet and the path of the file being completed, each after its token.
    private int Repository(Builder builder, FillInMiddleContext request, string separator)
    {
        if (request.Snippets.Count == 0 && request.Path is null && request.Repository is null)
        {
            return 0;
        }

        if (Tokens.Repository is { } repository && request.Repository is { } name)
        {
            builder.Special(repository);
            builder.Plain(name);
            builder.Plain("\n");
        }

        foreach (var snippet in request.Snippets)
        {
            builder.Special(separator);
            builder.Plain(snippet.Path);
            builder.Plain("\n");
            builder.Plain(snippet.Text);
            builder.Plain("\n");
        }

        if (request.Path is { } path)
        {
            builder.Special(separator);
            builder.Plain(path);
            builder.Plain("\n");
        }

        return request.Snippets.Count;
    }

    // Without a file separator: each snippet as the family's marked text at the start of the prefix, then the file's own path.
    private int Marked(Builder builder, FillInMiddleContext request)
    {
        if (Tokens.SnippetMarker is not { } marker || (request.Snippets.Count == 0 && request.Path is null))
        {
            return 0;
        }

        foreach (var snippet in request.Snippets)
        {
            builder.Plain(marker);
            builder.Plain(snippet.Path);
            builder.Plain("\n");
            builder.Plain(snippet.Text);
            builder.Plain("\n");
        }

        if (request.Path is { } path)
        {
            builder.Plain(marker);
            builder.Plain(path);
            builder.Plain("\n");
        }

        return request.Snippets.Count;
    }

    // The prompt's text and ids together: plain text is gathered until the next special token, then encoded in one piece
    // (as the tokenizer splits a text at its special tokens), so the ids are the text's.
    private sealed class Builder(IPretrainedTokenizer tokenizer, int capacity)
    {
        private readonly StringBuilder _text = new(capacity);
        private int _plainFrom;

        public List<int> Ids { get; } = new(capacity / 3 + 8);

        public int Length => _text.Length;

        public void Plain(string text) => _text.Append(text);

        public void Special(string token) => Special(token, tokenizer.IdOf(token)!.Value);

        public void Special(string token, int id)
        {
            Flush();
            _text.Append(token);
            Ids.Add(id);
            _plainFrom = _text.Length;
        }

        public string Text()
        {
            Flush();
            return _text.ToString();
        }

        private void Flush()
        {
            if (_text.Length > _plainFrom)
            {
                string piece = _text.ToString(_plainFrom, _text.Length - _plainFrom);
                tokenizer.EncodeRange(piece, 0, piece.Length, Ids);
                _plainFrom = _text.Length;
            }
        }
    }
}

/// <summary>
/// The fill-in-the-middle formats by name (<see cref="IFillInMiddleFormat"/>). The library registers none: each is a
/// family's (Idrak.Text.Plugins registers "qwen", "deepseek", "gemma" and "mistral", each with its family, on
/// <see cref="FillInMiddleLayout"/>), or an app's. A model's format is detected from its tokenizer
/// (<see cref="Detect"/>, the most recently registered format asked first); a model whose family registers none has
/// no format (null), and asking for one by name that nobody registered is refused as not registered.
/// </summary>
public static class FillInMiddleFormats
{
    // In the order they are asked: the most recently registered first.
    private static readonly KindTable<IFillInMiddleFormat> Table = new(nameof(FillInMiddleFormats), "Fill-in-the-middle format",
        "add it with FillInMiddleFormats.Register", Guard, StringComparer.Ordinal, newestFirst: true);

    private static IFillInMiddleFormat Guard(Slot slot, IFillInMiddleFormat app, IFillInMiddleFormat library) => new GuardedFormat(slot, app, library);

    /// <summary>
    /// Registers <paramref name="format"/> under its <see cref="IKind.Name"/>: under a registered name it takes that
    /// format's place; a new name is asked before every format registered so far. Names are matched exactly.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IFillInMiddleFormat format) => Table.Register(format, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the format <paramref name="name"/>; false when none is registered.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered format names, in the order they are asked.</summary>
    public static IReadOnlyCollection<string> Names => Table.Names;

    /// <summary>Who registered the format <paramref name="name"/>: the registering assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>The library's format <paramref name="name"/>: always null (the library registers none).</summary>
    public static IFillInMiddleFormat? Default(string name) => Table.Default(name);

    /// <summary>What happens when the app's format <paramref name="name"/> fails, as every registry's policy (<see cref="SlotPolicy"/>).</summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary>The format registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotRegisteredException">None is: the message names the registered ones.</exception>
    public static IFillInMiddleFormat Get(ReadOnlySpan<char> name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(FillInMiddleFormats), name.ToString(), Table.Names,
            $"No fill-in-the-middle format '{name}' is registered ({(Table.Names.Count == 0 ? "none" : string.Join(", ", Table.Names))}); add it with "
            + "FillInMiddleFormats.Register, or reference Idrak.Text.Plugins and call TextPlugins.RegisterAll().");

    /// <summary>The prompt the format registered as <paramref name="name"/> makes for <paramref name="context"/>.</summary>
    public static IFillInMiddlePrompt Create(ReadOnlySpan<char> name, FillInMiddleContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        return Get(name).Create(context);
    }

    /// <summary>
    /// The name of the first registered format (the most recently registered first) whose tokens <paramref name="tokenizer"/>
    /// has; null when none has them (the model has no registered format: a caller falls back to completing the prefix
    /// alone, and says so).
    /// </summary>
    public static string? Detect(IPretrainedTokenizer tokenizer)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        foreach (var format in Table.Kinds)
        {
            if (format.Detect(tokenizer))
            {
                return format.Name;
            }
        }

        return null;
    }

    /// <summary>The format <see cref="Detect"/> finds for <paramref name="tokenizer"/>.</summary>
    /// <exception cref="NotRegisteredException">No registered format's tokens are in the tokenizer.</exception>
    public static IFillInMiddleFormat For(IPretrainedTokenizer tokenizer) => Detect(tokenizer) is { } name ? Get(name)
        : throw new NotRegisteredException(nameof(FillInMiddleFormats), "", Table.Names,
            $"No fill-in-the-middle format registered for this tokenizer (registered: {(Table.Names.Count == 0 ? "none" : string.Join(", ", Table.Names))}, "
            + "none of whose tokens it has); register the model's family (reference Idrak.Text.Plugins and call TextPlugins.RegisterAll()) or add its "
            + "format with FillInMiddleFormats.Register.");

    /// <summary><paramref name="options"/> with the prompt's stops added to <see cref="GenerationOptions.Stop"/> (each once).</summary>
    public static GenerationOptions Options(IFillInMiddlePrompt prompt, GenerationOptions options)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentNullException.ThrowIfNull(options);
        return options with { Stop = [.. options.Stop.Concat(prompt.Stop).Distinct(StringComparer.Ordinal)] };
    }

    private sealed class GuardedFormat(Slot slot, IFillInMiddleFormat app, IFillInMiddleFormat library) : IFillInMiddleFormat
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public IReadOnlySet<string> Keys => app.Keys;

        public bool Detect(IPretrainedTokenizer tokenizer) => slot.Call(() => app.Detect(tokenizer), () => library.Detect(tokenizer), Comparisons.Exact);

        public IFillInMiddlePrompt Create(FillInMiddleContext request) => slot.Call(() => app.Create(request), () => library.Create(request));
    }
}
