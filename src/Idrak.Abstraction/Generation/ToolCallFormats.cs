// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Generation;

/// <summary>
/// Reads the tool calls a model writes in its answer, as the answer streams. <see cref="ChatOutputParser"/> takes the
/// reasoning in the template's think tags out first and feeds the rest here; one parser is made per reply (by the
/// template's format, <see cref="IToolCallFormat"/>), so it may keep state. Text that may still turn out to be part of a
/// call is held back until it can be decided; text that is not a call, or a call that does not parse, is answer text.
/// </summary>
public interface IToolCallParser
{
    /// <summary>
    /// Processes the next piece of the answer: what it adds to the answer text, to the reasoning (for formats with their
    /// own reasoning channel) and the calls it completes. The piece is read in place: a parser that holds text back
    /// copies what it holds, never the span itself.
    /// </summary>
    ChatDelta Feed(ReadOnlySpan<char> text);

    /// <summary>The end of the turn: the text held back, and the calls still open (a call may run to the end of the turn).</summary>
    ChatDelta Finish();
}

/// <summary>What a tool-call parser is made for: the chat template and the request's tools.</summary>
/// <param name="Template">The template the prompt was rendered with.</param>
/// <param name="Tools">The tools of the request (empty when none were given); parsers may read argument types from their schemas.</param>
/// <param name="ToolNames">The names a call may have; null accepts any name.</param>
public sealed record ToolCallContext(IChatTemplate Template, IReadOnlyList<ToolDefinition> Tools, IReadOnlyCollection<string>? ToolNames)
{
    /// <summary>Whether a call may name <paramref name="name"/> (any name when <see cref="ToolNames"/> is null).</summary>
    public bool Allows(string name) => ToolNames is null || ToolNames.Contains(name);

    /// <summary>The request's tool named <paramref name="name"/>, or null.</summary>
    public ToolDefinition? Tool(string name)
    {
        foreach (var tool in Tools)
        {
            if (tool.Name == name)
            {
                return tool;
            }
        }

        return null;
    }
}

/// <summary>
/// What a tool-call format's detector sees of a chat template (see <see cref="IToolCallFormat.Detect"/>).
/// </summary>
/// <param name="Source">The template's text (Jinja source), or null when the template is not text.</param>
/// <param name="Call">
/// The assistant turn rendered with one probe call (<see cref="FunctionName"/> with <see cref="ArgumentName"/> set to
/// <see cref="ArgumentValue"/>), from where it differs from the same turn with a plain answer to where they agree
/// again; null when the template renders no tool calls or could not be rendered.
/// </param>
public sealed record ToolCallProbe(string? Source, string? Call)
{
    /// <summary>The probe call's function name.</summary>
    public const string FunctionName = "ns_probe_function";

    /// <summary>The probe call's only argument.</summary>
    public const string ArgumentName = "ns_probe_argument";

    /// <summary>The probe argument's value (a string).</summary>
    public const string ArgumentValue = "ns-probe-value";

    /// <summary>Whether the rendered probe call contains <paramref name="text"/>, or, when there is none, the template's source does.</summary>
    public bool Shows(ReadOnlySpan<char> text) => (Call ?? Source).AsSpan().Contains(text, StringComparison.Ordinal);
}

/// <summary>
/// A tool-call format chat output is parsed in, registered by its <see cref="IKind.Name"/> in <see cref="ToolCallFormats"/>:
/// <see cref="Detect"/> says from a chat template (its source and a rendered probe call) whether the model writes calls
/// this way, and <see cref="IKind{TRequest, T}.Create"/> makes one reply's parser.
/// </summary>
public interface IToolCallFormat : IKind<ToolCallContext, IToolCallParser>
{
    /// <summary>Whether the template <paramref name="probe"/> shows writes its calls in this format.</summary>
    bool Detect(ToolCallProbe probe);
}

/// <summary>
/// The tool-call formats chat output is parsed in, by name (<see cref="IToolCallFormat"/>). A chat template names its
/// format (<see cref="IChatTemplate.ToolCallFormatName"/>); a model's own Jinja template is detected, asking the most
/// recently registered format first and "json" last. Registered:
/// <list type="bullet">
/// <item>"json": JSON calls in the template's <see cref="ToolCallFormat"/> (tags, a bare answer, a list); asked last, it takes any template.</item>
/// <item>"pythonic": a Python list of calls, <c>[get_weather(city="Paris"), now()]</c>.</item>
/// </list>
/// A family's own format is its plug-in's (Idrak.Text.Plugins registers "qwen3-coder", "mistral", "harmony" and
/// "deepseek", each with its family, on <see cref="MarkedToolCallParser"/> and <see cref="ToolCallText"/>). Register
/// another with <see cref="Register"/>; registering a built-in name overrides the built-in, which stays behind it until
/// the app unregisters it.
/// </summary>
public static class ToolCallFormats
{
    /// <summary>JSON calls in the template's <see cref="ToolCallFormat"/>.</summary>
    public const string Json = "json";

    /// <summary>A Python list of calls.</summary>
    public const string Pythonic = "pythonic";

    // In the order they are asked: the most recently registered first, "json" (which takes any template) last.
    private static readonly KindTable<IToolCallFormat> Table = BuiltIn();

    private static KindTable<IToolCallFormat> BuiltIn()
    {
        var table = new KindTable<IToolCallFormat>(nameof(ToolCallFormats), "Tool-call format", "add it with ToolCallFormats.Register", Guard, StringComparer.Ordinal,
            newestFirst: true);
        table.RegisterDefault(new JsonFormat());   // each goes first: the last registered is asked first
        table.RegisterDefault(new PythonicFormat());
        return table;
    }

    // Detection is one call; a parser holds one reply's state, so it falls back when it is made.
    private static IToolCallFormat Guard(Slot slot, IToolCallFormat app, IToolCallFormat library) => new GuardedFormat(slot, app, library);

    /// <summary>
    /// Registers <paramref name="format"/> under its <see cref="IKind.Name"/>. Under a registered name it takes that
    /// format's place (a built-in stays behind it, see <see cref="SetPolicy"/>); a new name is asked before every format
    /// registered so far. Names are matched exactly.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IToolCallFormat format) => Table.Register(format, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's format <paramref name="name"/> (a built-in name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered format names, in the order they are asked.</summary>
    public static IReadOnlyCollection<string> Names => Table.Names;

    /// <summary>Who registered the format <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>The library's format <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IToolCallFormat? Default(string name) => Table.Default(name);

    /// <summary>
    /// What happens when the app's format <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set: the error reaches the caller;
    /// <see cref="SlotPolicy.FallBack"/> retries on the library's): detection falls back per call, a parser when it is made.
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary>The format registered as <paramref name="name"/>; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names the registered ones.</exception>
    public static IToolCallFormat Get(ReadOnlySpan<char> name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(ToolCallFormats), name.ToString(), Table.Names,
            $"No tool-call format '{name}' is registered ({string.Join(", ", Table.Names)}); add it with ToolCallFormats.Register, or reference Idrak.Text.Plugins and call TextPlugins.RegisterAll().");

    /// <summary>One reply's parser in the format registered as <paramref name="name"/>.</summary>
    public static IToolCallParser Create(ReadOnlySpan<char> name, ToolCallContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        return Get(name).Create(context);
    }

    /// <summary>
    /// The name of the first registered format whose detector accepts <paramref name="probe"/> (the most recently
    /// registered first); "json" when none does.
    /// </summary>
    public static string Detect(ToolCallProbe probe)
    {
        ArgumentNullException.ThrowIfNull(probe);
        foreach (var format in Table.Kinds)
        {
            if (format.Detect(probe))
            {
                return format.Name;
            }
        }

        return Json;
    }

    private sealed class JsonFormat : IToolCallFormat
    {
        public string Name => Json;

        public string Summary => "JSON calls in the template's tool-call shape (tags, a bare answer, a list); takes any template";

        public IReadOnlySet<string> Keys => OptionKeys.None;

        public bool Detect(ToolCallProbe probe) => true;

        public IToolCallParser Create(ToolCallContext context) => new JsonToolCallParser(context);
    }

    private sealed class PythonicFormat : IToolCallFormat
    {
        public string Name => Pythonic;

        public string Summary => "a Python list of calls: [get_weather(city=\"Paris\"), now()]";

        public IReadOnlySet<string> Keys => OptionKeys.None;

        public bool Detect(ToolCallProbe probe) => probe.Call.AsSpan().Contains(ToolCallProbe.FunctionName + "(" + ToolCallProbe.ArgumentName + "=", StringComparison.Ordinal);

        public IToolCallParser Create(ToolCallContext context) => new PythonicToolCallParser(context);
    }

    private sealed class GuardedFormat(Slot slot, IToolCallFormat app, IToolCallFormat library) : IToolCallFormat
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public IReadOnlySet<string> Keys => app.Keys;

        public bool Detect(ToolCallProbe probe) => slot.Call(() => app.Detect(probe), () => library.Detect(probe), Comparisons.Exact);

        public IToolCallParser Create(ToolCallContext context) => slot.Call(() => app.Create(context), () => library.Create(context));
    }
}
