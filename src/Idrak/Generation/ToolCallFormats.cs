// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Generation;

/// <summary>
/// Reads the tool calls a model writes in its answer, as the answer streams. <see cref="ChatOutputParser"/> takes the
/// reasoning in the template's think tags out first and feeds the rest here; one parser is made per reply (by
/// <see cref="ChatTemplate.CreateToolCallParser"/>), so it may keep state. Text that may still turn out to be part of a
/// call is held back until it can be decided; text that is not a call, or a call that does not parse, is answer text.
/// </summary>
public interface IToolCallParser
{
    /// <summary>
    /// Processes the next piece of the answer: what it adds to the answer text, to the reasoning (for formats with their
    /// own reasoning channel) and the calls it completes.
    /// </summary>
    ChatDelta Feed(string text);

    /// <summary>The end of the turn: the text held back, and the calls still open (a call may run to the end of the turn).</summary>
    ChatDelta Finish();
}

/// <summary>What a tool-call parser is made for: the chat template and the request's tools.</summary>
/// <param name="Template">The template the prompt was rendered with.</param>
/// <param name="Tools">The tools of the request (empty when none were given); parsers may read argument types from their schemas.</param>
/// <param name="ToolNames">The names a call may have; null accepts any name.</param>
public sealed record ToolCallContext(ChatTemplate Template, IReadOnlyList<ToolDefinition> Tools, IReadOnlyCollection<string>? ToolNames)
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
/// What a tool-call format's detector sees of a chat template (see <see cref="ToolCallFormats.Detect"/>).
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
    public bool Shows(string text) => (Call ?? Source)?.Contains(text, StringComparison.Ordinal) ?? false;
}

/// <summary>
/// The tool-call formats chat output is parsed in, by name. Each has a detector, which says from a chat template (its
/// source and a rendered probe call, see <see cref="ToolCallProbe"/>) whether the model writes calls this way, and a
/// factory for one reply's parser. A <see cref="ChatTemplate"/> names its format
/// (<see cref="ChatTemplate.ToolCallFormatName"/>); a model's own Jinja template is detected, asking the most recently
/// registered format first and "json" last. Registered:
/// <list type="bullet">
/// <item>"json": JSON calls in the template's <see cref="ToolCallFormat"/> (tags, a bare answer, a list); asked last, it takes any template.</item>
/// <item>"pythonic": a Python list of calls, <c>[get_weather(city="Paris"), now()]</c> (Llama 3.2 and 4 pythonic templates).</item>
/// <item>"qwen3-coder": <c>&lt;tool_call&gt;&lt;function=name&gt;&lt;parameter=key&gt;value&lt;/parameter&gt;&lt;/function&gt;&lt;/tool_call&gt;</c>.</item>
/// <item>"mistral": <c>[TOOL_CALLS]</c> followed by a JSON list of calls or by <c>name[ARGS]{…}</c> per call.</item>
/// <item>"harmony": GPT-OSS channels, calls as <c>&lt;|channel|&gt;commentary to=functions.name … &lt;|message|&gt;{…}&lt;|call|&gt;</c>, reasoning in the analysis channel.</item>
/// <item>"deepseek": <c>&lt;｜tool▁calls▁begin｜&gt;&lt;｜tool▁call▁begin｜&gt;…&lt;｜tool▁sep｜&gt;…&lt;｜tool▁call▁end｜&gt;…</c> (V3 and R1 with a fenced JSON block, V3.1 with bare JSON).</item>
/// </list>
/// Register another with <see cref="Register"/>; registering a built-in name replaces the built-in.
/// </summary>
public static class ToolCallFormats
{
    /// <summary>JSON calls in the template's <see cref="ToolCallFormat"/>.</summary>
    public const string Json = "json";

    /// <summary>A Python list of calls.</summary>
    public const string Pythonic = "pythonic";

    /// <summary>Qwen3-Coder's XML calls with one element per parameter.</summary>
    public const string Qwen3Coder = "qwen3-coder";

    /// <summary>Mistral's [TOOL_CALLS] calls.</summary>
    public const string Mistral = "mistral";

    /// <summary>GPT-OSS's harmony channels.</summary>
    public const string Harmony = "harmony";

    /// <summary>DeepSeek's special-token calls.</summary>
    public const string DeepSeek = "deepseek";

    private sealed record Entry(string Name, Func<ToolCallProbe, bool> Detect, Func<ToolCallContext, IToolCallParser> Create);

    // In the order they are asked: the most recently registered first, "json" (which takes any template) last.
    private static readonly List<Entry> Registry =
    [
        new(DeepSeek, p => p.Shows(DeepSeekToolCallParser.CallBegin), c => new DeepSeekToolCallParser(c)),
        new(Harmony, p => p.Shows("to=functions."), _ => new HarmonyToolCallParser()),
        new(Mistral, p => p.Shows(MistralToolCallParser.Marker), c => new MistralToolCallParser(c)),
        new(Qwen3Coder, p => p.Call is { } call ? call.Contains("<function=" + ToolCallProbe.FunctionName + ">", StringComparison.Ordinal)
            : p.Shows("<function=") && p.Shows("<parameter="), c => new Qwen3CoderToolCallParser(c)),
        new(Pythonic, p => p.Call?.Contains(ToolCallProbe.FunctionName + "(" + ToolCallProbe.ArgumentName + "=", StringComparison.Ordinal) ?? false,
            c => new PythonicToolCallParser(c)),
        new(Json, _ => true, c => new JsonToolCallParser(c)),
    ];

    /// <summary>
    /// Registers the format <paramref name="name"/>: <paramref name="detect"/> says whether a template writes calls this
    /// way, <paramref name="create"/> makes one reply's parser. It replaces the format of the same name (in its place),
    /// or is asked before every format registered so far. Names are matched exactly.
    /// </summary>
    public static void Register(string name, Func<ToolCallProbe, bool> detect, Func<ToolCallContext, IToolCallParser> create)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(detect);
        ArgumentNullException.ThrowIfNull(create);
        lock (Registry)
        {
            int at = Registry.FindIndex(e => e.Name == name);
            if (at >= 0)
            {
                Registry[at] = new Entry(name, detect, create);
            }
            else
            {
                Registry.Insert(0, new Entry(name, detect, create));
            }
        }
    }

    /// <summary>Removes the format registered as <paramref name="name"/>; false when there is none.</summary>
    public static bool Unregister(string name)
    {
        lock (Registry)
        {
            return Registry.RemoveAll(e => e.Name == name) > 0;
        }
    }

    /// <summary>The registered format names, in the order they are asked.</summary>
    public static IReadOnlyCollection<string> Names
    {
        get
        {
            lock (Registry)
            {
                return [.. Registry.Select(e => e.Name)];
            }
        }
    }

    /// <summary>One reply's parser in the format registered as <paramref name="name"/>.</summary>
    public static IToolCallParser Create(string name, ToolCallContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        Entry entry;
        lock (Registry)
        {
            entry = Registry.Find(e => e.Name == name)
                ?? throw new NotSupportedException($"No tool-call format '{name}' is registered ({string.Join(", ", Registry.Select(e => e.Name))}); add it with ToolCallFormats.Register.");
        }

        return entry.Create(context);
    }

    /// <summary>
    /// The name of the first registered format whose detector accepts <paramref name="probe"/> (the most recently
    /// registered first); "json" when none does.
    /// </summary>
    public static string Detect(ToolCallProbe probe)
    {
        ArgumentNullException.ThrowIfNull(probe);
        Entry[] entries;
        lock (Registry)
        {
            entries = [.. Registry];
        }

        // Asked outside the lock: a detector is user code.
        foreach (var entry in entries)
        {
            if (entry.Detect(probe))
            {
                return entry.Name;
            }
        }

        return Json;
    }
}
