// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Generation;

namespace Idrak.Cli.Shared;

/// <summary>
/// The generation settings a command that answers prompts accepts (chat, run, batch, compare, complete, agent; serve
/// and bench may reuse them): sampling (<c>--temperature</c>, <c>--top-k</c>, <c>--top-p</c>, <c>--seed</c>), length
/// (<c>--max-tokens</c>), the system prompt (<c>--system</c>, <c>-s</c>), tool assemblies (<c>--tools FILE.dll</c>,
/// repeatable) and the reasoning mode (<c>--think</c>, <c>--no-think</c>). Mutable, so chat's <c>/set</c> changes them.
/// </summary>
internal sealed class GenerationSettings
{
    /// <summary>The value options.</summary>
    public static readonly string[] ValueOptions = ["--temperature", "--top-k", "--top-p", "--max-tokens", "--seed", "--system", "--tools"];

    /// <summary>The flags.</summary>
    public static readonly string[] Flags = ["--think", "--no-think"];

    /// <summary>Their short forms.</summary>
    public static readonly IReadOnlyDictionary<string, string> ShortForms = new Dictionary<string, string> { ["-s"] = "--system" };

    /// <summary>The options' lines for a command's usage text.</summary>
    public const string Help =
        "  -s, --system TEXT      the system prompt\n" +
        "      --temperature T    sampling temperature (default 0.6; 0 picks the most likely token)\n" +
        "      --top-k N          sample among the N most likely tokens (default 20; 0 = all)\n" +
        "      --top-p P          nucleus sampling (default 0.95; 1 = off)\n" +
        "      --max-tokens N     longest answer in tokens (default: until the model stops, at most 4096)\n" +
        "      --seed N           random seed for reproducible answers\n" +
        "      --think            ask for reasoning (shown apart); --no-think: suppress it (default: the model's own)\n" +
        "      --tools FILE.dll   tools the model may call: the [Tool] methods of an assembly (repeatable)\n";

    /// <summary>
    /// The environment variables that affect a command that loads and runs a language model (the last lines of its
    /// help; plans/idrak-cli.md, "Environment variables").
    /// </summary>
    public const string EnvironmentHelp =
        "Environment:\n" +
        "  models     IDRAK_CACHE, HF_TOKEN, HF_HOME, HF_HUB_CACHE, HF_ENDPOINT (where models are cached and downloaded from)\n" +
        "  running    IDRAK_MATMUL, IDRAK_OFFLOAD, IDRAK_VULKAN_DEFAULT (precision of products, offloading, default device)\n" +
        "  the tool   IDRAK_CONFIG, IDRAK_TRACE, NO_COLOR (the config file, error stacks, no colour)\n";

    /// <summary>The setting names <c>/set</c> knows.</summary>
    public static readonly string[] Names = ["temperature", "top-k", "top-p", "max-tokens", "seed", "think"];

    /// <summary>Softmax temperature; 0 means greedy.</summary>
    public float Temperature { get; set; } = 0.6f;

    /// <summary>Top-k (0 = all tokens).</summary>
    public int TopK { get; set; } = 20;

    /// <summary>Top-p (1 = off).</summary>
    public float TopP { get; set; } = 0.95f;

    /// <summary>Longest answer in tokens (-1: until a stop sequence or <see cref="TextGenerator.MaxTokens"/>).</summary>
    public int MaxTokens { get; set; } = -1;

    /// <summary>Random seed (null: random).</summary>
    public int? Seed { get; set; }

    /// <summary>The system prompt, or null.</summary>
    public string? System { get; set; }

    /// <summary>Reasoning: true asks for it, false suppresses it, null leaves it to the model.</summary>
    public bool? Think { get; set; }

    /// <summary>Assemblies whose tools the model may call.</summary>
    public IReadOnlyList<string> ToolAssemblies { get; set; } = [];

    /// <summary>The settings from the command line.</summary>
    public static GenerationSettings From(CommandContext context)
    {
        var settings = new GenerationSettings();
        foreach (string name in Names.Where(n => n != "think"))
        {
            if (context.Option("--" + name) is { } value && settings.Set(name, value) is { } error)
            {
                throw new UsageException(error);
            }
        }

        if (context.Flag("--think") && context.Flag("--no-think"))
        {
            throw new UsageException("Give --think or --no-think, not both.");
        }

        settings.Think = context.Flag("--think") ? true : context.Flag("--no-think") ? false : null;
        settings.System = context.Option("--system");
        settings.ToolAssemblies = context.Options("--tools");
        return settings;
    }

    /// <summary>Sets <paramref name="name"/> (one of <see cref="Names"/>) from text; returns an error message, or null.</summary>
    public string? Set(string name, string value)
    {
        var invariant = CultureInfo.InvariantCulture;
        bool Float(out float f) => float.TryParse(value, NumberStyles.Float, invariant, out f) && float.IsFinite(f);
        bool Int(out int i) => int.TryParse(value, NumberStyles.Integer, invariant, out i);
        switch (name)
        {
            case "temperature" when Float(out float t) && t >= 0:
                Temperature = t;
                return null;
            case "top-k" when Int(out int k) && k >= 0:
                TopK = k;
                return null;
            case "top-p" when Float(out float p) && p is > 0 and <= 1:
                TopP = p;
                return null;
            case "max-tokens" when Int(out int n) && (n > 0 || n == -1):
                MaxTokens = n;
                return null;
            case "seed" when value is "random" or "none":
                Seed = null;
                return null;
            case "seed" when Int(out int s):
                Seed = s;
                return null;
            case "think" when value is "on" or "true" or "off" or "false" or "default":
                Think = value is "on" or "true" ? true : value is "off" or "false" ? false : null;
                return null;
            case "temperature":
                return $"--temperature needs a number of 0 or more, not '{value}'.";
            case "top-k":
                return $"--top-k needs a whole number of 0 or more, not '{value}'.";
            case "top-p":
                return $"--top-p needs a number above 0 and at most 1, not '{value}'.";
            case "max-tokens":
                return $"--max-tokens needs a positive whole number (or -1 for no limit), not '{value}'.";
            case "seed":
                return $"--seed needs a whole number (or 'random'), not '{value}'.";
            case "think":
                return $"think is on, off or default, not '{value}'.";
            default:
                return $"Unknown setting '{name}'; the settings are {string.Join(", ", Names)}.";
        }
    }

    /// <summary>The library's options for these settings with a context window of <paramref name="context"/> tokens.</summary>
    public GenerationOptions ToOptions(int context) => new()
    {
        Temperature = Temperature <= 0 ? 1f : Temperature,
        TopK = Temperature <= 0 ? 1 : TopK,
        TopP = Temperature <= 0 ? 1f : TopP,
        RepeatPenalty = 1f,
        Seed = Seed,
        NumCtx = context,
        NumPredict = MaxTokens,
    };

    /// <summary>The settings as name and value pairs (for <c>/set</c> without arguments, and JSON output).</summary>
    public IEnumerable<(string Name, string Value)> Describe()
    {
        var invariant = CultureInfo.InvariantCulture;
        yield return ("temperature", Temperature.ToString(invariant));
        yield return ("top-k", TopK.ToString(invariant));
        yield return ("top-p", TopP.ToString(invariant));
        yield return ("max-tokens", MaxTokens < 0 ? "unlimited" : MaxTokens.ToString(invariant));
        yield return ("seed", Seed?.ToString(invariant) ?? "random");
        yield return ("think", Think switch { true => "on", false => "off", null => "default" });
    }

    /// <summary>A copy (compare runs two sets side by side).</summary>
    public GenerationSettings Clone() => (GenerationSettings)MemberwiseClone();
}
