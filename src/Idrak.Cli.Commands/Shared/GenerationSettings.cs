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
    public static readonly string[] ValueOptions =
    [
        "--temperature", "--top-k", "--top-p", "--min-p", "--max-tokens", "--system", "--tools", "--repeat-penalty", "--presence-penalty", "--frequency-penalty",
        "--repeat-last-n", "--repeat-stop",
    ];

    /// <summary>The flags.</summary>
    public static readonly string[] Flags = ["--think", "--no-think"];

    /// <summary>Their short forms.</summary>
    public static readonly IReadOnlyDictionary<string, string> ShortForms = new Dictionary<string, string> { ["-s"] = "--system" };

    /// <summary>The options' lines for a command's usage text.</summary>
    public const string Help =
        "  -s, --system TEXT      the system prompt\n" +
        "      --temperature T    sampling temperature (default: the model's own, else 0.6; 0 picks the most likely token)\n" +
        "      --top-k N          sample among the N most likely tokens (default: the model's, else 20; 0 = all)\n" +
        "      --top-p P          nucleus sampling (default: the model's, else 0.95; 1 = off)\n" +
        "      --max-tokens N     longest answer in tokens (default: until the model stops, at most 4096)\n" +
        "      --seed N           random seed for reproducible answers\n" +
        "      --repeat-penalty F divides the scores of tokens seen in the last --repeat-last-n tokens (default: the model's, else 1 = off)\n" +
        "      --presence-penalty F, --frequency-penalty F  subtracted once from each recent token's score, or once per\n" +
        "                         time it was seen (default 0 = off); --repeat-last-n N: how far back they look (default 64)\n" +
        "      --repeat-stop N    end the answer when one phrase repeats N times in a row (default 0 = off)\n" +
        "      --think            ask for reasoning (shown apart); --no-think: suppress it (default: the model's own)\n" +
        "      --tools FILE.dll   tools the model may call: the tools an assembly declares (repeatable)\n";

    /// <summary>The setting names <c>/set</c> knows.</summary>
    public static readonly string[] Names =
        ["temperature", "top-k", "top-p", "min-p", "max-tokens", "seed", "think", "repeat-penalty", "presence-penalty", "frequency-penalty", "repeat-last-n", "repeat-stop"];

    /// <summary>Softmax temperature; 0 means greedy.</summary>
    public float Temperature { get; set; } = 0.6f;

    /// <summary>Top-k (0 = all tokens).</summary>
    public int TopK { get; set; } = 20;

    /// <summary>Top-p (1 = off).</summary>
    public float TopP { get; set; } = 0.95f;

    /// <summary>Longest answer in tokens (-1: until a stop sequence or the context window is full).</summary>
    public int MaxTokens { get; set; } = -1;

    /// <summary>Random seed (null: random).</summary>
    public int? Seed { get; set; }

    /// <summary>Divides the scores of recently seen tokens (1 = off).</summary>
    public float RepeatPenalty { get; set; } = 1f;

    /// <summary>Subtracted once from every recently seen token's score (0 = off).</summary>
    public float PresencePenalty { get; set; }

    /// <summary>Subtracted once per recent occurrence (0 = off).</summary>
    public float FrequencyPenalty { get; set; }

    /// <summary>How many recent tokens the penalties look at.</summary>
    public int RepeatLastN { get; set; } = 64;

    /// <summary>End the answer when one phrase repeats this many times in a row (0 = off).</summary>
    public int RepeatStop { get; set; }

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
        string? error = SetValue(name, value);
        if (error is null)
        {
            _given.Add(name);
        }

        return error;
    }

    /// <summary>Min-p (0 = off).</summary>
    public float MinP { get; set; }

    // The settings the user gave (the command line, /set): the model's recommended sampling never replaces them.
    private HashSet<string> _given = new(StringComparer.Ordinal);

    /// <summary>The sampling values the user gave (the command line or <c>/set</c>), as <see cref="SamplingSettings"/>.</summary>
    public SamplingSettings Given => new()
    {
        Temperature = _given.Contains("temperature") ? Temperature : null,
        TopK = _given.Contains("top-k") ? TopK : null,
        TopP = _given.Contains("top-p") ? TopP : null,
        MinP = _given.Contains("min-p") ? MinP : null,
        RepeatPenalty = _given.Contains("repeat-penalty") ? RepeatPenalty : null,
    };

    /// <summary>
    /// Takes the model's <paramref name="recommended"/> sampling (<see cref="SamplingSettings"/>, its generation_config.json)
    /// for every sampling value the user did not give (<see cref="Given"/>); the CLI's own defaults stay for the rest.
    /// Says in --verbose which values came from the model. Returns what was taken ("" when nothing; null without a recommendation).
    /// </summary>
    public string? TakeRecommended(CommandContext? context, SamplingSettings? recommended, string model)
    {
        if (recommended is null)
        {
            return null;
        }

        var baseline = new GenerationOptions { Temperature = Temperature <= 0 ? 1f : Temperature, TopK = TopK, TopP = TopP, MinP = MinP, RepeatPenalty = RepeatPenalty };
        var resolved = SamplingSettings.Resolve(baseline, Given, recommended);
        var taken = resolved.FromModel.ToList();
        if (taken.Count == 0)
        {
            context?.Detail($"{model} recommends its own sampling ({SamplingSettings.FileName}); the values given were kept");
            return "";
        }

        var options = resolved.Options;
        bool greedy = taken.Any(s => s.Name == "greedy");
        bool keep = greedy || Temperature <= 0 && _given.Contains("temperature");   // greedy decoding: the CLI's temperature 0
        Temperature = greedy ? 0f : keep ? Temperature : options.Temperature;
        TopK = keep ? TopK : options.TopK;
        TopP = keep ? TopP : options.TopP;
        MinP = keep ? MinP : options.MinP;
        RepeatPenalty = options.RepeatPenalty;
        string said = string.Join(", ", taken.Select(s => $"{s.Name} {s.Value}"));
        context?.Detail($"sampling from {model}'s {SamplingSettings.FileName}: {said} (now: {resolved.Describe()})");
        return said;
    }

    private string? SetValue(string name, string value)
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
            case "min-p" when Float(out float m) && m is >= 0 and <= 1:
                MinP = m;
                return null;
            case "min-p":
                return $"--min-p needs a number from 0 to 1 (0 = off), not '{value}'.";
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
            case "repeat-penalty" when Float(out float r) && r is >= 0.5f and <= 3f:
                RepeatPenalty = r;
                return null;
            case "presence-penalty" when Float(out float pp) && pp is >= -2f and <= 2f:
                PresencePenalty = pp;
                return null;
            case "frequency-penalty" when Float(out float fp) && fp is >= -2f and <= 2f:
                FrequencyPenalty = fp;
                return null;
            case "repeat-last-n" when Int(out int last) && last >= 0:
                RepeatLastN = last;
                return null;
            case "repeat-stop" when Int(out int times) && (times == 0 || times >= 2):
                RepeatStop = times;
                return null;
            case "repeat-penalty":
                return $"--repeat-penalty needs a number from 0.5 to 3 (1 = off), not '{value}'.";
            case "presence-penalty" or "frequency-penalty":
                return $"--{name} needs a number from -2 to 2 (0 = off), not '{value}'.";
            case "repeat-last-n":
                return $"--repeat-last-n needs a whole number of 0 or more, not '{value}'.";
            case "repeat-stop":
                return $"--repeat-stop needs 0 (off) or a whole number of 2 or more, not '{value}'.";
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
        MinP = Temperature <= 0 ? 0f : MinP,
        RepeatPenalty = RepeatPenalty,
        PresencePenalty = PresencePenalty,
        FrequencyPenalty = FrequencyPenalty,
        RepeatLastN = RepeatLastN,
        RepeatStop = RepeatStop,
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
        yield return ("min-p", MinP.ToString(invariant));
        yield return ("max-tokens", MaxTokens < 0 ? "unlimited" : MaxTokens.ToString(invariant));
        yield return ("seed", Seed?.ToString(invariant) ?? "random");
        yield return ("think", Think switch { true => "on", false => "off", null => "default" });
        yield return ("repeat-penalty", RepeatPenalty.ToString(invariant));
        yield return ("presence-penalty", PresencePenalty.ToString(invariant));
        yield return ("frequency-penalty", FrequencyPenalty.ToString(invariant));
        yield return ("repeat-last-n", RepeatLastN.ToString(invariant));
        yield return ("repeat-stop", RepeatStop == 0 ? "off" : RepeatStop.ToString(invariant));
    }

    /// <summary>A copy (compare runs two sets side by side).</summary>
    public GenerationSettings Clone()
    {
        var copy = (GenerationSettings)MemberwiseClone();
        copy._given = new HashSet<string>(_given, StringComparer.Ordinal);
        return copy;
    }
}
