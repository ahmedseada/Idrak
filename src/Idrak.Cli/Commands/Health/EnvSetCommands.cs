// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands.Health;

/// <summary>
/// <c>idrak env set</c> and <c>idrak env unset</c>: save environment variables in the config ("env"), where every run
/// of the tool applies them (<see cref="SavedEnvironment"/>). This is the same in every terminal and shell, unlike
/// <c>$env:NAME=</c>, <c>export</c> or <c>set</c>. <c>--user</c> also sets them in the Windows user environment, for
/// programs started without the tool.
/// </summary>
internal abstract class EnvSaveCommand : Command
{
    public override IReadOnlyCollection<string> ValueOptions => ["--profile"];

    public override IReadOnlyCollection<string> Flags => ["--user"];

    // The object the variables are saved in: the config's "env", or a profile's.
    protected static JsonObject Saved(CommandContext context, bool create)
    {
        JsonObject scope = context.Config.Root;
        if (context.Option("--profile") is { } profile)
        {
            if (scope["profiles"] is not JsonObject profiles)
            {
                profiles = [];
                if (create)
                {
                    scope["profiles"] = profiles;
                }
            }

            if (profiles[profile] is not JsonObject named)
            {
                named = [];
                if (create)
                {
                    profiles[profile] = named;
                }
            }

            scope = named;
        }

        if (scope[SavedEnvironment.Key] is not JsonObject saved)
        {
            saved = [];
            if (create)
            {
                scope[SavedEnvironment.Key] = saved;
            }
        }

        return saved;
    }

    // The variable a name stands for, or a usage error naming the problem.
    protected static EnvironmentVariables.Variable Saveable(string name)
    {
        var variable = EnvironmentVariables.Find(name.Trim().ToUpperInvariant()) ?? EnvironmentVariables.Find(name.Trim())
            ?? throw new UsageException($"Not a variable Idrak reads: {name} (idrak help env lists them).");
        return SavedEnvironment.WhyNot(variable) is { } why ? throw new UsageException(why + ".") : variable;
    }

    // Sets (value) or removes (null) the variable in the Windows user environment, or says how on other systems.
    protected static string? User(CommandContext context, string name, string? value)
    {
        if (!context.Flag("--user"))
        {
            return null;
        }

        if (OperatingSystem.IsWindows())
        {
            Environment.SetEnvironmentVariable(name, value, EnvironmentVariableTarget.User);
            return value is null ? "removed from the Windows user environment (new terminals)" : "also set in the Windows user environment (new terminals)";
        }

        return value is null ? $"--user: remove the line setting {name} from your shell's profile (~/.profile, ~/.bashrc or ~/.zshrc)"
            : $"--user: add  export {name}='{value}'  to your shell's profile (~/.profile, ~/.bashrc or ~/.zshrc); the tool cannot change other programs' environment there";
    }

    protected static string Where(CommandContext context) =>
        context.Option("--profile") is { } profile ? $"{context.Config.Path}, profile {profile}" : context.Config.Path;
}

/// <summary><c>idrak env set [NAME [VALUE]]</c>: saves variables, asking for them when NAME or VALUE is left out.</summary>
internal sealed class EnvSetCommand : EnvSaveCommand
{
    // The variables offered first in the questions: the ones a run or a test usually needs.
    internal static readonly string[] Common =
    [
        "IDRAK_DEVICES", "IDRAK_FILTER", "IDRAK_CUDA_DEBUG", "IDRAK_WINDOW_KERNELS", "IDRAK_VULKAN_DEFAULT", "IDRAK_CACHE", "IDRAK_PORT",
        "IDRAK_TIMEOUT", "IDRAK_TRACE", "IDRAK_LANG", "IDRAK_LANG_RENDER",
    ];

    public override string Name => "env set";

    public override string Summary => "Save environment variables for every idrak run, in any terminal (asks for them, with suggested values)";

    public override string Usage => """
        [NAME [VALUE]] [--user] [--profile NAME]

        Saves variables in the config ("env"). Every idrak command sets them before it starts, and the programs it
        starts (idrak test's test runner, the agent's commands) get them too, the same in PowerShell, cmd, bash,
        zsh or fish. A variable already set in the terminal wins over the saved value.

        With no NAME, it asks: pick a variable (its number, its name or part of it; ? lists all), then its value
        (Enter takes the suggested one, - removes it), then whether to set another. With NAME and no VALUE, it
        asks for the value only.

        Arguments:
          NAME   the variable (idrak help env lists them)
          VALUE  its value; - or an empty value removes it

        Options:
              --user          also set it in the Windows user environment, for programs started without idrak
                              (other systems: prints the line for the shell's profile)
              --profile NAME  save it in the named profile (applies when that profile is in use)

        Tokens and keys cannot be saved here (idrak login stores them).

        Examples:
          idrak env set
          idrak env set IDRAK_DEVICES cuda:0
          idrak env set IDRAK_CUDA_DEBUG
          idrak env set IDRAK_WINDOW_KERNELS 0 --profile laptop
          idrak env unset IDRAK_CUDA_DEBUG
        """;

    public override int Run(CommandContext context)
    {
        if (context.Positional.Count > 2)
        {
            throw new UsageException("env set takes one NAME and one VALUE; quote a value with spaces.");
        }

        var changes = new JsonArray();
        if (context.Positional.Count == 2)
        {
            changes.Add(Save(context, Saveable(context.Positional[0]), context.Positional[1]));
        }
        else if (context.Positional.Count == 1)
        {
            var variable = Saveable(context.Positional[0]);
            changes.Add(Save(context, variable, AskValue(context, variable)));
        }
        else
        {
            if (!Terminal.IsInteractive(context) || context.Json && Terminal.TestInput is null)
            {
                throw new UsageException("env set with no NAME asks for the variables, and there is no terminal to ask on; give NAME VALUE (e.g. idrak env set IDRAK_DEVICES cpu).");
            }

            context.Write($"Variables saved here apply to every idrak command, in any terminal (saved in {Where(context)}).");
            ListCommon(context);
            while (Pick(context) is { } variable)
            {
                changes.Add(Save(context, variable, AskValue(context, variable)));
                string again = Terminal.Ask(context, "Set another (y/n)?", "y");
                if (!again.StartsWith('y') && !again.StartsWith('Y'))
                {
                    break;
                }
            }

            if (changes.Count == 0)
            {
                context.Write("Nothing changed.");
            }
            else
            {
                context.Write("idrak env lists them; idrak env unset NAME removes one.");
            }
        }

        context.WriteJson(new JsonObject { ["file"] = context.Config.Path, ["profile"] = context.Option("--profile"), ["changes"] = changes });
        return ExitCodes.Ok;
    }

    // The common variables, numbered, with their values now.
    private static void ListCommon(CommandContext context)
    {
        var saved = SavedEnvironment.Read(context.Config);
        context.Write("");
        for (int i = 0; i < Common.Length; i++)
        {
            var variable = EnvironmentVariables.Find(Common[i])!;
            context.Write($"  {i + 1,2}  {variable.Name,-22} {Now(variable, saved),-16} {Cut(variable.Meaning, 70)}");
        }

        context.Write("");
    }

    // Asks for a variable until one is chosen; null when the answer is empty (finish).
    private static EnvironmentVariables.Variable? Pick(CommandContext context)
    {
        while (true)
        {
            string answer = Terminal.Ask(context, "Variable (number, name or part of a name; ? lists all; Enter ends):", "");
            if (answer.Length == 0)
            {
                return null;
            }

            if (answer == "?")
            {
                var saved = SavedEnvironment.Read(context.Config);
                foreach (var group in EnvironmentVariables.All.Where(v => SavedEnvironment.WhyNot(v) is null).GroupBy(v => v.Group))
                {
                    context.Write($"{group.Key}:");
                    foreach (var variable in group)
                    {
                        context.Write($"  {variable.Name,-34} {Now(variable, saved),-16} {Cut(variable.Meaning, 60)}");
                    }
                }

                continue;
            }

            if (int.TryParse(answer, NumberStyles.Integer, CultureInfo.InvariantCulture, out int number))
            {
                if (number >= 1 && number <= Common.Length)
                {
                    return EnvironmentVariables.Find(Common[number - 1]);
                }

                context.Write($"Pick 1 to {Common.Length}, or type a name.");
                continue;
            }

            if ((EnvironmentVariables.Find(answer.ToUpperInvariant()) ?? EnvironmentVariables.Find(answer)) is { } exact)
            {
                if (SavedEnvironment.WhyNot(exact) is { } why)
                {
                    context.Write(why + ".");
                    continue;
                }

                return exact;
            }

            var matches = EnvironmentVariables.All.Where(v => SavedEnvironment.WhyNot(v) is null
                && v.Name.Contains(answer.Replace(' ', '_'), StringComparison.OrdinalIgnoreCase)).ToList();
            if (matches.Count == 1)
            {
                return matches[0];
            }

            context.Write(matches.Count == 0 ? $"No variable matches '{answer}' (? lists all)."
                : $"'{answer}' matches {string.Join(", ", matches.Select(m => m.Name))}; type more of the name.");
        }
    }

    // Asks for the value of one variable, with what it does, its default, its value now and a suggestion.
    private static string AskValue(CommandContext context, EnvironmentVariables.Variable variable)
    {
        var saved = SavedEnvironment.Read(context.Config);
        context.Write($"{variable.Name}: {variable.Meaning}");
        context.Write($"  default: {variable.Default}; now: {Now(variable, saved)}");
        string suggested = Suggest(variable, saved, context.Config);
        string question = suggested.Length > 0 ? "Value (Enter takes the suggestion, - removes it)" : "Value (- removes it, Enter keeps it as it is):";
        string answer = Terminal.Ask(context, question, suggested, requireTerminal: "a VALUE after the NAME");
        return answer.Length == 0 && suggested.Length == 0 ? saved.GetValueOrDefault(variable.Name) ?? "-" : answer;
    }

    // Saves (or, for "-" or an empty value, removes) one variable and says so.
    private static JsonObject Save(CommandContext context, EnvironmentVariables.Variable variable, string value)
    {
        bool remove = value is "-" or "";
        var saved = Saved(context, create: !remove);
        bool had = saved.ContainsKey(variable.Name);
        if (remove)
        {
            saved.Remove(variable.Name);
        }
        else
        {
            saved[variable.Name] = value;
        }

        if (!remove || had)
        {
            context.Config.SaveChanges();
        }

        string? user = User(context, variable.Name, remove ? null : value);
        context.Write(remove ? had ? $"Removed {variable.Name} ({Where(context)})" : $"{variable.Name} was not saved; nothing to remove"
            : $"Saved {variable.Name}={value} ({Where(context)})");
        if (Environment.GetEnvironmentVariable(variable.Name) is { Length: > 0 } shell && !SavedEnvironment.Applied.Contains(variable.Name) && !remove && shell != value)
        {
            context.Write($"  note: this terminal sets {variable.Name}={shell}, which wins here until it is unset or the terminal is closed");
        }

        if (user is not null)
        {
            context.Write($"  {user}");
        }

        return new JsonObject { ["name"] = variable.Name, ["value"] = remove ? null : value, ["removed"] = remove && had };
    }

    // What a variable is now: its saved value, the terminal's, or not set.
    private static string Now(EnvironmentVariables.Variable variable, IReadOnlyDictionary<string, string> saved) =>
        saved.TryGetValue(variable.Name, out string? value) ? $"{Cut(value, 30)} (saved)"
        : Environment.GetEnvironmentVariable(variable.Name) is { Length: > 0 } shell && !SavedEnvironment.Applied.Contains(variable.Name) ? $"{Cut(shell, 30)} (terminal)"
        : "not set";

    /// <summary>
    /// The value offered for <paramref name="variable"/>: its saved value, else one that turns on what the meaning
    /// describes ("1 or true: ..." gives 1, "0 or false: ..." gives 0, "ac or battery: ..." gives ac), else a concrete
    /// default, else nothing.
    /// </summary>
    internal static string Suggest(EnvironmentVariables.Variable variable, IReadOnlyDictionary<string, string> saved, CliConfig config)
    {
        if (saved.TryGetValue(variable.Name, out string? value))
        {
            return value;
        }

        switch (variable.Name)
        {
            case "IDRAK_DEVICES":
                return config.Get("device") ?? "cpu";
            case "IDRAK_FILTER":
                return "";
            case "IDRAK_CACHE":
                return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "idrak");
            case "IDRAK_LANG":
                return "ar";
            case "IDRAK_PORT":
                return "7317";
        }

        string meaning = variable.Meaning;
        int colon = meaning.IndexOf(':');
        if (colon > 0 && colon < 30)
        {
            string first = meaning[..colon].Split(' ', StringSplitOptions.RemoveEmptyEntries)[0].Trim(',');
            if (first.Length is > 0 and < 12 && (char.IsDigit(first[0]) || char.IsLower(first[0])) && !first.Contains('/'))
            {
                return first;
            }
        }

        return variable.Default is { Length: > 0 } fallback && !fallback.Contains(' ') && fallback is not ("on" or "off") ? fallback : "";
    }

    private static string Cut(string text, int width) => text.Length > width ? text[..(width - 3)] + "..." : text;
}

/// <summary><c>idrak env unset NAME...</c>: removes saved variables.</summary>
internal sealed class EnvUnsetCommand : EnvSaveCommand
{
    public override string Name => "env unset";

    public override string Summary => "Remove environment variables saved by idrak env set";

    public override string Usage => """
        NAME... [--user] [--profile NAME]

        Options:
              --user          also remove them from the Windows user environment
              --profile NAME  remove them from the named profile

        Examples:
          idrak env unset IDRAK_CUDA_DEBUG
          idrak env unset IDRAK_DEVICES IDRAK_FILTER --user
        """;

    public override int Run(CommandContext context)
    {
        if (context.Positional.Count == 0)
        {
            throw new UsageException("env unset needs the NAME of a saved variable (idrak env lists them).");
        }

        var saved = Saved(context, create: false);
        var removed = new JsonArray();
        foreach (string name in context.Positional)
        {
            var variable = EnvironmentVariables.Find(name.ToUpperInvariant()) ?? EnvironmentVariables.Find(name)
                ?? throw new UsageException($"Not a variable Idrak reads: {name} (idrak help env lists them).");
            bool had = saved.Remove(variable.Name);
            context.Write(had ? $"Removed {variable.Name} ({Where(context)})" : $"{variable.Name} was not saved");
            if (User(context, variable.Name, null) is { } user)
            {
                context.Write($"  {user}");
            }

            if (had)
            {
                removed.Add(variable.Name);
            }
        }

        if (removed.Count > 0)
        {
            context.Config.SaveChanges();
        }

        context.WriteJson(new JsonObject { ["file"] = context.Config.Path, ["removed"] = removed });
        return ExitCodes.Ok;
    }
}

/// <summary>
/// <c>idrak env reset</c>: removes every saved variable (top level, a profile's, or with <c>--all-profiles</c> all of
/// them), so every variable is back to its default or the terminal's value. Asks first (<c>--yes</c>, <c>--dry-run</c>).
/// </summary>
internal sealed class EnvResetCommand : EnvSaveCommand
{
    public override string Name => "env reset";

    public override string Summary => "Remove every saved environment variable: all back to their defaults (asks first)";

    public override string Usage => """
        [--all-profiles] [--profile NAME] [--user] [--yes] [--dry-run]

        Removes the variables idrak env set saved, so each one is back to its default (or to the terminal's value).
        Lists them and asks first.

        Options:
              --profile NAME  only the named profile's saved variables
              --all-profiles  the top-level ones and every profile's
              --user          also remove them from the Windows user environment
          -y, --yes           no question
              --dry-run       list what would be removed, change nothing

        Examples:
          idrak env reset
          idrak env reset -y --user
          idrak env reset --all-profiles --dry-run
        """;

    public override IReadOnlyCollection<string> Flags => ["--user", "--all-profiles", .. Terminal.ConfirmFlags];

    public override IReadOnlyDictionary<string, string> ShortForms => Terminal.ConfirmShortForms;

    public override int Run(CommandContext context)
    {
        if (context.Positional.Count > 0)
        {
            throw new UsageException($"env reset takes no NAME (idrak env unset {context.Positional[0]} removes one).");
        }

        if (context.Flag("--all-profiles") && context.Option("--profile") is not null)
        {
            throw new UsageException("Give --profile NAME or --all-profiles, not both.");
        }

        // Each place variables are saved in: (label, the object holding "env").
        var places = new List<(string Label, JsonObject Holder)>();
        var root = context.Config.Root;
        if (context.Option("--profile") is { } profile)
        {
            if (root["profiles"]?[profile] is JsonObject named)
            {
                places.Add(($"profile {profile}", named));
            }
        }
        else
        {
            places.Add(("top level", root));
            if (context.Flag("--all-profiles") && root["profiles"] is JsonObject profiles)
            {
                places.AddRange(profiles.Where(p => p.Value is JsonObject).Select(p => ($"profile {p.Key}", (JsonObject)p.Value!)));
            }
        }

        var found = places.Select(p => (p.Label, p.Holder, Names: p.Holder[SavedEnvironment.Key] is JsonObject saved ? saved.Select(v => v.Key).ToList() : []))
            .Where(p => p.Names.Count > 0).ToList();
        var removed = new JsonArray();
        if (found.Count == 0)
        {
            context.Write($"No saved variables to remove ({Where(context)}).");
        }
        else
        {
            foreach (var (label, _, names) in found)
            {
                context.Write($"{label}: {string.Join(", ", names)}");
            }

            int count = found.Sum(p => p.Names.Count);
            if (Terminal.DryRun(context))
            {
                context.Write($"Would remove {count} saved variable{(count == 1 ? "" : "s")} (--dry-run: nothing changed).");
            }
            else if (!Terminal.Confirm(context, $"Remove {count} saved variable{(count == 1 ? "" : "s")}?"))
            {
                context.Write("Nothing changed.");
            }
            else
            {
                foreach (var (label, holder, names) in found)
                {
                    holder.Remove(SavedEnvironment.Key);
                    foreach (string name in names)
                    {
                        removed.Add(new JsonObject { ["name"] = name, ["place"] = label });
                    }
                }

                context.Config.SaveChanges();
                context.Write($"Removed {count} saved variable{(count == 1 ? "" : "s")} ({context.Config.Path}); each is back to its default.");
                foreach (string name in found.SelectMany(p => p.Names).Distinct())
                {
                    if (User(context, name, null) is { } user)
                    {
                        context.Write($"  {name}: {user}");
                    }
                }
            }
        }

        context.WriteJson(new JsonObject { ["file"] = context.Config.Path, ["dryRun"] = Terminal.DryRun(context), ["removed"] = removed });
        return ExitCodes.Ok;
    }
}
