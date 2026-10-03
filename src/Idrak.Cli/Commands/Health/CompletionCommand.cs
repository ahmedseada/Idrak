// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Diagnostics;

namespace Idrak.Cli.Commands.Health;

/// <summary>
/// <c>idrak completion bash|zsh|fish|pwsh</c>: prints a shell completion script. The scripts ask the tool itself
/// (<c>idrak completion --complete -- WORDS...</c>) for the candidates, so commands, options, device names and cached
/// model names stay current, plug-in commands included.
/// </summary>
internal sealed class CompletionCommand : Command
{
    private static readonly string[] Shells = ["bash", "zsh", "fish", "pwsh"];

    public override string Name => "completion";

    public override string Summary => "Prints a shell completion script (bash, zsh, fish, pwsh)";

    public override string Usage => """
        bash|zsh|fish|pwsh

        Options:
              --complete -- WORDS...  the candidates for the last word (what the scripts call)

        Install:
          bash   echo 'eval "$(idrak completion bash)"' >> ~/.bashrc
          zsh    echo 'eval "$(idrak completion zsh)"' >> ~/.zshrc
          fish   idrak completion fish > ~/.config/fish/completions/idrak.fish
          pwsh   idrak completion pwsh >> $PROFILE

        Examples:
          idrak completion bash
          idrak completion --complete -- cache c
        """;

    public override IReadOnlyCollection<string> Flags => ["--complete"];

    public override int Run(CommandContext context)
    {
        if (context.Flag("--complete"))
        {
            foreach (string candidate in Candidates(context, context.Positional))
            {
                context.Output.WriteLine(candidate);
            }

            return ExitCodes.Ok;
        }

        string shell = context.Argument(0, "the shell: bash, zsh, fish or pwsh");
        string script = shell switch
        {
            "bash" => """
                # idrak completion for bash: eval "$(idrak completion bash)"
                _idrak() {
                    local IFS=$'\n'
                    COMPREPLY=($(idrak completion --complete -- "${COMP_WORDS[@]:1:COMP_CWORD}" 2>/dev/null))
                }
                complete -o default -F _idrak idrak
                """,
            "zsh" => """
                # idrak completion for zsh: eval "$(idrak completion zsh)"
                _idrak() {
                    local -a candidates
                    candidates=("${(@f)$(idrak completion --complete -- "${(@)words[2,CURRENT]}" 2>/dev/null)}")
                    compadd -a candidates
                }
                compdef _idrak idrak
                """,
            "fish" => """
                # idrak completion for fish: idrak completion fish > ~/.config/fish/completions/idrak.fish
                complete -c idrak -f -a '(idrak completion --complete -- (commandline -opc)[2..-1] (commandline -ct) 2>/dev/null)'
                """,
            "pwsh" => """
                # idrak completion for PowerShell: idrak completion pwsh >> $PROFILE
                Register-ArgumentCompleter -Native -CommandName idrak -ScriptBlock {
                    param($wordToComplete, $commandAst, $cursorPosition)
                    $words = @($commandAst.CommandElements | Select-Object -Skip 1 | ForEach-Object { $_.ToString() })
                    if ($wordToComplete -eq '') { $words += '' }
                    idrak completion --complete -- @words 2>$null | ForEach-Object {
                        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
                    }
                }
                """,
            _ => throw new UsageException($"completion takes one of {string.Join(", ", Shells)}, not '{shell}'."),
        };
        if (context.Json)
        {
            context.WriteJson(new JsonObject { ["shell"] = shell, ["script"] = script });
        }
        else
        {
            context.Output.WriteLine(script);
        }

        return ExitCodes.Ok;
    }

    /// <summary>The candidates for the last of <paramref name="words"/> (the words after <c>idrak</c>, the last one partly typed).</summary>
    internal static IEnumerable<string> Candidates(CommandContext context, IReadOnlyList<string> words)
    {
        string current = words.Count > 0 ? words[^1] : "";
        var before = words.Take(Math.Max(0, words.Count - 1)).ToList();
        var commands = CommandTable.All;
        var leading = before.TakeWhile(w => !w.StartsWith('-')).ToList();
        var command = CommandLine.Find(commands, leading, out _);
        IEnumerable<string> all;
        if (command is null)
        {
            // Still naming the command: the next word of every name and alias that starts with the words so far.
            all = commands.SelectMany(c => c.Aliases.Prepend(c.Name)).Select(n => n.Split(' '))
                .Where(n => n.Length > leading.Count && n.Take(leading.Count).SequenceEqual(leading))
                .Select(n => n[leading.Count]);
            if (leading.Count == 0)
            {
                all = all.Append("help");
            }
            else if (leading is ["help"])
            {
                all = commands.Select(c => c.Name.Split(' ')[0]).Concat(HelpTopics.Names);
            }
        }
        else if (current.StartsWith('-'))
        {
            all = CommandContext.CommonValueOptions.Concat(CommandContext.CommonFlags).Concat(command.ValueOptions).Concat(command.Flags)
                .Concat(CommandContext.CommonShortForms.Keys).Concat(command.ShortForms.Keys);
        }
        else if (before.Count > 0 && Expand(command, before[^1]) is { } option && IsValueOption(command, option))
        {
            all = option switch
            {
                "--device" => DeviceListing.Backends.SelectMany(b => Enumerable.Range(0, b.Count).Select(i => $"{b.Kind}:{i}")).Prepend("cpu"),
                "--format" => ["text", "json", "csv", "md"],
                "--color" => ["auto", "always", "never"],
                "--weights" => Registries.Snapshot().First(c => c.Key == "weights").Names,
                "--kv" => Registries.Snapshot().First(c => c.Key == "kv").Names,
                "--profile" => context.Config.Root["profiles"] is JsonObject p ? p.Select(x => x.Key) : [],
                _ => [],
            };
        }
        else
        {
            all = command.Name switch
            {
                "completion" => Shells,
                "cache clear" => ["models", "tuning", "kernels", "all"],
                "login" or "logout" => LoginCommand.Services,
                "env" => EnvironmentVariables.All.Select(v => v.Name),
                _ => command.Usage.Contains("MODEL", StringComparison.Ordinal) ? ModelNames(context) : [],
            };
        }

        return all.Distinct().Where(c => c.StartsWith(current, StringComparison.Ordinal)).Order(StringComparer.Ordinal);
    }

    private static string? Expand(Command command, string word) =>
        word.StartsWith("--", StringComparison.Ordinal) ? word
        : CommandContext.CommonShortForms.TryGetValue(word, out string? common) ? common
        : command.ShortForms.TryGetValue(word, out string? own) ? own : null;

    private static bool IsValueOption(Command command, string option) => CommandContext.CommonValueOptions.Contains(option) || command.ValueOptions.Contains(option);

    /// <summary>Aliases from the config and the models in the cache (owner/name).</summary>
    internal static IEnumerable<string> ModelNames(CommandContext context)
    {
        var names = new List<string>();
        if (context.Config.Object("aliases") is { } aliases)
        {
            names.AddRange(aliases.Select(a => a.Key));
        }

        string models = Path.Combine(context.CacheFolder, "downloads", "huggingface", "models");
        if (Directory.Exists(models))
        {
            foreach (string owner in Directory.EnumerateDirectories(models))
            {
                names.AddRange(Directory.EnumerateDirectories(owner).Select(repo => $"{Path.GetFileName(owner)}/{Path.GetFileName(repo)}"));
            }
        }

        return names;
    }
}
