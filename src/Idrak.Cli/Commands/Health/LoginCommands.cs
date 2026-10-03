// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands.Health;

/// <summary>
/// <c>idrak login hf|github|kaggle</c>: stores a token where the libraries already look for it (never printed): the
/// Hugging Face token file (<c>HF_TOKEN_PATH</c>, or <c>HF_HOME</c>/token), Kaggle's kaggle.json
/// (<c>KAGGLE_CONFIG_DIR</c>, or ~/.kaggle), and for GitHub, which the library reads from <c>GITHUB_TOKEN</c> only, the
/// config's "tokens.github" (which the tool passes on as <c>GITHUB_TOKEN</c>).
/// </summary>
internal sealed class LoginCommand : Command
{
    internal static readonly string[] Services = ["hf", "github", "kaggle"];

    public override string Name => "login";

    public override string Summary => "Stores a token for the Hugging Face hub, GitHub or Kaggle where the library reads it";

    public override string Usage => """
        hf|github|kaggle

        The token is read from piped input or typed without echo, and never printed:
          hf       written to the Hugging Face token file (HF_TOKEN_PATH, or HF_HOME/token: ~/.cache/huggingface/token)
          github   kept in the config as tokens.github; idrak passes it on as GITHUB_TOKEN (the library reads only that)
          kaggle   asks for the user name too; written to kaggle.json (KAGGLE_CONFIG_DIR, or ~/.kaggle)
        A token in the environment (HF_TOKEN, GITHUB_TOKEN, KAGGLE_KEY) still wins over the stored one.

        Examples:
          idrak login hf
          echo "$TOKEN" | idrak login github
          idrak logout hf
        """;

    public override int Run(CommandContext context)
    {
        string service = Service(context);
        string path;
        switch (service)
        {
            case "hf":
                path = HuggingFaceTokenFile();
                Write(path, Terminal.AskSecret(context, "Hugging Face token"));
                break;
            case "github":
                string token = Terminal.AskSecret(context, "GitHub token");
                var tokens = context.Config.Root["tokens"] as JsonObject ?? [];
                tokens["github"] = token;
                context.Config.Root["tokens"] = tokens;
                context.Config.SaveChanges();
                path = context.Config.Path;
                break;
            default:
                string user = Terminal.IsInteractive(context) || Terminal.TestInput is not null ? Terminal.Ask(context, "Kaggle user name?", "", requireTerminal: "KAGGLE_USERNAME and KAGGLE_KEY in the environment") : "";
                if (user.Length == 0)
                {
                    throw new UsageException("Kaggle needs a user name; or set KAGGLE_USERNAME and KAGGLE_KEY in the environment.");
                }

                path = KaggleFile();
                Write(path, new JsonObject { ["username"] = user, ["key"] = Terminal.AskSecret(context, "Kaggle key") }.ToJsonString());
                break;
        }

        context.Write($"Stored the {service} token in {path}.");
        context.WriteJson(new JsonObject { ["service"] = service, ["file"] = path, ["stored"] = true });
        return ExitCodes.Ok;
    }

    internal static string Service(CommandContext context)
    {
        string service = context.Argument(0, "the service: hf, github or kaggle");
        return Services.Contains(service) ? service : throw new UsageException($"{context.Command.Name} takes hf, github or kaggle, not '{service}'.");
    }

    // Where the Hugging Face token is read from (Idrak.Datasets.HuggingFace.Token).
    internal static string HuggingFaceTokenFile() => Environment.GetEnvironmentVariable("HF_TOKEN_PATH")
        ?? Path.Combine(Environment.GetEnvironmentVariable("HF_HOME") ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface"), "token");

    // Where Kaggle's credentials are read from.
    internal static string KaggleFile() => Path.Combine(Environment.GetEnvironmentVariable("KAGGLE_CONFIG_DIR")
        ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".kaggle"), "kaggle.json");

    // Writes a credentials file readable by the user only (where the system has such permissions).
    private static void Write(string path, string text)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        File.WriteAllText(path, text);
        if (!OperatingSystem.IsWindows())
        {
            File.SetUnixFileMode(path, UnixFileMode.UserRead | UnixFileMode.UserWrite);
        }
    }
}

/// <summary><c>idrak logout hf|github|kaggle</c>: removes the token <c>login</c> stored.</summary>
internal sealed class LogoutCommand : Command
{
    public override string Name => "logout";

    public override string Summary => "Removes a stored token (Hugging Face hub, GitHub or Kaggle)";

    public override string Usage => """
        hf|github|kaggle [--yes] [--dry-run]

          -y, --yes   no question
          --dry-run   say what would be removed

        Examples:
          idrak logout hf
          idrak logout kaggle -y
        """;

    public override IReadOnlyCollection<string> Flags => Terminal.ConfirmFlags;

    public override IReadOnlyDictionary<string, string> ShortForms => Terminal.ConfirmShortForms;

    public override int Run(CommandContext context)
    {
        string service = LoginCommand.Service(context);
        string where = service switch { "hf" => LoginCommand.HuggingFaceTokenFile(), "kaggle" => LoginCommand.KaggleFile(), _ => $"{context.Config.Path} (tokens.github)" };
        bool stored = service == "github" ? context.Config.Root["tokens"]?["github"] is not null : File.Exists(where);
        bool removed = false;
        if (!stored)
        {
            context.Write($"No {service} token is stored ({where}).");
        }
        else if (Terminal.DryRun(context))
        {
            context.Write($"Would remove the {service} token from {where}.");
        }
        else if (Terminal.Confirm(context, $"Remove the {service} token from {where}?"))
        {
            if (service == "github")
            {
                (context.Config.Root["tokens"] as JsonObject)?.Remove("github");
                context.Config.SaveChanges();
            }
            else
            {
                File.Delete(where);
            }

            removed = true;
            context.Write($"Removed the {service} token.");
        }

        context.WriteJson(new JsonObject { ["service"] = service, ["file"] = where, ["removed"] = removed });
        return ExitCodes.Ok;
    }
}
