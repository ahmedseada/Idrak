// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands.Health;

/// <summary>
/// <c>idrak setup android</c>: the steps of installation/android-termux.md that are safe to automate (the .NET heap
/// limit and the Vulkan driver file in the shell's start-up file), printed, or done with <c>--yes</c>; the steps that
/// are not (Termux, Ubuntu, the SDK, building Turnip) are listed; then the phone checks of <c>doctor --android</c>.
/// </summary>
internal sealed class SetupAndroidCommand : Command
{
    /// <summary>The heap limit line (4 GiB), as the guide gives it.</summary>
    internal const string HeapLine = "export DOTNET_GCHeapHardLimit=0x100000000";

    public override string Name => "setup android";

    public override string Summary => "The Android (Termux) setup steps that are safe to automate, then the phone checks";

    public override string Usage => """
        [--yes] [--dry-run] [--rc FILE]

          -y, --yes   add the lines to the shell start-up file (each only once)
          --dry-run   print the lines, change nothing (the default without --yes)
          --rc FILE   the start-up file (default ~/.bashrc)

        Run inside the Ubuntu of proot-distro on the phone (installation/android-termux.md). After --yes, run
        source ~/.bashrc so the current shell has the variables.

        Examples:
          idrak setup android
          idrak setup android -y
        """;

    public override IReadOnlyCollection<string> ValueOptions => ["--rc"];

    public override IReadOnlyCollection<string> Flags => Terminal.ConfirmFlags;

    public override IReadOnlyDictionary<string, string> ShortForms => Terminal.ConfirmShortForms;

    /// <summary>~/.bashrc.</summary>
    internal static string DefaultRc => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".bashrc");

    public override int Run(CommandContext context)
    {
        string rc = context.Option("--rc") ?? DefaultRc;
        bool apply = context.Flag("--yes") && !Terminal.DryRun(context);
        var lines = new List<string> { HeapLine };
        string? turnip = FindTurnip();
        if (turnip is not null)
        {
            lines.Add($"export VK_ICD_FILENAMES={turnip}");
        }

        context.Write("Not automated (once, by hand; installation/android-termux.md):");
        context.Write("  1-2. Termux from F-Droid; pkg install proot-distro; proot-distro install ubuntu; proot-distro login ubuntu");
        context.Write("  3-4. apt update && apt install -y tzdata git curl ca-certificates libicu-dev dotnet-sdk-10.0");
        if (turnip is null)
        {
            context.Write("  7.   build Mesa Turnip with -Dfreedreno-kmds=kgsl (no Turnip driver file was found; VK_ICD_FILENAMES follows once it is)");
        }

        context.Write("");
        context.Write(apply ? $"Adding to {rc}:" : $"Lines for {rc} (idrak setup android --yes adds them):");
        var added = new JsonArray();
        foreach (string line in lines)
        {
            if (apply)
            {
                string done = AddLine(rc, line);
                added.Add(done);
                context.Write($"  {done}");
            }
            else
            {
                context.Write($"  {line}{(HasLine(rc, line) ? "   (already there)" : "")}");
            }
        }

        var checks = new List<DoctorCommand.Check>();
        DoctorCommand.Android(checks, rc);
        context.Write("");
        context.Write("Phone checks (this shell; after --yes, source the file first):");
        foreach (var c in checks)
        {
            context.Write($"  {c.Status.ToString().ToLowerInvariant(),-4}  {c.Name}: {c.Detail}");
        }

        context.WriteJson(new JsonObject
        {
            ["rc"] = rc,
            ["lines"] = new JsonArray([.. lines.Select(l => (JsonNode)l)]),
            ["applied"] = apply,
            ["changes"] = added,
            ["turnip"] = turnip,
            ["checks"] = new JsonArray([.. checks.Select(c => (JsonNode)new JsonObject { ["name"] = c.Name, ["status"] = c.Status.ToString().ToLowerInvariant(), ["detail"] = c.Detail })]),
        });
        return ExitCodes.Ok;
    }

    /// <summary>The Turnip driver file where the guide builds it (or a system Vulkan folder), or null.</summary>
    internal static string? FindTurnip()
    {
        string[] folders = ["/opt/turnip/share/vulkan/icd.d", "/usr/local/share/vulkan/icd.d", "/usr/share/vulkan/icd.d"];
        return folders.Where(Directory.Exists).SelectMany(f => Directory.EnumerateFiles(f, "freedreno_icd*.json")).Order(StringComparer.Ordinal).FirstOrDefault();
    }

    private static bool HasLine(string rc, string line) => File.Exists(rc) && File.ReadLines(rc).Any(l => l.Trim() == line);

    /// <summary>Appends <paramref name="line"/> to <paramref name="rc"/> unless it is there; returns what was done.</summary>
    internal static string AddLine(string rc, string line)
    {
        if (HasLine(rc, line))
        {
            return $"{line} (already in {rc})";
        }

        string existing = File.Exists(rc) ? File.ReadAllText(rc) : "";
        File.AppendAllText(rc, (existing.Length > 0 && !existing.EndsWith('\n') ? "\n" : "") + line + "\n");
        return $"{line} (added to {rc})";
    }
}
