// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Diagnostics;

namespace Idrak.Cli.Commands.Health;

/// <summary>
/// <c>idrak init</c>: writes the config file by asking for the default device, the cache folder and model aliases, or
/// with <c>--yes</c> from what <c>doctor</c> finds (the default device, the cache in use). Other keys are kept.
/// </summary>
internal sealed class InitCommand : Command
{
    public override string Name => "init";

    public override string Summary => "Writes the config file: default device, cache folder, model aliases";

    public override string Usage => """
        [--yes] [--dry-run] [--profile NAME]

        Options:
          -y, --yes           no questions: the default device and the cache folder doctor finds
              --dry-run       print what would be written, write nothing
              --profile NAME  write the values into a named profile (e.g. phone) instead of the top level

        Keys already in the file are kept; the ones asked for are replaced. The file is --config, IDRAK_CONFIG or
        ~/.idrak/config.json.

        Examples:
          idrak init
          idrak init -y
          idrak init --yes --profile phone -d vulkan:0
        """;

    public override IReadOnlyCollection<string> ValueOptions => ["--profile"];

    public override IReadOnlyCollection<string> Flags => Terminal.ConfirmFlags;

    public override IReadOnlyDictionary<string, string> ShortForms => Terminal.ConfirmShortForms;

    public override int Run(CommandContext context)
    {
        string device = Suggested(context), cache = Path.GetFullPath(context.CacheFolder);
        var aliases = new JsonObject();
        if (!context.Flag("--yes"))
        {
            var names = DeviceListing.All().Where(d => d.Error is null).Select(d => d.Device).ToList();
            context.ConsoleOutput.WriteLine($"Devices: {string.Join(", ", names)}");
            device = Terminal.Ask(context, "Default device?", device);
            Device.Parse(device);                                                // a usage mistake shows now, not at the next command
            cache = Terminal.Ask(context, "Cache folder?", cache);
            while (Terminal.Ask(context, "Model alias as NAME=MODEL (empty to finish)?", "") is { Length: > 0 } alias)
            {
                int equals = alias.IndexOf('=');
                if (equals <= 0 || equals == alias.Length - 1)
                {
                    context.ConsoleOutput.WriteLine("  write NAME=MODEL, e.g. qwen=Qwen/Qwen3-0.6B");
                    continue;
                }

                aliases[alias[..equals].Trim()] = new JsonObject { ["model"] = alias[(equals + 1)..].Trim() };
            }
        }

        var values = new JsonObject { ["device"] = device, ["cache"] = cache };
        if (aliases.Count > 0)
        {
            values["aliases"] = aliases;
        }

        string? profile = context.Option("--profile");
        string where = profile is null ? context.Config.Path : $"profile {profile} in {context.Config.Path}";
        var json = new JsonObject { ["file"] = context.Config.Path, ["profile"] = profile, ["values"] = values.DeepClone(), ["written"] = false };
        if (Terminal.DryRun(context))
        {
            context.Write($"Would write to {where}:");
            context.Write(values.ToJsonString(CommandContext.JsonOutput));
            context.WriteJson(json);
            return ExitCodes.Ok;
        }

        var root = context.Config.Root;
        JsonObject target = root;
        if (profile is not null)
        {
            var profiles = root["profiles"] as JsonObject ?? [];
            root["profiles"] = profiles;
            target = profiles[profile] as JsonObject ?? [];
            profiles[profile] = target;
        }

        foreach (var (key, value) in values)
        {
            if (key == "aliases" && target["aliases"] is JsonObject existing)
            {
                foreach (var (name, alias) in (JsonObject)value!)
                {
                    existing[name] = alias!.DeepClone();
                }
            }
            else
            {
                target[key] = value!.DeepClone();
            }
        }

        context.Config.SaveChanges();
        context.Write($"Wrote {where}: device {device}, cache {cache}{(aliases.Count > 0 ? $", aliases {string.Join(", ", aliases.Select(a => a.Key))}" : "")}.");
        json["written"] = true;
        context.WriteJson(json);
        return ExitCodes.Ok;
    }

    // The device doctor would suggest: --device when given, else the default device when it starts.
    private static string Suggested(CommandContext context)
    {
        var device = context.Device;
        return DeviceListing.Describe(device).Error is null ? device.ToString() : "cpu";
    }
}
