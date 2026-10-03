// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Cli.Commands.Measure;

/// <summary>
/// <c>idrak tuning show</c>: the kernel choices the backends measured on this machine and keep in their tuning caches
/// (the CPU's parallel cut-overs, the CUDA and Vulkan kernel and runtime choices), per device; <c>--reset</c> deletes
/// them so they are measured again on next use.
/// </summary>
internal sealed class TuningShowCommand : Command
{
    private static readonly string[] Backends = ["cpu", "cuda", "vulkan"];

    public override string Name => "tuning show";

    public override string Summary => "The kernel choices measured on this machine (the tuning caches); --reset clears them";

    public override string Usage =>
        "[options]\n\n" +
        "Reads the tuning caches the backends write as they measure their choices: the CPU's parallel cut-overs\n" +
        "(cpu/tuning.tsv), the CUDA kernel choices (tuning/cuda/*.txt, one file per GPU and driver) and the Vulkan\n" +
        "runtime and kernel choices (vulkan/tuning.tsv), all under the cache folder unless the variables below move them.\n" +
        "Each choice is shown with the device it was measured on.\n\n" +
        "Options:\n" +
        "      --backend NAME  only cpu, cuda or vulkan\n" +
        "      --reset         delete the caches (or the --backend one): choices are measured again on next use\n\n" +
        "Examples:\n" +
        "  idrak tuning show\n" +
        "  idrak tuning show --backend vulkan -j\n" +
        "  idrak tuning show --reset --backend cuda";

    public override IReadOnlyCollection<string> ValueOptions => ["--backend"];

    public override IReadOnlyCollection<string> Flags => ["--reset"];

    public override int Run(CommandContext context)
    {
        if (context.Positional.Count > 0)
        {
            throw new UsageException($"tuning show takes no arguments, not '{context.Positional[0]}'.");
        }

        string? only = context.Option("--backend")?.ToLowerInvariant();
        if (only is not null && !Backends.Contains(only))
        {
            throw new UsageException($"--backend takes cpu, cuda or vulkan, not '{only}'.");
        }

        var caches = Caches(context.CacheFolder).Where(c => only is null || c.Backend == only).ToList();
        var json = new JsonArray();
        if (context.Flag("--reset"))
        {
            foreach (var cache in caches)
            {
                var deleted = new List<string>();
                foreach (string file in cache.Files)
                {
                    File.Delete(file);
                    deleted.Add(file);
                }

                context.Write(cache.Path is null ? $"{cache.Backend}: no cache (turned off)"
                    : deleted.Count == 0 ? $"{cache.Backend}: nothing kept ({cache.Path})"
                    : $"{cache.Backend}: deleted {string.Join(", ", deleted)}");
                json.Add(new JsonObject
                {
                    ["backend"] = cache.Backend,
                    ["path"] = cache.Path,
                    ["deleted"] = new JsonArray([.. deleted.Select(d => (JsonNode)d)]),
                });
            }

            context.Write("Choices are measured again on next use.");
            context.WriteJson(new JsonObject { ["reset"] = true, ["caches"] = json });
            return ExitCodes.Ok;
        }

        foreach (var cache in caches)
        {
            var entries = cache.Files.SelectMany(f => cache.Read(f)).ToList();
            context.Write($"{cache.Backend}: {(cache.Path is null ? "cache turned off" : cache.Path)} ({entries.Count} choice{(entries.Count == 1 ? "" : "s")})");
            foreach (var device in entries.GroupBy(e => e.Device))
            {
                context.Write($"  {device.Key}");
                context.Table(["choice", "value"], device.Select(e => (IReadOnlyList<string>)[e.Choice, e.Value]));
            }

            json.Add(new JsonObject
            {
                ["backend"] = cache.Backend,
                ["path"] = cache.Path,
                ["files"] = new JsonArray([.. cache.Files.Select(f => (JsonNode)f)]),
                ["choices"] = new JsonArray([.. entries.Select(e => (JsonNode)new JsonObject
                {
                    ["device"] = e.Device,
                    ["choice"] = e.Choice,
                    ["value"] = e.Value,
                })]),
            });
        }

        context.WriteJson(new JsonObject { ["caches"] = json });
        return ExitCodes.Ok;
    }

    /// <summary>One kept choice: the device it was measured on, what was chosen and the value.</summary>
    internal sealed record Entry(string Device, string Choice, string Value);

    /// <summary>A backend's tuning cache: its path (null when turned off), its files and how to read one.</summary>
    internal sealed record Cache(string Backend, string? Path, IReadOnlyList<string> Files, Func<string, IEnumerable<Entry>> Read);

    /// <summary>The caches, where the backends look for them (the same variables and defaults, under <paramref name="cacheFolder"/>).</summary>
    internal static IEnumerable<Cache> Caches(string cacheFolder)
    {
        static bool Off(string? setting) => setting is "0" || string.Equals(setting, "false", StringComparison.OrdinalIgnoreCase)
            || string.Equals(setting, "off", StringComparison.OrdinalIgnoreCase);
        string? tuning = Environment.GetEnvironmentVariable("IDRAK_TUNING_CACHE");

        string? cpu = Off(tuning) ? null
            : Environment.GetEnvironmentVariable("IDRAK_CPU_TUNING_FILE") is { } cpuFile ? (cpuFile.Length > 0 ? cpuFile : null)
            : System.IO.Path.Combine(cacheFolder, "cpu", "tuning.tsv");
        yield return new Cache("cpu", cpu, cpu is not null && File.Exists(cpu) ? [cpu] : [], ReadCpu);

        string? cuda = Off(tuning) ? null
            : tuning is { Length: > 0 } && tuning is not "1" && !string.Equals(tuning, "true", StringComparison.OrdinalIgnoreCase) ? tuning
            : System.IO.Path.Combine(cacheFolder, "tuning", "cuda");
        yield return new Cache("cuda", cuda, cuda is not null && Directory.Exists(cuda) ? [.. Directory.GetFiles(cuda, "*.txt").Order(StringComparer.Ordinal)] : [], ReadCuda);

        string? vulkanSetting = Environment.GetEnvironmentVariable("IDRAK_VULKAN_TUNING_CACHE");
        string? vulkan = vulkanSetting is "0" or "false" ? null
            : !string.IsNullOrEmpty(vulkanSetting) ? vulkanSetting
            : System.IO.Path.Combine(cacheFolder, "vulkan", "tuning.tsv");
        yield return new Cache("vulkan", vulkan, vulkan is not null && File.Exists(vulkan) ? [vulkan] : [], ReadVulkan);
    }

    // "v2|processor|features|cores|L1 …|…|.NET …|N threads<TAB>elements<TAB>flops": the two parallel cut-overs per machine key.
    private static IEnumerable<Entry> ReadCpu(string path)
    {
        foreach (string line in File.ReadLines(path))
        {
            var fields = line.Split('\t');
            if (fields.Length != 3)
            {
                continue;
            }

            var key = fields[0].Split('|');
            string device = key.Length > 1 ? string.Join(", ", key.Skip(1).Where(p => p.Length > 0)) : fields[0];
            yield return new Entry(device, "parallel cut-over, element-wise (elements)", fields[1]);
            yield return new Entry(device, "parallel cut-over, products (m x n x k)", fields[2]);
        }
    }

    // A header (format, "device: …", "driver: …", "library: …"), then "Op variant a b c d e f = value" per choice.
    private static IEnumerable<Entry> ReadCuda(string path)
    {
        var lines = File.ReadAllLines(path);
        string Header(string name) => lines.FirstOrDefault(l => l.StartsWith(name + ": ", StringComparison.Ordinal))?[(name.Length + 2)..] ?? "?";
        string device = $"{Header("device")} (driver {Header("driver")}; {lines.FirstOrDefault() ?? "?"})";
        foreach (string line in lines.Skip(4))
        {
            int equals = line.LastIndexOf(" = ", StringComparison.Ordinal);
            if (equals > 0)
            {
                yield return new Entry(device, line[..equals].Trim(), line[(equals + 3)..].Trim());
            }
        }
    }

    // "key<TAB>value<TAB>device name": runtime choices keyed "device/driver/version/[power/]name", kernel choices
    // "kernels/device/driver/version/[power/]width/reductions/[matrix shape/]version/op variant a b c d e f".
    private static IEnumerable<Entry> ReadVulkan(string path)
    {
        foreach (string line in File.ReadLines(path))
        {
            var fields = line.Split('\t');
            if (fields.Length < 2)
            {
                continue;
            }

            string device = fields.Length > 2 && fields[2].Length > 0 ? fields[2] : "?";
            var parts = fields[0].Split('/');
            bool kernel = parts[0] == "kernels";
            // The device, driver and version, then the power source ("ac" or "battery") in files written since it was added.
            int skip = kernel ? 4 : 3;
            string power = parts.Length > skip && parts[skip] is "ac" or "battery" ? parts[skip++] : "";
            string choice = parts.Length > skip ? string.Join('/', parts.Skip(skip)) : fields[0];
            yield return new Entry(power.Length > 0 ? $"{device} (power: {power})" : device, (kernel ? "kernel " : "runtime ") + choice, fields[1]);
        }
    }
}
