// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Security.Cryptography;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Datasets;
using Idrak.LanguageModels;

namespace Idrak.Cli.Commands;

/// <summary>
/// <c>idrak verify MODEL</c>: checks a model's files: the JSON files parse, the weights' headers cover their files, a
/// sharded checkpoint's shards exist, no download was left partial; <c>--read</c> reads every tensor (all values finite),
/// <c>--hub</c> compares sizes and SHA-256 hashes with the hub's.
/// </summary>
internal sealed class VerifyCommand : Command
{
    public override string Name => "verify";

    public override string Summary => "Check a cached model's files: sizes, hashes where the hub gives them, readable tensors";

    public override string Usage => """
        MODEL [--read] [--hub] [--token TOKEN]

          MODEL              a cached Hugging Face id, a folder, a .gguf file or an alias
              --read         read every tensor and check that all values are finite (slower: reads the whole model)
              --hub          compare each file's size and SHA-256 with the hub's (needs the network; hub models only)
              --token TOKEN  for gated and private models

        Exit code 1 when a check fails; the failed checks name the file. A damaged download is fixed with
        idrak pull MODEL --force.

        Examples:
          idrak verify Qwen/Qwen3-0.6B
          idrak verify ./model.gguf --read --json

        Environment: IDRAK_CACHE (the cache), HF_ENDPOINT, HF_TOKEN (with --hub)
        """;

    public override IReadOnlyCollection<string> ValueOptions => ["--token"];

    public override IReadOnlyCollection<string> Flags => ["--read", "--hub"];

    public override int Run(CommandContext context)
    {
        string name = context.Argument(0, "MODEL");
        var local = ModelCache.Locate(context, name);
        var checks = new List<(string Check, bool Ok, string Detail)>();
        void Add(string check, bool ok, string detail) => checks.Add((check, ok, detail));

        string folder = local.File is null ? local.Folder : Path.GetDirectoryName(Path.GetFullPath(local.File))!;
        foreach (string json in local.File is null ? Directory.GetFiles(local.Folder, "*.json") : [])
        {
            try
            {
                JsonNode.Parse(File.ReadAllText(json));
                Add(Path.GetFileName(json), true, "valid JSON");
            }
            catch (System.Text.Json.JsonException e)
            {
                Add(Path.GetFileName(json), false, $"not valid JSON: {e.Message}");
            }
        }

        var partial = Directory.GetFiles(folder, "*.part", SearchOption.AllDirectories);
        Add("downloads", partial.Length == 0, partial.Length == 0 ? "none left partial" : $"partial: {string.Join(", ", partial.Select(Path.GetFileName))} (pull again to resume)");

        if (local.File is { } gguf)
        {
            VerifyGguf(gguf, context.Flag("--read"), Add);
        }
        else
        {
            VerifySafeTensors(local.Folder, context.Flag("--read"), Add);
        }

        if (context.Flag("--hub"))
        {
            VerifyHub(context, name, local, Add);
        }

        int failed = checks.Count(c => !c.Ok);
        context.Table(["Check", "Result", "Detail"], checks.Select(c => (IReadOnlyList<string>)[c.Check, c.Ok ? "ok" : "FAILED", c.Detail]));
        context.Write(failed == 0 ? $"{name}: all {checks.Count} checks passed." : $"{name}: {failed} of {checks.Count} checks failed. Download it again: idrak pull {name} --force");
        context.WriteJson(new JsonObject
        {
            ["model"] = name,
            ["path"] = local.File ?? local.Folder,
            ["ok"] = failed == 0,
            ["checks"] = new JsonArray([.. checks.Select(c => (JsonNode)new JsonObject { ["check"] = c.Check, ["ok"] = c.Ok, ["detail"] = c.Detail })]),
        });
        return failed == 0 ? ExitCodes.Ok : ExitCodes.Failed;
    }

    private static void VerifySafeTensors(string folder, bool read, Action<string, bool, string> add)
    {
        string index = Path.Combine(folder, "model.safetensors.index.json");
        if (File.Exists(index))
        {
            try
            {
                var shards = (JsonNode.Parse(File.ReadAllText(index))?["weight_map"] as JsonObject ?? []).Select(p => (string?)p.Value).OfType<string>().Distinct().ToList();
                var missing = shards.Where(s => !File.Exists(Path.Combine(folder, s))).ToList();
                add("shards", missing.Count == 0, missing.Count == 0 ? $"all {shards.Count} shards present" : $"missing: {string.Join(", ", missing)}");
                if (missing.Count > 0)
                {
                    return;
                }
            }
            catch (System.Text.Json.JsonException e)
            {
                add("shards", false, e.Message);
                return;
            }
        }

        SafeTensorsReader reader;
        try
        {
            reader = SafeTensorsReader.Open(folder);
        }
        catch (Exception e) when (e is IOException or InvalidDataException or NotSupportedException or System.Text.Json.JsonException or InvalidOperationException)
        {
            add("weights", false, e.Message);
            return;
        }

        using (reader)
        {
            foreach (var file in reader.Tensors.Values.GroupBy(t => t.File))
            {
                long length = new FileInfo(file.Key).Length, end = file.Max(t => t.Offset + t.Length);
                bool sizesMatch = file.All(t => t.Length == t.Count * (t.Type == SafeTensorType.F32 ? 4 : 2));
                add(Path.GetFileName(file.Key), end == length && sizesMatch,
                    end > length ? $"truncated: the header needs {end:N0} bytes, the file has {length:N0}"
                    : !sizesMatch ? "a tensor's byte range does not match its shape"
                    : end < length ? $"{length - end:N0} bytes after the last tensor" : $"{file.Count()} tensors, {ModelCache.Size(length)}");
            }

            if (read)
            {
                bool ok = Read(reader.Tensors.Keys, reader.Read, out string detail);
                add("values", ok, detail);
            }
        }
    }

    private static void VerifyGguf(string path, bool read, Action<string, bool, string> add)
    {
        GgufFile file;
        try
        {
            file = GgufFile.Open(path);
        }
        catch (Exception e) when (e is IOException or InvalidDataException or NotSupportedException or EndOfStreamException or OverflowException)
        {
            add(Path.GetFileName(path), false, $"not a readable GGUF file: {e.Message}");
            return;
        }

        using (file)
        {
            string arch = file.Get("general.architecture", "");
            add(Path.GetFileName(path), true, $"GGUF version {file.Version}, {file.Tensors.Count} tensors, {ModelCache.Size(new FileInfo(path).Length)}");
            add("architecture", GgufArchitectures.Names.Contains(arch, StringComparer.OrdinalIgnoreCase),
                GgufArchitectures.Names.Contains(arch, StringComparer.OrdinalIgnoreCase) ? arch : $"'{arch}' is not registered (supported: {string.Join(", ", GgufArchitectures.Names)})");
            if (read)
            {
                bool ok = Read(file.Tensors.Keys, file.Read, out string detail);
                add("values", ok, detail);
            }
        }
    }

    // Reads every tensor; false with the first problem (unreadable, or not finite).
    private static bool Read(IEnumerable<string> names, Func<string, float[]> read, out string detail)
    {
        long values = 0;
        int count = 0;
        foreach (string name in names)
        {
            float[] data;
            try
            {
                data = read(name);
            }
            catch (Exception e) when (e is IOException or InvalidDataException or NotSupportedException or EndOfStreamException or ArgumentException)
            {
                detail = $"{name}: {e.Message}";
                return false;
            }

            int bad = Array.FindIndex(data, v => !float.IsFinite(v));
            if (bad >= 0)
            {
                detail = $"{name}: value {bad} is {data[bad]}";
                return false;
            }

            values += data.Length;
            count++;
        }

        detail = $"all {count} tensors read, {ModelCache.Count(values)} values finite";
        return true;
    }

    // The hub's sizes and LFS hashes against the local files (Idrak's downloads of hub models).
    private static void VerifyHub(CommandContext context, string name, ModelCache.Local local, Action<string, bool, string> add)
    {
        string model = Shared.Models.Choose(context, name).Model;
        string repo = PullTarget.SplitFile(model)?.Repo ?? PullTarget.SplitTag(model).Repo;
        if (!ModelSource.IsModelId(repo))
        {
            add("hub", false, $"{model} is not a hub model (owner/name), so there is nothing to compare with");
            return;
        }

        string? token = Shared.Hub.Token(context);
        var downloader = ModelCache.Downloader(context);
        string commit = HuggingFace.ResolveRevisionAsync(repo, "models", "main", token, downloader).GetAwaiter().GetResult();
        string revision = Path.GetFileName(Path.GetDirectoryName(Path.GetFullPath(local.File ?? Path.Combine(local.Folder, "x")))!);
        if (!commit.StartsWith(revision, StringComparison.OrdinalIgnoreCase))
        {
            add("revision", true, $"the cached copy ({revision}) is not the hub's current main ({commit[..12]}); idrak pull {repo} updates it");
        }

        var files = Shared.Hub.FilesAsync(repo, revision.Length == 40 || commit.StartsWith(revision, StringComparison.OrdinalIgnoreCase) ? commit : "main", token, downloader)
            .GetAwaiter().GetResult();
        IEnumerable<string> localFiles = local.File is { } gguf ? [gguf] : Directory.GetFiles(local.Folder).Where(f => !f.EndsWith(".part", StringComparison.Ordinal));
        foreach (string file in localFiles)
        {
            string relative = Path.GetFileName(file);
            var remote = files.FirstOrDefault(f => f.Path == relative || f.Path.EndsWith("/" + relative, StringComparison.Ordinal));
            if (remote is null)
            {
                continue;                                                        // a file made locally (a prepared folder's)
            }

            long size = new FileInfo(file).Length;
            if (size != remote.Size)
            {
                add($"hub: {relative}", false, $"{size:N0} bytes, the hub has {remote.Size:N0}");
                continue;
            }

            if (remote.Sha256 is { } expected)
            {
                using var stream = File.OpenRead(file);
                string actual = Convert.ToHexStringLower(SHA256.HashData(stream));
                add($"hub: {relative}", actual == expected, actual == expected ? "size and SHA-256 match" : $"SHA-256 {actual[..12]}…, the hub has {expected[..12]}…");
            }
            else
            {
                add($"hub: {relative}", true, "size matches (the hub gives no hash for small files)");
            }
        }
    }
}
