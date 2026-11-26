// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Text;
using Idrak.Text.Generation;
using Idrak.Vision.Plugins.Qwen;

// Plan 16: Qwen-VL's multi-axis rotary positions: the grid rule checked against transformers 5.19's positions for the tiny
// random Qwen2.5-VL fixture's prompts (tools/vlm/make_tiny_qwen_vl.py, tests/Idrak.Tests/data/vlm-qwen), and the decoding
// context's rotary offsets.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] QwenVisionGroup =
    [
        ("qwen vl: the grid position rule places the fixture's prompts as transformers' get_rope_index does (position ids on three axes and the rope delta)", QwenPositionRule),
        ("qwen vl: the decoding context's rotary offsets: decode steps add the row's offset (recorded steps the one set last), truncating before a pass's images restores the offsets before it, inside them is refused, Reset clears them", QwenRotaryOffsets),
    ];

    internal static string QwenData(string name) => TestData($"vlm-qwen/{name}");

    internal static JsonNode QwenManifest() => JsonNode.Parse(File.ReadAllText(QwenData("manifest.json")))!;

    internal static int[] QwenInts(string path) => [.. ReadNpyInt64(path).Select(i => (int)i)];

    internal static void QwenPositionRule(Device device)
    {
        _ = device;
        var manifest = QwenManifest();
        var rule = new GridRule();
        int pad = (int)manifest["token_ids"]!["<|image_pad|>"]!;
        foreach (var (prompt, pages) in new[] { ("prompt-image", new[] { "page" }), ("prompt-two-images", new[] { "page", "page-tall" }) })
        {
            var facts = manifest["facts"]!["tiny-qwen2.5-vl"]!;
            int[] ids = QwenInts(QwenData($"reference/qwen2.5-vl-{prompt}-input_ids.npy"));
            int[] expected = QwenInts(QwenData($"reference/qwen2.5-vl-{prompt}-position_ids.npy"));
            var runs = new List<ImageRun>();
            int k = 0;
            for (int i = 0; i < ids.Length; i++)
            {
                if (ids[i] != pad)
                {
                    continue;
                }

                var grid = facts["pages"]![pages[k++]]!["grid_thw"]!.AsArray().Select(n => (int)n!).ToArray();
                int rows = grid[1] / 2, columns = grid[2] / 2, tokens = rows * columns;
                var relative = new float[3 * tokens];
                for (int t = 0; t < tokens; t++)
                {
                    (relative[tokens + t], relative[2 * tokens + t]) = (t / columns, t % columns);
                }

                runs.Add(new ImageRun(i, tokens, relative));
                i += tokens - 1;
            }

            var placed = new float[3 * ids.Length];
            int next = rule.Place(0, ids.Length, runs, 3, placed);
            Check(placed.Select(p => (int)p).SequenceEqual(expected), $"{prompt}: positions {string.Join(" ", placed.Take(40))}…");
            Check(next - ids.Length == (int)facts[prompt]!["rope_delta"]!, $"{prompt}: rope delta {next - ids.Length}, transformers' {facts[prompt]!["rope_delta"]}");
        }
    }

    internal static void QwenRotaryOffsets(Device device)
    {
        using var context = new DecodingContext(device, 2, 64);
        context.BeginStep(5);
        context.EndStep(5);
        Check(context.RotaryOffsets is null && context.Recordable, "no offsets until a pass sets them");
        context.SetRotaryOffsets([-3, -3], from: 2);                                   // a pass whose images start at 2 ended at 5
        Check(context.Recordable, "one offset for every row is recordable");
        context.BeginStep(1);
        Check(context.RotaryPositions is { } rp && rp.ToArray() is [2f], $"one step at 5 with offset -3 rotates by 2: {string.Join(",", context.RotaryPositions?.ToArray() ?? [])}");
        context.EndStep(1);
        context.BeginStep(2);
        Check(context.RotaryPositions!.ToArray().SequenceEqual([3f, 4f]), $"two steps at 6: {string.Join(",", context.RotaryPositions.ToArray())}");
        context.EndStep(2);
        context.SetRotaryOffsets([-3, -1], from: 6);                                   // a second pass, images from 6 to 8
        Check(!context.Recordable, "rows with different offsets are not recordable");
        context.BeginStep(2);
        Check(context.RotaryPositions!.ToArray().SequenceEqual([5f, 6f, 7f, 8f]), $"per row: {string.Join(",", context.RotaryPositions.ToArray())}");
        context.EndStep(2);
        Check(Fails(() => context.Truncate(7)), "inside a pass's images is refused");
        context.Truncate(6);
        Check(context.RotaryOffsets is [-3, -3], $"truncating to the second pass's images restores the first's offsets: {string.Join(",", context.RotaryOffsets ?? [])}");
        context.Truncate(1);
        Check(context.RotaryOffsets is null, "truncating before the first pass's images clears them");
        context.SetRotaryOffsets([4, 4], from: 0);
        context.Reset();
        Check(context.RotaryOffsets is null, "Reset clears them");
    }
}
