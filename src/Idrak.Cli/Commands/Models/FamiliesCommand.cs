// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Layers;

namespace Idrak.Cli.Commands;

/// <summary>
/// <c>idrak families</c>: the model families Idrak loads (<see cref="PretrainedArchitectures"/>, with plug-ins' too) and
/// what each supports: sliding-window attention, soft-capping, RoPE scaling, query/key norms, biases, mixture of experts,
/// and loading from GGUF (<see cref="GgufArchitectures"/>). Features are found by reading a small probe config that asks for all of them
/// through the family's own reader, so a plug-in's family is described the same way.
/// </summary>
internal sealed class FamiliesCommand : Command
{
    public override string Name => "families";

    public override string Summary => "Supported model families and what each supports (windows, soft-capping, RoPE scalings, experts, GGUF)";

    public override string Usage => """
        [FILTER]

        Arguments:
          FILTER  only families whose name contains this text (e.g. qwen)

        A family is the "architectures" name in a model's config.json. "own network": the family builds its own layers.
        Experts: the family reads mixture-of-experts models (a router choosing a few expert feed-forward blocks per token).
        Plug-ins (--plugin) add families, GGUF architectures and RoPE scalings; they are listed too.

        Examples:
          idrak families
          idrak families gemma --json
          idrak families -P ./MyFamily.dll
        """;

    public override int Run(CommandContext context)
    {
        string? filter = context.Positional.Count > 0 ? context.Positional[0] : null;
        var ggufByFamily = GgufArchitectures.Names
            .SelectMany(n => new[] { GgufArchitectures.Get(n).HuggingFace, GgufArchitectures.Get(n).WithExperts }.OfType<string>().Select(family => (Family: family, Name: n)))
            .GroupBy(p => p.Family, StringComparer.Ordinal)
            .ToDictionary(g => g.Key, g => g.Select(p => p.Name).Order(StringComparer.Ordinal).ToList(), StringComparer.Ordinal);
        var rows = new List<JsonObject>();
        foreach (string name in PretrainedArchitectures.Names.Order(StringComparer.Ordinal).Where(n => filter is null || n.Contains(filter, StringComparison.OrdinalIgnoreCase)))
        {
            var architecture = PretrainedArchitectures.Get(name);
            var row = new JsonObject { ["family"] = name, ["ownNetwork"] = architecture.Build is not null };
            var withExperts = ExpertSpec(architecture, name);
            try
            {
                var spec = withExperts ?? architecture.Spec(Probe(name), []);
                row["slidingWindow"] = spec.SlidingWindow is not null;
                row["softCapping"] = spec.AttentionSoftcap is not null || spec.LogitSoftcap is not null;
                row["ropeScaling"] = spec.Rope?.Scaling is not null;
                row["queryKeyNorm"] = spec.QkNorm;
                row["qkvBias"] = spec.QkvBias;
                row["tiedEmbeddings"] = spec.TieEmbeddings;
                row["experts"] = withExperts is not null;
            }
            catch (Exception e) when (e is not OutOfMemoryException)
            {
                row["probe"] = e.Message;
            }

            row["gguf"] = new JsonArray([.. (ggufByFamily.TryGetValue(name, out var gguf) ? gguf : []).Select(g => (JsonNode?)JsonValue.Create(g))]);
            rows.Add(row);
        }

        string Yes(JsonNode? value) => value is null ? "?" : (bool)value ? "yes" : "-";
        if (rows.Count == 0)
        {
            context.Write($"No registered family matches '{filter}'.");
        }
        else
        {
            context.Table(["Family", "Windows", "Soft-capping", "RoPE scaling", "Q/K norm", "QKV bias", "Experts", "GGUF", "Note"],
                rows.Select(r => (IReadOnlyList<string>)[(string)r["family"]!, Yes(r["slidingWindow"]), Yes(r["softCapping"]), Yes(r["ropeScaling"]),
                    Yes(r["queryKeyNorm"]), Yes(r["qkvBias"]), Yes(r["experts"]), ((JsonArray)r["gguf"]!).Count == 0 ? "-" : string.Join(", ", (JsonArray)r["gguf"]!),
                    (bool)r["ownNetwork"]! ? "own network" : (string?)r["probe"] ?? ""]));
        }

        var scalings = RopeScalings.Names.Order(StringComparer.Ordinal).ToList();
        context.Write($"RoPE scalings: {string.Join(", ", scalings)} (read from config.json's rope_scaling by every family marked yes)");
        context.Write($"GGUF architectures: {string.Join(", ", GgufArchitectures.Names.Order(StringComparer.Ordinal))}");
        context.WriteJson(new JsonObject
        {
            ["families"] = new JsonArray([.. rows]),
            ["ropeScalings"] = new JsonArray([.. scalings.Select(s => (JsonNode?)JsonValue.Create(s))]),
            ["ggufArchitectures"] = new JsonArray([.. GgufArchitectures.Names.Order(StringComparer.Ordinal).Select(s => (JsonNode?)JsonValue.Create(s))]),
        });
        return ExitCodes.Ok;
    }

    // The family's spec of the probe with Mixtral's and Qwen's expert keys added, when it reads them into a spec with
    // experts; null for families without experts, which refuse such a configuration.
    private static DecoderSpec? ExpertSpec(PretrainedArchitecture architecture, string name)
    {
        var probe = Probe(name);
        probe["num_local_experts"] = 4;
        probe["num_experts"] = 4;
        probe["num_experts_per_tok"] = 2;
        probe["moe_intermediate_size"] = 32;
        try
        {
            return architecture.Spec(probe, []) is { Experts: > 0 } spec ? spec : null;
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            return null;
        }
    }

    // A small config with every optional feature switched on, in the keys Hugging Face's configs use (parsed from text,
    // so numbers convert to whatever type a family's reader asks for, as they do from a real config.json).
    private static JsonObject Probe(string architecture)
    {
        var probe = (JsonObject)JsonNode.Parse("""
            {
              "vocab_size": 256, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
              "num_key_value_heads": 2, "head_dim": 16, "max_position_embeddings": 4096, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
              "rope_scaling": { "rope_type": "linear", "type": "linear", "factor": 2.0 },
              "sliding_window": 1024, "use_sliding_window": true, "max_window_layers": 0, "sliding_window_pattern": 2,
              "layer_types": ["sliding_attention", "full_attention"],
              "attn_logit_softcapping": 50.0, "final_logit_softcapping": 30.0, "query_pre_attn_scalar": 16,
              "hidden_act": "silu", "hidden_activation": "gelu_pytorch_tanh", "tie_word_embeddings": false
            }
            """)!;
        probe["architectures"] = new JsonArray(architecture);
        return probe;
    }
}
