// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Text.Plugins.Gemma;

// Plan 26, P6: the Gemma family is Idrak.Text.Plugins' (Gemma, Gemma 2, Gemma 3 and the text decoder of Gemma 3 with
// images) and its vision part Idrak.Vision.Plugins'; the library keeps none of it. The family's outputs are checked by
// the groups that always checked them (pretrained:, decoder:, vision contracts, ...).
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] GemmaGroup =
    [
        ("plugins: gemma: llama.cpp's Gemma 2 and 3 norm names, the rest the common ones (Gemma's too), the 1 added to every norm gain and to nothing else; the four norms under Gemma's names, the rest under the Llama names", GemmaPluginRegistration),
    ];

    internal static void GemmaPluginRegistration(Device device)
    {
        _ = device;
        Check(GemmaGguf.TensorName("blk.1.post_ffw_norm.weight") == "model.layers.1.post_feedforward_layernorm.weight"
              && GemmaGguf.TensorName("blk.0.ffn_norm.weight") == "model.layers.0.pre_feedforward_layernorm.weight"
              && GemmaGguf.TensorName("blk.2.post_attention_norm.weight") == "model.layers.2.post_attention_layernorm.weight"
              && GemmaGguf.TensorName("blk.0.attn_q.weight") == GgufArchitecture.CommonTensorName("blk.0.attn_q.weight")
              && GemmaGguf.StoredOffset("model.layers.0.self_attn.q_norm.weight") == 1f && GemmaGguf.StoredOffset("model.norm.weight") == 1f
              && GemmaGguf.StoredOffset("model.layers.0.mlp.up_proj.weight") == 0f,
            "llama.cpp's Gemma 2 and 3 norm names, the rest the common ones; the 1 added to every norm gain, to nothing else");
        Check(GemmaArchitectures.TensorName("layers.3.post_mlp_norm.weight") == "model.layers.3.post_feedforward_layernorm.weight"
              && GemmaArchitectures.TensorName("layers.0.mlp_norm.weight") == "model.layers.0.pre_feedforward_layernorm.weight"
              && GemmaArchitectures.TensorName("layers.1.attn.q.weight") == PretrainedFamilies.CommonTensorName("layers.1.attn.q.weight"),
            "the four norms under Gemma's names, the rest under the Llama names");
    }
}
