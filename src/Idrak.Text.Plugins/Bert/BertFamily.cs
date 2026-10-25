// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Devices;
using Idrak.Layers;
using Idrak.Layers.Abstractions;
using Idrak.Models;
using Idrak.Models.Abstractions;

namespace Idrak.Text.Plugins.Bert;

/// <summary>
/// The BERT family under one architecture name: its checkpoint's tensors (Hugging Face's names, with or without the
/// "bert." prefix of the task models, LayerNorm's "weight"/"bias" or the older "gamma"/"beta") become the library's
/// <see cref="SequenceEncoder"/> of post-norm <see cref="TransformerEncoderLayer"/>s, and its heads the library's
/// <see cref="EncoderHeads"/> kinds. Public API only.
/// </summary>
internal sealed class BertFamily(string architecture) : IPretrainedEncoderFamily
{
    public string Name { get; } = architecture;

    public ISequenceEncoder Build(PretrainedCheckpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        var sizes = Sizes.Read(checkpoint.Config);
        var store = checkpoint.Tensors;
        var device = checkpoint.Device;
        string p = Prefix(store);
        var built = new List<IDisposable>();                                       // taken back if a later tensor is missing
        try
        {
            var words = Keep(built, Embedding.FromWeights(Parameter(store, $"{p}embeddings.word_embeddings.weight", [sizes.Vocabulary, sizes.Width], device)));
            var positions = Keep(built, new PositionEmbedding(Parameter(store, $"{p}embeddings.position_embeddings.weight", [sizes.MaxPositions, sizes.Width], device)));
            var types = sizes.Types > 0
                ? Keep(built, Embedding.FromWeights(Parameter(store, $"{p}embeddings.token_type_embeddings.weight", [sizes.Types, sizes.Width], device)))
                : null;
            var norm = Keep(built, Norm(store, $"{p}embeddings.LayerNorm", sizes, device));
            var blocks = new List<Module>(sizes.Layers);
            for (int i = 0; i < sizes.Layers; i++)
            {
                string l = $"{p}encoder.layer.{i}.";
                var attention = new MultiHeadAttention(Keep(built, Attention(store, l, sizes, device)), Keep(built, Dense(store, $"{l}attention.output.dense", sizes.Width, sizes.Width, device)),
                    sizes.Heads);
                var block = TransformerEncoderLayer.FromLayers(
                    Keep(built, Norm(store, $"{l}attention.output.LayerNorm", sizes, device)), attention,
                    Keep(built, Norm(store, $"{l}output.LayerNorm", sizes, device)),
                    Keep(built, Dense(store, $"{l}intermediate.dense", sizes.Width, sizes.FeedForward, device)),
                    Keep(built, Dense(store, $"{l}output.dense", sizes.FeedForward, sizes.Width, device)),
                    BlockActivation(sizes.Activation), NormPlacement.After);
                blocks.Add(Keep(built, block));
            }

            return new SequenceEncoder(words, blocks, positions, types, norm, finalNorm: null, sizes.MaxPositions);
        }
        catch
        {
            foreach (var part in Enumerable.Reverse(built))
            {
                part.Dispose();
            }

            throw;
        }
    }

    public IReadOnlyList<EncoderHead> Heads(PretrainedCheckpoint checkpoint, ISequenceEncoder encoder)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        ArgumentNullException.ThrowIfNull(encoder);
        var sizes = Sizes.Read(checkpoint.Config);
        var store = checkpoint.Tensors;
        var device = checkpoint.Device;
        string pooler = $"{Prefix(store)}pooler.dense";
        bool hasPooler = store.Contains($"{pooler}.weight");
        var heads = new List<EncoderHead>();
        try
        {
            if (hasPooler)
            {
                heads.Add(EncoderHeads.Create(EncoderHeads.Pooler, new EncoderHeadRequest(sizes.Width, sizes.Width, new JsonObject { ["activation"] = "tanh" }, device, 0,
                    Reader(store, ("dense", pooler)))));
            }

            if (store.Contains("classifier.weight"))
            {
                if (!hasPooler)
                {
                    throw new InvalidDataException($"The {Name} checkpoint holds a classifier but no {pooler}.weight (the pooler it reads).");
                }

                int labels = Shape(store, "classifier.weight", 2)[0];
                heads.Add(EncoderHeads.Create(EncoderHeads.SequenceClassification, new EncoderHeadRequest(sizes.Width, labels,
                    new JsonObject { ["dense"] = true, ["activation"] = "tanh" }, device, 0, Reader(store, ("dense", pooler), ("out", "classifier")))));
            }
            else if (Name == BertPlugin.BertForSequenceClassification)
            {
                throw new InvalidDataException($"A {Name} checkpoint holds classifier.weight and classifier.bias; this one has none.");
            }

            if (store.Contains("cls.predictions.transform.dense.weight"))
            {
                heads.Add(MaskedLanguageModel(checkpoint, encoder, sizes));
            }
            else if (Name == BertPlugin.BertForMaskedLM)
            {
                throw new InvalidDataException($"A {Name} checkpoint holds cls.predictions.transform.dense.weight (its masked-language-model head); this one has none.");
            }

            return heads;
        }
        catch
        {
            foreach (var head in heads)
            {
                head.Dispose();
            }

            throw;
        }
    }

    // The sequence vector a BERT head reads: the first token's state ([CLS]).
    public string Pooler(JsonObject config) => SequencePoolers.First;

    // "masked-lm": the transform (dense, the activation, LayerNorm), then the decoder: the word embeddings read in place
    // when tied (config.json's tie_word_embeddings, true unless set), else cls.predictions.decoder.weight; the bias
    // cls.predictions.bias (or the decoder's).
    private EncoderHead MaskedLanguageModel(PretrainedCheckpoint checkpoint, ISequenceEncoder encoder, Sizes sizes)
    {
        var store = checkpoint.Tensors;
        bool tied = checkpoint.Config["tie_word_embeddings"] is not { } tie || tie.GetValueKind() != JsonValueKind.False;
        var words = tied ? (encoder as IWordEmbeddings)?.Words : null;              // tested once, when the head is built
        string bias = store.Contains("cls.predictions.bias") ? "cls.predictions.bias" : "cls.predictions.decoder.bias";
        string decoder = tied && words is null ? $"{Prefix(store)}embeddings.word_embeddings" : "cls.predictions.decoder";
        var settings = new JsonObject { ["transform"] = true, ["activation"] = HeadActivation(sizes.Activation), ["norm_epsilon"] = sizes.Epsilon };
        var weights = Reader(store, ("dense", "cls.predictions.transform.dense"), ("norm", "cls.predictions.transform.LayerNorm"));
        return EncoderHeads.Create(EncoderHeads.MaskedLanguageModel, new EncoderHeadRequest(sizes.Width, sizes.Vocabulary, settings, checkpoint.Device, 0,
            (name, shape) => name switch
            {
                "out.bias" => store.Contains(bias) ? Checked(store, bias, [.. shape], transposed: false) : null,
                "out.weight" => Checked(store, $"{decoder}.weight", [shape[0], shape[1]], transposed: true),          // [vocabulary, width] stored
                _ => weights(name, shape),
            })
        {
            TiedTo = words,
        });
    }

    // The checkpoint's names start with "bert." in the task models (BertForMaskedLM, ...), with nothing in BertModel's.
    private static string Prefix(ITensorStore store) =>
        store.Contains("bert.embeddings.word_embeddings.weight") ? "bert." : store.Contains("embeddings.word_embeddings.weight") ? ""
        : throw new InvalidDataException("The checkpoint has no embeddings.word_embeddings.weight (nor bert.embeddings.word_embeddings.weight): not a BERT checkpoint.");

    // query, key and value side by side, [width, 3 · width] (each read transposed from Hugging Face's [out, in]), with their biases.
    private static Linear Attention(ITensorStore store, string layer, Sizes sizes, Device device)
    {
        int w = sizes.Width;
        var weight = new float[w * 3 * w];
        var bias = new float[3 * w];
        string[] parts = ["query", "key", "value"];
        for (int part = 0; part < 3; part++)
        {
            string name = $"{layer}attention.self.{parts[part]}";
            var values = Checked(store, $"{name}.weight", [w, w], transposed: true);            // [in, out]
            for (int row = 0; row < w; row++)
            {
                values.AsSpan(row * w, w).CopyTo(weight.AsSpan(row * 3 * w + part * w, w));
            }

            (store.Contains($"{name}.bias") ? Checked(store, $"{name}.bias", [w], transposed: false) : new float[w]).CopyTo(bias.AsSpan(part * w, w));
        }

        return Linear.FromWeights(Tensor.Persistent(weight, [w, 3 * w], device, requiresGrad: true), Tensor.Persistent(bias, [3 * w], device, requiresGrad: true));
    }

    // A dense layer [inputs, outputs] from Hugging Face's [outputs, inputs] weight and its bias.
    private static Linear Dense(ITensorStore store, string name, int inputs, int outputs, Device device) =>
        Linear.FromWeights(Tensor.Persistent(Checked(store, $"{name}.weight", [inputs, outputs], transposed: true), [inputs, outputs], device, requiresGrad: true),
            store.Contains($"{name}.bias") ? Tensor.Persistent(Checked(store, $"{name}.bias", [outputs], transposed: false), [outputs], device, requiresGrad: true) : null);

    private static LayerNorm Norm(ITensorStore store, string name, Sizes sizes, Device device)
    {
        string gain = NormPart(store, name, "weight", "gamma"), shift = NormPart(store, name, "bias", "beta");
        return LayerNorm.FromWeights(Tensor.Persistent(Checked(store, gain, [sizes.Width], transposed: false), [sizes.Width], device, requiresGrad: true),
            Tensor.Persistent(Checked(store, shift, [sizes.Width], transposed: false), [sizes.Width], device, requiresGrad: true), (float)sizes.Epsilon);
    }

    // A LayerNorm's tensor under its name now ("weight", "bias") or in older checkpoints ("gamma", "beta").
    private static string NormPart(ITensorStore store, string name, string current, string older) =>
        store.Contains($"{name}.{older}") && !store.Contains($"{name}.{current}") ? $"{name}.{older}" : $"{name}.{current}";

    private static Tensor Parameter(ITensorStore store, string name, int[] shape, Device device) =>
        Tensor.Persistent(Checked(store, name, shape, transposed: false), shape, device, requiresGrad: true);

    // The tensor's values in `shape` (Idrak's layout): read as stored, or transposed from [shape[1], shape[0]].
    private static float[] Checked(ITensorStore store, string name, int[] shape, bool transposed)
    {
        if (!store.Contains(name))
        {
            throw new InvalidDataException($"The BERT checkpoint has no tensor {name}.");
        }

        int[] stored = store.ShapeOf(name);
        int[] expected = transposed ? [shape[1], shape[0]] : shape;
        if (!stored.SequenceEqual(expected))
        {
            throw new InvalidDataException($"The BERT tensor {name} is [{string.Join(", ", stored)}]; the configuration makes it [{string.Join(", ", expected)}].");
        }

        return transposed ? store.ReadTransposed(name) : store.Read(name);
    }

    private static int[] Shape(ITensorStore store, string name, int rank)
    {
        int[] shape = store.ShapeOf(name);
        return shape.Length == rank ? shape : throw new InvalidDataException($"The BERT tensor {name} is [{string.Join(", ", shape)}]; a rank-{rank} tensor is expected.");
    }

    // A head's tensors by the names its kind reads ("dense.weight", "norm.bias", "out.weight", ...) from the checkpoint's
    // (stem → stored stem): weights transposed to [in, out], LayerNorm parts by either naming.
    private static Func<string, IReadOnlyList<int>, float[]?> Reader(ITensorStore store, params (string Part, string Stored)[] map) => (name, shape) =>
    {
        int dot = name.IndexOf('.', StringComparison.Ordinal);
        string part = name[..dot], what = name[(dot + 1)..];
        foreach (var (kind, stored) in map)
        {
            if (kind != part)
            {
                continue;
            }

            string full = what switch
            {
                "weight" when kind == "norm" => NormPart(store, stored, "weight", "gamma"),
                "bias" when kind == "norm" => NormPart(store, stored, "bias", "beta"),
                _ => $"{stored}.{what}",
            };
            if (!store.Contains(full))
            {
                return null;
            }

            bool matrix = shape.Count == 2;
            return Checked(store, full, matrix ? [shape[0], shape[1]] : [.. shape], transposed: matrix);
        }

        return null;
    };

    private static T Keep<T>(List<IDisposable> built, T part)
        where T : IDisposable
    {
        built.Add(part);
        return part;
    }

    // The blocks' feed-forward activation for config.json's hidden_act; null: the block's own GELU with the tanh
    // approximation (fused into the first projection during inference).
    private static Module? BlockActivation(string name) => name switch
    {
        "gelu" => new ExactGELU(),
        "gelu_new" or "gelu_pytorch_tanh" => null,
        "relu" => new ReLU(),
        _ => throw Unsupported(name),
    };

    // The "masked-lm" head kind's name of the same activation.
    private static string HeadActivation(string name) => name switch
    {
        "gelu" => "gelu",
        "gelu_new" or "gelu_pytorch_tanh" => "gelu-tanh",
        "relu" => "relu",
        _ => throw Unsupported(name),
    };

    private static NotSupportedException Unsupported(string name) =>
        new($"BERT's hidden_act '{name}' is not one the family reads (gelu, gelu_new, gelu_pytorch_tanh, relu).");

    // The sizes config.json gives.
    private sealed record Sizes(int Vocabulary, int Width, int Heads, int Layers, int FeedForward, int MaxPositions, int Types, double Epsilon, string Activation)
    {
        public static Sizes Read(JsonObject config)
        {
            ArgumentNullException.ThrowIfNull(config);
            if (config["position_embedding_type"] is { } kind && kind.GetValueKind() == JsonValueKind.String && kind.GetValue<string>() != "absolute")
            {
                throw new NotSupportedException($"BERT's position_embedding_type '{kind.GetValue<string>()}' is not read; the family reads absolute (learned) positions.");
            }

            if (config["is_decoder"] is { } decoder && decoder.GetValueKind() == JsonValueKind.True)
            {
                throw new NotSupportedException("A BERT checkpoint with is_decoder true is a causal decoder; the family reads encoders.");
            }

            int width = Int(config, "hidden_size"), heads = Int(config, "num_attention_heads");
            if (width % heads != 0)
            {
                throw new InvalidDataException($"BERT's hidden_size {width} is not a multiple of num_attention_heads {heads}.");
            }

            double epsilon = config["layer_norm_eps"] is { } e ? e.GetValue<double>() : 1e-12;
            return new Sizes(Int(config, "vocab_size"), width, heads, Int(config, "num_hidden_layers"), Int(config, "intermediate_size"),
                Int(config, "max_position_embeddings"), config["type_vocab_size"] is null ? 2 : Int(config, "type_vocab_size"), epsilon,
                (string?)config["hidden_act"] ?? "gelu");
        }

        private static int Int(JsonObject config, string key)
        {
            if (config[key] is not { } node || node.GetValueKind() != JsonValueKind.Number)
            {
                throw new InvalidDataException($"config.json has no {key} (a number), which a BERT model needs.");
            }

            int value = node.GetValue<int>();
            return value >= 0 ? value : throw new InvalidDataException($"BERT's {key} is not negative; got {value.ToString(CultureInfo.InvariantCulture)}.");
        }
    }
}
