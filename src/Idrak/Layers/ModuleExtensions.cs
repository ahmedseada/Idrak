// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;

namespace Idrak.Layers;

/// <summary>
/// Fine-tuning helpers: freezing parameters, saving only what training changed, and LoRA adapters. Freezing sets
/// <see cref="Tensor.RequiresGrad"/> to false, exactly as the manual loop <c>foreach (var p in ...) p.RequiresGrad = false</c>;
/// the <see cref="Training.Trainer"/> factory constructor then gives the optimizer only the parameters still trainable.
/// </summary>
public static class ModuleExtensions
{
    private const uint TrainableMagic = 0x3150_4B49; // "IKP1": selected parameters with their indices

    /// <summary>Stops gradients for every parameter of <paramref name="module"/>; returns it for chaining.</summary>
    public static T Freeze<T>(this T module) where T : Module => SetTrainable(module, module.Parameters(), false);

    /// <summary>Re-enables gradients for every parameter of <paramref name="module"/>; returns it for chaining.</summary>
    public static T Unfreeze<T>(this T module) where T : Module => SetTrainable(module, module.Parameters(), true);

    /// <summary>Freezes the layers of <paramref name="model"/> in <paramref name="layers"/>, e.g. <c>model.Freeze(0..^1)</c> for all but the last.</summary>
    public static Sequential Freeze(this Sequential model, Range layers) => SetTrainable(model, Layers(model, layers).SelectMany(l => l.Parameters()), false);

    /// <summary>Unfreezes the layers of <paramref name="model"/> in <paramref name="layers"/>.</summary>
    public static Sequential Unfreeze(this Sequential model, Range layers) => SetTrainable(model, Layers(model, layers).SelectMany(l => l.Parameters()), true);

    /// <summary>The parameters that will be trained: those with <see cref="Tensor.RequiresGrad"/> set.</summary>
    public static IEnumerable<Tensor> TrainableParameters(this Module module) => module.Parameters().Where(p => p.RequiresGrad);

    /// <summary>Every module inside <paramref name="module"/> (itself included), depth first.</summary>
    public static IEnumerable<Module> Descendants(this Module module) =>
        new[] { module }.Concat(module.Children().SelectMany(Descendants));

    /// <summary>
    /// Runs the module with activation checkpointing: its intermediate results are not kept for the backward pass but
    /// recomputed then (one more forward pass of the module, far less memory: only its output is stored). Use it for the
    /// blocks of a deep model when training long sequences. Without gradient recording it is an ordinary forward pass.
    /// </summary>
    public static Tensor ForwardCheckpointed(this Module module, Tensor input) => Tensor.Checkpoint(module.Forward, input);

    // ------------------------------------------------------------------ saving only what changed

    /// <summary>
    /// Saves only the trainable parameters (for example a new head, or LoRA adapters), each with its position in
    /// <see cref="Module.Parameters"/>. Load them into a model with the same structure with <see cref="LoadTrainable(Module, string)"/>.
    /// </summary>
    public static void SaveTrainable(this Module module, string path)
    {
        using var stream = File.Create(path);
        module.SaveTrainable(stream);
    }

    /// <summary>Writes the trainable parameters to <paramref name="stream"/>; the stream stays open.</summary>
    public static void SaveTrainable(this Module module, Stream stream)
    {
        var selected = module.Parameters().Select((p, i) => (p, i)).Where(x => x.p.RequiresGrad).ToList();
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        writer.Write(TrainableMagic);
        writer.Write(selected.Count);
        foreach (var (p, index) in selected)
        {
            writer.Write(index);
            writer.Write(p.Rank);
            foreach (int d in p.Shape)
            {
                writer.Write(d);
            }

            var data = p.ToArray();
            if (!BitConverter.IsLittleEndian)
            {
                var bits = MemoryMarshal.Cast<float, int>(data.AsSpan());
                BinaryPrimitives.ReverseEndianness(bits, bits);
            }

            writer.Write(MemoryMarshal.AsBytes(data.AsSpan()));
        }
    }

    /// <summary>Loads parameters written by <see cref="SaveTrainable(Module, string)"/> into the same positions.</summary>
    public static void LoadTrainable(this Module module, string path)
    {
        using var stream = File.OpenRead(path);
        module.LoadTrainable(stream);
    }

    /// <summary>Reads parameters written by <see cref="SaveTrainable(Module, Stream)"/>; the stream stays open.</summary>
    public static void LoadTrainable(this Module module, Stream stream)
    {
        var parameters = module.Parameters().ToList();
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        if (reader.ReadUInt32() != TrainableMagic)
        {
            throw new InvalidDataException("Not a Idrak trainable-parameters file (written by SaveTrainable).");
        }

        int count = reader.ReadInt32();
        for (int n = 0; n < count; n++)
        {
            int index = reader.ReadInt32();
            if (index < 0 || index >= parameters.Count)
            {
                throw new InvalidDataException($"The file refers to parameter {index}, but the model has {parameters.Count}.");
            }

            var p = parameters[index];
            var shape = new int[reader.ReadInt32()];
            for (int i = 0; i < shape.Length; i++)
            {
                shape[i] = reader.ReadInt32();
            }

            if (!shape.AsSpan().SequenceEqual(p.Shape))
            {
                throw new InvalidDataException($"Parameter {index}: file has {Tensor.FormatShape(shape)}, model has {Tensor.FormatShape(p.Shape)}.");
            }

            var data = new float[p.Size];
            reader.BaseStream.ReadExactly(MemoryMarshal.AsBytes(data.AsSpan()));
            if (!BitConverter.IsLittleEndian)
            {
                var bits = MemoryMarshal.Cast<float, int>(data.AsSpan());
                BinaryPrimitives.ReverseEndianness(bits, bits);
            }

            p.Load(data);
        }
    }

    // ------------------------------------------------------------------ LoRA

    /// <summary>
    /// Adds a LoRA adapter to every <see cref="Linear"/> inside <paramref name="model"/> for which
    /// <paramref name="targets"/> returns true (including those inside attention and transformer layers), and returns
    /// how many were added. With <paramref name="freezeBase"/>, every parameter the model had before is frozen first,
    /// so only the adapters train. A starts with uniform values in ±1/√in, B with zeros: the outputs are unchanged
    /// until training. Create the optimizer after this call, so it sees the adapters.
    /// </summary>
    /// <param name="model">The model to adapt.</param>
    /// <param name="rank">The adapter rank r (e.g. 4–16).</param>
    /// <param name="alpha">Scaling numerator; the adapter's output is multiplied by alpha / r.</param>
    /// <param name="targets">Which layers get an adapter, e.g. <c>l =&gt; true</c> or <c>l =&gt; l.OutFeatures == 3 * dim</c>.</param>
    /// <param name="freezeBase">Freeze the existing parameters (the usual LoRA setup).</param>
    /// <param name="random">Source of A's initial values; pass a seeded <see cref="Random"/> for reproducible runs.</param>
    public static int AddLora(this Module model, int rank, float alpha, Func<Linear, bool> targets, bool freezeBase, Random? random = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rank);
        ArgumentNullException.ThrowIfNull(targets);
        random ??= Random.Shared;
        if (freezeBase)
        {
            model.Freeze();
        }

        int added = 0;
        foreach (var linear in model.Descendants().OfType<Linear>().Where(targets).ToList())
        {
            if (linear.Adapter is not null)
            {
                throw new InvalidOperationException($"{linear} already has a LoRA adapter.");
            }

            var device = linear.Device;
            float bound = 1f / MathF.Sqrt(linear.InFeatures);
            var a = new float[linear.InFeatures * rank];
            for (int i = 0; i < a.Length; i++)
            {
                a[i] = (random.NextSingle() * 2f - 1f) * bound;
            }

            linear.Adapter = new LoraAdapter(
                Tensor.Persistent(a, [linear.InFeatures, rank], device, requiresGrad: true),
                Tensor.Persistent(new float[rank * linear.OutFeatures], [rank, linear.OutFeatures], device, requiresGrad: true),
                rank, alpha / rank);
            added++;
        }

        return added;
    }

    /// <summary>
    /// Replaces the float weights of the <see cref="Linear"/> layers chosen by <paramref name="targets"/> (all when null)
    /// with int8 weights: one signed byte per weight and one float scale per output column (symmetric, max |w| / 127).
    /// Those weights take a quarter of the memory and are read 4× faster when generating token by token; the outputs
    /// change slightly (see the tests and the Quantization sample for measured differences). Biases, LoRA adapters,
    /// normalization, embeddings and convolutions stay float32. The quantized weights are fixed (not trainable), but
    /// gradients still flow through them, so LoRA adapters added afterwards can be trained (QLoRA-style fine-tuning).
    /// Save and load the model as usual (quantize a freshly built model before loading a quantized file). Returns the
    /// number of layers quantized.
    /// </summary>
    public static int QuantizeInt8(this Module model, Func<Linear, bool>? targets = null)
    {
        int count = 0;
        foreach (var linear in model.Descendants().OfType<Linear>().Where(l => l.Int8 is null && (targets?.Invoke(l) ?? true)).ToList())
        {
            linear.QuantizeInt8();
            count++;
        }

        return count;
    }

    /// <summary>
    /// Turns int8 weights back into float32 weights (the rounding done by <see cref="QuantizeInt8"/> is not undone), for
    /// example to fine-tune every weight or merge LoRA adapters. Returns the number of layers converted.
    /// </summary>
    public static int DequantizeInt8(this Module model, bool trainable = true)
    {
        int count = 0;
        foreach (var linear in model.Descendants().OfType<Linear>().Where(l => l.Int8 is not null).ToList())
        {
            linear.DequantizeInt8(trainable);
            count++;
        }

        return count;
    }

    /// <summary>
    /// Stores the weights of every <see cref="Linear"/> layer (or those <paramref name="targets"/> selects) as bfloat16:
    /// half the memory, and token-by-token decoding reads half the bytes; values keep about 3 significant digits. The
    /// weights become fixed (LoRA adapters on them still train). Returns how many layers changed.
    /// </summary>
    public static int ToBFloat16(this Module model, Func<Linear, bool>? targets = null)
    {
        int count = 0;
        foreach (var linear in model.Descendants().OfType<Linear>().Where(l => l.BFloat16 is null && (targets?.Invoke(l) ?? true)).ToList())
        {
            linear.ToBFloat16();
            count++;
        }

        return count;
    }

    /// <summary>
    /// Stores the weights of every <see cref="Linear"/> layer (or those <paramref name="targets"/> selects) as 4-bit values
    /// with one scale per 32 rows and column (see <see cref="Int4Weight"/>): about 5 bits per weight, and token-by-token
    /// decoding reads 8× less than float32. The weights become fixed; LoRA adapters on them still train (QLoRA). Returns
    /// how many layers changed.
    /// </summary>
    public static int QuantizeInt4(this Module model, Func<Linear, bool>? targets = null)
    {
        int count = 0;
        foreach (var linear in model.Descendants().OfType<Linear>().Where(l => l.Int4 is null && (targets?.Invoke(l) ?? true)).ToList())
        {
            linear.QuantizeInt4();
            count++;
        }

        return count;
    }

    /// <summary>Turns bfloat16 and 4-bit weights back into float32 weights (the rounding stays). Returns how many layers changed.</summary>
    public static int ToFloat32(this Module model, bool trainable = true)
    {
        int count = 0;
        foreach (var linear in model.Descendants().OfType<Linear>().Where(l => l.BFloat16 is not null || l.Int4 is not null).ToList())
        {
            linear.ToFloat32(trainable);
            count++;
        }

        foreach (var embedding in model.Descendants().OfType<Embedding>().Where(e => e.BFloat16 is not null).ToList())
        {
            embedding.ToFloat32(trainable);
            count++;
        }

        return count;
    }

    /// <summary>Folds every LoRA adapter into its layer's weight and removes it (smaller, faster model; same outputs).</summary>
    public static int MergeLora(this Module model)
    {
        int merged = 0;
        foreach (var linear in model.Descendants().OfType<Linear>().Where(l => l.Adapter is not null).ToList())
        {
            linear.MergeAdapter();
            merged++;
        }

        return merged;
    }

    private static T SetTrainable<T>(T module, IEnumerable<Tensor> parameters, bool trainable) where T : Module
    {
        foreach (var p in parameters)
        {
            p.RequiresGrad = trainable;
        }

        return module;
    }

    private static IEnumerable<Module> Layers(Sequential model, Range range)
    {
        var (offset, length) = range.GetOffsetAndLength(model.Count);
        return Enumerable.Range(offset, length).Select(i => model[i]);
    }
}
