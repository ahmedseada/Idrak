// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;

namespace Idrak.IdrakAudio;

/// <summary>
/// The IdrakAudio tensors by their checkpoint names, in PyTorch's layouts (a linear's weight [out, in], a convolution's
/// [out, in / groups, 1, k]): read from a checkpoint, or drawn from a seed for a new model; every parameter remembered with
/// its name and layout so the model is written back (<see cref="Stored"/>) as it was read.
/// </summary>
internal sealed class IdrakAudioWeights
{
    private readonly ITensorStore? _store;
    private readonly Random? _random;
    private readonly float _scale;
    private readonly Device _device;
    private readonly List<(string Name, Tensor Tensor, int[] Stored, bool Transposed)> _entries = [];

    private IdrakAudioWeights(ITensorStore? store, Random? random, float scale, Device device) => (_store, _random, _scale, _device) = (store, random, scale, device);

    /// <summary>The tensors of a checkpoint.</summary>
    public static IdrakAudioWeights Read(ITensorStore store, Device device) => new(store, null, 1f, device);

    /// <summary>New tensors: matrices a normal draw of <paramref name="scale"/> / √inputs, biases 0, norms 1 and 0.</summary>
    public static IdrakAudioWeights Draw(int seed, float scale, Device device) => new(null, new Random(seed), scale, device);

    /// <summary>The parameters made, in the order they were made.</summary>
    public IReadOnlyList<(string Name, Tensor Tensor, int[] Stored, bool Transposed)> Entries => _entries;

    /// <summary>A linear layer from <c>{prefix}.weight</c> [outputs, inputs] and <c>{prefix}.bias</c> [outputs] (none when <paramref name="bias"/> is false).</summary>
    public Linear Linear(string prefix, int inputs, int outputs, string? weightName = null, string? biasName = null, bool bias = true)
    {
        string name = weightName ?? prefix + ".weight";
        float[] values;
        if (_store is not null)
        {
            Expect(name, [outputs, inputs]);
            values = _store.ReadTransposed(name);                                // [inputs, outputs]
        }
        else
        {
            values = Normal(inputs * outputs, _scale / MathF.Sqrt(inputs));
        }

        var weight = Add(name, values, [inputs, outputs], [outputs, inputs], transposed: true);
        return Idrak.Layers.Linear.FromWeights(weight, bias ? Bias(biasName ?? prefix + ".bias", outputs) : null);
    }

    /// <summary>A table [rows, columns] stored as it is (an embedding's): drawn as a matrix of <paramref name="columns"/> inputs.</summary>
    public Tensor Table(string name, int rows, int columns)
    {
        float[] values;
        if (_store is not null)
        {
            Expect(name, [rows, columns]);
            values = _store.Read(name);
        }
        else
        {
            values = Normal(rows * columns, _scale / MathF.Sqrt(columns));
        }

        return Add(name, values, [rows, columns], [rows, columns], transposed: false);
    }

    /// <summary>A layer normalization from <c>{prefix}.weight</c> and <c>{prefix}.bias</c>.</summary>
    public LayerNorm Norm(string prefix, int features, float epsilon) =>
        LayerNorm.FromWeights(Vector(prefix + ".weight", features, 1f), Vector(prefix + ".bias", features, 0f), epsilon);

    /// <summary>A bias [length] (zeros when drawn).</summary>
    public Tensor Bias(string name, int length) => Vector(name, length, 0f);

    /// <summary>A convolution's filters, stored [outputs, inputsPerGroup, 1, kernel], as [outputs, inputsPerGroup · kernel].</summary>
    public Tensor Convolution(string name, int outputs, int inputsPerGroup, int kernel)
    {
        float[] values;
        if (_store is not null)
        {
            Expect(name, [outputs, inputsPerGroup, 1, kernel]);
            values = _store.Read(name);
        }
        else
        {
            values = Normal(outputs * inputsPerGroup * kernel, _scale / MathF.Sqrt(inputsPerGroup * kernel));
        }

        return Add(name, values, [outputs, inputsPerGroup * kernel], [outputs, inputsPerGroup, 1, kernel], transposed: false);
    }

    private Tensor Vector(string name, int length, float fill)
    {
        float[] values;
        if (_store is not null)
        {
            Expect(name, [length]);
            values = _store.Read(name);
        }
        else
        {
            values = new float[length];
            Array.Fill(values, fill);
        }

        return Add(name, values, [length], [length], transposed: false);
    }

    private Tensor Add(string name, float[] values, int[] shape, int[] stored, bool transposed)
    {
        var tensor = Tensor.Persistent(values, shape, _device, requiresGrad: true);
        _entries.Add((name, tensor, stored, transposed));
        return tensor;
    }

    private void Expect(string name, int[] shape)
    {
        if (!_store!.Contains(name))
        {
            throw new InvalidDataException($"The IdrakAudio checkpoint has no tensor '{name}'.");
        }

        var stored = _store.ShapeOf(name);
        if (!stored.SequenceEqual(shape))
        {
            throw new InvalidDataException($"The IdrakAudio checkpoint's '{name}' is {Tensor.FormatShape(stored)}; the config gives {Tensor.FormatShape(shape)}.");
        }
    }

    // Normal draws (Box-Muller, in a fixed order from the seed), times `std`.
    private float[] Normal(int count, float std)
    {
        var values = new float[count];
        for (int i = 0; i < count; i += 2)
        {
            double u = 1 - _random!.NextDouble(), v = _random.NextDouble();
            double radius = Math.Sqrt(-2 * Math.Log(u));
            values[i] = (float)(std * radius * Math.Cos(2 * Math.PI * v));
            if (i + 1 < count)
            {
                values[i + 1] = (float)(std * radius * Math.Sin(2 * Math.PI * v));
            }
        }

        return values;
    }

    /// <summary>Tensors of the checkpoint that no layer read (a config that does not match its weights), or none.</summary>
    public IReadOnlyList<string> Unread() =>
        _store is null ? [] : [.. _store.Names.Except(_entries.Select(e => e.Name), StringComparer.Ordinal).Order(StringComparer.Ordinal)];

    /// <summary>A parameter's values in its stored layout: a linear's weight [inputs, outputs] transposed back to [outputs, inputs].</summary>
    public static float[] Stored(Tensor tensor, bool transposed)
    {
        var values = tensor.ToArray();
        if (!transposed)
        {
            return values;
        }

        int rows = tensor.Shape[0], columns = tensor.Shape[1];
        var back = new float[values.Length];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < columns; c++)
            {
                back[c * rows + r] = values[r * columns + c];
            }
        }

        return back;
    }
}
