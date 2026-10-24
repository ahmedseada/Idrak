// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.
// The network ported from xiph's RNNoise v0.1.1 (src/rnn.c: Copyright (c) 2008-2011 Octasic Inc., 2012-2017 Jean-Marc
// Valin; the model file format of src/rnn_reader.c: Copyright (c) 2018 Gregor Richards; BSD-3-Clause, its text in
// README.md in this folder).

using System.Buffers.Text;
using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Idrak.Audio.Plugins.RnNoise;

/// <summary>
/// One layer of RNNoise's network: a dense layer (<c>Recurrent</c> null) or a GRU. Weights are RNNoise's signed bytes
/// as floats (exactly), in its order: input <c>j</c>'s row of every neuron's weight (<c>j·stride + i</c>; a GRU's
/// stride is 3·neurons: the update gate, the reset gate, then the output).
/// </summary>
internal sealed class RnNoiseLayer(int inputs, int neurons, int activation, float[] weights, float[]? recurrent, float[] bias)
{
    public const int Tanh = 0, Sigmoid = 1, Relu = 2;

    public int Inputs { get; } = inputs;

    public int Neurons { get; } = neurons;

    public int Activation { get; } = activation;

    public float[] Weights { get; } = weights;

    public float[]? Recurrent { get; } = recurrent;

    public float[] Bias { get; } = bias;
}

/// <summary>
/// RNNoise's network, read from its model file (the text <c>rnnoise_model_from_file</c> reads: the line
/// <c>rnnoise-nu model file version 1</c>, then for the dense input layer, the VAD GRU, the noise GRU, the denoise GRU,
/// the denoise output and the VAD output: inputs, neurons and activation (0 tanh, 1 sigmoid, 2 ReLU, another tanh),
/// then the weights, a GRU's recurrent weights, and the biases, as whole numbers of a signed byte, the weights scaled
/// by 1/256 when used). Its layers' sizes are checked against what RNNoise feeds them. Immutable; models are read once
/// per file and shared (<see cref="Load"/>).
/// </summary>
internal sealed class RnNoiseModel
{
    /// <summary>The features RNNoise computes per frame (the network's input).</summary>
    public const int Features = 42;

    /// <summary>The bands whose gains it gives.</summary>
    public const int Bands = 22;

    /// <summary>The first line of a model file.</summary>
    public const string Header = "rnnoise-nu model file version 1";

    private const float WeightsScale = 1f / 256;
    private static readonly ConcurrentDictionary<string, (DateTime Written, long Length, RnNoiseModel Model)> Loaded = new(StringComparer.Ordinal);

    private RnNoiseModel(string path, RnNoiseLayer[] layers)
    {
        Path = path;
        (InputDense, VadGru, NoiseGru, DenoiseGru, DenoiseOutput, VadOutput) = (layers[0], layers[1], layers[2], layers[3], layers[4], layers[5]);
    }

    /// <summary>The file it was read from.</summary>
    public string Path { get; }

    public RnNoiseLayer InputDense { get; }

    public RnNoiseLayer VadGru { get; }

    public RnNoiseLayer NoiseGru { get; }

    public RnNoiseLayer DenoiseGru { get; }

    public RnNoiseLayer DenoiseOutput { get; }

    public RnNoiseLayer VadOutput { get; }

    /// <summary>The model in <paramref name="path"/>, read once per process while the file does not change (rule 75).</summary>
    /// <exception cref="FileNotFoundException">No such file.</exception>
    /// <exception cref="InvalidDataException">The file is not an RNNoise model, or its layers do not fit RNNoise's network.</exception>
    public static RnNoiseModel Load(string path)
    {
        string full = System.IO.Path.GetFullPath(path);
        var file = new FileInfo(full);
        if (!file.Exists)
        {
            throw new FileNotFoundException($"The RNNoise model file {full} does not exist.", full);
        }

        if (Loaded.TryGetValue(full, out var known) && known.Written == file.LastWriteTimeUtc && known.Length == file.Length)
        {
            return known.Model;
        }

        var model = Parse(full, File.ReadAllBytes(full));
        Loaded[full] = (file.LastWriteTimeUtc, file.Length, model);
        return model;
    }

    /// <summary>A model from the bytes of a model file (<paramref name="path"/> names it in messages).</summary>
    /// <exception cref="InvalidDataException">Not an RNNoise model, or its layers do not fit RNNoise's network.</exception>
    public static RnNoiseModel Parse(string path, ReadOnlySpan<byte> bytes)
    {
        var reader = new Reader(path, bytes);
        reader.ExpectHeader();
        var input = reader.Layer("the dense input layer", gru: false);
        var vad = reader.Layer("the VAD GRU", gru: true);
        var noise = reader.Layer("the noise GRU", gru: true);
        var denoise = reader.Layer("the denoise GRU", gru: true);
        var gains = reader.Layer("the denoise output", gru: false);
        var voice = reader.Layer("the VAD output", gru: false);
        reader.ExpectEnd();

        Fits(path, "the dense input layer", input.Inputs, Features, "the 42 features");
        Fits(path, "the VAD GRU", vad.Inputs, input.Neurons, "the dense input layer's outputs");
        Fits(path, "the noise GRU", noise.Inputs, input.Neurons + vad.Neurons + Features, "the dense input layer's outputs, the VAD GRU's state and the 42 features");
        Fits(path, "the denoise GRU", denoise.Inputs, vad.Neurons + noise.Neurons + Features, "the VAD GRU's state, the noise GRU's state and the 42 features");
        Fits(path, "the denoise output", gains.Inputs, denoise.Neurons, "the denoise GRU's state");
        Fits(path, "the denoise output's neurons", gains.Neurons, Bands, "a gain per band (22)");
        Fits(path, "the VAD output", voice.Inputs, vad.Neurons, "the VAD GRU's state");
        Fits(path, "the VAD output's neurons", voice.Neurons, 1, "one voice probability");
        return new RnNoiseModel(path, [input, vad, noise, denoise, gains, voice]);
    }

    /// <summary>
    /// compute_rnn: the band gains (<paramref name="gains"/>, 22) and the voice probability of one frame's
    /// <paramref name="features"/> (42), the GRUs' states in <paramref name="state"/> carried to the next frame.
    /// </summary>
    public float Run(RnNoiseNetworkState state, ReadOnlySpan<float> features, Span<float> gains)
    {
        int dense = InputDense.Neurons, vad = VadGru.Neurons, noise = NoiseGru.Neurons;
        var denseOut = state.DenseOut.AsSpan(0, dense);
        Dense(InputDense, denseOut, features);
        Gru(VadGru, state.Vad, denseOut, state);
        Span<float> voice = stackalloc float[1];
        Dense(VadOutput, voice, state.Vad);

        var noiseInput = state.NoiseInput.AsSpan();
        denseOut.CopyTo(noiseInput);
        state.Vad.CopyTo(noiseInput[dense..]);
        features.CopyTo(noiseInput[(dense + vad)..]);
        Gru(NoiseGru, state.Noise, noiseInput, state);

        var denoiseInput = state.DenoiseInput.AsSpan();
        state.Vad.CopyTo(denoiseInput);
        state.Noise.CopyTo(denoiseInput[vad..]);
        features.CopyTo(denoiseInput[(vad + noise)..]);
        Gru(DenoiseGru, state.Denoise, denoiseInput, state);
        Dense(DenoiseOutput, gains, state.Denoise);
        return voice[0];
    }

    // compute_dense: each neuron's sum starts at its bias and takes the inputs in order (Product keeps that order for
    // every neuron), then the scale and the activation.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Dense(RnNoiseLayer layer, Span<float> output, ReadOnlySpan<float> input)
    {
        int n = layer.Neurons;
        output = output[..n];
        layer.Bias.AsSpan(0, n).CopyTo(output);
        Product(output, layer.Weights, n, input[..layer.Inputs], default);
        for (int i = 0; i < n; i++)
        {
            output[i] = Activate(layer.Activation, WeightsScale * output[i]);
        }
    }

    // compute_gru: the update and reset gates (bias, the inputs in order, then the state in order), then the output (bias,
    // the inputs, then the state times the reset gate: (w · s) · r, as the C code multiplies), each neuron's sum in the
    // C code's order.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Gru(RnNoiseLayer layer, float[] stateArray, ReadOnlySpan<float> input, RnNoiseNetworkState scratch)
    {
        int n = layer.Neurons, stride = 3 * n;
        var state = stateArray.AsSpan(0, n);
        var gates = scratch.Gates.AsSpan(0, 2 * n);
        var output = scratch.Output.AsSpan(0, n);
        input = input[..layer.Inputs];
        layer.Bias.AsSpan(0, 2 * n).CopyTo(gates);
        Product(gates, layer.Weights, stride, input, default);
        Product(gates, layer.Recurrent!, stride, state, default);
        for (int i = 0; i < 2 * n; i++)
        {
            gates[i] = SigmoidApprox(WeightsScale * gates[i]);              // z, then r
        }

        var z = gates[..n];
        var r = gates[n..];
        layer.Bias.AsSpan(2 * n, n).CopyTo(output);
        Product(output, layer.Weights.AsSpan(2 * n), stride, input, default);
        Product(output, layer.Recurrent!.AsSpan(2 * n), stride, state, r);
        for (int i = 0; i < n; i++)
        {
            float sum = Activate(layer.Activation, WeightsScale * output[i]);
            output[i] = z[i] * state[i] + (1 - z[i]) * sum;
        }

        output.CopyTo(state);
    }

    // sum[k] += w[j·stride + k] · x[j] (times gate[j] when a gate is given: (w · x) · g) for j = 0, 1, ... in order, every
    // k: each neuron's sum takes its terms in the C code's order, multiply then add (no fused multiply-add: the C code
    // rounds the product). The neurons go four SIMD vectors at a time, their sums held in registers over all the inputs
    // (the weights, stored input by input, read once), then one vector at a time, then one by one.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Product(Span<float> sum, ReadOnlySpan<float> w, int stride, ReadOnlySpan<float> x, ReadOnlySpan<float> gate)
    {
        int n = sum.Length, m = x.Length;
        bool gated = !gate.IsEmpty;
        if (m == 0)
        {
            return;
        }

        // The unchecked loads below stay inside w and sum: checked once here (rule 51).
        if ((long)(m - 1) * stride + n > w.Length || (gated && gate.Length < m))
        {
            throw new ArgumentException($"RNNoise's product reads {(long)(m - 1) * stride + n} weights of {w.Length}.", nameof(w));
        }

        ref float weights = ref MemoryMarshal.GetReference(w);
        ref float sums = ref MemoryMarshal.GetReference(sum);
        int k = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int v = Vector<float>.Count;
            for (; k + 4 * v <= n; k += 4 * v)
            {
                var a0 = Vector.LoadUnsafe(ref sums, (nuint)k);
                var a1 = Vector.LoadUnsafe(ref sums, (nuint)(k + v));
                var a2 = Vector.LoadUnsafe(ref sums, (nuint)(k + 2 * v));
                var a3 = Vector.LoadUnsafe(ref sums, (nuint)(k + 3 * v));
                for (int j = 0; j < m; j++)
                {
                    nuint at = (nuint)((long)j * stride + k);
                    var xj = new Vector<float>(x[j]);
                    if (gated)
                    {
                        var gj = new Vector<float>(gate[j]);
                        a0 += Vector.LoadUnsafe(ref weights, at) * xj * gj;
                        a1 += Vector.LoadUnsafe(ref weights, at + (nuint)v) * xj * gj;
                        a2 += Vector.LoadUnsafe(ref weights, at + (nuint)(2 * v)) * xj * gj;
                        a3 += Vector.LoadUnsafe(ref weights, at + (nuint)(3 * v)) * xj * gj;
                    }
                    else
                    {
                        a0 += Vector.LoadUnsafe(ref weights, at) * xj;
                        a1 += Vector.LoadUnsafe(ref weights, at + (nuint)v) * xj;
                        a2 += Vector.LoadUnsafe(ref weights, at + (nuint)(2 * v)) * xj;
                        a3 += Vector.LoadUnsafe(ref weights, at + (nuint)(3 * v)) * xj;
                    }
                }

                a0.StoreUnsafe(ref sums, (nuint)k);
                a1.StoreUnsafe(ref sums, (nuint)(k + v));
                a2.StoreUnsafe(ref sums, (nuint)(k + 2 * v));
                a3.StoreUnsafe(ref sums, (nuint)(k + 3 * v));
            }

            for (; k + v <= n; k += v)
            {
                var a = Vector.LoadUnsafe(ref sums, (nuint)k);
                for (int j = 0; j < m; j++)
                {
                    var term = Vector.LoadUnsafe(ref weights, (nuint)((long)j * stride + k)) * new Vector<float>(x[j]);
                    a += gated ? term * new Vector<float>(gate[j]) : term;
                }

                a.StoreUnsafe(ref sums, (nuint)k);
            }
        }

        for (; k < n; k++)
        {
            float s = sum[k];
            for (int j = 0; j < m; j++)
            {
                float term = w[j * stride + k] * x[j];
                s += gated ? term * gate[j] : term;
            }

            sum[k] = s;
        }
    }

    private static float Activate(int activation, float x) => activation switch
    {
        RnNoiseLayer.Sigmoid => SigmoidApprox(x),
        RnNoiseLayer.Relu => x < 0 ? 0 : x,
        _ => TansigApprox(x),
    };

    /// <summary>tansig_approx: tanh from RNNoise's table of 201 values (steps of 0.04) and a correction, as the C code does.</summary>
    public static float TansigApprox(float x)
    {
        // The tests are reversed to catch NaNs (as the C code's).
        if (!(x < 8))
        {
            return 1;
        }

        if (!(x > -8))
        {
            return -1;
        }

        float sign = 1;
        if (x < 0)
        {
            x = -x;
            sign = -1;
        }

        int i = (int)(.5f + 25 * x);                                             // floor(.5 + 25 x): the same, x being 0 or more
        x -= .04f * i;
        float y = TansigTable[i];
        float dy = 1 - y * y;
        y = y + x * dy * (1 - y * x);
        return sign * y;
    }

    /// <summary>sigmoid_approx: .5 + .5 tansig(x / 2), the halves in double as the C code computes them.</summary>
    public static float SigmoidApprox(float x) => (float)(.5 + .5 * TansigApprox((float)(.5 * x)));

    private static void Fits(string path, string what, int given, int needed, string why)
    {
        if (given != needed)
        {
            throw new InvalidDataException($"{path}: {what} has {given} where RNNoise's network needs {needed} ({why}); it is not a model of RNNoise's classic network.");
        }
    }

    // The whitespace-separated whole numbers of a model file, parsed from its bytes (rule 74).
    private ref struct Reader(string path, ReadOnlySpan<byte> bytes)
    {
        private readonly ReadOnlySpan<byte> _bytes = bytes;
        private int _at;

        public void ExpectHeader()
        {
            var header = "rnnoise-nu model file version "u8;
            if (_bytes.StartsWith("﻿"u8))
            {
                _at = 3;
            }

            if (!_bytes[_at..].StartsWith(header))
            {
                var start = _bytes[_at..Math.Min(_bytes.Length, _at + 40)];
                throw new InvalidDataException($"{path} is not an RNNoise model file: it starts '{System.Text.Encoding.UTF8.GetString(start).Split('\n')[0]}', not '{Header}'.");
            }

            _at += header.Length;
            int version = Next("the version");
            if (version != 1)
            {
                throw new InvalidDataException($"{path} is an RNNoise model file of version {version}; this reader takes version 1.");
            }
        }

        public RnNoiseLayer Layer(string what, bool gru)
        {
            int inputs = Size(what, "inputs"), neurons = Size(what, "neurons"), activation = Next($"{what}'s activation");
            int gates = gru ? 3 : 1;
            var weights = Values(what, "weights", inputs * neurons * gates);
            var recurrent = gru ? Values(what, "recurrent weights", neurons * neurons * gates) : null;
            var bias = Values(what, "biases", neurons * gates);
            return new RnNoiseLayer(inputs, neurons, activation is RnNoiseLayer.Sigmoid or RnNoiseLayer.Relu ? activation : RnNoiseLayer.Tanh, weights, recurrent, bias);
        }

        public void ExpectEnd()
        {
            SkipSpace();
            if (_at < _bytes.Length)
            {
                throw new InvalidDataException($"{path}: more numbers follow the VAD output's biases (an RNNoise model file ends there).");
            }
        }

        private int Size(string what, string which)
        {
            int value = Next($"{what}'s {which}");
            return value is >= 1 and <= 128 ? value : throw new InvalidDataException($"{path}: {what}'s {which} is {value}; RNNoise's reader takes 1 to 128.");
        }

        private float[] Values(string what, string which, int count)
        {
            var values = new float[count];
            for (int i = 0; i < count; i++)
            {
                int v = Next($"{what}'s {which} ({count})");
                values[i] = v is >= -128 and <= 127 ? v : throw new InvalidDataException($"{path}: {what}'s {which} hold {v}; RNNoise's weights are signed bytes (-128 to 127).");
            }

            return values;
        }

        private int Next(string what)
        {
            SkipSpace();
            if (_at >= _bytes.Length)
            {
                throw new InvalidDataException($"{path} ends before {what}: the file is cut short.");
            }

            if (!Utf8Parser.TryParse(_bytes[_at..], out int value, out int used) || (_at + used < _bytes.Length && !IsSpace(_bytes[_at + used])))
            {
                int end = _at;
                while (end < _bytes.Length && !IsSpace(_bytes[end]) && end - _at < 20)
                {
                    end++;
                }

                throw new InvalidDataException($"{path}: '{System.Text.Encoding.UTF8.GetString(_bytes[_at..end])}' where {what} should be a whole number.");
            }

            _at += used;
            return value;
        }

        private void SkipSpace()
        {
            while (_at < _bytes.Length && IsSpace(_bytes[_at]))
            {
                _at++;
            }
        }

        private static bool IsSpace(byte b) => b is (byte)' ' or (byte)'\n' or (byte)'\r' or (byte)'\t';
    }

    // RNNoise's tansig_table.h: tanh(0.04 i) for i = 0 to 200, as it rounds them. An array: read through a span
    // property it cost 2.3 times as much a call (measured).
    private static readonly float[] TansigTable =
    [
        0.000000f, 0.039979f, 0.079830f, 0.119427f, 0.158649f, 0.197375f, 0.235496f, 0.272905f, 0.309507f, 0.345214f,
        0.379949f, 0.413644f, 0.446244f, 0.477700f, 0.507977f, 0.537050f, 0.564900f, 0.591519f, 0.616909f, 0.641077f,
        0.664037f, 0.685809f, 0.706419f, 0.725897f, 0.744277f, 0.761594f, 0.777888f, 0.793199f, 0.807569f, 0.821040f,
        0.833655f, 0.845456f, 0.856485f, 0.866784f, 0.876393f, 0.885352f, 0.893698f, 0.901468f, 0.908698f, 0.915420f,
        0.921669f, 0.927473f, 0.932862f, 0.937863f, 0.942503f, 0.946806f, 0.950795f, 0.954492f, 0.957917f, 0.961090f,
        0.964028f, 0.966747f, 0.969265f, 0.971594f, 0.973749f, 0.975743f, 0.977587f, 0.979293f, 0.980869f, 0.982327f,
        0.983675f, 0.984921f, 0.986072f, 0.987136f, 0.988119f, 0.989027f, 0.989867f, 0.990642f, 0.991359f, 0.992020f,
        0.992631f, 0.993196f, 0.993718f, 0.994199f, 0.994644f, 0.995055f, 0.995434f, 0.995784f, 0.996108f, 0.996407f,
        0.996682f, 0.996937f, 0.997172f, 0.997389f, 0.997590f, 0.997775f, 0.997946f, 0.998104f, 0.998249f, 0.998384f,
        0.998508f, 0.998623f, 0.998728f, 0.998826f, 0.998916f, 0.999000f, 0.999076f, 0.999147f, 0.999213f, 0.999273f,
        0.999329f, 0.999381f, 0.999428f, 0.999472f, 0.999513f, 0.999550f, 0.999585f, 0.999617f, 0.999646f, 0.999673f,
        0.999699f, 0.999722f, 0.999743f, 0.999763f, 0.999781f, 0.999798f, 0.999813f, 0.999828f, 0.999841f, 0.999853f,
        0.999865f, 0.999875f, 0.999885f, 0.999893f, 0.999902f, 0.999909f, 0.999916f, 0.999923f, 0.999929f, 0.999934f,
        0.999939f, 0.999944f, 0.999948f, 0.999952f, 0.999956f, 0.999959f, 0.999962f, 0.999965f, 0.999968f, 0.999970f,
        0.999973f, 0.999975f, 0.999977f, 0.999978f, 0.999980f, 0.999982f, 0.999983f, 0.999984f, 0.999986f, 0.999987f,
        0.999988f, 0.999989f, 0.999990f, 0.999990f, 0.999991f, 0.999992f, 0.999992f, 0.999993f, 0.999994f, 0.999994f,
        0.999994f, 0.999995f, 0.999995f, 0.999996f, 0.999996f, 0.999996f, 0.999997f, 0.999997f, 0.999997f, 0.999997f,
        0.999997f, 0.999998f, 0.999998f, 0.999998f, 0.999998f, 0.999998f, 0.999998f, 0.999999f, 0.999999f, 0.999999f,
        0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f,
        1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
        1.000000f,
    ];
}

/// <summary>The network's state between frames (the three GRUs) and its scratch, one per channel; nothing is allocated per frame.</summary>
internal sealed class RnNoiseNetworkState
{
    public RnNoiseNetworkState(RnNoiseModel model)
    {
        int widest = Math.Max(model.VadGru.Neurons, Math.Max(model.NoiseGru.Neurons, model.DenoiseGru.Neurons));
        (Vad, Noise, Denoise) = (new float[model.VadGru.Neurons], new float[model.NoiseGru.Neurons], new float[model.DenoiseGru.Neurons]);
        DenseOut = new float[model.InputDense.Neurons];
        NoiseInput = new float[model.NoiseGru.Inputs];
        DenoiseInput = new float[model.DenoiseGru.Inputs];
        Gates = new float[2 * widest];
        Output = new float[widest];
    }

    public float[] Vad { get; }

    public float[] Noise { get; }

    public float[] Denoise { get; }

    public float[] DenseOut { get; }

    public float[] NoiseInput { get; }

    public float[] DenoiseInput { get; }

    public float[] Gates { get; }

    public float[] Output { get; }
}
