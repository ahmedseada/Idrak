// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Idrak.Layers;
using Tensor = Idrak.Abstraction.Tensor;   // not ONNX Runtime's Tensor

namespace Idrak.Onnx.Runtime;

/// <summary>
/// An ONNX model run by ONNX Runtime, as a Idrak <see cref="Module"/>: it plugs into <c>Predictor.For</c>, the
/// inference engine, <see cref="Sequential"/> and the ASP.NET Core endpoints like any other model. It is for inference
/// only (no gradients, no parameters). The model must have one input and one output; its input is float32 (any
/// other type is refused: not on the new design yet).
/// </summary>
public sealed class OnnxModule : Module
{
    private readonly InferenceSession _session;

    private OnnxModule(InferenceSession session)
    {
        _session = session;
        if (session.InputMetadata.Count != 1 || session.OutputMetadata.Count < 1)
        {
            session.Dispose();
            throw new NotSupportedException($"OnnxModule needs a model with one input; this one has {session.InputMetadata.Count}.");
        }

        var (inputName, input) = session.InputMetadata.First();
        var (outputName, output) = session.OutputMetadata.First();
        InputName = inputName;
        OutputName = outputName;
        InputShape = input.Dimensions;
        OutputShape = output.Dimensions;
        if (input.ElementType != typeof(float))
        {
            session.Dispose();
            throw new NotSupportedException($"The model's input is {input.ElementType.Name}: type not on the new design yet.");
        }

        Name = "onnx";
    }

    /// <summary>Loads an .onnx file. <paramref name="configure"/> can choose execution providers (CUDA, DirectML, …) and threads.</summary>
    public static OnnxModule Load(string path, Action<SessionOptions>? configure = null)
    {
        using var options = Options(configure);
        return new OnnxModule(new InferenceSession(path, options));
    }

    /// <summary>Loads a model from the bytes of an .onnx file.</summary>
    public static OnnxModule Load(byte[] model, Action<SessionOptions>? configure = null)
    {
        using var options = Options(configure);
        return new OnnxModule(new InferenceSession(model, options));
    }

    /// <summary>
    /// Exports <paramref name="model"/> with <see cref="OnnxExport"/> (for samples shaped <paramref name="sampleShape"/>)
    /// and loads the result: the same network, run by ONNX Runtime.
    /// </summary>
    public static OnnxModule From(Module model, int[] sampleShape, Action<SessionOptions>? configure = null) =>
        Load(OnnxExport.For(model).Input(sampleShape).ToBytes(), configure);

    /// <summary>The model's input name.</summary>
    public string InputName { get; }

    /// <summary>The model's output name.</summary>
    public string OutputName { get; }

    /// <summary>The input's declared shape (-1 for dynamic dimensions).</summary>
    public IReadOnlyList<int> InputShape { get; }

    /// <summary>The output's declared shape (-1 for dynamic dimensions).</summary>
    public IReadOnlyList<int> OutputShape { get; }

    /// <summary>The ONNX Runtime session, for anything else (metadata, profiling).</summary>
    public InferenceSession Session => _session;

    /// <summary>The model's metadata (custom key/value pairs written by the exporter).</summary>
    public IReadOnlyDictionary<string, string> Metadata => _session.ModelMetadata.CustomMetadataMap;

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        // The input is read into a pooled buffer (returned in finally) and handed to ONNX Runtime as a view of it.
        int size = input.Size;
        float[] values = ArrayPool<float>.Shared.Rent(size);
        try
        {
            input.CopyTo(values.AsSpan(0, size));
            var value = NamedOnnxValue.CreateFromTensor(InputName, new DenseTensor<float>(values.AsMemory(0, size), input.Shape));

            using var results = _session.Run([value], [OutputName]);
            var output = results[0].AsTensor<float>();

            // A dense output is read where it lies (one copy, onto the device) before the results are disposed,
            // not copied to an array first.
            return output is DenseTensor<float> dense
                ? Tensor.From(dense.Buffer.Span[..(int)dense.Length], output.Dimensions, input.Device)
                : Tensor.From(output.ToArray(), output.Dimensions.ToArray(), input.Device);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(values);
        }
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        _session.Dispose();
        base.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() => $"OnnxModule({InputName} [{string.Join(", ", InputShape)}] -> {OutputName} [{string.Join(", ", OutputShape)}])";

    private static SessionOptions Options(Action<SessionOptions>? configure)
    {
        var options = new SessionOptions();
        configure?.Invoke(options);
        return options;
    }
}
