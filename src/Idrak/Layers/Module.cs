using System.Buffers.Binary;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using Idrak.Diagnostics;

namespace Idrak.Layers;

/// <summary>
/// Base class for network building blocks. A module maps an input tensor to an output tensor
/// and owns the trainable parameters it uses.
/// </summary>
public abstract class Module : IDisposable
{
    private const uint FileMagic = 0x3257_4B49; // "IKW2": parameters followed by buffers (float32)
    private const uint FileMagic3 = 0x3357_4B49; // "IKW3": int8 layers, then tensors each with an element type
    private const uint FileMagic4 = 0x3457_4B49; // "IKW4": int8 layers, bfloat16 layers, then tensors as in IKW3
    private const uint FileMagic5 = 0x3557_4B49; // "IKW5": int8, bfloat16, int4 layers, bfloat16 embeddings, then tensors as in IKW3

    /// <summary>An optional name shown in summaries and telemetry.</summary>
    public string? Name { get; set; }

    /// <summary>The name, or a description of the module when no name was set.</summary>
    public string DisplayName => Name ?? ToString() ?? GetType().Name;

    /// <summary>
    /// True in training mode (the default), false in evaluation mode. Layers such as
    /// <see cref="Dropout"/> behave differently in each. Change it with <see cref="Train"/> and <see cref="Eval"/>.
    /// </summary>
    public bool IsTraining { get; private set; } = true;

    /// <summary>
    /// Computes the module's output for a batch of inputs, recording gradients when autograd is on.
    /// Publishes a <see cref="LayerForward"/> event when <see cref="TelemetryLevel.Layers"/> is enabled.
    /// </summary>
    public Tensor Forward(Tensor input)
    {
        using var offload = Offloading.Enter(this, input);           // offloaded weights staged on the device (else nothing)
        if (!Telemetry.IsEnabled(TelemetryLevel.Layers))
        {
            return ForwardCore(input);
        }

        int depth = Telemetry.EnterLayer();
        long start = Stopwatch.GetTimestamp();
        Tensor output;
        try
        {
            output = ForwardCore(input);
        }
        catch
        {
            Telemetry.LeaveLayer(depth);
            throw;
        }

        Telemetry.LayerForward(this, input, output, start, depth);
        return output;
    }

    /// <summary>Implements the forward computation. Override this in custom layers.</summary>
    protected abstract Tensor ForwardCore(Tensor input);

    /// <summary>
    /// Runs inference: evaluation mode, no gradient recording. Publishes an <see cref="InferenceCompleted"/>
    /// event when <see cref="TelemetryLevel.Inference"/> is enabled.
    /// </summary>
    public Tensor Predict(Tensor input)
    {
        bool wasTraining = IsTraining;
        long start = Telemetry.Start(TelemetryLevel.Inference);
        Eval();
        try
        {
            Tensor output;
            using (Autograd.NoGrad())
            using (var scope = new TensorScope())
            {
                // Intermediate activations are released immediately; only the result survives.
                output = Forward(input);
                if (!ReferenceEquals(output, input))
                {
                    scope.Keep(output);
                }
            }

            if (start != 0)
            {
                output.Device.Synchronize();
                Telemetry.Inference(new InferenceCompleted(DisplayName, input.Rank > 0 ? input.Shape[0] : 1,
                    input.Shape.ToArray(), output.Shape.ToArray(), output.Device, Stopwatch.GetElapsedTime(start)));
            }

            return output;
        }
        finally
        {
            Train(wasTraining);
        }
    }

    /// <summary>Predicts a batch given as a [rows, features] array and returns [rows, outputs].</summary>
    public float[,] Predict(float[,] input)
    {
        var device = Parameters().FirstOrDefault()?.Device ?? Device.Default;
        using var x = Tensor.From(input, device);
        using var y = Predict(x);
        return y.ToArray2D();
    }

    /// <summary>Switches this module and its children to training (true) or evaluation (false) mode.</summary>
    public void Train(bool training = true)
    {
        IsTraining = training;
        foreach (var child in Children())
        {
            child.Train(training);
        }
    }

    /// <summary>Switches to evaluation mode (same as <c>Train(false)</c>).</summary>
    public void Eval() => Train(false);

    /// <summary>Direct sub-modules, for containers. Leaf layers return none.</summary>
    public virtual IEnumerable<Module> Children() => [];

    /// <summary>The trainable tensors of this module and its children, in a stable order.</summary>
    public virtual IEnumerable<Tensor> Parameters() => Children().SelectMany(c => c.Parameters());

    /// <summary>
    /// Non-trainable state of this module and its children (e.g. BatchNorm running statistics).
    /// Buffers are saved with <see cref="Save(string)"/> and moved by <see cref="To"/>, but not optimized.
    /// </summary>
    public virtual IEnumerable<Tensor> Buffers() => Children().SelectMany(c => c.Buffers());

    /// <summary>Creates a trainable parameter (outside any <see cref="TensorScope"/>).</summary>
    protected static Tensor CreateParameter(float[] values, int[] shape, Device device) =>
        Tensor.Persistent(values, shape, device, requiresGrad: true);

    /// <summary>Creates a non-trainable buffer (outside any <see cref="TensorScope"/>).</summary>
    protected static Tensor CreateBuffer(float[] values, int[] shape, Device device) =>
        Tensor.Persistent(values, shape, device, requiresGrad: false);

    /// <summary>Returns <paramref name="tensor"/> on <paramref name="device"/>, disposing the original when it had to be copied.</summary>
    protected static Tensor MoveTensor(Tensor tensor, Device device)
    {
        if (tensor.Device == device)
        {
            return tensor;
        }

        var moved = Tensor.Persistent(tensor.ToArray(), tensor.Shape, device, tensor.RequiresGrad);
        tensor.Dispose();
        return moved;
    }

    /// <summary>Uniform values in [-bound, bound).</summary>
    protected static float[] UniformValues(int count, float bound, Random random)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = (random.NextSingle() * 2f - 1f) * bound;
        }

        return values;
    }

    /// <summary>A table of layers and parameter counts.</summary>
    public string Summary()
    {
        var sb = new StringBuilder();
        void Walk(Module m, int depth)
        {
            long own = m.Children().Any() ? 0 : m.ParameterCount;
            sb.Append(new string(' ', depth * 2)).Append(m.DisplayName);
            sb.Append(own > 0 ? $"  [{own:N0} params]" : "").AppendLine();
            foreach (var child in m.Children())
            {
                Walk(child, depth + 1);
            }
        }

        Walk(this, 0);
        sb.Append($"Total trainable parameters: {ParameterCount:N0}");
        return sb.ToString();
    }

    /// <summary>The total number of trainable values.</summary>
    public long ParameterCount => Parameters().Sum(p => (long)p.Size);

    /// <summary>Moves every parameter to <paramref name="device"/> and returns this module.</summary>
    public Module To(Device device)
    {
        MoveTo(device);
        return this;
    }

    /// <summary>Moves this module's own parameters; by default forwards the call to the children.</summary>
    protected internal virtual void MoveTo(Device device)
    {
        foreach (var child in Children())
        {
            child.MoveTo(device);
        }
    }

    /// <summary>Writes all parameter values to a binary file (float32).</summary>
    public void Save(string path) => Save(path, WeightFormat.Float32);

    /// <summary>
    /// Writes all parameter values to a binary file in <paramref name="format"/>: Float16 and BFloat16 halve the file
    /// (values are rounded; they are float32 again after loading). Int8 weights are always stored as they are.
    /// </summary>
    public void Save(string path, WeightFormat format)
    {
        using var stream = File.Create(path);
        Save(stream, format);
    }

    /// <summary>Writes all parameter values to <paramref name="stream"/> (the same format as <see cref="Save(string)"/>); the stream stays open.</summary>
    public void Save(Stream stream) => Save(stream, WeightFormat.Float32);

    /// <summary>Writes all parameter values to <paramref name="stream"/> in <paramref name="format"/>; the stream stays open.</summary>
    public void Save(Stream stream, WeightFormat format)
    {
        // IKW3: the int8 layers (by position among the Linear layers), then each tensor with its element type.
        var linears = this.Descendants().OfType<Linear>().ToList();
        var int8 = linears.Select((l, i) => (l, i)).Where(p => p.l.Int8 is not null).ToList();
        var half = linears.Select((l, i) => (l, i)).Where(p => p.l.BFloat16 is not null).ToList();
        var int4 = linears.Select((l, i) => (l, i)).Where(p => p.l.Int4 is not null).ToList();
        var tables = this.Descendants().OfType<Embedding>().Select((e, i) => (e, i)).Where(p => p.e.BFloat16 is not null).ToList();
        var exact = int8.SelectMany(p => new[] { p.l.Int8!.Packed, p.l.Int8.Scales }).Concat(half.Select(p => p.l.BFloat16!.Packed))
            .Concat(int4.SelectMany(p => new[] { p.l.Int4!.Packed, p.l.Int4.Scales }))
            .Concat(tables.Select(p => p.e.BFloat16!.Packed))
            .ToHashSet(ReferenceEqualityComparer.Instance);
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        var parameters = Parameters().Concat(Buffers()).ToList();
        bool five = int4.Count > 0 || tables.Count > 0;
        writer.Write(five ? FileMagic5 : half.Count > 0 ? FileMagic4 : FileMagic3);
        writer.Write(int8.Count);
        foreach (var (_, index) in int8)
        {
            writer.Write(index);
        }

        if (half.Count > 0 || five)
        {
            writer.Write(half.Count);
            foreach (var (_, index) in half)
            {
                writer.Write(index);
            }
        }

        if (five)
        {
            writer.Write(int4.Count);
            foreach (var (_, index) in int4)
            {
                writer.Write(index);
            }

            writer.Write(tables.Count);
            foreach (var (_, index) in tables)
            {
                writer.Write(index);
            }
        }

        writer.Write(parameters.Count);
        foreach (var p in parameters)
        {
            writer.Write(p.Rank);
            foreach (int d in p.Shape)
            {
                writer.Write(d);
            }

            var type = exact.Contains(p) ? WeightFormat.Float32 : format;
            writer.Write((byte)type);
            var values = p.ToArray();
            if (type == WeightFormat.Float32 && BitConverter.IsLittleEndian)
            {
                writer.Write(MemoryMarshal.AsBytes(values.AsSpan()));                  // the values are the stored bytes
            }
            else
            {
                writer.Write(Encode(values, type));
            }
        }
    }

    /// <summary>Reads parameter values written by <see cref="Save(string)"/> into this module. The architecture must match.</summary>
    public void Load(string path)
    {
        using var stream = File.OpenRead(path);
        Load(stream, path);
    }

    /// <summary>Reads parameter values written by <see cref="Save(Stream)"/> from <paramref name="stream"/>; the stream stays open.</summary>
    public void Load(Stream stream) => Load(stream, "The stream");

    private void Load(Stream stream, string source)
    {
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        uint magic = reader.ReadUInt32();
        if (magic is not (FileMagic or FileMagic3 or FileMagic4 or FileMagic5))
        {
            throw new InvalidDataException($"{source} is not a Idrak weights file.");
        }

        if (magic is FileMagic3 or FileMagic4 or FileMagic5)
        {
            // Layers stored as int8, int4 or bfloat16 get room for their packed weights first (allocated, not computed: the
            // file's values replace them).
            var linears = this.Descendants().OfType<Linear>().ToList();
            int quantized = reader.ReadInt32();
            for (int i = 0; i < quantized; i++)
            {
                int index = reader.ReadInt32();
                if (index >= linears.Count)
                {
                    throw new InvalidDataException($"The file quantizes Linear layer {index}, but the model has {linears.Count}.");
                }

                linears[index].LoadAsInt8();
            }

            int halves = magic is FileMagic4 or FileMagic5 ? reader.ReadInt32() : 0;
            for (int i = 0; i < halves; i++)
            {
                int index = reader.ReadInt32();
                if (index >= linears.Count)
                {
                    throw new InvalidDataException($"The file stores Linear layer {index} as bfloat16, but the model has {linears.Count}.");
                }

                linears[index].LoadAsBFloat16();
            }

            int nibbles = magic == FileMagic5 ? reader.ReadInt32() : 0;
            for (int i = 0; i < nibbles; i++)
            {
                int index = reader.ReadInt32();
                if (index >= linears.Count)
                {
                    throw new InvalidDataException($"The file stores Linear layer {index} as int4, but the model has {linears.Count}.");
                }

                linears[index].LoadAsInt4();
            }

            var embeddings = this.Descendants().OfType<Embedding>().ToList();
            int tables = magic == FileMagic5 ? reader.ReadInt32() : 0;
            for (int i = 0; i < tables; i++)
            {
                int index = reader.ReadInt32();
                if (index >= embeddings.Count)
                {
                    throw new InvalidDataException($"The file stores Embedding {index} as bfloat16, but the model has {embeddings.Count}.");
                }

                embeddings[index].LoadAsBFloat16();
            }
        }

        var parameters = Parameters().Concat(Buffers()).ToList();
        int count = reader.ReadInt32();
        if (count != parameters.Count)
        {
            throw new InvalidDataException($"The file has {count} parameter tensors but the model has {parameters.Count}.");
        }

        foreach (var p in parameters)
        {
            var shape = new int[reader.ReadInt32()];
            for (int i = 0; i < shape.Length; i++)
            {
                shape[i] = reader.ReadInt32();
            }

            if (!shape.AsSpan().SequenceEqual(p.Shape))
            {
                throw new InvalidDataException($"Shape mismatch: file has {Tensor.FormatShape(shape)}, model has {Tensor.FormatShape(p.Shape)}.");
            }

            var type = magic is FileMagic3 or FileMagic4 or FileMagic5 ? (WeightFormat)reader.ReadByte() : WeightFormat.Float32;
            var bytes = new byte[p.Size * (type == WeightFormat.Float32 ? 4 : 2)];
            reader.BaseStream.ReadExactly(bytes);
            if (type == WeightFormat.Float32 && BitConverter.IsLittleEndian)
            {
                p.Load(MemoryMarshal.Cast<byte, float>(bytes.AsSpan()));             // the stored bytes are the values
            }
            else
            {
                p.Load(Decode(bytes, p.Size, type));
            }
        }
    }

    // Little-endian bytes of the values in `format` (bfloat16 rounds to nearest even; NaN stays NaN).
    private static byte[] Encode(float[] values, WeightFormat format)
    {
        var bytes = new byte[values.Length * (format == WeightFormat.Float32 ? 4 : 2)];
        for (int i = 0; i < values.Length; i++)
        {
            switch (format)
            {
                case WeightFormat.Float32:
                    BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4), values[i]);
                    break;
                case WeightFormat.Float16:
                    BinaryPrimitives.WriteHalfLittleEndian(bytes.AsSpan(i * 2), (Half)values[i]);
                    break;
                default:
                    uint bits = BitConverter.SingleToUInt32Bits(values[i]);
                    ushort b16 = float.IsNaN(values[i]) ? (ushort)0x7FC0 : (ushort)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
                    BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(i * 2), b16);
                    break;
            }
        }

        return bytes;
    }

    private static float[] Decode(byte[] bytes, int count, WeightFormat format)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = format switch
            {
                WeightFormat.Float32 => BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4)),
                WeightFormat.Float16 => (float)BinaryPrimitives.ReadHalfLittleEndian(bytes.AsSpan(i * 2)),
                WeightFormat.BFloat16 => BitConverter.UInt32BitsToSingle((uint)BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(i * 2)) << 16),
                _ => throw new InvalidDataException($"Unknown weight format {(int)format}."),
            };
        }

        return values;
    }

    /// <summary>Releases the device memory held by the parameters.</summary>
    public virtual void Dispose()
    {
        foreach (var p in Parameters().Concat(Buffers()))
        {
            p.Dispose();
        }

        GC.SuppressFinalize(this);
    }
}
