// Adad piece 1's reference harness. Runs the parts Adad copied as the Idrak build in IDRAK_BIN has them (default: the adad
// worktree's CLI build, loaded at run time; Adad never references Idrak) and prints what they give as SHA-256, for Adad's
// tests to compare with (stop point S3: a copied part's results differ from its source's):
//   ptx        the kernels' texts (line ends as LF): the native kernel's 9 shapes, the x pack's kernel, the q4_k widening's
//              module (header, pieces' table, bfloat16 widening), the splits' sum module
//   types      q4_k's encoder over the seeded rows (the source's recorded bytes), its decoder over those blocks and over
//              random valid blocks (the float bits)
//   control    the source's control lane (CudaBackend.StoredSuperLane, native: the x pack, the widening, the native kernel,
//              the sum) on random valid q4_k rows and random x at each test shape: the output's float bits; run with
//              IDRAK_AUTOTUNE=0 (the native kernel's first shape, unsplit) or with a fresh tuning folder and IDRAK_TUNE_LOG=1
//              (its measured shape and split, in the log)
using System.Buffers.Binary;
using System.Globalization;
using System.Reflection;
using System.Runtime.Loader;
using System.Security.Cryptography;
using System.Text;

string bin = Environment.GetEnvironmentVariable("IDRAK_BIN") ?? @"D:\Projects\idrak\test-output\worktrees\adad\src\Idrak.Cli\bin\Release\net10.0";
AssemblyLoadContext.Default.Resolving += (context, name) =>
    File.Exists(Path.Combine(bin, name.Name + ".dll")) ? context.LoadFromAssemblyPath(Path.Combine(bin, name.Name + ".dll")) : null;

return Harness.Run(args.Length > 0 ? args[0] : "ptx");

internal static class Harness
{
    private const BindingFlags Any = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Instance;

    // The test shapes (m, n, k): real products of the two models and edges (rows and columns past a tile, a short k).
    internal static readonly (int M, int N, int K)[] Shapes =
    [
        (128, 1024, 2048), (512, 2048, 2048), (512, 6144, 2048), (512, 2048, 6144), (128, 3072, 3072), (512, 1024, 3072),
        (128, 3072, 8192), (200, 300, 768), (37, 70, 256),
    ];

    public static int Run(string mode)
    {
        _ = Assembly.Load("Idrak");                                         // its registrations (the GGUF types)
        var gpu = Assembly.Load("Idrak.Gpu");
        Console.WriteLine($"build: {gpu.Location} {gpu.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion}");
        var ptx = gpu.GetType("Idrak.Gpu.Cuda.PtxKernels", throwOnError: true)!;
        var q4k = ElementTypes.Get("q4_k");
        var super = q4k.Super!;
        if (mode == "ptx")
        {
            var kindType = gpu.GetType("Idrak.Gpu.Cuda.PtxKernels+NativeKind", throwOnError: true)!;
            object bf16 = kindType.GetField("BF16", Any)!.GetValue(null)!;
            var configs = (Array)ptx.GetField("NativeMmaConfigs", Any)!.GetValue(null)!;
            var nativeMma = ptx.GetMethod("NativeMma", Any)!;
            for (int i = 0; i < configs.Length; i++)
            {
                object?[] call = [bf16, configs.GetValue(i), null, null, false];
                string text = (string)nativeMma.Invoke(null, call)!;
                Console.WriteLine($"native {i} {call[2]} {Sha(text)}");
            }

            var typedDecode = gpu.GetType("Idrak.Gpu.Cuda.TypedDecode", throwOnError: true)!;
            object decode = typedDecode.GetMethod("Of", Any)!.Invoke(null, [ElementType.Float32])!;
            object pack = ptx.GetMethod("NativePackModule", Any)!.Invoke(null, [decode, bf16])!;
            string packText = (string)pack.GetType().GetField("Item2")!.GetValue(pack)!;
            string xPart = packText[..packText.IndexOf(".visible .entry native_pack_w_", StringComparison.Ordinal)];
            Console.WriteLine($"pack_x native_pack_x_bf16 {Sha(xPart)}");
            string code = (string)ptx.GetMethod("SuperCode", Any)!.Invoke(null, [super])!;
            string header = (string)ptx.GetField("NativeHeader", Any)!.GetValue(null)!;
            string pieces = (string)ptx.GetMethod("SuperPiecesDeclaration", Any)!.Invoke(null, [super])!;
            string widen = (string)ptx.GetMethod("SuperWiden", Any)!.Invoke(null, [super, $"super_widen_{code}_bf16", true])!;
            Console.WriteLine($"widen super_widen_{code}_bf16 {Sha(header + pieces + widen)}");
            var sum = new StringBuilder();
            sum.AppendLine(".version 6.0");
            sum.AppendLine(".target sm_50");
            sum.AppendLine(".address_size 64");
            sum.AppendLine();
            ptx.GetMethod("TypedSum", Any)!.Invoke(null, [sum]);
            Console.WriteLine($"sum typed_sum_parts {Sha(sum.ToString())}");
            return 0;
        }

        if (mode == "types")
        {
            using var encoded = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            using var decoded = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            foreach (var (rows, k, seed) in new[] { (48, 2048, 1), (8, 6144, 2) })
            {
                var values = EncoderRows(rows, k, seed);
                int rowBytes = (int)q4k.ByteCount(k);
                var bytes = new byte[rows * rowBytes];
                for (int r = 0; r < rows; r++)
                {
                    q4k.Encode(values.AsSpan(r * k, k), bytes.AsSpan(r * rowBytes, rowBytes));
                }

                encoded.AppendData(bytes);
                var floats = new float[rows * k];
                super.Decode(bytes, floats);
                decoded.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(floats.AsSpan()));
            }

            Console.WriteLine($"encode q4_k seeded {Convert.ToHexStringLower(encoded.GetHashAndReset())}");
            Console.WriteLine($"decode q4_k seeded {Convert.ToHexStringLower(decoded.GetHashAndReset())}");
            var random = RandomBlocks(super, 64, 2048, 5);
            var randomFloats = new float[64 * 2048];
            super.Decode(random, randomFloats);
            Console.WriteLine($"decode q4_k random {Sha(System.Runtime.InteropServices.MemoryMarshal.AsBytes(randomFloats.AsSpan()))}");
            return 0;
        }

        if (mode == "control")
        {
            var device = Device.Cuda();
            var backend = device.Backend;
            var lane = backend.GetType().GetMethod("StoredSuperLane", Any)!;
            foreach (var (m, n, k) in Shapes)
            {
                var x = RandomRows(m, k, 1000 + k + m);
                var w = RandomBlocks(super, n, k, n * 31 + k);
                float[] y;
                using (var scope = new TensorScope())
                {
                    var input = Tensor.From(x, [m, k], device);
                    var weight = StoredWeight.FromStored(q4k, w, k, n, null, device);
                    var output = Tensor.Empty([m, n], device);
                    bool ran = (bool)lane.Invoke(backend, [q4k, input.Storage, weight.StoredValues.Storage, output.Storage, m, n, k, true])!;
                    if (!ran)
                    {
                        Console.WriteLine($"control {m} {n} {k} did not run");
                        return 1;
                    }

                    y = output.ToArray();
                }

                Console.WriteLine($"control {m} {n} {k} {Sha(System.Runtime.InteropServices.MemoryMarshal.AsBytes(y.AsSpan()))}");
            }

            return 0;
        }

        Console.WriteLine($"unknown mode {mode}");
        return 1;
    }

    private static string Sha(string text) => Sha(Encoding.ASCII.GetBytes(text.Replace("\r\n", "\n", StringComparison.Ordinal)));

    private static string Sha(ReadOnlySpan<byte> bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));

    // The seeded rows of the encoders' test (recorded before the encoders were made faster).
    internal static float[] EncoderRows(int rows, int k, int seed)
    {
        var random = new Random(seed);
        var values = new float[rows * k];
        for (int i = 0; i < values.Length; i++)
        {
            float v = (float)(random.NextDouble() + random.NextDouble() + random.NextDouble() + random.NextDouble() - 2.0) * 0.03f;
            values[i] = i % 97 == 0 ? v * 9f : v;
        }

        values.AsSpan(0, 256).Clear();
        values.AsSpan(256, 256).Fill(0.0125f);
        (values[512], values[520]) = (0.5f, -0.5f);
        (values[680], values[690], values[700]) = (-0.5f, 0.5f, -0.5f);
        values.AsSpan(736, 32).Fill(1e-40f);
        values[770] = float.NaN;
        values[1030] = float.PositiveInfinity;
        return values;
    }

    // Random valid blocks: random bytes, d and dmin random small finite halves (as Adad's bench and tests make them).
    internal static byte[] RandomBlocks(SuperBlock s, int rows, int k, int seed)
    {
        var random = new Random(seed);
        var bytes = new byte[(long)rows * (k / s.Values) * s.Bytes];
        random.NextBytes(bytes);
        for (long at = 0; at < bytes.Length; at += s.Bytes)
        {
            BinaryPrimitives.WriteHalfLittleEndian(bytes.AsSpan((int)at + s.ScaleOffset), (Half)(random.NextSingle() * 0.002f));
            if (s.MinOffset >= 0)
            {
                BinaryPrimitives.WriteHalfLittleEndian(bytes.AsSpan((int)at + s.MinOffset), (Half)(random.NextSingle() * 0.002f));
            }
        }

        return bytes;
    }

    internal static float[] RandomRows(int rows, int k, int seed)
    {
        var random = new Random(seed);
        var x = new float[(long)rows * k];
        for (long i = 0; i < x.Length; i++)
        {
            x[i] = (float)(random.NextDouble() + random.NextDouble() + random.NextDouble() + random.NextDouble() - 2.0);
        }

        return x;
    }
}
