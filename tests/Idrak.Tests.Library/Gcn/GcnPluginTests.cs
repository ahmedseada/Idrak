// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Globalization;
using Idrak;
using Idrak.Gpu;
using Idrak.Gpu.Abstractions;
using Idrak.Gpu.Hip;
using Idrak.PluginTests;
using Idrak.Tests.Gcn;

// A plug-in's own GCN kernels (GcnKernel, plan 35 G7b), as CudaKernel runs a plug-in's PTX: the arguments read from the
// text's .arg lines and every launch checked against them (count, kinds, names in the refusals), the declared block,
// text refused with its line and column, assembled for every gfx9 processor, cached under a key of its own; run in the
// GCN interpreter against the CPU (the documentation's example and the outside plug-in's two kernels), and on HIP
// devices through IKernelLauncher<GcnKernel>.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] GcnPluginGroup =
    [
        ("gcn plug-in kernels: GcnKernel reads its arguments (names, kinds, offsets) and block from the text's .arg and block lines; arguments that do not match are refused naming the kernel and the argument, a block the text does not declare too; text whose declarations do not read is refused at construction and text that does not assemble at assembly, each with its line and column; IDRAK_HIP_KERNELS leaves plug-in kernels on; another device is refused", GcnPluginShapes),
        ("gcn plug-in kernels: the documentation's example and the outside plug-in's GCN kernels assemble for gfx900, gfx906, gfx908 and gfx90c into code objects whose metadata declares them as their text does (another kernel's object refused); the cache key of a plug-in's code object changes with its text, the target, the device, the format and the versions, and is never the library set's", GcnPluginAssembles),
        ("gcn plug-in kernels: the documentation's example (a scale) and the outside plug-in's unpacking and row scaling run in the interpreter for gfx90c and gfx908 over 1, 255, 1,000 and about 70,000 elements (grids of 1 to 5 blocks, grid-stride), equal to the CPU bit for bit; nothing past the count written, no missing wait, race or access out of range", GcnPluginInterpreter),
        ("hip plug-in kernels: on HIP devices a plug-in's GcnKernel launches through IKernelLauncher<GcnKernel>, equal to the CPU; its function loads once a device and text (assembled, then read from the kernel cache), and a profile times it under its name", HipPluginLaunch),
    ];

    // The documentation's example (GcnKernel's remarks, installation/hip.md): y = scale · x for i < n, one element a
    // work-item in blocks of 256 (no grid stride: ceil(n / 256) blocks).
    internal const string GcnExampleScale = """
        .kernel my_scale
        .arg x, 8, global
        .arg y, 8, global
        .arg scale, 4, value
        .arg n, 4, value
        .reqd_workgroup_size 256, 1, 1
        .user_sgpr kernarg_segment_ptr
        .system_sgpr workgroup_id_x
          s_load_dwordx4 %s[0:3], s[0:1], 0x0          ; x, y
          s_load_dwordx2 %s[4:5], s[0:1], 0x10         ; scale, n
          s_lshl_b32 %s6, s2, 8                        ; the block's first element
          v_add_u32_e32 %v0, %s6, v0                   ; this work-item's element
          v_cmp_gt_u32_e32 vcc, %s5, %v0
          s_and_saveexec_b64 %s[8:9], vcc              ; the work-items below n
          s_cbranch_execz .Ldone
          v_lshlrev_b32_e32 %v1, 2, %v0
          global_load_dword %v2, %v1, %s[0:1]
          v_mul_f32_e32 %v3, %s4, %v2
          global_store_dword %v1, %v3, %s[2:3]
        .Ldone:
          s_endpgm
        """;

    // The processors the plug-in kernels are assembled for: every gfx9 processor the plan names.
    private static readonly string[] PluginIsas = ["gfx900", "gfx906", "gfx908", "gfx90c"];

    // The kernels checked here: the example and the outside plug-in's two.
    private static GcnKernel[] PluginGcnKernels() => [new GcnKernel(GcnExampleScale, "my_scale"), .. PluginKernels.GcnKernels];

    internal static void GcnPluginShapes(Device device)
    {
        _ = device;
        var example = new GcnKernel(GcnExampleScale, "my_scale", "example scale");
        Check(example.Parameters == "ppww" && example.ParameterNames.SequenceEqual(["x", "y", "scale", "n"]) && example.Entry == "my_scale" && example.Name == "example scale",
            $"the example's arguments: {example.Parameters} ({string.Join(", ", example.ParameterNames)}), entry {example.Entry}, name {example.Name}");
        Check(example.Declarations.Arguments.Select(a => (a.Offset, a.Size, a.Kind)).SequenceEqual(
                [(0, 8, GcnArgumentKind.GlobalBuffer), (8, 8, GcnArgumentKind.GlobalBuffer), (16, 4, GcnArgumentKind.ByValue), (20, 4, GcnArgumentKind.ByValue)])
              && example.Declarations.RequiredWorkgroupSize == (256, 1, 1) && example.Declarations.MaxFlatWorkgroupSize == 256,
            $"the example's layout: {string.Join(", ", example.Declarations.Arguments)}, block {example.Declarations.RequiredWorkgroupSize}");

        // Every kind and offset rule: a buffer 'p', a 4-byte value 'w', an 8-byte one 'l' (aligned to 8), other sizes '?', an offset given.
        var kinds = new GcnKernel(".kernel kinds\n.arg a, 4, value\n.arg b, 8, value\n.arg c, 8, global\n.arg d, 2, value\n.arg e, 4, value, 40\n.lds 512\n  s_endpgm\n", "kinds");
        Check(kinds.Parameters == "wlp?w" && kinds.Declarations.Arguments.Select(a => a.Offset).SequenceEqual([0, 8, 16, 24, 40]) && kinds.Declarations.LdsBytes == 512,
            $"argument kinds and offsets: {kinds.Parameters}, {string.Join(", ", kinds.Declarations.Arguments.Select(a => a.Offset))}, LDS {kinds.Declarations.LdsBytes}");

        // Arguments checked as the launch checks them, each refusal naming the kernel and the argument.
        var cpu = Device.Cpu.Backend;
        var storage = cpu.Allocate(8, zeroed: true);
        try
        {
            KernelArgument[] good = [storage, storage, 2f, 8];
            KernelArgument.CheckAgainst(example.Name, example.Parameters, good, example.ParameterNames);
            KernelArgument.CheckAgainst(example.Name, example.Parameters, [storage, storage, 2, 8u], example.ParameterNames);   // a 4-byte value takes an integer too
            string? count = Refused(() => KernelArgument.CheckAgainst(example.Name, example.Parameters, [storage, storage, 2f], example.ParameterNames));
            Check(count is not null && count.Contains("'example scale'", StringComparison.Ordinal) && count.Contains("n: a 32-bit integer or a float", StringComparison.Ordinal)
                  && count.Contains("3 arguments given", StringComparison.Ordinal), $"one argument short: {count}");
            string? kind = Refused(() => KernelArgument.CheckAgainst(example.Name, example.Parameters, [storage, 2f, 2f, 8], example.ParameterNames));
            Check(kind is not null && kind.Contains("'example scale'", StringComparison.Ordinal) && kind.Contains("parameter 1 (y) takes a storage", StringComparison.Ordinal),
                $"a float for a buffer: {kind}");
            string? wide = Refused(() => KernelArgument.CheckAgainst(kinds.Name, kinds.Parameters, [1, 2, storage, 4, 5], kinds.ParameterNames));
            Check(wide is not null && wide.Contains("(b) takes a 64-bit integer", StringComparison.Ordinal), $"an int for an 8-byte value: {wide}");
            string? odd = Refused(() => KernelArgument.CheckAgainst(kinds.Name, kinds.Parameters, [1, 2L, storage, 4, 5], kinds.ParameterNames));
            Check(odd is not null && odd.Contains("(d) takes an unsupported type", StringComparison.Ordinal), $"a 2-byte value: {odd}");

            // Another device refuses it before anything else.
            string? other = Refused(() => example.Launch(cpu, 1, 1, 1, 256, 1, 1, good));
            Check(other is not null && other.Contains("runs on a HIP device", StringComparison.Ordinal) && other.Contains(cpu.Kind, StringComparison.Ordinal), $"on the CPU: {other}");
            Check(GcnKernel.UnavailableReason(cpu) is { } why && why.Contains("not a HIP device", StringComparison.Ordinal), $"the CPU's reason: {GcnKernel.UnavailableReason(cpu)}");
        }
        finally
        {
            storage.Release();
        }

        // The declared block: exactly the required one; at most the flat size otherwise (256 without either).
        Check(example.BlockRefused(256, 1, 1) is null && example.BlockRefused(0, 1, 1) is null, "the required block is taken");
        Check(example.BlockRefused(128, 2, 1) is { } shape && shape.Contains("'example scale'", StringComparison.Ordinal) && shape.Contains("256 × 1 × 1", StringComparison.Ordinal),
            $"another shape of 256: {example.BlockRefused(128, 2, 1)}");
        var flat = new GcnKernel(".kernel flat\n.max_flat_workgroup_size 512\n.system_vgpr workitem_id_y\n  s_endpgm\n", "flat");
        Check(flat.BlockRefused(32, 16, 1) is null && flat.BlockRefused(16, 16, 2) is null && flat.BlockRefused(32, 17, 1) is { } past && past.Contains("at most 512", StringComparison.Ordinal),
            $"a flat block of 512: {flat.BlockRefused(32, 17, 1)}");
        Check(kinds.BlockRefused(256, 1, 1) is null && kinds.BlockRefused(257, 1, 1) is not null, "256 without a block directive");

        // Text refused with its line and column: the declarations at construction, the instructions at assembly.
        (string Text, string Entry, string Where)[] declarations =
        [
            (".kernel k\n.arg a, 8, global\n.arg b, 4, bytes\n  s_endpgm\n", "k", "line 3, column 12"),       // not an argument kind
            (".kernel k\n.arg a, 4, global\n  s_endpgm\n", "k", "line 2, column 9"),                          // a buffer is 8 bytes
            ("  ; a comment\n.arg a, 8, global\n.kernel k\n  s_endpgm\n", "k", "line 2, column 1"),            // .kernel first
            (".kernel k\n.reqd_workgroup_size 64, 64, 1\n  s_endpgm\n", "k", "line 2, column 1"),             // past 1024 work-items
            ("  s_endpgm\n", "k", "line 1, column 1"),                                                        // no .kernel
        ];
        foreach (var (text, entry, where) in declarations)
        {
            string? message = Refused(() => _ = new GcnKernel(text, entry));
            Check(message is not null && message.Contains(where, StringComparison.Ordinal) && message.Contains("GCN kernel 'k'", StringComparison.Ordinal),
                $"refused at {where}: {message ?? "taken"}\n{text}");
        }

        string? entryRefused = Refused(() => _ = new GcnKernel(GcnExampleScale, "my_other"));
        Check(entryRefused is not null && entryRefused.Contains(".kernel my_scale", StringComparison.Ordinal), $"another entry: {entryRefused}");

        var bad = new GcnKernel(".kernel bad\n.arg y, 8, global\n.user_sgpr kernarg_segment_ptr\n  s_load_dwordx2 %s[0:1], s[0:1], 0x0\n  v_bogus_f32 %v0, %v1\n  s_endpgm\n", "bad");
        try
        {
            bad.Assemble(GcnIsaVersion.Parse("gfx90c"));
            Check(false, "an unknown instruction assembled");
        }
        catch (GcnSyntaxException e)
        {
            Check(e.Line == 5 && e.Column == 3 && e.Message.Contains("line 5, column 3", StringComparison.Ordinal), $"an unknown instruction: {e.Message}");
        }

        // IDRAK_HIP_KERNELS turns the library's kernels off, not a plug-in's: a target a family handles runs them either way.
        Check(HipBackend.KernelsSettingReason("0") is not null && HipBackend.TargetUnavailable("hip:0", "gfx90c", out _, out _) is null,
            "IDRAK_HIP_KERNELS=0 is the library's kernels' reason only; gfx90c runs plug-in kernels");

        static string? Refused(Action action)
        {
            try
            {
                action();
                return null;
            }
            catch (ArgumentException e)
            {
                return e.Message;
            }
        }
    }

    internal static void GcnPluginAssembles(Device device)
    {
        _ = device;
        var kernels = PluginGcnKernels();
        foreach (string processor in PluginIsas)
        {
            var isa = GcnIsaVersion.Parse(processor);
            var target = GcnTargets.For(isa);
            var images = new List<byte[]>();
            foreach (var kernel in kernels)
            {
                var code = kernel.Assemble(isa);
                Check(code.Name == kernel.Entry && code.Arguments.SequenceEqual(kernel.Declarations.Arguments) && code.RequiredWorkgroupSize == (256, 1, 1),
                    $"{processor} {kernel.Name}: {code.Name}, arguments {string.Join(", ", code.Arguments)}, block {code.RequiredWorkgroupSize}");
                foreach (int format in new[] { 5, 4 })
                {
                    byte[] image = GcnCodeObject.Write(target, isa, format, [code]);
                    string? problem = HipBackend.PluginLayout(image, kernel, out var layout);
                    Check(problem is null && layout is { } l && l.Name == kernel.Entry && l.ArgumentBytes >= kernel.Declarations.Arguments[^1].Offset + kernel.Declarations.Arguments[^1].Size,
                        $"{processor} {kernel.Name}, format {format}: {problem}");
                    var read = GcnCodeObjectReader.Read(image, kernel.Entry);
                    Check(read.Name == kernel.Entry && read.Code.Length > 0, $"{processor} {kernel.Name}, format {format}: found by the interpreter's reader");
                    if (format == 5)
                    {
                        images.Add(image);
                    }
                }
            }

            // A code object of another kernel is not this one's.
            Check(HipBackend.PluginLayout(images[0], kernels[1], out _) is { } other && other.Contains(kernels[1].Entry, StringComparison.Ordinal),
                $"{processor}: the example's object for {kernels[1].Name}: {HipBackend.PluginLayout(images[0], kernels[1], out _)}");
        }

        // The cache key: the text, the target, the device, the format and the versions; never the library set's.
        var source = new GcnKernelSource(kernels[0].Entry, kernels[0].Text);
        string Key(GcnKernelSource s, string format = "format=5", string target = "gfx90c", string device = "a GPU", string versions = "runtime 1, driver 2") =>
            HipKernelCache.KeyOf([s], [format, "plug-in"], target, device, versions);
        string key = Key(source);
        string[] others =
        [
            Key(new GcnKernelSource(kernels[0].Entry, System.Text.Encoding.UTF8.GetBytes(kernels[0].Gcn.Replace("%s4, %v2", "%s4, %v2 ; another", StringComparison.Ordinal)))),
            Key(source, format: "format=4"), Key(source, target: "gfx906"), Key(source, device: "another GPU"), Key(source, versions: "runtime 1, driver 3"),
            HipKernelCache.KeyOf(GcnKernels.All, ["format=5"], "gfx90c", "a GPU", "runtime 1, driver 2"),
        ];
        Check(key == Key(source) && others.All(o => o != key) && others.Distinct().Count() == others.Length, $"plug-in keys: {key} against {string.Join(", ", others)}");
    }

    internal static void GcnPluginInterpreter(Device device)
    {
        _ = device;
        int compared = 0;
        foreach (string processor in new[] { "gfx90c", "gfx908" })
        {
            var isa = GcnIsaVersion.Parse(processor);
            var target = GcnTargets.For(isa);
            foreach (var kernel in PluginGcnKernels())
            {
                var code = kernel.Assemble(isa);
                var image = GcnCodeObjectReader.Read(GcnCodeObject.Write(target, isa, 5, [code]), kernel.Entry);
                int argumentBytes = GcnCodeObject.KernargBytes(code);
                (int N, int Blocks, int Width)[] runs = kernel.Entry switch
                {
                    "my_scale" => [(1, 1, 0), (255, 1, 0), (1000, 4, 0), (70001, 274, 0)],
                    "outside_scale_rows" => [(1, 1, 1), (255, 1, 7), (1000, 2, 1000), (70001, 5, 13), (69888, 3, 65537)],
                    _ => [(1, 1, 0), (255, 1, 0), (1001, 2, 0), (70001, 5, 0)],
                };
                foreach (var (n, blocks, width) in runs)
                {
                    compared += PluginRun(kernel, image, argumentBytes, processor, n, blocks, width);
                }
            }
        }

        Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"    {compared:N0} elements on two processors, each equal to the CPU"));
    }

    // One run in the interpreter: the inputs mapped between sentinels, the kernel, its output compared bit for bit with
    // the CPU's computation of the same thing; the elements compared.
    private static int PluginRun(GcnKernel kernel, GcnKernelImage image, int argumentBytes, string processor, int n, int blocks, int width)
    {
        const float Sentinel = -98765.5f, Scale = -1.375f;
        const int Pad = 67;
        var random = new Random(n + blocks + width);
        float Next() => random.NextSingle() * 8f - 4f;
        var memory = new GcnMemory();
        var arguments = new byte[argumentBytes];
        var layout = kernel.Declarations.Arguments;
        ulong Map(string name, float[] values, int pad = 0)
        {
            var mapped = new float[values.Length + pad];
            Array.Fill(mapped, Sentinel);
            values.CopyTo(mapped, 0);
            return memory.Map<float>(name, mapped);
        }

        void Put(int i, ulong value)
        {
            if (layout[i].Size == 8)
            {
                BinaryPrimitives.WriteUInt64LittleEndian(arguments.AsSpan(layout[i].Offset), value);
            }
            else
            {
                BinaryPrimitives.WriteUInt32LittleEndian(arguments.AsSpan(layout[i].Offset), (uint)value);
            }
        }

        float[] expected;
        ulong output;
        switch (kernel.Entry)
        {
            case "my_scale":
            {
                float[] x = [.. Enumerable.Range(0, n).Select(_ => Next())];
                expected = [.. x.Select(v => Scale * v)];
                Put(0, Map("x", x));
                Put(1, output = Map("y", new float[n], Pad));
                Put(2, BitConverter.SingleToUInt32Bits(Scale));
                Put(3, (uint)n);
                break;
            }

            case "outside_scale_rows":
            {
                int rows = (n + width - 1) / width;
                float[] values = [.. Enumerable.Range(0, n).Select(_ => Next())], scales = [.. Enumerable.Range(0, rows).Select(_ => Next())];
                expected = [.. values.Select((v, i) => v * scales[i / width])];
                Put(0, Map("rows", values));
                Put(1, Map("scales", scales));
                Put(2, output = Map("output", new float[n], Pad));
                Put(3, (uint)n);
                Put(4, (uint)width);
                break;
            }

            default:
            {
                float[] words = [.. Enumerable.Range(0, (n + 1) / 2).Select(_ => BitConverter.UInt32BitsToSingle((uint)random.Next() & 0x7F7F7F7Fu))];
                expected = [.. Enumerable.Range(0, n).Select(i =>
                {
                    uint word = BitConverter.SingleToUInt32Bits(words[i >> 1]);
                    return BitConverter.UInt32BitsToSingle((i & 1) == 0 ? word << 16 : word & 0xFFFF0000u);
                })];
                Put(0, Map("packed", words));
                Put(1, output = Map("values", new float[n], Pad));
                Put(2, (uint)n);
                break;
            }
        }

        string what = string.Create(CultureInfo.InvariantCulture, $"{kernel.Name} on {processor}, {n} elements{(width > 0 ? $" in rows of {width}" : "")} in {blocks} blocks");
        try
        {
            GcnInterpreter.Run(image, memory, new GcnDispatch { Workgroups = (blocks, 1, 1), WorkgroupSize = (256, 1, 1), Arguments = arguments, Seed = (ulong)(n + blocks) });
        }
        catch (GcnInterpreterException e)
        {
            throw new Exception($"{what}: the interpreter reports {e.Kind}: {e.Message}", e);
        }

        float[] actual = memory.Read<float>(output, n + Pad);
        Check(actual.AsSpan(n).IndexOfAnyExcept(Sentinel) < 0, $"{what}: a float past the count was written");
        for (int i = 0; i < n; i++)
        {
            Check(BitConverter.SingleToInt32Bits(actual[i]) == BitConverter.SingleToInt32Bits(expected[i]),
                string.Create(CultureInfo.InvariantCulture, $"{what}: element {i} is {actual[i]:R}, the CPU's {expected[i]:R}"));
        }

        return n;
    }

    internal static void HipPluginLaunch(Device device)
    {
        if (device.Type != DeviceType.Hip)
        {
            return;
        }

        var hip = (HipBackend)device.Backend;
        if (GcnKernel.UnavailableReason(hip) is { } why)
        {
            Skip($"plug-in GCN kernels do not run on {hip.Name}: {why}");
        }

        var random = new Random(5);
        var owned = new List<Storage>();
        Storage On(float[] values)
        {
            var s = hip.Allocate(Math.Max(1, values.Length), zeroed: false);
            owned.Add(s);
            hip.Upload(values, s);
            return s;
        }

        float[] Read(Storage s, int n)
        {
            var values = new float[n];
            hip.Download(s, values);
            return values;
        }

        void Same(float[] expected, float[] actual, string what)
        {
            for (int i = 0; i < expected.Length; i++)
            {
                Check(BitConverter.SingleToInt32Bits(actual[i]) == BitConverter.SingleToInt32Bits(expected[i]),
                    string.Create(CultureInfo.InvariantCulture, $"{what}: element {i} is {actual[i]:R}, the CPU's {expected[i]:R}"));
            }
        }

        try
        {
            // The example: ceil(n / 256) blocks of 256, through IKernelLauncher<GcnKernel>.
            var example = new GcnKernel(GcnExampleScale, "my_scale");
            Check(hip is IKernelLauncher<GcnKernel>, "HipBackend launches GCN kernels");
            const int N = 70001;
            float[] x = [.. Enumerable.Range(0, N).Select(_ => random.NextSingle() * 8f - 4f)];
            var dx = On(x);
            var dy = On(new float[N]);
            example.Launch(hip, (N + 255) / 256, 1, 1, 256, 1, 1, [dx, dy, 0.625f, N]);
            Same([.. x.Select(v => 0.625f * v)], Read(dy, N), "the example on the device");
            var loaded = example.LastFunction;
            example.Launch(hip, (N + 255) / 256, 1, 1, 256, 1, 1, [dx, dy, -2f, N]);
            Check(loaded is not null && ReferenceEquals(example.LastFunction, loaded), "the function is looked up once a device");
            Same([.. x.Select(v => -2f * v)], Read(dy, N), "the example launched again");

            // The outside plug-in's kernels, grid-stride over fewer blocks than elements.
            var unpack = PluginKernels.GcnKernels[0];
            var scale = PluginKernels.GcnKernels[1];
            float[] words = [.. Enumerable.Range(0, (N + 1) / 2).Select(_ => BitConverter.UInt32BitsToSingle((uint)random.Next() & 0x7F7F7F7Fu))];
            var dw = On(words);
            var dv = On(new float[N]);
            unpack.Launch(hip, 7, 1, 1, 256, 1, 1, [dw, dv, N]);
            Same([.. Enumerable.Range(0, N).Select(i =>
            {
                uint word = BitConverter.SingleToUInt32Bits(words[i >> 1]);
                return BitConverter.UInt32BitsToSingle((i & 1) == 0 ? word << 16 : word & 0xFFFF0000u);
            })], Read(dv, N), "the plug-in's unpacking");
            const int Width = 13;
            float[] scales = [.. Enumerable.Range(0, (N + Width - 1) / Width).Select(_ => random.NextSingle() * 2f - 1f)];
            var ds = On(scales);
            var dout = On(new float[N]);
            scale.Launch(hip, 9, 1, 1, 256, 1, 1, [dx, ds, dout, N, Width]);
            Same([.. x.Select((v, i) => v * scales[i / Width])], Read(dout, N), "the plug-in's row scaling");

            // A profile times it under its name.
            var profiler = (IKernelProfiler)hip;
            profiler.StartProfile();
            example.Launch(hip, 1, 1, 1, 256, 1, 1, [dx, dy, 1f, 256]);
            var profile = profiler.StopProfile();
            Check(hip.ProfileUnavailableReason is not null || (profile.TryGetValue(example.Name, out var entry) && entry.Calls == 1),
                $"profiled under its name: {string.Join(", ", profile.Keys)}");

            // Loaded through a cache of its own: assembled and kept, then read back.
            string folder = Path.Combine(Path.GetTempPath(), $"idrak-gcn-plugin-{Environment.ProcessId}");
            try
            {
                var cache = new HipKernelCache(folder);
                var fresh = new GcnKernel(GcnExampleScale.Replace("; x, y", "; x, y (a fresh text)", StringComparison.Ordinal), "my_scale");
                Check(hip.LoadPlugin(fresh, cache) is not null && Directory.GetFiles(folder, "*.co").Length == 1, "assembled and kept in the cache");
                Check(hip.LoadPlugin(fresh, cache) is not null && Directory.GetFiles(folder, "*.co").Length == 1, "read back from the cache");
            }
            finally
            {
                if (Directory.Exists(folder))
                {
                    Directory.Delete(folder, recursive: true);
                }
            }
        }
        finally
        {
            foreach (var s in owned)
            {
                hip.Return(s);
            }
        }
    }
}
