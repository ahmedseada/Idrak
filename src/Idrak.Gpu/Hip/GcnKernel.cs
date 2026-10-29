// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using Idrak.Abstraction.Operations;
using Idrak.Gpu.Abstractions;

namespace Idrak.Gpu.Hip;

/// <summary>
/// A compute kernel for a HIP device (<see cref="Launch(Backend, uint, uint, uint, uint, uint, uint, ReadOnlySpan{KernelArgument}, uint)"/>):
/// GCN text, AMD's own assembly, written with virtual registers as PTX is (<c>%v12</c> a vector register, <c>%s3</c> a
/// scalar one), in the syntax of the library's own HIP kernels, and the name its <c>.kernel</c> line gives. Its
/// arguments are read from the text's <c>.arg name, bytes, global|value[, offset]</c> lines, and every launch is checked
/// against them: a <c>global</c> argument takes a storage, a 4-byte <c>value</c> an integer or a float, an 8-byte one a
/// 64-bit integer; its block against the block the text declares (<c>.reqd_workgroup_size</c> exactly, else at most
/// <c>.max_flat_workgroup_size</c>, 256 when the text says neither). Each device assembles a text once, with Idrak's own
/// assembler, for the target it reports (no SDK, no compiler: the graphics driver's runtime alone loads it), keeps the
/// code object in the HIP kernel cache on disk (IDRAK_HIP_KERNEL_CACHE; the key hashes the text, the target, the device
/// and the runtime, driver and library versions) and loads the module once. A launch goes on the device's stream, in
/// order with its other work, is timed under <see cref="Name"/> while a profile runs, and reaches the telemetry when the
/// runtime refuses it, as the library's own kernels do. IDRAK_HIP_KERNELS=0 turns off the library's kernels, not a
/// plug-in's, as Vulkan's setting leaves a plug-in's SPIR-V and CUDA's plug-in PTX always runs. A plug-in ships the GCN
/// of its own and launches it from a kernel it registers for the "hip" kind
/// (<see cref="Kernels.Register{TKernel}(Operation{TKernel}, string, TKernel, Func{Backend, bool})"/>), with the
/// requirement <c>b =&gt; GcnKernel.UnavailableReason(b) is null</c> so a device no registered GCN family handles runs the
/// operation's default kernel. It launches on any backend that launches GCN kernels (<see cref="IKernelLauncher{TKernel}"/>).
/// Text that does not assemble for a device, a code object the runtime refuses in every format and a refused launch
/// raise <see cref="HipException"/>; the assembler's refusal names the line and column.
/// </summary>
/// <example>
/// <code>
/// const string ScaleText = """
///     .kernel my_scale
///     .arg x, 8, global
///     .arg y, 8, global
///     .arg scale, 4, value
///     .arg n, 4, value
///     .reqd_workgroup_size 256, 1, 1
///     .user_sgpr kernarg_segment_ptr
///     .system_sgpr workgroup_id_x
///       s_load_dwordx4 %s[0:3], s[0:1], 0x0          ; x, y
///       s_load_dwordx2 %s[4:5], s[0:1], 0x10         ; scale, n
///       s_lshl_b32 %s6, s2, 8                        ; the block's first element
///       v_add_u32_e32 %v0, %s6, v0                   ; this work-item's element
///       v_cmp_gt_u32_e32 vcc, %s5, %v0
///       s_and_saveexec_b64 %s[8:9], vcc              ; the work-items below n
///       s_cbranch_execz .Ldone
///       v_lshlrev_b32_e32 %v1, 2, %v0
///       global_load_dword %v2, %v1, %s[0:1]
///       v_mul_f32_e32 %v3, %s4, %v2
///       global_store_dword %v1, %v3, %s[2:3]
///     .Ldone:
///       s_endpgm
///     """;
/// static readonly GcnKernel Scale = new(ScaleText, "my_scale");
///
/// Kernels.Register(MyOps.Scale, "hip", (b, x, y, scale, n) =&gt; Scale.Launch(b, (uint)((n + 255) / 256), 1, 1, 256, 1, 1, [x, y, scale, n]),
///     b =&gt; GcnKernel.UnavailableReason(b) is null);
/// </code>
/// </example>
public sealed class GcnKernel : IDeviceKernel
{
    /// <summary>A kernel from GCN text and the name its <c>.kernel</c> line gives; <paramref name="name"/> is for messages and the profiler (the entry's when null).</summary>
    /// <exception cref="ArgumentException">
    /// The text's declarations are refused (the message names the line and column), or its <c>.kernel</c> is not
    /// <paramref name="entry"/>.
    /// </exception>
    public GcnKernel(string gcn, string entry, string? name = null)
    {
        ArgumentNullException.ThrowIfNull(gcn);
        ArgumentException.ThrowIfNullOrWhiteSpace(entry);
        Name = name ?? entry;
        Text = Encoding.UTF8.GetBytes(gcn);
        try
        {
            Declarations = ReadDeclarations(Text);
        }
        catch (GcnSyntaxException e)
        {
            throw new ArgumentException($"GCN kernel '{Name}': {e.Message}.", nameof(gcn), e);
        }

        if (!string.Equals(Declarations.Name, entry, StringComparison.Ordinal))
        {
            throw new ArgumentException($"GCN kernel '{Name}': the text declares .kernel {Declarations.Name}, not '{entry}' (one kernel a text).", nameof(entry));
        }

        Gcn = gcn;
        Entry = entry;
        Parameters = Letters(Declarations.Arguments);
        ParameterNames = [.. Declarations.Arguments.Select(a => a.Name ?? "")];
    }

    /// <summary>The GCN text.</summary>
    public string Gcn { get; }

    /// <summary>The kernel launched (its <c>.kernel</c> name).</summary>
    public string Entry { get; }

    /// <summary>A name for messages and the profiler.</summary>
    public string Name { get; }

    /// <summary>
    /// Its arguments as the text's <c>.arg</c> lines declare them, one letter each: 'p' a storage (<c>global</c>), 'w' a
    /// 32-bit integer or a float (a 4-byte <c>value</c>), 'l' a 64-bit integer (an 8-byte <c>value</c>), '?' a size a
    /// launch cannot pass (1, 2 or more than 8 bytes).
    /// </summary>
    public string Parameters { get; }

    /// <summary>The arguments' names, as the <c>.arg</c> lines give them, in order.</summary>
    public IReadOnlyList<string> ParameterNames { get; }

    /// <summary>
    /// Queues the kernel on the HIP device <paramref name="backend"/> over gridX × gridY × gridZ blocks of
    /// blockX × blockY × blockZ work-items, with <paramref name="arguments"/> in the order the text declares them
    /// (storages of that device, integers, floats) and <paramref name="sharedBytes"/> of dynamic shared memory (LDS,
    /// after the <c>.lds</c> bytes the text declares). It runs in order with the device's other work; reading a storage
    /// back waits for it. A grid with a zero dimension launches nothing. The first launch on a device assembles the text
    /// there (or reads its code object from the cache) and loads it.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// <paramref name="backend"/> does not launch GCN kernels, a storage is not on it, or the arguments do not match the
    /// ones the text declares (the message names the kernel and the argument).
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// The block is not the one the text declares, or the grid, the block or the shared memory exceed the device's limits.
    /// </exception>
    /// <exception cref="HipException">The text does not assemble for the device's target (its line and column), or the runtime refused the code object or the launch.</exception>
    public void Launch(Backend backend, uint gridX, uint gridY, uint gridZ, uint blockX, uint blockY, uint blockZ, ReadOnlySpan<KernelArgument> arguments, uint sharedBytes = 0)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (backend is not IKernelLauncher<GcnKernel> hip)
        {
            throw new ArgumentException($"GCN kernel '{Name}' runs on a HIP device, not on {backend.Name} ({backend.Kind}).", nameof(backend));
        }

        KernelArgument.CheckAgainst(Name, Parameters, arguments, ParameterNames);
        if (BlockRefused(blockX, blockY, blockZ) is { } refused)
        {
            throw new ArgumentOutOfRangeException(nameof(blockX), refused);
        }

        hip.Launch(this, gridX, gridY, gridZ, blockX, blockY, blockZ, arguments, sharedBytes);
    }

    /// <summary>
    /// Why GCN kernels cannot run on <paramref name="backend"/> (null when they can): it does not launch GCN kernels, or
    /// it is a HIP device whose reported target no registered GCN family handles ("not registered", naming it). A plug-in
    /// registers its kernel with the requirement <c>b =&gt; GcnKernel.UnavailableReason(b) is null</c>, so such a device
    /// runs the operation's default kernel instead of failing. IDRAK_HIP_KERNELS does not make a reason: it turns off the
    /// library's own kernels only.
    /// </summary>
    public static string? UnavailableReason(Backend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        return backend switch
        {
            not IKernelLauncher<GcnKernel> => $"{backend.Name} ({backend.Kind}) is not a HIP device",
            HipBackend hip => hip.PluginKernelsUnavailableReason,
            _ => null,
        };
    }

    /// <summary>The text's UTF-8 bytes, as the assembler reads them.</summary>
    internal byte[] Text { get; }

    /// <summary>What the text declares: its name, its arguments at their offsets, its block and its LDS.</summary>
    internal GcnKernelDeclarations Declarations { get; }

    /// <summary>The function loaded for it on the device that launched it last (looked up without a dictionary; one
    /// reference, so a thread never reads one device's owner with another's function).</summary>
    internal LoadedFunction? LastFunction;

    /// <summary>A function of this kernel loaded on a device.</summary>
    internal sealed record LoadedFunction(HipBackend Owner, HipBackend.GcnFunction Function);

    /// <summary>
    /// Why a block of <paramref name="x"/> × <paramref name="y"/> × <paramref name="z"/> work-items is not one the text
    /// declares (null when it is): not exactly its <c>.reqd_workgroup_size</c>, or past its <c>.max_flat_workgroup_size</c>.
    /// A grid with a zero dimension launches nothing, so an empty block is refused only with work to do (the device's check).
    /// </summary>
    internal string? BlockRefused(uint x, uint y, uint z)
    {
        ulong items = (ulong)x * y * z;
        if (Declarations.RequiredWorkgroupSize is { } required)
        {
            return (x, y, z) == ((uint)required.X, (uint)required.Y, (uint)required.Z) || items == 0 ? null
                : string.Create(CultureInfo.InvariantCulture,
                    $"GCN kernel '{Name}': blocks of {x} × {y} × {z} work-items; its text is written for {required.X} × {required.Y} × {required.Z} (.reqd_workgroup_size).");
        }

        return items <= (ulong)Declarations.MaxFlatWorkgroupSize ? null
            : string.Create(CultureInfo.InvariantCulture,
                $"GCN kernel '{Name}': blocks of {x} × {y} × {z} work-items; its text allows at most {Declarations.MaxFlatWorkgroupSize} (.max_flat_workgroup_size, 256 when not given).");
    }

    /// <summary>
    /// The text assembled for <paramref name="isa"/> by the registered family that handles it: what a device of that
    /// target loads. A text the assembler refuses raises <see cref="GcnSyntaxException"/> (its line and column) or
    /// <see cref="GcnAllocationException"/> (its registers past the declared block's budget).
    /// </summary>
    /// <exception cref="NotRegisteredException">No registered family handles the target.</exception>
    internal GcnKernelCode Assemble(GcnIsaVersion isa)
    {
        var target = GcnTargets.For(isa);
        using var asm = new GcnAssembler(target, isa, bytes: Math.Max(256, Text.Length / 2));
        return asm.AssembleKernel(Text);
    }

    // The declarations, read by the assembler of a registered family (the directives are Idrak's syntax, the same for every family).
    private static GcnKernelDeclarations ReadDeclarations(byte[] text)
    {
        foreach (var target in GcnTargets.All)
        {
            GcnAssembler asm;
            try
            {
                asm = new GcnAssembler(target, bytes: 64);
            }
            catch (NotSupportedException)
            {
                continue;                                                          // a family whose instructions Idrak does not write
            }

            using (asm)
            {
                return asm.ReadDeclarations(text);
            }
        }

        throw new NotRegisteredException(nameof(GcnTargets), "any", GcnTargets.Names, "No registered GCN family's assembler reads a kernel's text.");
    }

    // One letter an argument (see Parameters).
    private static string Letters(IReadOnlyList<GcnArgument> arguments)
    {
        var letters = new char[arguments.Count];
        for (int i = 0; i < letters.Length; i++)
        {
            var argument = arguments[i];
            letters[i] = argument.Kind == GcnArgumentKind.GlobalBuffer ? 'p'
                : argument.Size switch
                {
                    4 => 'w',
                    8 => 'l',
                    _ => '?',
                };
        }

        return new string(letters);
    }

    /// <inheritdoc />
    public override string ToString() => Name;
}

/// <summary>
/// What a kernel's text declares before its instructions (<see cref="GcnAssembler.ReadDeclarations"/>): its name, its
/// arguments at their offsets, the most work-items a block may have, the only block shape it runs with (when it requires
/// one) and the shared memory (LDS) it declares.
/// </summary>
internal sealed record GcnKernelDeclarations(string Name, IReadOnlyList<GcnArgument> Arguments, int MaxFlatWorkgroupSize, (int X, int Y, int Z)? RequiredWorkgroupSize, int LdsBytes);
