// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.CompilerServices;

namespace Idrak.Gpu.Abstractions;

/// <summary>How a target string sets one of its features (xnack, sramecc): <c>xnack+</c> on, <c>xnack-</c> off, absent any.</summary>
public enum GcnFeatureSetting
{
    /// <summary>Not named: code that runs either way (the code object says "any").</summary>
    Any,

    /// <summary>Named with a minus: the device runs with the feature off.</summary>
    Off,

    /// <summary>Named with a plus: the device runs with the feature on.</summary>
    On,
}

/// <summary>
/// The instruction-set version a HIP device reports, read from its target string (the runtime's gcnArchName, such as
/// <c>gfx90c:xnack-</c>): the major version in decimal, then the minor version and the stepping as one hex digit each
/// (9.0.12, 10.3.0), and the features it names. It selects the <see cref="IGcnTarget"/> family and the code object's
/// machine value; it never names a card.
/// </summary>
/// <param name="Major">The major version (9 for GCN 5 and CDNA, 10 to 12 for the later families).</param>
/// <param name="Minor">The minor version (0 to 15).</param>
/// <param name="Stepping">The stepping (0 to 15).</param>
/// <param name="Xnack">Whether the device replays faulting memory accesses (xnack).</param>
/// <param name="Sramecc">Whether the device's memory error correction mode is on (sramecc).</param>
public readonly record struct GcnIsaVersion(int Major, int Minor, int Stepping, GcnFeatureSetting Xnack = GcnFeatureSetting.Any, GcnFeatureSetting Sramecc = GcnFeatureSetting.Any)
{
    /// <summary>The processor's name without features (the target string's first part).</summary>
    public string Processor => string.Create(CultureInfo.InvariantCulture, $"gfx{Major}{Minor:x}{Stepping:x}");

    /// <summary>The target id as LLVM writes it: the processor, then each named feature (sramecc before xnack).</summary>
    public override string ToString()
    {
        string processor = Processor;
        return (Sramecc, Xnack) switch
        {
            (GcnFeatureSetting.Any, GcnFeatureSetting.Any) => processor,
            (GcnFeatureSetting.Any, _) => $"{processor}:xnack{Sign(Xnack)}",
            (_, GcnFeatureSetting.Any) => $"{processor}:sramecc{Sign(Sramecc)}",
            _ => $"{processor}:sramecc{Sign(Sramecc)}:xnack{Sign(Xnack)}",
        };

        static char Sign(GcnFeatureSetting setting) => setting == GcnFeatureSetting.On ? '+' : '-';
    }

    /// <summary>
    /// Reads a target string (<c>gfx90c:xnack-</c>, <c>gfx942:sramecc+:xnack-</c>, <c>gfx1100</c>) from the span, with no
    /// string made per part. False for anything else: an empty string, a generic target (<c>gfx9-generic</c>), a feature
    /// other than xnack and sramecc, or one named twice.
    /// </summary>
    public static bool TryParse(ReadOnlySpan<char> target, out GcnIsaVersion version)
    {
        version = default;
        int colon = target.IndexOf(':');
        ReadOnlySpan<char> processor = colon < 0 ? target : target[..colon];
        if (!processor.StartsWith("gfx", StringComparison.Ordinal) || processor.Length < 6)
        {
            return false;
        }

        // The digits after the prefix: the major version in decimal, then one hex digit each for minor and stepping.
        ReadOnlySpan<char> digits = processor[3..];
        if (digits[0] == '0' || !int.TryParse(digits[..^2], NumberStyles.None, CultureInfo.InvariantCulture, out int major)
            || !char.IsAsciiHexDigitLower(digits[^2]) || !char.IsAsciiHexDigitLower(digits[^1]))
        {
            return false;
        }

        var xnack = GcnFeatureSetting.Any;
        var sramecc = GcnFeatureSetting.Any;
        ReadOnlySpan<char> rest = colon < 0 ? [] : target[(colon + 1)..];
        while (colon >= 0)
        {
            int next = rest.IndexOf(':');
            ReadOnlySpan<char> feature = next < 0 ? rest : rest[..next];
            if (feature.Length < 2 || feature[^1] is not ('+' or '-'))
            {
                return false;
            }

            var setting = feature[^1] == '+' ? GcnFeatureSetting.On : GcnFeatureSetting.Off;
            ReadOnlySpan<char> name = feature[..^1];
            if (name.SequenceEqual("xnack") && xnack == GcnFeatureSetting.Any)
            {
                xnack = setting;
            }
            else if (name.SequenceEqual("sramecc") && sramecc == GcnFeatureSetting.Any)
            {
                sramecc = setting;
            }
            else
            {
                return false;
            }

            if (next < 0)
            {
                break;
            }

            rest = rest[(next + 1)..];
        }

        version = new GcnIsaVersion(major, HexValue(digits[^2]), HexValue(digits[^1]), xnack, sramecc);
        return true;

        static int HexValue(char c) => c <= '9' ? c - '0' : c - 'a' + 10;
    }

    /// <summary>Reads a target string (see <see cref="TryParse"/>).</summary>
    /// <exception cref="FormatException">It is not one; the message quotes it.</exception>
    public static GcnIsaVersion Parse(ReadOnlySpan<char> target) => TryParse(target, out var version)
        ? version
        : throw new FormatException($"'{target}' is not a GCN target string (a processor such as the runtime reports, then :xnack± or :sramecc±).");
}

/// <summary>
/// What one family of GCN targets allows a kernel: its register files and how a kernel's use of them is counted.
/// </summary>
/// <param name="Vgprs">Vector registers a lane may address.</param>
/// <param name="VgprGranule">Vector registers are allocated, and counted in the kernel descriptor, in blocks of this many.</param>
/// <param name="Sgprs">Scalar registers a kernel may address, those the family reserves (VCC and the others) excluded.</param>
/// <param name="SgprGranule">Scalar registers are counted in the kernel descriptor in blocks of this many.</param>
/// <param name="MaxWavesPerSimd">The most waves one SIMD holds at once, whatever the registers allow.</param>
public sealed record GcnRegisterFile(int Vgprs, int VgprGranule, int Sgprs, int SgprGranule, int MaxWavesPerSimd);

/// <summary>
/// The kernel descriptor's bits a family decides, which the code object writer combines with the ones every family
/// shares (the segment sizes, the user and system registers, the float modes).
/// </summary>
/// <param name="Rsrc1">Bits of COMPUTE_PGM_RSRC1: the register blocks and the family's mode bits.</param>
/// <param name="Rsrc3">COMPUTE_PGM_RSRC3 (zero for families without one).</param>
/// <param name="CodeProperties">Bits of the kernel code properties, such as the wave-size bit of families that run both.</param>
public readonly record struct GcnDescriptorBits(uint Rsrc1, uint Rsrc3, ushort CodeProperties);

/// <summary>
/// One family of GCN instruction sets (gfx9: GCN 5 and CDNA's first targets; later gfx10, gfx11, gfx12), chosen by the
/// version the device reports (<see cref="GcnIsaVersion"/>), never by its name: its wave sizes, register files, kernel
/// descriptor encoding and the code object's ELF flags. Register one with <see cref="GcnTargets.Register"/>; the
/// assembler and the code object writer look a device's family up there.
/// </summary>
public interface IGcnTarget
{
    /// <summary>The family's name, e.g. "gfx9" (its registry key).</summary>
    string Family { get; }

    /// <summary>The wave sizes the family runs, the default first.</summary>
    IReadOnlyList<int> WaveSizes { get; }

    /// <summary>The instruction that fills the gaps between kernels' code (s_nop 0), as a little-endian dword.</summary>
    uint PaddingWord { get; }

    /// <summary>Whether the family writes code for <paramref name="isa"/>: its version and feature settings.</summary>
    bool Handles(GcnIsaVersion isa);

    /// <summary>The register files of <paramref name="isa"/> for waves of <paramref name="waveSize"/> lanes.</summary>
    GcnRegisterFile Registers(GcnIsaVersion isa, int waveSize);

    /// <summary>
    /// The scalar registers a kernel holds: the <paramref name="sgprs"/> it addresses (the next free one) and those the
    /// family reserves beside them when the kernel uses VCC or flat scratch (and the xnack mask while xnack is not off).
    /// </summary>
    int SgprCount(GcnIsaVersion isa, int sgprs, bool vcc, bool flatScratch);

    /// <summary>
    /// How many waves of a kernel one SIMD holds at once, from its <paramref name="vgprs"/> and its scalar registers
    /// (<paramref name="sgprs"/>, as <see cref="SgprCount"/> gives them).
    /// </summary>
    int WavesPerSimd(GcnIsaVersion isa, int vgprs, int sgprs, int waveSize);

    /// <summary>
    /// The kernel descriptor's family bits for a kernel of <paramref name="vgprs"/> vector registers and
    /// <paramref name="sgprs"/> scalar ones (as <see cref="SgprCount"/> gives them) in waves of <paramref name="waveSize"/>.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">A count exceeds the register file, or the wave size is not the family's.</exception>
    GcnDescriptorBits Descriptor(GcnIsaVersion isa, int vgprs, int sgprs, int waveSize);

    /// <summary>
    /// The code object's ELF header flags for <paramref name="isa"/>: its machine value and the xnack and sramecc bits of
    /// its feature settings, in code object format <paramref name="codeObjectVersion"/>.
    /// </summary>
    /// <exception cref="ArgumentException">The target sets a feature its processor does not have.</exception>
    /// <exception cref="NotSupportedException">The format is not one the family writes.</exception>
    uint ElfFlags(GcnIsaVersion isa, int codeObjectVersion);
}

/// <summary>
/// The GCN target families by name. Built in (Idrak.Gpu): "gfx9" (GCN 5 and CDNA's gfx9 targets without a unified
/// register file). Register another family, or replace one, with <see cref="Register"/>; families an app registers are
/// asked first. A device whose version no family handles is refused, naming the target it reported.
/// </summary>
public static partial class GcnTargets
{
    // The library's families are its defaults (BuiltIn, in src/Idrak.Gpu/Hip/GcnTargets.cs), registered when the table is made.
    private static readonly SlotTable<string, IGcnTarget> Table = BuiltIn();

    /// <summary>The registered family names, the newest first.</summary>
    public static IReadOnlyList<string> Names => Table.Keys;

    /// <summary>The registered families, in the order they are asked.</summary>
    public static IReadOnlyList<IGcnTarget> All => Table.Values;

    /// <summary>Registers <paramref name="target"/> under its family name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IGcnTarget target)
    {
        ArgumentNullException.ThrowIfNull(target);
        ArgumentException.ThrowIfNullOrWhiteSpace(target.Family);
        Table.Register(target.Family, target, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's family of that name (the library's, if any, answers again); false when the app registered none.</summary>
    public static bool Unregister(string family) => Table.Unregister(family);

    /// <summary>The family registered as <paramref name="family"/>, or null.</summary>
    public static IGcnTarget? Find(string family) => Table.Find(family);

    /// <summary>The library's family of that name, whatever an app registered over it; null when the library has none.</summary>
    public static IGcnTarget? Default(string family) => Table.Default(family);

    /// <summary>Who registered the family: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string family) => Table.Origin(family);

    /// <summary>The first registered family that handles <paramref name="isa"/>; false when none does.</summary>
    public static bool TryFor(GcnIsaVersion isa, [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out IGcnTarget? target)
    {
        foreach (var candidate in Table.Values)
        {
            if (candidate.Handles(isa))
            {
                target = candidate;
                return true;
            }
        }

        target = null;
        return false;
    }

    /// <summary>The family that handles <paramref name="isa"/>.</summary>
    /// <exception cref="NotRegisteredException">No registered family does; the message names the target and the families.</exception>
    public static IGcnTarget For(GcnIsaVersion isa) => TryFor(isa, out var target) ? target : throw NotRegistered(isa.ToString(), isa);

    /// <summary>The family that handles the target string a device reports (<c>gfx90c:xnack-</c>).</summary>
    /// <exception cref="FormatException">The string is not a target string.</exception>
    /// <exception cref="NotRegisteredException">No registered family handles it; the message names the reported target.</exception>
    public static IGcnTarget For(string reported)
    {
        ArgumentNullException.ThrowIfNull(reported);
        var isa = GcnIsaVersion.Parse(reported);
        return TryFor(isa, out var target) ? target : throw NotRegistered(reported, isa);
    }

    private static NotRegisteredException NotRegistered(string reported, GcnIsaVersion isa) =>
        new(nameof(GcnTargets), reported, Names,
            $"The GCN target '{reported}' (ISA {isa.Major}.{isa.Minor}.{isa.Stepping}) is not registered: no GCN target family handles it "
            + $"(registered: {string.Join(", ", Names)}); add one with GcnTargets.Register.");
}
