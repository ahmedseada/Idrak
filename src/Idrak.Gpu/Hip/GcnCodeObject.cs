// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Globalization;
using System.Numerics;
using System.Text;
using Idrak.Gpu.Abstractions;

namespace Idrak.Gpu.Hip;

/// <summary>What a kernel argument is, as the code object's metadata names it (<c>.value_kind</c>).</summary>
internal enum GcnArgumentKind
{
    /// <summary>A pointer to device memory (global_buffer).</summary>
    GlobalBuffer,

    /// <summary>A value copied into the kernel arguments (by_value).</summary>
    ByValue,
}

/// <summary>The memory a pointer argument points into (<c>.address_space</c>); none for a value.</summary>
internal enum GcnAddressSpace
{
    None,
    Global,
    Constant,
}

/// <summary>
/// One kernel argument: where it sits in the kernel arguments, how many bytes, what it is, and its name for the metadata
/// (none: no <c>.name</c> written).
/// </summary>
internal readonly record struct GcnArgument(int Offset, int Size, GcnArgumentKind Kind, GcnAddressSpace AddressSpace, string? Name = null)
{
    /// <summary>A pointer to global memory at <paramref name="offset"/>.</summary>
    public static GcnArgument Buffer(int offset, string? name = null) => new(offset, 8, GcnArgumentKind.GlobalBuffer, GcnAddressSpace.Global, name);

    /// <summary>A value of <paramref name="size"/> bytes at <paramref name="offset"/>.</summary>
    public static GcnArgument Value(int offset, int size, string? name = null) => new(offset, size, GcnArgumentKind.ByValue, GcnAddressSpace.None, name);
}

/// <summary>
/// The user SGPRs the hardware loads before a kernel starts, in this order from s0 (each value is its bit in the kernel
/// code properties): the private segment buffer (4 registers), the dispatch packet, queue and kernel argument pointers
/// (2 each), the dispatch id (2), flat scratch's init (2) and the private segment size (1).
/// </summary>
[Flags]
internal enum GcnUserSgprs : ushort
{
    None = 0,
    PrivateSegmentBuffer = 1 << 0,
    DispatchPtr = 1 << 1,
    QueuePtr = 1 << 2,
    KernargSegmentPtr = 1 << 3,
    DispatchId = 1 << 4,
    FlatScratchInit = 1 << 5,
    PrivateSegmentSize = 1 << 6,
}

/// <summary>
/// One kernel for <see cref="GcnCodeObject.Write(IGcnTarget, GcnIsaVersion, int, ReadOnlySpan{GcnKernelCode})"/>: its
/// machine code and what the kernel descriptor and the metadata say about it. The defaults are those of LLVM's
/// <c>.amdhsa_kernel</c> directives, except float32 denormals, kept as clang keeps them (the CPU's results).
/// </summary>
internal sealed record GcnKernelCode
{
    /// <summary>The kernel's name: its code symbol, and its descriptor's with ".kd" after it.</summary>
    public required string Name { get; init; }

    /// <summary>The machine code, whole dwords.</summary>
    public required ReadOnlyMemory<byte> Code { get; init; }

    /// <summary>Vector registers it uses (the next free one).</summary>
    public required int Vgprs { get; init; }

    /// <summary>Scalar registers it addresses (the next free one), without those the family reserves (VCC and the others).</summary>
    public required int Sgprs { get; init; }

    /// <summary>Whether it uses VCC (its family then reserves the registers for it).</summary>
    public bool UsesVcc { get; init; }

    /// <summary>Whether it uses flat scratch.</summary>
    public bool UsesFlatScratch { get; init; }

    /// <summary>The kernel arguments, by offset.</summary>
    public IReadOnlyList<GcnArgument> Arguments { get; init; } = [];

    /// <summary>Bytes of kernel arguments; null: up to the end of the last argument.</summary>
    public int? KernargBytes { get; init; }

    /// <summary>Shared memory (LDS) the kernel declares, in bytes (the group segment's fixed size).</summary>
    public int LdsBytes { get; init; }

    /// <summary>Scratch memory per lane, in bytes; above zero the private segment's wave offset is loaded too.</summary>
    public int PrivateBytes { get; init; }

    /// <summary>The user SGPRs it enables.</summary>
    public GcnUserSgprs UserSgprs { get; init; } = GcnUserSgprs.KernargSegmentPtr;

    /// <summary>How many workgroup ids the hardware loads after the user SGPRs (0 to 3: x, then y, then z).</summary>
    public int WorkgroupIds { get; init; } = 1;

    /// <summary>How many work-item ids the hardware loads into v0, v1, v2 (1 to 3).</summary>
    public int WorkitemIds { get; init; } = 1;

    /// <summary>Lanes per wave.</summary>
    public int WaveSize { get; init; } = 64;

    /// <summary>The most work-items a workgroup of this kernel may have.</summary>
    public int MaxFlatWorkgroupSize { get; init; } = 256;

    /// <summary>The only workgroup shape the kernel runs with (<c>.reqd_workgroup_size</c>); null: any up to the most.</summary>
    public (int X, int Y, int Z)? RequiredWorkgroupSize { get; init; }

    /// <summary>Scalar registers spilled to memory (<c>.sgpr_spill_count</c>).</summary>
    public int SgprSpills { get; init; }

    /// <summary>Vector registers spilled to memory (<c>.vgpr_spill_count</c>).</summary>
    public int VgprSpills { get; init; }

    /// <summary>FLOAT_DENORM_MODE_32 (0 flushes float32 denormals in and out, 3 keeps them).</summary>
    public int Float32Denormals { get; init; } = 3;

    /// <summary>FLOAT_DENORM_MODE_16_64 (the same for float16 and float64).</summary>
    public int Float16And64Denormals { get; init; } = 3;
}

/// <summary>
/// Writes the ELF code object the HIP runtime loads (hipModuleLoadData) from kernels' machine code, as LLVM's
/// <c>ld.lld -shared</c> lays one out: the note with the metadata (MessagePack), the dynamic symbols, their hash table and
/// names, the kernel descriptors (one 64-byte descriptor a kernel, in .rodata), the code (each kernel 256-byte aligned)
/// and the dynamic table, in three loadable segments (read; read and execute; read and write), then the static symbols
/// and the section headers. Formats 4 and 5; nothing outside .NET is called. The image is one array of its exact size.
/// </summary>
internal static partial class GcnCodeObject
{
    /// <summary>The newest format written, and the default.</summary>
    internal const int NewestFormat = 5;

    /// <summary>The formats written, newest first.</summary>
    internal static ReadOnlySpan<int> Formats => [5, 4];

    /// <summary>
    /// The formats to try, in order, for the setting IDRAK_HIP_CODE_OBJECT (<paramref name="setting"/>): unset, empty,
    /// "auto" or "default", every format written, newest first (a runtime older than format 5 takes 4); a version number,
    /// that one alone.
    /// </summary>
    /// <exception cref="NotSupportedException">Format 6, or another version this writer does not write.</exception>
    /// <exception cref="ArgumentException">Not a version or "auto"; the message names the value.</exception>
    internal static int[] FormatsFor(string? setting)
    {
        var text = setting.AsSpan().Trim();
        if (text.IsEmpty || text.Equals("auto", StringComparison.OrdinalIgnoreCase) || text.Equals("default", StringComparison.OrdinalIgnoreCase))
        {
            return Formats.ToArray();
        }

        if (!int.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out int version))
        {
            throw new ArgumentException($"IDRAK_HIP_CODE_OBJECT={setting}: give a code object version (5 or 4) or 'auto'.", nameof(setting));
        }

        _ = AbiVersion(version);                                         // refuses a format not written, naming it
        return [version];
    }

    /// <summary>The formats to try, in order, as the environment's IDRAK_HIP_CODE_OBJECT asks (see <see cref="FormatsFor"/>).</summary>
    internal static int[] FormatsFromEnvironment() => FormatsFor(Environment.GetEnvironmentVariable("IDRAK_HIP_CODE_OBJECT"));

    private const int Page = 0x1000, HeaderBytes = 64, ProgramHeaderBytes = 56, SectionHeaderBytes = 64, SymbolBytes = 24, DynamicBytes = 16;
    private const int DescriptorBytes = 64, CodeAlign = 256, ProgramHeaders = 8, SectionCount = 11, DynamicEntries = 6;

    // ELF values (the System V ABI and LLVM's AMDGPU additions).
    private const byte ElfClass64 = 2, ElfData2Lsb = 1, ElfVersion = 1, OsAbiAmdgpuHsa = 64;
    private const ushort ElfTypeShared = 3, MachineAmdgpu = 224;
    private const uint PtLoad = 1, PtDynamic = 2, PtNote = 4, PtPhdr = 6, PtGnuStack = 0x6474E551, PtGnuRelro = 0x6474E552;
    private const uint PfX = 1, PfW = 2, PfR = 4;
    private const uint ShtProgbits = 1, ShtSymtab = 2, ShtStrtab = 3, ShtHash = 5, ShtDynamic = 6, ShtNote = 7, ShtDynsym = 11;
    private const ulong ShfWrite = 1, ShfAlloc = 2, ShfExecInstr = 4;
    private const long DtNull = 0, DtHash = 4, DtStrtab = 5, DtSymtab = 6, DtStrsz = 10, DtSyment = 11;
    private const byte SttNotype = 0, SttObject = 1, SttFunc = 2, StbLocal = 0, StbGlobal = 1, StvDefault = 0, StvHidden = 2, StvProtected = 3;
    private const uint NtAmdgpuMetadata = 32;

    // The section indices, in the order of the section headers.
    private const ushort NoteSection = 1, DynsymSection = 2, HashSection = 3, DynstrSection = 4, RodataSection = 5, TextSection = 6, DynamicSection = 7, SymtabSection = 8,
        ShstrtabSection = 9, StrtabSection = 10;

    private static ReadOnlySpan<byte> SectionNames => "\0.note\0.dynsym\0.hash\0.dynstr\0.rodata\0.text\0.dynamic\0.symtab\0.shstrtab\0.strtab\0"u8;

    private static ReadOnlySpan<byte> DynamicName => "_DYNAMIC\0"u8;

    /// <summary>The code object of <paramref name="kernels"/> for <paramref name="isa"/>, through the family that handles it (<see cref="GcnTargets.For(GcnIsaVersion)"/>).</summary>
    internal static byte[] Write(GcnIsaVersion isa, int format, params ReadOnlySpan<GcnKernelCode> kernels) => Write(GcnTargets.For(isa), isa, format, kernels);

    /// <summary>The code object of <paramref name="kernels"/> for <paramref name="isa"/> in format <paramref name="format"/> (4 or 5), encoded by <paramref name="target"/>.</summary>
    /// <exception cref="NotSupportedException">Format 6 or another this writer does not write.</exception>
    /// <exception cref="ArgumentException">No kernels, a kernel the family cannot describe, or a target the family does not handle.</exception>
    internal static byte[] Write(IGcnTarget target, GcnIsaVersion isa, int format, ReadOnlySpan<GcnKernelCode> kernels)
    {
        ArgumentNullException.ThrowIfNull(target);
        byte abiVersion = AbiVersion(format);
        if (kernels.IsEmpty)
        {
            throw new ArgumentException("A code object holds at least one kernel.", nameof(kernels));
        }

        if (!target.Handles(isa))
        {
            throw new ArgumentException($"The GCN family {target.Family} does not handle '{isa}'.", nameof(isa));
        }

        uint flags = target.ElfFlags(isa, format);
        int count = kernels.Length;
        var kernelSgprs = new int[count];
        var family = new GcnDescriptorBits[count];
        int nameBytes = 0;
        for (int i = 0; i < count; i++)
        {
            var kernel = kernels[i];
            Check(kernel, target, kernels[..i]);
            kernelSgprs[i] = target.SgprCount(isa, kernel.Sgprs, kernel.UsesVcc, kernel.UsesFlatScratch);
            family[i] = target.Descriptor(isa, kernel.Vgprs, kernelSgprs[i], kernel.WaveSize);
            nameBytes += Encoding.UTF8.GetByteCount(kernel.Name);
        }

        string targetId = $"{GcnTargets.Triple}--{isa}";
        var measure = new MessagePack([]);
        WriteMetadata(ref measure, kernels, kernelSgprs, targetId, format);
        int metadataBytes = measure.Length;

        // The layout: offsets in the file, and addresses (a section's address equals its offset modulo the page).
        int symbols = 1 + 2 * count;                                     // the null symbol, each kernel's code and descriptor
        int note = HeaderBytes + ProgramHeaders * ProgramHeaderBytes;
        int noteBytes = 12 + GcnTargets.NoteOwner.Length + AlignUp(metadataBytes, 4);
        int dynsym = AlignUp(note + noteBytes, 8);
        int hash = AlignUp(dynsym + symbols * SymbolBytes, 4);
        int hashBytes = (2 + 2 * symbols) * 4;                           // as many buckets as symbols, as ld.lld makes
        int dynstr = hash + hashBytes;
        int dynstrBytes = 1 + 2 * nameBytes + count * (1 + 4);           // name\0, name.kd\0
        int rodata = AlignUp(dynstr + dynstrBytes, DescriptorBytes);
        int readEnd = rodata + count * DescriptorBytes;

        var codeAt = new int[count];
        int text = AlignUp(readEnd, CodeAlign), textEnd = text;
        for (int i = 0; i < count; i++)
        {
            codeAt[i] = AlignUp(textEnd, CodeAlign);
            textEnd = codeAt[i] + kernels[i].Code.Length;
        }

        long textAddress = AlignUp(readEnd, Page) + text % Page;
        long textEndAddress = textAddress + (textEnd - text);
        int dynamic = AlignUp(textEnd, 8);
        int dynamicBytes = DynamicEntries * DynamicBytes;
        long dynamicAddress = AlignUp(textEndAddress, Page) + dynamic % Page;
        int symtab = AlignUp(dynamic + dynamicBytes, 8);
        int symtabBytes = (symbols + 1) * SymbolBytes;                   // with _DYNAMIC, as ld.lld writes it
        int shstrtab = symtab + symtabBytes;
        int strtab = shstrtab + SectionNames.Length;
        int strtabBytes = dynstrBytes + DynamicName.Length;
        int sectionHeaders = AlignUp(strtab + strtabBytes, 8);
        var image = new byte[sectionHeaders + SectionCount * SectionHeaderBytes];
        Span<byte> file = image;

        // ELF header.
        file[0] = 0x7F;
        "ELF"u8.CopyTo(file[1..]);
        file[4] = ElfClass64;
        file[5] = ElfData2Lsb;
        file[6] = ElfVersion;
        file[7] = OsAbiAmdgpuHsa;
        file[8] = abiVersion;
        BinaryPrimitives.WriteUInt16LittleEndian(file[16..], ElfTypeShared);
        BinaryPrimitives.WriteUInt16LittleEndian(file[18..], MachineAmdgpu);
        BinaryPrimitives.WriteUInt32LittleEndian(file[20..], ElfVersion);
        BinaryPrimitives.WriteUInt64LittleEndian(file[32..], HeaderBytes);                  // program headers
        BinaryPrimitives.WriteUInt64LittleEndian(file[40..], (ulong)sectionHeaders);
        BinaryPrimitives.WriteUInt32LittleEndian(file[48..], flags);
        BinaryPrimitives.WriteUInt16LittleEndian(file[52..], HeaderBytes);
        BinaryPrimitives.WriteUInt16LittleEndian(file[54..], ProgramHeaderBytes);
        BinaryPrimitives.WriteUInt16LittleEndian(file[56..], ProgramHeaders);
        BinaryPrimitives.WriteUInt16LittleEndian(file[58..], SectionHeaderBytes);
        BinaryPrimitives.WriteUInt16LittleEndian(file[60..], SectionCount);
        BinaryPrimitives.WriteUInt16LittleEndian(file[62..], ShstrtabSection);

        // Program headers: the headers themselves, the three loadable segments, then the dynamic table, its read-only
        // range after relocation, the stack's flags and the note.
        var headers = file[HeaderBytes..];
        Segment(headers, 0, PtPhdr, PfR, HeaderBytes, HeaderBytes, ProgramHeaders * ProgramHeaderBytes, ProgramHeaders * ProgramHeaderBytes, 8);
        Segment(headers, 1, PtLoad, PfR, 0, 0, readEnd, readEnd, Page);
        Segment(headers, 2, PtLoad, PfR | PfX, text, textAddress, textEnd - text, textEnd - text, Page);
        Segment(headers, 3, PtLoad, PfR | PfW, dynamic, dynamicAddress, dynamicBytes, dynamicBytes, Page);
        Segment(headers, 4, PtDynamic, PfR | PfW, dynamic, dynamicAddress, dynamicBytes, dynamicBytes, 8);
        Segment(headers, 5, PtGnuRelro, PfR, dynamic, dynamicAddress, dynamicBytes, dynamicBytes, 1);
        Segment(headers, 6, PtGnuStack, PfR | PfW, 0, 0, 0, 0, 0);
        Segment(headers, 7, PtNote, PfR, note, note, noteBytes, noteBytes, 4);

        // The note: its owner's and description's sizes, its type, its owner's name, then the metadata.
        BinaryPrimitives.WriteUInt32LittleEndian(file[note..], (uint)GcnTargets.NoteOwner.IndexOf((byte)0) + 1);   // with its terminator
        BinaryPrimitives.WriteUInt32LittleEndian(file[(note + 4)..], (uint)metadataBytes);
        BinaryPrimitives.WriteUInt32LittleEndian(file[(note + 8)..], NtAmdgpuMetadata);
        GcnTargets.NoteOwner.CopyTo(file[(note + 12)..]);
        var pack = new MessagePack(file.Slice(note + 12 + GcnTargets.NoteOwner.Length, metadataBytes));
        WriteMetadata(ref pack, kernels, kernelSgprs, targetId, format);

        // Names: each kernel's code symbol, then each descriptor's; the static names add _DYNAMIC after them.
        var nameAt = new int[symbols];
        int at = 1;
        for (int i = 0; i < 2 * count; i++)
        {
            nameAt[1 + i] = at;
            at += Encoding.UTF8.GetBytes(kernels[i % count].Name, file[(dynstr + at)..]);
            if (i >= count)
            {
                ".kd"u8.CopyTo(file[(dynstr + at)..]);
                at += 3;
            }

            at++;                                                        // the terminator (already zero)
        }

        file.Slice(dynstr, dynstrBytes).CopyTo(file[strtab..]);
        DynamicName.CopyTo(file[(strtab + dynstrBytes)..]);

        // Symbols, descriptors and the hash table.
        var dynamicSymbols = file.Slice(dynsym, symbols * SymbolBytes);
        var staticSymbols = file.Slice(symtab, symtabBytes);
        Symbol(staticSymbols, 1, (uint)dynstrBytes, StbLocal, SttNotype, StvHidden, DynamicSection, dynamicAddress, 0);
        long rodataAddress = rodata;                                    // the first segment's addresses are its offsets
        for (int i = 0; i < count; i++)
        {
            var kernel = kernels[i];
            long codeAddress = textAddress + (codeAt[i] - text);
            long descriptorAddress = rodataAddress + i * DescriptorBytes;
            for (int table = 0; table < 2; table++)
            {
                var symbolTable = table == 0 ? dynamicSymbols : staticSymbols;
                int first = table == 0 ? 1 : 2;
                Symbol(symbolTable, first + i, (uint)nameAt[1 + i], StbGlobal, SttFunc, StvProtected, TextSection, codeAddress, kernel.Code.Length);
                Symbol(symbolTable, first + count + i, (uint)nameAt[1 + count + i], StbGlobal, SttObject, StvDefault, RodataSection, descriptorAddress, DescriptorBytes);
            }

            kernel.Code.Span.CopyTo(file[codeAt[i]..]);
            Descriptor(file.Slice(rodata + i * DescriptorBytes, DescriptorBytes), kernel, family[i], codeAddress - descriptorAddress);
        }

        for (int i = 1; i < count; i++)                                // the space between kernels: the family's no-op
        {
            for (int gap = codeAt[i - 1] + kernels[i - 1].Code.Length; gap < codeAt[i]; gap += 4)
            {
                BinaryPrimitives.WriteUInt32LittleEndian(file[gap..], target.PaddingWord);
            }
        }

        HashTable(file.Slice(hash, hashBytes), file.Slice(dynstr, dynstrBytes), nameAt);

        // The dynamic table: where the dynamic symbols, their names and their hash table are.
        var entries = file.Slice(dynamic, dynamicBytes);
        Dynamic(entries, 0, DtSymtab, dynsym);
        Dynamic(entries, 1, DtSyment, SymbolBytes);
        Dynamic(entries, 2, DtStrtab, dynstr);
        Dynamic(entries, 3, DtStrsz, dynstrBytes);
        Dynamic(entries, 4, DtHash, hash);
        Dynamic(entries, 5, DtNull, 0);

        // Section headers.
        SectionNames.CopyTo(file[shstrtab..]);
        var sections = file[sectionHeaders..];
        Section(sections, NoteSection, ".note"u8, ShtNote, ShfAlloc, note, note, noteBytes, 0, 0, 4, 0);
        Section(sections, DynsymSection, ".dynsym"u8, ShtDynsym, ShfAlloc, dynsym, dynsym, symbols * SymbolBytes, DynstrSection, 1, 8, SymbolBytes);
        Section(sections, HashSection, ".hash"u8, ShtHash, ShfAlloc, hash, hash, hashBytes, DynsymSection, 0, 4, 4);
        Section(sections, DynstrSection, ".dynstr"u8, ShtStrtab, ShfAlloc, dynstr, dynstr, dynstrBytes, 0, 0, 1, 0);
        Section(sections, RodataSection, ".rodata"u8, ShtProgbits, ShfAlloc, rodata, rodata, count * DescriptorBytes, 0, 0, DescriptorBytes, 0);
        Section(sections, TextSection, ".text"u8, ShtProgbits, ShfAlloc | ShfExecInstr, text, textAddress, textEnd - text, 0, 0, CodeAlign, 0);
        Section(sections, DynamicSection, ".dynamic"u8, ShtDynamic, ShfAlloc | ShfWrite, dynamic, dynamicAddress, dynamicBytes, DynstrSection, 0, 8, DynamicBytes);
        Section(sections, SymtabSection, ".symtab"u8, ShtSymtab, 0, symtab, 0, symtabBytes, StrtabSection, 2, 8, SymbolBytes);
        Section(sections, ShstrtabSection, ".shstrtab"u8, ShtStrtab, 0, shstrtab, 0, SectionNames.Length, 0, 0, 1, 0);
        Section(sections, StrtabSection, ".strtab"u8, ShtStrtab, 0, strtab, 0, strtabBytes, 0, 0, 1, 0);
        return image;
    }

    /// <summary>The ELF header's ABI version of a code object format (LLVM's ELFABIVERSION_AMDGPU_HSA_V4 and V5).</summary>
    /// <exception cref="NotSupportedException">Format 6 or another not written.</exception>
    internal static byte AbiVersion(int format) => format switch
    {
        4 => 2,
        5 => 3,
        6 => throw new NotSupportedException(
            "GCN code object format 6 is not written: Idrak writes formats 5 and 4 (format 6 adds generic targets and their version to the header). "
            + "Set IDRAK_HIP_CODE_OBJECT to 5 or 4."),
        _ => throw new NotSupportedException($"GCN code object format {format} is not written: Idrak writes formats 5 and 4."),
    };

    // A kernel the writer can describe: its name, code, arguments and settings within what the format and family allow.
    private static void Check(GcnKernelCode kernel, IGcnTarget target, ReadOnlySpan<GcnKernelCode> before)
    {
        ArgumentNullException.ThrowIfNull(kernel);
        string name = kernel.Name;
        if (string.IsNullOrEmpty(name) || name.Contains('\0', StringComparison.Ordinal))
        {
            throw new ArgumentException("A kernel's name is empty or holds a NUL.", nameof(kernel));
        }

        foreach (var other in before)
        {
            if (string.Equals(other.Name, name, StringComparison.Ordinal))
            {
                throw new ArgumentException($"Two kernels are named '{name}'.", nameof(kernel));
            }
        }

        if (kernel.Code.IsEmpty || kernel.Code.Length % 4 != 0)
        {
            throw new ArgumentException($"Kernel '{name}': its code is {kernel.Code.Length} bytes, not whole dwords.", nameof(kernel));
        }

        if (!target.WaveSizes.Contains(kernel.WaveSize))
        {
            throw new ArgumentException($"Kernel '{name}': waves of {kernel.WaveSize} lanes; the {target.Family} family runs {string.Join(" or ", target.WaveSizes)}.", nameof(kernel));
        }

        if (kernel.LdsBytes < 0 || kernel.PrivateBytes < 0 || kernel.WorkgroupIds is < 0 or > 3 || kernel.WorkitemIds is < 1 or > 3
            || kernel.MaxFlatWorkgroupSize is < 1 or > 1024 || kernel.Float32Denormals is < 0 or > 3 || kernel.Float16And64Denormals is < 0 or > 3
            || kernel.SgprSpills < 0 || kernel.VgprSpills < 0
            || kernel.RequiredWorkgroupSize is { } shape && (shape.X < 1 || shape.Y < 1 || shape.Z < 1 || (long)shape.X * shape.Y * shape.Z > kernel.MaxFlatWorkgroupSize))
        {
            throw new ArgumentException($"Kernel '{name}': a segment size, id count, spill count, workgroup size or float mode is out of its field's range.", nameof(kernel));
        }

        int end = 0;
        foreach (var argument in kernel.Arguments)
        {
            int alignment = (int)Math.Min(16u, BitOperations.RoundUpToPowerOf2((uint)Math.Max(1, argument.Size)));
            if (argument.Size <= 0 || argument.Offset < end || argument.Offset % alignment != 0
                || (argument.Kind == GcnArgumentKind.GlobalBuffer) != (argument.AddressSpace != GcnAddressSpace.None) || argument.Kind == GcnArgumentKind.GlobalBuffer && argument.Size != 8)
            {
                throw new ArgumentException($"Kernel '{name}': the argument at {argument.Offset} ({argument.Size} bytes, {argument.Kind}) overlaps the one before, is not aligned "
                                            + "to its size, or is a pointer of another size or without an address space.", nameof(kernel));
            }

            end = argument.Offset + argument.Size;
        }

        if (kernel.KernargBytes < end)
        {
            throw new ArgumentException($"Kernel '{name}': {kernel.KernargBytes} bytes of kernel arguments, but its arguments end at {end}.", nameof(kernel));
        }
    }

    private static int KernargBytes(GcnKernelCode kernel) =>
        kernel.KernargBytes ?? (kernel.Arguments.Count == 0 ? 0 : kernel.Arguments[^1].Offset + kernel.Arguments[^1].Size);

    // The kernel's alignment of its arguments: 8 (what LLVM writes for any kernel), or 16 for an argument of 16 bytes or more.
    private static int KernargAlign(GcnKernelCode kernel)
    {
        foreach (var argument in kernel.Arguments)
        {
            if (argument.Size >= 16)
            {
                return 16;
            }
        }

        return 8;
    }

    /// <summary>
    /// The 64-byte kernel descriptor: the group (LDS), private and kernel-argument sizes, where the code starts from the
    /// descriptor, then COMPUTE_PGM_RSRC3, RSRC1 and RSRC2 and the kernel code properties.
    /// </summary>
    private static void Descriptor(Span<byte> descriptor, GcnKernelCode kernel, GcnDescriptorBits family, long codeOffset)
    {
        int userSgprs = BitOperations.PopCount((uint)(kernel.UserSgprs & GcnUserSgprs.PrivateSegmentBuffer)) * 4
                        + BitOperations.PopCount((uint)(kernel.UserSgprs & ~GcnUserSgprs.PrivateSegmentBuffer & ~GcnUserSgprs.PrivateSegmentSize)) * 2
                        + BitOperations.PopCount((uint)(kernel.UserSgprs & GcnUserSgprs.PrivateSegmentSize));
        uint rsrc1 = family.Rsrc1 | ((uint)kernel.Float32Denormals << 16) | ((uint)kernel.Float16And64Denormals << 18);
        uint rsrc2 = (kernel.PrivateBytes > 0 ? 1u : 0)                // ENABLE_PRIVATE_SEGMENT (the wave's scratch offset)
                     | ((uint)userSgprs << 1)                          // USER_SGPR_COUNT
                     | ((1u << kernel.WorkgroupIds) - 1) << 7         // ENABLE_SGPR_WORKGROUP_ID_X, _Y, _Z
                     | ((uint)(kernel.WorkitemIds - 1) << 11);         // ENABLE_VGPR_WORKITEM_ID
        BinaryPrimitives.WriteUInt32LittleEndian(descriptor, (uint)kernel.LdsBytes);
        BinaryPrimitives.WriteUInt32LittleEndian(descriptor[4..], (uint)kernel.PrivateBytes);
        BinaryPrimitives.WriteUInt32LittleEndian(descriptor[8..], (uint)KernargBytes(kernel));
        BinaryPrimitives.WriteInt64LittleEndian(descriptor[16..], codeOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(descriptor[44..], family.Rsrc3);
        BinaryPrimitives.WriteUInt32LittleEndian(descriptor[48..], rsrc1);
        BinaryPrimitives.WriteUInt32LittleEndian(descriptor[52..], rsrc2);
        BinaryPrimitives.WriteUInt16LittleEndian(descriptor[56..], (ushort)((ushort)kernel.UserSgprs | family.CodeProperties));
    }

    // The System V hash table ld.lld writes: one bucket a symbol, each symbol chained in front of the bucket's earlier ones.
    private static void HashTable(Span<byte> hash, ReadOnlySpan<byte> names, ReadOnlySpan<int> nameAt)
    {
        int symbols = nameAt.Length;
        BinaryPrimitives.WriteUInt32LittleEndian(hash, (uint)symbols);
        BinaryPrimitives.WriteUInt32LittleEndian(hash[4..], (uint)symbols);
        var buckets = hash.Slice(8, symbols * 4);
        var chains = hash.Slice(8 + symbols * 4, symbols * 4);
        for (int i = 1; i < symbols; i++)
        {
            var name = names[nameAt[i]..];
            int bucket = (int)(ElfHash(name[..name.IndexOf((byte)0)]) % (uint)symbols);
            BinaryPrimitives.WriteUInt32LittleEndian(chains[(i * 4)..], BinaryPrimitives.ReadUInt32LittleEndian(buckets[(bucket * 4)..]));
            BinaryPrimitives.WriteUInt32LittleEndian(buckets[(bucket * 4)..], (uint)i);
        }
    }

    /// <summary>The System V ABI's symbol hash.</summary>
    internal static uint ElfHash(ReadOnlySpan<byte> name)
    {
        uint h = 0;
        foreach (byte c in name)
        {
            h = (h << 4) + c;
            uint g = h & 0xF0000000;
            if (g != 0)
            {
                h ^= g >> 24;
            }

            h &= ~g;
        }

        return h;
    }

    private static void Segment(Span<byte> headers, int index, uint type, uint flags, long offset, long address, long fileBytes, long memoryBytes, long align)
    {
        var header = headers.Slice(index * ProgramHeaderBytes, ProgramHeaderBytes);
        BinaryPrimitives.WriteUInt32LittleEndian(header, type);
        BinaryPrimitives.WriteUInt32LittleEndian(header[4..], flags);
        BinaryPrimitives.WriteInt64LittleEndian(header[8..], offset);
        BinaryPrimitives.WriteInt64LittleEndian(header[16..], address);
        BinaryPrimitives.WriteInt64LittleEndian(header[24..], address);   // the physical address, as ld.lld writes it
        BinaryPrimitives.WriteInt64LittleEndian(header[32..], fileBytes);
        BinaryPrimitives.WriteInt64LittleEndian(header[40..], memoryBytes);
        BinaryPrimitives.WriteInt64LittleEndian(header[48..], align);
    }

    private static void Section(Span<byte> headers, int index, ReadOnlySpan<byte> name, uint type, ulong flags, long offset, long address, long bytes, uint link, uint info, long align,
        long entryBytes)
    {
        var header = headers.Slice(index * SectionHeaderBytes, SectionHeaderBytes);
        int nameAt = SectionNames.IndexOf(name);                      // each name is in the table once, after a NUL
        BinaryPrimitives.WriteUInt32LittleEndian(header, (uint)nameAt);
        BinaryPrimitives.WriteUInt32LittleEndian(header[4..], type);
        BinaryPrimitives.WriteUInt64LittleEndian(header[8..], flags);
        BinaryPrimitives.WriteInt64LittleEndian(header[16..], address);
        BinaryPrimitives.WriteInt64LittleEndian(header[24..], offset);
        BinaryPrimitives.WriteInt64LittleEndian(header[32..], bytes);
        BinaryPrimitives.WriteUInt32LittleEndian(header[40..], link);
        BinaryPrimitives.WriteUInt32LittleEndian(header[44..], info);
        BinaryPrimitives.WriteInt64LittleEndian(header[48..], align);
        BinaryPrimitives.WriteInt64LittleEndian(header[56..], entryBytes);
    }

    private static void Symbol(Span<byte> table, int index, uint name, byte binding, byte type, byte visibility, ushort section, long value, long bytes)
    {
        var symbol = table.Slice(index * SymbolBytes, SymbolBytes);
        BinaryPrimitives.WriteUInt32LittleEndian(symbol, name);
        symbol[4] = (byte)((binding << 4) | type);
        symbol[5] = visibility;
        BinaryPrimitives.WriteUInt16LittleEndian(symbol[6..], section);
        BinaryPrimitives.WriteInt64LittleEndian(symbol[8..], value);
        BinaryPrimitives.WriteInt64LittleEndian(symbol[16..], bytes);
    }

    private static void Dynamic(Span<byte> table, int index, long tag, long value)
    {
        BinaryPrimitives.WriteInt64LittleEndian(table[(index * DynamicBytes)..], tag);
        BinaryPrimitives.WriteInt64LittleEndian(table[(index * DynamicBytes + 8)..], value);
    }

    private static int AlignUp(int value, int align) => (value + align - 1) / align * align;

    private static long AlignUp(long value, long align) => (value + align - 1) / align * align;
}
