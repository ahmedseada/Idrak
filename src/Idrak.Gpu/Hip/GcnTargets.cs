// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Abstractions;

// The library's GCN target families (the registry itself is in src/Idrak.Gpu/Abstractions/IGcnTarget.cs) and what they
// share: the processors' machine values (GcnTargets.MachineValues.cs) and the ELF flags made of them. Each family lives
// in a file of its own (GcnTargets.Gfx9.cs). These files are the only ones that name processors in code.
public static partial class GcnTargets
{
    /// <summary>The code object formats a family writes ELF flags for: 4 and 5 (the same flags; 6 adds generic targets).</summary>
    internal static ReadOnlySpan<int> FlagFormats => [4, 5];

    private static SlotTable<string, IGcnTarget> BuiltIn()
    {
        var table = new SlotTable<string, IGcnTarget>(nameof(GcnTargets), comparer: StringComparer.Ordinal, newestFirst: true,
            unguarded: "a family's encodings are written into code objects whole, with no call to fall back at");
        var gfx9 = new Gfx9();
        table.RegisterDefault(gfx9.Family, gfx9);
        return table;
    }

    /// <summary>The processors whose machine values the library knows, by name.</summary>
    internal static IEnumerable<string> Processors => Machines.Select(m => m.Processor);

    /// <summary>The machine value of <paramref name="isa"/>'s processor and whether it has xnack and sramecc settings; false for an unknown processor.</summary>
    internal static bool TryMachine(GcnIsaVersion isa, out byte machine, out bool xnack, out bool sramecc)
    {
        for (int i = 0; i < MachineVersions.Length; i++)
        {
            var known = MachineVersions[i];
            if (known.Major == isa.Major && known.Minor == isa.Minor && known.Stepping == isa.Stepping)
            {
                (_, machine, xnack, sramecc) = Machines[i];
                return true;
            }
        }

        (machine, xnack, sramecc) = (0, false, false);
        return false;
    }

    /// <summary>
    /// The ELF header flags of code for <paramref name="isa"/> in code object format 4 or 5: its machine value, then the
    /// xnack setting (0x100 any, 0x200 off, 0x300 on) and the sramecc one (0x400 any, 0x800 off, 0xC00 on) where the
    /// processor has them.
    /// </summary>
    /// <exception cref="ArgumentException">The processor is unknown, or the target sets a feature it does not have.</exception>
    /// <exception cref="NotSupportedException">Another format.</exception>
    internal static uint ElfFlagsOf(GcnIsaVersion isa, int codeObjectVersion)
    {
        if (!FlagFormats.Contains(codeObjectVersion))
        {
            throw new NotSupportedException($"GCN code object format {codeObjectVersion} has no ELF flags here (formats {string.Join(", ", FlagFormats.ToArray())} do).");
        }

        if (!TryMachine(isa, out byte machine, out bool xnack, out bool sramecc))
        {
            throw new ArgumentException($"'{isa}' has no known machine value (the processors known are LLVM 18's).", nameof(isa));
        }

        return machine | Feature(isa.Xnack, xnack, 0x100, "xnack") | Feature(isa.Sramecc, sramecc, 0x400, "sramecc");

        uint Feature(GcnFeatureSetting setting, bool present, uint any, string name) => (setting, present) switch
        {
            (_, true) => any * (uint)(setting switch { GcnFeatureSetting.Off => 2, GcnFeatureSetting.On => 3, _ => 1 }),
            (GcnFeatureSetting.Any, false) => 0,
            _ => throw new ArgumentException($"'{isa}' sets {name}, which its processor does not have.", nameof(isa)),
        };
    }
}
