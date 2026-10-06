// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace Idrak.Abstraction.Devices.Cpu;

/// <summary>Which register-tiled float32 product kernel the CPU runs (chosen by the instruction sets it reports).</summary>
internal enum TiledKernel : byte
{
    /// <summary>None: the portable <see cref="System.Numerics.Vector{T}"/> kernel only (no hardware FMA with 256-bit or wider vectors, or 128-bit on ARM64).</summary>
    None,

    /// <summary>8 rows × 32 columns in 16 of the 32 AVX-512 registers.</summary>
    Avx512,

    /// <summary>6 rows × 16 columns in 12 of the 16 AVX2 registers (FMA3).</summary>
    Avx2,

    /// <summary>8 rows × 8 columns in 16 of the 32 ARM64 NEON registers.</summary>
    Neon,
}

/// <summary>
/// Every tiling, blocking and threading choice of the CPU backend, from what the machine reports (<see cref="CpuInfo"/>:
/// threads, vector width and registers, cache sizes) or measured on it. No processor is named; sizes are fractions of the
/// caches or multiples of the vector width and thread count. The two parallel cut-overs (element-wise kernels and
/// products) depend on how fast this machine starts a parallel loop against how fast one core streams, so they are
/// measured the first time they are needed (about two seconds, once per machine), kept per machine, runtime and thread count in the
/// library's cache folder (~/.cache/idrak/cpu, see <see cref="CacheFile"/>), and the cache-size formula is the fallback (IDRAK_AUTOTUNE=0, one thread, or an override).
/// <para>
/// None of these choices changes a result: they decide how work is split and tiled, and every product element is still
/// one FMA chain over k, every element-wise output one operation. The few values that decide a summation order are not
/// tuning but part of what the CPU computes (it is the reference other devices are checked against), so they are the
/// same on every machine and never measured: see "Numerical contract" below.
/// </para>
/// </summary>
internal static class CpuTuning
{
    /// <summary>Measure the parallel cut-overs (default); off (IDRAK_AUTOTUNE=0): the cache-size formulas only.</summary>
    internal static bool Autotune = Environment.GetEnvironmentVariable("IDRAK_AUTOTUNE") is not ("0" or "false");

    /// <summary>Overrides the element-wise parallel cut-over (IDRAK_CPU_PARALLEL_ELEMENTS; tests and benchmarks).</summary>
    internal static int? ParallelElementsOverride = EnvironmentInt("IDRAK_CPU_PARALLEL_ELEMENTS");

    /// <summary>Overrides the product parallel cut-over in m·n·k (IDRAK_CPU_PARALLEL_FLOPS; tests and benchmarks).</summary>
    internal static long? ParallelFlopsOverride = EnvironmentInt("IDRAK_CPU_PARALLEL_FLOPS");

    /// <summary>Forces a tiled product kernel (tests: every kernel the machine can run gives the same results).</summary>
    internal static TiledKernel? TiledKernelOverride;

    /// <summary>
    /// Where measured cut-overs are kept: cpu/tuning.tsv in the library's cache folder (IDRAK_CACHE, or ~/.cache/idrak),
    /// or the file IDRAK_CPU_TUNING_FILE names; null (not kept, measured again by each process) when IDRAK_TUNING_CACHE=0
    /// or the file name is empty.
    /// </summary>
    internal static string? CacheFile = DefaultCacheFile();

    internal static string? DefaultCacheFile()
    {
        if (Environment.GetEnvironmentVariable("IDRAK_TUNING_CACHE") is "0" or "false")
        {
            return null;
        }

        if (Environment.GetEnvironmentVariable("IDRAK_CPU_TUNING_FILE") is { } file)
        {
            return file.Length > 0 ? file : null;
        }

        string root = Environment.GetEnvironmentVariable("IDRAK_CACHE") is { Length: > 0 } cache
            ? cache
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile, Environment.SpecialFolderOption.DoNotVerify), ".cache", "idrak");
        return Path.Combine(root, "cpu", "tuning.tsv");
    }

    /// <summary>
    /// A size counts as faster in parallel only when the parallel run takes at most this share of the sequential time (10%
    /// faster): a split also costs the work around it what a timing of the kernel alone does not show (the pool's workers
    /// woken for little, caches shared), so an early split costs more than a late one.
    /// </summary>
    private const double Margin = 0.9;

    private static readonly ConcurrentDictionary<int, Lazy<Cutovers>> s_measured = new();

    [ThreadStatic]
    private static bool t_measuring;

    /// <summary>The machine every choice is derived from.</summary>
    internal static CpuInfo Machine => CpuInfo.Current;

    /// <summary>The two parallel cut-overs and where they came from.</summary>
    internal readonly record struct Cutovers(int Elements, long Flops, string Source);

    // ---------------------------------------------------------------------------------------------------------------
    // Formulas (each takes the machine so tests can check them for other machines).

    /// <summary>
    /// Element-wise work (≈ element operations) from which kernels split across threads, when not measured: where the
    /// operands of a binary kernel (two inputs, an output, about 16 bytes per element) fill one core's L2, since below that
    /// one core streams them from its own cache faster than a parallel loop starts. 1 MiB L2 → 65,536.
    /// </summary>
    internal static int ParallelElementsFormula(CpuInfo c) => Math.Max(4 * c.VectorFloats, c.L2 / 16);

    /// <summary>
    /// Products (m·n·k) from which row or column blocks run in parallel, when not measured: twice the element-wise cut-over
    /// (a fused multiply-add from registers is about half the cost of an element streamed from memory). 1 MiB L2 → 131,072.
    /// </summary>
    internal static long ParallelFlopsFormula(CpuInfo c) => 2L * ParallelElementsFormula(c);

    /// <summary>
    /// The tiled product's A tile in bytes: a quarter of a core's L2, so the packed rows stay there while the B panels,
    /// the C rows and the prefetched lines take the rest. 1 MiB L2 → 256 KiB.
    /// </summary>
    internal static int TileBytes(CpuInfo c) => Math.Max(4096, c.L2 / 4);

    /// <summary>
    /// Most kernel row blocks in one row tile: the tile's C rows of one B panel, which every row block of the tile updates
    /// in turn, stay in the L1 with the panel's next rows. 32 KiB L1 → 16 (the former constant); scales with the L1.
    /// </summary>
    internal static int MaxRowBlocks(CpuInfo c) => Math.Max(1, c.L1 / 2048);

    /// <summary>
    /// The tile grid of an m × n × k product for a kernel of <paramref name="rows"/> × <paramref name="columns"/> on
    /// <paramref name="threads"/> threads: (rows, columns) of C per tile. Row tiles: as few as keep each packed A tile within
    /// <see cref="TileBytes"/> and at most <see cref="MaxRowBlocks"/> kernel row blocks. Column tiles: the columns split so there are at least two tiles per thread in all and the tile
    /// count a multiple of the threads where at most twice the column tiles allow it (whole rounds, no thread idle in the
    /// last one), whole panels, at least two panels wide. (A grid of exactly one tile per thread, each as large as the caches allow, was
    /// tried and was 15–30% slower for 512³–1024³ on the shared test container: fewer, larger tiles balance worse when
    /// other work takes a core. Sharing the rows evenly among the row tiles, timed in one process against the former grid,
    /// gained nothing at 128³–512³ (2–4% slower); the whole rounds gave 1024³ about 9%.)
    /// </summary>
    internal static (int Rows, int Columns) TileGrid(CpuInfo c, int m, int n, int k, int rows, int columns, int threads)
    {
        int mc = Math.Clamp(TileBytes(c) / Math.Max(1, k * sizeof(float)) / rows, 1, MaxRowBlocks(c)) * rows;
        int rowTiles = (m + mc - 1) / mc;
        int split = Math.Max(1, (2 * threads + rowTiles - 1) / rowTiles);
        // Whole rounds (every thread takes as many tiles) by splitting the columns further, up to twice as many column
        // tiles (each tile packs its A rows again) and while tiles stay two panels wide; otherwise the first split.
        int widest = Math.Max(1, (n + 2 * columns - 1) / (2 * columns));
        for (int more = split; more <= Math.Min(widest, 2 * split); more++)
        {
            if (rowTiles * more % threads == 0)
            {
                split = more;
                break;
            }
        }

        int nc = Math.Max(2 * columns, ((n + split - 1) / split + columns - 1) / columns * columns);
        return (mc, nc);
    }

    /// <summary>
    /// Side of the square tiles a transpose copies: source and destination tiles (2 · side² floats) within half the L1 data
    /// cache, a power of two. 32 KiB L1 → 32; 64 KiB → 32; 128 KiB → 64.
    /// </summary>
    internal static int TransposeTile(CpuInfo c)
    {
        int side = 4;
        while (2 * (side * 2) * (side * 2) * sizeof(float) <= c.L1 / 2)
        {
            side *= 2;
        }

        return side;
    }

    /// <summary>
    /// Fewest columns in a parallel work item of the few-row products (and the int8 / int4 / bfloat16 ones): eight vectors,
    /// so each item's inner loop runs whole vectors and amortizes its broadcasts. 8 floats per vector → 64.
    /// </summary>
    internal static int MinColumns(CpuInfo c) => 8 * c.VectorFloats;

    // ---------------------------------------------------------------------------------------------------------------
    // Numerical contract: which algorithm (and so which summation order or rounding) a shape gets. These are facts of the
    // CPU's results, not of its speed, and the CPU is the reference, so they are the same on every machine; deriving them
    // from the hardware or a timing would make the reference differ between machines. Each one names the two orders it
    // chooses between.

    /// <summary>
    /// Reductions (Σx, Σx²) sum sequentially below this many values; above, in chunks of at least a quarter of it (at most
    /// two per thread), each chunk in double, the chunks in order. The chunks only run on the threads when the measured
    /// cut-over (<see cref="ParallelElements"/>) says so; their boundaries never depend on it.
    /// </summary>
    internal const int ReductionElements = 1 << 16;

    /// <summary>
    /// Rows of a product with a transposed B up to which B is read in place: each output is then a dot product over k
    /// summed in vector lanes; with more rows B is transposed once and each output is one FMA chain over k.
    /// </summary>
    internal const int TransposedInPlace = 16;

    /// <summary>
    /// Rows the packed products stream their weights for, row by row (int4: products added without FMA); with more rows
    /// the weights are expanded once and the register-tiled product (FMA) runs. Also <see cref="BackendCapabilities.FewRows"/>,
    /// which decides the same split for the layers.
    /// </summary>
    internal const int FewRows = 8;

    /// <summary>
    /// Head sizes up to which the decoding-attention kernel runs (<see cref="BackendCapabilities.DecodeAttentionHeadDim"/>);
    /// larger heads take the [t, t] weights (a different summation order). The CPU's kernel takes any head size; the
    /// split stays where it has always been so the reference's results do not change.
    /// </summary>
    internal const int DecodeAttentionHeads = 256;

    /// <summary>
    /// Head sizes up to which the tiled (prompt and training) attention kernel runs
    /// (<see cref="BackendCapabilities.TiledAttentionHeadDim"/>); as <see cref="DecodeAttentionHeads"/>.
    /// </summary>
    internal const int TiledAttentionHeads = 128;

    /// <summary>
    /// The column-split few-row product applies beta · C without FMA (the other paths fuse it), so when beta ≠ 0 the rule
    /// that sends a product there is part of its result: at most this many rows, at least <see cref="ColumnSplitColumns"/>
    /// columns, m·n·k at least <see cref="ColumnSplitWork"/>, more than one thread. With beta = 0 (every forward product)
    /// the paths agree and the choice follows the measured cut-over and the vector width.
    /// </summary>
    internal const int ColumnSplitRows = 4;

    /// <summary>See <see cref="ColumnSplitRows"/>.</summary>
    internal const int ColumnSplitColumns = 128;

    /// <summary>See <see cref="ColumnSplitRows"/>.</summary>
    internal const long ColumnSplitWork = 1L << 17;

    // ---------------------------------------------------------------------------------------------------------------
    // This machine's values.

    /// <summary>See <see cref="ParallelElementsFormula"/>; measured on this machine unless overridden.</summary>
    internal static int ParallelElements => ParallelElementsOverride ?? Current().Elements;

    /// <summary>See <see cref="ParallelFlopsFormula"/>; measured on this machine unless overridden.</summary>
    internal static long ParallelFlops => ParallelFlopsOverride ?? Current().Flops;

    /// <summary>
    /// Whether element-wise work of <paramref name="work"/> element operations splits across threads. Work below the
    /// smallest size a measurement tries never does, and never starts one (small tensors stay cheap).
    /// </summary>
    internal static bool SplitElements(long work) =>
        ComputeResources.AllowParallel && (ParallelElementsOverride is { } forced ? work >= forced : work >= SmallestElements && work >= ParallelElements);

    /// <summary>The products' cut-over in m·n·k for a product of <paramref name="work"/> (no measuring below the smallest size tried).</summary>
    internal static long ProductCutover(long work) =>
        ParallelFlopsOverride is { } forced ? forced : work < SmallestFlops || !ComputeResources.AllowParallel ? long.MaxValue : ParallelFlops;

    // The smallest sizes a measurement tries (1/32 of the formulas): below them nothing is measured or split.
    private static readonly long SmallestElements = Math.Max(4L * CpuInfo.Current.VectorFloats, ParallelElementsFormula(CpuInfo.Current) / 32);
    private static readonly long SmallestFlops = Math.Max(64, ParallelFlopsFormula(CpuInfo.Current) / 32);

    /// <summary>See <see cref="MinColumns(CpuInfo)"/>.</summary>
    internal static readonly int ColumnBlock = MinColumns(CpuInfo.Current);

    /// <summary>See <see cref="TransposeTile(CpuInfo)"/>.</summary>
    internal static readonly int TransposeSide = TransposeTile(CpuInfo.Current);

    /// <summary>
    /// Rows whose sums the register-tiled few-row kernels keep in registers at once: two accumulators per row plus two weight
    /// vectors and a broadcast must fit the vector registers (8 rows need 19: AVX-512 and ARM64 have 32; 4 rows need 11:
    /// AVX2 and SSE have 16).
    /// </summary>
    internal static int RegisterRowsFormula(CpuInfo c) => c.VectorRegisters >= 2 * 8 + 3 ? 8 : 4;

    /// <summary>See <see cref="RegisterRowsFormula"/> (or the override, for tests).</summary>
    internal static int RegisterRows => RegisterRowsOverride ?? s_registerRows;

    private static readonly int s_registerRows = RegisterRowsFormula(CpuInfo.Current);

    /// <summary>Forces the rows per pass of the register-tiled few-row kernels (tests: 4 and 8 give the same results).</summary>
    internal static int? RegisterRowsOverride;

    /// <summary>
    /// Weight rows (k) the register-tiled few-row kernels take per pass before their sums go back to memory: half a cache
    /// line of each input row's values. A pass walks that many weight rows side by side, one stream each, and reloads the
    /// sums once per pass (4 vector loads and stores per row against 2 FMAs per weight row). 64-byte lines → 8 (on the
    /// 4-thread container, 8 was the fastest of 2, 4, 8 and 16 for 1–8 rows; 16 was about twice as slow).
    /// </summary>
    internal static int KChunkFormula(CpuInfo c) => Math.Max(1, c.CacheLine / (2 * sizeof(float)));

    /// <summary>See <see cref="KChunkFormula"/> (or the override, for tests).</summary>
    internal static int KChunk => KChunkOverride ?? s_kChunk;

    private static readonly int s_kChunk = KChunkFormula(CpuInfo.Current);

    /// <summary>Forces the k chunk of the few-row kernels (tests: every chunk gives the same results).</summary>
    internal static int? KChunkOverride = EnvironmentInt("IDRAK_CPU_KCHUNK");

    /// <summary>
    /// Whether the few-row kernels use 512-bit vectors where <see cref="System.Numerics.Vector{T}"/> is narrower (AVX-512
    /// accelerated), or the override (tests).
    /// </summary>
    internal static bool WidePanels => WidePanelsOverride ?? (Vector512.IsHardwareAccelerated && System.Numerics.Vector<float>.Count < Vector512<float>.Count);

    /// <summary>Forces the 512-bit few-row panels on or off (tests; on only where AVX-512 is accelerated).</summary>
    internal static bool? WidePanelsOverride;

    /// <summary>The tiled product kernel the instruction sets allow (best first), or the override.</summary>
    internal static TiledKernel Kernel => TiledKernelOverride ?? s_kernel;

    private static readonly TiledKernel s_kernel =
        Vector512.IsHardwareAccelerated && Avx512F.IsSupported && Fma.IsSupported ? TiledKernel.Avx512
        : Vector256.IsHardwareAccelerated && Avx2.IsSupported && Fma.IsSupported ? TiledKernel.Avx2
        : AdvSimd.Arm64.IsSupported && Vector128.IsHardwareAccelerated ? TiledKernel.Neon
        : TiledKernel.None;

    /// <summary>Rows and columns of C a tiled kernel keeps in registers.</summary>
    internal static (int Rows, int Columns) KernelShape(TiledKernel kernel) => kernel switch
    {
        TiledKernel.Avx512 => (8, 32),
        TiledKernel.Avx2 => (6, 16),
        TiledKernel.Neon => (8, 8),
        _ => (0, 0),
    };

    /// <summary>
    /// Runs the few-row bfloat16 products with the streaming loop instead of the register-tiled kernels (tests: the same
    /// results; also the path where SIMD is not accelerated).
    /// </summary>
    internal static bool StreamingFewRows;

    /// <summary>One line for benchmarks and diagnostics: the machine and every choice made from it.</summary>
    internal static string Describe()
    {
        var c = Machine;
        var cut = ParallelElementsOverride is null && ParallelFlopsOverride is null ? Current() : new Cutovers(ParallelElements, ParallelFlops, "override");
        var (rows, columns) = KernelShape(Kernel);
        return $"CPU tuning: {c}\n"
            + $"  parallel from {cut.Elements:N0} element operations and {cut.Flops:N0} product m·n·k ({cut.Source}; formula {ParallelElementsFormula(c):N0} / {ParallelFlopsFormula(c):N0})\n"
            + $"  tiled kernel {Kernel}{(rows > 0 ? $" {rows}x{columns}" : "")}, A tile {TileBytes(c) >> 10} KiB (at most {MaxRowBlocks(c)} row blocks), ≥ 2 tiles per thread in whole rounds; "
            + $"column block {ColumnBlock}; B read in place up to {TransposedInPlace} rows; few rows {FewRows} ({RegisterRows} in registers"
            + $"{(WidePanels ? ", 512-bit panels" : "")}); transpose tile {TransposeSide}";
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Measuring.

    /// <summary>Forgets the measured cut-overs of this process (tests).</summary>
    internal static void Forget()
    {
        s_measured.Clear();
        s_last = null;
    }

    private static Cutovers Formulas(string source) => new(ParallelElementsFormula(Machine), ParallelFlopsFormula(Machine), source);

    /// <summary>This machine's cut-overs and where they came from (measured, read back, or a formula).</summary>
    internal static Cutovers CurrentCutovers => Current();

    private static Cutovers Current()
    {
        if (!ComputeResources.AllowParallel)
        {
            return Formulas("one thread");
        }

        if (!Autotune || t_measuring)
        {
            return Formulas("formula");
        }

        int threads = ComputeResources.MaxCpuThreads;
        if (s_last is { } last && last.Threads == threads)
        {
            return last.Value;                                               // every kernel call: one read, no lookup
        }

        var value = s_measured.GetOrAdd(threads, t => new Lazy<Cutovers>(() => LoadOrMeasure(t))).Value;
        s_last = new Known(threads, value);
        return value;
    }

    private sealed record Known(int Threads, Cutovers Value);

    private static volatile Known? s_last;

    /// <summary>The cache file's key for this machine, runtime and thread count: what the machine reports, nothing else.</summary>
    internal static string CacheKey(int threads)
    {
        var c = Machine;
        return $"v3|{c.Model}|{c.InstructionSets}|{c.LogicalProcessors}/{c.PhysicalCores}|L1 {c.L1}|L2 {c.L2}|L3 {c.L3}|.NET {Environment.Version}|{threads} threads|power {PowerSource.Current}";
    }

    private static Cutovers LoadOrMeasure(int threads)
    {
        string key = CacheKey(threads);
        if (Load(key) is { } known)
        {
            return known;
        }

        var measured = Measure(threads);
        if (measured.Source == Measured)
        {
            Save(key, measured);                                             // an unsteady measurement is not kept
        }

        return measured;
    }

    /// <summary>Reads the kept cut-overs for <paramref name="key"/> (null when absent or unreadable).</summary>
    internal static Cutovers? Load(string key)
    {
        try
        {
            if (string.IsNullOrEmpty(CacheFile) || !File.Exists(CacheFile))
            {
                return null;
            }

            foreach (var line in File.ReadLines(CacheFile))
            {
                var parts = line.Split('\t');
                if (parts.Length == 3 && parts[0] == key
                    && int.TryParse(parts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int elements) && elements > 0
                    && long.TryParse(parts[2], NumberStyles.Integer, CultureInfo.InvariantCulture, out long flops) && flops > 0)
                {
                    return new Cutovers(elements, flops, "measured earlier on this machine");
                }
            }
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
        }

        return null;
    }

    /// <summary>Keeps the cut-overs for <paramref name="key"/> (replacing an older line for it); failures are ignored.</summary>
    internal static void Save(string key, Cutovers cutovers)
    {
        try
        {
            if (string.IsNullOrEmpty(CacheFile))
            {
                return;
            }

            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(CacheFile))!);
            var lines = File.Exists(CacheFile) ? File.ReadAllLines(CacheFile).Where(l => !l.StartsWith(key + "\t", StringComparison.Ordinal)).ToList() : [];
            lines.Add(string.Create(CultureInfo.InvariantCulture, $"{key}\t{cutovers.Elements}\t{cutovers.Flops}"));
            string temporary = CacheFile + "." + Environment.ProcessId + ".tmp";
            File.WriteAllLines(temporary, lines);
            File.Move(temporary, CacheFile, overwrite: true);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or NotSupportedException or ArgumentException)
        {
        }
    }

    private const string Measured = "measured";

    // Both cut-overs on this machine with `threads` threads: for sizes from 1/32 to 32 times the formula's (powers of two;
    // for products one row against a square B, see MeasureFlops), the kernel runs on one thread and split as it would be
    // at that cut-over, in turns over seven rounds, and the median of each is compared. The cut-over is the smallest size
    // at which the parallel run is at least 10% faster there and at the next size (see FirstFaster); the formula when no
    // size is. The whole measurement runs three times and the median is kept: other work on a shared machine moves a
    // single pass by a power of two or two either way (32,768 to 524,288 on the 4-thread test container, whose
    // element-wise work splits well from 65,536–131,072), and both tails cost (too early made the work around it 2×
    // slower, too late leaves small kernels on one core); passes more than 16× apart give the formula, not kept, and the
    // next process measures again. A pass before them is not timed (thread pool start-up, unoptimized code). About two
    // seconds in all, once per machine, runtime and thread count.
    private static Cutovers Measure(int threads)
    {
        t_measuring = true;
        try
        {
            // A first pass, not kept: the thread pool starts its workers on demand and the kernels start in unoptimized
            // code, which a first timed pass would count as the cost of every parallel run (cut-overs set too high).
            MeasureElements(threads);
            MeasureFlops(threads);
            long[] e = [MeasureElements(threads), MeasureElements(threads), MeasureElements(threads)];
            long[] f = [MeasureFlops(threads), MeasureFlops(threads), MeasureFlops(threads)];
            Array.Sort(e);
            Array.Sort(f);
            static bool Steady(long[] v) => v[2] <= 16 * v[0];
            return Steady(e) && Steady(f)
                ? new Cutovers((int)e[1], f[1], Measured)
                : Formulas($"formula (measurements unsteady: {string.Join(" / ", e.Select(x => x.ToString("N0", CultureInfo.InvariantCulture)))}, "
                    + $"{string.Join(" / ", f.Select(x => x.ToString("N0", CultureInfo.InvariantCulture)))})");
        }
        finally
        {
            t_measuring = false;
        }
    }

    private static int MeasureElements(int threads)
    {
        int formula = ParallelElementsFormula(Machine);
        var sizes = new List<int>();
        for (long n = Math.Max(4L * Machine.VectorFloats, formula / 32); n <= (long)formula * 32 && n <= int.MaxValue / 2; n *= 2)
        {
            sizes.Add((int)n);
        }

        var x = new float[sizes[^1]];
        var y = new float[sizes[^1]];
        x.AsSpan().Fill(1e-7f);
        var options = new ParallelOptions { MaxDegreeOfParallelism = threads };
        // Each repeat takes the next n-element window of the buffers (formula × 32 elements each, several times a core's
        // L2), as a kernel meets tensors another kernel just wrote: the same window again and again would sit in one
        // core's L2 and favour the sequential run, which real element-wise work (fresh outputs) does not.
        bool Faster(int n)
        {
            int chunks = Math.Min(threads * 2, 4);                       // as CpuBackend.Run splits n at a cut-over of n
            int size = (n + chunks - 1) / chunks;
            int repeats = Math.Max(1, formula * 64 / n);
            int windows = Math.Max(1, x.Length / n);
            double sequential = Median(() =>
            {
                for (int r = 0; r < repeats; r++)
                {
                    int o = r % windows * n;
                    new CpuBackend.AxpyLoop(x, y, 1f).Execute(o, o + n);
                }
            }, () =>
            {
                for (int r = 0; r < repeats; r++)
                {
                    int o = r % windows * n;
                    var kernel = new CpuBackend.AxpyLoop(x, y, 1f);
                    Parallel.For(0, chunks, options, c => kernel.Execute(o + c * size, o + Math.Min(n, (c + 1) * size)));
                }
            }, out double parallel);
            return parallel < sequential * Margin;
        }

        return (int)FirstFaster(sizes.Select(n => (long)n).ToList(), n => Faster((int)n), formula);
    }

    // The product's cut-over is timed on one row against a square B (m = 1, n = k = √size): the column split spreads that
    // shape evenly over the threads, so the timing shows what starting the threads costs against the work, which is what
    // the cut-over decides. (Cubes of the same sizes are a poor probe: at a few hundred thousand m·n·k the tiled grid has
    // one or two tiles, so a parallel run gains little there whatever the machine.) It is also the decoding shape, the
    // product that runs once per token.
    private static long MeasureFlops(int threads)
    {
        long formula = ParallelFlopsFormula(Machine);
        var sides = new List<long>();
        for (double f = Math.Max(64, formula / 32.0); f <= formula * 32.0; f *= 2)
        {
            long edge = (long)Math.Round(Math.Sqrt(f));
            if (sides.Count == 0 || edge > sides[^1])
            {
                sides.Add(edge);
            }
        }

        int largest = (int)sides[^1];
        var random = new Random(1);
        var a = new float[largest];
        var b = new float[largest * largest];
        var c = new float[largest];
        for (int i = 0; i < a.Length; i++)
        {
            a[i] = random.NextSingle() - 0.5f;
        }

        for (int i = 0; i < b.Length; i++)
        {
            b[i] = random.NextSingle() - 0.5f;
        }

        var options = new ParallelOptions { MaxDegreeOfParallelism = threads };
        bool Faster(long side)
        {
            int s = (int)side;
            int repeats = (int)Math.Max(1, formula * 4 / (side * side));
            double sequential = Median(() =>
            {
                for (int r = 0; r < repeats; r++)
                {
                    CpuMatMul.Multiply(a, 0, b, 0, c, 0, 1, s, s, false, false, 0f, long.MaxValue, options);
                }
            }, () =>
            {
                for (int r = 0; r < repeats; r++)
                {
                    CpuMatMul.Multiply(a, 0, b, 0, c, 0, 1, s, s, false, false, 0f, 0, options);
                }
            }, out double parallel);
            return parallel < sequential * Margin;
        }

        long side = FirstFaster(sides, Faster, -1);
        return side < 0 ? formula : side * side;
    }

    // The smallest size at which `faster` holds there and at the next size (two in a row, so one lucky timing does not
    // decide; the largest size alone when it is the only one), walking up from the smallest; `fallback` when none does.
    // Walking up, not down: past the caches a memory-bound kernel can be as fast on one core as on all of them (the
    // bandwidth is shared), which says nothing about the sizes below, where every core streams its own cache.
    private static long FirstFaster(List<long> sizes, Func<long, bool> faster, long fallback)
    {
        bool previous = false;
        for (int i = 0; i < sizes.Count; i++)
        {
            bool now = faster(sizes[i]);
            if (now && previous)
            {
                return sizes[i - 1];
            }

            if (now && i == sizes.Count - 1)
            {
                return sizes[i];
            }

            previous = now;
        }

        return fallback;
    }

    // Median milliseconds of `first` and `second` over seven rounds in turns, after one run of each outside the timing.
    private static double Median(Action first, Action second, out double secondMedian)
    {
        first();
        second();
        const int Rounds = 7;
        var a = new double[Rounds];
        var b = new double[Rounds];
        for (int r = 0; r < Rounds; r++)
        {
            long t0 = Stopwatch.GetTimestamp();
            first();
            long t1 = Stopwatch.GetTimestamp();
            second();
            long t2 = Stopwatch.GetTimestamp();
            a[r] = t1 - t0;
            b[r] = t2 - t1;
        }

        Array.Sort(a);
        Array.Sort(b);
        secondMedian = b[Rounds / 2];
        return a[Rounds / 2];
    }

    private static int? EnvironmentInt(string name) =>
        int.TryParse(Environment.GetEnvironmentVariable(name), NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) && value > 0 ? value : null;
}
