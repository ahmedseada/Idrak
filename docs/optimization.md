# Optimization rules

**Read this before writing or changing code in the library.** It holds the C# performance rules for .NET 10 (each
checked on this repository's SDK, see "Verification" below) and, after them, Idrak's own rules learned from measured
work, with the measurements behind them. A rule here does not replace measuring: every speed or memory change still
comes with before and after numbers (see [CONTRIBUTING.md](../CONTRIBUTING.md)), and with proof that its results did
not change.

## Verification

Checked on 2026-10-05 with the .NET SDK 10.0.112 (runtime 10.0.12, C# 14) on x64 with AVX-512, by compiling and running
every API, language rule and example below (Release build), and the analyzer and failure cases separately:

- **Confirmed by running**: rules 4, 6, 7-14 (the async-shell example, fed lines split across reads and a last line
  without a newline, read them all), 15 (alternate lookups on all five collections, 0 bytes allocated), 16 (0 bytes),
  17-20 (`params ReadOnlySpan<T>`: 0 bytes), 22-29, 35-36 and 38 (both SIMD examples equal the scalar results, also
  with `DOTNET_EnableHWIntrinsic=0`), 45 (`stackalloc` in `finally` is error CS0255), 48, 52 (an AVX-512 intrinsic
  with AVX-512 turned off throws `PlatformNotSupportedException`), 53, 54, 55-59, 62-63, and the table's "tiny
  short-lived arrays" (see its note).
- **Corrected or refined**: rule 42 (the per-instruction-set switch names), rule 63 (the calling thread only, measured), rule 65
  (which analyzers the setting adds), the table's note on stack-allocated arrays Each note sits next to its rule, marked **Checked:**.
- **Not run (guidance, not a testable claim)**: 0-3, 5, 21, 30, 31-34, 37, 39-41, 43-44, 46-47, 49-51, 60-61, 64, 66-69.

---

# C# performance rules: .NET 10 (C# 14), async and I/O heavy

**Scope:** pure C# and the built-in .NET 10 base class library only. No NuGet packages, no external libraries, no extra tools.

**Legend:** CRITICAL = can corrupt data or crash · WARN = can be slower or wrong · NOTE = minor
"~" numbers are rules of thumb. Always measure.

---

## 0. The dependency rule

0. Use only what ships with .NET 10: `System.*` namespaces, the C# compiler, and the analyzers built into the SDK. Never add a NuGet package or an external library for performance work.

---

## 1. Pick the type

1. Sync, read-only, hot path → `ReadOnlySpan<char>`, or `ReadOnlySpan<byte>` for UTF-8.
2. The value must cross an `await` → `ReadOnlyMemory<char>` / `Memory<byte>`.
3. Stored values, dictionary keys, public output → `string`.
4. A span parameter already accepts strings, arrays (implicit in C# 14), `stackalloc` buffers and slices.
5. Add a `string` overload only if you return or store the original string.
6. Numbers: `Span<T>` / `ReadOnlySpan<T>` work the same way for `int`, `double`, etc.

---

## 2. Architecture: async shell, sync core

7. Async methods only `await` and move bytes (`Stream.ReadAsync(Memory<byte>)`).
8. All parsing lives in **sync** methods that take spans.
9. Data can end mid-message → parse the complete messages, then move the leftover bytes to the front of the buffer and read more.
10. Rent the read buffer from `ArrayPool<byte>.Shared` and return it in `finally`.
11. Stay in UTF-8 bytes for I/O: `"abc"u8` literals, `int.TryParse(ReadOnlySpan<byte>, out int)`.
12. Convert bytes to a string once, at the edge: `Encoding.UTF8.GetString(span)`.
13. Span locals compile in async methods (C# 13+), but never across an `await`. Keep span work in sync helpers anyway; it is clearer.
14. Data that arrives as several segments → `ReadOnlySequence<byte>` + `SequenceReader<byte>` (both built in).

```csharp
using System.Buffers;

// Async shell: only awaits and buffer moves
public static async Task ReadLinesAsync(Stream stream, CancellationToken ct)
{
    byte[] buffer = ArrayPool<byte>.Shared.Rent(64 * 1024);
    int filled = 0;
    try
    {
        while (true)
        {
            if (filled == buffer.Length) buffer = Grow(buffer, filled);   // line longer than buffer

            int read = await stream.ReadAsync(buffer.AsMemory(filled), ct);
            if (read == 0) break;
            filled += read;

            int consumed = ProcessLines(buffer.AsSpan(0, filled));        // sync, span-based

            // Move the unfinished line to the front of the buffer
            buffer.AsSpan(consumed, filled - consumed).CopyTo(buffer);
            filled -= consumed;
        }

        if (filled > 0) ParseLine(buffer.AsSpan(0, filled));             // last line without '\n'
    }
    finally
    {
        ArrayPool<byte>.Shared.Return(buffer);
    }
}

static byte[] Grow(byte[] old, int used)
{
    byte[] bigger = ArrayPool<byte>.Shared.Rent(old.Length * 2);
    old.AsSpan(0, used).CopyTo(bigger);
    ArrayPool<byte>.Shared.Return(old);
    return bigger;
}

// Sync core: returns how many bytes were fully consumed
static int ProcessLines(ReadOnlySpan<byte> data)
{
    int consumed = 0;
    int newline;
    while ((newline = data[consumed..].IndexOf((byte)'\n')) >= 0)
    {
        ParseLine(data.Slice(consumed, newline).TrimEnd((byte)'\r'));
        consumed += newline + 1;
    }
    return consumed;
}

// Hot path: UTF-8 bytes, zero allocations. Example line: "42,admin"
static void ParseLine(ReadOnlySpan<byte> line)
{
    int comma = line.IndexOf((byte)',');
    if (comma < 0) return;

    if (!int.TryParse(line[..comma], out int id)) return;
    ReadOnlySpan<byte> role = line[(comma + 1)..];

    if (role.SequenceEqual("admin"u8)) { /* ... */ }
}
```

---

## 3. String tools

15. Dictionary lookup by span, with no string allocated:
    ```csharp
    var lookup = dict.GetAlternateLookup<ReadOnlySpan<char>>();   // create once, reuse it
    lookup.TryGetValue(span, out var value);
    ```
    Works on `Dictionary`, `HashSet`, `ConcurrentDictionary`, `FrozenDictionary`, `FrozenSet`.
16. Split with no allocation: `foreach (Range r in span.Split(',')) { var part = span[r]; }`
17. Many chars → `SearchValues<char>`. Many substrings → `SearchValues<string>`. Keep them in `static readonly` fields.
18. Build strings with `string.Create`, or `stackalloc` + `TryFormat` / `MemoryExtensions.TryWrite`.
19. `params ReadOnlySpan<T>` (C# 13): methods with variable arguments allocate no array.
20. `allows ref struct` (C# 13): generic code can take spans as `T`.
21. Don't use `in string` or `ref string`. A string is already a reference.

---

## 4. Number tools

22. `CollectionsMarshal.AsSpan(list)` → a span over a `List<T>`.
23. `CollectionsMarshal.GetValueRefOrAddDefault(dict, key, out _)` → update a value with ONE hash lookup.
24. `MemoryMarshal.Cast<byte, int>` → read raw bytes as numbers (machine byte order).
25. `BinaryPrimitives.ReadInt32BigEndian` etc. → fixed byte order (files, network).
26. `[InlineArray(N)]` struct → a fixed buffer with no heap use, usable as a span.
27. Generic math (`where T : INumber<T>`) → one method for all number types, no boxing.
28. `value.TryFormat(Span<char> or Span<byte>, out written, format, CultureInfo.InvariantCulture)`.
29. `T.TryParse` on `ReadOnlySpan<char>` or `ReadOnlySpan<byte>`.
30. Money → `decimal`, or `long` cents. NEVER `double`.

---

## 5. SIMD (one instruction processes several numbers)

### Order of choice
31. **First:** built-in methods that are already SIMD: `IndexOf`, `IndexOfAny`, `Contains`, `SequenceEqual`, `SearchValues`, `Base64`, the UTF-8 APIs.
32. **Second:** `Vector<T>` (its width depends on the machine; read `Vector<T>.Count`), or `Vector128<T>` / `Vector256<T>` / `Vector512<T>` (fixed widths).
33. **Last:** platform intrinsics (`Avx2`, `Avx512F`, `AdvSimd`). Always guard them with `.IsSupported`.

### Rules
34. Check `VectorXXX.IsHardwareAccelerated` and keep a scalar fallback. ARM (NEON) accelerates 128-bit vectors only.
35. Loop shape: a vector main loop, then a scalar tail for the leftover items.
36. Branch-free code: compare → mask → `ConditionalSelect` / `ExtractMostSignificantBits`.
37. Layout: a struct of arrays (`float[] X, float[] Y`) beats an array of structs.
38. Integer lanes can overflow, so widen them. Float SIMD sums differ slightly from scalar sums.
39. Use 2–4 accumulators in hot loops (measure it).
40. Fuse operations into one loop. Two separate passes over big arrays cost two trips through memory.
41. `Vector512` is fast only on AVX-512 CPUs, and can lower the clock on some older Intel CPUs. Measure on your target hardware.
42. Test both paths. `DOTNET_EnableHWIntrinsic=0` forces the scalar fallback.
    > **Checked:** `DOTNET_EnableHWIntrinsic=0` makes every `IsHardwareAccelerated` false (and `Vector<int>.Count` 4,
    > a software vector). To turn off one instruction set on .NET 10, the switch names have changed:
    > `DOTNET_EnableAVX512F=0` does nothing any more, `DOTNET_EnableAVX512=0` turns AVX-512 off, `DOTNET_EnableAVX2=0`
    > turns AVX2 (and AVX-512) off. `DOTNET_PreferredVectorBitWidth=256` keeps AVX-512 instructions but makes
    > `Vector512.IsHardwareAccelerated` false. On the AVX-512 machine `Vector<int>.Count` was 8: `Vector<T>` stays 256
    > bits wide by default.

```csharp
using System.Numerics;
using System.Runtime.Intrinsics;

// Sum with Vector256 + scalar tail (int -> long, no overflow)
static long Sum(ReadOnlySpan<int> s)
{
    int i = 0;
    long sum = 0;
    if (Vector256.IsHardwareAccelerated && s.Length >= Vector256<int>.Count)
    {
        var acc = Vector256<long>.Zero;
        for (; i <= s.Length - Vector256<int>.Count; i += Vector256<int>.Count)
        {
            var (lo, hi) = Vector256.Widen(Vector256.Create(s.Slice(i)));
            acc += lo + hi;
        }
        sum = Vector256.Sum(acc);
    }
    for (; i < s.Length; i++) sum += s[i];   // scalar tail
    return sum;
}

// Count values > limit, branch-free
static int CountAbove(ReadOnlySpan<int> s, int limit)
{
    int i = 0, count = 0;
    if (Vector256.IsHardwareAccelerated && s.Length >= Vector256<int>.Count)
    {
        var lim = Vector256.Create(limit);
        for (; i <= s.Length - Vector256<int>.Count; i += Vector256<int>.Count)
        {
            var mask = Vector256.GreaterThan(Vector256.Create(s.Slice(i)), lim);
            count += BitOperations.PopCount(mask.ExtractMostSignificantBits());
        }
    }
    for (; i < s.Length; i++) if (s[i] > limit) count++;
    return count;
}
```

---

## 6. Safety rules

43. **CRITICAL:** Never keep a span or `Memory` that points into a pooled buffer after `Return`. Copy to a string first.
44. **CRITICAL:** Pooled arrays → return them in `finally`, never twice, never use them after returning.
45. **CRITICAL:** `stackalloc` must be small (~1 KB or less). Never put it in a loop, and never use a user-controlled size. A `StackOverflowException` kills the process. It is not allowed in `catch` / `finally`.
46. **CRITICAL:** Never add or remove list items while holding `CollectionsMarshal.AsSpan`.
47. **CRITICAL:** Never change a dictionary while holding the ref from `GetValueRefOrAddDefault`.
48. **CRITICAL:** `MemoryMarshal.Cast` uses the machine byte order (wrong for network/file formats) and drops leftover bytes.
49. **CRITICAL:** Always pass `CultureInfo.InvariantCulture` to parse/format calls for I/O.
50. **CRITICAL:** `ValueTask` → await it once, no `.Result` before it completes, never await it twice.
51. **CRITICAL:** `Unsafe` / `LoadUnsafe` / `GetReference` skip bounds checks. Use them only in measured, tested hot spots.
52. **CRITICAL:** Intrinsics without an `.IsSupported` check throw `PlatformNotSupportedException`.
53. **CRITICAL (C# 14):** `array.Contains(x)` can now bind to the span version. Inside expression trees / `IQueryable`, call `Enumerable.Contains` explicitly.
    > **Checked:** `a => a.Contains(2)` became `a => op_Implicit(a).Contains(2)`; `Compile()` ran it, but
    > `Compile(preferInterpretation: true)` threw `ArgumentException` (a `ReadOnlySpan<int>` violates a generic
    > constraint), as an interpreting query provider would. With `Enumerable.Contains` both work.
54. **CRITICAL (C# 14):** An implicit array → `Span<T>` conversion on a covariant array throws `ArrayTypeMismatchException`. `ReadOnlySpan<T>` is safe.
55. **WARN:** `ArrayPool.Rent(n)` can return a bigger array. Track your own length.
56. **WARN:** Rented arrays contain old data. For secrets, use `Return(arr, clearArray: true)`.
57. **WARN:** `GetAlternateLookup` throws if the comparer can't handle spans (custom comparers). Use `TryGetAlternateLookup` if unsure. There is no built-in UTF-8 key lookup.
58. **WARN:** `TryFormat` returns `false` when the buffer is too small. Always check the result.
59. **WARN:** `ReadOnlyMemory` keeps its WHOLE parent alive (a 10-char slice can hold 50 MB).

---

## 7. When the ORIGINAL wins

| Situation | Use instead | Speed | Memory |
|---|---|---|---|
| You need a string at the end anyway | `string` | same or better | same |
| Small slice of a huge string, kept | `Substring` / `ToString` | similar | much better (parent freed) |
| Small file read once | `File.ReadAllBytes` | similar | fine |
| Line reading that is not hot | `StreamReader.ReadLine` | similar | fine |
| Tiny short-lived arrays | `new T[]` (the JIT may put it on the stack) | equal or faster | better (the pool retains memory) |

> **Checked:** a non-escaping `new int[4]` allocated nothing once its method was compiled with full optimization, but
> its first calls (quick, unoptimized code) allocated 40 bytes each; measure warm, and do not count on it in cold code.

| Situation | Use instead | Speed | Memory |
|---|---|---|---|
| Big or unknown buffer size | `ArrayPool` / `new` | safe | no stack overflow |
| Small loops (~<32 items) | plain loop | equal or faster | same |
| Branchy per-item logic | plain loop | faster | same |
| Scattered data (lists, dictionaries, class objects) | plain loop | SIMD can't load it | same |
| Search / compare | built-in `IndexOf` / `SequenceEqual` | already SIMD | same |
| Method usually truly waits | `Task`, not `ValueTask` | same | same |
| Non-hot formatting | string interpolation | fine | small allocation is OK |
| Big struct passed around a lot | `class` | no copying | — |
| Money | `decimal` / `long` cents | correctness wins | — |

---

## 8. Measure (built-in tools only)

60. Measure in a Release build, without a debugger attached.
61. Warm up first, so the JIT finishes optimizing the code before you time it.
62. Time with `Stopwatch.GetTimestamp()` / `Stopwatch.GetElapsedTime(start)`.
63. Count allocations with `GC.GetAllocatedBytesForCurrentThread()` before and after.
    > **Checked:** it counts the calling thread only. A `Parallel.For` allocating 64 x 100,000 bytes showed 1.4 MB on
    > the calling thread (which runs part of the work) and 6.4 MB in `GC.GetTotalAllocatedBytes(precise: true)`: use
    > the latter around work spread over threads.
64. Re-measure after every runtime upgrade. An old "win" can become a loss.

```csharp
using System.Diagnostics;

static void Measure(string name, Action action, int iterations = 1_000_000)
{
    for (int i = 0; i < 10_000; i++) action();             // warm-up

    long bytes = GC.GetAllocatedBytesForCurrentThread();
    long start = Stopwatch.GetTimestamp();

    for (int i = 0; i < iterations; i++) action();

    TimeSpan time = Stopwatch.GetElapsedTime(start);
    bytes = GC.GetAllocatedBytesForCurrentThread() - bytes;

    Console.WriteLine($"{name}: {time.TotalNanoseconds / iterations:F1} ns/op, {bytes / iterations} B/op");
}
```

---

## 9. Stay fast over time

65. Turn on the performance analyzers built into the SDK (no package needed):
    ```xml
    <PropertyGroup>
      <AnalysisModePerformance>All</AnalysisModePerformance>
    </PropertyGroup>
    ```
    They flag CA1845, CA1846, CA1835, CA1870, CA2014 and CA1865–CA1867.
    > **Checked:** with `AnalysisModePerformance` set to `All`, CA1835, CA1845, CA1846, CA1865, CA1866 and CA1870 were
    > reported (and CA1861). CA2014 (`stackalloc` in a loop) is a reliability rule, reported with or without the
    > setting. CA1867 was not reported by any case tried, also with `AnalysisLevel` `latest-all`.
66. Prefer safe span code. The JIT removes more bounds checks every release.
67. Prefer built-in APIs. They get faster every release, for free.
68. Never hard-code a vector width.
69. Ship on .NET 10 LTS (supported to Nov 2028).

---

---

## 10. Idrak's rules (learned from measured work)

Each came from a measurement in this repository or its samples; the numbers are the evidence, from the reads of
handwritten pages in the CNN samples' MultiLanguageOcr (an RTX 5050 laptop and a 4-core x64 container), October 2026.

70. **No platform dependence.** Every built-in piece works alike on Windows, Linux, macOS and Android, in C#, without
    an outside program (PowerShell, ImageMagick, ffmpeg) or a platform's imaging or codecs. A sample may use such a tool
    as a test case to compare against, never as the library's way of doing a thing. (Reading JPEG through Windows
    imaging cost most of a new page's 238-662 ms and runs only on Windows; a C# codec is the library's answer.)
71. **The same result, proven.** An optimization changes no output: compare before and after on the same inputs (a
    hash of the produced pixels, every box, every token), not by looking. Speed without this is not a change to merge.
72. **Walk memory in its order.** Read a row-major image row by row; when an algorithm runs along columns, keep its
    working arrays column by column. (A text layout summing columns pixel by pixel: 37 ms; summing whole rows: 3.7 ms.
    A sideways-reach pass over row-major arrays in column order: 40-66 ms; the same pass on column-ordered arrays:
    18 ms.)
73. **One pass, not a walk from every element.** Where each element's answer follows from its neighbour's (a reach, a
    run length, a prefix sum), compute it in one sweep instead of searching from every element.
74. **Parse bytes, not strings.** Data files are read as UTF-8 bytes in blocks, numbers parsed from the bytes, values
    stored in the smallest type that holds them. (A CSV of 20,000 x 784 pixels: `ReadLine` + `Split` + `double.TryParse`
    took 1.3-2.2 s and allocated 700 MB; parsing the bytes into one byte per value, 0.3-0.45 s and 32 MB, the same
    values.)
75. **Do work once.** Convert an image to grey once, find a region map once, load a model once per process; share the
    result between the stages that need it.
76. **Measure cold and warm apart.** The first call pays for compiling methods (tiered JIT) and GPU kernels; report it
    separately from the steady speed (a page's first classify: 54 ms, warm: 11 ms; a layout's first call 50 ms, warm
    3.7 ms). Tools that run once per process compile loop methods optimized at once
    (`TieredCompilationQuickJitForLoops` false) or ship ReadyToRun.
77. **Device memory does not grow with use.** A memory pool reuses freed blocks across sizes, reports what it holds
    (in use, cached, peak) and can be trimmed. (After five pages of different batch sizes the CUDA pool held 861 MB
    cached for 8 MB in use; after one page, 245 MB.)
78. **Measure with the library's telemetry,** and keep it free when nobody listens (one static read and a bitwise AND
    per site). What the telemetry cannot show yet is a gap to fill, not a reason to time with ad hoc stopwatches in the
    library.
79. **Spread independent work, then check it paid.** Rows of an image, regions of a batch: `Parallel.For` where the
    pieces are independent and large enough, measured on a machine with few cores too.
