// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Generation;

/// <summary>
/// A kind of token sampler, registered by its <see cref="IKind.Name"/> in <see cref="TokenSamplers"/>: makes the sampler
/// of one generation from a <see cref="SamplerRequest"/> (<see cref="IKind{TRequest, T}.Create"/>).
/// </summary>
public interface ITokenSamplerKind : IKind<SamplerRequest, ITokenSampler>;

/// <summary>
/// The token samplers generation makes, by name. <see cref="DefaultName"/> is the built-in <see cref="TokenSampler"/>
/// (temperature, top-k, top-p, min-p and penalties, on the device), which Idrak's <c>TextGenerator.CreateSampler</c> makes
/// unless set. Register a sampler kind under <see cref="DefaultName"/> to override it for every generation: the built-in
/// stays behind it until <see cref="Unregister"/>. A sampler keeps state through a generation (its steps, its
/// random stream, the history penalties look at), so under <see cref="SlotPolicy.FallBack"/> it falls back when it is
/// made, not half-way, and
/// <see cref="SlotPolicy.Shadow"/> compares whole generations.
/// </summary>
public static class TokenSamplers
{
    /// <summary>The name of the sampler every generation makes unless told otherwise: "default".</summary>
    public const string DefaultName = "default";

    private static readonly KindTable<ITokenSamplerKind> Registry = BuiltIn();

    private static KindTable<ITokenSamplerKind> BuiltIn()
    {
        var table = new KindTable<ITokenSamplerKind>(nameof(TokenSamplers), "Token sampler", "add it with TokenSamplers.Register", Guard, StringComparer.Ordinal);
        table.RegisterDefault(new BuiltInKind("the device sampler, failing on logits that are not finite (Idrak 0.3)", Version1), version: 1);
        table.RegisterDefault(new BuiltInKind("the device sampler: temperature, top-k, top-p, min-p and penalties; never samples NaN", TokenSampler.Create), version: 2, since: "0.4.0");
        return table;
    }

    // Version 1 of the default (Idrak 0.3): the same sampler, failing on logits that are not finite.
    private static ITokenSampler Version1(SamplerRequest request)
    {
        var sampler = (TokenSampler)TokenSampler.Create(request);
        sampler.RejectNonFinite = true;
        return sampler;
    }

    /// <summary>
    /// Registers <paramref name="kind"/> under its <see cref="IKind.Name"/>, made for each generation. Under
    /// <see cref="DefaultName"/> it overrides the built-in for every generation, which stays behind it
    /// (see <see cref="SetPolicy"/>).
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ITokenSamplerKind kind) => Registry.Register(kind, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's sampler <paramref name="name"/> (<see cref="DefaultName"/> gets the built-in back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered sampler names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Names;

    /// <summary>The sampler kind <paramref name="name"/>; an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name: the message names the registered ones.</exception>
    public static ITokenSamplerKind Get(ReadOnlySpan<char> name) =>
        Registry.TryGet(name, out var kind) ? kind
            : throw new NotSupportedException($"No token sampler '{name}' is registered ({string.Join(", ", Registry.Names)}); add it with TokenSamplers.Register.");

    /// <summary>The sampler for <paramref name="request"/>: the one registered as <see cref="DefaultName"/>.</summary>
    public static ITokenSampler Create(SamplerRequest request) => Get(DefaultName).Create(request);

    /// <summary>The library's sampler kind <paramref name="name"/>, whatever an app registered over it (for an app's sampler to delegate to); null when the library has none.</summary>
    public static ITokenSamplerKind? Default(string name) => Registry.Default(name);

    /// <summary>
    /// Version <paramref name="version"/> of the library's sampler kind <paramref name="name"/>; null when there is none. The
    /// default sampler has two: 1, which fails on logits that are not finite (Idrak 0.3), and 2, the default since 0.4.0,
    /// which never samples NaN and lets +∞ win. Register an older version over the default to get its behavior back:
    /// <c>TokenSamplers.Register(TokenSamplers.Default(TokenSamplers.DefaultName, 1)!)</c>.
    /// </summary>
    public static ITokenSamplerKind? Default(string name, int version) => Registry.Default(name, version);

    /// <summary>The version of the library's sampler <paramref name="name"/> (0 when the library has none).</summary>
    public static int DefaultVersion(string name) => Registry.DefaultVersion(name);

    /// <summary>Who registered the sampler <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>
    /// What happens when the app's sampler <paramref name="name"/> fails to be made (<see cref="SlotPolicy.Throw"/> unless
    /// set: the error reaches the caller; <see cref="SlotPolicy.FallBack"/> retries on the built-in), or whether it only
    /// runs beside the built-in (<see cref="SlotPolicy.Shadow"/>: the built-in
    /// chooses every token; on <paramref name="shadowRate"/> of the generations the app's sampler sees the same logits, and
    /// the tokens each chose, times and allocations are reported when the generation ends).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(name, policy, shadowRate);

    private static ITokenSamplerKind Guard(Slot slot, ITokenSamplerKind app, ITokenSamplerKind library) => new GuardedKind(slot, app, library);

    // The built-in sampler, by version.
    private sealed class BuiltInKind(string summary, Func<SamplerRequest, ITokenSampler> create) : ITokenSamplerKind
    {
        public string Name => DefaultName;

        public string Summary => summary;

        public IReadOnlySet<string> Keys => OptionKeys.None;

        public ITokenSampler Create(SamplerRequest request) => create(request);
    }

    private sealed class GuardedKind(Slot slot, ITokenSamplerKind app, ITokenSamplerKind library) : ITokenSamplerKind
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public IReadOnlySet<string> Keys => app.Keys;

        public ITokenSampler Create(SamplerRequest request)
        {
            if (slot.Policy != SlotPolicy.Shadow)
            {
                return slot.Call(() => app.Create(request), () => library.Create(request));
            }

            var answer = library.Create(request);
            if (!slot.Samples())
            {
                return answer;
            }

            long start = Stopwatch.GetTimestamp(), bytes = GC.GetAllocatedBytesForCurrentThread();
            try
            {
                return new ShadowSampler(slot, answer, app.Create(request));
            }
            catch (Exception e) when (e is not OperationCanceledException)
            {
                slot.ReportComparison(null, TimeSpan.Zero, Stopwatch.GetElapsedTime(start), 0, GC.GetAllocatedBytesForCurrentThread() - bytes, e);
                return answer;
            }
        }
    }

    // The built-in sampler answers; the app's sees the same calls, and the tokens each chose are compared step by step.
    // Reading the chosen ids every step synchronizes the device, so a shadowed generation is not recorded as a graph.
    private sealed class ShadowSampler(Slot slot, ITokenSampler library, ITokenSampler app) : ITokenSampler
    {
        private TimeSpan _libraryTime, _appTime;
        private long _libraryBytes, _appBytes;
        private int _steps, _differed;
        private string? _first;
        private Exception? _error;
        private bool _disposed;

        public int Rows => library.Rows;

        public int Vocabulary => library.Vocabulary;

        public Tensor Ids => library.Ids;

        public bool Recordable => false;

        public void Sample(Tensor logits)
        {
            Measure(() => library.Sample(logits), ref _libraryTime, ref _libraryBytes);
            if (_error is not null)
            {
                return;
            }

            if (Mine(() => app.Sample(logits)))
            {
                var chosen = library.Ids.ToArray();
                var other = app.Ids.ToArray();
                if (Comparisons.Difference(chosen, other, 0f) is { } difference)
                {
                    _differed++;
                    _first ??= $"step {_steps}: {difference}";
                }
            }

            _steps++;
        }

        public void SetHistory(IReadOnlyList<int> tokens)
        {
            library.SetHistory(tokens);
            Mine(() => app.SetHistory(tokens));
        }

        public void Reset()
        {
            library.Reset();
            Mine(() => app.Reset());
        }

        public SampledToken[][] Read(int fromStep, int toStep) => library.Read(fromStep, toStep);

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            slot.ReportComparison(_differed == 0 ? null : $"{_differed} of {_steps} steps chose other tokens (first {_first})",
                _libraryTime, _appTime, _libraryBytes, _appBytes, _error);
            library.Dispose();
            app.Dispose();
        }

        // Runs the app's sampler unless it failed before; false when it throws now (reported at the end).
        private bool Mine(Action call)
        {
            if (_error is not null)
            {
                return false;
            }

            try
            {
                Measure(call, ref _appTime, ref _appBytes);
                return true;
            }
            catch (Exception e) when (e is not OperationCanceledException)
            {
                _error = e;
                return false;
            }
        }

        private static void Measure(Action call, ref TimeSpan time, ref long bytes)
        {
            long start = Stopwatch.GetTimestamp(), before = GC.GetAllocatedBytesForCurrentThread();
            try
            {
                call();
            }
            finally
            {
                time += Stopwatch.GetElapsedTime(start);
                bytes += GC.GetAllocatedBytesForCurrentThread() - before;
            }
        }
    }
}
