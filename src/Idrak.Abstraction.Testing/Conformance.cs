// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Operations;
using Idrak.Abstraction.Testing.Devices;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// Checks an implementation against the library default on fixed and random cases, with tolerances, and reports every
/// failing case. Nothing here asserts: each method returns a <see cref="ConformanceReport"/> for any test framework (or
/// none) to check, print and save (<see cref="ConformanceReport.SaveFailures"/>).
/// </summary>
/// <example>
/// <code>
/// var report = Conformance.Check(Device.Get("mydevice"));
/// Console.WriteLine(report);
/// report.ThrowIfFailed();
/// </code>
/// </example>
public static class Conformance
{
    /// <summary>
    /// Checks a device against the CPU, operation by operation: every case of <paramref name="options"/> runs on the CPU
    /// with each operation call recorded, then each call runs on <paramref name="device"/> through its dispatcher
    /// (<c>Backend.Name(...)</c>: a registered kernel, the device's own, a composition or the host fallback) on the same
    /// values, and what it writes must agree within the operation's tolerance. Operations no case reaches, or that have
    /// no kernel on the device (<see cref="Operations.Kernels.Chain"/>), are reported as skipped.
    /// </summary>
    public static ConformanceReport Check(Device device, DeviceCheckOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(device);
        return DeviceConformance.Check(device.Backend, $"{device} ({device.Name})", options ?? new DeviceCheckOptions());
    }

    /// <summary>The same as <see cref="Check(Device, DeviceCheckOptions?)"/> for a backend that is not registered as a device.</summary>
    public static ConformanceReport Check(Backend backend, DeviceCheckOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        return DeviceConformance.Check(backend, $"{backend.Kind} ({backend.Name})", options ?? new DeviceCheckOptions());
    }

    /// <summary>
    /// Checks the kernel a plug-in operation runs on <paramref name="device"/> (<see cref="PluginOperation{TKernel}.KernelFor"/>:
    /// a kernel registered for its kind, or the default) against the operation's default kernel on the CPU, on
    /// <paramref name="cases"/> cases. <paramref name="run"/> makes one case from its seed on the backend it is given
    /// (uploads the inputs, calls the kernel with that backend, downloads the outputs) and returns the outputs; they must
    /// agree within <paramref name="tolerance"/> (<see cref="Comparisons.Close(float, float, float)"/>).
    /// </summary>
    /// <example>
    /// <code>
    /// Conformance.Check(MyOps.ScaleRows, Device.Get("vulkan:0"), (backend, kernel, seed) =&gt;
    /// {
    ///     // inputs from new Random(seed), uploaded to backend; kernel(backend, ...); the outputs downloaded
    /// }).ThrowIfFailed();
    /// </code>
    /// </example>
    public static ConformanceReport Check<TKernel>(PluginOperation<TKernel> operation, Device device, Func<Backend, TKernel, int, float[]> run, int cases = 8, float tolerance = 1e-5f)
        where TKernel : Delegate
    {
        ArgumentNullException.ThrowIfNull(operation);
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(run);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(cases);
        var backend = device.Backend;
        var source = Kernels.Chain(backend)[operation.Index].Source;
        var failures = new List<ConformanceFailure>();
        for (int seed = 0; seed < cases; seed++)
        {
            string? difference;
            try
            {
                var expected = run(Device.Cpu.Backend, operation.DefaultKernel, seed);
                var actual = run(backend, operation.KernelFor(backend), seed);
                difference = Comparisons.Difference(expected, actual, tolerance);
            }
            catch (Exception e) when (e is not OutOfMemoryException)
            {
                difference = $"{e.GetType().Name}: {e.Message}";
            }

            if (difference is not null)
            {
                failures.Add(new ConformanceFailure(operation.Name, $"seed {seed}", difference));
            }
        }

        var entry = new CheckEntry(operation.Name, failures.Count == 0 ? CheckStatus.Passed : CheckStatus.Failed,
            $"the {source.ToString().ToLowerInvariant()} kernel against the default on the CPU, {cases} cases");
        return new ConformanceReport($"{operation.Name} on {device} ({device.Name})", [entry], failures, cases);
    }

    /// <summary>Runs a contract's suite (fixed cases, then random ones) on an implementation.</summary>
    public static ConformanceReport Check<T>(T implementation, ContractSuite<T> suite, ContractCheckOptions? options = null)
        where T : class
    {
        ArgumentNullException.ThrowIfNull(suite);
        options ??= new ContractCheckOptions();
        return suite.Check(implementation, [.. suite.Cases(options)], "", options.MaxFailures);
    }

    /// <summary>
    /// Checks a tokenizer: ids in range, the same ids every time, decoding the same text from a list or a span, text
    /// stable through a round trip, and, given a <paramref name="reference"/> (the library tokenizer it replaces), the
    /// same ids and text as it (<see cref="TokenizerSuite"/>).
    /// </summary>
    public static ConformanceReport Check(ITokenizer tokenizer, ITokenizer? reference = null, ContractCheckOptions? options = null) =>
        Check(tokenizer, new TokenizerSuite(reference), options);

    /// <summary>
    /// Checks the RoPE scaling method registered as <paramref name="type"/> (or <paramref name="method"/>, when given)
    /// against the library's method of that name (<see cref="RopeScalings.Default"/>): as many frequencies, finite and
    /// positive, the same frequencies and attention factor within a tolerance, the same per position, and safe to call
    /// from several threads (<see cref="RopeScalingSuite"/>).
    /// </summary>
    public static ConformanceReport CheckRopeScaling(string type, RopeScalingMethod? method = null, ContractCheckOptions? options = null) =>
        Check(method ?? RopeScalings.Get(type), new RopeScalingSuite(type), options);

    /// <summary>
    /// Checks the kind of message part registered as <paramref name="name"/> (or <paramref name="kind"/>, when given): it
    /// reads parts of its kind, writes JSON that reads back to an equal part, the same every time, and agrees with the
    /// library's kind of that name when there is one (<see cref="ChatPartKindSuite"/>). A kind of your own needs
    /// <paramref name="samples"/>, parts of it as JSON.
    /// </summary>
    public static ConformanceReport CheckChatPartKind(string name, IChatPartKind? kind = null, IReadOnlyList<System.Text.Json.Nodes.JsonObject>? samples = null,
        ContractCheckOptions? options = null) =>
        Check(kind ?? ChatParts.Get(name), new ChatPartKindSuite(name, samples), options);
}
