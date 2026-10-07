// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Testing;

/// <summary>A case of a contract check, as data: everything an implementation is given, in JSON, so it can be saved.</summary>
/// <param name="Name">The case's name in reports.</param>
/// <param name="Data">Its inputs.</param>
public sealed record ContractCase(string Name, JsonObject Data);

/// <summary>How <see cref="Conformance.Check{T}(T, ContractSuite{T}, ContractCheckOptions?)"/> runs a suite.</summary>
public sealed record ContractCheckOptions
{
    /// <summary>The seed of the random cases.</summary>
    public int Seed { get; init; } = 1;

    /// <summary>How many random cases run after the fixed ones.</summary>
    public int RandomCases { get; init; } = 20;

    /// <summary>Whether the random cases may be large (long texts, wide vocabularies).</summary>
    public bool Large { get; init; }

    /// <summary>The failing cases reported per property; the others are counted.</summary>
    public int MaxFailures { get; init; } = 3;
}

/// <summary>
/// The checks of one contract (a sampler, a tokenizer, a RoPE scaling, or one of your own): fixed and random cases, as
/// data, and how to run one on an implementation, comparing with the library default where the contract has one. To add
/// a contract, subclass it: <see cref="FixedCases"/>, <see cref="RandomCase"/> and <see cref="Run"/>; then
/// <see cref="Conformance.Check{T}(T, ContractSuite{T}, ContractCheckOptions?)"/>, <see cref="Stress.Run{T}(T, ContractSuite{T}, StressOptions?, CancellationToken)"/>
/// and <see cref="Regression.Replay{T}(string, T, ContractSuite{T})"/> work with it.
/// </summary>
/// <typeparam name="T">The contract: an interface, an abstract class or a delegate.</typeparam>
public abstract class ContractSuite<T>
    where T : class
{
    /// <summary>The contract's name: the kind of its saved cases ("token-sampler").</summary>
    public abstract string Name { get; }

    /// <summary>Where the implementation's tensors live: the device whose memory a stress run watches (the CPU by default).</summary>
    public virtual Device Device => Device.Cpu;

    /// <summary>The fixed cases: edge cases first, then typical ones.</summary>
    public abstract IEnumerable<ContractCase> FixedCases();

    /// <summary>A random case drawn from <paramref name="random"/>; larger inputs when <paramref name="large"/>.</summary>
    public abstract ContractCase RandomCase(Random random, bool large);

    /// <summary>
    /// Runs <paramref name="case"/> on <paramref name="implementation"/> and records each property checked in
    /// <paramref name="checks"/>. An exception is reported as a failure of the case.
    /// </summary>
    public abstract void Run(T implementation, ContractCase @case, CaseChecks checks);

    /// <summary>The fixed cases, then <paramref name="options"/>' random ones.</summary>
    public IEnumerable<ContractCase> Cases(ContractCheckOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        foreach (var @case in FixedCases())
        {
            yield return @case;
        }

        var random = new Random(options.Seed);
        for (int i = 0; i < options.RandomCases; i++)
        {
            yield return RandomCase(new Random(random.Next()), options.Large);
        }
    }

    /// <summary>Runs <paramref name="cases"/> on <paramref name="implementation"/> into a report.</summary>
    public ConformanceReport Check(T implementation, IReadOnlyList<ContractCase> cases, string subject, int maxFailures = 3)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(cases);
        var checks = new CaseChecks(Name, maxFailures);
        foreach (var @case in cases)
        {
            RunOne(implementation, @case, checks);
        }

        return checks.Report($"{Describe(implementation)} ({Name}){(subject.Length > 0 ? ", " + subject : "")}", cases.Count);
    }

    /// <summary>Runs one case; returns its time.</summary>
    internal TimeSpan RunOne(T implementation, ContractCase @case, CaseChecks checks)
    {
        var started = Stopwatch.GetTimestamp();
        checks.Begin(@case);
        try
        {
            Run(implementation, @case, checks);
        }
        catch (Exception ex)
        {
            checks.Fail("runs without throwing", $"threw {ex.GetType().Name}: {ex.Message}");
        }

        return Stopwatch.GetElapsedTime(started);
    }

    /// <summary>How the implementation is named in reports: its type's name, or the method's for a delegate.</summary>
    protected virtual string Describe(T implementation) =>
        implementation is Delegate d ? $"{d.Method.DeclaringType?.Name}.{d.Method.Name}" : implementation.GetType().Name;
}

/// <summary>Collects what each case of a <see cref="ContractSuite{T}"/> found, property by property.</summary>
public sealed class CaseChecks
{
    private readonly string _kind;
    private readonly int _maxFailures;
    private readonly Dictionary<string, (int Passed, int Failed, string? Skipped)> _properties = new(StringComparer.Ordinal);
    private readonly List<string> _order = [];
    private readonly List<ConformanceFailure> _failures = [];
    private readonly AsyncLocal<ContractCase?> _case = new();                // per thread: stress runs cases on several

    internal CaseChecks(string kind, int maxFailures)
    {
        _kind = kind;
        _maxFailures = maxFailures;
    }

    /// <summary>The failures so far.</summary>
    public IReadOnlyList<ConformanceFailure> Failures
    {
        get
        {
            lock (_properties)
            {
                return [.. _failures];
            }
        }
    }

    /// <summary>Records that <paramref name="property"/> holds on the current case.</summary>
    public void Pass(string property) => Record(property, null, null);

    /// <summary>Records that <paramref name="property"/> fails on the current case, with what differs.</summary>
    public void Fail(string property, string message) => Record(property, message ?? "failed", null);

    /// <summary>Records <paramref name="property"/> as holding when <paramref name="holds"/>, else failing with <paramref name="message"/>.</summary>
    public void Check(string property, bool holds, string message) => Record(property, holds ? null : message, null);

    /// <summary>Records <paramref name="property"/> as holding when <paramref name="difference"/> is null (a <see cref="Comparisons"/> result).</summary>
    public void Compare(string property, string? difference) => Record(property, difference, null);

    /// <summary>Records that <paramref name="property"/> was not checked on the current case, and why.</summary>
    public void Skip(string property, string reason) => Record(property, null, reason ?? "skipped");

    internal void Begin(ContractCase @case) => _case.Value = @case;

    internal ConformanceReport Report(string subject, int cases)
    {
        lock (_properties)
        {
            var entries = _order.Select(p => _properties[p] switch
            {
                { Failed: > 0 } r => new CheckEntry(p, CheckStatus.Failed, $"{r.Failed} of {r.Passed + r.Failed} cases fail"),
                { Passed: > 0 } r => new CheckEntry(p, CheckStatus.Passed, $"{r.Passed} cases"),
                var r => new CheckEntry(p, CheckStatus.Skipped, r.Skipped ?? "no case checks it"),
            }).ToList();
            return new ConformanceReport(subject, entries, [.. _failures], cases);
        }
    }

    private void Record(string property, string? problem, string? skipped)
    {
        var @case = _case.Value;
        lock (_properties)
        {
            if (!_properties.TryGetValue(property, out var r))
            {
                _order.Add(property);
            }

            if (skipped is not null)
            {
                _properties[property] = r with { Skipped = r.Skipped ?? skipped };
                return;
            }

            _properties[property] = problem is null ? r with { Passed = r.Passed + 1 } : r with { Failed = r.Failed + 1 };
            if (problem is not null && r.Failed < _maxFailures)
            {
                _failures.Add(new ConformanceFailure(property, @case?.Name ?? "", problem,
                    @case is null ? null : new RegressionCase(_kind, @case.Name, (System.Text.Json.Nodes.JsonObject)@case.Data.DeepClone())));
            }
        }
    }
}
