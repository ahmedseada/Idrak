// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Globalization;
using Idrak.Abstraction.Operations;

namespace Idrak.Abstraction;

/// <summary>
/// One sum type measured for an operation on a device (plan 37, decision 7): the time a call took summing in
/// <paramref name="Type"/>, and its error against the widest sum type the device reports, on the run's own data.
/// </summary>
/// <param name="Type">The sum type.</param>
/// <param name="Seconds">The time of a warm call.</param>
/// <param name="Error">The largest relative error against the widest sum (0 for the widest itself).</param>
public sealed record SumTypeMeasurement(ElementType Type, double Seconds, double Error);

/// <summary>
/// The type an operation sums in on a device (plan 37, decision 7: no fixed sum type). Each backend reports, per
/// operation, the sum types its kernels take (<see cref="Devices.BackendCapabilities.SumTypes"/>, from what the device
/// reports at run time); <see cref="Auto"/> picks the fastest whose error against the widest is within
/// <see cref="Tolerance"/>, from measurements on the device (<see cref="Record"/>: at load or by <c>idrak tune</c>); a
/// run's option or <see cref="EnvironmentVariable"/> selects one instead. The choice says per device and operation which
/// type is used and how (<see cref="Report"/>): "picked by auto" with the measurement, or "selected by the user".
/// </summary>
public static class SumTypes
{
    /// <summary>The variable naming the sum type: "IDRAK_SUM_DTYPE" (auto, or a type the device reports for the operation).</summary>
    public const string EnvironmentVariable = "IDRAK_SUM_DTYPE";

    /// <summary>The variable setting <see cref="Tolerance"/>: "IDRAK_SUM_TOLERANCE" (a relative error, such as 1e-3).</summary>
    public const string ToleranceVariable = "IDRAK_SUM_TOLERANCE";

    /// <summary>The choice that picks the sum type per device and operation (the default): the fastest within the tolerance, measured.</summary>
    public const string Auto = "auto";

    private static readonly ConcurrentDictionary<(string Device, string Operation), SumTypeMeasurement[]> s_measured = new();
    private static double? s_tolerance;

    /// <summary>
    /// The largest relative error a sum type may have against the widest the device reports for auto to pick it: set here
    /// or by <see cref="ToleranceVariable"/>; when neither is, the resolution of the type the values are kept in
    /// (<see cref="Resolution"/> of <see cref="ElementTypes.Choose"/>'s pick), so a sum loses nothing the values keep.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">A tolerance that is negative or not finite.</exception>
    public static double? Tolerance
    {
        get => s_tolerance;
        set => s_tolerance = value is null || double.IsFinite(value.Value) && value.Value >= 0 ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "A sum type's tolerance is a relative error of 0 or more.");
    }

    /// <summary>
    /// The tolerance a choice on <paramref name="device"/> uses: <see cref="Tolerance"/>, else
    /// <see cref="ToleranceVariable"/>, else the resolution of the type the run keeps its values in on the device.
    /// </summary>
    /// <exception cref="ArgumentException">The variable holds no relative error.</exception>
    public static double ToleranceOn(Device? device = null)
    {
        if (s_tolerance is { } set)
        {
            return set;
        }

        string? text = Environment.GetEnvironmentVariable(ToleranceVariable);
        if (!string.IsNullOrWhiteSpace(text))
        {
            return double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double value) && double.IsFinite(value) && value >= 0
                ? value
                : throw new ArgumentException($"{ToleranceVariable} '{text}': a relative error of 0 or more (1e-3, say).");
        }

        return Resolution(ElementTypes.Choose(device: device).Type);
    }

    /// <summary>
    /// A float type's resolution at 1: the distance from 1 to the next value it holds (2^-23 for float32, 2^-7 for
    /// bfloat16), found by storing 1 + 2^-k in the type and reading it back, so a registered type needs no table; 1 for a
    /// type that is not a float or has no conversions.
    /// </summary>
    public static double Resolution(ElementType type)
    {
        ArgumentNullException.ThrowIfNull(type);
        if (!type.IsFloat || type.IsBlock || type.ToFloat32 is null || type.FromFloat32 is null)
        {
            return 1;
        }

        Span<byte> stored = stackalloc byte[8];
        Span<float> back = stackalloc float[1];
        double resolution = 1;
        for (int k = 1; k < 24; k++)
        {
            float value = 1f + MathF.Pow(2f, -k);
            type.Encode([value], stored);
            type.Decode(stored, back);
            if (back[0] != value)
            {
                break;
            }

            resolution = Math.Pow(2, -k);
        }

        return resolution;
    }

    /// <summary>Keeps <paramref name="measurements"/> of <paramref name="operation"/> on <paramref name="device"/> for <see cref="Choose"/>'s auto (the latest replace earlier ones).</summary>
    public static void Record(Device device, Operation operation, IReadOnlyList<SumTypeMeasurement> measurements)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(operation);
        ArgumentNullException.ThrowIfNull(measurements);
        s_measured[(device.ToString(), operation.Name)] = [.. measurements];
    }

    /// <summary>The measurements kept for <paramref name="operation"/> on <paramref name="device"/>, or empty.</summary>
    public static IReadOnlyList<SumTypeMeasurement> Measurements(Device device, Operation operation)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(operation);
        return s_measured.TryGetValue((device.ToString(), operation.Name), out var kept) ? kept : [];
    }

    /// <summary>
    /// The fastest of <paramref name="measurements"/> whose error is within <paramref name="tolerance"/> (ties: the first
    /// given); when none is, the one with the least error.
    /// </summary>
    /// <exception cref="ArgumentException">No measurement.</exception>
    public static SumTypeMeasurement Pick(IReadOnlyList<SumTypeMeasurement> measurements, double tolerance)
    {
        ArgumentNullException.ThrowIfNull(measurements);
        if (measurements.Count == 0)
        {
            throw new ArgumentException("No sum type was measured.", nameof(measurements));
        }

        SumTypeMeasurement? best = null;
        foreach (var m in measurements)
        {
            if (m.Error <= tolerance && (best is null || m.Seconds < best.Seconds))
            {
                best = m;
            }
        }

        return best ?? measurements.MinBy(m => m.Error)!;
    }

    /// <summary>
    /// The type <paramref name="operation"/> sums in on <paramref name="device"/> (the CPU when null), among those its
    /// backend reports (<see cref="Devices.BackendCapabilities.SumTypesOf"/>): <paramref name="asked"/> (a run's option,
    /// named <paramref name="source"/>) when given, else <see cref="EnvironmentVariable"/>, else <see cref="Auto"/>: the
    /// fastest within <see cref="ToleranceOn"/> from the measurements kept for it (<see cref="Record"/>), the only type
    /// when the device reports one, else the first the device reports (its own order) until it is measured, said so.
    /// </summary>
    /// <exception cref="ArgumentException">A type the device does not report for the operation (the message names those it does).</exception>
    /// <exception cref="NotSupportedException">The device reports no sum type for the operation.</exception>
    public static ElementTypeChoice Choose(Operation operation, Device? device = null, string? asked = null, string source = "--sum-dtype")
    {
        ArgumentNullException.ThrowIfNull(operation);
        device ??= Device.Cpu;
        var reported = device.Backend.Capabilities.SumTypesOf(operation);
        if (reported.Count == 0)
        {
            throw new NotSupportedException($"{device} reports no sum type for {operation.Name}: its backend lists them in BackendCapabilities.SumTypes.");
        }

        string role = $"sum type for {operation.Name}";
        string native = $"{string.Join(", ", reported.Select(t => t.Name))} for {operation.Name}";
        string? value = asked, from = source;
        if (string.IsNullOrWhiteSpace(value))
        {
            value = Environment.GetEnvironmentVariable(EnvironmentVariable);
            from = EnvironmentVariable;
        }

        if (string.IsNullOrWhiteSpace(value) || value.Trim().Equals(Auto, StringComparison.OrdinalIgnoreCase))
        {
            var measured = Measurements(device, operation).Where(m => reported.Contains(m.Type)).ToList();
            string why;
            ElementType type;
            if (measured.Count > 0)
            {
                double tolerance = ToleranceOn(device);
                var pick = Pick(measured, tolerance);
                type = pick.Type;
                why = string.Create(CultureInfo.InvariantCulture,
                    $"the fastest within {tolerance:G3} of the widest sum on {device}: {string.Join(", ", measured.Select(m => $"{m.Type.Name} {m.Seconds * 1e3:G3} ms (error {m.Error:G3})"))}");
            }
            else
            {
                type = reported[0];
                why = reported.Count == 1 ? $"the only sum type {device} reports for {operation.Name}" : $"not measured yet: the first of {native} that {device} reports";
            }

            return new ElementTypeChoice(type, Picked: true, Source: null) { Device = device.ToString(), Native = native, Role = role, Measured = why };
        }

        var selected = ElementTypes.Find(value.AsSpan().Trim());
        if (selected is null || !reported.Contains(selected))
        {
            throw new ArgumentException($"{from} '{value}': {device} sums {operation.Name} in {string.Join(", ", reported.Select(t => t.Name))} (or {Auto}: the fastest within the tolerance, measured).");
        }

        return new ElementTypeChoice(selected, Picked: false, Source: from) { Device = device.ToString(), Native = native, Role = role };
    }

    /// <summary>Publishes <paramref name="choice"/> for <paramref name="device"/> to telemetry, as <see cref="ElementTypes.Report"/> does.</summary>
    public static void Report(ElementTypeChoice choice, Device device) => ElementTypes.Report(choice, device);
}
