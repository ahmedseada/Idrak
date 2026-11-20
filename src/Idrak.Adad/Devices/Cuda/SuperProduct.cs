// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using Idrak.Adad.Abstractions;
using Idrak.Adad.Measure;
using Idrak.Adad.Types;

namespace Idrak.Adad.Devices.Cuda;

/// <summary>A measured lane choice's key: the description's stable id (decision 8), the class of m, n and k.</summary>
public readonly record struct ProductKey(int Description, int M, int N, int K);

/// <summary>A measured variant choice's key: the product's key and the lane.</summary>
public readonly record struct VariantKey(ProductKey Product, int Lane);

/// <summary>
/// The prompt product y = x · Wᵀ for weights of a super-block description on CUDA (decision 23): one default, the control
/// (today's winning path), and the overrides A and B, each matched by the description's fields (decision 7). Per shape the
/// measured choice decides which runs (decision 5: every lane that passes its tests is a candidate): first each lane's own
/// variant (its kernel shape and k split, timed warm on the real operands, as the proof's native choice), then the lanes
/// against the default (timed cold, as the proof's choice among a super-block's lanes). Every choice is keyed by the
/// description's stable id and the shape; the description is resolved once per call (its lanes found by its id), and a
/// call formats no string and builds no list.
/// </summary>
public sealed class SuperProduct(CudaDevice device, Tuner tuner)
{
    /// <summary>The lanes, in order: 0 the default.</summary>
    public static readonly string[] LaneNames = ["control", "A", "B"];

    private readonly Dictionary<int, Lanes> _lanes = [];
    private readonly Dictionary<ProductKey, int> _choice = [];
    private readonly Dictionary<VariantKey, int> _variant = [];

    /// <summary>Whether a choice not yet made is measured (else the default runs, and each lane's device shape unsplit).</summary>
    public bool Measuring { get; set; } = true;

    // A description's lanes: each prepared once, or null with the reason it is no candidate.
    internal sealed class Lanes(SuperBlock description, SuperLane?[] lanes, string[] reasons)
    {
        public SuperBlock Description { get; } = description;

        public SuperLane?[] Of { get; } = lanes;

        public string[] Reasons { get; } = reasons;
    }

    /// <summary>The lanes of <paramref name="s"/>, prepared on its first use (kernels generated from the description, loaded, the device shapes measured).</summary>
    internal Lanes Prepare(SuperBlock s)
    {
        int id = s.Id;
        if (_lanes.TryGetValue(id, out var lanes))
        {
            return lanes;
        }

        var made = new SuperLane?[] { new ControlLane(device, tuner), new DecodeLane(device, tuner), new Int8Lane(device, tuner) };
        var reasons = new string[made.Length];
        for (int i = 0; i < made.Length; i++)
        {
            made[i]!.MeasureShape = Measuring;
            if (!made[i]!.Prepare(s, out reasons[i]))
            {
                made[i] = null;
            }
        }

        if (made[0] is null)
        {
            throw new CudaException($"the default lane of the product does not run on {device.Name}: {reasons[0]}");
        }

        lanes = new Lanes(s, made, reasons);
        _lanes[id] = lanes;
        return lanes;
    }

    /// <summary>y = x · Wᵀ by the measured choice for this shape (measured now when not yet made and <see cref="Measuring"/>).</summary>
    public void Run(SuperBlock s, in ProductOperands op)
    {
        var lanes = Prepare(s);
        var key = new ProductKey(s.Id, SizeClass.Of(op.M), op.N, op.K);
        if (!_choice.TryGetValue(key, out int lane))
        {
            lane = Choose(lanes, key, op);
        }

        RunLane(lanes, lane, op);
    }

    /// <summary>y = x · Wᵀ by lane <paramref name="lane"/> (its own variant measured for this shape on first use).</summary>
    public void RunLane(SuperBlock s, int lane, in ProductOperands op) => RunLane(Prepare(s), lane, op);

    private void RunLane(Lanes lanes, int lane, in ProductOperands op)
    {
        var l = lanes.Of[lane] ?? throw new CudaException($"lane {LaneNames[lane]} is no candidate: {lanes.Reasons[lane]}");
        l.Run(VariantOf(lanes, lane, op), op);
    }

    /// <summary>Lane <paramref name="lane"/>'s variant for this shape: the stored choice, else measured now (warm, on these operands).</summary>
    public int VariantOf(SuperBlock s, int lane, in ProductOperands op) => VariantOf(Prepare(s), lane, op);

    private int VariantOf(Lanes lanes, int lane, in ProductOperands op)
    {
        var key = new VariantKey(new ProductKey(lanes.Description.Id, SizeClass.Of(op.M), op.N, op.K), lane);
        if (_variant.TryGetValue(key, out int variant))
        {
            return variant;
        }

        var l = lanes.Of[lane]!;
        int[] variants = l.Variants(op.M, op.N, op.K, out int fallback);
        if (!Measuring || !tuner.CanMeasure)
        {
            return fallback;
        }

        var operands = op;
        variant = tuner.Measure(string.Create(CultureInfo.InvariantCulture, $"{LaneNames[lane]} {op.M}x{op.N}x{op.K}: shape and split"), variants, fallback,
            v => l.Run(v, operands), cold: false);
        _variant[key] = variant;
        return variant;
    }

    private int Choose(Lanes lanes, ProductKey key, in ProductOperands op)
    {
        if (!Measuring || !tuner.CanMeasure)
        {
            return 0;
        }

        var candidates = new List<int>(lanes.Of.Length);
        for (int i = 0; i < lanes.Of.Length; i++)
        {
            if (lanes.Of[i] is not null)
            {
                candidates.Add(i);
            }
        }

        var operands = op;
        int lane = tuner.Measure(string.Create(CultureInfo.InvariantCulture, $"lane {op.M}x{op.N}x{op.K}"), candidates.ToArray(), 0, c => RunLane(lanes, c, operands), cold: true);
        _choice[key] = lane;
        return lane;
    }

    /// <summary>The lane chosen for a shape, or −1 when not measured.</summary>
    public int ChosenLane(SuperBlock s, int m, int n, int k) => _choice.TryGetValue(new ProductKey(s.Id, SizeClass.Of(m), n, k), out int lane) ? lane : -1;

    /// <summary>Lane <paramref name="lane"/> of <paramref name="s"/>, or null with the reason it is no candidate.</summary>
    internal SuperLane? Lane(SuperBlock s, int lane, out string reason)
    {
        var lanes = Prepare(s);
        reason = lanes.Reasons[lane];
        return lanes.Of[lane];
    }

    /// <summary>The choices made, as the store's lines (the description's id, m's class, n, k).</summary>
    public string SaveChoices()
    {
        var sb = new StringBuilder();
        foreach (var (key, lane) in _choice.OrderBy(e => (e.Key.Description, e.Key.N, e.Key.K, e.Key.M)))
        {
            sb.AppendLine(CultureInfo.InvariantCulture, $"lane {key.Description} {key.M} {key.N} {key.K} = {lane}");
        }

        foreach (var (key, variant) in _variant.OrderBy(e => (e.Key.Product.Description, e.Key.Product.N, e.Key.Product.K, e.Key.Product.M, e.Key.Lane)))
        {
            sb.AppendLine(CultureInfo.InvariantCulture, $"variant {key.Product.Description} {key.Product.M} {key.Product.N} {key.Product.K} {key.Lane} = {variant}");
        }

        return sb.ToString();
    }

    /// <summary>Reads choices saved by <see cref="SaveChoices"/> (a line it does not know is skipped).</summary>
    public int LoadChoices(string text)
    {
        int read = 0;
        foreach (string line in text.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            int Int(int i) => int.Parse(parts[i], CultureInfo.InvariantCulture);
            if (parts is ["lane", _, _, _, _, "=", _])
            {
                _choice[new ProductKey(Int(1), Int(2), Int(3), Int(4))] = Int(6);
                read++;
            }
            else if (parts is ["variant", _, _, _, _, _, "=", _])
            {
                _variant[new VariantKey(new ProductKey(Int(1), Int(2), Int(3), Int(4)), Int(5))] = Int(7);
                read++;
            }
        }

        return read;
    }
}
