// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Operations;
using Idrak.Abstraction.Testing.Devices;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// A case of the device check: a small program on the CPU (tensors, autograd, or operations called on its backend). The
/// kit runs it on the CPU, records every operation call it makes with the values of its storages, then makes each call
/// again on the device under test with the same values and compares what it writes. A case may also check the CPU's
/// results against plain loops (<see cref="DeviceCaseContext.ExpectClose"/>), which keeps the reference honest.
/// </summary>
/// <param name="name">The case's name in reports and saved cases.</param>
/// <param name="run">The program; it builds everything on <see cref="DeviceCaseContext.Device"/> (the CPU).</param>
/// <param name="random">
/// Whether the program draws its shapes from <see cref="DeviceCaseContext.Random"/>: such a case runs
/// <see cref="DeviceCheckOptions.RandomRuns"/> times with different seeds, a fixed one once.
/// </param>
public sealed class DeviceCase(string name, Action<DeviceCaseContext> run, bool random = false)
{
    /// <summary>The case's name.</summary>
    public string Name { get; } = name;

    /// <summary>The program.</summary>
    public Action<DeviceCaseContext> Run { get; } = run;

    /// <summary>Whether the program draws its shapes at random.</summary>
    public bool Random { get; } = random;

    /// <inheritdoc />
    public override string ToString() => Name;
}

/// <summary>What a <see cref="DeviceCase"/> runs with: the CPU, a seeded random source, and checks against plain loops.</summary>
public sealed class DeviceCaseContext : IDisposable
{
    private readonly List<Storage> _storages = [];
    private readonly CallRecorder? _recorder;

    internal DeviceCaseContext(int seed, bool large, CallRecorder? recorder)
    {
        _recorder = recorder;
        Seed = seed;
        Random = new Random(seed);
        Large = large;
    }

    /// <summary>The device the case runs on: the CPU, whose kernels are the library default.</summary>
    public Device Device => Device.Cpu;

    /// <summary>The CPU's backend, for calling an operation directly (<c>Backend.Softmax(...)</c>).</summary>
    public Backend Backend => Device.Cpu.Backend;

    /// <summary>The seed of <see cref="Random"/>.</summary>
    public int Seed { get; }

    /// <summary>The random source for values and shapes, seeded per run.</summary>
    public Random Random { get; }

    /// <summary>Whether large shapes are wanted (<see cref="DeviceCheckOptions.Large"/>).</summary>
    public bool Large { get; }

    /// <summary><paramref name="count"/> values drawn uniformly from [-scale, scale].</summary>
    public float[] Values(int count, float scale = 1f)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = (Random.NextSingle() * 2f - 1f) * scale;
        }

        return values;
    }

    /// <summary>A size from [<paramref name="min"/>, <paramref name="max"/>], odd sizes (primes and one past a power of two) as likely as any.</summary>
    public int Size(int min, int max)
    {
        int[] odd = [.. new[] { 1, 3, 5, 7, 13, 17, 31, 33, 63, 65, 127, 129, 255, 257 }.Where(s => s >= min && s <= max)];
        return odd.Length > 0 && Random.Next(3) == 0 ? odd[Random.Next(odd.Length)] : Random.Next(min, max + 1);
    }

    /// <summary>A CPU storage holding <paramref name="values"/>, released when the case ends.</summary>
    public Storage Storage(ReadOnlySpan<float> values)
    {
        var storage = Backend.Allocate(values.Length, zeroed: false);
        Backend.Upload(values, storage);
        _storages.Add(storage);
        return storage;
    }

    /// <summary>A CPU storage of <paramref name="length"/> zeros, released when the case ends.</summary>
    public Storage Zeros(int length)
    {
        var storage = Backend.Allocate(length, zeroed: true);
        _storages.Add(storage);
        return storage;
    }

    /// <summary>The values of a storage.</summary>
    public static float[] Read(Storage storage)
    {
        ArgumentNullException.ThrowIfNull(storage);
        var values = new float[storage.Length];
        storage.Backend.Download(storage, values);
        return values;
    }

    /// <summary>Fails the case, as the library default's own failure, unless <paramref name="condition"/> holds.</summary>
    /// <exception cref="ConformanceException">The condition does not hold.</exception>
    public void Expect(bool condition, string message)
    {
        if (!condition)
        {
            throw new ConformanceException(message);
        }
    }

    /// <summary>
    /// Fails the case unless the CPU's <paramref name="actual"/> values agree with <paramref name="expected"/> (computed
    /// with plain loops) within <paramref name="tolerance"/> (<see cref="Comparisons.Close"/>).
    /// </summary>
    /// <exception cref="ConformanceException">They do not agree.</exception>
    public void ExpectClose(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, float tolerance, string what)
    {
        if (Comparisons.Difference(expected, actual, tolerance) is { } difference)
        {
            throw new ConformanceException($"{what}: {difference}");
        }
    }

    /// <summary>
    /// Calls <paramref name="operation"/>, an operation the CPU may have no kernel for (it returns false there, as the
    /// fused products and <c>MatMulBias</c> do), then, when it returned false, <paramref name="reference"/>, which writes
    /// what the operation must write into the same storages with other operations. The device's kernel for the operation
    /// is then compared with what the reference wrote.
    /// </summary>
    /// <param name="descriptor">The operation <paramref name="operation"/> calls (<c>Ops.MatMulBias</c>).</param>
    /// <param name="operation">Calls it on the CPU's backend; returns what it returned.</param>
    /// <param name="reference">Writes its results with other operations (it must not change the operation's inputs).</param>
    public void Composed(Operation descriptor, Func<Backend, bool> operation, Action<Backend> reference)
    {
        ArgumentNullException.ThrowIfNull(descriptor);
        ArgumentNullException.ThrowIfNull(operation);
        ArgumentNullException.ThrowIfNull(reference);
        int index = _recorder?.Count ?? 0;
        if (operation(Backend))
        {
            return;                                                            // the CPU has a kernel for it: compared as it is
        }

        reference(Backend);
        _recorder?.Rewrite(index, descriptor);
    }

    /// <summary>Releases the storages the case made.</summary>
    public void Dispose()
    {
        foreach (var storage in _storages)
        {
            storage.Release();
        }

        _storages.Clear();
    }
}
