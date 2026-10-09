// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics.CodeAnalysis;
using Idrak.Models;
using Idrak.Nlp.Abstractions;

namespace Idrak.Nlp;

// The "memory" feature cache: copies of the values on the host, the least recently used evicted to stay within a byte
// budget (given, or half of the system memory the machine reports available when the cache is made).
internal sealed class MemoryFeatureCache : IFeatureCache
{
    private sealed record Entry(FeatureCacheKey Key, float[] Values, int[] Shape)
    {
        public long Bytes => Values.LongLength * sizeof(float);
    }

    private readonly Lock _gate = new();
    private readonly Dictionary<FeatureCacheKey, LinkedListNode<Entry>> _map = [];
    private readonly LinkedList<Entry> _recent = new();                   // the most recently used first
    private long _bytes;

    public MemoryFeatureCache(FeatureCacheOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        Budget = options.Budget ?? MeasuredBudget();
        ArgumentOutOfRangeException.ThrowIfNegative(Budget, nameof(options.Budget));
    }

    public string Name => FeatureCaches.Memory;

    // The most bytes of values kept.
    public long Budget { get; }

    public int Count
    {
        get
        {
            lock (_gate)
            {
                return _map.Count;
            }
        }
    }

    public long Bytes => Interlocked.Read(ref _bytes);

    public bool Contains(FeatureCacheKey key)
    {
        ArgumentNullException.ThrowIfNull(key);
        lock (_gate)
        {
            return _map.ContainsKey(key);
        }
    }

    public bool TryGet(FeatureCacheKey key, Device device, [NotNullWhen(true)] out Tensor? value)
    {
        ArgumentNullException.ThrowIfNull(key);
        ArgumentNullException.ThrowIfNull(device);
        Entry entry;
        lock (_gate)
        {
            if (!_map.TryGetValue(key, out var node))
            {
                value = null;
                return false;
            }

            _recent.Remove(node);
            _recent.AddFirst(node);
            entry = node.Value;
        }

        value = Tensor.From(entry.Values, entry.Shape, device);                 // entries are never changed: read outside the lock
        return true;
    }

    public bool Put(FeatureCacheKey key, Tensor value)
    {
        ArgumentNullException.ThrowIfNull(key);
        ArgumentNullException.ThrowIfNull(value);
        var entry = new Entry(key, value.ToArray(), value.Shape.ToArray());
        if (entry.Bytes > Budget)
        {
            return false;
        }

        lock (_gate)
        {
            if (_map.Remove(key, out var old))
            {
                _recent.Remove(old);
                Interlocked.Add(ref _bytes, -old.Value.Bytes);
            }

            _map[key] = _recent.AddFirst(entry);
            Interlocked.Add(ref _bytes, entry.Bytes);
            while (_bytes > Budget && _recent.Last is { } last)
            {
                _recent.RemoveLast();
                _map.Remove(last.Value.Key);
                Interlocked.Add(ref _bytes, -last.Value.Bytes);
            }
        }

        return true;
    }

    public void Clear()
    {
        lock (_gate)
        {
            _map.Clear();
            _recent.Clear();
            Interlocked.Exchange(ref _bytes, 0);
        }
    }

    public void Dispose() => Clear();

    public override string ToString() => $"memory feature cache: {Count} entries, {Bytes:N0} of {Budget:N0} bytes";

    // Half of the system memory available now, as the CPU device reports it (within ComputeResources.CpuMemoryLimit).
    private static long MeasuredBudget()
    {
        long available = Device.Cpu.Backend.AvailableMemory() ?? GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
        if (ComputeResources.CpuMemoryLimit is { } limit)
        {
            available = Math.Min(available, limit);
        }

        return Math.Max(0, available / 2);
    }
}

// The "disk" feature cache: one safetensors file per entry (float32, the tensor "features", the key in the metadata), in
// a folder (given, or FeatureCaches.DefaultFolder) under two-hex-digit subfolders of the key's id. A file is written under
// a temporary name and moved over the old one when whole, so readers never see half a file; one whose metadata does not
// match its key is a miss. With a budget, the oldest files go first to make room.
internal sealed class DiskFeatureCache : IFeatureCache
{
    private const string Format = "idrak-features/1", TensorName = "features";

    public DiskFeatureCache(FeatureCacheOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (options.Budget is { } budget)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(budget, nameof(options.Budget));
        }

        Folder = Path.GetFullPath(options.Folder ?? FeatureCaches.DefaultFolder);
        Budget = options.Budget;
        Directory.CreateDirectory(Folder);
    }

    public string Name => FeatureCaches.Disk;

    // Where the files are.
    public string Folder { get; }

    // The most bytes of files kept, or null for what the disk takes.
    public long? Budget { get; }

    public int Count => Files().Count();

    public long Bytes => Files().Sum(f => f.Length);

    public bool Contains(FeatureCacheKey key)
    {
        ArgumentNullException.ThrowIfNull(key);
        return File.Exists(PathOf(key));
    }

    public bool TryGet(FeatureCacheKey key, Device device, [NotNullWhen(true)] out Tensor? value)
    {
        ArgumentNullException.ThrowIfNull(key);
        ArgumentNullException.ThrowIfNull(device);
        value = null;
        string path = PathOf(key);
        if (!File.Exists(path))
        {
            return false;
        }

        try
        {
            using var reader = SafeTensorsReader.Open(path);
            if (!reader.Metadata.TryGetValue("format", out var format) || format != Format
                || !reader.Metadata.TryGetValue("key", out var stored) || stored != key.ToJson().ToJsonString()
                || !reader.Tensors.TryGetValue(TensorName, out var info))
            {
                return false;
            }

            value = Tensor.From(reader.Read(TensorName), info.Shape, device);
            return true;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or InvalidDataException or System.Text.Json.JsonException or ArgumentException)
        {
            return false;                                                       // replaced or removed meanwhile, or not a cache file: a miss
        }
    }

    public bool Put(FeatureCacheKey key, Tensor value)
    {
        ArgumentNullException.ThrowIfNull(key);
        ArgumentNullException.ThrowIfNull(value);
        float[] values = value.ToArray();
        int[] shape = value.Shape.ToArray();
        long bytes = values.LongLength * sizeof(float);
        if (Budget is { } budget && !MakeRoom(budget, bytes, PathOf(key)))
        {
            return false;
        }

        string path = PathOf(key), temporary = $"{path}.{Guid.NewGuid():N}.tmp";
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            SafeTensorsWriter.Write(temporary, [(TensorName, shape, values)], SafeTensorType.F32,
                new Dictionary<string, string> { ["format"] = Format, ["key"] = key.ToJson().ToJsonString() });
            File.Move(temporary, path, overwrite: true);
            return true;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            TryDelete(temporary);                                               // the disk is full, or another writer holds the file
            return false;
        }
    }

    public void Clear()
    {
        foreach (var file in Files())
        {
            TryDelete(file.FullName);
        }
    }

    public void Dispose()
    {
    }

    public override string ToString() => $"disk feature cache in {Folder}";

    private string PathOf(FeatureCacheKey key)
    {
        string id = key.Id;
        return Path.Combine(Folder, id[..2], id + ".safetensors");
    }

    private IEnumerable<FileInfo> Files() => Directory.Exists(Folder)
        ? new DirectoryInfo(Folder).EnumerateDirectories().Where(d => d.Name.Length == 2).SelectMany(d => d.EnumerateFiles("*.safetensors"))
        : [];

    // Removes the oldest files (by last write) until `bytes` more fit in `budget` (the file being replaced counts as gone);
    // false when they cannot fit at all.
    private bool MakeRoom(long budget, long bytes, string replaced)
    {
        if (bytes > budget)
        {
            return false;
        }

        var files = Files().Where(f => !string.Equals(f.FullName, replaced, StringComparison.Ordinal)).OrderBy(f => f.LastWriteTimeUtc).ToList();
        long used = files.Sum(f => f.Length);
        foreach (var file in files)
        {
            if (used + bytes <= budget)
            {
                break;
            }

            if (TryDelete(file.FullName))
            {
                used -= file.Length;
            }
        }

        return used + bytes <= budget;
    }

    private static bool TryDelete(string path)
    {
        try
        {
            File.Delete(path);
            return true;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }
}
