// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Globalization;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using Idrak.Layers.Abstractions;

namespace Idrak.Models;

/// <summary>
/// What a load's packed-weight cache did (<see cref="PretrainedModel.PackedCache"/>): the weights it read back (a warm
/// load), those it packed and kept (a cold one), and where.
/// </summary>
/// <param name="Folder">The model's folder in the cache (<see cref="PretrainedOptions.PackedCacheFolder"/>).</param>
/// <param name="Read">Packed weights read from the cache, not packed again.</param>
/// <param name="Written">Packed weights made and kept for the next load.</param>
/// <param name="Problem">Why weights could not be kept, the first time (a full disk, a folder that cannot be written), or null.</param>
public sealed record PackedWeightCacheUse(string Folder, int Read, int Written, string? Problem);

/// <summary>
/// The packed weights (int8, int4, any format with a <see cref="PackedWeightLayouts">layout</see>) of one model's loads,
/// kept on disk (<see cref="PretrainedOptions.PackedCache"/>), so the next load of the same checkpoint reads them instead
/// of reading the checkpoint's values and packing again: the same bits, since packing is a function of the values, the
/// format and its packing version.
/// <para>
/// One file a packed tensor, in a folder of the model's (its checkpoint folder) and a folder of the format's. An entry is
/// found by a name made from what never changes for that tensor (its name and role); its header holds the whole key: the
/// stored tensor's identity (its file's path, size and time of change, its place and type there:
/// <see cref="Abstractions.IWeightSource.IdentityOf"/>), the shape, the format and its layout's version, and this file layout's version.
/// A key that differs in any part is a miss: the weights are packed again and the entry written over, so a checkpoint
/// edited, a format's packing changed or a newer library never reads an older entry. After the load the entries of the
/// formats it used that it did not use are removed (<see cref="Prune"/>), and once a process the entries of models whose
/// folder is gone (<see cref="SweepGone"/>). An entry is read through a memory map straight
/// into the weight's device copy (no copy of the file on the managed heap), and written to a temporary file first, then
/// moved over the entry (a reader never sees half of one). A source that cannot say a tensor's identity (an adapter merged
/// into it, a store that does not know its files) is not cached. The path, size and time of change identify a file, not a
/// hash of its bytes: hashing would read the whole checkpoint on every load, which is what the cache saves.
/// </para>
/// </summary>
internal sealed class PackedWeightCache
{
    private const uint Magic = 0x4357_5049;                 // "IPWC"
    private const int FileVersion = 1;
    private const int Alignment = 64;                       // the values start on a cache line
    private const string SourceFile = "source.txt";         // the model folder an entry folder is for
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, bool> s_swept = new(StringComparer.Ordinal);
    private readonly string _folder;
    private readonly HashSet<string> _used = new(StringComparer.Ordinal);
    private readonly Lock _gate = new();
    private int _hits, _written;
    private string? _problem;
    private bool _sourceWritten;

    /// <summary>A cache for the model in <paramref name="model"/> (its checkpoint folder), under <paramref name="root"/>.</summary>
    public PackedWeightCache(string root, string model)
    {
        Model = Path.GetFullPath(model);
        _folder = Path.Combine(Path.GetFullPath(root), Hash(Model)[..16]);
    }

    /// <summary>The model folder the entries are for.</summary>
    public string Model { get; }

    /// <summary>What this load read and kept so far.</summary>
    public PackedWeightCacheUse Use
    {
        get
        {
            lock (_gate)
            {
                return new PackedWeightCacheUse(_folder, _hits, _written, _problem);
            }
        }
    }

    /// <summary>The cache a load asks for, or null when it keeps none: its option, else <see cref="ComputeResources.PackedWeightCache"/>.</summary>
    public static PackedWeightCache? For(PretrainedOptions options, string model)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (!(options.PackedCache ?? ComputeResources.PackedWeightCache) || options.Deferred)
        {
            return null;
        }

        string root = Path.GetFullPath(options.PackedCacheFolder ?? ComputeResources.PackedWeightCacheFolder);
        if (s_swept.TryAdd(root, true))
        {
            SweepGone(root);                                // once a process for each cache folder
        }

        return new PackedWeightCache(root, model);
    }

    /// <summary>
    /// Removes the entries of models whose folder is gone (a model deleted, a temporary one), so the cache does not keep
    /// what no load can ask for again: each model's folder names its model in a small file.
    /// </summary>
    internal static void SweepGone(string root)
    {
        if (!Directory.Exists(root))
        {
            return;
        }

        foreach (string folder in Directory.EnumerateDirectories(root))
        {
            try
            {
                string source = Path.Combine(folder, SourceFile);
                if (File.Exists(source) && File.ReadAllText(source).Trim() is { Length: > 0 } model && !Directory.Exists(model) && !File.Exists(model))
                {
                    Directory.Delete(folder, recursive: true);
                }
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                // in use by another process: the next sweep
            }
        }
    }

    /// <summary>
    /// The kept weights of <paramref name="tensor"/>'s <paramref name="role"/> ("weight", "tied head") packed from the stored
    /// tensor <paramref name="identity"/>, [rows, columns] in <paramref name="layout"/>'s format, on <paramref name="device"/>;
    /// null on a miss (none kept, or kept for another key).
    /// </summary>
    public unsafe IKernelPackedWeight? TryRead(string tensor, string role, string identity, IPackedWeightLayout layout, int rows, int columns, Device device)
    {
        string path = PathOf(tensor, role, layout);
        lock (_gate)
        {
            _used.Add(path);
        }

        if (!File.Exists(path))
        {
            return null;
        }

        try
        {
            using var map = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            using var view = map.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            byte* start = null;
            view.SafeMemoryMappedViewHandle.AcquirePointer(ref start);
            try
            {
                byte* file = start + view.PointerOffset;
                long length = (long)view.SafeMemoryMappedViewHandle.ByteLength - view.PointerOffset;
                var header = new ReadOnlySpan<byte>(file, (int)Math.Min(length, 1 << 16));
                if (!Entry(header, Key(identity, layout, rows, columns), out long packedLength, out long scalesLength, out int dataStart)
                    || packedLength > int.MaxValue || scalesLength > int.MaxValue || dataStart + 4 * (packedLength + scalesLength) > length)
                {
                    return null;
                }

                // The values as mapped, read once into the weights' device memory.
                var packed = new ReadOnlySpan<float>(file + dataStart, (int)packedLength);
                var scales = new ReadOnlySpan<float>(file + dataStart + 4 * packedLength, (int)scalesLength);
                var weight = layout.FromStored(packed, scales, rows, columns, device);
                if (weight is not null)
                {
                    lock (_gate)
                    {
                        _hits++;
                    }
                }

                return weight;
            }
            finally
            {
                view.SafeMemoryMappedViewHandle.ReleasePointer();
            }
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or ArgumentException)
        {
            return null;                                    // unreadable, empty or being written over: packed again (and written over)
        }
    }

    /// <summary>Keeps <paramref name="weight"/> as the entry of <paramref name="tensor"/>'s <paramref name="role"/> packed from <paramref name="identity"/>.</summary>
    public void Write(string tensor, string role, string identity, IPackedWeightLayout layout, IKernelPackedWeight weight)
    {
        ArgumentNullException.ThrowIfNull(layout);
        ArgumentNullException.ThrowIfNull(weight);
        string path = PathOf(tensor, role, layout);
        string temporary = string.Create(CultureInfo.InvariantCulture, $"{path}.{Environment.ProcessId}.{Environment.CurrentManagedThreadId}.tmp");
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            if (!_sourceWritten)
            {
                File.WriteAllText(Path.Combine(_folder, SourceFile), Model);
                _sourceWritten = true;
            }

            byte[] key = Encoding.UTF8.GetBytes(Key(identity, layout, weight.Rows, weight.Columns));
            int dataStart = (12 + key.Length + 16 + Alignment - 1) / Alignment * Alignment;
            using (var stream = new FileStream(temporary, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 16))
            {
                Span<byte> head = stackalloc byte[12];
                BinaryPrimitives.WriteUInt32LittleEndian(head, Magic);
                BinaryPrimitives.WriteInt32LittleEndian(head[4..], FileVersion);
                BinaryPrimitives.WriteInt32LittleEndian(head[8..], key.Length);
                stream.Write(head);
                stream.Write(key);
                Span<byte> lengths = stackalloc byte[16];
                BinaryPrimitives.WriteInt64LittleEndian(lengths, weight.PackedValues.Size);
                BinaryPrimitives.WriteInt64LittleEndian(lengths[8..], weight.ScaleValues?.Size ?? 0);
                stream.Write(lengths);
                stream.Write(new byte[dataStart - stream.Position]);
                WriteValues(stream, weight.PackedValues);
                if (weight.ScaleValues is { } scales)
                {
                    WriteValues(stream, scales);
                }
            }

            File.Move(temporary, path, overwrite: true);
            lock (_gate)
            {
                _written++;
            }
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            lock (_gate)
            {
                _problem ??= e.Message;
            }

            try
            {
                File.Delete(temporary);
            }
            catch (Exception cleanup) when (cleanup is IOException or UnauthorizedAccessException)
            {
                // left for the next prune
            }
        }
    }

    /// <summary>
    /// Removes the entries of the formats this load used that it did not use (another model's shape, a tensor no longer
    /// there), and temporary files over an hour old; entries of formats it did not use stay for the loads that use them.
    /// </summary>
    public void Prune()
    {
        string[] folders;
        lock (_gate)
        {
            folders = [.. _used.Select(p => Path.GetDirectoryName(p)!).Distinct(StringComparer.Ordinal)];
        }

        foreach (string folder in folders)
        {
            if (!Directory.Exists(folder))
            {
                continue;
            }

            foreach (string file in Directory.EnumerateFiles(folder))
            {
                bool keep;
                lock (_gate)
                {
                    keep = _used.Contains(file);
                }

                keep |= file.EndsWith(".tmp", StringComparison.Ordinal) && File.GetLastWriteTimeUtc(file) > DateTime.UtcNow.AddHours(-1);
                if (!keep)
                {
                    try
                    {
                        File.Delete(file);
                    }
                    catch (Exception e) when (e is IOException or UnauthorizedAccessException)
                    {
                        // in use by another process: the next prune
                    }
                }
            }
        }
    }

    /// <summary>"packed weights: 12 read from the cache, 0 packed and kept (…/packed/1a2b…)", for the load's notes.</summary>
    public string Describe()
    {
        var use = Use;
        return string.Create(CultureInfo.InvariantCulture, $"packed weights: {use.Read} read from the cache, {use.Written} packed and kept ({use.Folder})")
            + (use.Problem is { } problem ? $"; not kept: {problem}" : "");
    }

    // The values of a packed tensor, from its memory when it is in host memory (no copy), else copied once.
    private static void WriteValues(FileStream stream, Tensor values)
    {
        if (values.Storage is IHostStorage host && values.Storage.Length == values.Size)
        {
            stream.Write(MemoryMarshal.AsBytes(host.HostMemory[..values.Size]));
            return;
        }

        stream.Write(MemoryMarshal.AsBytes(values.ToArray().AsSpan()));
    }

    // The entry's whole key: the source's identity, the shape, the format and its layout's version.
    private static string Key(string identity, IPackedWeightLayout layout, int rows, int columns) =>
        string.Create(CultureInfo.InvariantCulture, $"{identity}|[{rows}, {columns}]|{layout.Name.ToLowerInvariant()} v{layout.Version}");

    // Entries of a format in a folder of their own: the format's name when it makes a plain folder name, else its hash.
    private string PathOf(string tensor, string role, IPackedWeightLayout layout)
    {
        string format = layout.Name.ToLowerInvariant();
        string folder = format.Length is > 0 and <= 32 && format.All(c => char.IsAsciiLetterOrDigit(c) || c is '-' or '_') ? format : "f" + Hash(format)[..16];
        return Path.Combine(_folder, folder, Hash(tensor + "|" + role)[..24] + ".ipw");
    }

    // Whether the entry's header holds `key`, and where its values are.
    private static bool Entry(ReadOnlySpan<byte> file, string key, out long packedLength, out long scalesLength, out int dataStart)
    {
        (packedLength, scalesLength, dataStart) = (0, 0, 0);
        if (file.Length < 12 || BinaryPrimitives.ReadUInt32LittleEndian(file) != Magic || BinaryPrimitives.ReadInt32LittleEndian(file[4..]) != FileVersion)
        {
            return false;
        }

        int keyLength = BinaryPrimitives.ReadInt32LittleEndian(file[8..]);
        if (keyLength < 0 || 12L + keyLength + 16 > file.Length || !file.Slice(12, keyLength).SequenceEqual(Encoding.UTF8.GetBytes(key)))
        {
            return false;
        }

        packedLength = BinaryPrimitives.ReadInt64LittleEndian(file[(12 + keyLength)..]);
        scalesLength = BinaryPrimitives.ReadInt64LittleEndian(file[(20 + keyLength)..]);
        dataStart = (12 + keyLength + 16 + Alignment - 1) / Alignment * Alignment;
        return packedLength >= 0 && scalesLength >= 0;
    }

    private static string Hash(string text) => Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(text)));
}
