// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;

namespace Idrak.Models;

/// <summary>A key a trie search found: its value and its length in bytes (the key is the input's first Length bytes).</summary>
internal readonly record struct TrieMatch(int Value, int Length);

/// <summary>
/// A double-array trie over UTF-8 byte keys with non-negative int values, in darts-clone's layout: one uint32 unit per
/// node, the same layout SentencePiece's precompiled character map stores. It is read from those bytes
/// (<see cref="ReadCharsmap"/>, <see cref="ReadUnits"/>) or built once from sorted keys (<see cref="Build(byte[][], int[])"/>)
/// for vocabularies; one type serves both. Searching allocates nothing: <see cref="CommonPrefixSearch"/> writes into the
/// caller's span and <see cref="ExactMatch"/> returns the value.
/// </summary>
/// <remarks>
/// A unit holds a label (its low 8 bits, with bit 31 set for a value unit so that no byte matches it), a has-leaf flag
/// (bit 8), and the XOR offset from its position to its children's base (bits 10-31, times 256 when bit 9 is set). From a
/// node at position p with base b = p ^ offset, the child for byte c is at b ^ c, and its unit's label must be c; a key
/// that ends at a node has its value in the unit at b (label 0). Keys therefore contain no 0 byte. A built trie's unused
/// units carry labels no search byte can match there, as darts-clone writes them.
/// </remarks>
internal sealed class DoubleArrayTrie
{
    private const uint LabelMask = 0x8000_00FFu;
    private const uint HasLeafBit = 1u << 8;
    private const uint ValueMask = 0x7FFF_FFFFu;
    private const int BlockSize = 256;
    private const int OpenBlocks = 16;                 // darts-clone's NUM_EXTRA_BLOCKS: the blocks a new node may use
    private const int NearOffsets = 1 << 21;           // offsets below this are stored as they are
    private const int FarOffsets = 1 << 29;            // larger ones must be multiples of 256 and below this

    private readonly uint[] _units;

    private DoubleArrayTrie(uint[] units) => _units = units;

    /// <summary>The number of units (four bytes each).</summary>
    public int Units => _units.Length;

    /// <summary>
    /// Reads SentencePiece's precompiled character map: a little-endian uint32 with the trie's size in bytes, the trie's
    /// units (little-endian uint32), then the normalized strings, each ended by a 0 byte; a match's value is the offset of
    /// its string in <paramref name="normalized"/>.
    /// </summary>
    public static DoubleArrayTrie ReadCharsmap(ReadOnlySpan<byte> charsmap, out ReadOnlySpan<byte> normalized)
    {
        if (charsmap.Length < 4)
        {
            throw new InvalidDataException($"Precompiled charsmap of {charsmap.Length} bytes: its 4-byte trie size at byte 0 is cut off.");
        }

        uint size = BinaryPrimitives.ReadUInt32LittleEndian(charsmap);
        if (size == 0 || size % 4 != 0 || size > (uint)(charsmap.Length - 4))
        {
            throw new InvalidDataException(
                $"Precompiled charsmap of {charsmap.Length} bytes: the trie size at byte 0 is {size} (it must be a positive multiple of 4 and at most {charsmap.Length - 4}).");
        }

        var trie = ReadUnits(charsmap.Slice(4, (int)size));
        normalized = charsmap[(4 + (int)size)..];
        return trie;
    }

    /// <summary>Reads darts-clone units: little-endian uint32s, the first one the root.</summary>
    public static DoubleArrayTrie ReadUnits(ReadOnlySpan<byte> bytes)
    {
        if (bytes.Length < 4 || bytes.Length % 4 != 0)
        {
            throw new InvalidDataException($"Double-array trie of {bytes.Length} bytes: its size must be a positive multiple of 4 (one uint32 a unit).");
        }

        var units = new uint[bytes.Length / 4];
        MemoryMarshal.Cast<byte, uint>(bytes).CopyTo(units);
        if (!BitConverter.IsLittleEndian)
        {
            BinaryPrimitives.ReverseEndianness(units, units);
        }

        return new DoubleArrayTrie(units);
    }

    /// <summary>
    /// Builds a trie from keys sorted in ordinal byte order (strictly ascending: no duplicates), each non-empty and without
    /// a 0 byte, with values from 0 to <see cref="int.MaxValue"/>. The work is done once; the builder's arrays are let go.
    /// </summary>
    public static DoubleArrayTrie Build(byte[][] keys, int[] values)
    {
        ArgumentNullException.ThrowIfNull(keys);
        ArgumentNullException.ThrowIfNull(values);
        if (keys.Length != values.Length)
        {
            throw new ArgumentException($"{keys.Length} keys but {values.Length} values.", nameof(values));
        }

        for (int i = 0; i < keys.Length; i++)
        {
            var key = keys[i] ?? throw new ArgumentException($"Key {i} is null.", nameof(keys));
            if (key.Length == 0)
            {
                throw new ArgumentException($"Key {i} is empty.", nameof(keys));
            }

            if (key.AsSpan().IndexOf((byte)0) is int zero and >= 0)
            {
                throw new ArgumentException($"Key {i} has a 0 byte at {zero} (a trie key cannot hold one).", nameof(keys));
            }

            if (values[i] < 0)
            {
                throw new ArgumentException($"Value {i} is {values[i]} (values are 0 to {int.MaxValue}).", nameof(values));
            }

            if (i > 0 && keys[i - 1].AsSpan().SequenceCompareTo(key) >= 0)
            {
                throw new ArgumentException(
                    keys[i - 1].AsSpan().SequenceEqual(key)
                        ? $"Key {i} repeats key {i - 1}."
                        : $"Key {i} sorts before key {i - 1} (keys must be in ordinal byte order).", nameof(keys));
            }
        }

        return new DoubleArrayTrie(new Builder(keys, values).Run());
    }

    /// <summary>
    /// Builds a trie from strings (encoded as UTF-8 and sorted here) with their values; a string that repeats another, is
    /// empty or holds U+0000 is refused.
    /// </summary>
    public static DoubleArrayTrie Build(IReadOnlyList<string> keys, IReadOnlyList<int> values)
    {
        ArgumentNullException.ThrowIfNull(keys);
        ArgumentNullException.ThrowIfNull(values);
        if (keys.Count != values.Count)
        {
            throw new ArgumentException($"{keys.Count} keys but {values.Count} values.", nameof(values));
        }

        var bytes = new byte[keys.Count][];
        var order = new int[keys.Count];
        for (int i = 0; i < bytes.Length; i++)
        {
            bytes[i] = Encoding.UTF8.GetBytes(keys[i] ?? throw new ArgumentException($"Key {i} is null.", nameof(keys)));
            order[i] = i;
        }

        Array.Sort(order, (a, b) => bytes[a].AsSpan().SequenceCompareTo(bytes[b]) is var c && c != 0 ? c : a.CompareTo(b));
        var sortedKeys = new byte[order.Length][];
        var sortedValues = new int[order.Length];
        for (int i = 0; i < order.Length; i++)
        {
            sortedKeys[i] = bytes[order[i]];
            sortedValues[i] = values[order[i]];
            if (i > 0 && sortedKeys[i - 1].AsSpan().SequenceEqual(sortedKeys[i]))
            {
                throw new ArgumentException($"Key {order[i]} ('{keys[order[i]]}') repeats key {order[i - 1]}.", nameof(keys));
            }
        }

        return Build(sortedKeys, sortedValues);
    }

    /// <summary>
    /// Finds every key that is a prefix of <paramref name="key"/>, shortest first, in one walk. Writes up to
    /// <paramref name="matches"/>' length of them and returns how many there are (which may be more). Allocates nothing.
    /// </summary>
    public int CommonPrefixSearch(ReadOnlySpan<byte> key, Span<TrieMatch> matches)
    {
        var units = _units;
        int found = 0;
        uint node = Offset(units[0]);
        for (int i = 0; i < key.Length; i++)
        {
            uint label = key[i];
            node ^= label;
            if (node >= (uint)units.Length)
            {
                break;
            }

            uint unit = units[node];
            if ((unit & LabelMask) != label)
            {
                break;
            }

            node ^= Offset(unit);
            if ((unit & HasLeafBit) != 0 && node < (uint)units.Length)
            {
                if (found < matches.Length)
                {
                    matches[found] = new TrieMatch((int)(units[node] & ValueMask), i + 1);
                }

                found++;
            }
        }

        return found;
    }

    /// <summary>The value of exactly <paramref name="key"/>, or -1 when it is not a key. Allocates nothing.</summary>
    public int ExactMatch(ReadOnlySpan<byte> key)
    {
        var units = _units;
        uint node = Offset(units[0]);
        uint unit = 0;
        for (int i = 0; i < key.Length; i++)
        {
            uint label = key[i];
            node ^= label;
            if (node >= (uint)units.Length)
            {
                return -1;
            }

            unit = units[node];
            if ((unit & LabelMask) != label)
            {
                return -1;
            }

            node ^= Offset(unit);
        }

        return key.Length > 0 && (unit & HasLeafBit) != 0 && node < (uint)units.Length ? (int)(units[node] & ValueMask) : -1;
    }

    /// <summary>
    /// Every key and its value, in ordinal byte order, found by walking every path (for checks and diagnostics; it
    /// allocates). A path longer than <paramref name="maxKeyLength"/> bytes means the units loop, and is refused.
    /// </summary>
    public List<(byte[] Key, int Value)> Entries(int maxKeyLength = 4096)
    {
        var units = _units;
        var result = new List<(byte[] Key, int Value)>();
        var stack = new Stack<(uint Node, byte[] Key)>();
        stack.Push((Offset(units[0]), []));
        while (stack.Count > 0)
        {
            var (node, key) = stack.Pop();
            for (uint c = 1; c < 256; c++)
            {
                uint child = node ^ c;
                if (child >= (uint)units.Length || (units[child] & LabelMask) != c)
                {
                    continue;
                }

                if (key.Length >= maxKeyLength)
                {
                    throw new InvalidDataException($"Double-array trie: a path longer than {maxKeyLength} bytes (the units loop) at unit {child}.");
                }

                byte[] longer = [.. key, (byte)c];
                uint next = child ^ Offset(units[child]);
                if ((units[child] & HasLeafBit) != 0 && next < (uint)units.Length)
                {
                    result.Add((longer, (int)(units[next] & ValueMask)));
                }

                stack.Push((next, longer));
            }
        }

        result.Sort((a, b) => a.Key.AsSpan().SequenceCompareTo(b.Key));
        return result;
    }

    // The XOR offset a unit stores: bits 10-31, shifted left 8 more when bit 9 is set.
    private static uint Offset(uint unit) => (unit >> 10) << (int)((unit & (1u << 9)) >> 6);

    /// <summary>
    /// darts-clone's construction on sorted keys: depth first, each node's children placed in one of the last
    /// <see cref="OpenBlocks"/> blocks of 256 units (a base's children all lie in its block, since a byte changes only the
    /// low 8 bits), at a base no other node uses, so a label check is enough to tell nodes apart.
    /// </summary>
    private sealed class Builder(byte[][] keys, int[] values)
    {
        private uint[] _units = new uint[BlockSize];
        private bool[] _used = new bool[BlockSize];        // positions holding a unit
        private bool[] _bases = new bool[BlockSize];       // positions used as a base
        private int[] _free = [BlockSize];                 // free positions per block
        private int _blocks = 1;

        public uint[] Run()
        {
            _used[0] = true;                                   // the root
            _free[0]--;

            // Base 0 is never given out: the root's unit has label 0, so from base 0 a 0 byte would reach the root and
            // match it, and a search would start over inside the key. (An empty trie's root keeps offset 0, so its base
            // is 0 as well, and the unused units' labels are fixed against it.)
            _bases[0] = true;
            var labels = new int[257];
            var starts = new int[258];
            var work = new Stack<(int Node, int Low, int High, int Depth)>();
            if (keys.Length > 0)
            {
                work.Push((0, 0, keys.Length, 0));
            }

            while (work.Count > 0)
            {
                var (node, low, high, depth) = work.Pop();
                int count = 0;
                int i = low;
                if (keys[low].Length == depth)
                {
                    labels[count] = 0;                         // the key that ends here: its value goes at the base
                    starts[count++] = low;
                    i++;
                }

                while (i < high)
                {
                    byte c = keys[i][depth];
                    labels[count] = c;
                    starts[count++] = i;
                    while (i < high && keys[i][depth] == c)
                    {
                        i++;
                    }
                }

                starts[count] = high;
                int @base = FindBase(node, labels.AsSpan(0, count));
                _bases[@base] = true;
                _units[node] |= EncodeOffset((uint)(node ^ @base));
                for (int k = count - 1; k >= 0; k--)
                {
                    int position = @base ^ labels[k];
                    _used[position] = true;
                    _free[position / BlockSize]--;
                    if (labels[k] == 0)
                    {
                        _units[position] = (uint)values[starts[k]] | 0x8000_0000u;
                        _units[node] |= HasLeafBit;
                    }
                    else
                    {
                        _units[position] = (uint)labels[k];
                        work.Push((position, starts[k], starts[k + 1], depth + 1));
                    }
                }
            }

            FixUnused();
            return _units;
        }

        private int FindBase(int node, ReadOnlySpan<int> labels)
        {
            for (int block = Math.Max(0, _blocks - OpenBlocks); block < _blocks; block++)
            {
                if (TryBlock(block, node, labels, out int found))
                {
                    return found;
                }
            }

            AddBlock();
            return TryBlock(_blocks - 1, node, labels, out int fresh)
                ? fresh
                : throw new InvalidOperationException($"Double-array trie: {_blocks} blocks of 256 units are more than the offsets can reach.");
        }

        private bool TryBlock(int block, int node, ReadOnlySpan<int> labels, out int found)
        {
            found = 0;
            if (_free[block] < labels.Length)
            {
                return false;
            }

            int start = block * BlockSize;
            int distance = node ^ start;                   // the offset's bits above the low 8
            bool near = distance < NearOffsets;
            if (!near && distance >= FarOffsets)
            {
                return false;
            }

            // A far offset must be a multiple of 256: the base's low byte must equal the node's.
            int first = near ? 0 : node & 0xFF;
            int last = near ? BlockSize - 1 : node & 0xFF;
            for (int low = first; low <= last; low++)
            {
                int candidate = start | low;
                if (_bases[candidate])
                {
                    continue;
                }

                bool fits = true;
                foreach (int label in labels)
                {
                    if (_used[candidate ^ label])
                    {
                        fits = false;
                        break;
                    }
                }

                if (fits)
                {
                    found = candidate;
                    return true;
                }
            }

            return false;
        }

        private void AddBlock()
        {
            int size = (_blocks + 1) * BlockSize;
            if (size > _units.Length)
            {
                int capacity = Math.Max(size, _units.Length * 2);
                Array.Resize(ref _units, capacity);
                Array.Resize(ref _used, capacity);
                Array.Resize(ref _bases, capacity);
                Array.Resize(ref _free, capacity / BlockSize);
            }

            _free[_blocks++] = BlockSize;
        }

        // Unused units get labels that no search byte can match there: in each block, a base nobody uses (u), and every
        // unused position p labelled p ^ u, so from any used base b in the block a byte c reaches b ^ c with label b ^ c ^ u,
        // which is not c because b and u differ in their low byte. Then the array is cut to whole blocks.
        private void FixUnused()
        {
            for (int block = 0; block < _blocks; block++)
            {
                int start = block * BlockSize;
                int spare = -1;
                for (int low = 0; low < BlockSize && spare < 0; low++)
                {
                    spare = _bases[start | low] ? -1 : start | low;
                }

                for (int position = start; position < start + BlockSize && spare >= 0; position++)
                {
                    if (!_used[position])
                    {
                        _units[position] = (uint)((position ^ spare) & 0xFF);
                    }
                }
            }

            Array.Resize(ref _units, _blocks * BlockSize);
        }

        private static uint EncodeOffset(uint offset) =>
            offset < NearOffsets ? offset << 10 : (offset << 2) | (1u << 9);
    }
}
