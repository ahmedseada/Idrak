// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Security.Cryptography;

namespace Idrak.Abstraction.Cluster;

/// <summary>What a message on the wire is.</summary>
public enum WireMessageKind : ushort
{
    /// <summary>A rank's first message: its rank, the world size and the HMAC of a nonce under the key.</summary>
    Hello = 1,

    /// <summary>The coordinator's answer: every rank's address, in rank order.</summary>
    Welcome = 2,

    /// <summary>A piece of a transfer.</summary>
    Chunk = 3,

    /// <summary>A chunk received.</summary>
    Ack = 4,

    /// <summary>A rank reached a barrier.</summary>
    Barrier = 5,

    /// <summary>A rank is alive.</summary>
    Heartbeat = 6,

    /// <summary>A rank leaves.</summary>
    Bye = 7,

    /// <summary>A rank failed; the payload says why, in UTF-8.</summary>
    Error = 8,
}

/// <summary>What a chunk's payload holds.</summary>
public enum WireElement : byte
{
    /// <summary>Bytes as they are.</summary>
    Bytes = 0,

    /// <summary>Float32 values, little-endian.</summary>
    Float32 = 1,

    /// <summary>BFloat16 values, little-endian.</summary>
    BFloat16 = 2,
}

/// <summary>A message's header (<see cref="WireFormat"/>).</summary>
/// <param name="Kind">What the message is.</param>
/// <param name="Rank">The sending rank.</param>
/// <param name="Tag">The tag it was sent under.</param>
/// <param name="Sequence">The sender's message number for this peer and tag.</param>
/// <param name="ChunkIndex">The chunk's place in its transfer.</param>
/// <param name="ChunkCount">The transfer's chunks.</param>
/// <param name="Element">What the payload holds.</param>
/// <param name="Op">The reduction the payload takes part in.</param>
/// <param name="Flags">Bits a transport may use; 0 when none.</param>
/// <param name="PayloadBytes">The bytes after the header.</param>
public readonly record struct WireHeader(WireMessageKind Kind, int Rank, int Tag, uint Sequence, uint ChunkIndex, uint ChunkCount,
    WireElement Element, ReduceOp Op, ushort Flags, uint PayloadBytes);

/// <summary>
/// The bytes every cluster transport speaks (plan 23), public so a plug-in transport speaks them too: a 36-byte
/// little-endian header, then the payload. Layout: magic "IDRK" (4), version (u16), kind (u16), rank (i32), tag (i32),
/// sequence (u32), chunk index (u32), chunk count (u32), element (u8), op (u8), flags (u16), payload bytes (u32). A peer
/// proves it holds the cluster's key in its hello: an HMAC-SHA256 of (nonce, rank, world size) under the key.
/// </summary>
public static class WireFormat
{
    /// <summary>The header's bytes.</summary>
    public const int HeaderBytes = 36;

    /// <summary>The version this library writes and the newest it reads.</summary>
    public const ushort Version = 1;

    /// <summary>The bytes of a hello's nonce.</summary>
    public const int NonceBytes = 32;

    /// <summary>The bytes of a hello's HMAC.</summary>
    public const int MacBytes = 32;

    private const uint Magic = 0x4B524449;   // "IDRK" read little-endian

    /// <summary>Writes <paramref name="header"/> into the first <see cref="HeaderBytes"/> of <paramref name="destination"/>.</summary>
    /// <exception cref="ArgumentException"><paramref name="destination"/> is shorter than a header.</exception>
    public static void WriteHeader(Span<byte> destination, in WireHeader header)
    {
        if (destination.Length < HeaderBytes)
        {
            throw new ArgumentException($"A header needs {HeaderBytes} bytes; the buffer has {destination.Length}.", nameof(destination));
        }

        BinaryPrimitives.WriteUInt32LittleEndian(destination, Magic);
        BinaryPrimitives.WriteUInt16LittleEndian(destination[4..], Version);
        BinaryPrimitives.WriteUInt16LittleEndian(destination[6..], (ushort)header.Kind);
        BinaryPrimitives.WriteInt32LittleEndian(destination[8..], header.Rank);
        BinaryPrimitives.WriteInt32LittleEndian(destination[12..], header.Tag);
        BinaryPrimitives.WriteUInt32LittleEndian(destination[16..], header.Sequence);
        BinaryPrimitives.WriteUInt32LittleEndian(destination[20..], header.ChunkIndex);
        BinaryPrimitives.WriteUInt32LittleEndian(destination[24..], header.ChunkCount);
        destination[28] = (byte)header.Element;
        destination[29] = (byte)header.Op;
        BinaryPrimitives.WriteUInt16LittleEndian(destination[30..], header.Flags);
        BinaryPrimitives.WriteUInt32LittleEndian(destination[32..], header.PayloadBytes);
    }

    /// <summary>
    /// Reads a header from the start of <paramref name="source"/>; false when fewer than <see cref="HeaderBytes"/> bytes
    /// have arrived yet (read more and call again).
    /// </summary>
    /// <exception cref="InvalidDataException">The bytes are not an Idrak cluster message, or a newer version wrote them; the message names both versions.</exception>
    public static bool TryReadHeader(ReadOnlySpan<byte> source, out WireHeader header)
    {
        header = default;
        if (source.Length < HeaderBytes)
        {
            return false;
        }

        if (BinaryPrimitives.ReadUInt32LittleEndian(source) != Magic)
        {
            throw new InvalidDataException("Not an Idrak cluster message (no \"IDRK\" at its start): a peer that is not an Idrak rank, or a stream out of step.");
        }

        ushort version = BinaryPrimitives.ReadUInt16LittleEndian(source[4..]);
        if (version is 0 or > Version)
        {
            throw new InvalidDataException($"A peer speaks cluster wire version {version}; this Idrak speaks version {Version}. Run the same Idrak version on every rank.");
        }

        ushort kind = BinaryPrimitives.ReadUInt16LittleEndian(source[6..]);
        byte element = source[28], op = source[29];
        if (!Enum.IsDefined((WireMessageKind)kind) || !Enum.IsDefined((WireElement)element) || !Enum.IsDefined((ReduceOp)op))
        {
            throw new InvalidDataException($"A cluster message has an unknown kind {kind}, element {element} or reduction {op} for version {version}.");
        }

        header = new WireHeader((WireMessageKind)kind, BinaryPrimitives.ReadInt32LittleEndian(source[8..]), BinaryPrimitives.ReadInt32LittleEndian(source[12..]),
            BinaryPrimitives.ReadUInt32LittleEndian(source[16..]), BinaryPrimitives.ReadUInt32LittleEndian(source[20..]),
            BinaryPrimitives.ReadUInt32LittleEndian(source[24..]), (WireElement)element, (ReduceOp)op, BinaryPrimitives.ReadUInt16LittleEndian(source[30..]),
            BinaryPrimitives.ReadUInt32LittleEndian(source[32..]));
        return true;
    }

    /// <summary>Writes into <paramref name="mac"/> the HMAC-SHA256 under <paramref name="key"/> of (nonce, rank, world size), the proof a hello carries.</summary>
    /// <exception cref="ArgumentException">The nonce is not <see cref="NonceBytes"/> bytes, or <paramref name="mac"/> is shorter than <see cref="MacBytes"/>.</exception>
    public static void SignHello(ClusterKey key, ReadOnlySpan<byte> nonce, int rank, int worldSize, Span<byte> mac)
    {
        ArgumentNullException.ThrowIfNull(key);
        if (mac.Length < MacBytes)
        {
            throw new ArgumentException($"An HMAC needs {MacBytes} bytes.", nameof(mac));
        }

        Span<byte> message = stackalloc byte[NonceBytes + 8];
        HelloMessage(nonce, rank, worldSize, message);
        HMACSHA256.HashData(key.Bytes, message, mac);
    }

    /// <summary>Whether <paramref name="mac"/> proves the sender holds <paramref name="key"/> (compared in constant time).</summary>
    public static bool VerifyHello(ClusterKey key, ReadOnlySpan<byte> nonce, int rank, int worldSize, ReadOnlySpan<byte> mac)
    {
        ArgumentNullException.ThrowIfNull(key);
        if (mac.Length != MacBytes || nonce.Length != NonceBytes)
        {
            return false;
        }

        Span<byte> expected = stackalloc byte[MacBytes];
        SignHello(key, nonce, rank, worldSize, expected);
        return CryptographicOperations.FixedTimeEquals(expected, mac);
    }

    /// <summary>A fresh random nonce for a hello.</summary>
    public static void NewNonce(Span<byte> nonce)
    {
        if (nonce.Length != NonceBytes)
        {
            throw new ArgumentException($"A nonce is {NonceBytes} bytes.", nameof(nonce));
        }

        RandomNumberGenerator.Fill(nonce);
    }

    private static void HelloMessage(ReadOnlySpan<byte> nonce, int rank, int worldSize, Span<byte> message)
    {
        if (nonce.Length != NonceBytes)
        {
            throw new ArgumentException($"A nonce is {NonceBytes} bytes.", nameof(nonce));
        }

        nonce.CopyTo(message);
        BinaryPrimitives.WriteInt32LittleEndian(message[NonceBytes..], rank);
        BinaryPrimitives.WriteInt32LittleEndian(message[(NonceBytes + 4)..], worldSize);
    }
}
