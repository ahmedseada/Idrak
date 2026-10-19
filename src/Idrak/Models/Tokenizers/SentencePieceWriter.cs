// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Text;

namespace Idrak.Models;

/// <summary>
/// Writes a SentencePiece <c>ModelProto</c> (the bytes of a tokenizer.model) from its pieces and the trainer and normalizer
/// fields encoding reads, with sentencepiece 0.2.2's field numbers (<see cref="SentencePieceModel"/> reads them back). GGUF
/// files hold a SentencePiece tokenizer as these fields (plan 24, T3); the trainers write their models with it (T4).
/// </summary>
internal static class SentencePieceWriter
{
    // What a ModelProto needs beyond its pieces.
    public sealed record Settings
    {
        public SentencePieceAlgorithm Algorithm { get; init; } = SentencePieceAlgorithm.Unigram;
        public bool ByteFallback { get; init; }
        public int UnkId { get; init; }
        public int BosId { get; init; } = -1;
        public int EosId { get; init; } = -1;
        public int PadId { get; init; } = -1;
        public bool TreatWhitespaceAsSuffix { get; init; }
        public string NormalizerName { get; init; } = "identity";
        public byte[] PrecompiledCharsmap { get; init; } = [];
        public bool AddDummyPrefix { get; init; } = true;
        public bool RemoveExtraWhitespaces { get; init; } = true;
        public bool EscapeWhitespaces { get; init; } = true;
    }

    public static byte[] Write(IReadOnlyList<string> pieces, IReadOnlyList<float> scores, IReadOnlyList<SentencePieceType> types, Settings settings)
    {
        ArgumentNullException.ThrowIfNull(pieces);
        ArgumentNullException.ThrowIfNull(scores);
        ArgumentNullException.ThrowIfNull(types);
        ArgumentNullException.ThrowIfNull(settings);
        if (scores.Count != pieces.Count || types.Count != pieces.Count)
        {
            throw new ArgumentException($"{pieces.Count} pieces, {scores.Count} scores and {types.Count} types.");
        }

        using var model = new MemoryStream();
        using var piece = new MemoryStream();
        for (int i = 0; i < pieces.Count; i++)
        {
            piece.SetLength(0);
            Bytes(piece, 1, Encoding.UTF8.GetBytes(pieces[i]));
            Float(piece, 2, scores[i]);
            Varint(piece, 3, (ulong)types[i]);
            Bytes(model, 1, piece.ToArray());
        }

        using var trainer = new MemoryStream();
        Varint(trainer, 3, (ulong)settings.Algorithm);
        Varint(trainer, 24, settings.TreatWhitespaceAsSuffix ? 1UL : 0UL);
        Varint(trainer, 35, settings.ByteFallback ? 1UL : 0UL);
        Varint(trainer, 40, (ulong)(long)settings.UnkId);                   // negative ids as ten-byte varints, as protobuf writes int32
        Varint(trainer, 41, (ulong)(long)settings.BosId);
        Varint(trainer, 42, (ulong)(long)settings.EosId);
        Varint(trainer, 43, (ulong)(long)settings.PadId);
        foreach (var (field, id) in new[] { (45, settings.UnkId), (46, settings.BosId), (47, settings.EosId), (48, settings.PadId) })
        {
            if (id >= 0 && id < pieces.Count)
            {
                Bytes(trainer, field, Encoding.UTF8.GetBytes(pieces[id]));
            }
        }

        Bytes(model, 2, trainer.ToArray());

        using var normalizer = new MemoryStream();
        Bytes(normalizer, 1, Encoding.UTF8.GetBytes(settings.NormalizerName));
        if (settings.PrecompiledCharsmap.Length > 0)
        {
            Bytes(normalizer, 2, settings.PrecompiledCharsmap);
        }

        Varint(normalizer, 3, settings.AddDummyPrefix ? 1UL : 0UL);
        Varint(normalizer, 4, settings.RemoveExtraWhitespaces ? 1UL : 0UL);
        Varint(normalizer, 5, settings.EscapeWhitespaces ? 1UL : 0UL);
        Bytes(model, 3, normalizer.ToArray());
        return model.ToArray();
    }

    private static void Key(Stream stream, int field, int wire) => Raw(stream, ((ulong)field << 3) | (uint)wire);

    private static void Varint(Stream stream, int field, ulong value)
    {
        Key(stream, field, 0);
        Raw(stream, value);
    }

    private static void Float(Stream stream, int field, float value)
    {
        Key(stream, field, 5);
        Span<byte> bytes = stackalloc byte[4];
        BinaryPrimitives.WriteSingleLittleEndian(bytes, value);
        stream.Write(bytes);
    }

    private static void Bytes(Stream stream, int field, ReadOnlySpan<byte> value)
    {
        Key(stream, field, 2);
        Raw(stream, (ulong)value.Length);
        stream.Write(value);
    }

    private static void Raw(Stream stream, ulong value)
    {
        while (value >= 0x80)
        {
            stream.WriteByte((byte)(value | 0x80));
            value >>= 7;
        }

        stream.WriteByte((byte)value);
    }
}
