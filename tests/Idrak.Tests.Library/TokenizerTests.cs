// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Testing;
using Idrak.Models;
using Idrak.Models.Abstractions;

// Plan 24, phase L0: the tokenizer models, the file readers and the one loader.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] TokenizerGroup =
    [
        ("tokenizers: the tokenizer.json reader reads a list vocabulary as pieces and scores, the model's other fields kept in its JSON", TokenizersFileReaders),
    ];

    // As tools wrote bpe-baseline.json: SHA-256 of each text's id count and ids (int32 little-endian), or of "!", the
    // refusal's type and message and a NUL; and of each decoded text (UTF-8) followed by a NUL.
    internal static (string Ids, string Decoded) BaselineHashes(IPretrainedTokenizer tokenizer, string[] texts)
    {
        using var ids = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        using var decoded = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var four = new byte[4];
        foreach (string text in texts)
        {
            IReadOnlyList<int> encoded;
            try
            {
                encoded = tokenizer.Encode(text);
            }
            catch (Exception e) when (e is InvalidDataException or NotSupportedException)
            {
                ids.AppendData(Encoding.UTF8.GetBytes("!" + e.GetType().Name + ": " + e.Message + "\u0000"));
                continue;
            }

            BitConverter.TryWriteBytes(four, encoded.Count);
            ids.AppendData(four);
            foreach (int id in encoded)
            {
                BitConverter.TryWriteBytes(four, id);
                ids.AppendData(four);
            }

            decoded.AppendData(Encoding.UTF8.GetBytes(tokenizer.Decode(encoded) + "\u0000"));
        }

        return (Convert.ToHexStringLower(ids.GetHashAndReset()), Convert.ToHexStringLower(decoded.GetHashAndReset()));
    }

    internal static void TokenizersFileReaders(Device device)
    {
        _ = device;
        var temporary = Directory.CreateTempSubdirectory("idrak-tokenizers-");
        try
        {
            // A Unigram tokenizer.json: the list vocabulary becomes pieces and scores.
            string unigram = Path.Combine(temporary.FullName, "unigram.json");
            File.WriteAllText(unigram, """
                {"added_tokens": [], "normalizer": null, "pre_tokenizer": null, "post_processor": null, "decoder": null,
                 "model": {"type": "Unigram", "unk_id": 0, "vocab": [["<unk>", 0.0], ["▁a", -1.5], ["b", -2.25]], "byte_fallback": false}}
                """);
            var definition = new TokenizerJsonReader().Read(unigram, new TokenizerOptions());
            Check(definition.ModelType == "Unigram" && definition.Vocabulary is null && definition.Pieces is { Count: 3 } pieces
                  && pieces[1] == ("▁a", -1.5) && pieces[2] == ("b", -2.25), "a list vocabulary reads to pieces and scores in id order");
            Check((int?)definition.Json["model"]?["unk_id"] == 0, "the model's other fields stay in its JSON");
        }
        finally
        {
            temporary.Delete(recursive: true);
        }
    }
}
