// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// Reads a SentencePiece model file (<c>tokenizer.model</c>, <c>spiece.model</c>, <c>sentencepiece.bpe.model</c>) into the
/// definition a tokenizer.json of the same tokenizer would give, in the shape transformers' converters write: a Unigram
/// model as "Unigram" (pieces and scores), a BPE model as "BPE" (vocabulary, and merges derived from the scores), the
/// control, unknown and user-defined pieces as added tokens. The pipeline is chosen to reproduce SentencePiece itself
/// (what <c>sentencepiece</c> gives for the file), not a converted tokenizer.json:
/// <list type="bullet">
/// <item>the charsmap as a <c>Precompiled</c> normalizer in SentencePiece's mode (<c>"mode": "sentencepiece"</c>);</item>
/// <item>the normalizer spec's whitespace handling as standard normalizers: extra spaces removed (<c>Replace</c> with
/// <c>\A +</c>, <c> {2,}</c>, and the trailing space symbols), the dummy prefix (<c>Prepend</c>), spaces escaped as ▁
/// (<c>Replace</c>);</item>
/// <item>a <c>Metaspace</c> pre-tokenizer that only splits before ▁, when no piece holds ▁ past its first character (then
/// encoding word by word gives what SentencePiece gives for the whole text), else none;</item>
/// <item>a <c>Metaspace</c> decoder for a Unigram model without byte fallback, else Llama's sequence (<c>Replace</c> ▁,
/// <c>ByteFallback</c>, <c>Fuse</c>, <c>Strip</c>); the unknown piece decodes as SentencePiece writes it (" ⁇ ").</item>
/// </list>
/// No post-processor: SentencePiece adds no BOS or EOS by default. Every component is one Hugging Face's tokenizers reads,
/// so a saved tokenizer.json loads there too (with its own Precompiled mode).
/// </summary>
internal sealed class SentencePieceReader : ITokenizerFileReader
{
    // The names model folders give the file, in the order they are looked for.
    private static readonly string[] FileNames = ["tokenizer.model", "spiece.model", "sentencepiece.bpe.model"];

    private const string Space = "▁";

    public string Name => TokenizerFiles.SentencePiece;

    public string Summary => "SentencePiece model (tokenizer.model, spiece.model, sentencepiece.bpe.model): Unigram or BPE";

    public string? Find(string path)
    {
        if (Directory.Exists(path))
        {
            foreach (var name in FileNames)
            {
                string file = Path.Combine(path, name);
                if (File.Exists(file))
                {
                    return file;
                }
            }

            return null;
        }

        return File.Exists(path) && path.EndsWith(".model", StringComparison.OrdinalIgnoreCase) ? path : null;
    }

    public TokenizerDefinition Read(string file, TokenizerOptions options) => FromModel(SentencePieceModel.Load(file));

    // The definition of a model already read.
    internal static TokenizerDefinition FromModel(SentencePieceModel model)
    {
        var trainer = model.TrainerSpec;
        var spec = model.NormalizerSpec;
        bool suffix = trainer.TreatWhitespaceAsSuffix;
        int unknown = UnknownId(model);
        var json = new JsonObject
        {
            ["version"] = "1.0",
            ["truncation"] = null,
            ["padding"] = null,
            ["added_tokens"] = AddedTokens(model),
            ["normalizer"] = Normalizer(spec, suffix),
            ["pre_tokenizer"] = spec.EscapeWhitespaces && !suffix && !SpaceInside(model.Pieces)
                ? new JsonObject { ["type"] = "Metaspace", ["replacement"] = Space, ["prepend_scheme"] = "never", ["split"] = true }
                : null,
            ["post_processor"] = null,
            ["decoder"] = Decoder(model, unknown, suffix),
        };

        switch (trainer.ModelType)
        {
            case SentencePieceAlgorithm.Unigram:
                json["model"] = new JsonObject
                {
                    ["type"] = "Unigram",
                    ["unk_id"] = unknown >= 0 ? unknown : null,
                    ["byte_fallback"] = trainer.ByteFallback,
                };
                var pieces = new (string, double)[model.Count];
                for (int id = 0; id < pieces.Length; id++)
                {
                    pieces[id] = (model.Pieces[id], model.Scores[id]);
                }

                return new TokenizerDefinition { Source = model.Source, Json = json, Pieces = pieces };

            case SentencePieceAlgorithm.Bpe:
                json["model"] = new JsonObject
                {
                    ["type"] = TokenizerModels.Bpe,
                    ["dropout"] = null,
                    ["unk_token"] = unknown >= 0 ? model.Pieces[unknown] : null,
                    ["continuing_subword_prefix"] = null,
                    ["end_of_word_suffix"] = null,
                    ["fuse_unk"] = true,
                    ["byte_fallback"] = trainer.ByteFallback,
                    ["ignore_merges"] = false,
                };
                var vocabulary = new Dictionary<string, int>(model.Count);
                for (int id = 0; id < model.Count; id++)
                {
                    vocabulary[model.Pieces[id]] = id;
                }

                return new TokenizerDefinition { Source = model.Source, Json = json, Vocabulary = vocabulary, Merges = Merges(model, vocabulary) };

            default:
                throw new NotSupportedException(
                    $"{model.Source} is a SentencePiece {trainer.ModelType.ToString().ToUpperInvariant()} model; the sentencepiece reader reads UNIGRAM and BPE models.");
        }
    }

    // The unknown piece's id: the piece typed unknown, else the trainer's unk_id when it is a piece; -1 when there is none.
    private static int UnknownId(SentencePieceModel model)
    {
        int at = Array.IndexOf(model.Types, SentencePieceType.Unknown);
        return at >= 0 ? at : (uint)model.TrainerSpec.UnkId < (uint)model.Count ? model.TrainerSpec.UnkId : -1;
    }

    // Control, unknown and user-defined pieces, as the converters list them: matched as written (not normalized), special
    // but for the user-defined ones.
    private static JsonArray AddedTokens(SentencePieceModel model)
    {
        var added = new JsonArray();
        for (int id = 0; id < model.Count; id++)
        {
            var type = model.Types[id];
            if (type is SentencePieceType.Control or SentencePieceType.Unknown or SentencePieceType.UserDefined)
            {
                added.Add((JsonNode)new JsonObject
                {
                    ["id"] = id,
                    ["content"] = model.Pieces[id],
                    ["single_word"] = false,
                    ["lstrip"] = false,
                    ["rstrip"] = false,
                    ["normalized"] = false,
                    ["special"] = type != SentencePieceType.UserDefined,
                });
            }
        }

        return added;
    }

    // SentencePiece's normalizer (normalizer.cc) as standard normalizers, in its order: the charsmap; leading spaces
    // dropped and runs of spaces made one; the dummy prefix; spaces escaped; trailing space symbols dropped; the dummy
    // suffix. Null when there is nothing to do.
    private static JsonObject? Normalizer(SentencePieceNormalizerSpec spec, bool suffix)
    {
        string space = spec.EscapeWhitespaces ? Space : " ";
        var steps = new JsonArray();
        if (spec.PrecompiledCharsmap.Length > 0)
        {
            steps.Add((JsonNode)new JsonObject
            {
                ["type"] = "Precompiled",
                ["precompiled_charsmap"] = Convert.ToBase64String(spec.PrecompiledCharsmap),
                [PrecompiledCharsMap.ModeField] = PrecompiledCharsMap.SentencePieceMode,
            });
        }

        if (spec.RemoveExtraWhitespaces)
        {
            steps.Add((JsonNode)Replace(@"\A +", "", regex: true));
            steps.Add((JsonNode)Replace(" {2,}", " ", regex: true));
        }

        if (spec.AddDummyPrefix && !suffix)
        {
            steps.Add((JsonNode)new JsonObject { ["type"] = "Prepend", ["prepend"] = space });   // not on empty text, as SentencePiece
        }

        if (spec.EscapeWhitespaces)
        {
            steps.Add((JsonNode)Replace(" ", Space, regex: false));
        }

        if (spec.RemoveExtraWhitespaces)
        {
            steps.Add((JsonNode)Replace($"(?:{space})+\\z", "", regex: true));
        }

        if (spec.AddDummyPrefix && suffix)
        {
            steps.Add((JsonNode)Replace(@"(?<=[\s\S])\z", space, regex: true));
        }

        return steps.Count == 0 ? null : new JsonObject { ["type"] = "Sequence", ["normalizers"] = steps };
    }

    private static JsonObject Replace(string pattern, string content, bool regex) => new()
    {
        ["type"] = "Replace",
        ["pattern"] = new JsonObject { [regex ? "Regex" : "String"] = pattern },
        ["content"] = content,
    };

    // Whether some piece holds the space symbol past its first character (then splitting at ▁ could cut a piece).
    private static bool SpaceInside(string[] pieces)
    {
        foreach (var piece in pieces)
        {
            if (piece.Length > 1 && piece.AsSpan(1).Contains(Space[0]))
            {
                return true;
            }
        }

        return false;
    }

    // SentencePiece's decoding: ▁ back to spaces, bytes back to text, the dummy prefix's space dropped (or the suffix's),
    // and the unknown piece as its surface (" ⁇ ").
    private static JsonObject Decoder(SentencePieceModel model, int unknown, bool suffix)
    {
        var spec = model.NormalizerSpec;
        bool leading = !suffix && (spec.AddDummyPrefix || spec.RemoveExtraWhitespaces);
        var steps = new JsonArray();
        if (unknown >= 0 && model.Pieces[unknown] != model.TrainerSpec.UnkSurface)
        {
            steps.Add((JsonNode)Replace(model.Pieces[unknown], model.TrainerSpec.UnkSurface, regex: false));
        }

        if (model.TrainerSpec.ModelType == SentencePieceAlgorithm.Unigram && !model.TrainerSpec.ByteFallback && !suffix && spec.EscapeWhitespaces)
        {
            steps.Add((JsonNode)new JsonObject { ["type"] = "Metaspace", ["replacement"] = Space, ["prepend_scheme"] = leading ? "always" : "never", ["split"] = true });
        }
        else
        {
            if (spec.EscapeWhitespaces)
            {
                steps.Add((JsonNode)Replace(Space, " ", regex: false));
            }

            if (model.TrainerSpec.ByteFallback)
            {
                steps.Add((JsonNode)new JsonObject { ["type"] = "ByteFallback" });
            }

            steps.Add((JsonNode)new JsonObject { ["type"] = "Fuse" });
            if (leading || (suffix && spec.AddDummyPrefix))
            {
                steps.Add((JsonNode)new JsonObject { ["type"] = "Strip", ["content"] = " ", ["start"] = leading ? 1 : 0, ["stop"] = leading ? 0 : 1 });
            }
        }

        return steps.Count == 1 ? (JsonObject)steps[0]!.DeepClone() : new JsonObject { ["type"] = "Sequence", ["decoders"] = steps };
    }

    // The merges of a SentencePiece BPE model, as transformers 4's generate_merges derives them: every split of every piece
    // (at a character, in piece order) whose two halves are pieces, sorted by the piece's score, then the left half's
    // length, then the right half's (in characters), all descending, ties kept in that order.
    internal static (string Left, string Right)[] Merges(SentencePieceModel model, Dictionary<string, int> vocabulary)
    {
        var lookup = vocabulary.GetAlternateLookup<ReadOnlySpan<char>>();
        var found = new List<(string Left, string Right, float Score, int LeftLength, int RightLength)>();
        for (int id = 0; id < model.Count; id++)
        {
            string piece = model.Pieces[id];
            int characters = CodePoints(piece);
            int left = 0;
            for (int at = 1; at < piece.Length; at++)
            {
                if (char.IsLowSurrogate(piece[at]))
                {
                    continue;                                  // inside a character
                }

                left++;
                if (lookup.TryGetValue(piece.AsSpan(0, at), out string? l, out _) && lookup.TryGetValue(piece.AsSpan(at), out string? r, out _))
                {
                    found.Add((l, r, model.Scores[id], left, characters - left));   // the vocabulary's strings: none is made
                }
            }
        }

        var order = new int[found.Count];
        for (int i = 0; i < order.Length; i++)
        {
            order[i] = i;
        }

        var items = found.ToArray();
        Array.Sort(order, (a, b) =>
        {
            var (x, y) = (items[a], items[b]);
            int c = y.Score.CompareTo(x.Score);
            c = c != 0 ? c : y.LeftLength.CompareTo(x.LeftLength);
            c = c != 0 ? c : y.RightLength.CompareTo(x.RightLength);
            return c != 0 ? c : a.CompareTo(b);
        });

        var merges = new (string, string)[order.Length];
        for (int i = 0; i < merges.Length; i++)
        {
            merges[i] = (items[order[i]].Left, items[order[i]].Right);
        }

        return merges;

        static int CodePoints(string s)
        {
            int count = s.Length;
            foreach (char c in s)
            {
                count -= char.IsLowSurrogate(c) ? 1 : 0;
            }

            return count;
        }
    }
}
