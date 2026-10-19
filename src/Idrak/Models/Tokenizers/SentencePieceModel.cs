// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Models;

/// <summary>What a SentencePiece piece is (<c>ModelProto.SentencePiece.Type</c>; the numbers are the file's).</summary>
internal enum SentencePieceType : byte
{
    /// <summary>An ordinary piece, matched in text.</summary>
    Normal = 1,

    /// <summary>The unknown piece (one per model).</summary>
    Unknown = 2,

    /// <summary>A control symbol (&lt;s&gt;, &lt;/s&gt;, …): never matched in text.</summary>
    Control = 3,

    /// <summary>A user-defined symbol: always one piece when it appears in text.</summary>
    UserDefined = 4,

    /// <summary>A piece the model never produces.</summary>
    Unused = 5,

    /// <summary>A byte-fallback piece, &lt;0x00&gt; to &lt;0xFF&gt;.</summary>
    Byte = 6,
}

/// <summary>The algorithm a SentencePiece model was trained with (<c>TrainerSpec.ModelType</c>).</summary>
internal enum SentencePieceAlgorithm
{
    /// <summary>The unigram language model (T5, ALBERT, XLM-R).</summary>
    Unigram = 1,

    /// <summary>Byte-pair encoding (Llama 1 and 2).</summary>
    Bpe = 2,

    /// <summary>Whole words.</summary>
    Word = 3,

    /// <summary>Single characters.</summary>
    Char = 4,
}

/// <summary>
/// The training settings a SentencePiece model keeps (<c>TrainerSpec</c>). Most describe how it was trained; encoding
/// reads the special pieces and ids, <see cref="ByteFallback"/>, <see cref="SplitDigits"/>,
/// <see cref="TreatWhitespaceAsSuffix"/> and the algorithm. A field absent from the file has the default the proto declares.
/// </summary>
internal sealed class SentencePieceTrainerSpec
{
    /// <summary>Field 1: the training input files.</summary>
    public string[] Input { get; private set; } = [];

    /// <summary>Field 7: the input format ("text" or "tsv").</summary>
    public string InputFormat { get; private set; } = "";

    /// <summary>Field 2: the output name the trainer was given.</summary>
    public string ModelPrefix { get; private set; } = "";

    /// <summary>Field 3: the algorithm.</summary>
    public SentencePieceAlgorithm ModelType { get; private set; } = SentencePieceAlgorithm.Unigram;

    /// <summary>Field 4: the vocabulary size asked for.</summary>
    public int VocabSize { get; private set; } = 8000;

    /// <summary>Field 5: the languages the trainer was told to accept.</summary>
    public string[] AcceptLanguage { get; private set; } = [];

    /// <summary>Field 6: the self-test sample size.</summary>
    public int SelfTestSampleSize { get; private set; }

    /// <summary>Field 50: whether differential privacy was on in training.</summary>
    public bool EnableDifferentialPrivacy { get; private set; }

    /// <summary>Field 51: the differential-privacy noise level.</summary>
    public float DifferentialPrivacyNoiseLevel { get; private set; }

    /// <summary>Field 52: the differential-privacy clipping threshold.</summary>
    public ulong DifferentialPrivacyClippingThreshold { get; private set; }

    /// <summary>Field 10: the share of characters the alphabet covers.</summary>
    public float CharacterCoverage { get; private set; } = 0.9995f;

    /// <summary>Field 11: the sentences sampled for training (0: all).</summary>
    public ulong InputSentenceSize { get; private set; }

    /// <summary>Field 19: whether the input was shuffled.</summary>
    public bool ShuffleInputSentence { get; private set; } = true;

    /// <summary>Field 12 (deprecated).</summary>
    public int MiningSentenceSize { get; private set; }

    /// <summary>Field 13 (deprecated).</summary>
    public int TrainingSentenceSize { get; private set; }

    /// <summary>Field 14: the seed vocabulary's size (unigram).</summary>
    public int SeedSentencepieceSize { get; private set; } = 1_000_000;

    /// <summary>Field 15: the share kept per pruning round (unigram).</summary>
    public float ShrinkingFactor { get; private set; } = 0.75f;

    /// <summary>Field 18: the longest sentence trained on, in bytes.</summary>
    public int MaxSentenceLength { get; private set; } = 4192;

    /// <summary>Field 16: the trainer's threads.</summary>
    public int NumThreads { get; private set; } = 16;

    /// <summary>Field 17: EM steps per round (unigram).</summary>
    public int NumSubIterations { get; private set; } = 2;

    /// <summary>Field 20: the longest piece, in characters.</summary>
    public int MaxSentencepieceLength { get; private set; } = 16;

    /// <summary>Field 21: pieces do not cross a Unicode script change.</summary>
    public bool SplitByUnicodeScript { get; private set; } = true;

    /// <summary>Field 23: pieces do not cross a number boundary.</summary>
    public bool SplitByNumber { get; private set; } = true;

    /// <summary>Field 22: pieces do not cross whitespace.</summary>
    public bool SplitByWhitespace { get; private set; } = true;

    /// <summary>Field 24: the whitespace mark ends a piece instead of starting it.</summary>
    public bool TreatWhitespaceAsSuffix { get; private set; }

    /// <summary>Field 26: pieces made only of whitespace are allowed.</summary>
    public bool AllowWhitespaceOnlyPieces { get; private set; }

    /// <summary>Field 25: every digit is its own piece.</summary>
    public bool SplitDigits { get; private set; }

    /// <summary>Field 53: a delimiter that pre-splits the training text.</summary>
    public string PretokenizationDelimiter { get; private set; } = "";

    /// <summary>Field 30: the control symbols.</summary>
    public string[] ControlSymbols { get; private set; } = [];

    /// <summary>Field 31: the user-defined symbols.</summary>
    public string[] UserDefinedSymbols { get; private set; } = [];

    /// <summary>Field 36: characters the alphabet must keep.</summary>
    public string RequiredChars { get; private set; } = "";

    /// <summary>Field 35: unknown characters are spelled with &lt;0xNN&gt; byte pieces.</summary>
    public bool ByteFallback { get; private set; }

    /// <summary>Field 32: scores were written to the vocabulary file.</summary>
    public bool VocabularyOutputPieceScore { get; private set; } = true;

    /// <summary>Field 33: the vocabulary size was a hard limit.</summary>
    public bool HardVocabLimit { get; private set; } = true;

    /// <summary>Field 34: every piece was kept (word and char models).</summary>
    public bool UseAllVocab { get; private set; }

    /// <summary>Field 40: the unknown piece's id.</summary>
    public int UnkId { get; private set; }

    /// <summary>Field 41: the beginning-of-sentence id (-1: none).</summary>
    public int BosId { get; private set; } = 1;

    /// <summary>Field 42: the end-of-sentence id (-1: none).</summary>
    public int EosId { get; private set; } = 2;

    /// <summary>Field 43: the padding id (-1: none).</summary>
    public int PadId { get; private set; } = -1;

    /// <summary>Field 45: the unknown piece.</summary>
    public string UnkPiece { get; private set; } = "<unk>";

    /// <summary>Field 46: the beginning-of-sentence piece.</summary>
    public string BosPiece { get; private set; } = "<s>";

    /// <summary>Field 47: the end-of-sentence piece.</summary>
    public string EosPiece { get; private set; } = "</s>";

    /// <summary>Field 48: the padding piece.</summary>
    public string PadPiece { get; private set; } = "<pad>";

    /// <summary>Field 44: the text an unknown piece decodes to.</summary>
    public string UnkSurface { get; private set; } = " ⁇ ";

    /// <summary>Field 49: the trainer's large-corpus mode.</summary>
    public bool TrainExtremelyLargeCorpus { get; private set; }

    /// <summary>Reads the message's fields into this spec (a field met twice keeps its last value, a repeated one adds).</summary>
    internal void Read(ProtoReader reader)
    {
        List<string>? input = null, languages = null, control = null, user = null;
        while (reader.Next(out int field, out var wire))
        {
            switch ((field, wire))
            {
                case (1, ProtoWireType.LengthDelimited): (input ??= [.. Input]).Add(reader.String()); break;
                case (7, ProtoWireType.LengthDelimited): InputFormat = reader.String(); break;
                case (2, ProtoWireType.LengthDelimited): ModelPrefix = reader.String(); break;
                case (3, ProtoWireType.Varint): ModelType = Algorithm(reader.Int32(), ModelType); break;
                case (4, ProtoWireType.Varint): VocabSize = reader.Int32(); break;
                case (5, ProtoWireType.LengthDelimited): (languages ??= [.. AcceptLanguage]).Add(reader.String()); break;
                case (6, ProtoWireType.Varint): SelfTestSampleSize = reader.Int32(); break;
                case (50, ProtoWireType.Varint): EnableDifferentialPrivacy = reader.Bool(); break;
                case (51, ProtoWireType.Fixed32): DifferentialPrivacyNoiseLevel = reader.Float(); break;
                case (52, ProtoWireType.Varint): DifferentialPrivacyClippingThreshold = reader.Varint(); break;
                case (10, ProtoWireType.Fixed32): CharacterCoverage = reader.Float(); break;
                case (11, ProtoWireType.Varint): InputSentenceSize = reader.Varint(); break;
                case (19, ProtoWireType.Varint): ShuffleInputSentence = reader.Bool(); break;
                case (12, ProtoWireType.Varint): MiningSentenceSize = reader.Int32(); break;
                case (13, ProtoWireType.Varint): TrainingSentenceSize = reader.Int32(); break;
                case (14, ProtoWireType.Varint): SeedSentencepieceSize = reader.Int32(); break;
                case (15, ProtoWireType.Fixed32): ShrinkingFactor = reader.Float(); break;
                case (18, ProtoWireType.Varint): MaxSentenceLength = reader.Int32(); break;
                case (16, ProtoWireType.Varint): NumThreads = reader.Int32(); break;
                case (17, ProtoWireType.Varint): NumSubIterations = reader.Int32(); break;
                case (20, ProtoWireType.Varint): MaxSentencepieceLength = reader.Int32(); break;
                case (21, ProtoWireType.Varint): SplitByUnicodeScript = reader.Bool(); break;
                case (23, ProtoWireType.Varint): SplitByNumber = reader.Bool(); break;
                case (22, ProtoWireType.Varint): SplitByWhitespace = reader.Bool(); break;
                case (24, ProtoWireType.Varint): TreatWhitespaceAsSuffix = reader.Bool(); break;
                case (26, ProtoWireType.Varint): AllowWhitespaceOnlyPieces = reader.Bool(); break;
                case (25, ProtoWireType.Varint): SplitDigits = reader.Bool(); break;
                case (53, ProtoWireType.LengthDelimited): PretokenizationDelimiter = reader.String(); break;
                case (30, ProtoWireType.LengthDelimited): (control ??= [.. ControlSymbols]).Add(reader.String()); break;
                case (31, ProtoWireType.LengthDelimited): (user ??= [.. UserDefinedSymbols]).Add(reader.String()); break;
                case (36, ProtoWireType.LengthDelimited): RequiredChars = reader.String(); break;
                case (35, ProtoWireType.Varint): ByteFallback = reader.Bool(); break;
                case (32, ProtoWireType.Varint): VocabularyOutputPieceScore = reader.Bool(); break;
                case (33, ProtoWireType.Varint): HardVocabLimit = reader.Bool(); break;
                case (34, ProtoWireType.Varint): UseAllVocab = reader.Bool(); break;
                case (40, ProtoWireType.Varint): UnkId = reader.Int32(); break;
                case (41, ProtoWireType.Varint): BosId = reader.Int32(); break;
                case (42, ProtoWireType.Varint): EosId = reader.Int32(); break;
                case (43, ProtoWireType.Varint): PadId = reader.Int32(); break;
                case (45, ProtoWireType.LengthDelimited): UnkPiece = reader.String(); break;
                case (46, ProtoWireType.LengthDelimited): BosPiece = reader.String(); break;
                case (47, ProtoWireType.LengthDelimited): EosPiece = reader.String(); break;
                case (48, ProtoWireType.LengthDelimited): PadPiece = reader.String(); break;
                case (44, ProtoWireType.LengthDelimited): UnkSurface = reader.String(); break;
                case (49, ProtoWireType.Varint): TrainExtremelyLargeCorpus = reader.Bool(); break;
                default: reader.Skip(field, wire); break;
            }
        }

        Input = input is null ? Input : [.. input];
        AcceptLanguage = languages is null ? AcceptLanguage : [.. languages];
        ControlSymbols = control is null ? ControlSymbols : [.. control];
        UserDefinedSymbols = user is null ? UserDefinedSymbols : [.. user];
    }

    // proto2 enums are closed: a number the enum does not declare leaves the field as it was (protobuf keeps it as an
    // unknown field).
    private static SentencePieceAlgorithm Algorithm(int value, SentencePieceAlgorithm current) =>
        value is >= 1 and <= 4 ? (SentencePieceAlgorithm)value : current;
}

/// <summary>
/// How a SentencePiece model normalizes text (<c>NormalizerSpec</c>): the precompiled character map (a darts-clone trie
/// plus the strings it maps to, <see cref="DoubleArrayTrie.ReadCharsmap"/>), and the whitespace rules applied after it.
/// </summary>
internal sealed class SentencePieceNormalizerSpec
{
    /// <summary>Field 1: the rule's name (nmt_nfkc, nfkc, nmt_nfkc_cf, nfkc_cf, identity, or user_defined for a TSV rule).</summary>
    public string Name { get; private set; } = "";

    /// <summary>Field 2: the precompiled character map (empty for identity).</summary>
    public byte[] PrecompiledCharsmap { get; private set; } = [];

    /// <summary>Field 3: a whitespace mark is put before the text.</summary>
    public bool AddDummyPrefix { get; private set; } = true;

    /// <summary>Field 4: leading, trailing and repeated spaces are removed.</summary>
    public bool RemoveExtraWhitespaces { get; private set; } = true;

    /// <summary>Field 5: spaces become the whitespace mark U+2581.</summary>
    public bool EscapeWhitespaces { get; private set; } = true;

    /// <summary>Field 6: the TSV rule file the map was compiled from (its name as the trainer was given it).</summary>
    public string NormalizationRuleTsv { get; private set; } = "";

    /// <summary>Reads the message's fields into this spec (a field met twice keeps its last value).</summary>
    internal void Read(ProtoReader reader)
    {
        while (reader.Next(out int field, out var wire))
        {
            switch ((field, wire))
            {
                case (1, ProtoWireType.LengthDelimited): Name = reader.String(); break;
                case (2, ProtoWireType.LengthDelimited): PrecompiledCharsmap = reader.LengthDelimited().ToArray(); break;
                case (3, ProtoWireType.Varint): AddDummyPrefix = reader.Bool(); break;
                case (4, ProtoWireType.Varint): RemoveExtraWhitespaces = reader.Bool(); break;
                case (5, ProtoWireType.Varint): EscapeWhitespaces = reader.Bool(); break;
                case (6, ProtoWireType.LengthDelimited): NormalizationRuleTsv = reader.String(); break;
                default: reader.Skip(field, wire); break;
            }
        }
    }
}

/// <summary>
/// A SentencePiece model file (<c>tokenizer.model</c>, <c>spiece.model</c>, <c>sentencepiece.bpe.model</c>): its
/// <c>ModelProto</c> read from bytes, in C# (no protobuf package, no native library). The pieces are three flat arrays
/// by id, sized from a first pass over the file, so a 250,000-piece vocabulary is three arrays and its strings.
/// </summary>
/// <remarks>
/// Field numbers and defaults are sentencepiece_model.proto's as of sentencepiece 0.2.2 (ModelProto: pieces 1,
/// trainer_spec 2, normalizer_spec 3, self_test_data 4, denormalizer_spec 5; SentencePiece: piece 1, score 2 (float),
/// type 3). Unknown fields, and known fields with a wire type their declaration does not have, are skipped as protobuf
/// skips them; self-test data is skipped. Errors are <see cref="InvalidDataException"/> naming the file and the byte.
/// </remarks>
internal sealed class SentencePieceModel
{
    private SentencePieceModel(string source, int count)
    {
        Source = source;
        Pieces = new string[count];
        Scores = new float[count];
        Types = new SentencePieceType[count];
    }

    /// <summary>The file, for messages.</summary>
    public string Source { get; }

    /// <summary>The pieces by id.</summary>
    public string[] Pieces { get; }

    /// <summary>The pieces' scores by id (log probabilities for unigram, merge ranks as negative numbers for BPE).</summary>
    public float[] Scores { get; }

    /// <summary>The pieces' types by id.</summary>
    public SentencePieceType[] Types { get; }

    /// <summary>The training settings.</summary>
    public SentencePieceTrainerSpec TrainerSpec { get; } = new();

    /// <summary>The normalization.</summary>
    public SentencePieceNormalizerSpec NormalizerSpec { get; } = new();

    /// <summary>The denormalization applied when decoding, when the file has one.</summary>
    public SentencePieceNormalizerSpec? DenormalizerSpec { get; private set; }

    /// <summary>The number of pieces.</summary>
    public int Count => Pieces.Length;

    /// <summary>Reads a model file (one read of its bytes).</summary>
    public static SentencePieceModel Load(string path) => Read(File.ReadAllBytes(path), path);

    /// <summary>Reads a model from its bytes; <paramref name="source"/> names it in messages.</summary>
    public static SentencePieceModel Read(ReadOnlySpan<byte> data, string source)
    {
        try
        {
            var model = new SentencePieceModel(source, CountPieces(data));
            model.ReadFields(new ProtoReader(data));
            return model;
        }
        catch (InvalidDataException e)
        {
            throw new InvalidDataException($"SentencePiece model '{source}': {e.Message}", e);
        }
    }

    // The first pass: how many pieces (field 1) the file holds, skipping everything.
    private static int CountPieces(ReadOnlySpan<byte> data)
    {
        var reader = new ProtoReader(data);
        int count = 0;
        while (reader.Next(out int field, out var wire))
        {
            if (field == 1 && wire == ProtoWireType.LengthDelimited)
            {
                count++;
            }

            reader.Skip(field, wire);
        }

        return count;
    }

    private void ReadFields(ProtoReader reader)
    {
        int index = 0;
        while (reader.Next(out int field, out var wire))
        {
            switch ((field, wire))
            {
                case (1, ProtoWireType.LengthDelimited):
                    ReadPiece(reader.Message(), index++);
                    break;
                case (2, ProtoWireType.LengthDelimited):
                    TrainerSpec.Read(reader.Message());
                    break;
                case (3, ProtoWireType.LengthDelimited):
                    NormalizerSpec.Read(reader.Message());
                    break;
                case (5, ProtoWireType.LengthDelimited):
                    (DenormalizerSpec ??= new()).Read(reader.Message());
                    break;
                default:
                    reader.Skip(field, wire);
                    break;
            }
        }
    }

    private void ReadPiece(ProtoReader reader, int index)
    {
        string piece = "";
        float score = 0f;
        var type = SentencePieceType.Normal;
        while (reader.Next(out int field, out var wire))
        {
            switch ((field, wire))
            {
                case (1, ProtoWireType.LengthDelimited): piece = reader.String(); break;
                case (2, ProtoWireType.Fixed32): score = reader.Float(); break;
                case (3, ProtoWireType.Varint):
                    int value = reader.Int32();
                    type = value is >= 1 and <= 6 ? (SentencePieceType)value : type;    // a closed proto2 enum
                    break;
                default: reader.Skip(field, wire); break;
            }
        }

        Pieces[index] = piece;
        Scores[index] = score;
        Types[index] = type;
    }
}
