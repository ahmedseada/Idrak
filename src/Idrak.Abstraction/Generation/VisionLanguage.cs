// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text;
using Idrak.Abstraction.Data;

namespace Idrak.Abstraction.Generation;

/// <summary>
/// How many tokens one image becomes in a vision-language model's prompt, and how they are arranged: a family with a
/// fixed size gives every image the same count (Gemma 3: 256 tokens, a 16 x 16 grid), a family with dynamic resolution
/// a count per image (from its size), with the grid its tokens cover when the family has one (rows and columns, or
/// frames, rows and columns), outermost first.
/// </summary>
/// <param name="Tokens">The image's tokens in the prompt (rows of its <see cref="ImageFeatures.Features"/>).</param>
public sealed record ImageTokenLayout(int Tokens)
{
    private readonly int[] _grid = [];

    /// <summary>The image's tokens.</summary>
    public int Tokens { get; } = Tokens > 0 ? Tokens : throw new ArgumentOutOfRangeException(nameof(Tokens), Tokens, "An image takes at least one token.");

    /// <summary>
    /// The grid the tokens cover, outermost first ([rows, columns], or [frames, rows, columns]), tokens in row-major order;
    /// empty when the family has none. Its product is <see cref="Tokens"/>.
    /// </summary>
    public IReadOnlyList<int> Grid
    {
        get => _grid;
        init
        {
            ArgumentNullException.ThrowIfNull(value);
            if (value.Count > 0 && (value.Any(v => v <= 0) || value.Aggregate(1L, (a, v) => a * v) != Tokens))
            {
                throw new ArgumentException($"A grid of [{string.Join(", ", value)}] does not hold {Tokens} tokens.", nameof(value));
            }

            _grid = [.. value];
        }
    }

    /// <inheritdoc />
    public bool Equals(ImageTokenLayout? other) => other is not null && Tokens == other.Tokens && _grid.AsSpan().SequenceEqual(other._grid);

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Tokens, _grid.Length > 0 ? _grid[0] : 0, _grid.Length > 1 ? _grid[^1] : 0);

    /// <inheritdoc />
    public override string ToString() => _grid.Length == 0 ? $"{Tokens} tokens" : $"{Tokens} tokens ({string.Join(" x ", _grid)})";
}

/// <summary>
/// One image as a vision encoder gives it (<see cref="IVisionEncoder.Encode"/>): the embeddings of its tokens in the
/// language model's prompt, its token layout, and, for a family that places image tokens by several axes (M-RoPE:
/// time, height, width), their position ids. Owns its tensors: dispose it when done.
/// </summary>
public sealed class ImageFeatures : IDisposable
{
    /// <summary>Wraps an image's features (owned from now on).</summary>
    /// <param name="features">[tokens, width]: one embedding per image token, the language model's width, on its device.</param>
    /// <param name="layout">The image's token layout; its <see cref="ImageTokenLayout.Tokens"/> is the features' rows.</param>
    /// <param name="positions">
    /// Null (the usual case): the image's tokens take the next positions of the prompt, one each, as text does. Otherwise
    /// [axes, tokens]: each token's position on each axis of a multi-axis rotary embedding (M-RoPE), whole numbers as
    /// floats, counted from the image's start (0 for its first row and column); the image position rule
    /// (<see cref="ImagePositionRules"/>) places them in the prompt. Read on the host once per prompt.
    /// </param>
    /// <exception cref="ArgumentException">The shapes do not agree with each other or with the layout.</exception>
    public ImageFeatures(Tensor features, ImageTokenLayout layout, Tensor? positions = null)
    {
        ArgumentNullException.ThrowIfNull(features);
        ArgumentNullException.ThrowIfNull(layout);
        if (features.Rank != 2 || features.Shape[0] != layout.Tokens)
        {
            throw new ArgumentException($"An image's features are [tokens, width] with {layout.Tokens} tokens ({layout}), got {Tensor.FormatShape(features.Shape)}.", nameof(features));
        }

        if (positions is not null && (positions.Rank != 2 || positions.Shape[1] != layout.Tokens))
        {
            throw new ArgumentException($"An image's position ids are [axes, {layout.Tokens}], got {Tensor.FormatShape(positions.Shape)}.", nameof(positions));
        }

        Features = features;
        Layout = layout;
        Positions = positions;
    }

    /// <summary>[tokens, width]: the image tokens' embeddings.</summary>
    public Tensor Features { get; }

    /// <summary>The image's token layout.</summary>
    public ImageTokenLayout Layout { get; }

    /// <summary>[axes, tokens] position ids for a multi-axis rotary embedding, from the image's start, or null: the prompt's next positions.</summary>
    public Tensor? Positions { get; }

    /// <summary>The image's tokens.</summary>
    public int Tokens => Layout.Tokens;

    /// <inheritdoc />
    public void Dispose()
    {
        Features.Dispose();
        Positions?.Dispose();
    }
}

/// <summary>
/// The image side of a vision-language model: decoded images in, each image's token features out, with as many tokens
/// per image as its family gives it (<see cref="Blocks"/>). It does its family's own preprocessing (resize, crop,
/// normalize, tiling or dynamic resolution) on the decoded pixels it is given. One image may become several blocks of
/// image tokens in the prompt (a family that adds views of an image, such as an overview and parts of it, or tiles):
/// each block is a run of image tokens of its own, with its own <see cref="ImageFeatures"/>, and the family's prompt
/// format writes the text around and between them (<see cref="IImagePromptFormat.Expand"/>). Most families give one
/// block per image. A family's own options (<see cref="VisionOptions"/>, given per request over the encoder's own) reach
/// both methods; an encoder refuses keys it does not take, naming those it does. A family makes one from its
/// checkpoint; the testing kit checks one (<c>Conformance.CheckVisionEncoder</c>: deterministic, counts matching the
/// blocks, a batch equal to single images, every device equal to the CPU).
/// </summary>
public interface IVisionEncoder : IDisposable
{
    /// <summary>The features' width: the language model's (each image token's embedding).</summary>
    int Width { get; }

    /// <summary>Where the features are made.</summary>
    Device Device { get; }

    /// <summary>
    /// The blocks of image tokens <paramref name="image"/> becomes under <paramref name="options"/> (over the encoder's
    /// own), in prompt order, each with its token count and arrangement, without encoding it: one block for most
    /// families.
    /// </summary>
    /// <exception cref="ArgumentException">An option is not one the family takes, or its value is not valid.</exception>
    IReadOnlyList<ImageTokenLayout> Blocks(ImageData image, VisionOptions? options = null);

    /// <summary>
    /// The features of <paramref name="images"/> under <paramref name="options"/>: one <see cref="ImageFeatures"/> per
    /// block (the caller disposes them), the images in order and each image's blocks in the order <see cref="Blocks"/>
    /// gives them, each with that block's layout. Runs without recording gradients. An image encodes the same alone and
    /// in a batch.
    /// </summary>
    /// <exception cref="ArgumentException">An option is not one the family takes, or its value is not valid.</exception>
    IReadOnlyList<ImageFeatures> Encode(IReadOnlyList<ImageData> images, VisionOptions? options = null);

    /// <summary>
    /// The position ids of a block's tokens on each axis of a multi-axis rotary embedding, [axes · tokens] (axis a of
    /// token j at a · tokens + j), counted from the block's start: what <see cref="Encode"/> gives as
    /// <see cref="ImageFeatures.Positions"/>, from the block's layout alone, so a caller that makes the features in stages
    /// (fine-tuning) places the tokens without encoding. Null (the default) for a family whose image tokens take the
    /// prompt's next positions, as text does.
    /// </summary>
    float[]? AxisPositions(ImageTokenLayout layout) => null;
}

/// <summary>
/// A vision encoder that also shows its stages, for comparing them one by one with a reference (<c>idrak vlm check</c>
/// against transformers): the pixel values its preprocessing gives, the vision tower's output, and the features. A
/// family's encoder may implement it; nothing requires it.
/// </summary>
public interface IVisionEncoderStages
{
    /// <summary>
    /// The pixel values the encoder's preprocessing gives <paramref name="image"/> under <paramref name="options"/>:
    /// [blocks, channels, height, width], one per block of <see cref="IVisionEncoder.Blocks"/>, on the CPU.
    /// </summary>
    Tensor PixelValues(ImageData image, VisionOptions? options = null);

    /// <summary>The vision tower's output for pixel values [images, channels, height, width] (before any projection), without gradients.</summary>
    Tensor Tower(Tensor pixelValues);

    /// <summary>The features [images, tokens, width] of pixel values [images, channels, height, width], without gradients.</summary>
    Tensor Features(Tensor pixelValues);
}

/// <summary>
/// How a vision-language model's prompt holds its images: the chat template writes one marker per image part, and the
/// model reads more than the marker (the image's tokens, and whatever its processor puts around them). <see cref="Expand"/>
/// does that second step on the rendered text, before it is tokenized; the embeddings of the
/// <see cref="ImageToken"/> runs are then replaced by the images' features. Each family's registration gives its own
/// (with its token ids from its configuration); <see cref="ImageTokenFormat"/> is a building block that covers the
/// common shapes.
/// </summary>
public interface IImagePromptFormat
{
    /// <summary>A name for messages (the family's).</summary>
    string Name { get; }

    /// <summary>The token whose embeddings an image's features replace (one per image token).</summary>
    int ImageToken { get; }

    /// <summary>
    /// The rendered prompt <paramref name="prompt"/> with each image's marker expanded to the text the model reads: each
    /// of the image's blocks (<see cref="IVisionEncoder.Blocks"/>) as its <see cref="ImageTokenLayout.Tokens"/> image
    /// tokens (written as the tokenizer's text of <see cref="ImageToken"/>, so that tokenizing it gives their ids), in
    /// order, with the family's text around and between them; the images in order. Two blocks never touch (text or
    /// another token separates them), so each is a run of image tokens of its own.
    /// </summary>
    /// <param name="prompt">The rendered prompt.</param>
    /// <param name="images">Each image's blocks, the images in prompt order.</param>
    /// <param name="tokenizer">The model's tokenizer.</param>
    /// <exception cref="InvalidOperationException">The prompt does not hold one marker per image.</exception>
    /// <exception cref="NotSupportedException">An image has several blocks and the format writes one per image.</exception>
    string Expand(string prompt, IReadOnlyList<IReadOnlyList<ImageTokenLayout>> images, ITokenizer tokenizer);
}

/// <summary>
/// An <see cref="ImageTokenFormat"/> that writes an image of several blocks (views) as one text: a family whose processor
/// puts text between an image's views derives its format from <see cref="ImageTokenFormat"/> and implements this.
/// </summary>
public interface IImageBlocksJoin
{
    /// <summary>
    /// The text of an image given each block's text (as <see cref="ImageTokenFormat.Block"/> writes it) in order. It keeps
    /// every block's text whole and in order, with something between any two.
    /// </summary>
    string Join(IReadOnlyList<string> blocks);
}

/// <summary>
/// An image prompt format built from token ids: each marker token the chat template wrote (in order, one per image)
/// becomes its block (<see cref="Block"/>): <see cref="Before"/>, the begin token (if any), the block's count of image
/// tokens, the end token (if any) and <see cref="After"/>. Gemma 3's processor is the marker <c>&lt;start_of_image&gt;</c>,
/// begin and end <c>&lt;start_of_image&gt;</c> and <c>&lt;end_of_image&gt;</c>, "\n\n" around; LLaVA's is the marker
/// <c>&lt;image&gt;</c> replaced by the image tokens alone (the marker is the image token). An image of several blocks
/// is written from its blocks' texts by a format that implements <see cref="IImageBlocksJoin"/> (a family's own format
/// derives from this class); any other writes one block per image.
/// </summary>
/// <param name="name">A name for messages.</param>
/// <param name="marker">The token the chat template writes once per image.</param>
/// <param name="imageToken">The image token (repeated once per image token).</param>
/// <param name="begin">The token before the image tokens, or null for none.</param>
/// <param name="end">The token after the image tokens, or null for none.</param>
/// <param name="before">Text before the begin token.</param>
/// <param name="after">Text after the end token.</param>
public class ImageTokenFormat(string name, int marker, int imageToken, int? begin = null, int? end = null, string before = "", string after = "") : IImagePromptFormat
{

    /// <inheritdoc />
    public string Name { get; } = name ?? throw new ArgumentNullException(nameof(name));

    /// <summary>The token the chat template writes once per image.</summary>
    public int Marker { get; } = marker;

    /// <inheritdoc />
    public int ImageToken { get; } = imageToken;

    /// <summary>The token before the image tokens, or null.</summary>
    public int? Begin { get; } = begin;

    /// <summary>The token after the image tokens, or null.</summary>
    public int? End { get; } = end;

    /// <summary>Text before the begin token.</summary>
    public string Before { get; } = before ?? "";

    /// <summary>Text after the end token.</summary>
    public string After { get; } = after ?? "";

    /// <summary>One block's text: <see cref="Before"/>, the begin token, <paramref name="layout"/>'s image tokens, the end token, <see cref="After"/>.</summary>
    public string Block(ImageTokenLayout layout, ITokenizer tokenizer)
    {
        ArgumentNullException.ThrowIfNull(layout);
        ArgumentNullException.ThrowIfNull(tokenizer);
        string image = Token(tokenizer, ImageToken, "image");
        string begin = Begin is { } b ? Token(tokenizer, b, "begin-image") : "", end = End is { } e ? Token(tokenizer, e, "end-image") : "";
        var text = new StringBuilder(Before.Length + begin.Length + layout.Tokens * image.Length + end.Length + After.Length).Append(Before).Append(begin);
        return text.Insert(text.Length, image, layout.Tokens).Append(end).Append(After).ToString();
    }

    /// <inheritdoc />
    public string Expand(string prompt, IReadOnlyList<IReadOnlyList<ImageTokenLayout>> images, ITokenizer tokenizer)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentNullException.ThrowIfNull(images);
        ArgumentNullException.ThrowIfNull(tokenizer);
        string markerText = Token(tokenizer, Marker, "image marker");
        var at = new List<int>();
        for (int i = prompt.IndexOf(markerText, StringComparison.Ordinal); i >= 0; i = prompt.IndexOf(markerText, i + markerText.Length, StringComparison.Ordinal))
        {
            at.Add(i);
        }

        if (at.Count != images.Count)
        {
            throw new InvalidOperationException($"The rendered prompt holds {at.Count} image markers ({markerText}) for {images.Count} images: the chat template does not write one per image part.");
        }

        if (images.Count == 0)
        {
            return prompt;
        }

        var text = new StringBuilder(prompt.Length);
        var joiner = this as IImageBlocksJoin;                                // tested once per prompt
        int from = 0;
        for (int k = 0; k < at.Count; k++)
        {
            var blocks = images[k];
            string written = blocks?.Count switch
            {
                null or 0 => throw new ArgumentException($"Image {k} has no blocks of image tokens.", nameof(images)),
                1 => Block(blocks[0], tokenizer),
                _ when joiner is not null => joiner.Join([.. blocks.Select(block => Block(block, tokenizer))]),
                _ => throw new NotSupportedException($"Image {k} has {blocks.Count} blocks of image tokens; the prompt format {Name} writes one per image (it is no IImageBlocksJoin)."),
            };
            text.Append(prompt, from, at[k] - from).Append(written);
            from = at[k] + markerText.Length;
        }

        return text.Append(prompt, from, prompt.Length - from).ToString();
    }

    /// <inheritdoc />
    public override string ToString() => $"ImageTokenFormat({Name})";

    private static string Token(ITokenizer tokenizer, int id, string what) =>
        tokenizer.TokenOf(id) is { Length: > 0 } text ? text
            : throw new InvalidOperationException($"The tokenizer has no {what} token {id} (vocabulary {tokenizer.VocabularySize}).");
}

/// <summary>
/// How a vision-language family's image tokens attend in the prompt: which keys each row sees in a sequence holding image
/// blocks. Text rows are causal in every rule the library ships; what differs is the image rows (Gemma 3: an image's tokens
/// see each other in both directions; LLaVA: plainly causal). Each family picks its rule (by name, from
/// <see cref="ImageAttentionRules"/>), so the decoder never assumes one.
/// </summary>
public interface IImageAttentionRule
{
    /// <summary>The rule's name in <see cref="ImageAttentionRules"/>.</summary>
    string Name { get; }

    /// <summary>
    /// True when image rows attend exactly as text rows (plain causal attention, windowed on sliding-window layers): the
    /// decoder then only substitutes the images' features and keeps its causal kernels; <see cref="Spans"/> is not asked.
    /// </summary>
    bool Causal { get; }

    /// <summary>
    /// Each row's range of keys for one sequence of <paramref name="rows"/> positions (keys are the same positions) holding
    /// <paramref name="blocks"/> (each image's first position and token count, in order, not overlapping), on a layer that
    /// sees <paramref name="window"/> keys back (its own included; 0: every earlier key).
    /// </summary>
    KeySpans Spans(int rows, IReadOnlyList<(int Start, int Length)> blocks, int window);
}

/// <summary>
/// The attention rules for image tokens, by name. The library has two: <see cref="Causal"/> (image rows attend as the
/// decoder's text rows do, the decoder's own behaviour) and <see cref="ImageBlocks"/> (each image's rows see their whole
/// image, through <see cref="KeySpans.ImageBlocks"/>, causal elsewhere; within the window on sliding-window layers). A
/// family names the rule its image tokens attend by in its vision part; a family whose tokens attend otherwise registers
/// its own rule here, and an app can take any name's place. Nothing falls back to another rule: an unknown name is an
/// error naming this registry.
/// </summary>
public static class ImageAttentionRules
{
    /// <summary>Image tokens attend as text does: causal (and windowed on sliding-window layers). LLaVA's rule.</summary>
    public const string Causal = "causal";

    /// <summary>
    /// Each image's tokens see their whole image (both directions), the rest causal, and on sliding-window layers only the
    /// keys within the window (<see cref="KeySpans.ImageBlocks"/>): a key is seen when it is not after the row or shares the
    /// row's image, and lies within the window. Gemma 3's mask (transformers' <c>token_type_ids</c>) and IdrakVision's.
    /// </summary>
    public const string ImageBlocks = "image-blocks";

    private static readonly SlotTable<string, IImageAttentionRule> Table = BuiltIn();

    private static SlotTable<string, IImageAttentionRule> BuiltIn()
    {
        var table = new SlotTable<string, IImageAttentionRule>(nameof(ImageAttentionRules), Guard, StringComparer.Ordinal);
        table.RegisterDefault(Causal, new CausalRule());
        table.RegisterDefault(ImageBlocks, new ImageBlocksRule());
        return table;
    }

    // Each Spans is one call: an app's rule falls back (or is shadowed) per call. Causal is read when the rule is chosen.
    private static IImageAttentionRule Guard(Slot slot, IImageAttentionRule app, IImageAttentionRule library) => new GuardedRule(slot, app, library);

    /// <summary>Registers <paramref name="rule"/> under its <see cref="IImageAttentionRule.Name"/>; under a library name it takes that rule's place.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IImageAttentionRule rule)
    {
        ArgumentNullException.ThrowIfNull(rule);
        ArgumentException.ThrowIfNullOrEmpty(rule.Name);
        Table.Register(rule.Name, rule, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's rule <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The rule <paramref name="name"/>, or null when none is registered.</summary>
    public static IImageAttentionRule? Find(string name) => Table.Find(name);

    /// <summary>The rule <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name; the message names <see cref="Register"/>.</exception>
    public static IImageAttentionRule Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(ImageAttentionRules), name, Table.Keys,
            $"No image attention rule '{name}' is registered ({string.Join(", ", Table.Keys)}); add it with ImageAttentionRules.Register.");

    /// <summary>The library's rule <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IImageAttentionRule? Default(string name) => Table.Default(name);

    /// <summary>Who registered the rule <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>What happens when the app's rule <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set).</summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    private sealed class CausalRule : IImageAttentionRule
    {
        public string Name => ImageAttentionRules.Causal;

        public bool Causal => true;

        public KeySpans Spans(int rows, IReadOnlyList<(int Start, int Length)> blocks, int window) => KeySpans.Causal(rows, window);

        public override string ToString() => "causal image tokens";
    }

    private sealed class ImageBlocksRule : IImageAttentionRule
    {
        public string Name => ImageAttentionRules.ImageBlocks;

        public bool Causal => false;

        public KeySpans Spans(int rows, IReadOnlyList<(int Start, int Length)> blocks, int window) => KeySpans.ImageBlocks(rows, blocks, window);

        public override string ToString() => "image blocks seen whole";
    }

    private sealed class GuardedRule(Slot slot, IImageAttentionRule app, IImageAttentionRule library) : IImageAttentionRule
    {
        public string Name => app.Name;

        public bool Causal => app.Causal;

        public KeySpans Spans(int rows, IReadOnlyList<(int Start, int Length)> blocks, int window) =>
            slot.Call(() => app.Spans(rows, blocks, window), () => library.Spans(rows, blocks, window), (a, b) => a.Starts.SequenceEqual(b.Starts) && a.Ends.SequenceEqual(b.Ends) ? null : "different key ranges");
    }
}

/// <summary>
/// One image of a sequence for an <see cref="IImagePositionRule"/>: where its tokens are in the sequence and each token's
/// position on each axis, counted from the image's start (<see cref="ImageFeatures.Positions"/>).
/// </summary>
/// <param name="Offset">The image's first token, counted from the start of the tokens being placed.</param>
/// <param name="Length">The image's tokens.</param>
/// <param name="Relative">[axes · length]: axis a of token j at <c>a · Length + j</c>, whole numbers from 0.</param>
public readonly record struct ImageRun(int Offset, int Length, ReadOnlyMemory<float> Relative);

/// <summary>
/// How the tokens of a sequence holding images are placed on the axes of a multi-axis rotary embedding (M-RoPE,
/// <see cref="RopeSettings.Sections"/>): text and image tokens alike get one position per axis, from the sequence's next
/// rotary position on. The decoder rotates each frequency pair by its axis' position, and decodes the tokens after the
/// sequence from the rotary position the rule returns (the difference to the sequence's length is kept by
/// <see cref="DecodingContext.RotaryOffsets"/>).
/// </summary>
public interface IImagePositionRule
{
    /// <summary>The rule's name in <see cref="ImagePositionRules"/>.</summary>
    string Name { get; }

    /// <summary>
    /// Places <paramref name="count"/> tokens starting at rotary position <paramref name="start"/>: writes
    /// <paramref name="positions"/>[a · count + i] for every axis a and token i, the images at their offsets in order (the
    /// other tokens are text), and returns the rotary position after the last token.
    /// </summary>
    int Place(int start, int count, IReadOnlyList<ImageRun> images, int axes, Span<float> positions);
}

/// <summary>
/// The image position rules, by name. The library has one, <see cref="Grid"/>, the multi-axis placement of Qwen2-VL and
/// Qwen2.5-VL's <c>get_rope_index</c>, which a prompt whose image features carry axis positions uses; an app may take its
/// place (or register another rule for its own use). An unknown name is an error naming this registry.
/// </summary>
public static class ImagePositionRules
{
    /// <summary>
    /// Text tokens take the next position on every axis; an image's tokens take the position after the text before it
    /// plus their own axis positions; the text after the image continues from one past the largest position the image
    /// took. The rotary position after a sequence is one past its largest position.
    /// </summary>
    public const string Grid = "grid";

    private static readonly SlotTable<string, IImagePositionRule> Table = BuiltIn();

    private static SlotTable<string, IImagePositionRule> BuiltIn()
    {
        var table = new SlotTable<string, IImagePositionRule>(nameof(ImagePositionRules), Guard, StringComparer.Ordinal);
        table.RegisterDefault(Grid, new GridRule());
        return table;
    }

    private static IImagePositionRule Guard(Slot slot, IImagePositionRule app, IImagePositionRule library) => new GuardedRule(slot, app, library);

    /// <summary>Registers <paramref name="rule"/> under its <see cref="IImagePositionRule.Name"/>; under a library name it takes that rule's place.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IImagePositionRule rule)
    {
        ArgumentNullException.ThrowIfNull(rule);
        ArgumentException.ThrowIfNullOrEmpty(rule.Name);
        Table.Register(rule.Name, rule, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's rule <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The rule <paramref name="name"/>, or null when none is registered.</summary>
    public static IImagePositionRule? Find(string name) => Table.Find(name);

    /// <summary>The rule <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name; the message names <see cref="Register"/>.</exception>
    public static IImagePositionRule Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(ImagePositionRules), name, Table.Keys,
            $"No image position rule '{name}' is registered ({string.Join(", ", Table.Keys)}); add it with ImagePositionRules.Register.");

    /// <summary>The library's rule <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IImagePositionRule? Default(string name) => Table.Default(name);

    /// <summary>Who registered the rule <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>What happens when the app's rule <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set).</summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    private sealed class GridRule : IImagePositionRule
    {
        public string Name => Grid;

        public int Place(int start, int count, IReadOnlyList<ImageRun> images, int axes, Span<float> positions)
        {
            if (positions.Length < axes * count)
            {
                throw new ArgumentException($"The positions hold {positions.Length} values; {axes} axes x {count} tokens need {axes * count}.", nameof(positions));
            }

            int next = start, i = 0;
            foreach (var image in images)
            {
                if (image.Offset < i || image.Offset + image.Length > count || image.Relative.Length != axes * image.Length)
                {
                    throw new ArgumentException($"An image of {image.Length} tokens at {image.Offset} (with {image.Relative.Length} axis positions) does not fit "
                        + $"{count} tokens after the previous image, {axes} axes each.", nameof(images));
                }

                for (; i < image.Offset; i++, next++)
                {
                    for (int a = 0; a < axes; a++)
                    {
                        positions[a * count + i] = next;
                    }
                }

                // The image's tokens from here on, each axis by its own position; the text after it starts one past the
                // largest position any of its tokens took.
                var relative = image.Relative.Span;
                float largest = 0;
                for (int a = 0; a < axes; a++)
                {
                    for (int j = 0; j < image.Length; j++)
                    {
                        float p = relative[a * image.Length + j];
                        positions[a * count + i + j] = next + p;
                        largest = Math.Max(largest, p);
                    }
                }

                i += image.Length;
                next += (int)largest + 1;
            }

            for (; i < count; i++, next++)
            {
                for (int a = 0; a < axes; a++)
                {
                    positions[a * count + i] = next;
                }
            }

            return next;
        }

        public override string ToString() => "grid positions (M-RoPE)";
    }

    private sealed class GuardedRule(Slot slot, IImagePositionRule app, IImagePositionRule library) : IImagePositionRule
    {
        public string Name => app.Name;

        public int Place(int start, int count, IReadOnlyList<ImageRun> images, int axes, Span<float> positions)
        {
            // Each rule writes its own copy (the slot may run both, and compare them); the answer's is kept.
            (int Next, float[] Values) Run(IImagePositionRule rule)
            {
                var values = new float[axes * count];
                return (rule.Place(start, count, images, axes, values), values);
            }

            var answer = slot.Call(() => Run(app), () => Run(library),
                (a, b) => a.Next == b.Next && a.Values.AsSpan().SequenceEqual(b.Values) ? null : "different positions");
            answer.Values.CopyTo(positions);
            return answer.Next;
        }
    }
}
