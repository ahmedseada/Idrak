// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using Idrak.Models;

namespace Idrak.Generation.Abstractions;

/// <summary>
/// How a vision-language model's prompt holds its images: the chat template writes one marker per image part (Gemma 3's
/// writes <c>&lt;start_of_image&gt;</c>), and the model reads more than the marker (Gemma 3's processor turns each into
/// <c>"\n\n&lt;start_of_image&gt;"</c>, <see cref="ImageTokenIds.TokensPerImage"/> image soft tokens and
/// <c>"&lt;end_of_image&gt;\n\n"</c>). <see cref="Expand"/> does that second step on the rendered text, before it is
/// tokenized; the image soft tokens' embeddings are then replaced by the image's features
/// (<see cref="Idrak.Layers.ImagePrefill"/>). Registered by model type in <see cref="ImagePromptFormats"/>.
/// </summary>
public interface IImagePromptFormat
{
    /// <summary>The model type it is registered under (config.json's <c>model_type</c>, "gemma3").</summary>
    string Name { get; }

    /// <summary>
    /// The rendered prompt <paramref name="prompt"/> with each of its <paramref name="images"/> images' markers expanded
    /// to the text the model reads, the image tokens included (written as the tokenizer's text of
    /// <paramref name="tokens"/>, so that tokenizing it gives their ids).
    /// </summary>
    /// <exception cref="InvalidOperationException">The prompt does not hold one marker per image.</exception>
    string Expand(string prompt, int images, ImageTokenIds tokens, ITokenizer tokenizer);
}

/// <summary>
/// Gemma 3's image prompt (and any model whose processor works the same way): each begin-image token the chat template
/// wrote becomes <see cref="Before"/>, the begin-image token, <see cref="ImageTokenIds.TokensPerImage"/> image tokens, the
/// end-image token and <see cref="After"/>, as transformers' <c>Gemma3Processor</c> replaces <c>boi_token</c> with
/// <c>full_image_sequence</c>.
/// </summary>
/// <param name="name">The model type it is registered under.</param>
/// <param name="before">Text before the begin-image token ("\n\n" for Gemma 3).</param>
/// <param name="after">Text after the end-image token ("\n\n" for Gemma 3).</param>
public sealed class ImageMarkerFormat(string name, string before = "\n\n", string after = "\n\n") : IImagePromptFormat
{
    /// <inheritdoc />
    public string Name { get; } = name;

    /// <summary>Text before the begin-image token.</summary>
    public string Before { get; } = before;

    /// <summary>Text after the end-image token.</summary>
    public string After { get; } = after;

    /// <inheritdoc />
    public string Expand(string prompt, int images, ImageTokenIds tokens, ITokenizer tokenizer)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentNullException.ThrowIfNull(tokens);
        ArgumentNullException.ThrowIfNull(tokenizer);
        string begin = Token(tokenizer, tokens.BeginImage, "begin-image"), image = Token(tokenizer, tokens.ImageToken, "image"),
            end = Token(tokenizer, tokens.EndImage, "end-image");
        int markers = 0;
        for (int at = prompt.IndexOf(begin, StringComparison.Ordinal); at >= 0; at = prompt.IndexOf(begin, at + begin.Length, StringComparison.Ordinal))
        {
            markers++;
        }

        if (markers != images)
        {
            throw new InvalidOperationException($"The rendered prompt holds {markers} image markers ({begin}) for {images} images: the chat template does not write one per image part.");
        }

        if (images == 0)
        {
            return prompt;
        }

        var block = new System.Text.StringBuilder(Before.Length + begin.Length + image.Length * tokens.TokensPerImage + end.Length + After.Length);
        block.Append(Before).Append(begin).Insert(block.Length, image, tokens.TokensPerImage).Append(end).Append(After);
        return prompt.Replace(begin, block.ToString(), StringComparison.Ordinal);
    }

    private static string Token(ITokenizer tokenizer, int id, string what) =>
        tokenizer.TokenOf(id) is { Length: > 0 } text ? text
            : throw new InvalidOperationException($"The tokenizer has no {what} token {id} (vocabulary {tokenizer.VocabularySize}).");
}

/// <summary>
/// The image prompt formats by model type (config.json's <c>model_type</c>). "gemma3" (<see cref="ImageMarkerFormat"/>
/// with "\n\n" around each image) is the library's. Register a format of your own for another family, or over a
/// library one (it stays behind yours, see <see cref="SetPolicy"/>).
/// </summary>
public static class ImagePromptFormats
{
    /// <summary>Gemma 3's model type.</summary>
    public const string Gemma3 = "gemma3";

    private static readonly SlotTable<string, IImagePromptFormat> Table = BuiltIn();

    private static SlotTable<string, IImagePromptFormat> BuiltIn()
    {
        var table = new SlotTable<string, IImagePromptFormat>(nameof(ImagePromptFormats), Guard, StringComparer.Ordinal);
        table.RegisterDefault(Gemma3, new ImageMarkerFormat(Gemma3));
        return table;
    }

    // Expanding a prompt is one call: an app's format falls back (or is shadowed) per call.
    private static IImagePromptFormat Guard(Slot slot, IImagePromptFormat app, IImagePromptFormat library) => new GuardedFormat(slot, app, library);

    /// <summary>
    /// Registers <paramref name="format"/> under its <see cref="IImagePromptFormat.Name"/>; under a library name it takes
    /// that format's place (the library's stays behind it).
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IImagePromptFormat format)
    {
        ArgumentNullException.ThrowIfNull(format);
        ArgumentException.ThrowIfNullOrEmpty(format.Name);
        Table.Register(format.Name, format, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's format <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered model types.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The format of model type <paramref name="name"/>, or null when none is registered.</summary>
    public static IImagePromptFormat? Find(string name) => Table.Find(name);

    /// <summary>The format of model type <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is registered; the message names <see cref="Register"/>.</exception>
    public static IImagePromptFormat Get(string name) => Table.Find(name)
        ?? throw new NotSupportedException($"No image prompt format is registered for model type '{name}' ({string.Join(", ", Table.Keys)}); add one with ImagePromptFormats.Register.");

    /// <summary>The library's format <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IImagePromptFormat? Default(string name) => Table.Default(name);

    /// <summary>Who registered the format <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>What happens when the app's format <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set: the error reaches the caller; <see cref="SlotPolicy.FallBack"/> retries on the library's).</summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    private sealed class GuardedFormat(Slot slot, IImagePromptFormat app, IImagePromptFormat library) : IImagePromptFormat
    {
        public string Name => app.Name;

        public string Expand(string prompt, int images, ImageTokenIds tokens, ITokenizer tokenizer) =>
            slot.Call(() => app.Expand(prompt, images, tokens, tokenizer), () => library.Expand(prompt, images, tokens, tokenizer), Comparisons.Exact);
    }
}
