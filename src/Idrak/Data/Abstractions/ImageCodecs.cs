// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Data.Abstractions;

/// <summary>What an image file's header says.</summary>
/// <param name="Width">Pixels per row.</param>
/// <param name="Height">Rows.</param>
/// <param name="Channels">Channels once decoded: 1 for grey, 3 for colour (alpha is not counted).</param>
/// <param name="Format">The codec's name, e.g. "png".</param>
public readonly record struct ImageInfo(int Width, int Height, int Channels, string Format);

/// <summary>
/// Reads one image format; register it with <see cref="ImageCodecs.Register"/>. A codec decides from a file's first
/// bytes whether the file is in its format, so the extension only decides which files a folder source picks up.
/// </summary>
public interface IImageCodec
{
    /// <summary>The format's name, e.g. "png" (names ignore case).</summary>
    string Name { get; }

    /// <summary>The file extensions of the format, with the dot, e.g. ".png".</summary>
    IReadOnlyCollection<string> Extensions { get; }

    /// <summary>
    /// The size and channels from the start of a file (its first 64 KiB, or all of a smaller file), or null when the
    /// bytes are not this format.
    /// </summary>
    ImageInfo? ReadInfo(ReadOnlySpan<byte> header);

    /// <summary>Decodes a whole file; a variant the codec does not read is an <see cref="InvalidDataException"/>.</summary>
    ImageData Decode(ReadOnlySpan<byte> file);
}

/// <summary>
/// The image formats that <c>ImageFolderSource</c> and the command-line tool read, by name (ignoring case). Built
/// in, without dependencies: "png" (every bit depth and colour type, interlaced or not, inflated with the zlib in .NET),
/// "bmp" (1, 4, 8, 16, 24 and 32 bits, uncompressed or with bit fields) and "netpbm" (PGM and PPM, text or binary).
/// Alpha is dropped. JPEG is not built in: register a codec for it (or any other format) with <see cref="Register"/>.
/// </summary>
public static class ImageCodecs
{
    /// <summary>The bytes of a file's start that <see cref="IImageCodec.ReadInfo"/> receives.</summary>
    public const int HeaderBytes = 64 * 1024;

    // The most recently registered first; the built-ins png, bmp, netpbm in that order.
    private static readonly SlotTable<string, IImageCodec> Registry = BuiltIn();

    private static SlotTable<string, IImageCodec> BuiltIn()
    {
        var table = new SlotTable<string, IImageCodec>(nameof(ImageCodecs), (slot, app, library) => new GuardedCodec(slot, app, library),
            StringComparer.OrdinalIgnoreCase, newestFirst: true);
        foreach (var codec in (IImageCodec[])[new NetpbmCodec(), new BmpCodec(), new PngCodec()])
        {
            table.RegisterDefault(codec.Name, codec);
        }

        return table;
    }

    /// <summary>
    /// Registers <paramref name="codec"/>: one of a registered name takes its place (a built-in stays behind it as its
    /// fallback, see <see cref="SetPolicy"/>), a new one is asked first.
    /// </summary>
    public static void Register(IImageCodec codec)
    {
        ArgumentNullException.ThrowIfNull(codec);
        ArgumentException.ThrowIfNullOrWhiteSpace(codec.Name);
        Registry.Register(codec.Name, codec);
    }

    /// <summary>Removes the app's codec <paramref name="name"/> (a built-in name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered codec names, the first asked first.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The extensions of the registered codecs (with the dot, lower case).</summary>
    public static IReadOnlyCollection<string> Extensions => [.. Registry.Values.SelectMany(c => c.Extensions).Select(e => e.ToLowerInvariant()).Distinct()];

    /// <summary>The codec registered as <paramref name="name"/> (any case).</summary>
    public static IImageCodec Get(string name) => Registry.Find(name)
        ?? throw new NotSupportedException($"No image codec '{name}' is registered ({string.Join(", ", Names)}); add it with ImageCodecs.Register.");

    /// <summary>The library's codec <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IImageCodec? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the codec <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>What happens when the app's codec <paramref name="name"/> fails (<see cref="SlotPolicy.FallBack"/> to the library's unless set).</summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(name, policy, shadowRate);

    /// <summary>Whether a registered codec lists the extension of <paramref name="path"/>.</summary>
    public static bool CanDecode(string path)
    {
        string extension = Path.GetExtension(path);
        return extension.Length > 0 && Registry.Values.Any(c => c.Extensions.Contains(extension, StringComparer.OrdinalIgnoreCase));
    }

    /// <summary>The size and channels of an image file, or null when no registered codec knows its format.</summary>
    public static ImageInfo? ReadInfo(string path)
    {
        var head = new byte[HeaderBytes];
        int length;
        using (var stream = File.OpenRead(path))
        {
            length = stream.ReadAtLeast(head, head.Length, throwOnEndOfStream: false);
        }

        foreach (var codec in Codecs())
        {
            if (codec.ReadInfo(head.AsSpan(0, length)) is { } info)
            {
                return info;
            }
        }

        return null;
    }

    /// <summary>Decodes an image file with the codec that knows its format.</summary>
    /// <exception cref="InvalidDataException">No registered codec reads the file; the message names the file and the codecs.</exception>
    public static ImageData Decode(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var header = bytes.AsSpan(0, Math.Min(bytes.Length, HeaderBytes));
        foreach (var codec in Codecs())
        {
            if (codec.ReadInfo(header) is not null)
            {
                try
                {
                    return codec.Decode(bytes);
                }
                catch (Exception e) when (e is IndexOutOfRangeException or ArgumentOutOfRangeException or ArgumentException or EndOfStreamException)
                {
                    throw new InvalidDataException($"{path}: a damaged {codec.Name} file ({e.Message}).", e);
                }
                catch (InvalidDataException e)
                {
                    throw new InvalidDataException($"{path}: {e.Message}", e);
                }
            }
        }

        throw new InvalidDataException($"{path}: not an image format of the registered codecs ({string.Join(", ", Names)}); register one with ImageCodecs.Register.");
    }

    /// <summary>Decodes an image file and resizes it to <paramref name="channels"/> x <paramref name="height"/> x <paramref name="width"/> (<see cref="ImageData.Resize(int, int, int)"/>).</summary>
    public static float[] Load(string path, int channels, int height, int width) => Decode(path).Resize(channels, height, width);

    private static IReadOnlyList<IImageCodec> Codecs() => Registry.Values;

    // An app's codec over a built-in: each call falls back on its own, or is compared with the built-in's.
    private sealed class GuardedCodec(Slot slot, IImageCodec app, IImageCodec library) : IImageCodec
    {
        public string Name => app.Name;

        public IReadOnlyCollection<string> Extensions => app.Extensions;

        public ImageInfo? ReadInfo(ReadOnlySpan<byte> header)
        {
            if (slot.Policy == SlotPolicy.Shadow)
            {
                var run = slot.Shadow();
                var answer = library.ReadInfo(header);
                if (run is not null)
                {
                    run.Answered();
                    try
                    {
                        run.Done(Comparisons.Exact(answer, app.ReadInfo(header)));
                    }
                    catch (Exception e) when (e is not OperationCanceledException)
                    {
                        run.Failed(e);
                    }
                }

                return answer;
            }

            try
            {
                return app.ReadInfo(header);
            }
            catch (Exception e) when (slot.FallsBack(e))
            {
                return library.ReadInfo(header);
            }
        }

        public ImageData Decode(ReadOnlySpan<byte> file)
        {
            if (slot.Policy == SlotPolicy.Shadow)
            {
                var run = slot.Shadow();
                var answer = library.Decode(file);
                if (run is not null)
                {
                    run.Answered();
                    try
                    {
                        var other = app.Decode(file);
                        run.Done(Comparisons.Exact((answer.Channels, answer.Height, answer.Width), (other.Channels, other.Height, other.Width))
                                 ?? Comparisons.Difference(answer.Pixels, other.Pixels, 1e-5f));
                    }
                    catch (Exception e) when (e is not OperationCanceledException)
                    {
                        run.Failed(e);
                    }
                }

                return answer;
            }

            try
            {
                return app.Decode(file);
            }
            catch (Exception e) when (slot.FallsBack(e))
            {
                return library.Decode(file);
            }
        }
    }
}
