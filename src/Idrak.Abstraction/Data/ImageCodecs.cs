// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Data;

/// <summary>What an image file's header says.</summary>
/// <param name="Width">Pixels per row.</param>
/// <param name="Height">Rows.</param>
/// <param name="Channels">Channels once decoded: 1 for grey, 3 for colour (alpha is not counted).</param>
/// <param name="Format">The codec's name, e.g. "png".</param>
public readonly record struct ImageInfo(int Width, int Height, int Channels, string Format);

/// <summary>
/// Decoded pixels: [Channels, Height, Width] values in [0, 1], grey (1 channel) or red, green and blue (3 channels).
/// </summary>
public sealed class ImageData
{
    /// <summary>Wraps decoded pixels (used directly, not copied).</summary>
    public ImageData(float[] pixels, int channels, int height, int width)
    {
        ArgumentNullException.ThrowIfNull(pixels);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        if (pixels.Length != (long)channels * height * width)
        {
            throw new ArgumentException($"{pixels.Length} values are not {channels} x {height} x {width}.", nameof(pixels));
        }

        Pixels = pixels;
        Channels = channels;
        Height = height;
        Width = width;
    }

    /// <summary>[Channels, Height, Width] values in [0, 1].</summary>
    public float[] Pixels { get; }

    /// <summary>1 (grey) or 3 (red, green, blue).</summary>
    public int Channels { get; }

    /// <summary>Rows.</summary>
    public int Height { get; }

    /// <summary>Pixels per row.</summary>
    public int Width { get; }

    /// <summary>
    /// The image as <paramref name="channels"/> x <paramref name="height"/> x <paramref name="width"/> values written
    /// to <paramref name="destination"/>: grey repeated to colour, colour averaged to grey, and the size changed per
    /// axis: an axis that grows (or keeps its size) by bilinear sampling with the corners aligned, an axis that shrinks
    /// by the mean of the pixels each output pixel covers, so that no thin line falls between the samples.
    /// </summary>
    public void Resize(int channels, int height, int width, Span<float> destination)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        if (destination.Length != channels * height * width)
        {
            throw new ArgumentException($"The destination holds {destination.Length} values, not {channels} x {height} x {width}.", nameof(destination));
        }

        int c = Channels, h = Height, w = Width;
        var pixels = Pixels;
        var rows = Taps(h, height);
        var columns = Taps(w, width);
        for (int oc = 0; oc < channels; oc++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    float value = 0;
                    foreach (var (yy, wy) in rows[y])
                    {
                        float row = 0;
                        foreach (var (xx, wx) in columns[x])
                        {
                            row += Sample(yy, xx) * wx;
                        }

                        value += row * wy;
                    }

                    destination[oc * height * width + y * width + x] = value;

                    float Sample(int yy, int xx)
                    {
                        if (channels == c || c == 1)
                        {
                            return pixels[(c == 1 ? 0 : oc) * h * w + yy * w + xx];
                        }

                        float sum = 0;                                                  // colour to grey: the mean of the channels
                        for (int ic = 0; ic < c; ic++)
                        {
                            sum += pixels[ic * h * w + yy * w + xx];
                        }

                        return sum / c;
                    }
                }
            }
        }
    }

    // For each output position along an axis of `from` pixels resized to `to`, the source pixels it reads and their
    // weights (summing to 1). Growing or keeping the size: the two neighbours of bilinear sampling with the corners
    // aligned. Shrinking: every source pixel the output pixel's span [i, i + 1) * from / to covers, weighted by how much.
    private static (int Index, float Weight)[][] Taps(int from, int to)
    {
        var taps = new (int Index, float Weight)[to][];
        for (int i = 0; i < to; i++)
        {
            if (to >= from)
            {
                float s = to == 1 ? 0 : i * (from - 1) / (float)(to - 1);
                int i0 = (int)s, i1 = Math.Min(i0 + 1, from - 1);
                float f = s - i0;
                taps[i] = [(i0, 1 - f), (i1, f)];
                continue;
            }

            double start = i * (double)from / to, end = (i + 1) * (double)from / to;
            var covered = new List<(int Index, float Weight)>();
            for (int j = (int)start; j < Math.Min(from, (int)Math.Ceiling(end)); j++)
            {
                double overlap = Math.Min(end, j + 1) - Math.Max(start, j);
                if (overlap > 0)
                {
                    covered.Add((j, (float)(overlap / (end - start))));
                }
            }

            taps[i] = [.. covered];
        }

        return taps;
    }

    /// <summary>The image resized as <see cref="Resize(int, int, int, Span{float})"/> says, in a new array.</summary>
    public float[] Resize(int channels, int height, int width)
    {
        var result = new float[channels * height * width];
        Resize(channels, height, width, result);
        return result;
    }
}

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

    private static readonly List<IImageCodec> Registry = [new PngCodec(), new BmpCodec(), new NetpbmCodec()];

    /// <summary>Registers <paramref name="codec"/>, replacing one of the same name; a codec registered later is asked first.</summary>
    public static void Register(IImageCodec codec)
    {
        ArgumentNullException.ThrowIfNull(codec);
        ArgumentException.ThrowIfNullOrWhiteSpace(codec.Name);
        lock (Registry)
        {
            Registry.RemoveAll(c => string.Equals(c.Name, codec.Name, StringComparison.OrdinalIgnoreCase));
            Registry.Insert(0, codec);
        }
    }

    /// <summary>Removes the codec <paramref name="name"/>; returns whether it was registered.</summary>
    public static bool Unregister(string name)
    {
        lock (Registry)
        {
            return Registry.RemoveAll(c => string.Equals(c.Name, name, StringComparison.OrdinalIgnoreCase)) > 0;
        }
    }

    /// <summary>The registered codec names, the first asked first.</summary>
    public static IReadOnlyCollection<string> Names
    {
        get
        {
            lock (Registry)
            {
                return [.. Registry.Select(c => c.Name)];
            }
        }
    }

    /// <summary>The extensions of the registered codecs (with the dot, lower case).</summary>
    public static IReadOnlyCollection<string> Extensions
    {
        get
        {
            lock (Registry)
            {
                return [.. Registry.SelectMany(c => c.Extensions).Select(e => e.ToLowerInvariant()).Distinct()];
            }
        }
    }

    /// <summary>The codec registered as <paramref name="name"/> (any case).</summary>
    public static IImageCodec Get(string name)
    {
        lock (Registry)
        {
            return Registry.FirstOrDefault(c => string.Equals(c.Name, name, StringComparison.OrdinalIgnoreCase))
                ?? throw new NotSupportedException($"No image codec '{name}' is registered ({string.Join(", ", Registry.Select(c => c.Name))}); add it with ImageCodecs.Register.");
        }
    }

    /// <summary>Whether a registered codec lists the extension of <paramref name="path"/>.</summary>
    public static bool CanDecode(string path)
    {
        string extension = Path.GetExtension(path);
        lock (Registry)
        {
            return extension.Length > 0 && Registry.Any(c => c.Extensions.Contains(extension, StringComparer.OrdinalIgnoreCase));
        }
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

    private static IImageCodec[] Codecs()
    {
        lock (Registry)
        {
            return [.. Registry];
        }
    }
}
