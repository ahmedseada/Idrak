// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using Idrak.Data;
using Idrak.Data.Abstractions;
using Idrak.Generation;
using Idrak.Generation.Abstractions;
using Idrak.Models;

namespace Idrak.Cli.Shared;

/// <summary>
/// Images given to a vision-language model on the command line (<c>run --image</c>, <c>chat --image</c>, <c>/image</c>):
/// the files as chat image parts, decoded by the library's codecs (<see cref="ImageCodecs"/>) and turned upright by their
/// EXIF orientation as transformers' <c>load_image</c> does for a path (<c>ImageOps.exif_transpose</c>); the codecs and
/// the processors themselves apply no orientation, so this is done here, where files are read.
/// </summary>
internal static class ImageInputs
{
    /// <summary>The files as chat image parts; a usage error names a file that does not exist.</summary>
    public static IReadOnlyList<ChatImage> Read(IEnumerable<string> paths)
    {
        var images = new List<ChatImage>();
        foreach (string path in paths)
        {
            images.Add(File.Exists(path) ? ChatImage.FromFile(path) : throw new UsageException($"Image file not found: {path}"));
        }

        return images;
    }

    /// <summary>A chat image's pixels: decoded by the codec that reads its bytes (<see cref="ImageCodecs"/>), then turned by its EXIF orientation.</summary>
    /// <exception cref="InvalidDataException">No registered codec reads it.</exception>
    public static ImageData Decode(ChatImage image)
    {
        ArgumentNullException.ThrowIfNull(image);
        return Orient(ImageCodecs.Decode(image.Data.Span), Orientation(image.Data.Span));
    }

    /// <summary>
    /// The EXIF orientation of an image file (1 to 8; 1 when it has none): tag 0x0112 of the first IFD, read from a JPEG's
    /// APP1 "Exif" segment or a PNG's eXIf chunk.
    /// </summary>
    public static int Orientation(ReadOnlySpan<byte> file)
    {
        try
        {
            if (file.Length > 4 && file[0] == 0xFF && file[1] == 0xD8)
            {
                int at = 2;
                while (at + 4 <= file.Length && file[at] == 0xFF)
                {
                    int marker = file[at + 1];
                    if (marker is 0xD8 or 0x01 or >= 0xD0 and <= 0xD7 or 0xFF)
                    {
                        at += marker == 0xFF ? 1 : 2;                               // fill bytes and markers without a length
                        continue;
                    }

                    if (marker is 0xDA or 0xD9)
                    {
                        break;                                                      // the scan: no EXIF after it
                    }

                    int length = BinaryPrimitives.ReadUInt16BigEndian(file[(at + 2)..]);
                    var segment = file.Slice(at + 4, Math.Min(length - 2, file.Length - at - 4));
                    if (marker == 0xE1 && segment.StartsWith("Exif\0\0"u8))
                    {
                        return Tiff(segment[6..]);
                    }

                    at += 2 + length;
                }
            }
            else if (file.Length > 8 && file[..8].SequenceEqual((ReadOnlySpan<byte>)[0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A]))
            {
                int at = 8;
                while (at + 8 <= file.Length)
                {
                    int length = (int)BinaryPrimitives.ReadUInt32BigEndian(file[at..]);
                    var type = file.Slice(at + 4, 4);
                    if (length < 0 || at + 8 + length > file.Length || type.SequenceEqual("IEND"u8))
                    {
                        break;
                    }

                    if (type.SequenceEqual("eXIf"u8))
                    {
                        var data = file.Slice(at + 8, length);
                        return Tiff(data.StartsWith("Exif\0\0"u8) ? data[6..] : data);
                    }

                    at += 12 + length;
                }
            }
        }
        catch (ArgumentOutOfRangeException)
        {
            // A damaged segment: no orientation.
        }

        return 1;
    }

    // Tag 0x0112 of a TIFF structure's first IFD (a SHORT, or a LONG as some writers store it); 1 when absent or out of range.
    private static int Tiff(ReadOnlySpan<byte> tiff)
    {
        if (tiff.Length < 8)
        {
            return 1;
        }

        bool little = tiff[0] == 'I' && tiff[1] == 'I';
        if (!little && !(tiff[0] == 'M' && tiff[1] == 'M'))
        {
            return 1;
        }

        long ifd = U32(tiff, 4, little);
        if (ifd < 8 || ifd + 2 > tiff.Length)
        {
            return 1;
        }

        int entries = U16(tiff, (int)ifd, little);
        for (int i = 0; i < entries; i++)
        {
            int entry = (int)ifd + 2 + 12 * i;
            if (entry + 12 > tiff.Length)
            {
                break;
            }

            if (U16(tiff, entry, little) != 0x0112)
            {
                continue;
            }

            long value = U16(tiff, entry + 2, little) switch { 3 => U16(tiff, entry + 8, little), 4 => U32(tiff, entry + 8, little), _ => 1 };
            return value is >= 1 and <= 8 ? (int)value : 1;
        }

        return 1;
    }

    private static int U16(ReadOnlySpan<byte> data, int at, bool little) =>
        little ? BinaryPrimitives.ReadUInt16LittleEndian(data[at..]) : BinaryPrimitives.ReadUInt16BigEndian(data[at..]);

    private static long U32(ReadOnlySpan<byte> data, int at, bool little) =>
        little ? BinaryPrimitives.ReadUInt32LittleEndian(data[at..]) : BinaryPrimitives.ReadUInt32BigEndian(data[at..]);

    /// <summary>
    /// The image turned upright for EXIF orientation <paramref name="orientation"/>, as Pillow's <c>exif_transpose</c>:
    /// 2 mirrored, 3 turned half a turn, 4 flipped, 5 transposed, 6 turned a quarter clockwise, 7 transversed, 8 turned a
    /// quarter anticlockwise; 1 and anything else unchanged.
    /// </summary>
    public static ImageData Orient(ImageData image, int orientation)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (orientation is < 2 or > 8)
        {
            return image;
        }

        int c = image.Channels, h = image.Height, w = image.Width;
        bool swap = orientation >= 5;
        int oh = swap ? w : h, ow = swap ? h : w;
        var source = image.Pixels;
        var pixels = new float[source.Length];
        for (int y = 0; y < oh; y++)
        {
            for (int x = 0; x < ow; x++)
            {
                // The source pixel (row, column) shown at (y, x).
                var (sy, sx) = orientation switch
                {
                    2 => (y, w - 1 - x),
                    3 => (h - 1 - y, w - 1 - x),
                    4 => (h - 1 - y, x),
                    5 => (x, y),
                    6 => (h - 1 - x, y),
                    7 => (h - 1 - x, w - 1 - y),
                    _ => (x, w - 1 - y),
                };
                for (int k = 0; k < c; k++)
                {
                    pixels[(k * oh + y) * ow + x] = source[(k * h + sy) * w + sx];
                }
            }
        }

        return new ImageData(pixels, c, oh, ow);
    }

    /// <summary>
    /// Makes the image encoder of a model: given the model and the preprocessor its images are read with, a function from
    /// decoded images to their features [images, tokens per image, text width] on the model's device, and what to dispose
    /// when done. Null: the library's (<see cref="PretrainedVision.CreateEncoder"/>). Tests set it to feed reference features.
    /// </summary>
    internal static Func<PretrainedModel, ImagePreprocessor, (Func<IReadOnlyList<ImageData>, Tensor> Encode, IDisposable? Owner)>? EncoderFactory { get; set; }

    /// <summary>
    /// How a chat generator reads <paramref name="model"/>'s images (<see cref="ChatGenerator.Images"/>), or null with
    /// the reason when it reads none: a text model, or a family without a registered image prompt format. The encoder is
    /// built when the first image is read (and disposed with the returned owner): <paramref name="grayscale"/> turns
    /// images grey first (<see cref="PretrainedVision.Preprocessor"/>, Pillow's <c>convert("L")</c>).
    /// </summary>
    public static ModelImages? For(PretrainedModel model, bool grayscale, out string? reason)
    {
        ArgumentNullException.ThrowIfNull(model);
        reason = null;
        if (model.Vision is not { } vision)
        {
            return null;
        }

        string type = (string?)model.Config["model_type"] ?? "";
        if (ImagePromptFormats.Find(type) is not { } format)
        {
            reason = $"no image prompt format is registered for its model type '{type}' ({string.Join(", ", ImagePromptFormats.Names)})";
            return null;
        }

        return new ModelImages(model, vision, format, vision.Preprocessor(grayscale));
    }
}

/// <summary>A model's image reading for chat (<see cref="Images"/>), owning the encoder it builds on first use.</summary>
internal sealed class ModelImages : IDisposable
{
    private readonly Lock _lock = new();
    private (Func<IReadOnlyList<ImageData>, Tensor> Encode, IDisposable? Owner)? _encoder;

    public ModelImages(PretrainedModel model, PretrainedVision vision, IImagePromptFormat format, ImagePreprocessor preprocessor)
    {
        Preprocessor = preprocessor;
        Images = new ChatImages(vision.ImageTokens, format, images =>
        {
            Func<IReadOnlyList<ImageData>, Tensor> encode;
            lock (_lock)
            {
                _encoder ??= (ImageInputs.EncoderFactory ?? LibraryEncoder)(model, preprocessor);
                encode = _encoder.Value.Encode;
            }

            return encode([.. images.Select(ImageInputs.Decode)]);
        });
    }

    /// <summary>How images become pixel values (the model's preprocessing, grey when asked).</summary>
    public ImagePreprocessor Preprocessor { get; }

    /// <summary>What the chat generator reads images with.</summary>
    public ChatImages Images { get; }

    private static (Func<IReadOnlyList<ImageData>, Tensor>, IDisposable?) LibraryEncoder(PretrainedModel model, ImagePreprocessor preprocessor)
    {
        var encoder = model.Vision!.CreateEncoder(model.Device, preprocessor);
        return (encoder.Encode, encoder);
    }

    public void Dispose()
    {
        lock (_lock)
        {
            _encoder?.Owner?.Dispose();
            _encoder = null;
        }
    }
}
