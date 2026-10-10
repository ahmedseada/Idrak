// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Text.Json;
using Idrak.Data;
using Idrak.Nlp;

namespace Idrak.Samples.ArabicOcr;

/// <summary>
/// A page (or a line) to read: its name, its encoded image and, when it came from fine-tuning data, the conversation's
/// prompt and the expected answer.
/// </summary>
/// <param name="Name">A file name's stem, or <c>DATA-0003</c> for record 3 of a data file.</param>
/// <param name="Source">The image file, or <c>val.json #3</c>.</param>
/// <param name="Image">The encoded image (bytes as stored).</param>
/// <param name="Truth">The expected answer (a data file's assistant turn), or null.</param>
/// <param name="Prompt">The data's user text beside the image (its <c>&lt;image&gt;</c> placeholder removed), or null.</param>
/// <param name="System">The data's system message, or null.</param>
internal sealed record Page(string Name, string Source, ChatImage Image, string? Truth = null, string? Prompt = null, string? System = null)
{
    /// <summary>The pixels: the registered codecs, then the EXIF orientation (as the vision-language reader sees them).</summary>
    public ImageData Decode() => ChatImageDecoder.Decode(Image);
}

/// <summary>
/// The pages the commands read: image files, folders of them (sorted by name), and fine-tuning data files (.json,
/// .jsonl: ShareGPT or chat messages, detected by <see cref="TuningDataFormats"/>; their images in the data's folder, or
/// in <c>--images</c>: a folder or a .zip read in place).
/// </summary>
internal static class Pages
{
    /// <summary>Whether <paramref name="path"/> is a fine-tuning data file rather than an image.</summary>
    public static bool IsDataFile(string path) => Path.GetExtension(path).ToLowerInvariant() is ".json" or ".jsonl";

    /// <summary>The image files of a folder, sorted by name (ordinal), or the file itself.</summary>
    public static IEnumerable<string> ImageFiles(string path, bool recursive = false) =>
        Directory.Exists(path)
            ? Directory.EnumerateFiles(path, "*", recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly)
                .Where(ImageCodecs.CanDecode).Order(StringComparer.Ordinal)
            : File.Exists(path) ? [path] : throw new UsageException($"Not found: {path}");

    /// <summary>Every page of the inputs, in order (data records lazily, one at a time).</summary>
    public static IEnumerable<Page> Read(IReadOnlyList<string> inputs, string? images)
    {
        if (inputs.Count == 0)
        {
            throw new UsageException("Give the images to read: files, folders, or a data file (.json, .jsonl).");
        }

        foreach (string input in inputs)
        {
            if (IsDataFile(input))
            {
                foreach (var page in FromData(input, images))
                {
                    yield return page;
                }

                continue;
            }

            foreach (string file in ImageFiles(input))
            {
                yield return new Page(Path.GetFileNameWithoutExtension(file), file, ChatImage.FromFile(file));
            }
        }
    }

    /// <summary>
    /// How many pages the inputs hold, for progress, without reading an image: image files, and a data file's records (a
    /// .json array's elements, a .jsonl file's non-blank lines; a record without an image is skipped later). Null when a
    /// data file is neither.
    /// </summary>
    public static int? Count(IReadOnlyList<string> inputs)
    {
        int total = 0;
        foreach (string input in inputs)
        {
            if (!IsDataFile(input))
            {
                total += ImageFiles(input).Count();
                continue;
            }

            if (Records(input) is not { } records)
            {
                return null;
            }

            total += records;
        }

        return total;
    }

    // The records of a data file, counted over its bytes (rule 74: bytes, not strings).
    private static int? Records(string path)
    {
        if (!File.Exists(path))
        {
            return null;
        }

        if (Path.GetExtension(path).Equals(".jsonl", StringComparison.OrdinalIgnoreCase))
        {
            int lines = 0;
            bool content = false;
            byte[] buffer = ArrayPool<byte>.Shared.Rent(64 * 1024);
            try
            {
                using var stream = File.OpenRead(path);
                int read;
                while ((read = stream.Read(buffer, 0, buffer.Length)) > 0)
                {
                    foreach (byte b in buffer.AsSpan(0, read))
                    {
                        if (b == (byte)'\n')
                        {
                            lines += content ? 1 : 0;
                            content = false;
                        }
                        else if (b is not ((byte)' ' or (byte)'\r' or (byte)'\t'))
                        {
                            content = true;
                        }
                    }
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(buffer);
            }

            return lines + (content ? 1 : 0);
        }

        var reader = new Utf8JsonReader(File.ReadAllBytes(path), new JsonReaderOptions { CommentHandling = JsonCommentHandling.Skip });
        if (!reader.Read() || reader.TokenType != JsonTokenType.StartArray)
        {
            return null;
        }

        int elements = 0;
        while (reader.Read() && reader.TokenType != JsonTokenType.EndArray)
        {
            elements++;
            reader.Skip();
        }

        return elements;
    }

    /// <summary>The records of a fine-tuning data file as pages: each record's first image, its prompt and its answer.</summary>
    public static IEnumerable<Page> FromData(string path, string? images)
    {
        if (!File.Exists(path))
        {
            throw new UsageException($"Not found: {path}");
        }

        string stem = Path.GetFileNameWithoutExtension(path);
        string file = Path.GetFileName(path);
        int record = 0;
        foreach (var transcript in TuningDataFormats.Read(path, options: new TuningDataOptions { Images = images }))
        {
            record++;
            var user = transcript.Messages.FirstOrDefault(m => m.Role == "user" && m.Parts.OfType<ChatImage>().Any());
            if (user is null)
            {
                continue;                                                                    // a record without an image: nothing to read
            }

            var image = user.Parts.OfType<ChatImage>().First();
            string prompt = string.Join("\n", user.Parts.OfType<ChatText>().Select(t => t.Text.Replace("<image>", "", StringComparison.Ordinal).Trim()).Where(t => t.Length > 0));
            string? system = transcript.Messages.FirstOrDefault(m => m.Role == "system")?.Content;
            string? truth = transcript.Messages.LastOrDefault(m => m.Role == "assistant")?.Content;
            yield return new Page($"{stem}-{record:D4}", $"{file} #{record}", image, truth, prompt.Length > 0 ? prompt : null, system);
        }
    }

    /// <summary>
    /// The known texts of a data file's pages by their image's identity (<see cref="ChatImage.Hash"/>), so a scan given as a
    /// file finds its page's answer.
    /// </summary>
    public static Dictionary<string, Page> ByImage(string path, string? images)
    {
        var pages = new Dictionary<string, Page>(StringComparer.Ordinal);
        foreach (var page in FromData(path, images))
        {
            pages.TryAdd(page.Image.Hash, page);
        }

        return pages;
    }
}
