// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Samples.ArabicOcr;

/// <summary>A line image with its transcription and where the transcription came from.</summary>
/// <param name="Kind">"corrected" (a <c>.txt</c>), "aligned" (a <c>.aligned.txt</c> draft snapped to a page's known text) or "draft" (a <c>.draft.txt</c>: the vision-language reader's reading).</param>
internal sealed record LineSample(string Image, string Text, string Kind);

/// <summary>
/// The files of a line folder, as <c>cut</c> writes them: <c>NAME.png</c>, and beside it the transcription. <c>NAME.txt</c>
/// is the corrected (accepted) text, and an empty one means "not transcribed yet". Drafts are kept apart so they are never
/// taken for corrected text: <c>NAME.aligned.txt</c> (the vision-language reader's reading snapped to the page's known
/// text) and <c>NAME.draft.txt</c> (its reading as it is). Accepting a draft is renaming it to <c>NAME.txt</c> (or the
/// <c>accept</c> command).
/// </summary>
internal static class LineFiles
{
    public const string Corrected = ".txt";
    public const string Aligned = ".aligned.txt";
    public const string Draft = ".draft.txt";

    /// <summary>The transcription file of <paramref name="image"/> of a kind (<see cref="Corrected"/>, <see cref="Aligned"/>, <see cref="Draft"/>).</summary>
    public static string Of(string image, string kind) => Path.Combine(Path.GetDirectoryName(image) ?? "", Path.GetFileNameWithoutExtension(image) + kind);

    /// <summary>Whether <paramref name="image"/> has a corrected (non-empty) <c>.txt</c>.</summary>
    public static bool IsCorrected(string image) => OcrText.ReadTranscription(Of(image, Corrected)).Length > 0;

    /// <summary>
    /// The lines of <paramref name="folder"/> (and its subfolders) with a transcription: the corrected one, or with
    /// <paramref name="drafts"/> the aligned draft, else the draft, for lines not corrected yet. Lines without any are left
    /// out. The transcriptions are read in parallel (tens of thousands of small files); <paramref name="progress"/> gets
    /// (read, total) as they come in. The order is the folder's.
    /// </summary>
    public static IReadOnlyList<LineSample> Read(string folder, bool drafts, Action<int, int>? progress = null)
    {
        if (!Directory.Exists(folder))
        {
            throw new UsageException($"No folder {folder}: give a folder of line images with their .txt transcriptions (cut writes one).");
        }

        string[] images = [.. Pages.ImageFiles(folder, recursive: true)];
        var found = new LineSample?[images.Length];
        int read = 0;
        progress?.Invoke(0, images.Length);
        Parallel.For(0, images.Length, ComputeResources.ParallelOptions, i =>
        {
            foreach (var (kind, name) in Kinds)
            {
                if (kind != Corrected && !drafts)
                {
                    break;
                }

                string text = OcrText.ReadTranscription(Of(images[i], kind));
                if (text.Length > 0)
                {
                    found[i] = new LineSample(images[i], text, name);
                    break;
                }
            }

            int done = Interlocked.Increment(ref read);
            progress?.Invoke(done, images.Length);
        });

        var samples = new List<LineSample>(images.Length);
        foreach (var sample in found)
        {
            if (sample is not null)
            {
                samples.Add(sample);
            }
        }

        return samples;
    }

    private static readonly (string Kind, string Name)[] Kinds = [(Corrected, "corrected"), (Aligned, "aligned"), (Draft, "draft")];
}
