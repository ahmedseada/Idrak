// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Nlp;
using Idrak.Nlp.Abstractions;

namespace Idrak.Samples.LegalOcr;

/// <summary>One evaluation page: its image, the instruction the data gives with it, and the expected answer.</summary>
public sealed record EvaluationPage(int Index, string Name, ChatImage Image, string? Prompt, string Expected);

/// <summary>What the page list shows of an evaluation page.</summary>
public sealed record EvaluationPageInfo(int Index, string Name, string? Prompt, int ExpectedCharacters, string Preview);

/// <summary>
/// The evaluation pages: the conversations of the settings' data file (ShareGPT or messages, through the library's
/// <see cref="TuningDataFormats"/>) with their images from a folder or a zip, read once on first use; a conversation
/// counts when its user turn has an image and an assistant turn follows.
/// </summary>
public sealed class EvaluationPages(LegalOcrSettings settings)
{
    private readonly Lock _lock = new();
    private IReadOnlyList<EvaluationPage>? _pages;

    /// <summary>Whether the settings name a data file that exists.</summary>
    public bool Configured => settings.EvaluationData is { } file && File.Exists(file);

    /// <summary>Where the pages come from, for the page.</summary>
    public string? Source => settings.EvaluationData;

    /// <summary>Why the data file could not be read (a missing image, a bad record), or null.</summary>
    public string? Error { get; private set; }

    /// <summary>
    /// The pages (empty when none are configured, or when the file could not be read: <see cref="Error"/> says why, and
    /// the next call reads it again).
    /// </summary>
    public IReadOnlyList<EvaluationPage> All
    {
        get
        {
            if (!Configured)
            {
                return [];
            }

            lock (_lock)
            {
                if (_pages is null)
                {
                    try
                    {
                        _pages = Read(settings);
                        Error = null;
                    }
                    catch (Exception ex) when (ex is IOException or InvalidDataException or NotSupportedException or UnauthorizedAccessException)
                    {
                        Error = ex.Message;
                        return [];
                    }
                }

                return _pages;
            }
        }
    }

    /// <summary>The page at <paramref name="index"/>, or null.</summary>
    public EvaluationPage? Get(int index) => index >= 0 && index < All.Count ? All[index] : null;

    /// <summary>The pages as the list shows them.</summary>
    public IEnumerable<EvaluationPageInfo> List() => All.Select(p =>
        new EvaluationPageInfo(p.Index, p.Name, p.Prompt, p.Expected.Length, AnswerTexts.Of(p.Expected, AnswerTexts.Values) is var v && v.Length > 120 ? v[..120] + "…" : v));

    // The data file's conversations as the library's answer scorer splits them (the messages before the last assistant
    // turn are the prompt, its text the reference), those whose prompt has an image.
    private static List<EvaluationPage> Read(LegalOcrSettings settings)
    {
        string file = settings.EvaluationData!;
        var transcripts = TuningDataFormats.Detect(file).Read(file, new TuningDataOptions { Images = settings.EvaluationImages });
        var pages = new List<EvaluationPage>();
        foreach (var (prompt, reference, _, _) in new TuningAnswerScorer(transcripts).Items)
        {
            var user = prompt.LastOrDefault(m => m.Role == "user");
            if (user?.Parts.OfType<ChatImage>().FirstOrDefault() is not { } image)
            {
                continue;
            }

            string text = string.Join("\n", user.Parts.OfType<ChatText>().Select(t => t.Text)).Trim();
            pages.Add(new EvaluationPage(pages.Count, $"{Path.GetFileNameWithoutExtension(file)}-{pages.Count + 1:D4}", image, text.Length > 0 ? text : null, reference));
        }

        return pages;
    }
}
