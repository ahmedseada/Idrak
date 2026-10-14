// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Imaging;
using Idrak.Nlp;
using Idrak.Nlp.Abstractions;

namespace Idrak.Samples.LegalOcr;

/// <summary>A "Make training pages" request from the page: each one null keeps the settings' value (<see cref="SyntheticSettings"/>).</summary>
public sealed record SyntheticRequest
{
    public string? FontsFolder { get; init; }

    public string? PartsFile { get; init; }

    public string? Output { get; init; }

    public int? Variants { get; init; }

    public List<string>? Layouts { get; init; }

    public List<string>? Scans { get; init; }

    public double? Grey { get; init; }

    public string? Page { get; init; }

    public int? Seed { get; init; }

    public int? Quality { get; init; }

    public int? Limit { get; init; }

    /// <summary>Pages a preview shows.</summary>
    public int? Count { get; init; }

    /// <summary>The settings with this request's values over them.</summary>
    public SyntheticSettings Over(SyntheticSettings s) => s with
    {
        FontsFolder = FontsFolder ?? s.FontsFolder, PartsFile = PartsFile ?? s.PartsFile, Output = Output is { Length: > 0 } o ? o : s.Output,
        Variants = Variants ?? s.Variants, Layouts = Layouts is { Count: > 0 } l ? l : s.Layouts, Scans = Scans is { Count: > 0 } c ? c : s.Scans,
        Grey = Grey ?? s.Grey, Page = Page ?? s.Page, Seed = Seed ?? s.Seed, Quality = Quality ?? s.Quality, Limit = Limit ?? s.Limit,
    };
}

/// <summary>Where making pages is.</summary>
public enum SyntheticState
{
    Idle,
    Running,
    Finished,
    Stopped,
    Failed,
}

/// <summary>Making pages as the page shows it.</summary>
public sealed record SyntheticStatus(
    SyntheticState State, string Message, int Pages, int Total, int Skipped, double Seconds, double? SecondsLeft, double PagesPerSecond, string? Output, string? LastSkip);

/// <summary>A preview page: the page as a JPEG data URL, its text and how it was made.</summary>
public sealed record SyntheticPreview(string Image, string Text, string Layout, string Scan, string Description, int Width, int Height);

/// <summary>
/// The Fine-tune tab's "Make training pages": the training records' answer text drawn as new pages by the library
/// (<see cref="SyntheticPages"/>), previewed a few at a time or made in the background with live progress, written beside
/// the app (<see cref="SyntheticSettings.Output"/>), where a tuning run takes them with the training pages.
/// </summary>
public sealed class SyntheticService : IDisposable
{
    private readonly LegalOcrSettings _settings;
    private readonly string _contentRoot;
    private readonly Lock _lock = new();
    private readonly Dictionary<string, IReadOnlyList<IFontFace>> _fonts = new(StringComparer.Ordinal);   // read once a folder
    private SyntheticStatus _status = new(SyntheticState.Idle, "No pages made yet.", 0, 0, 0, 0, null, 0, null, null);
    private CancellationTokenSource? _stop;
    private Task? _task;

    public SyntheticService(LegalOcrSettings settings, AdapterStore adapters)
    {
        _settings = settings;
        _contentRoot = adapters.ContentRoot;
    }

    /// <summary>The made records' file (full path) for <paramref name="settings"/>.</summary>
    public static string DataPath(LegalOcrSettings settings, string contentRoot) => DataPath(settings.Synthetic.Output, contentRoot);

    private static string DataPath(string output, string contentRoot)
    {
        string path = Path.GetFullPath(output, contentRoot);
        return path.EndsWith(".jsonl", StringComparison.OrdinalIgnoreCase) ? path : path + ".jsonl";
    }

    public SyntheticStatus Status
    {
        get
        {
            lock (_lock)
            {
                return _status;
            }
        }
    }

    /// <summary>What the panel starts from: the settings, the layouts and scan profiles there are, and what is found where.</summary>
    public object Defaults()
    {
        var s = _settings.Synthetic.Effective();
        string? fonts = s.FontsFolder is { Length: > 0 } f ? Path.GetFullPath(f, _contentRoot) : null;
        string? parts = s.PartsFile is { Length: > 0 } p ? Path.GetFullPath(p, _contentRoot) : null;
        string data = DataPath(_settings, _contentRoot);
        return new
        {
            settings = s,
            layouts = PageLayouts.Names.Select(n => new { name = n, summary = PageLayouts.Get(n).Summary }),
            scans = ScanProfiles.Names.Select(n => new { name = n, summary = ScanProfiles.Get(n).Summary }).Append(new { name = "none", summary = "the page as drawn" }),
            pageKeys = SyntheticPages.StyleKeys,
            fontsFolder = fonts,
            fontsFound = fonts is not null && Directory.Exists(fonts) ? Directory.EnumerateFiles(fonts).Count(IsFont) : 0,
            partsFile = parts,
            partsFound = parts is not null && File.Exists(parts),
            trainingData = _settings.TrainingData,
            trainingDataFound = _settings.TrainingData is { } t && File.Exists(t),
            output = data,
            outputPages = File.Exists(data) ? File.ReadLines(data).Count(l => l.Length > 0) : 0,
        };
    }

    private static bool IsFont(string path) => Path.GetExtension(path).ToLowerInvariant() is ".ttf" or ".otf" or ".ttc";

    /// <summary>The library's options for <paramref name="request"/> over the settings.</summary>
    /// <exception cref="ArgumentException">A setting is not valid, or the fonts or the training data are missing (the message says which).</exception>
    public SyntheticPageOptions Options(SyntheticRequest request)
    {
        var s = request.Over(_settings.Synthetic.Effective());
        if (s.FontsFolder is not { Length: > 0 } fontsFolder)
        {
            throw new ArgumentException("Set the fonts folder (Synthetic.FontsFolder in appsettings.json, or on the page): a folder of .ttf, .otf or .ttc fonts with Arabic.");
        }

        string folder = Path.GetFullPath(fontsFolder, _contentRoot);
        if (!Directory.Exists(folder))
        {
            throw new ArgumentException($"The fonts folder {folder} is not there.");
        }

        IReadOnlyList<IFontFace> fonts;
        lock (_lock)
        {
            if (!_fonts.TryGetValue(folder, out fonts!))
            {
                _fonts[folder] = fonts = SyntheticPages.Fonts(folder);
            }
        }

        if (fonts.Count == 0)
        {
            throw new ArgumentException($"No font was read from {folder} (.ttf, .otf, .ttc).");
        }

        try
        {
            var model = _settings.Models.FirstOrDefault(m => m.Id == _settings.Tuning.Base);
            var options = new SyntheticPageOptions
            {
                Style = SyntheticPages.Style(new PageStyle { Fonts = fonts }, s.Page),
                Parts = s.PartsFile is { Length: > 0 } parts ? SyntheticPageParts.Read(Path.GetFullPath(parts, _contentRoot)) : SyntheticPageParts.Empty,
                Variants = s.Variants, Layouts = s.Layouts, Scans = s.Scans, Grey = s.Grey, Seed = s.Seed, Quality = s.Quality, Workers = s.Workers, Limit = s.Limit,
                FullTextKey = model?.FullText,
            };
            return options;
        }
        catch (Exception e) when (e is InvalidDataException or FileNotFoundException or DirectoryNotFoundException or NotSupportedException)
        {
            throw new ArgumentException(e.Message, e);
        }
    }

    private string TrainingData() => _settings.TrainingData is { } t && File.Exists(t)
        ? t
        : throw new ArgumentException($"The training data {_settings.TrainingData ?? "(none)"} was not found: set TrainingData in appsettings.json.");

    /// <summary>The first records' pages (<see cref="SyntheticRequest.Count"/>, 3 unless given), each as the first of its variants would be made.</summary>
    public IReadOnlyList<SyntheticPreview> Preview(SyntheticRequest request)
    {
        var options = Options(request);
        string data = TrainingData();
        int count = Math.Clamp(request.Count ?? 3, 1, 12);
        var format = SyntheticRecordFormats.Detect(data);
        var previews = new List<SyntheticPreview>();
        foreach (var (record, index) in format.Read(data, options.Prompt, options.AnswerKey).Take(count).Select((r, i) => (r, i)))
        {
            try
            {
                var page = SyntheticPages.Make(record.Answer, options, index, 0, record.Where);
                string image = "data:image/jpeg;base64," + Convert.ToBase64String(JpegEncoder.Encode(page.Image, 85, optimize: true));
                previews.Add(new SyntheticPreview(image, page.Text, page.Layout, page.Scan, page.Description, page.Image.Width, page.Image.Height));
            }
            catch (InvalidDataException e)
            {
                previews.Add(new SyntheticPreview("", e.Message, "", "", "left out", 0, 0));
            }
        }

        return previews;
    }

    /// <summary>Starts making the pages in the background.</summary>
    /// <exception cref="InvalidOperationException">Pages are being made.</exception>
    /// <exception cref="ArgumentException">A setting is not valid.</exception>
    public SyntheticStatus Start(SyntheticRequest request)
    {
        var options = Options(request);
        string data = TrainingData();
        string output = DataPath(request.Over(_settings.Synthetic).Output, _contentRoot);
        lock (_lock)
        {
            if (_status.State == SyntheticState.Running)
            {
                throw new InvalidOperationException("Pages are being made: stop first.");
            }

            _stop = new CancellationTokenSource();
            _status = new SyntheticStatus(SyntheticState.Running, $"Counting the records of {Path.GetFileName(data)}…", 0, 0, 0, 0, null, 0, output, null);
            var token = _stop.Token;
            var clock = Stopwatch.StartNew();
            _task = Task.Run(() =>
            {
                try
                {
                    var result = SyntheticPages.Write(data, output, options, new Reports(this, output), token);
                    Set(_status with
                    {
                        State = SyntheticState.Finished, Seconds = result.Elapsed.TotalSeconds, SecondsLeft = 0, Pages = result.Pages, Skipped = result.Skips.Count,
                        Message = $"{result.Pages:N0} pages from {result.Records:N0} records in {result.Elapsed.TotalSeconds:F0} s{(result.Skips.Count > 0 ? $", {result.Skips.Count} left out" : "")}: {result.Data}",
                    });
                }
                catch (OperationCanceledException)
                {
                    Set(_status with { State = SyntheticState.Stopped, Seconds = clock.Elapsed.TotalSeconds, SecondsLeft = null, Message = $"Stopped after {_status.Pages:N0} pages; those are written: {output}" });
                }
                catch (Exception e)
                {
                    Set(_status with { State = SyntheticState.Failed, Seconds = clock.Elapsed.TotalSeconds, SecondsLeft = null, Message = e.Message });
                }
            });
            return _status;
        }
    }

    /// <summary>Stops making pages; the pages made so far stay written.</summary>
    public SyntheticStatus Stop()
    {
        lock (_lock)
        {
            _stop?.Cancel();
            return _status;
        }
    }

    /// <summary>Waits for the pages being made (tests).</summary>
    public Task WaitAsync() => _task ?? Task.CompletedTask;

    private void Set(SyntheticStatus status)
    {
        lock (_lock)
        {
            _status = status;
        }
    }

    public void Dispose()
    {
        _stop?.Cancel();
        try
        {
            _task?.Wait(TimeSpan.FromSeconds(5));
        }
        catch (AggregateException)
        {
        }

        _stop?.Dispose();
    }

    private sealed class Reports(SyntheticService service, string output) : IProgress<SyntheticPagesProgress>
    {
        public void Report(SyntheticPagesProgress p) => service.Set(service.Status with
        {
            Pages = p.Pages, Total = p.Total, Skipped = p.Skipped, Seconds = p.Elapsed.TotalSeconds, PagesPerSecond = Math.Round(p.PagesPerSecond, 2),
            SecondsLeft = p.Remaining.TotalSeconds, LastSkip = p.LastSkip, Output = output,
            Message = $"Drawing pages: {p.Pages:N0} of {p.Total:N0}{(p.Skipped > 0 ? $", {p.Skipped} records left out" : "")}",
        });
    }
}
