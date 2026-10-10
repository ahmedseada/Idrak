// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json;
using Idrak.Nlp;

namespace Idrak.Samples.LegalOcr;

/// <summary>What the app records beside an adapter it tuned (<see cref="AdapterStore.RecordFile"/>): how to read with it and how its run went.</summary>
public sealed record AdapterRecord
{
    /// <summary>The adapter's name (its folder's).</summary>
    public required string Name { get; init; }

    /// <summary>The model of the switch it was tuned on.</summary>
    public string? Base { get; init; }

    /// <summary>The base checkpoint (a Hugging Face id or a folder), as its manifest names it.</summary>
    public required string BaseModel { get; init; }

    /// <summary>The instruction the training pages were sent with (reading with the adapter uses it).</summary>
    public string Prompt { get; init; } = "Extract details to JSON.";

    /// <summary>The image transforms it was tuned with.</summary>
    public string Preprocessing { get; init; } = "";

    /// <summary>The most tokens an answer may take when reading.</summary>
    public int MaxTokens { get; init; } = 2048;

    /// <summary>When the run started.</summary>
    public DateTimeOffset? Created { get; init; }

    /// <summary>Whether the run was stopped before its last step (the adapter is what was trained so far).</summary>
    public bool Stopped { get; init; }

    /// <summary>The optimizer steps taken, of the run's total.</summary>
    public int Steps { get; init; }

    /// <summary>The run's optimizer steps.</summary>
    public int TotalSteps { get; init; }

    /// <summary>The training time in seconds.</summary>
    public double Seconds { get; init; }

    /// <summary>The last training loss.</summary>
    public float? Loss { get; init; }

    /// <summary>The evaluation loss before and after.</summary>
    public float? EvaluationLossBefore { get; init; }

    /// <summary>The last evaluation loss.</summary>
    public float? EvaluationLoss { get; init; }

    /// <summary>The CER of the scored evaluation pages before training (the base model's).</summary>
    public double? CerBefore { get; init; }

    /// <summary>The last CER of the scored evaluation pages.</summary>
    public double? Cer { get; init; }

    /// <summary>The run's settings.</summary>
    public TuningDefaults? Settings { get; init; }
}

/// <summary>
/// The adapters the app tuned (or any adapter folder with the library's <see cref="TuningManifest"/> put there): one
/// folder each under the settings' <see cref="LegalOcrSettings.AdaptersFolder"/>, each a model of the switch
/// ("adapter:" and the folder's name) read as its base model with the adapter merged.
/// </summary>
public sealed class AdapterStore(LegalOcrSettings settings, string contentRoot)
{
    /// <summary>The app's record in an adapter folder.</summary>
    public const string RecordFile = "legal-ocr-adapter.json";

    /// <summary>The switch's prefix of an adapter's id.</summary>
    public const string Prefix = "adapter:";

    private static readonly JsonSerializerOptions Json = new()
    {
        WriteIndented = true, Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    /// <summary>The adapters' folder (full path).</summary>
    public string Folder => Path.GetFullPath(settings.AdaptersFolder, contentRoot);

    /// <summary>The adapters found, newest first: their folders and records (made from the manifest for a folder the app did not write).</summary>
    public IReadOnlyList<(string Folder, AdapterRecord Record)> List()
    {
        if (!Directory.Exists(Folder))
        {
            return [];
        }

        var found = new List<(string, AdapterRecord, DateTime)>();
        foreach (string folder in Directory.EnumerateDirectories(Folder))
        {
            if (Read(folder) is { } record)
            {
                found.Add((folder, record, Directory.GetLastWriteTimeUtc(folder)));
            }
        }

        return [.. found.OrderByDescending(f => f.Item3).Select(f => (f.Item1, f.Item2))];
    }

    /// <summary>The adapters as models of the switch.</summary>
    public IEnumerable<ReaderModel> Models() => List().Select(a => Model(a.Folder, a.Record));

    /// <summary>The adapter model of that switch id, or null.</summary>
    public ReaderModel? Find(string id) =>
        id.StartsWith(Prefix, StringComparison.OrdinalIgnoreCase) && Path.Combine(Folder, id[Prefix.Length..]) is var folder && IsName(id[Prefix.Length..])
            && Read(folder) is { } record ? Model(folder, record) : null;

    /// <summary>The folder a new adapter of that name goes to.</summary>
    /// <exception cref="ArgumentException">Not a name of letters, digits, '-', '_' and '.'.</exception>
    public string PathFor(string name) => IsName(name) ? Path.Combine(Folder, name)
        : throw new ArgumentException($"'{name}' is not an adapter name: letters, digits, '-', '_' and '.' only.");

    /// <summary>Writes the app's record into an adapter folder.</summary>
    public static void Write(string folder, AdapterRecord record) =>
        File.WriteAllText(Path.Combine(folder, RecordFile), JsonSerializer.Serialize(record, Json));

    /// <summary>An adapter folder's record, or null when it holds no adapter.</summary>
    public static AdapterRecord? Read(string folder)
    {
        if (TuningManifest.Read(folder) is not { } manifest)
        {
            return null;
        }

        string file = Path.Combine(folder, RecordFile);
        var images = TuningImages.Read(folder);
        var record = File.Exists(file) ? JsonSerializer.Deserialize<AdapterRecord>(File.ReadAllText(file), Json) : null;
        return (record ?? new AdapterRecord
        {
            Name = Path.GetFileName(folder), BaseModel = manifest.BaseModel, Preprocessing = images?.Pipeline.ToString() ?? "",
            Created = Directory.GetCreationTimeUtc(folder),
        }) with { Name = Path.GetFileName(folder), BaseModel = manifest.BaseModel };
    }

    // An adapter as a model of the switch: its base checkpoint with the adapter merged as the weights load.
    private static ReaderModel Model(string folder, AdapterRecord record)
    {
        bool local = Directory.Exists(record.BaseModel) || File.Exists(record.BaseModel);
        string about = $"Our LoRA on {record.BaseModel}"
                       + (record.Steps > 0 ? $", {record.Steps:N0}{(record.Stopped ? $" of {record.TotalSteps:N0}" : "")} steps" : "")
                       + (record.Cer is { } cer ? $", CER {cer:P2}" : "")
                       + (record.CerBefore is { } before ? $" (base {before:P2})" : "");
        return new ReaderModel
        {
            Id = Prefix + record.Name, Name = $"Ours · {record.Name}", Folder = local ? record.BaseModel : null, Repo = local ? null : record.BaseModel, Adapter = folder,
            About = about, Preprocessing = record.Preprocessing, Prompt = record.Prompt, MaxTokens = record.MaxTokens,
        };
    }

    private static bool IsName(string name) =>
        name.Length is > 0 and <= 80 && name.All(c => char.IsAsciiLetterOrDigit(c) || c is '-' or '_' or '.') && name.Trim('.').Length > 0;
}
