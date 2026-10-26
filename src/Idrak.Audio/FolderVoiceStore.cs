// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Audio.Abstractions;
using Idrak.Models;

namespace Idrak.Audio;

/// <summary>
/// The built-in voice store (plan 21, S9): one folder per voice under <see cref="Root"/>, named as the voice, holding
/// voice.json, embedding.safetensors (an embedding voice), model/ (a checkpoint voice: the fine-tuned synthesizer) and
/// clips/ (only when the clips were kept). A voice is
/// written to a hidden folder first and moved into place, so a half-written voice is never listed.
/// </summary>
public sealed class FolderVoiceStore : ICheckpointVoiceStore
{
    /// <summary>The manifest's file name.</summary>
    public const string ManifestFile = "voice.json";

    /// <summary>An embedding voice's embedding (one F32 tensor "embedding" of the encoder's width).</summary>
    public const string EmbeddingFile = "embedding.safetensors";

    /// <summary>The version of voice.json this store writes.</summary>
    public const int ManifestVersion = 1;

    private static readonly JsonSerializerOptions Indented = new() { WriteIndented = true };

    /// <summary>A store at <paramref name="root"/> (made when the first voice is added), registered as <paramref name="name"/>.</summary>
    public FolderVoiceStore(string root, string name = VoiceStores.FolderName)
    {
        ArgumentException.ThrowIfNullOrEmpty(root);
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        Root = Path.GetFullPath(root);
        Name = name;
    }

    /// <summary>
    /// Where the built-in store keeps voices: <c>IDRAK_VOICES</c> when set, else voices/ under the library's cache
    /// folder (<see cref="ComputeResources.CacheFolder"/>).
    /// </summary>
    public static string DefaultRoot => Environment.GetEnvironmentVariable("IDRAK_VOICES") is { Length: > 0 } voices
        ? Path.GetFullPath(voices)
        : Path.Combine(ComputeResources.CacheFolder, "voices");

    /// <inheritdoc />
    public string Name { get; }

    /// <summary>The folder the voices are in.</summary>
    public string Root { get; }

    /// <inheritdoc />
    public string Location => Root;

    /// <inheritdoc />
    public IReadOnlyList<VoiceProfile> List()
    {
        if (!Directory.Exists(Root))
        {
            return [];
        }

        var voices = new List<VoiceProfile>();
        foreach (string folder in Directory.EnumerateDirectories(Root))
        {
            if (!Path.GetFileName(folder).StartsWith('.') && File.Exists(Path.Combine(folder, ManifestFile)))
            {
                voices.Add(Read(folder));
            }
        }

        voices.Sort((a, b) => string.Compare(a.Name, b.Name, StringComparison.OrdinalIgnoreCase));
        return voices;
    }

    /// <inheritdoc />
    public VoiceProfile? Find(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        if (!IsPlainName(name) || !Directory.Exists(Root))
        {
            return null;
        }

        // The folder as it is spelled on disk (a file system that ignores case would open Root/name as given, so the profile
        // would name the folder as asked, not as installed): the exact name first, else the name ignoring case.
        string? match = null;
        foreach (string folder in Directory.EnumerateDirectories(Root))
        {
            string folderName = Path.GetFileName(folder);
            if (string.Equals(folderName, name, StringComparison.OrdinalIgnoreCase) && File.Exists(Path.Combine(folder, ManifestFile)))
            {
                if (string.Equals(folderName, name, StringComparison.Ordinal))
                {
                    return Read(folder);
                }

                match ??= folder;
            }
        }

        return match is null ? null : Read(match);
    }

    /// <inheritdoc />
    public VoiceProfile Add(string name, IReadOnlyList<string> clips, ISpeakerEncoder encoder, VoiceAddOptions options, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(clips);
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(options);
        CheckName(name);
        if (options.Consent is not { } consent || string.IsNullOrWhiteSpace(consent.Speaker) || string.IsNullOrWhiteSpace(consent.Statement))
        {
            throw new ArgumentException($"Voice '{name}' has no consent: a voice is a person's likeness, so VoiceAddOptions.Consent (who is speaking and the statement they agreed to) is required.", nameof(options));
        }

        if (clips.Count == 0)
        {
            throw new ArgumentException($"Voice '{name}' needs at least one clip.", nameof(clips));
        }

        if (Find(name) is { } existing && !options.Overwrite)
        {
            throw new IOException($"Voice '{existing.Name}' is installed already at {existing.Folder}; set VoiceAddOptions.Overwrite to replace it.");
        }

        var audio = new AudioData[clips.Count];
        var hashes = new string[clips.Count];
        for (int i = 0; i < clips.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            using (var stream = File.OpenRead(clips[i]))
            {
                hashes[i] = Convert.ToHexStringLower(SHA256.HashData(stream));
            }

            audio[i] = Prepare(AudioCodecs.Decode(clips[i]), encoder.SampleRate, options);
            if (audio[i].Duration.TotalSeconds < options.MinClipSeconds)
            {
                throw new ArgumentException($"Clip {clips[i]} holds {audio[i].Duration.TotalSeconds:0.##} s of sound{(options.TrimSilence ? " after its silence is cut" : "")}, "
                    + $"under VoiceAddOptions.MinClipSeconds ({options.MinClipSeconds} s).", nameof(clips));
            }
        }

        var embedding = encoder.Embed(audio, cancellationToken);
        if (embedding.Length != encoder.EmbeddingSize)
        {
            throw new InvalidDataException($"Encoder '{encoder.Name}' gave {embedding.Length} values for an embedding of {encoder.EmbeddingSize}.");
        }

        var agreements = new float[clips.Count];
        for (int i = 0; i < clips.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            agreements[i] = clips.Count == 1 ? 1f : Cosine(encoder.Embed([audio[i]], cancellationToken), embedding);
            if (agreements[i] < options.MinAgreement)
            {
                throw new ArgumentException($"Clip {clips[i]} agrees with the voice by {agreements[i]:0.###} (the cosine of its embedding with the voice's), under VoiceAddOptions.MinAgreement "
                    + $"({options.MinAgreement}): another speaker or noise; leave it out or lower the setting.", nameof(clips));
            }
        }

        Directory.CreateDirectory(Root);
        string staging = Path.Combine(Root, $".{name}.{Guid.NewGuid():N}");
        Directory.CreateDirectory(staging);
        try
        {
            SafeTensorsWriter.Write(Path.Combine(staging, EmbeddingFile), [("embedding", [embedding.Length], embedding)], SafeTensorType.F32);
            var clipList = new JsonArray();
            for (int i = 0; i < clips.Count; i++)
            {
                string file = Path.GetFileName(clips[i]);
                if (options.KeepClips)
                {
                    Directory.CreateDirectory(Path.Combine(staging, "clips"));
                    file = UniqueName(Path.Combine(staging, "clips"), file);
                    File.Copy(clips[i], Path.Combine(staging, "clips", file));
                }

                clipList.Add((JsonNode)new JsonObject
                {
                    ["file"] = file,
                    ["sha256"] = hashes[i],
                    ["seconds"] = Math.Round(audio[i].Duration.TotalSeconds, 3),
                    ["agreement"] = Math.Round(agreements[i], 4),
                });
            }

            var manifest = new JsonObject
            {
                ["version"] = ManifestVersion,
                ["name"] = name,
                ["kind"] = VoiceKinds.Embedding,
                ["languages"] = new JsonArray([.. options.Languages.Select(l => (JsonNode?)JsonValue.Create(l))]),
                ["created"] = DateTimeOffset.UtcNow.ToString("o", CultureInfo.InvariantCulture),
                ["consent"] = new JsonObject
                {
                    ["speaker"] = consent.Speaker,
                    ["statement"] = consent.Statement,
                    ["agreed"] = (consent.Agreed == default ? DateTimeOffset.UtcNow : consent.Agreed).ToString("o", CultureInfo.InvariantCulture),
                },
                ["encoder"] = new JsonObject
                {
                    ["family"] = encoder.Name,
                    ["folder"] = encoder.Folder,
                    ["sampleRate"] = encoder.SampleRate,
                },
                ["embedding"] = new JsonObject { ["file"] = EmbeddingFile, ["size"] = embedding.Length },
                ["clips"] = clipList,
                ["clipsKept"] = options.KeepClips,
            };
            File.WriteAllText(Path.Combine(staging, ManifestFile), manifest.ToJsonString(Indented));
            return Place(staging, name, overwrite: true);
        }
        finally
        {
            if (Directory.Exists(staging))
            {
                Directory.Delete(staging, recursive: true);
            }
        }
    }

    /// <summary>A checkpoint voice's model folder within the voice's folder.</summary>
    public const string CheckpointFolder = "model";

    /// <inheritdoc />
    public VoiceProfile AddCheckpoint(string name, string modelFolder, VoiceCheckpointOptions options)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelFolder);
        ArgumentNullException.ThrowIfNull(options);
        CheckName(name);
        if (options.Consent is not { } consent || string.IsNullOrWhiteSpace(consent.Speaker) || string.IsNullOrWhiteSpace(consent.Statement))
        {
            throw new ArgumentException($"Voice '{name}' has no consent: a model trained on a person's recordings speaks as them, so VoiceCheckpointOptions.Consent "
                + "(who is speaking and the statement they agreed to) is required.", nameof(options));
        }

        if (!File.Exists(Path.Combine(modelFolder, "config.json")))
        {
            throw new FileNotFoundException($"{modelFolder} holds no config.json: give a trained model folder (a checkpoint the trainer wrote).", Path.Combine(modelFolder, "config.json"));
        }

        if (Find(name) is { } existing && !options.Overwrite)
        {
            throw new IOException($"Voice '{existing.Name}' is installed already at {existing.Folder}; set VoiceCheckpointOptions.Overwrite to replace it.");
        }

        Directory.CreateDirectory(Root);
        string staging = Path.Combine(Root, $".{name}.{Guid.NewGuid():N}");
        Directory.CreateDirectory(staging);
        try
        {
            var excluded = new HashSet<string>(options.Exclude, StringComparer.Ordinal);
            string model = Path.Combine(staging, CheckpointFolder);
            CopyFolder(modelFolder, model, file => !excluded.Contains(Path.GetRelativePath(modelFolder, file).Replace('\\', '/')));
            long bytes = Directory.EnumerateFiles(model, "*", SearchOption.AllDirectories).Sum(f => new FileInfo(f).Length);
            var manifest = new JsonObject
            {
                ["version"] = ManifestVersion,
                ["name"] = name,
                ["kind"] = VoiceKinds.Checkpoint,
                ["languages"] = new JsonArray([.. options.Languages.Select(l => (JsonNode?)JsonValue.Create(l))]),
                ["created"] = DateTimeOffset.UtcNow.ToString("o", CultureInfo.InvariantCulture),
                ["consent"] = new JsonObject
                {
                    ["speaker"] = consent.Speaker,
                    ["statement"] = consent.Statement,
                    ["agreed"] = (consent.Agreed == default ? DateTimeOffset.UtcNow : consent.Agreed).ToString("o", CultureInfo.InvariantCulture),
                },
                ["checkpoint"] = CheckpointFolder,
                ["checkpoint_bytes"] = bytes,
                ["base"] = options.Base,
                ["voice"] = options.Voice,
                ["training"] = options.Training?.DeepClone(),
            };
            File.WriteAllText(Path.Combine(staging, ManifestFile), manifest.ToJsonString(Indented));
            return Place(staging, name, overwrite: true);
        }
        finally
        {
            if (Directory.Exists(staging))
            {
                Directory.Delete(staging, recursive: true);
            }
        }
    }

    /// <inheritdoc />
    public float[] ReadEmbedding(VoiceProfile voice)
    {
        ArgumentNullException.ThrowIfNull(voice);
        string path = Path.Combine(voice.Folder, voice.Manifest["embedding"]?["file"]?.GetValue<string>() ?? EmbeddingFile);
        if (!File.Exists(path))
        {
            throw new InvalidDataException($"Voice '{voice.Name}' has no embedding ({path}); it is of kind '{voice.Kind}'.");
        }

        using var reader = SafeTensorsReader.Open(path);
        var values = reader.Read("embedding");
        if (voice.EmbeddingSize > 0 && values.Length != voice.EmbeddingSize)
        {
            throw new InvalidDataException($"Voice '{voice.Name}': {path} holds {values.Length} values, voice.json says {voice.EmbeddingSize}.");
        }

        return values;
    }

    /// <inheritdoc />
    public VoiceProfile Import(string folderOrZip, bool overwrite = false)
    {
        ArgumentException.ThrowIfNullOrEmpty(folderOrZip);
        Directory.CreateDirectory(Root);
        string staging = Path.Combine(Root, $".import.{Guid.NewGuid():N}");
        try
        {
            if (Directory.Exists(folderOrZip))
            {
                CopyFolder(folderOrZip, staging);
            }
            else if (File.Exists(folderOrZip))
            {
                ZipFile.ExtractToDirectory(folderOrZip, staging);                // refuses entries that would land outside the folder
                if (!File.Exists(Path.Combine(staging, ManifestFile)) && Directory.GetDirectories(staging) is [var only] && File.Exists(Path.Combine(only, ManifestFile)))
                {
                    string inner = staging + ".inner";                           // a zip of the folder itself, not its contents
                    Directory.Move(only, inner);
                    Directory.Delete(staging, recursive: true);
                    Directory.Move(inner, staging);
                }
            }
            else
            {
                throw new FileNotFoundException($"There is no voice folder or zip at {folderOrZip}.", folderOrZip);
            }

            var profile = Read(staging, checkName: false);
            CheckName(profile.Name);
            if (profile.Consent is null)
            {
                throw new InvalidDataException($"{folderOrZip}: voice '{profile.Name}' has no consent in its voice.json; a voice is installed only with one.");
            }

            if (profile.Kind == VoiceKinds.Embedding)
            {
                ReadEmbedding(profile);
            }
            else if (profile.CheckpointFolder is { } model && !File.Exists(Path.Combine(model, "config.json")))
            {
                throw new InvalidDataException($"{folderOrZip}: voice '{profile.Name}' is of kind {VoiceKinds.Checkpoint} and holds no model ({Path.GetRelativePath(staging, model)}/config.json).");
            }

            if (!overwrite && Find(profile.Name) is { } existing)
            {
                throw new IOException($"Voice '{existing.Name}' is installed already at {existing.Folder}; import with overwrite to replace it.");
            }

            return Place(staging, profile.Name, overwrite);
        }
        finally
        {
            if (Directory.Exists(staging))
            {
                Directory.Delete(staging, recursive: true);
            }
        }
    }

    /// <inheritdoc />
    public void Export(string name, string zip)
    {
        ArgumentException.ThrowIfNullOrEmpty(zip);
        var voice = Find(name) ?? throw NotInstalled(name);
        if (File.Exists(zip))
        {
            File.Delete(zip);
        }

        if (Path.GetDirectoryName(Path.GetFullPath(zip)) is { Length: > 0 } parent)
        {
            Directory.CreateDirectory(parent);
        }

        ZipFile.CreateFromDirectory(voice.Folder, zip, CompressionLevel.Optimal, includeBaseDirectory: false);
    }

    /// <inheritdoc />
    public bool Remove(string name)
    {
        if (Find(name) is not { } voice)
        {
            return false;
        }

        Directory.Delete(voice.Folder, recursive: true);
        return true;
    }

    /// <summary>Whether <paramref name="name"/> can name a voice: letters, digits, '.', '-' and '_', starting with a letter or digit, at most 64 characters.</summary>
    public static bool IsPlainName(string name) =>
        name is { Length: > 0 and <= 64 } && char.IsAsciiLetterOrDigit(name[0]) && name.All(c => char.IsAsciiLetterOrDigit(c) || c is '.' or '-' or '_');

    /// <summary>
    /// A clip as an encoder reads it: mixed to one channel, at <paramref name="sampleRate"/>, and (with
    /// <see cref="VoiceAddOptions.TrimSilence"/>) without the 10 ms windows at its ends whose peak is more than
    /// <see cref="VoiceAddOptions.SilenceDecibels"/> under the clip's.
    /// </summary>
    public static AudioData Prepare(AudioData clip, int sampleRate, VoiceAddOptions options)
    {
        ArgumentNullException.ThrowIfNull(clip);
        ArgumentNullException.ThrowIfNull(options);
        if (clip.Channels > 1)
        {
            var mono = GC.AllocateUninitializedArray<float>(clip.Length);
            clip.ToMono(mono);
            clip = new AudioData(mono, 1, clip.SampleRate);
        }

        if (clip.SampleRate != sampleRate)
        {
            clip = Resampler.Resample(clip, sampleRate);
        }

        if (!options.TrimSilence || clip.Length == 0)
        {
            return clip;
        }

        var samples = clip.Channel(0);
        float peak = 0;
        foreach (float s in samples)
        {
            peak = Math.Max(peak, Math.Abs(s));
        }

        float floor = peak * MathF.Pow(10f, -options.SilenceDecibels / 20f);
        int window = Math.Max(1, sampleRate / 100), windows = (samples.Length + window - 1) / window;
        int first = 0, last = windows - 1;
        while (first < windows && WindowPeak(samples, first, window) < floor)
        {
            first++;
        }

        while (last > first && WindowPeak(samples, last, window) < floor)
        {
            last--;
        }

        if (first >= windows)
        {
            return clip.Slice(0, 0);
        }

        int from = first * window, to = Math.Min(samples.Length, (last + 1) * window);
        return from == 0 && to == samples.Length ? clip : clip.Slice(from, to - from);
    }

    /// <summary>The cosine of two embeddings (0 when either is zero).</summary>
    public static float Cosine(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Embeddings of {a.Length} and {b.Length} values.", nameof(b));
        }

        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }

        return na == 0 || nb == 0 ? 0f : (float)(dot / Math.Sqrt(na * nb));
    }

    private static float WindowPeak(ReadOnlySpan<float> samples, int index, int window)
    {
        var part = samples.Slice(index * window, Math.Min(window, samples.Length - index * window));
        float peak = 0;
        foreach (float s in part)
        {
            peak = Math.Max(peak, Math.Abs(s));
        }

        return peak;
    }

    // Moves a written folder into place as `name`, replacing a voice of that name (ignoring case) when `overwrite`.
    private VoiceProfile Place(string staging, string name, bool overwrite)
    {
        if (Find(name) is { } existing)
        {
            if (!overwrite)
            {
                throw new IOException($"Voice '{existing.Name}' is installed already at {existing.Folder}.");
            }

            Directory.Delete(existing.Folder, recursive: true);
        }

        string target = Path.Combine(Root, name);
        Directory.Move(staging, target);
        return Read(target);
    }

    private VoiceProfile Read(string folder, bool checkName = true)
    {
        string path = Path.Combine(folder, ManifestFile);
        JsonObject manifest;
        try
        {
            manifest = JsonNode.Parse(File.ReadAllText(path))?.AsObject() ?? throw new InvalidDataException($"{path} is not a JSON object.");
        }
        catch (JsonException e)
        {
            throw new InvalidDataException($"{path} is not valid JSON: {e.Message}", e);
        }

        string name = checkName ? Path.GetFileName(folder) : manifest["name"]?.GetValue<string>() ?? throw new InvalidDataException($"{path} names no voice (\"name\").");
        string kind = manifest["kind"]?.GetValue<string>() ?? throw new InvalidDataException($"{path} gives no \"kind\" ({VoiceKinds.Speaker}, {VoiceKinds.Embedding}, {VoiceKinds.Checkpoint}).");
        var languages = manifest["languages"] is JsonArray l ? [.. l.Select(v => v!.GetValue<string>())] : Array.Empty<string>();
        return new VoiceProfile(name, kind, languages, Path.GetFullPath(folder), manifest);
    }

    private static void CheckName(string name)
    {
        if (!IsPlainName(name))
        {
            throw new ArgumentException($"'{name}' is not a voice name: letters, digits, '.', '-' and '_', starting with a letter or digit, at most 64 characters.", nameof(name));
        }
    }

    private static string UniqueName(string folder, string file)
    {
        string candidate = file;
        for (int n = 2; File.Exists(Path.Combine(folder, candidate)); n++)
        {
            candidate = $"{Path.GetFileNameWithoutExtension(file)}-{n}{Path.GetExtension(file)}";
        }

        return candidate;
    }

    private static void CopyFolder(string from, string to, Func<string, bool>? copy = null)
    {
        Directory.CreateDirectory(to);
        foreach (string file in Directory.EnumerateFiles(from))
        {
            if (copy?.Invoke(file) != false)
            {
                File.Copy(file, Path.Combine(to, Path.GetFileName(file)));
            }
        }

        foreach (string folder in Directory.EnumerateDirectories(from))
        {
            CopyFolder(folder, Path.Combine(to, Path.GetFileName(folder)), copy);
        }
    }

    private InvalidOperationException NotInstalled(string name) =>
        new($"Voice '{name}' is not installed in {Root} (installed: {(List() is { Count: > 0 } all ? string.Join(", ", all.Select(v => v.Name)) : "none")}).");
}
