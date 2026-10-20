// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Cli.Commands.Train;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands.Voices;

/// <summary>What the voice commands share: the store option, how a voice is shown, the clips' agreement line.</summary>
internal static class VoiceText
{
    /// <summary>The help line of <c>--voices</c>, aligned as the commands' options are.</summary>
    public const string StoreHelp = "      --voices DIR       the installed voices' folder (default IDRAK_VOICES, else voices/ under the cache)";

    /// <summary>The agreement under which a clip is flagged when no <c>--warn-agreement</c> is given.</summary>
    public const double WarnAgreement = 0.8;

    /// <summary>The voice's lines: kind, languages, encoder, embedding, consent, clips (flagged under <paramref name="warn"/>), folder.</summary>
    public static IEnumerable<string> Lines(VoiceProfile voice, double warn)
    {
        yield return $"{voice.Name}: {voice.Kind}, languages [{string.Join(", ", voice.Languages)}], encoder {voice.Encoder ?? "-"}"
                     + (voice.EncoderFolder is null ? "" : $" ({voice.EncoderFolder})") + (voice.EmbeddingSize > 0 ? $", embedding {voice.EmbeddingSize} wide" : "");
        yield return voice.Consent is { } consent
            ? string.Create(CultureInfo.InvariantCulture, $"  consent: {consent.Speaker} agreed {consent.Agreed:u}: \"{consent.Statement}\"")
            : "  consent: none recorded";
        if (voice.CheckpointFolder is { } model)
        {
            var training = voice.Manifest["training"];
            yield return $"  model: {model} (fine-tuned from {voice.Base ?? "?"}"
                         + (training?["steps"] is { } steps ? string.Create(CultureInfo.InvariantCulture, $", {steps} steps on {training["utterances"]} utterances") : "") + ")";
        }

        foreach (var clip in voice.Clips)
        {
            yield return string.Create(CultureInfo.InvariantCulture, $"  clip {clip.File}: {clip.Seconds:0.0} s, agreement {clip.Agreement:0.000}")
                         + (clip.Agreement < warn ? "  <- far from the others: another speaker or noise?" : "");
        }

        yield return $"  folder: {voice.Folder}";
    }

    /// <summary>The voice as JSON: its manifest's figures, the clips with their flags, the folder.</summary>
    public static JsonObject Json(VoiceProfile voice, double warn) => new()
    {
        ["name"] = voice.Name,
        ["kind"] = voice.Kind,
        ["languages"] = new JsonArray([.. voice.Languages.Select(l => (JsonNode)l)]),
        ["encoder"] = voice.Encoder,
        ["encoder_folder"] = voice.EncoderFolder,
        ["embedding_size"] = voice.EmbeddingSize,
        ["consent"] = voice.Consent is { } c ? new JsonObject { ["speaker"] = c.Speaker, ["statement"] = c.Statement, ["agreed"] = c.Agreed.ToString("o", CultureInfo.InvariantCulture) } : null,
        ["clips"] = new JsonArray([.. voice.Clips.Select(c => (JsonNode)new JsonObject
        {
            ["file"] = c.File, ["sha256"] = c.Sha256, ["seconds"] = c.Seconds, ["agreement"] = c.Agreement, ["flagged"] = c.Agreement < warn,
        })]),
        ["folder"] = voice.Folder,
        ["checkpoint"] = voice.CheckpointFolder,
        ["base"] = voice.Base,
        ["training"] = voice.Manifest["training"]?.DeepClone(),
    };

    /// <summary>The voice of <paramref name="name"/> in the store, or a failure naming the store and its voices.</summary>
    public static VoiceProfile Find(IVoiceStore store, string name) => store.Find(name)
        ?? throw new InvalidOperationException($"Voice '{name}' is not installed in {store.Location} (installed: "
                                               + (store.List() is { Count: > 0 } all ? string.Join(", ", all.Select(v => v.Name)) : "none") + ").");

    /// <summary>A store's refusal with the command line's option names in place of the library's.</summary>
    public static string Options(string message) => message.Split(" (Parameter", 2)[0]
        .Replace("set VoiceAddOptions.Overwrite to replace it", "give --overwrite to replace it", StringComparison.Ordinal)
        .Replace("VoiceAddOptions.MinClipSeconds", "--min-seconds", StringComparison.Ordinal)
        .Replace("VoiceAddOptions.MinAgreement", "--min-agreement", StringComparison.Ordinal)
        .Replace("VoiceAddOptions.Consent", "--consent", StringComparison.Ordinal);
}

/// <summary><c>idrak voice add NAME --from CLIP... --converter DIR --consent TEXT</c>: installs a person's voice from a few clips.</summary>
internal sealed class VoiceAddCommand : Command
{
    public override string Name => "voice add";

    public override string Summary => "Install a person's voice from a few clips of them, with their consent, for idrak speak --voice";

    public override string Usage => """
        NAME --from CLIP|DIR... --converter DIR --consent TEXT|--consent-file F [options]

        Decodes the clips (WAV, FLAC; MP3 with its plug-in), cuts the silence at their ends, embeds them with the voice
        converter's own speaker encoder, and installs the voice: voice.json (the consent, the clips' hashes, lengths
        and agreement, the converter) and the embedding. Each clip's agreement is the cosine of its embedding with the
        voice's: a clip far below the others is another speaker or noise, and is flagged. A voice is a person's
        likeness: --consent (who agreed, to what) is required, stored and shown by voice show; the library cannot verify
        it. The converter family is a plug-in (Idrak.Audio.Plugins' OpenVoice is built in; another with -P): its folder
        holds config.json and the weights.

        Arguments:
          NAME                   the voice's name (letters, digits, . - _)

        Options:
              --from CLIP|DIR    a clip, a folder of clips or a glob (repeatable; clips after NAME count too); 1 to 3
                                 minutes of one person's normal speech in all
              --converter DIR    the voice converter's folder (or a hub id in the cache), read by its plug-in
              --consent TEXT     the speaker's agreement, in their words
              --consent-file F   the agreement read from a UTF-8 file
              --speaker WHO      who is speaking (default: NAME)
          -l, --language CODE    a language the voice is meant for (repeatable; informational: conversion works on any)
              --keep-clips       copy the clips into the voice's folder (they are the person's recordings: off by default)
              --no-trim          keep the silence at the clips' ends
              --silence-db D     how far under a clip's peak its end counts as silence, dB (default 40)
              --min-seconds S    refuse a clip shorter than S seconds after trimming (default 1)
              --min-agreement A  refuse a clip whose agreement is under A (default 0: none is refused)
              --warn-agreement A flag a clip whose agreement is under A (default 0.8)
              --overwrite        replace an installed voice of the same name
        """ + "\n" + VoiceText.StoreHelp + "\n" + """

        Examples:
          idrak voice add ahmed --from clips/ --converter ./openvoice-v2 -l ar -l en --consent "I, Ahmed, agree that this voice is used for speech made with Idrak."
          idrak voice add sara --from a.wav --from b.flac --converter ./openvoice-v2 --consent-file sara-consent.txt -d cuda
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } =
        ["--from", "--converter", "--consent", "--consent-file", "--speaker", "--language", "--silence-db", "--min-seconds", "--min-agreement", "--warn-agreement", "--voices"];

    public override IReadOnlyCollection<string> Flags { get; } = ["--keep-clips", "--no-trim", "--overwrite"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-l"] = "--language" };

    public override int Run(CommandContext context)
    {
        string name = context.Argument(0, "NAME (the voice's name)");
        if (!FolderVoiceStore.IsPlainName(name))
        {
            throw new UsageException($"'{name}' is not a voice name: letters, digits, '.', '-' and '_', starting with a letter or digit, at most 64 characters.");
        }

        var clips = AudioFiles.Expand(context.Options("--from").Concat(context.Positional.Skip(1)));
        if (clips.Count == 0)
        {
            throw new UsageException("Give the clips with --from CLIP|DIR (one person's speech, 1 to 3 minutes in all).");
        }

        string statement = context.Option("--consent")
            ?? (context.Option("--consent-file") is { } file ? File.Exists(file) ? File.ReadAllText(file).Trim() : throw new UsageException($"Consent file not found: {file}") : null)
            ?? throw new UsageException("A voice is a person's likeness: give their agreement with --consent TEXT or --consent-file F (it is stored with the voice and shown by voice show).");
        if (statement.Length == 0)
        {
            throw new UsageException("The consent is empty: give the speaker's agreement, in their words.");
        }

        string converterArgument = context.Option("--converter") ?? throw new UsageException("Missing --converter DIR (the voice converter's folder; its family a registered plug-in's).");
        double warn = AudioFiles.Number(context, "--warn-agreement", VoiceText.WarnAgreement, -1, 1);
        var defaults = new VoiceAddOptions();
        var options = new VoiceAddOptions
        {
            Consent = new VoiceConsent(context.Option("--speaker") ?? name, statement, DateTimeOffset.UtcNow),
            Languages = [.. context.Options("--language")],
            KeepClips = context.Flag("--keep-clips"),
            TrimSilence = !context.Flag("--no-trim"),
            SilenceDecibels = (float)AudioFiles.Number(context, "--silence-db", defaults.SilenceDecibels, double.Epsilon),
            MinClipSeconds = AudioFiles.Number(context, "--min-seconds", defaults.MinClipSeconds, 0),
            MinAgreement = (float)AudioFiles.Number(context, "--min-agreement", defaults.MinAgreement, -1, 1),
            Overwrite = context.Flag("--overwrite"),
        };
        var store = AudioFiles.VoiceStore(context);
        string converterFolder = Directory.Exists(converterArgument) ? converterArgument : ModelChoices.Resolve(context, converterArgument);
        var device = context.Device;
        try
        {
            IVoiceConverter converter;
            using (var loading = new ProgressLine(context, $"loading the converter {converterArgument} on {device}", unit: ProgressUnit.Elapsed))
            {
                converter = VoiceConverters.Load(converterFolder, new VoiceConverterLoadOptions { Device = device });
                loading.Clear();
            }

            using (converter)
            {
                VoiceProfile voice;
                using (var embedding = new ProgressLine(context, $"embedding {clips.Count} clip(s) for '{name}' with {converter.Name}", unit: ProgressUnit.Elapsed))
                {
                    voice = store.Add(name, clips, converter, options, context.TimeoutToken);
                    embedding.Clear();
                }

                context.Write($"installed '{voice.Name}' in {store.Location}");
                foreach (string line in VoiceText.Lines(voice, warn))
                {
                    context.Write(line);
                }

                int flagged = voice.Clips.Count(c => c.Agreement < warn);
                if (flagged > 0)
                {
                    context.Error(string.Create(CultureInfo.InvariantCulture,
                        $"{flagged} clip(s) agree with the voice by less than {warn}: listen to them; reinstall without them (--overwrite) if they are another speaker or noise."));
                }

                context.Write($"next: idrak speak MODEL \"...\" --voice {voice.Name} -o {voice.Name}.wav");
                context.WriteJson(VoiceText.Json(voice, warn));
                return ExitCodes.Ok;
            }
        }
        catch (NotSupportedException e) when (ImageModelFiles.NotRegistered(e))
        {
            return ImageModelFiles.Refuse(context, e);
        }
        catch (Exception e) when (e is ArgumentException or IOException)
        {
            context.Error(VoiceText.Options(e.Message));
            return ExitCodes.Failed;
        }
    }
}

/// <summary><c>idrak voice list</c>: the installed voices.</summary>
internal sealed class VoiceListCommand : Command
{
    public override string Name => "voice list";

    public override IReadOnlyCollection<string> Aliases => ["voice ls"];

    public override string Summary => "The installed voices: kind, languages, clips and the converter they were made with";

    public override string Usage => """
        [options]

        Options:
        """ + "\n" + VoiceText.StoreHelp + "\n" + """

        Examples:
          idrak voice list
          idrak voice ls --format csv
          idrak voice list -j
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } = ["--voices"];

    public override int Run(CommandContext context)
    {
        var store = AudioFiles.VoiceStore(context);
        var voices = store.List();
        if (voices.Count == 0)
        {
            context.Write($"no voices installed in {store.Location} (idrak voice add NAME --from CLIPS --converter DIR --consent TEXT)");
        }
        else
        {
            context.Table(["Voice", "Kind", "Languages", "Clips", "Encoder"], voices.Select(v => (IReadOnlyList<string>)
                [v.Name, v.Kind, string.Join(",", v.Languages), string.Create(CultureInfo.InvariantCulture, $"{v.Clips.Count} ({v.Clips.Sum(c => c.Seconds):0.0} s)"), v.Encoder ?? "-"]));
            context.Write($"({store.Location})");
        }

        context.WriteJson(new JsonObject
        {
            ["store"] = store.Location,
            ["voices"] = new JsonArray([.. voices.Select(v => (JsonNode)VoiceText.Json(v, VoiceText.WarnAgreement))]),
        });
        return ExitCodes.Ok;
    }
}

/// <summary><c>idrak voice show NAME</c>: one voice, its consent and its clips' agreement.</summary>
internal sealed class VoiceShowCommand : Command
{
    public override string Name => "voice show";

    public override string Summary => "One installed voice: its consent, clips and their agreement, converter and folder";

    public override string Usage => """
        NAME [options]

        Options:
              --warn-agreement A flag a clip whose agreement is under A (default 0.8)
        """ + "\n" + VoiceText.StoreHelp + "\n" + """

        Examples:
          idrak voice show ahmed
          idrak voice show ahmed -j
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } = ["--voices", "--warn-agreement"];

    public override int Run(CommandContext context)
    {
        string name = context.Argument(0, "NAME (an installed voice)");
        double warn = AudioFiles.Number(context, "--warn-agreement", VoiceText.WarnAgreement, -1, 1);
        var voice = VoiceText.Find(AudioFiles.VoiceStore(context), name);
        foreach (string line in VoiceText.Lines(voice, warn))
        {
            context.Write(line);
        }

        context.WriteJson(VoiceText.Json(voice, warn));
        return ExitCodes.Ok;
    }
}

/// <summary><c>idrak voice remove NAME</c>: deletes an installed voice.</summary>
internal sealed class VoiceRemoveCommand : Command
{
    public override string Name => "voice remove";

    public override IReadOnlyCollection<string> Aliases => ["voice rm"];

    public override string Summary => "Remove an installed voice (its folder: manifest, embedding and kept clips)";

    public override string Usage => """
        NAME [options]

        Options:
        """ + "\n" + VoiceText.StoreHelp + "\n" + """

        Examples:
          idrak voice remove ahmed
          idrak voice rm old-voice
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } = ["--voices"];

    public override int Run(CommandContext context)
    {
        string name = context.Argument(0, "NAME (an installed voice)");
        var store = AudioFiles.VoiceStore(context);
        bool removed = store.Remove(name);
        context.WriteJson(new JsonObject { ["name"] = name, ["removed"] = removed, ["store"] = store.Location });
        if (!removed)
        {
            context.Error($"Voice '{name}' is not installed in {store.Location}.");
            return ExitCodes.Failed;
        }

        context.Write($"removed '{name}' from {store.Location}");
        return ExitCodes.Ok;
    }
}

/// <summary><c>idrak voice export NAME -o voice.zip</c>: one voice as a zip another machine imports.</summary>
internal sealed class VoiceExportCommand : Command
{
    public override string Name => "voice export";

    public override string Summary => "Write an installed voice to a zip (manifest, consent, embedding) for idrak voice import";

    public override string Usage => """
        NAME -o ZIP [options]

        Options:
          -o, --out ZIP          the zip to write
        """ + "\n" + VoiceText.StoreHelp + "\n" + """

        Examples:
          idrak voice export ahmed -o ahmed-voice.zip
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } = ["--out", "--voices"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-o"] = "--out" };

    public override int Run(CommandContext context)
    {
        string name = context.Argument(0, "NAME (an installed voice)");
        string zip = context.Option("--out") ?? (context.Positional.Count > 1 ? context.Positional[1] : throw new UsageException("Missing -o, --out ZIP (the zip to write)."));
        var store = AudioFiles.VoiceStore(context);
        var voice = VoiceText.Find(store, name);
        if (Path.GetDirectoryName(Path.GetFullPath(zip)) is { } folder)
        {
            Directory.CreateDirectory(folder);
        }

        store.Export(voice.Name, zip);
        context.Write($"wrote {Path.GetFullPath(zip)} ({Units.Bytes(new FileInfo(zip).Length)}); install it elsewhere with idrak voice import");
        context.WriteJson(new JsonObject { ["name"] = voice.Name, ["out"] = Path.GetFullPath(zip), ["bytes"] = new FileInfo(zip).Length });
        return ExitCodes.Ok;
    }
}

/// <summary><c>idrak voice import ZIP|DIR</c>: installs a voice a zip or a voice folder holds.</summary>
internal sealed class VoiceImportCommand : Command
{
    public override string Name => "voice import";

    public override string Summary => "Install a voice from a zip idrak voice export wrote, or from a voice folder";

    public override string Usage => """
        ZIP|DIR [options]

        A voice without a consent record is refused.

        Options:
              --overwrite        replace an installed voice of the same name
        """ + "\n" + VoiceText.StoreHelp + "\n" + """

        Examples:
          idrak voice import ahmed-voice.zip
          idrak voice import ./shared/voices/sara --overwrite
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } = ["--voices"];

    public override IReadOnlyCollection<string> Flags { get; } = ["--overwrite"];

    public override int Run(CommandContext context)
    {
        string source = context.Argument(0, "ZIP|DIR (a zip voice export wrote, or a voice folder)");
        if (!File.Exists(source) && !Directory.Exists(source))
        {
            throw new UsageException($"Not found: {source}");
        }

        var store = AudioFiles.VoiceStore(context);
        try
        {
            var voice = store.Import(source, context.Flag("--overwrite"));
            context.Write($"installed '{voice.Name}' in {store.Location}");
            foreach (string line in VoiceText.Lines(voice, VoiceText.WarnAgreement))
            {
                context.Write(line);
            }

            context.WriteJson(VoiceText.Json(voice, VoiceText.WarnAgreement));
            return ExitCodes.Ok;
        }
        catch (Exception e) when (e is IOException or InvalidDataException or ArgumentException)
        {
            context.Error(VoiceText.Options(e.Message).Replace("set overwrite", "give --overwrite", StringComparison.Ordinal));
            return ExitCodes.Failed;
        }
    }
}

/// <summary>
/// <c>idrak voice pull OWNER/NAME[/FOLDER]</c>: downloads a voice folder published on the Hugging Face hub (a
/// <c>voice.json</c> with its consent and embedding, as <c>voice export</c> writes them) into the cache and installs it,
/// as <c>idrak pull</c> takes a model.
/// </summary>
internal sealed class VoicePullCommand : Command
{
    public override string Name => "voice pull";

    public override string Summary => "Download a voice folder published on the Hugging Face hub and install it";

    public override string Usage => """
        OWNER/NAME[/FOLDER] [options]

        The repository (or its folder) holds one voice folder: voice.json with the consent record, the embedding and
        any kept clips, as idrak voice export writes them. Its files are downloaded into the cache (resumed when
        interrupted), then installed as idrak voice import installs a folder; a voice without a consent record is refused.

        Options:
              --revision REV     a branch, tag or commit (default main)
              --token TOKEN      for gated and private repositories (default: the config's hf_token, HF_TOKEN or the saved login)
              --overwrite        replace an installed voice of the same name
          -f, --force            download again even when cached
        """ + "\n" + VoiceText.StoreHelp + "\n" + """

        Examples:
          idrak voice pull someone/voices/sara
          idrak voice pull someone/sara-voice --overwrite
        """;

    public override IReadOnlyCollection<string> ValueOptions { get; } = ["--voices", "--revision", "--token"];

    public override IReadOnlyCollection<string> Flags { get; } = ["--overwrite", "--force"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-f"] = "--force" };

    public override int Run(CommandContext context)
    {
        string target = context.Argument(0, "OWNER/NAME[/FOLDER] (a voice folder on the Hugging Face hub)");
        var (repo, folder) = PullTarget.SplitFolder(target) ?? (target.Trim('/'), "");
        if (!Idrak.Models.ModelSource.IsModelId(repo))
        {
            throw new UsageException($"'{target}' is not a Hugging Face id (owner/name, or owner/name/folder); a voice zip or folder on disk installs with idrak voice import.");
        }

        if (context.Offline)
        {
            throw new UsageException("voice pull downloads; --offline forbids it (idrak voice import installs a folder on disk).");
        }

        var downloader = ModelCache.Downloader(context, context.Flag("--force"));
        string? token = Hub.Token(context);
        string revision = context.Option("--revision") ?? "main";
        string commit = Idrak.Data.HuggingFace.ResolveRevisionAsync(repo, "models", revision, token, downloader).GetAwaiter().GetResult();
        var files = Idrak.Data.HuggingFace.ListFilesAsync(repo, "models", commit, token, downloader).GetAwaiter().GetResult();
        string prefix = folder.Length == 0 ? "" : folder.TrimEnd('/') + "/";
        var manifests = files.Where(f => f.Path.StartsWith(prefix, StringComparison.Ordinal) && Path.GetFileName(f.Path) == FolderVoiceStore.ManifestFile).ToList();
        if (manifests.Count != 1)
        {
            throw new UsageException(manifests.Count == 0
                ? $"{target} holds no voice ({FolderVoiceStore.ManifestFile} not found{(prefix.Length == 0 ? "" : $" under {folder}")})."
                : $"{target} holds {manifests.Count} voices; name one: {string.Join(", ", manifests.Select(m => $"{repo}/{Path.GetDirectoryName(m.Path)!.Replace('\\', '/')}"))}.");
        }

        string voiceFolder = Path.GetDirectoryName(manifests[0].Path)!.Replace('\\', '/');
        string voicePrefix = voiceFolder.Length == 0 ? "" : voiceFolder + "/";
        var wanted = files.Where(f => f.Path.StartsWith(voicePrefix, StringComparison.Ordinal)).ToList();
        context.Write($"{repo}{(voiceFolder.Length == 0 ? "" : "/" + voiceFolder)} at {commit[..Math.Min(12, commit.Length)]}: {wanted.Count} files ({Units.Bytes(wanted.Sum(f => f.Size))})");
        string? local = null;
        foreach (var file in wanted)
        {
            string path = Idrak.Data.HuggingFace.DownloadFileAsync(repo, file.Path, "models", commit, token, downloader).GetAwaiter().GetResult();
            if (file == manifests[0])
            {
                local = Path.GetDirectoryName(path);
            }
        }

        var store = AudioFiles.VoiceStore(context);
        try
        {
            var voice = store.Import(local!, context.Flag("--overwrite"));
            context.Write($"installed '{voice.Name}' in {store.Location}");
            foreach (string line in VoiceText.Lines(voice, VoiceText.WarnAgreement))
            {
                context.Write(line);
            }

            var json = VoiceText.Json(voice, VoiceText.WarnAgreement);
            json["source"] = $"{repo}{(voiceFolder.Length == 0 ? "" : "/" + voiceFolder)}@{commit}";
            context.WriteJson(json);
            return ExitCodes.Ok;
        }
        catch (Exception e) when (e is IOException or InvalidDataException or ArgumentException)
        {
            context.Error(VoiceText.Options(e.Message).Replace("set overwrite", "give --overwrite", StringComparison.Ordinal));
            return ExitCodes.Failed;
        }
    }
}
