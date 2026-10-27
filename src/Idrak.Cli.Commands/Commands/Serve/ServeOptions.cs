// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Net;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Datasets;
using Idrak.Generation;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Text;
using Idrak.Text.Generation;
using Idrak.Text.Inference;
using Seq2SeqRun = Idrak.Cli.Commands.Run.Seq2SeqRun;

namespace Idrak.Cli.Commands.Serve;

/// <summary>
/// One model of a server: the name clients use, what was asked for, and what is read at startup (the template, and
/// whether it reads images: a vision-language model whose vision family is registered and whose template renders images).
/// </summary>
internal sealed record ServedModel(string Name, string Source, ModelChoices.ModelChoice Choice, string Path, string Folder, IChatTemplate? Template, JsonObject? Config)
{
    /// <summary>Whether its chat takes images.</summary>
    public bool ReadsImages => Config is not null && VisionFamilies.HasVision(Config) && Architecture(Config) is { } architecture
        && VisionFamilies.Find(architecture) is not null && Template?.PartKinds.Contains(ChatParts.Image) == true;

    /// <summary>Whether its chat takes sounds: an audio-language model whose audio family is registered and whose template renders sounds.</summary>
    public bool ReadsAudio => Config is not null && Idrak.Audio.Abstractions.AudioFamilies.HasAudio(Config) && Architecture(Config) is { } architecture
        && Idrak.Audio.Abstractions.AudioFamilies.Find(architecture) is not null && Template?.PartKinds.Contains(ChatParts.Audio) == true;

    /// <summary>The architecture name config.json gives (its first "architectures" entry), or null.</summary>
    public static string? Architecture(JsonObject config) => config["architectures"] is JsonArray { Count: > 0 } names ? (string?)names[0] : null;

    /// <summary>
    /// The architecture when it is an encoder-decoder a plug-in registered (<c>PretrainedSeq2Seq</c>), else null: such a
    /// model is served as the engine's "seq2seq" kind (prompts on /v1/completions and MCP generate), not as a chat model.
    /// </summary>
    public string? EncoderDecoder { get; init; }

    /// <summary>The engine kind it is served as: "chat", or "seq2seq" for an encoder-decoder.</summary>
    public string Kind => EncoderDecoder is null ? ChatKind : Seq2SeqKind;

    /// <summary>The kind of a model served as a chat model.</summary>
    public const string ChatKind = "chat";

    /// <summary>The kind of an encoder-decoder (the inference engine's name for it).</summary>
    public const string Seq2SeqKind = "seq2seq";
}

/// <summary>The server's settings from the command line.</summary>
internal sealed record ServeSettings(string Host, int Port, string? ApiKey, IReadOnlyList<string> Cors, int MaxConcurrency, TimeSpan? KeepAlive, string KeepAliveText)
{
    /// <summary>SHA-256 hashes (hex) of the keys added with <c>idrak server keys add</c>; any of them, or <see cref="ApiKey"/>, is accepted.</summary>
    public IReadOnlyList<string> KeyHashes { get; init; } = [];

    /// <summary>Whether GET /metrics answers (--metrics).</summary>
    public bool Metrics { get; init; }

    /// <summary>The request log file (--log-requests), or null.</summary>
    public string? RequestLog { get; init; }

    /// <summary>Whether the request log holds the request bodies (--log-content).</summary>
    public bool LogContent { get; init; }

    /// <summary>Whether requests need a key.</summary>
    public bool NeedsKey => ApiKey is not null || KeyHashes.Count > 0;

    /// <summary>The largest request body in bytes (--max-request-mb); a larger one is a 413.</summary>
    public long MaxRequestBytes { get; init; } = ServeOptions.DefaultMaxRequestMegabytes * 1024L * 1024;

    /// <summary>The most images one request may hold (--max-images).</summary>
    public int MaxImages { get; init; } = ServeOptions.DefaultMaxImages;

    /// <summary>Whether image_url parts may give http(s) addresses the server downloads (--allow-image-urls).</summary>
    public bool AllowImageUrls { get; init; }

    /// <summary>An encoder-decoder's beams for every request that gives none (--beams; 1: the sampler).</summary>
    public int Beams { get; init; } = 1;

    /// <summary>The beams' length penalty for every request that gives none (--length-penalty).</summary>
    public float LengthPenalty { get; init; } = 1f;
}

/// <summary>
/// What <c>idrak serve</c> and <c>idrak ui</c> read before a server starts: their options, the port, the settings and the
/// models named (each found or downloaded, its template read). The server itself is the host's (<see cref="CommandServers"/>:
/// the idrak tool's ASP.NET Core one).
/// </summary>
internal static class ServeOptions
{
    /// <summary>
    /// The port when none is chosen: Idrak's own, so a server does not collide with another local model server
    /// (<c>-p 11434</c> serves clients that expect that port).
    /// </summary>
    public const int DefaultPort = 7317;

    /// <summary>The environment variable choosing the port when <c>--port</c> is not given.</summary>
    public const string PortVariable = "IDRAK_PORT";

    /// <summary>The config key choosing the port when neither <c>--port</c> nor <see cref="PortVariable"/> is given.</summary>
    public const string PortKey = "serve.port";

    /// <summary>The options of serve and ui (besides the model options).</summary>
    public static readonly string[] ValueOptions = ["--host", "--port", "--api-key", "--cors", "--max-concurrency", "--keep-alive", "--log-requests",
        "--max-request-mb", "--max-images", ModelChoices.VisionOption, ModelChoices.ImageTransformOption, Seq2SeqRun.BeamsOption, Seq2SeqRun.LengthPenaltyOption];

    /// <summary>The flags of serve and ui.</summary>
    public static readonly string[] Flags = ["--metrics", "--log-content", "--grayscale", "--allow-image-urls"];

    /// <summary>The largest request body when --max-request-mb is not given, in MB (a phone photo as a data URL fits).</summary>
    public const int DefaultMaxRequestMegabytes = 32;

    /// <summary>The most images in one request when --max-images is not given.</summary>
    public const int DefaultMaxImages = 8;

    /// <summary>How long the server waits for one http(s) image URL (--allow-image-urls).</summary>
    public static readonly TimeSpan ImageUrlTimeout = TimeSpan.FromSeconds(30);

    /// <summary>Their short forms.</summary>
    public static readonly Dictionary<string, string> ShortForms = new() { ["-p"] = "--port", ["-H"] = "--host" };

    /// <summary>
    /// The port a server listens on: <c>--port</c>, else <see cref="PortVariable"/>, else the config's
    /// <see cref="PortKey"/>, else <see cref="DefaultPort"/> (usage errors for values that are not ports).
    /// </summary>
    public static int Port(CommandContext context) => ExplicitPort(context) ?? ConfiguredPort(context) ?? DefaultPort;

    /// <summary>The port given with <c>--port</c> or <see cref="PortVariable"/>, or null.</summary>
    public static int? ExplicitPort(CommandContext context) =>
        context.Option("--port") is not null ? Checked(context.IntOption("--port", DefaultPort), "--port")
        : Environment.GetEnvironmentVariable(PortVariable) is { Length: > 0 } text ? Parsed(text, PortVariable)
        : null;

    /// <summary>The config's <see cref="PortKey"/> (a number, or a number as text), or null.</summary>
    public static int? ConfiguredPort(CommandContext context)
    {
        var parts = PortKey.Split('.');
        return context.Config.Object(parts[0])?[parts[1]] is { } node
            ? Parsed(node is JsonValue v && v.TryGetValue(out string? text) ? text : node.ToJsonString(), $"{PortKey} in {context.Config.Path}")
            : null;
    }

    private static int Parsed(string text, string source) =>
        int.TryParse(text.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out int port) ? Checked(port, source)
        : throw new UsageException($"{source} must be a port number (0 to 65535), not '{text}'.");

    private static int Checked(int port, string source) =>
        port is >= 0 and <= 65535 ? port : throw new UsageException($"{source} must be between 0 and 65535, not {port}.");

    /// <summary>The settings from the command's options (usage errors for bad values).</summary>
    public static ServeSettings ReadSettings(CommandContext context)
    {
        string host = context.Option("--host") ?? "127.0.0.1";
        int port = Port(context);

        int concurrency = context.IntOption("--max-concurrency", 0);
        if (concurrency < 0)
        {
            throw new UsageException("--max-concurrency must be 0 (no limit) or more.");
        }

        string keepText = context.Option("--keep-alive") ?? "5m";
        TimeSpan? keep;
        try
        {
            keep = KeepAlive.Parse(keepText);
        }
        catch (FormatException e)
        {
            throw new UsageException($"--keep-alive: {e.Message}");
        }

        int requestMb = context.IntOption("--max-request-mb", DefaultMaxRequestMegabytes);
        if (requestMb < 1)
        {
            throw new UsageException("--max-request-mb must be 1 or more.");
        }

        int images = context.IntOption("--max-images", DefaultMaxImages);
        if (images < 0)
        {
            throw new UsageException("--max-images must be 0 (no images) or more.");
        }

        var (beams, penalty) = Seq2SeqRun.Beams(context);

        return new ServeSettings(host, port, context.Option("--api-key") ?? Environment.GetEnvironmentVariable("IDRAK_API_KEY"),
            [.. context.Options("--cors").SelectMany(o => o.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))],
            concurrency, keep, keepText)
        {
            KeyHashes = [.. ServerKeys.Hashes(context)],
            Metrics = context.Flag("--metrics"),
            RequestLog = context.Option("--log-requests"),
            MaxRequestBytes = requestMb * 1024L * 1024,
            MaxImages = images,
            AllowImageUrls = context.Flag("--allow-image-urls"),
            Beams = beams,
            LengthPenalty = penalty,
            LogContent = context.Flag("--log-content") && (context.Option("--log-requests") is not null
                ? true : throw new UsageException("--log-content adds the request bodies to the request log; give the log with --log-requests FILE.")),
        };
    }

    /// <summary>
    /// The models named on the command line: NAME=MODEL serves MODEL as NAME; otherwise an alias keeps its name, a file
    /// or folder is named after it, and a Hugging Face id is used as is. Each is found (or downloaded) and its chat
    /// template read now; the weights load on the first request.
    /// </summary>
    public static List<ServedModel> ReadModels(CommandContext context, IReadOnlyList<string> arguments)
    {
        if (arguments.Count == 0)
        {
            throw new UsageException("Name at least one model to serve (a Hugging Face id, a folder, a GGUF file or an alias).");
        }

        var models = new List<ServedModel>();
        foreach (string argument in arguments)
        {
            string name, source = argument;
            int equals = argument.IndexOf('=');
            if (equals > 0 && !argument[..equals].Contains('/') && !argument[..equals].Contains('\\'))
            {
                (name, source) = (argument[..equals], argument[(equals + 1)..]);
            }
            else
            {
                name = context.Config.Object("aliases")?[argument] is not null ? argument
                    : File.Exists(argument) ? System.IO.Path.GetFileNameWithoutExtension(argument)
                    : Directory.Exists(argument) ? System.IO.Path.GetFileName(System.IO.Path.TrimEndingDirectorySeparator(System.IO.Path.GetFullPath(argument)))
                    : argument;
            }

            if (models.Any(m => m.Name == name))
            {
                throw new UsageException($"Two models are served as '{name}'; name one with NAME=MODEL.");
            }

            var choice = ModelChoices.Choose(context, source);
            if (choice.Kv is { } kv)
            {
                _ = KeyValueLayouts.Get(kv);                    // an unknown format fails now, not on the first request
            }

            string path = ModelChoices.Resolve(context, choice.Model);
            string folder = CheckpointFormats.For(path).Prepare(path);
            var tokenizer = PretrainedTokenizers.TryLoad(folder);
            var template = JinjaChatTemplate.Load(folder, tokenizer);
            string configFile = System.IO.Path.Combine(folder, "config.json");
            var config = File.Exists(configFile) ? JsonNode.Parse(File.ReadAllText(configFile)) as JsonObject : null;
            try
            {
                // A vision-language model is read by its own registered vision family (-P PLUGIN): none fails now, not on the first request.
                _ = config is not null && ServedModel.Architecture(config) is { } architecture ? VisionFamilies.For(architecture, config) : null;
            }
            catch (NotSupportedException e)
            {
                throw new UsageException($"{name}: {e.Message}");
            }

            // An encoder-decoder a plug-in registered: served as one (weight choices it does not take refused now, as run does).
            string? encoderDecoder = Seq2SeqRun.Architecture(folder);
            if (encoderDecoder is not null)
            {
                Seq2SeqRun.CheckChoice(choice, name, encoderDecoder);
                if (tokenizer is null)
                {
                    throw new UsageException($"{name}: {folder} has no tokenizer (tokenizer.json or a SentencePiece model), which an encoder-decoder ({encoderDecoder}) is served with.");
                }
            }

            models.Add(new ServedModel(name, source, choice, path, folder, template, config) { EncoderDecoder = encoderDecoder });
        }

        foreach (string option in new[] { Seq2SeqRun.BeamsOption, Seq2SeqRun.LengthPenaltyOption })
        {
            if (context.Option(option) is not null && models.All(m => m.EncoderDecoder is null))
            {
                throw new UsageException($"{option}: no encoder-decoder is served (it sets an encoder-decoder's beam search; {string.Join(", ", models.Select(m => m.Name))} answer chat).");
            }
        }

        return models;
    }
}

/// <summary>
/// A small file per running server (<c>CACHE/servers/PORT.json</c>: its URL, process and models), so the server
/// commands find it without --port.
/// </summary>
internal static class ServerState
{
    public static string Folder(CommandContext context) => Path.Combine(context.CacheFolder, "servers");

    public static string Write(CommandContext context, string url, int port, IEnumerable<string> models)
    {
        string file = Path.Combine(Folder(context), $"{port}.json");
        try
        {
            Directory.CreateDirectory(Folder(context));
            File.WriteAllText(file, new JsonObject
            {
                ["url"] = url,
                ["pid"] = Environment.ProcessId,
                ["started"] = DateTimeOffset.UtcNow.ToString("O", CultureInfo.InvariantCulture),
                ["models"] = new JsonArray([.. models.Select(m => (JsonNode)m)]),
            }.ToJsonString());
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            context.Detail($"could not write {file}: {e.Message}");
        }

        return file;
    }

    public static void Remove(string file)
    {
        try
        {
            File.Delete(file);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
        }
    }

    /// <summary>The URL of the most recently started server whose process still runs, or null.</summary>
    public static string? Find(CommandContext context)
    {
        if (!Directory.Exists(Folder(context)))
        {
            return null;
        }

        var running = new List<(DateTimeOffset Started, string Url)>();
        foreach (string file in Directory.EnumerateFiles(Folder(context), "*.json"))
        {
            try
            {
                var state = JsonNode.Parse(File.ReadAllText(file))!;
                if (Alive((int)state["pid"]!))
                {
                    running.Add((DateTimeOffset.Parse((string)state["started"]!, CultureInfo.InvariantCulture), (string)state["url"]!));
                }
            }
            catch (Exception e) when (e is JsonException or IOException or FormatException or InvalidOperationException or NullReferenceException)
            {
                // a file being written or left over: skip it
            }
        }

        return running.Count == 0 ? null : running.MaxBy(r => r.Started).Url;
    }

    private static bool Alive(int pid)
    {
        try
        {
            using var process = Process.GetProcessById(pid);
            return !process.HasExited;
        }
        catch (ArgumentException)
        {
            return false;
        }
    }
}
