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

    /// <summary>Whether requests need a key (an empty key is refused when the settings are read, never taken as one).</summary>
    public bool NeedsKey => ApiKey is { Length: > 0 } || KeyHashes.Count > 0;

    /// <summary>
    /// Browser origins (normalized, <c>scheme://host[:port]</c>, or <c>*</c> for any) whose pages may call the control
    /// endpoints besides the server's own page and the origins <see cref="Cors"/> names (--control-origins,
    /// <see cref="ServeOptions.ControlOriginsVariable"/>; default none).
    /// </summary>
    public IReadOnlyList<string> ControlOrigins { get; init; } = [];

    /// <summary>
    /// Whether listening where other machines reach (<see cref="PublicBind"/>) with no key refuses to start, instead of the
    /// warning (--require-key, <see cref="ServeOptions.RequireKeyVariable"/>).
    /// </summary>
    public bool RequireKey { get; init; }

    /// <summary>
    /// Whether a JSON route reads its body whatever its Content-Type says (--any-content-type,
    /// <see cref="ServeOptions.AnyContentTypeVariable"/>); off, another declared type is a 415, so a cross-site form cannot post to it.
    /// </summary>
    public bool AnyContentType { get; init; }

    /// <summary>Whether <see cref="Host"/> is reachable from other machines (any address but a loopback one or localhost).</summary>
    public bool PublicBind => ServeOptions.IsPublicHost(Host);

    /// <summary>The largest request body in bytes (--max-request-mb); a larger one is a 413.</summary>
    public long MaxRequestBytes { get; init; } = ServeOptions.DefaultMaxRequestMegabytes * 1024L * 1024;

    /// <summary>The most images one request may hold (--max-images).</summary>
    public int MaxImages { get; init; } = ServeOptions.DefaultMaxImages;

    /// <summary>Whether image_url parts may give http(s) addresses the server downloads (--allow-image-urls).</summary>
    public bool AllowImageUrls { get; init; }

    /// <summary>
    /// Whether those downloads may reach loopback, link-local, private, multicast and reserved addresses
    /// (--allow-private-image-urls, <see cref="ServeOptions.PrivateImageUrlsVariable"/>; off: refused, a 400 naming the URL).
    /// </summary>
    public bool AllowPrivateImageUrls { get; init; }

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
    public static readonly string[] ValueOptions = ["--host", "--port", "--api-key", "--cors", "--control-origins", "--max-concurrency", "--keep-alive", "--log-requests",
        "--max-request-mb", "--max-images", ModelChoices.VisionOption, ModelChoices.ImageTransformOption, Seq2SeqRun.BeamsOption, Seq2SeqRun.LengthPenaltyOption];

    /// <summary>The flags of serve and ui.</summary>
    public static readonly string[] Flags = ["--metrics", "--log-content", "--grayscale", "--allow-image-urls", "--allow-private-image-urls", "--require-key", "--any-content-type"];

    /// <summary>The environment variable giving the key when <c>--api-key</c> is not given.</summary>
    public const string ApiKeyVariable = "IDRAK_API_KEY";

    /// <summary>The environment variable turning <c>--require-key</c> on (1, true, on or yes) or off (0, false, off or no).</summary>
    public const string RequireKeyVariable = "IDRAK_SERVE_REQUIRE_KEY";

    /// <summary>The environment variable giving the control origins (comma-separated) when <c>--control-origins</c> is not given.</summary>
    public const string ControlOriginsVariable = "IDRAK_SERVE_CONTROL_ORIGINS";

    /// <summary>The environment variable turning <c>--any-content-type</c> on (1, true, on or yes) or off (0, false, off or no).</summary>
    public const string AnyContentTypeVariable = "IDRAK_SERVE_ANY_CONTENT_TYPE";

    /// <summary>The environment variable turning <c>--allow-private-image-urls</c> on (1, true, on or yes) or off (0, false, off or no).</summary>
    public const string PrivateImageUrlsVariable = "IDRAK_SERVE_PRIVATE_IMAGE_URLS";

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

        // An empty key would make the server look protected while an empty bearer matches it: refused, never taken.
        string? key = context.Option("--api-key") ?? Environment.GetEnvironmentVariable(ApiKeyVariable);
        if (key is not null && string.IsNullOrWhiteSpace(key))
        {
            string source = context.Option("--api-key") is not null ? "--api-key" : ApiKeyVariable;
            throw new UsageException($"{source} is empty: an empty key would let every request in. Give a key with --api-key KEY or {ApiKeyVariable}, "
                + $"or leave both out ({ApiKeyVariable} unset) to serve without one.");
        }

        var settings = new ServeSettings(host, port, key,
            [.. context.Options("--cors").SelectMany(o => o.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))],
            concurrency, keep, keepText)
        {
            ControlOrigins = ControlOrigins(context),
            RequireKey = Switch(context, "--require-key", RequireKeyVariable),
            AnyContentType = Switch(context, "--any-content-type", AnyContentTypeVariable),
            KeyHashes = [.. ServerKeys.Hashes(context)],
            Metrics = context.Flag("--metrics"),
            RequestLog = context.Option("--log-requests"),
            MaxRequestBytes = requestMb * 1024L * 1024,
            MaxImages = images,
            AllowImageUrls = context.Flag("--allow-image-urls"),
            AllowPrivateImageUrls = Switch(context, "--allow-private-image-urls", PrivateImageUrlsVariable) && (context.Flag("--allow-image-urls")
                ? true : context.Flag("--allow-private-image-urls") ? throw new UsageException("--allow-private-image-urls lets image URL downloads reach private addresses; "
                    + "downloads are off without --allow-image-urls.") : false),
            Beams = beams,
            LengthPenalty = penalty,
            LogContent = context.Flag("--log-content") && (context.Option("--log-requests") is not null
                ? true : throw new UsageException("--log-content adds the request bodies to the request log; give the log with --log-requests FILE.")),
        };

        // MCP listens on its own address (--mcp-address, which refuses a remote one by itself); --host is the HTTP server's.
        if (settings.RequireKey && settings.PublicBind && !settings.NeedsKey && !context.Flag("--mcp"))
        {
            throw new UsageException($"--require-key ({RequireKeyVariable}): --host {host} is reachable from other machines and no API key is set. Give one with "
                + $"--api-key KEY or {ApiKeyVariable} (or add one with 'idrak server keys add'), or listen on 127.0.0.1.");
        }

        return settings;
    }

    /// <summary>
    /// Whether <paramref name="host"/> is reachable from other machines: every address but a loopback one (127.0.0.0/8,
    /// ::1) and localhost; the wildcards (0.0.0.0, *, +, ::, [::]) and host names included.
    /// </summary>
    public static bool IsPublicHost(string host)
    {
        string name = host.Trim();
        if (name.Equals("localhost", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        if (name.Length > 1 && name[0] == '[' && name[^1] == ']')
        {
            name = name[1..^1];
        }

        return !(IPAddress.TryParse(name, out var address) && IPAddress.IsLoopback(address));
    }

    /// <summary>The warning printed first when the server listens where other machines reach with no key.</summary>
    public static string PublicBindWarning(string host) =>
        $"idrak serve: warning: listening on {host}, which other machines reach, with no API key: anyone who reaches this port can use, load, unload "
        + $"and stop the models. Give a key with --api-key KEY or {ApiKeyVariable} (or 'idrak server keys add'), or make this start fail with "
        + $"--require-key ({RequireKeyVariable}=1).";

    /// <summary>
    /// An origin as browsers send it in the <c>Origin</c> header (<c>scheme://host[:port]</c>, the host in lower case, a
    /// default port left out), or null when <paramref name="text"/> is not an http(s) origin (such as "null").
    /// </summary>
    public static string? NormalizeOrigin(string text) =>
        Uri.TryCreate(text.Trim(), UriKind.Absolute, out var uri) && (uri.Scheme == Uri.UriSchemeHttp || uri.Scheme == Uri.UriSchemeHttps)
            ? uri.GetLeftPart(UriPartial.Authority) : null;

    // --control-origins (repeatable or comma-separated), else IDRAK_SERVE_CONTROL_ORIGINS: each an origin or *.
    private static string[] ControlOrigins(CommandContext context)
    {
        IReadOnlyList<string> values = context.Options("--control-origins");
        string source = "--control-origins";
        if (values.Count == 0 && Environment.GetEnvironmentVariable(ControlOriginsVariable) is { Length: > 0 } text)
        {
            values = [text];
            source = ControlOriginsVariable;
        }

        return [.. values.SelectMany(v => v.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)).Select(v => v == "*" ? v
            : NormalizeOrigin(v) ?? throw new UsageException($"{source}: '{v}' is not an origin (http://HOST[:PORT] or https://HOST[:PORT], or * for any)."))];
    }

    // A flag, or its variable: 1, true, on or yes turn it on; 0, false, off or no (or unset) leave it off.
    private static bool Switch(CommandContext context, string flag, string variable) =>
        context.Flag(flag) || (Environment.GetEnvironmentVariable(variable)?.Trim().ToLowerInvariant() switch
        {
            null or "" or "0" or "false" or "off" or "no" => false,
            "1" or "true" or "on" or "yes" => true,
            var other => throw new UsageException($"{variable} takes 1, true, on or yes (or 0, false, off or no), not '{other}'."),
        });

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
