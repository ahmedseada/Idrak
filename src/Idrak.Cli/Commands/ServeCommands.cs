// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Cli.Commands.Serve;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands;

/// <summary>The serve, server, api, ui and mcp serve commands (plans/idrak-cli.md, "Serve").</summary>
internal static class ServeCommands
{
    public static IReadOnlyList<Command> All { get; } =
    [
        new ServeCommand(),
        new UiCommand(),
        new ServerPsCommand(),
        new ServerStopCommand(),
        new ServerLoadCommand(),
        new ServerUnloadCommand(),
        new ServerKeysAddCommand(),
        new ServerKeysListCommand(),
        new ServerKeysRemoveCommand(),
        new ApiCommand(),
        new PingCommand(),
        new McpServeCommand(),
    ];
}

/// <summary><c>idrak serve MODEL [MODEL...]</c>: the chat API and the OpenAI-style API on one port.</summary>
internal sealed class ServeCommand : Command
{
    public override string Name => "serve";

    public override IReadOnlyCollection<string> Aliases => ["s"];

    public override string Summary => "Serve models over the chat API and the OpenAI-style API on one port";

    /// <summary>The options of the server (serve and ui).</summary>
    public const string OptionsHelp = """
          -H, --host NAME            address to listen on (default 127.0.0.1; 0.0.0.0 for every interface)
          -p, --port N               port (default IDRAK_PORT, else the config's serve.port, else 7317; 0 picks a
                                     free one; -p 11434 for clients that expect that port)
              --api-key KEY          require "Authorization: Bearer KEY" (default IDRAK_API_KEY when set); keys
                                     added with 'idrak server keys add' are accepted too
              --metrics              GET /metrics: requests, tokens, tokens per second, latency, queue and memory
                                     in the plain text format monitoring tools scrape
              --log-requests FILE    one JSON line per model request (time, model, status, tokens, speed)
              --log-content          add the request bodies (prompts) to that log
              --cors ORIGIN          allow browser pages from ORIGIN (repeatable or comma-separated; * for any)
              --max-concurrency N    model requests answered at once; the rest wait (default 0: no limit)
              --keep-alive DURATION  unload a model after this long without use: 30s, 5m, 1h, 0 (after each
                                     request) or -1 (never) (default 5m)
              --max-request-mb N     the largest request body, images included (default 32); larger is a 413
              --max-images N         the most images in one request (default 8)
              --grayscale            read a vision model's images grey (some OCR fine-tunes ask for it; an alias
                                     can keep it); a request can also ask with "grayscale": true
              --image-transform P    image transforms run on every image first, in order, as Pillow does them
                                     (such as grayscale,max_width=1024,contrast=1.5; an alias can keep them); a
                                     request gives its own as "image_transforms" (they replace these; "none": none)
              --vision-option K=V    an option of a vision model's family for every image (repeatable; the family
                                     names the keys it takes; with the Gemma 3 plug-in, do_pan_and_scan=true adds
                                     crops of a tall or wide page); a request gives its own as "vision_options":
                                     {"KEY": VALUE}, over these
              --allow-image-urls     let image_url parts name http(s) addresses the server downloads (off: data
                                     URLs and uploads only; each download is held to --max-request-mb and 30 s)
          -w, --weights FORMAT       int8, int4, bf16 or a registered packed format
          -k, --kv FORMAT            the KV cache format (float32, int8, bfloat16 or a registered one)
              --context N            the longest context to allocate
              --adapter DIR          merge a LoRA or DoRA adapter into the weights (one tuned on images: its
                                     trained projector and image preparation too)
        """;

    public override string Usage => $"""
        MODEL [MODEL...] [options]

        Serves each MODEL (a Hugging Face id, a folder, a GGUF file, a .ikm package or an alias; NAME=MODEL serves it
        as NAME) on one port. Each model loads on its first request and unloads after --keep-alive without use.

        Options:
        {OptionsHelp}

        Endpoints:
          chat API           POST /api/chat (streamed lines, tool calls, think, keep_alive), GET /api/tags,
                             GET /api/ps, GET /api/version
          OpenAI-style API   GET /v1/models, POST /v1/chat/completions (stream, tools, image_url parts as data
                             URLs), POST /v1/completions; POST /v1/embeddings answers 404 (no embedding model
                             can be served yet)
          image upload       POST /v1/chat/upload, a multipart/form-data form: image (one or more files),
                             prompt, and optionally system, stream (true: server-sent events), max_tokens,
                             temperature, grayscale, vision_options (a JSON object), image_transforms,
                             model; answers as
                             /v1/chat/completions
          web chat           GET /ui
          control            GET /idrak/ps, POST /idrak/load, /idrak/unload, /idrak/stop, GET /idrak/status
          metrics            GET /metrics (with --metrics)
        A vision-language model (Gemma 3, such as an OCR fine-tune) takes images on /v1/chat/completions,
        /v1/chat/upload and /api/chat (PNG, JPEG, BMP, PPM/PGM; turned upright by their EXIF orientation); a
        text-only model answers an image with 400. Tool calls are returned to the client, which runs the tools. Point clients at http://127.0.0.1:7317 (the
        chat API's base address, or http://127.0.0.1:7317/v1 for the OpenAI-style API), or serve on the port they
        expect with -p.

        MCP:
              --mcp                  serve over the Model Context Protocol on standard input/output instead of
                                     HTTP: the models as the tools generate and chat (each takes a "model" argument
                                     when several are served), until the client disconnects; status lines go to the
                                     error output
              --tools FILE.dll       with --mcp, also serve the [Tool] methods of an assembly (repeatable)

        Examples:
          idrak serve Qwen/Qwen3-0.6B
          idrak serve qwen -p 11434                      # for clients that expect the chat API on port 11434
          idrak s qwen phi -p 8080 --api-key $KEY        # two models on one port
          idrak serve tiny=./tiny.gguf -d vulkan:0 -k int8 --keep-alive 30m --cors http://localhost:3000
          idrak serve ocr=bakrianoo/arabic-legal-documents-ocr-1.0 -d cuda:0 -w bf16 --grayscale
          idrak serve -P Idrak.Gemma3Vision.dll ocr --image-transform "grayscale,max_width=1024,contrast=1.5"
          idrak serve qwen --mcp --tools ./MyTools.dll  # an MCP client's configuration starts it this way
        """;

    public override IReadOnlyCollection<string> ValueOptions => [.. ServeHost.ValueOptions, .. ModelChoices.ValueOptions, "--tools"];

    public override IReadOnlyCollection<string> Flags => [.. ServeHost.Flags, "--mcp"];

    public override IReadOnlyDictionary<string, string> ShortForms => ServeHost.ShortForms.Concat(ModelChoices.ShortForms).ToDictionary();

    public override int Run(CommandContext context)
    {
        var settings = ServeHost.ReadSettings(context);
        if (!context.Flag("--mcp"))
        {
            return context.Options("--tools").Count > 0 ? throw new UsageException("--tools serves tools over MCP; add --mcp.")
                : new ServeHost(context, settings, ServeHost.ReadModels(context, context.Positional)).Run();
        }

        var tools = Shared.ToolAssemblies.Load(context.Options("--tools"));
        return new ServeHost(context, settings, ServeHost.ReadModels(context, context.Positional)).RunMcp(tools);
    }
}

/// <summary><c>idrak ui MODEL</c>: the server with its web chat page opened in the browser.</summary>
internal sealed class UiCommand : Command
{
    public override string Name => "ui";

    public override string Summary => "Serve models and open a small web chat page in the browser";

    public override string Usage => $"""
        MODEL [MODEL...] [options]

        Starts the same server as 'idrak serve' (every serve option applies) and opens its chat page (/ui) in the
        browser; the page streams answers from the OpenAI-style API.

        Options:
              --no-browser           print the page's address instead of opening it
        {ServeCommand.OptionsHelp}

        Examples:
          idrak ui Qwen/Qwen3-0.6B
          idrak ui qwen -p 8080 --no-browser
        """;

    public override IReadOnlyCollection<string> ValueOptions => [.. ServeHost.ValueOptions, .. ModelChoices.ValueOptions];

    public override IReadOnlyCollection<string> Flags => [.. ServeHost.Flags, "--no-browser"];

    public override IReadOnlyDictionary<string, string> ShortForms => ServeHost.ShortForms.Concat(ModelChoices.ShortForms).ToDictionary();

    public override int Run(CommandContext context)
    {
        var settings = ServeHost.ReadSettings(context);
        var models = ServeHost.ReadModels(context, context.Positional);
        return new ServeHost(context, settings, models).Run(url =>
        {
            string page = url + "/ui";
            if (context.Flag("--no-browser") || !Open(page))
            {
                context.Write($"Open {page} in a browser.");
            }
        });
    }

    private static bool Open(string url)
    {
        try
        {
            using var _ = Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
            return true;
        }
        catch (Exception e) when (e is System.ComponentModel.Win32Exception or InvalidOperationException or PlatformNotSupportedException)
        {
            return false;
        }
    }
}
