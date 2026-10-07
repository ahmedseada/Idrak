// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.IO.Pipelines;
using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Server;
using Idrak.AspNetCore;
using Idrak.Cli;
using Idrak.Cli.Commands.Serve;
using Idrak.Generation;
using Idrak.Inference;
using Idrak.Mcp;

// The tool contracts of Idrak.Abstraction (IToolRegistry, IToolChatModel, ChatTools.WithTools) and the bridges built
// on them alone: server-side tool execution in the ASP.NET Core chat endpoint, the engine's models as MCP tools
// (IModelCatalog) in process, and idrak serve --mcp over pipes.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] ToolContractGroup =
    [
        ("tool contracts: ChatTools.WithTools runs tools on the server (rounds, the round limit, streaming, the request kept)", d => { if (d == Device.Cpu) WithToolsRounds(); }),
        ("tool contracts: MapChatApi with server-side tools (the model's own through IToolChatModel, or given in the options; streamed and not)", d => { if (d == Device.Cpu) ChatApiServerTools(); }),
        ("mcp: the engine's models as MCP tools (generate, chat, embed) listed and called in process over the model catalog", ModelToolsOverMcp),
        ("cli serve: serve --mcp serves the models (generate, chat) and --tools assemblies over pipes", d => { if (d == Device.Cpu) CliServeMcp(); }),
    ];

    private static IToolRegistry WeatherTools(List<string> asked) => ToolRegistry.Create()
        .Add("weather", "The weather in a city.", new JsonObject
        {
            ["type"] = "object",
            ["properties"] = new JsonObject { ["city"] = new JsonObject { ["type"] = "string" } },
            ["required"] = new JsonArray("city"),
        }, (args, _) =>
        {
            lock (asked)
            {
                asked.Add((string)args["city"]!);
            }

            return Task.FromResult("sunny");
        })
        .Build();

    private static void WithToolsRounds()
    {
        var asked = new List<string>();
        var call = FakeChatModel.ToolCall("weather", new JsonObject { ["city"] = "Cairo" });
        var fake = FakeChatModel.Script(call, FakeChatModel.Answer("It is sunny.", "checked"));
        var request = new ChatRequest([new ChatMessage("user", "weather?")], [new ToolDefinition("client_tool")]);
        var chunks = fake.WithTools(WeatherTools(asked), maxToolRounds: 2).StreamAsync(request).ToBlockingEnumerable().ToList();
        var final = chunks[^1];
        Check(final.Done && final.Message!.Content == "It is sunny." && final.Message.ToolCalls is null && final.Delta.ToolCalls.Count == 0, "the final answer, without tool calls");
        Check(chunks.Take(chunks.Count - 1).All(c => !c.Done && c.Delta.ToolCalls.Count == 0) && string.Concat(chunks.Select(c => c.Delta.Content)) == "It is sunny.",
            "the stream carries content only");
        Check(asked.SequenceEqual(["Cairo"]) && fake.Requests.Count == 2 && fake.Requests.All(r => r.Tools!.Single().Name == "weather"), "the server's tools reach the model, the client's do not");
        Check(fake.Requests[1].Messages is [{ Role: "user" }, { Role: "assistant" }, { Role: "tool", Content: "sunny", ToolName: "weather" }], "the tool result goes back to the model");
        Check(request.Messages.Count == 1, "the request's messages are not changed");

        var limited = FakeChatModel.Script(call, call);
        var last = limited.WithTools(WeatherTools(asked), maxToolRounds: 1).ChatAsync(request).GetAwaiter().GetResult();
        Check(last.Message!.ToolCalls!.Single().Name == "weather" && last.Delta.ToolCalls.Count == 1 && limited.Requests.Count == 2,
            "when the rounds run out, the last reply keeps its tool calls");
        bool threw = false;
        try { limited.WithTools(WeatherTools(asked), -1); } catch (ArgumentOutOfRangeException) { threw = true; }
        Check(threw, "negative rounds are rejected");
    }

    // A chat kind of one's own that carries tools: the endpoint finds them through IToolChatModel.
    private sealed class ScriptedToolKind(FakeChatModel fake, IToolRegistry? tools) : EngineModel<FakeChatModel>("scripted", new EngineHosting { SharedCopies = true }), IToolChatModel
    {
        public IToolRegistry? Tools => tools;

        public override FakeChatModel LoadCopy() => fake;

        public override void UnloadCopy(FakeChatModel copy)
        {
        }

        public override ModelDescription Describe(FakeChatModel copy) => new(Name, Kind, 0, Device.Cpu, null);

        public async IAsyncEnumerable<ChatChunk> StreamAsync(ChatRequest request, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            using var lease = await Host.AcquireAsync(cancellationToken);
            await foreach (var chunk in lease.Copy.StreamAsync(request, lease.Token))
            {
                yield return chunk;
            }

            lease.Completed(1, 1, TimeSpan.Zero);
        }
    }

    private static void ChatApiServerTools()
    {
        var asked = new List<string>();
        var call = FakeChatModel.ToolCall("weather", new JsonObject { ["city"] = "Cairo" });
        var own = FakeChatModel.Script(call, FakeChatModel.Answer("It is sunny."), call, FakeChatModel.Answer("Still sunny."));
        var given = FakeChatModel.Script(call, call);
        var bare = FakeChatModel.Script();
        var builder = WebApplication.CreateSlimBuilder();
        builder.Logging.ClearProviders();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        builder.Services.AddIdrak()
            .Add("own", _ => new ScriptedToolKind(own, WeatherTools(asked)))
            .Add("given", _ => new ScriptedToolKind(given, null))
            .Add("bare", _ => new ScriptedToolKind(bare, null));
        var app = builder.Build();
        app.MapChatApi("/own", "own", o => o.Tools(ToolExecution.Server, maxRounds: 3));
        app.MapChatApi("/given", "given", o => o.Tools(ToolExecution.Server, maxRounds: 1, tools: WeatherTools(asked)));
        app.MapChatApi("/bare", "bare", o => o.Tools(ToolExecution.Server, maxRounds: 3));
        app.StartAsync().GetAwaiter().GetResult();
        try
        {
            using var http = new HttpClient { BaseAddress = new Uri(app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First()) };
            var ask = new { model = "m", messages = new[] { new { role = "user", content = "weather?" } }, stream = false };
            var reply = JsonNode.Parse(http.PostAsJsonAsync("/own/chat", ask).Result.Content.ReadAsStringAsync().Result)!;
            Check((string?)reply["message"]!["content"] == "It is sunny." && reply["message"]!["tool_calls"] is null && (bool)reply["done"]!, $"the model's own tools ran: {reply}");
            Check(asked.SequenceEqual(["Cairo"]) && own.Requests[1].Messages[^1] is { Role: "tool", Content: "sunny" }, "the tool's result reached the model");

            string lines = http.PostAsJsonAsync("/own/chat", ask with { stream = true }).Result.Content.ReadAsStringAsync().Result;
            var parsed = lines.Split('\n', StringSplitOptions.RemoveEmptyEntries).Select(l => JsonNode.Parse(l)!).ToList();
            Check((bool)parsed[^1]["done"]! && parsed.All(l => l["message"]!["tool_calls"] is null)
                && string.Concat(parsed.Select(l => (string?)l["message"]!["content"])) == "Still sunny.", $"streamed: {lines}");

            var limited = JsonNode.Parse(http.PostAsJsonAsync("/given/chat", ask).Result.Content.ReadAsStringAsync().Result)!;
            Check((string?)limited["message"]!["tool_calls"]![0]!["function"]!["name"] == "weather" && asked.Count == 3,
                $"tools given in the options ran, and the last call came back at the round limit: {limited}");

            var none = http.PostAsJsonAsync("/bare/chat", ask).Result;
            Check((int)none.StatusCode == 500 && none.Content.ReadAsStringAsync().Result.Contains("has no tools"), "server-side execution without tools is an error");
        }
        finally
        {
            app.StopAsync().GetAwaiter().GetResult();
            ((IAsyncDisposable)app).DisposeAsync().AsTask().GetAwaiter().GetResult();
        }
    }

    // An embedding kind of one's own: the MCP bridge offers "embed" for it.
    private sealed class EmbedderKind() : EngineModel<FixedEmbedder>("embedder", new EngineHosting { SharedCopies = true }), IEmbedder
    {
        public override FixedEmbedder LoadCopy() => new();

        public override void UnloadCopy(FixedEmbedder copy)
        {
        }

        public override ModelDescription Describe(FixedEmbedder copy) => new(Name, Kind, 0, Device.Cpu, null);

        public async ValueTask<float[][]> EmbedAsync(IReadOnlyList<string> texts, CancellationToken cancellationToken = default)
        {
            using var lease = await Host.AcquireAsync(cancellationToken);
            lease.Completed(texts.Count, texts.Count, TimeSpan.Zero);
            return await lease.Copy.EmbedAsync(texts, lease.Token);
        }
    }

    private static void ModelToolsOverMcp(Device device) => ModelToolsOverMcpAsync(device).GetAwaiter().GetResult();

    private static async Task ModelToolsOverMcpAsync(Device device)
    {
        await using var engine = await InferenceEngine.Create()
            .ChatModel("chat", () => { var (m, t) = TinyLanguageModel(device); return new TextGenerator(m, t, 32); })
            .TextModel("text", () => { var (m, t) = TinyLanguageModel(device); return new TextGenerator(m, t, 32); })
            .Add("words", new EmbedderKind())
            .BuildAsync();
        IModelCatalog catalog = engine;
        Check(catalog.KindOf("words") == "embedder" && catalog.KindOf("chat") == "chat" && catalog.TryGetModel<IEmbedder>("words", out _) && !catalog.TryGetModel<IEmbedder>("chat", out _),
            "the engine as a model catalog");

        var toServer = new Pipe();
        var toClient = new Pipe();
        var options = new McpServerOptions { ServerInfo = new Implementation { Name = "models", Version = "1.0" }, ToolCollection = [] };
        foreach (var tool in McpTools.ServerTools(catalog))
        {
            options.ToolCollection.Add(tool);
        }

        await using var server = McpServer.Create(new StreamServerTransport(toServer.Reader.AsStream(), toClient.Writer.AsStream(), "models"), options);
        using var stop = new CancellationTokenSource();
        var running = server.RunAsync(stop.Token);
        await using (var source = await McpTools.ConnectAsync(new StreamClientTransport(toServer.Writer.AsStream(), toClient.Reader.AsStream())))
        {
            var tools = (await source.ListToolsAsync()).ToDictionary(t => t.Definition.Name);
            Check(tools.Keys.Order().SequenceEqual([McpTools.ChatTool, McpTools.EmbedTool, McpTools.GenerateTool]), $"tools {string.Join(",", tools.Keys)}");
            var generateSchema = tools["generate"].Definition.Parameters!;
            Check(generateSchema["properties"]!["model"]!["enum"]!.AsArray().Select(n => (string)n!).SequenceEqual(["chat", "text"])
                && generateSchema["required"]!.AsArray().Select(n => (string)n!).SequenceEqual(["model", "prompt"]), $"generate: both models, model required: {generateSchema}");
            Check(tools["chat"].Definition.Parameters!["required"]!.AsArray().Select(n => (string)n!).SequenceEqual(["messages"]), "chat: one model, model optional");

            var options3 = new GenerationOptions { NumPredict = 4, Seed = 3 };
            string generated = await source.CallAsync("generate", new JsonObject { ["model"] = "text", ["prompt"] = "abc", ["max_tokens"] = 4, ["seed"] = 3 });
            var (direct, _, stats) = await engine.Model<ITextModel>("text").GenerateAsync("abc", options3);
            Check(generated == direct && stats.GeneratedTokens == 4, $"generate over MCP = the engine's answer: '{generated}' vs '{direct}'");

            string chatted = await source.CallAsync("chat", new JsonObject
            {
                ["messages"] = new JsonArray(new JsonObject { ["role"] = "user", ["content"] = "hi" }),
                ["max_tokens"] = 4,
                ["seed"] = 3,
            });
            var answer = await engine.Model<IChatModel>("chat").ChatAsync(new ChatRequest([new ChatMessage("user", "hi")], Options: options3));
            Check(chatted == answer.Message!.Content, $"chat over MCP = the engine's answer: '{chatted}' vs '{answer.Message.Content}'");

            var embedded = JsonNode.Parse(await source.CallAsync("embed", new JsonObject { ["input"] = new JsonArray("abc", "de") }))!;
            Check((float)embedded["embeddings"]![0]![0]! == 3f && (float)embedded["embeddings"]![1]![0]! == 2f && engine.Stats("words").Requests == 1, $"embed over MCP: {embedded}");

            foreach (var (name, arguments, error) in new[]
            {
                ("generate", new JsonObject { ["model"] = "nope", ["prompt"] = "abc" }, "one of"),
                ("generate", new JsonObject { ["prompt"] = "abc" }, "'model' is required"),
                ("chat", new JsonObject { ["messages"] = new JsonArray() }, "must not be empty"),
            })
            {
                string message = "";
                try { await source.CallAsync(name, arguments); } catch (InvalidOperationException e) { message = e.Message; }
                Check(message.Contains(error, StringComparison.Ordinal), $"{name} with bad arguments is an error: '{message}'");
            }
        }

        await stop.CancelAsync();
        try
        {
            await running;
        }
        catch (OperationCanceledException)
        {
        }
    }

    private static void CliServeMcp()
    {
        string tools = typeof(CliServeTestTools).Assembly.Location;
        Check(ServeCli("serve", TestData("gguf/tiny-qwen3-q8.gguf"), "--tools", tools).Code == 2, "--tools without --mcp is a usage error");

        var toServer = new Pipe();
        var toClient = new Pipe();
        var previous = McpServeCommand.Streams;
        McpServeCommand.Streams = () => (toServer.Reader.AsStream(), toClient.Writer.AsStream());
        try
        {
            var output = new StringWriter();
            var error = new StringWriter();
            var serving = Task.Run(() => CommandLine.Run(["serve", "tiny=" + TestData("gguf/tiny-qwen3-q8.gguf"), "--mcp", "--tools", tools, "--cache", TempFolder()], output, error));
            Task.Run(async () =>
            {
                await using var source = await McpTools.ConnectAsync(new StreamClientTransport(toServer.Writer.AsStream(), toClient.Reader.AsStream()));
                var listed = (await source.ListToolsAsync()).Select(t => t.Definition.Name).ToList();
                Check(listed.Contains("generate") && listed.Contains("chat") && listed.Contains("serve_add") && !listed.Contains("embed"), $"models and tools over MCP: {string.Join(",", listed)}");
                Check(await source.CallAsync("serve_add", new JsonObject { ["a"] = 2, ["b"] = 40 }) == "42", "an assembly's tool over MCP");
                _ = await source.CallAsync("chat", new JsonObject
                {
                    ["messages"] = new JsonArray(new JsonObject { ["role"] = "user", ["content"] = "hi" }),
                    ["max_tokens"] = 3,
                });
                _ = await source.CallAsync("generate", new JsonObject { ["model"] = "tiny", ["prompt"] = "abc", ["max_tokens"] = 3 });
            }).GetAwaiter().GetResult();
            toServer.Writer.Complete();
            Check(serving.Wait(TimeSpan.FromSeconds(60)) && serving.Result == 0, $"serve --mcp ends with its input: {error}");
            Check(output.ToString().Length == 0 && error.ToString().Contains("loaded tiny") && error.ToString().Contains("over MCP"),
                $"status lines go to the error output: out '{output}', error '{error}'");
        }
        finally
        {
            McpServeCommand.Streams = previous;
        }
    }
}
