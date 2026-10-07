// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Nlp;
using Idrak.Retrieval;

namespace Idrak.Cli.Commands.Retrieval;

/// <summary><c>idrak rag ask "question" --index INDEX --model MODEL</c>: answers from the index's passages with citations.</summary>
internal sealed class RagAskCommand : Command
{
    public override string Name => "rag ask";

    public override string Summary => "Answer a question from an index's passages with a chat model, citing them";

    public override string Usage =>
        "\"question\" --index INDEX --model MODEL [options]\n\n" +
        "Finds the best passages, shows them numbered to the chat model and streams its answer; the passages it cites\n" +
        "as [n] are listed after it.\n\n" +
        "Options:\n" +
        "      --index INDEX     the index file (required)\n" +
        "  -m, --model MODEL     the chat model (required)\n" +
        "  -w, --weights FORMAT  its weight format; -k, --kv FORMAT: its key/value cache format\n" +
        "      --context N       its context length; --adapter DIR: merge an adapter first\n" +
        "      --top N           passages shown to the model (default 4)\n" +
        "  -s, --system TEXT     a system message\n" +
        "      --max-tokens N    longest answer (default 512)\n" +
        "      --temperature T   sampling temperature (default 0: greedy)\n" +
        "      --think / --no-think  the reasoning mode passed to the chat template\n\n" +
        "Examples:\n" +
        "  idrak rag ask \"How do I reset?\" --index docs.idx -m org/model --no-think\n" +
        "  idrak rag ask \"What does IDRAK_CACHE change?\" --index docs.idx -m mymodel --top 6 -j";

    public override IReadOnlyCollection<string> ValueOptions =>
        [.. ModelChoices.ValueOptions, "--index", "--model", "--top", "--system", "--max-tokens", "--temperature"];

    public override IReadOnlyCollection<string> Flags => ["--think", "--no-think"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } =
        new Dictionary<string, string>(ModelChoices.ShortForms) { ["-m"] = "--model", ["-s"] = "--system" };

    public override int Run(CommandContext context)
    {
        string question = string.Join(' ', context.Positional);
        if (question.Length == 0)
        {
            throw new UsageException("Missing the question, e.g. idrak rag ask \"How do I reset?\" --index docs.idx -m MODEL");
        }

        string path = context.Option("--index") ?? throw new UsageException("Missing --index INDEX (built with 'idrak rag index').");
        string modelName = context.Option("--model") ?? throw new UsageException("Missing --model MODEL (the chat model that answers).");
        int top = context.IntOption("--top", 4), maxTokens = context.IntOption("--max-tokens", 512);
        if (top <= 0 || maxTokens <= 0)
        {
            throw new UsageException("--top and --max-tokens need numbers above 0.");
        }

        float temperature = context.Option("--temperature") is { } t
            ? float.TryParse(t, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float value) && value >= 0 ? value
                : throw new UsageException($"--temperature needs a number of 0 or more, not '{t}'.")
            : 0f;
        if (context.Flag("--think") && context.Flag("--no-think"))
        {
            throw new UsageException("Choose --think or --no-think, not both.");
        }

        var choice = ModelChoices.Choose(context, modelName);
        using var model = ModelChoices.Load(context, choice);
        using var opened = RagIndexFile.Open(context, path, (choice.Model, choice.Weights, model));
        var chat = model.CreateChat(ModelChoices.CacheLayout(choice), model.MaxPositions);
        var options = temperature == 0f
            ? new GenerationOptions { Temperature = 0f, TopK = 1, TopP = 1f, RepeatPenalty = 1f, NumPredict = maxTokens, NumCtx = model.MaxPositions }
            : new GenerationOptions { Temperature = temperature, NumPredict = maxTokens, NumCtx = model.MaxPositions, Seed = context.Seed };
        var builder = Rag.For(chat).Retrieve(opened.Index, top).Options(options)
            .Think(context.Flag("--think") ? true : context.Flag("--no-think") ? false : null)
            .Label(c => $"{c.DocumentId}#{c.Position}");
        if (context.Option("--system") is { } system)
        {
            builder.System(system);
        }

        var pipeline = builder.Build();
        bool streaming = !context.Json && !context.Quiet;
        foreach (var delta in Sync(pipeline.StreamAsync(question)))
        {
            if (streaming && delta.Content.Length > 0)
            {
                context.Output.Write(delta.Content);
            }
        }

        var answer = pipeline.LastAnswer ?? throw new InvalidOperationException("The model gave no answer.");
        if (streaming)
        {
            context.Output.WriteLine();
        }
        else if (context.Quiet && !context.Json)
        {
            context.Output.WriteLine(answer.Text);
        }

        context.Write(answer.Cited.Count == 0 ? $"\n(no passage cited; {answer.Passages.Count} shown to the model)" : "\nCited:");
        foreach (var c in answer.Cited)
        {
            context.Write($"  [{c.Number}] {c.Label} (score {c.Score:G4}): {RagIndexFile.Snippet(c.Chunk.Text)}");
        }

        JsonObject Citation(Citation c) => new()
        {
            ["number"] = c.Number,
            ["label"] = c.Label,
            ["document"] = c.Chunk.DocumentId,
            ["position"] = c.Chunk.Position,
            ["score"] = c.Score,
            ["text"] = c.Chunk.Text,
        };
        context.WriteJson(new JsonObject
        {
            ["question"] = question,
            ["answer"] = answer.Text,
            ["thinking"] = answer.Message.Thinking,
            ["done_reason"] = answer.DoneReason,
            ["model"] = choice.Model,
            ["index"] = path,
            ["cited"] = new JsonArray([.. answer.Cited.Select(c => (JsonNode)Citation(c))]),
            ["passages"] = new JsonArray([.. answer.Passages.Select(c => (JsonNode)Citation(c))]),
        });
        return ExitCodes.Ok;
    }

    // The deltas of an asynchronous stream, read in order on this thread.
    private static IEnumerable<ChatDelta> Sync(IAsyncEnumerable<ChatDelta> stream)
    {
        var enumerator = stream.GetAsyncEnumerator();
        try
        {
            while (enumerator.MoveNextAsync().AsTask().GetAwaiter().GetResult())
            {
                yield return enumerator.Current;
            }
        }
        finally
        {
            enumerator.DisposeAsync().AsTask().GetAwaiter().GetResult();
        }
    }
}
