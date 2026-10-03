// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Generation;

namespace Idrak.Cli.Commands.Run;

/// <summary><c>idrak batch MODEL --input prompts.jsonl --out answers.jsonl</c>: many prompts with batched generation, resumable.</summary>
internal sealed class BatchCommand : Command
{
    public override string Name => "batch";

    public override string Summary => "Answers many prompts (JSON Lines) with batched generation, resumable";

    public override string Usage => """
        MODEL --input prompts.jsonl --out answers.jsonl [options]

        Each input line is a prompt: a JSON string, {"prompt": "...", "id": ...} or {"messages": [...]} in the chat
        layout (a plain text line is a prompt too). Each answer is appended to --out as one line with the line's index,
        its id when given, the prompt, the answer and the figures, as soon as its group is done; run the same command
        again after an interruption and the lines already answered are skipped.

        Options:
          -i, --input FILE       the prompts (JSON Lines)
          -o, --out FILE         the answers (JSON Lines; appended to)
              --batch-size N     prompts generated together (default 8; 1 when the model cannot batch)
          -w, --weights FORMAT   int8, int4, bf16 or a registered packed format (default: as stored)
          -k, --kv FORMAT        KV cache format: float32, int8, bfloat16 or a registered one (default float32)
              --context N        context window in tokens (default 4096, at most the model's)
              --adapter DIR      merge a LoRA adapter into the weights as they are read
        """ + "\n" + GenerationSettings.Help.Replace("      --tools FILE.dll   tools the model may call: the [Tool] methods of an assembly (repeatable)\n", "") + """

        Examples:
          idrak batch Qwen/Qwen3-0.6B --input prompts.jsonl --out answers.jsonl
          idrak batch qwen -i prompts.jsonl -o answers.jsonl --batch-size 16 --temperature 0
        """;

    public override IReadOnlyCollection<string> ValueOptions =>
        [.. Models.ValueOptions, .. GenerationSettings.ValueOptions.Where(o => o != "--tools"), "--input", "--out", "--batch-size"];

    public override IReadOnlyCollection<string> Flags => GenerationSettings.Flags;

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string>(
        Models.ShortForms.Concat(GenerationSettings.ShortForms).Append(KeyValuePair.Create("-i", "--input")).Append(KeyValuePair.Create("-o", "--out")));

    public override int Run(CommandContext context)
    {
        string name = context.Argument(0, "MODEL (a Hugging Face id, a folder, a .gguf file or an alias)");
        string input = context.Option("--input") ?? throw new UsageException("Missing --input prompts.jsonl.");
        string output = context.Option("--out") ?? throw new UsageException("Missing --out answers.jsonl.");
        int batchSize = context.IntOption("--batch-size", 8);
        if (batchSize < 1)
        {
            throw new UsageException("--batch-size needs a positive whole number.");
        }

        if (!File.Exists(input))
        {
            throw new UsageException($"Input file not found: {input}");
        }

        var settings = GenerationSettings.From(context);
        var prompts = ReadPrompts(input, settings.System);
        var done = Answered(output);
        var pending = prompts.Where(p => !done.Contains(p.Index)).ToList();
        context.Write($"{prompts.Count} prompts, {done.Count} already answered in {output}, {pending.Count} to go.");
        if (pending.Count == 0)
        {
            context.WriteJson(Result(prompts.Count, done.Count, 0, 0, 0, output));
            return ExitCodes.Ok;
        }

        using var loaded = LoadedChat.Load(context, name, settings);
        if (!loaded.Chat.Generator.SupportsBatches)
        {
            batchSize = 1;
        }

        var options = settings.ToOptions(loaded.Context);
        int answered = 0, tokens = 0;
        var watch = Stopwatch.StartNew();
        using var writer = new StreamWriter(output, append: true);
        foreach (var group in pending.Chunk(batchSize))
        {
            var requests = group.Select(p => new ChatRequest(p.Messages, null, settings.Think, options)).ToList();
            IReadOnlyList<ChatChunk> replies = requests.Count == 1 ? [loaded.Chat.Chat(requests[0])] : loaded.Chat.ChatBatch(requests);
            for (int i = 0; i < group.Length; i++)
            {
                var (prompt, reply) = (group[i], replies[i]);
                var line = new JsonObject { ["index"] = prompt.Index };
                if (prompt.Id is not null)
                {
                    line["id"] = prompt.Id.DeepClone();
                }

                line["prompt"] = prompt.Messages[^1].Content;
                line["answer"] = reply.Message?.Content ?? "";
                if (reply.Message?.Thinking is { } thinking)
                {
                    line["thinking"] = thinking;
                }

                line["done_reason"] = reply.DoneReason;
                line["prompt_tokens"] = reply.Stats?.PromptTokens ?? 0;
                line["generated_tokens"] = reply.Stats?.GeneratedTokens ?? 0;
                writer.WriteLine(line.ToJsonString());
                tokens += reply.Stats?.GeneratedTokens ?? 0;
                answered++;
            }

            writer.Flush();
            context.Write($"  {done.Count + answered}/{prompts.Count} answered, {tokens / Math.Max(watch.Elapsed.TotalSeconds, 1e-9):F1} tokens/s");
        }

        double seconds = watch.Elapsed.TotalSeconds;
        context.Write($"Wrote {answered} answers to {output} in {seconds:F1} s ({tokens} tokens, {tokens / Math.Max(seconds, 1e-9):F1} tokens/s).");
        context.WriteJson(Result(prompts.Count, done.Count, answered, tokens, seconds, output));
        return ExitCodes.Ok;
    }

    private static JsonObject Result(int prompts, int skipped, int answered, int tokens, double seconds, string output) => new()
    {
        ["prompts"] = prompts,
        ["skipped"] = skipped,
        ["answered"] = answered,
        ["generated_tokens"] = tokens,
        ["seconds"] = Math.Round(seconds, 3),
        ["tokens_per_second"] = Math.Round(tokens / Math.Max(seconds, 1e-9), 2),
        ["out"] = output,
    };

    private sealed record Prompt(int Index, JsonNode? Id, List<ChatMessage> Messages);

    // Every non-empty line of the input as a conversation to answer.
    private static List<Prompt> ReadPrompts(string path, string? system)
    {
        var prompts = new List<Prompt>();
        int index = 0;
        foreach (string raw in File.ReadLines(path))
        {
            string line = raw.Trim();
            if (line.Length == 0)
            {
                continue;
            }

            JsonNode? json = null;
            if (line[0] is '{' or '"' or '[')
            {
                try
                {
                    json = JsonNode.Parse(line);
                }
                catch (JsonException e)
                {
                    throw new InvalidOperationException($"{path}, line {index + 1}: not JSON ({e.Message}).");
                }
            }

            List<ChatMessage> messages = json switch
            {
                JsonObject { } o when o["messages"] is JsonArray list => [.. list.Select(ChatHistory.Message)],
                JsonObject { } o when o["prompt"] is JsonValue p => [new ChatMessage("user", (string?)p ?? "")],
                JsonValue v when v.TryGetValue(out string? text) => [new ChatMessage("user", text)],
                null => [new ChatMessage("user", line)],
                _ => throw new InvalidOperationException($"{path}, line {index + 1}: expected a string, {{\"prompt\": ...}} or {{\"messages\": [...]}}."),
            };
            if (system is not null && messages.All(m => m.Role != "system"))
            {
                messages.Insert(0, new ChatMessage("system", system));
            }

            prompts.Add(new Prompt(index++, json is JsonObject withId ? withId["id"] : null, messages));
        }

        return prompts;
    }

    // The indexes already in the output (a torn last line from an interruption is dropped and answered again).
    private static HashSet<int> Answered(string path)
    {
        var done = new HashSet<int>();
        if (!File.Exists(path))
        {
            return done;
        }

        var good = new List<string>();
        bool torn = false;
        foreach (string line in File.ReadLines(path))
        {
            try
            {
                if (JsonNode.Parse(line)?["index"] is JsonValue v && v.TryGetValue(out int index))
                {
                    done.Add(index);
                    good.Add(line);
                    continue;
                }
            }
            catch (JsonException)
            {
            }

            torn = true;
        }

        if (torn)
        {
            File.WriteAllLines(path, good);
        }

        return done;
    }
}
