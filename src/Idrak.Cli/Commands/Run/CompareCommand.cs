// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Generation;

namespace Idrak.Cli.Commands.Run;

/// <summary><c>idrak compare MODEL_A MODEL_B "prompt"</c>: the same prompt on two models (or two settings of one), side by side.</summary>
internal sealed class CompareCommand : Command
{
    public override string Name => "compare";

    public override string Summary => "The same prompt on two models (or two settings of one), answers side by side with speed";

    public override string Usage => """
        MODEL_A MODEL_B ["prompt"] [options]

        Loads each model in turn (one in memory at a time), answers the prompt with the same settings and seed, and
        prints both answers with their speed. MODEL_B may be the same model as MODEL_A with other settings: aliases carry
        their own --weights and --kv, and --weights-b / --kv-b change B only. The prompt may also come from --input or
        standard input.

        Options:
          -i, --input FILE       read (more of) the prompt from FILE (repeatable)
          -w, --weights FORMAT   weight format of both (default: as stored); --weights-b FORMAT: of B only
          -k, --kv FORMAT        KV cache format of both (default float32); --kv-b FORMAT: of B only
              --context N        context window in tokens (default 4096, at most the model's)
              --width N          width of the side-by-side text (default 100; 0 prints the answers one after the other)
        """ + "\n" + GenerationSettings.Help.Replace("      --tools FILE.dll   tools the model may call: the [Tool] methods of an assembly (repeatable)\n", "") + """

        Examples:
          idrak compare Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B "Explain recursion in one paragraph"
          idrak compare qwen qwen --weights-b int4 "Explain recursion" --seed 1
        """ + "\n\n" + GenerationSettings.EnvironmentHelp;

    public override IReadOnlyCollection<string> ValueOptions =>
        [.. Models.ValueOptions, .. GenerationSettings.ValueOptions.Where(o => o != "--tools"), "--input", "--weights-b", "--kv-b", "--width"];

    public override IReadOnlyCollection<string> Flags => GenerationSettings.Flags;

    public override IReadOnlyDictionary<string, string> ShortForms { get; } =
        new Dictionary<string, string>(Models.ShortForms.Concat(GenerationSettings.ShortForms).Append(KeyValuePair.Create("-i", "--input")));

    public override int Run(CommandContext context)
    {
        string a = context.Argument(0, "MODEL_A"), b = context.Argument(1, "MODEL_B");
        string prompt = PromptInput.Read(context, 2);
        var settings = GenerationSettings.From(context);
        settings.Seed ??= 1;                                      // the same draw for both, so differences come from the models
        int width = context.IntOption("--width", 100);
        var choiceB = Models.Choose(context, b);
        choiceB = choiceB with { Weights = context.Option("--weights-b") ?? choiceB.Weights, Kv = context.Option("--kv-b") ?? choiceB.Kv };
        var results = new List<(string Label, ChatAnswer Answer)>();
        foreach (var (label, name, choice) in new[] { ("A", a, Models.Choose(context, a)), ("B", b, choiceB) })
        {
            using var loaded = LoadedChat.Load(context, name, settings, choice);
            var messages = new List<ChatMessage>();
            if (settings.System is { Length: > 0 } system)
            {
                messages.Add(new ChatMessage("system", system));
            }

            messages.Add(new ChatMessage("user", prompt));
            var answer = new ChatResponder(loaded.Chat, null).Answer(messages, settings, loaded.Context);
            string title = $"{label}: {choice.Model}{(choice.Weights is { } w ? $" -w {w}" : "")}{(choice.Kv is { } k ? $" -k {k}" : "")}";
            results.Add((title, answer));
        }

        if (width <= 0)
        {
            foreach (var (title, answer) in results)
            {
                context.Write($"== {title}  ({answer.GeneratedTokens} tokens, {answer.TokensPerSecond:F1} tokens/s)");
                context.Write(answer.Message.Content);
                context.Write("");
            }
        }
        else
        {
            int column = Math.Max(10, (width - 3) / 2);
            var left = Wrap(results[0].Answer.Message.Content, column);
            var right = Wrap(results[1].Answer.Message.Content, column);
            context.Write($"{Fit(results[0].Label, column)} | {Fit(results[1].Label, column)}");
            context.Write($"{new string('-', column)}-+-{new string('-', column)}");
            for (int i = 0; i < Math.Max(left.Count, right.Count); i++)
            {
                context.Write($"{(i < left.Count ? left[i] : "").PadRight(column)} | {(i < right.Count ? right[i] : "")}".TrimEnd());
            }

            context.Write($"{new string('-', column)}-+-{new string('-', column)}");
            context.Write($"{Fit(Speed(results[0].Answer), column)} | {Speed(results[1].Answer)}");
        }

        context.WriteJson(new JsonObject
        {
            ["prompt"] = prompt,
            ["results"] = new JsonArray([.. results.Select(r =>
            {
                var json = new JsonObject { ["label"] = r.Label, ["text"] = r.Answer.Message.Content };
                foreach (var (key, value) in r.Answer.Figures())
                {
                    json[key] = value?.DeepClone();
                }

                return (JsonNode)json;
            })]),
        });
        return ExitCodes.Ok;
    }

    private static string Speed(ChatAnswer answer) => $"{answer.GeneratedTokens} tokens, {answer.TokensPerSecond:F1} tokens/s";

    private static string Fit(string text, int width) => text.Length > width ? text[..(width - 3)] + "..." : text.PadRight(width);

    // Words wrapped to lines of at most width characters (longer words are cut).
    internal static List<string> Wrap(string text, int width)
    {
        var lines = new List<string>();
        foreach (string paragraph in text.Replace("\r", "").Split('\n'))
        {
            var line = new System.Text.StringBuilder();
            foreach (string word in paragraph.Split(' ', StringSplitOptions.RemoveEmptyEntries))
            {
                for (string rest = word; rest.Length > 0;)
                {
                    if (line.Length > 0 && line.Length + 1 + rest.Length > width)
                    {
                        lines.Add(line.ToString());
                        line.Clear();
                    }

                    string piece = rest.Length > width ? rest[..width] : rest;
                    line.Append(line.Length > 0 ? " " : "").Append(piece);
                    rest = rest[piece.Length..];
                }
            }

            lines.Add(line.ToString());
        }

        return lines;
    }
}
