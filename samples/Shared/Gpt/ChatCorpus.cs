// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json.Nodes;
using Idrak.Generation;

namespace Idrak.Samples.Gpt;

/// <summary>
/// Synthetic chat transcripts in the ChatML format of <see cref="ChatMLTemplate"/>, so a small character-level GPT can
/// learn the conventions of a chat model: answering turns, reasoning inside think tags, calling a web_fetch tool when
/// tools are offered, and answering from a tool result with a citation. The facts are invented; the format is real.
/// </summary>
public static class ChatCorpus
{
    public static readonly string[] Systems =
    [
        "You are a helpful assistant. Cite sources as [1], [2] when a research pack is present.",
        "You are a helpful assistant.",
        "You are a concise assistant.",
    ];

    public static readonly ToolDefinition WebFetch = new("web_fetch", "Fetch a page from the allowlisted search results.",
        JsonNode.Parse("""{"type":"object","properties":{"url":{"type":"string","description":"Absolute http(s) URL from the allowlist."}},"required":["url"]}"""));

    private static readonly string[] Products = ["Ollama", "Idrak", "Python", "Node", "Redis", "Postgres", "Docker", "Kafka", "Nginx", "Rust"];

    /// <summary>Returns <paramref name="count"/> transcripts, each ending with the assistant turn and the end-of-turn marker.</summary>
    public static string Generate(int count, int seed)
    {
        var random = new Random(seed);
        var template = new ChatMLTemplate();
        var sb = new StringBuilder();
        for (int i = 0; i < count; i++)
        {
            var (prompt, answer) = Sample(random, template);
            sb.Append(prompt).Append(answer).Append("<|im_end|>\n");
        }

        return sb.ToString();
    }

    private static (string Prompt, string Answer) Sample(Random random, ChatTemplate template)
    {
        string system = Systems[random.Next(Systems.Length)];
        bool tools = random.Next(3) > 0;                     // two thirds offer the tool
        bool? think = random.Next(4) switch { 0 => false, 1 => null, _ => true };
        string product = Products[random.Next(Products.Length)];
        string version = $"{random.Next(0, 20)}.{random.Next(0, 30)}.{random.Next(0, 10)}";
        string slug = product.ToLowerInvariant();
        var messages = new List<ChatMessage> { new("system", system) };
        ToolDefinition[] offered = tools ? [WebFetch] : [];
        string Think(string text) => think == false ? "" : $"<think>\n{text}\n</think>\n\n";

        switch (random.Next(5))
        {
            case 0 or 1:
            {
                // Latest-version questions: call the tool when available, otherwise say so.
                string question = random.Next(2) == 0 ? $"What is the latest {product} version?" : $"Which version of {product} is the newest?";
                messages.Add(new("user", question));
                if (!tools)
                {
                    return (template.Render(messages, offered, think),
                        Think($"The user asks for the latest {product} release. I have no tool to check it.") +
                        $"I cannot check the latest {product} version without web access. Please see the {product} release page.");
                }

                string url = $"https://{slug}.com/releases";
                if (random.Next(2) == 0)
                {
                    // First turn: decide to fetch.
                    return (template.Render(messages, offered, think),
                        Think($"The user wants the latest {product} version. I should fetch the release page.") +
                        $"<tool_call>\n{{\"name\": \"web_fetch\", \"arguments\": {{\"url\": \"{url}\"}}}}\n</tool_call>");
                }

                // Second turn: answer from the tool result with a citation.
                messages.Add(new("assistant", "", ToolCalls: [new ToolCall("web_fetch", new JsonObject { ["url"] = url })]));
                messages.Add(new("tool", $"[1] {url}: {product} {version} is the latest release.", ToolName: "web_fetch"));
                return (template.Render(messages, offered, think),
                    Think($"The page says {product} {version} is the latest release.") +
                    $"The latest {product} version is {version} [1].");
            }

            case 2:
            {
                int a = random.Next(1, 50), b = random.Next(1, 50);
                messages.Add(new("user", $"What is {a} + {b}?"));
                return (template.Render(messages, offered, think), Think($"{a} + {b} = {a + b}.") + $"{a} + {b} = {a + b}.");
            }

            case 3:
            {
                string[] greetings = ["Hello!", "Hi there.", "Good morning!"];
                messages.Add(new("user", greetings[random.Next(greetings.Length)]));
                return (template.Render(messages, offered, think), Think("The user greets me.") + "Hello! How can I help you today?");
            }

            default:
            {
                messages.Add(new("user", $"What is {product}?"));
                return (template.Render(messages, offered, think),
                    Think($"The user asks what {product} is.") + $"{product} is a software project. Ask me about its latest version and I can look it up.");
            }
        }
    }
}
