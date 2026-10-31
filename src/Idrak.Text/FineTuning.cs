// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Cluster;
using Idrak.Data;
using Idrak.Datasets.Abstractions;
using Idrak.Diagnostics;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Text.Abstractions;
using Idrak.Optimizers;

namespace Idrak.Text;

/// <summary>
/// One conversation to fine-tune on: messages (system, user, assistant with optional reasoning and tool calls, tool
/// results) and the tools that were available. Read agent transcripts with <see cref="ReadJsonLines"/>.
/// </summary>
/// <param name="Messages">The conversation.</param>
/// <param name="Tools">The tools offered to the model (rendered by the chat template as the model saw them).</param>
/// <param name="Think">The reasoning mode passed to the template (enable_thinking), or null for the template's default.</param>
public sealed record ChatTranscript(IReadOnlyList<ChatMessage> Messages, IReadOnlyList<ToolDefinition> Tools, bool? Think = null)
{
    /// <summary>
    /// Reads one transcript: <c>{"messages": [...], "tools": [...]}</c> in the OpenAI / Hugging Face chat format
    /// (content as a string or a list of parts, each read by <see cref="ChatParts.FromJson"/>: text, images or a registered kind; reasoning in reasoning_content, thinking or reasoning; tool calls as
    /// [{"function": {"name", "arguments"}}] with arguments as an object or a JSON string; tool results with role "tool"
    /// and a name or a tool_call_id), or ShareGPT's <c>{"conversations": [{"from": "human" | "gpt" | "system", "value"}]}</c>.
    /// Optional "enable_thinking" / "think" sets <see cref="Think"/>.
    /// </summary>
    public static ChatTranscript FromJson(JsonObject json) => Parse(json, null);

    // FromJson, with `content` reading each message's "content" and each ShareGPT turn's "value" into its parts (the
    // tuning data formats resolve images there); null reads them as the chat JSON does (ChatParts).
    internal static ChatTranscript Parse(JsonObject json, Func<JsonNode?, IReadOnlyList<ChatPart>>? content)
    {
        ArgumentNullException.ThrowIfNull(json);
        var messages = new List<ChatMessage>();
        var callNames = new Dictionary<string, string>(StringComparer.Ordinal);
        if (json["messages"] is JsonArray list)
        {
            foreach (var node in list)
            {
                messages.Add(Message(node as JsonObject ?? throw new InvalidDataException("A message is not an object."), callNames, content));
            }
        }
        else if (json["conversations"] is JsonArray shareGpt)
        {
            foreach (var node in shareGpt)
            {
                string from = (string?)node?["from"] ?? "";
                string role = from switch
                {
                    "human" or "user" => "user",
                    "gpt" or "assistant" or "model" => "assistant",
                    "system" => "system",
                    "tool" or "observation" or "function_response" => "tool",
                    _ => throw new InvalidDataException($"Unknown ShareGPT speaker '{from}'."),
                };
                messages.Add(content is null ? new ChatMessage(role, (string?)node?["value"] ?? "") : new ChatMessage(role, content(node?["value"])));
            }
        }
        else
        {
            throw new InvalidDataException("A transcript needs \"messages\" (or ShareGPT \"conversations\").");
        }

        var tools = new List<ToolDefinition>();
        foreach (var node in json["tools"] as JsonArray ?? [])
        {
            var tool = node?["function"] as JsonObject ?? node as JsonObject ?? throw new InvalidDataException("A tool is not an object.");
            tools.Add(new ToolDefinition((string?)tool["name"] ?? throw new InvalidDataException("A tool has no name."),
                (string?)tool["description"], tool["parameters"]?.DeepClone()));
        }

        bool? think = json["enable_thinking"] is JsonValue e ? (bool)e : json["think"] is JsonValue t ? (bool)t : null;
        return new ChatTranscript(messages, tools, think);
    }

    private static ChatMessage Message(JsonObject m, Dictionary<string, string> callNames, Func<JsonNode?, IReadOnlyList<ChatPart>>? read)
    {
        string role = (string?)m["role"] ?? throw new InvalidDataException("A message has no role.");
        IReadOnlyList<ChatPart> content;
        try
        {
            content = read is null ? ChatParts.ContentFromJson(m["content"]) : read(m["content"]);
        }
        catch (Exception ex) when (ex is FormatException or NotSupportedException)
        {
            throw new InvalidDataException($"Unsupported content in a {role} message: {ex.Message}", ex);
        }

        string? thinking = (string?)m["reasoning_content"] ?? (string?)m["thinking"] ?? (string?)m["reasoning"];
        List<ToolCall>? calls = null;
        if (m["tool_calls"] is JsonArray toolCalls && toolCalls.Count > 0)
        {
            calls = [];
            foreach (var node in toolCalls)
            {
                var function = node?["function"] as JsonObject ?? node as JsonObject ?? throw new InvalidDataException("A tool call is not an object.");
                string name = (string?)function["name"] ?? throw new InvalidDataException("A tool call has no name.");
                var arguments = function["arguments"] switch
                {
                    JsonObject o => (JsonObject)o.DeepClone(),
                    JsonValue s when (string?)s is { } text => ParseArguments(text, name),
                    null => new JsonObject(),
                    _ => throw new InvalidDataException($"The arguments of a call to {name} are neither an object nor a JSON string."),
                };
                calls.Add(new ToolCall(name, arguments));
                if ((string?)node?["id"] is { } id)
                {
                    callNames[id] = name;
                }
            }
        }

        string? toolName = (string?)m["name"];
        if (toolName is null && (string?)m["tool_call_id"] is { } callId && callNames.TryGetValue(callId, out var called))
        {
            toolName = called;
        }

        return new ChatMessage(role, content, string.IsNullOrEmpty(thinking) ? null : thinking, calls, role == "tool" ? toolName : null);
    }

    private static JsonObject ParseArguments(string text, string name)
    {
        try
        {
            return JsonNode.Parse(text) as JsonObject ?? throw new InvalidDataException($"The arguments of a call to {name} are not a JSON object.");
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException($"The arguments of a call to {name} are not valid JSON: {ex.Message}", ex);
        }
    }

    /// <summary>Reads one transcript per non-empty line of a JSON Lines file (see <see cref="FromJson"/>).</summary>
    public static IEnumerable<ChatTranscript> ReadJsonLines(string path)
    {
        foreach (var (node, number) in JsonLines.Read(path))
        {
            ChatTranscript transcript;
            try
            {
                transcript = FromJson(node as JsonObject ?? throw new InvalidDataException("The line is not a JSON object."));
            }
            catch (Exception ex) when (ex is JsonException or InvalidDataException or InvalidOperationException or FormatException)
            {
                throw new InvalidDataException($"{path}, line {number}: {ex.Message}", ex);
            }

            yield return transcript;
        }
    }
}

/// <summary>
/// Turns transcripts into <see cref="TrainingSequence"/>s with a model's own chat template, so the fine-tuning data has
/// exactly the layout the model sees when it is used (tool definitions, tool-call syntax, reasoning blocks). The
/// assistant's turns are found from the template itself: the text its generation prompt adds (the assistant header)
/// and the text that ends an assistant message, discovered by rendering a probe conversation.
/// </summary>
public sealed class ChatTranscriptEncoder
{
    private const string UserProbe = "⁠ns-user-probe⁠", AssistantProbe = "⁠ns-assistant-probe⁠";

    /// <summary>Creates the encoder for <paramref name="template"/> and <paramref name="tokenizer"/>.</summary>
    public ChatTranscriptEncoder(JinjaChatTemplate template, ITokenizer tokenizer)
    {
        Template = template;
        Tokenizer = tokenizer;
        var user = new ChatMessage("user", UserProbe);
        string prompt = template.Render([user], [], null, addGenerationPrompt: true);
        string bare = template.Render([user], [], null, addGenerationPrompt: false);
        string pair = template.Render([user, new ChatMessage("assistant", AssistantProbe)], [], null, addGenerationPrompt: false);
        int answer = pair.IndexOf(AssistantProbe, StringComparison.Ordinal);
        if (answer < 0)
        {
            throw new InvalidOperationException("The chat template does not render an assistant message's content.");
        }

        string header = prompt.StartsWith(bare, StringComparison.Ordinal) ? prompt[bare.Length..] : "";
        if (header.Trim().Length == 0)
        {
            // No generation prompt (e.g. [INST] … [/INST] templates): the text between the user's content and the answer.
            int userEnd = pair.IndexOf(UserProbe, StringComparison.Ordinal) + UserProbe.Length;
            header = pair[userEnd..answer].Trim();
        }

        string end = pair[(answer + AssistantProbe.Length)..].TrimEnd();
        AssistantHeader = header;
        AssistantEnd = end.Length > 0 ? end : template.EosToken;
        if (AssistantHeader.Length == 0)
        {
            throw new InvalidOperationException("Could not find how the chat template starts an assistant turn.");
        }
    }

    /// <summary>The template.</summary>
    public JinjaChatTemplate Template { get; }

    /// <summary>The tokenizer.</summary>
    public ITokenizer Tokenizer { get; }

    /// <summary>The text that starts an assistant turn (for ChatML: "&lt;|im_start|&gt;assistant\n").</summary>
    public string AssistantHeader { get; }

    /// <summary>The text that ends an assistant turn (for ChatML: "&lt;|im_end|&gt;"); trained, so the model learns to stop.</summary>
    public string AssistantEnd { get; }

    /// <summary>
    /// How transcripts with images are encoded (a vision-language model, <see cref="TuningVision.Create"/>): each image's
    /// marker is expanded by the family's prompt format to its blocks of image tokens, as generation does, and the
    /// sequences carry their images (<see cref="TrainingSequence.Images"/>). Null (the default): text only, and a
    /// transcript with an image is refused.
    /// </summary>
    public TuningVision? Vision { get; init; }

    /// <summary>
    /// The sources of soft tokens for the other kinds of message part (an audio part through Idrak.Audio's
    /// <c>ChatAudioParts</c>; one <see cref="ISoftPartSource"/> per kind): each part's marker is expanded by its source's
    /// format to its blocks of soft tokens, as generation does, and the sequences record the parts with their blocks'
    /// positions (<see cref="TrainingSequence.Parts"/>). None (the default): a transcript with such a part is refused.
    /// </summary>
    public IReadOnlyList<ISoftPartSource> Parts
    {
        get => _parts;
        init
        {
            ArgumentNullException.ThrowIfNull(value);
            foreach (var source in value)
            {
                ArgumentNullException.ThrowIfNull(source, nameof(value));
                if (source.Kind is ChatParts.Text or ChatParts.Image || value.Count(s => s.Kind == source.Kind) > 1)
                {
                    throw new ArgumentException($"Parts holds one source per kind other than text and image; '{source.Kind}' is {(source.Kind is ChatParts.Text or ChatParts.Image ? "not one" : "given twice")}.", nameof(value));
                }
            }

            _parts = value;
        }
    }

    private IReadOnlyList<ISoftPartSource> _parts = [];

    /// <summary>The rendered transcript and the character ranges [start, end) of the assistant's turns in it.</summary>
    public (string Text, IReadOnlyList<(int Start, int End)> Spans) Render(ChatTranscript transcript)
    {
        string text = Template.Render(transcript.Messages, transcript.Tools, transcript.Think, addGenerationPrompt: false);
        return (text, Spans(text));
    }

    // The assistant's turns in a rendered (or expanded) transcript: from each assistant header to the end of the turn.
    private List<(int Start, int End)> Spans(string text)
    {
        var spans = new List<(int, int)>();
        for (int at = text.IndexOf(AssistantHeader, StringComparison.Ordinal); at >= 0; at = text.IndexOf(AssistantHeader, at, StringComparison.Ordinal))
        {
            int start = at + AssistantHeader.Length;
            int end = AssistantEnd.Length == 0 ? -1 : text.IndexOf(AssistantEnd, start, StringComparison.Ordinal);
            end = end < 0 ? text.Length : end + AssistantEnd.Length;
            spans.Add((start, end));
            at = end;
        }

        return spans;
    }

    /// <summary>
    /// Whether <see cref="Encode"/> shortens a transcript that is too long by cutting the end of the user message before
    /// the last assistant turn (keeping its start) until the whole answer fits, instead of cutting the transcript's end,
    /// which loses the answer. On by default.
    /// </summary>
    public bool ShortenToFit { get; init; } = true;

    /// <summary>
    /// The transcript's tokens, with the assistant's turns marked as trained, at most <paramref name="maxLength"/> + 1
    /// tokens (inputs and targets are the sequence shifted by one); null when nothing trainable remains. A transcript that
    /// is too long is shortened in its last user message (see <see cref="ShortenToFit"/>), else cut at the end.
    /// </summary>
    /// <remarks>
    /// With images (<see cref="Vision"/>), each image's marker becomes its blocks of image tokens before tokenizing, and
    /// the sequence records where they lie; a sequence cut at the end keeps only whole blocks before the cut (the cut
    /// moves to the first block that does not fit).
    /// </remarks>
    /// <exception cref="InvalidOperationException">The transcript holds images and <see cref="Vision"/> is not set.</exception>
    /// <exception cref="InvalidDataException">An image lies in an assistant turn (images are read, never trained on).</exception>
    public TrainingSequence? Encode(ChatTranscript transcript, int maxLength)
    {
        var (tokens, trained, images, parts, turns) = Tokens(transcript);
        if (tokens.Count > maxLength + 1 && ShortenToFit && Shortened(transcript, tokens.Count, Answers(trained, turns, transcript), maxLength) is { } fitted)
        {
            return fitted;
        }

        return Sequence(tokens, trained, images, parts, maxLength);
    }

    // The tokens of the assistant's turns (every turn trained: Tokens' default), when they are exactly the transcript's
    // assistant messages; -1 when the rendered text holds a turn of another kind (an assistant header in the user's text).
    // Each turn is tokenized on its own, so shortening the user's message never changes this count.
    private static int Answers(List<bool> trained, int turns, ChatTranscript transcript)
    {
        if (turns != transcript.Messages.Count(m => m.Role == "assistant"))
        {
            return -1;
        }

        int count = 0;
        foreach (bool train in CollectionsMarshal.AsSpan(trained))
        {
            count += train ? 1 : 0;
        }

        return count;
    }

    // The first maxLength + 1 tokens as a sequence (null when nothing trainable remains), with the image blocks that lie
    // whole among its inputs (the cut moves back to the start of the first block that does not).
    private static TrainingSequence? Sequence(List<int> tokens, List<bool> trained, List<TrainingImage> images, List<TrainingPart> parts, int maxLength)
    {
        int keep = Math.Min(tokens.Count, maxLength + 1);
        if (images.Count == 0 && parts.Count == 0)
        {
            var text = new TrainingSequence([.. tokens.Take(keep)], [.. trained.Take(keep)]);
            return text.TrainedTokens > 0 ? text : null;
        }

        // The cut moves back to the start of the first block (of an image or a part) that does not lie whole among the inputs.
        foreach (var block in images.SelectMany(i => i.Blocks).Concat(parts.SelectMany(p => p.Blocks)).OrderBy(b => b.Position))
        {
            if (block.Position + block.Tokens > keep - 1)
            {
                keep = Math.Min(keep, block.Position);
                break;
            }
        }

        var kept = new List<TrainingImage>(images.Count);
        foreach (var image in images)
        {
            var blocks = image.Blocks.Where(b => b.Position + b.Tokens <= keep - 1).ToList();
            if (blocks.Count == image.Blocks.Count)
            {
                kept.Add(image);
            }
            else if (blocks.Count > 0)
            {
                kept.Add(image with { Blocks = blocks });
            }
        }

        var keptParts = new List<TrainingPart>(parts.Count);
        foreach (var part in parts)
        {
            var blocks = part.Blocks.Where(b => b.Position + b.Tokens <= keep - 1).ToList();
            if (blocks.Count == part.Blocks.Count)
            {
                keptParts.Add(part);
            }
            else if (blocks.Count > 0)
            {
                keptParts.Add(part with { Blocks = blocks });
            }
        }

        var sequence = new TrainingSequence([.. tokens.Take(keep)], [.. trained.Take(keep)]) { Images = kept.Count == 0 ? [] : kept, Parts = keptParts.Count == 0 ? [] : keptParts };
        return sequence.TrainedTokens > 0 ? sequence : null;
    }

    /// <summary>
    /// <paramref name="transcript"/> with its last user message shortened (its start kept) so that the whole transcript
    /// fits in <paramref name="maxLength"/> + 1 tokens; the transcript itself when it fits already; null when even an
    /// empty message leaves no room (or the transcript has no user message before its last assistant turn).
    /// </summary>
    public ChatTranscript? Fit(ChatTranscript transcript, int maxLength)
    {
        ArgumentNullException.ThrowIfNull(transcript);
        var (tokens, trained, _, _, turns) = Tokens(transcript);
        if (tokens.Count <= maxLength + 1)
        {
            return transcript;
        }

        return ShortenedTranscript(transcript, tokens.Count, Answers(trained, turns, transcript), maxLength).Transcript;
    }

    private TrainingSequence? Shortened(ChatTranscript transcript, int total, int answers, int maxLength)
    {
        var (fitted, tokens, trained, images, parts) = ShortenedTranscript(transcript, total, answers, maxLength);
        return fitted is null ? null : Sequence(tokens, trained, images, parts, maxLength);
    }

    // Cuts the end of the last user message before the last assistant turn until everything fits: a first cut from the
    // tokens over, then 10% less each time the estimate falls short (tokens do not map to characters exactly). `total` is
    // the whole transcript's count of tokens (tokenized once, by the caller); `answers` the assistant turns' (-1: unknown):
    // when they alone pass the budget no cut can fit, and nothing more is rendered.
    private (ChatTranscript? Transcript, List<int> Tokens, List<bool> Trained, List<TrainingImage> Images, List<TrainingPart> Parts) ShortenedTranscript(ChatTranscript transcript, int total,
        int answers, int maxLength)
    {
        if (answers > maxLength + 1)
        {
            return (null, [], [], [], []);
        }

        var messages = transcript.Messages;
        int answer = -1;
        for (int i = messages.Count - 1; i >= 0; i--)
        {
            if (messages[i].Role == "assistant")
            {
                answer = i;
                break;
            }
        }

        int user = -1;
        for (int i = answer - 1; i >= 0; i--)
        {
            if (messages[i].Role == "user")
            {
                user = i;
                break;
            }
        }

        if (answer < 0 || user < 0 || messages[user].Content.Length == 0)
        {
            return (null, [], [], [], []);
        }

        string content = messages[user].Content;
        int contentTokens = Math.Max(1, Tokenizer.Encode(content).Count);
        int keepTokens = contentTokens - (total - (maxLength + 1)) - 2;
        int length = keepTokens <= 0 ? 0 : (int)((long)content.Length * keepTokens / contentTokens);
        while (true)
        {
            var shorter = messages.ToArray();
            shorter[user] = shorter[user] with { Content = content[..length] };
            var candidate = transcript with { Messages = shorter };
            var (tokens, trained, images, parts, _) = Tokens(candidate);
            if (tokens.Count <= maxLength + 1)
            {
                return (candidate, tokens, trained, images, parts);
            }

            if (length == 0)
            {
                return (null, [], [], [], []);
            }

            length = length * 9 / 10;
        }
    }

    // Every token of the rendered transcript (each image's marker expanded to its blocks of image tokens), the assistant's
    // turns marked as trained (only the last `answerTurns` of them when given), the images with their blocks' positions,
    // and how many assistant turns the text holds.
    private (List<int> Tokens, List<bool> Trained, List<TrainingImage> Images, List<TrainingPart> Parts, int Turns) Tokens(ChatTranscript transcript, int answerTurns = int.MaxValue)
    {
        var (text, spans) = Render(transcript);
        List<ChatImage>? images = null;
        List<IReadOnlyList<ImageTokenLayout>>? layouts = null;
        if (transcript.Messages.Any(m => m.Parts.Any(p => p is ChatImage)))
        {
            // The image parts in the order the template renders them, each as the blocks the family's encoder makes of it.
            images = [.. transcript.Messages.SelectMany(m => m.Parts).OfType<ChatImage>()];
            var vision = Vision ?? throw new InvalidOperationException($"The transcript holds {images.Count} image(s), and the encoder reads text only: set "
                + "ChatTranscriptEncoder.Vision (TuningVision.Create with a vision-language model) to train on images.");
            layouts = [.. images.Select(i => vision.Blocks(i))];
            text = vision.Expand(text, layouts, Tokenizer);
            spans = Spans(text);
        }

        // The parts of other kinds, each kind's in the order the template renders them, expanded by their source.
        List<(ISoftPartSource Source, List<ChatPart> Parts, List<IReadOnlyList<ImageTokenLayout>> Layouts)>? soft = null;
        foreach (var part in transcript.Messages.SelectMany(m => m.Parts))
        {
            if (part is ChatText or ChatImage || soft?.Any(s => s.Source.Kind == part.Kind) == true)
            {
                continue;
            }

            var source = _parts.FirstOrDefault(s => s.Kind == part.Kind) ?? throw new InvalidOperationException(
                $"The transcript holds a '{part.Kind}' part ({part}), and the encoder has no source for that kind: set ChatTranscriptEncoder.Parts "
                + "(an ISoftPartSource per kind, such as Idrak.Audio's ChatAudioParts for audio) to train on it.");
            var own = transcript.Messages.SelectMany(m => m.Parts).Where(p => p.Kind == source.Kind).ToList();
            var ownLayouts = own.Select(source.Blocks).ToList();
            text = source.Format.Expand(text, ownLayouts, Tokenizer);
            spans = Spans(text);
            (soft ??= []).Add((source, own, ownLayouts));
        }

        int firstTrained = Math.Max(0, spans.Count - answerTurns);
        var tokens = new List<int>();
        var trained = new List<bool>();
        void Add(int from, int to, bool train)
        {
            if (to > from)
            {
                int count = tokens.Count;
                if (Tokenizer is IPretrainedTokenizer pretrained)
                {
                    pretrained.EncodeRange(text, from, to - from, tokens);         // no copy of the span's text
                }
                else
                {
                    tokens.AddRange(Tokenizer.Encode(text[from..to]));
                }

                trained.AddRange(Enumerable.Repeat(train, tokens.Count - count));
            }
        }

        int position = 0;
        for (int s = 0; s < spans.Count; s++)
        {
            var (start, end) = spans[s];
            Add(position, start, false);
            Add(start, end, s >= firstTrained);
            position = end;
        }

        Add(position, text.Length, false);
        var parts = new List<TrainingPart>();
        foreach (var (source, own, ownLayouts) in soft ?? [])
        {
            parts.AddRange(LocateParts(tokens, trained, source, own, ownLayouts));
        }

        return (tokens, trained, images is null ? [] : Locate(tokens, trained, images, layouts!), parts, spans.Count);
    }

    // Where each part's blocks of soft tokens lie in the tokens (as Locate finds images), in the parts' order; a part next
    // to a trained token is refused: parts are read, not trained on.
    private static List<TrainingPart> LocateParts(List<int> tokens, List<bool> trained, ISoftPartSource source, List<ChatPart> parts, List<IReadOnlyList<ImageTokenLayout>> layouts)
    {
        int softToken = source.Format.ImageToken;
        var located = new List<TrainingPart>(parts.Count);
        int at = 0;
        for (int k = 0; k < parts.Count; k++)
        {
            var blocks = new List<(int Position, int Tokens)>(layouts[k].Count);
            foreach (var layout in layouts[k])
            {
                while (at < tokens.Count && tokens[at] != softToken)
                {
                    at++;
                }

                int end = at;
                while (end < tokens.Count && end < at + layout.Tokens && tokens[end] == softToken)
                {
                    end++;
                }

                if (end - at != layout.Tokens)
                {
                    throw new InvalidDataException($"'{source.Kind}' part {k + 1} of the transcript: {end - at} soft tokens at {at} where its block takes {layout.Tokens} "
                        + "(the text holds the soft token of its own, or the tokenizer does not keep it whole).");
                }

                for (int i = Math.Max(0, at - 1); i < Math.Min(tokens.Count, end + 1); i++)
                {
                    if (trained[i])
                    {
                        throw new InvalidDataException($"'{source.Kind}' part {k + 1} of the transcript lies in an assistant turn: parts are read in the prompt, never trained on.");
                    }
                }

                blocks.Add((at, layout.Tokens));
                at = end;
            }

            located.Add(new TrainingPart(parts[k], blocks));
        }

        if (tokens.IndexOf(softToken, at) is var extra and >= 0)
        {
            throw new InvalidDataException($"The transcript's text holds '{source.Kind}' soft tokens past its {parts.Count} part(s) (at {extra}).");
        }

        return located;
    }

    // Where each image's blocks lie in the tokens: the image tokens, in order, taken block by block by their counts (so
    // blocks that touch are told apart). An image next to a trained token is refused: images are read, not trained on.
    private List<TrainingImage> Locate(List<int> tokens, List<bool> trained, List<ChatImage> images, List<IReadOnlyList<ImageTokenLayout>> layouts)
    {
        var vision = Vision!;
        int imageToken = vision.Format.ImageToken;
        var located = new List<TrainingImage>(images.Count);
        int at = 0;
        for (int k = 0; k < images.Count; k++)
        {
            var blocks = new List<(int Position, int Tokens)>(layouts[k].Count);
            var axes = new List<float[]?>(layouts[k].Count);
            foreach (var layout in layouts[k])
            {
                while (at < tokens.Count && tokens[at] != imageToken)
                {
                    at++;
                }

                int end = at;
                while (end < tokens.Count && end < at + layout.Tokens && tokens[end] == imageToken)
                {
                    end++;
                }

                if (end - at != layout.Tokens)
                {
                    throw new InvalidDataException($"Image {k + 1} of the transcript: {end - at} image tokens at {at} where its block takes {layout.Tokens} "
                        + "(the text holds image tokens of its own, or the tokenizer does not keep the image token whole).");
                }

                for (int i = Math.Max(0, at - 1); i < Math.Min(tokens.Count, end + 1); i++)
                {
                    if (trained[i])
                    {
                        throw new InvalidDataException($"Image {k + 1} of the transcript lies in an assistant turn: images are read in the prompt, never trained on.");
                    }
                }

                blocks.Add((at, layout.Tokens));
                axes.Add(vision.Encoder.AxisPositions(layout));
                at = end;
            }

            located.Add(new TrainingImage(images[k], vision.Images, blocks) { AxisPositions = axes.Any(a => a is not null) ? axes : null });
        }

        if (tokens.IndexOf(imageToken, at) is var extra and >= 0)
        {
            throw new InvalidDataException($"The transcript's text holds image tokens past its {images.Count} image(s) (at {extra}).");
        }

        return located;
    }

    /// <summary>
    /// Plain text for continued pre-training or domain adaptation: the template's BOS token, the text and its EOS token,
    /// every token trained. Longer texts become several sequences of at most <paramref name="maxLength"/> + 1 tokens that
    /// together predict every token once.
    /// </summary>
    public IEnumerable<TrainingSequence> EncodeText(string text, int maxLength)
    {
        var ids = Tokenizer.Encode(Template.BosToken + text + Template.EosToken);
        for (int start = 0; start + 1 < ids.Count; start += maxLength)
        {
            int count = Math.Min(maxLength + 1, ids.Count - start);
            if (count < 2)
            {
                yield break;
            }

            var tokens = new int[count];
            var trained = new bool[count];
            for (int i = 0; i < count; i++)
            {
                tokens[i] = ids[start + i];
            }

            Array.Fill(trained, true);
            yield return new TrainingSequence(tokens, trained);
        }
    }

    /// <summary>
    /// <see cref="EncodeRow"/> for many rows on all cores (rendering the chat template and tokenizing is most of the time
    /// data preparation takes), each row with its sequences, in the rows' order. The rows are read on the calling thread.
    /// </summary>
    public IEnumerable<(JsonObject Row, IReadOnlyList<TrainingSequence> Sequences)> EncodeRows(IEnumerable<JsonObject> rows, int maxLength,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(rows);
        return rows.AsParallel().AsOrdered().WithCancellation(cancellationToken)
            .Select(row => (row, (IReadOnlyList<TrainingSequence>)[.. EncodeRow(row, maxLength)]));
    }

    /// <summary>
    /// A preference row as a <see cref="PreferencePair"/>: the chosen and the rejected answer after the same prompt, read by
    /// <see cref="ChatRows.Preference"/> (TRL's layouts: <c>{"prompt", "chosen", "rejected"}</c> with message lists or
    /// strings, or chosen and rejected as whole conversations that share their prompt; "tools" and "enable_thinking" as in
    /// <see cref="ChatTranscript.FromJson"/>). Each sequence trains only its answer's assistant turns, not assistant turns
    /// inside the prompt. A pair that is too long is shortened in its last user message (see <see cref="ShortenToFit"/>),
    /// else cut at the end. Null when the row is not a preference pair or an answer has no trainable token left.
    /// </summary>
    public PreferencePair? EncodePreference(JsonObject row, int maxLength)
    {
        ArgumentNullException.ThrowIfNull(row);
        if (ChatRows.Preference(row) is not { } pair)
        {
            return null;
        }

        TrainingSequence? Answer(string name)
        {
            var messages = new JsonArray();
            foreach (var message in pair["prompt"]!.AsArray().Concat(pair[name]!.AsArray()))
            {
                messages.Add(message!.DeepClone());
            }

            var json = new JsonObject { ["messages"] = messages, ["tools"] = pair["tools"]?.DeepClone() };
            if ((row["enable_thinking"] ?? row["think"]) is JsonValue think)
            {
                json["enable_thinking"] = think.DeepClone();
            }

            var transcript = ChatTranscript.FromJson(json);
            int turns = transcript.Messages.Count(m => m.Role == "assistant") - pair["prompt"]!.AsArray().Count(m => (string?)m?["role"] == "assistant");
            if (ShortenToFit && Fit(transcript, maxLength) is { } fitted)
            {
                transcript = fitted;
            }

            var (tokens, trained, images, parts, _) = Tokens(transcript, turns);
            return Sequence(tokens, trained, images, parts, maxLength);
        }

        return Answer("chosen") is { } chosen && Answer("rejected") is { } rejected ? new PreferencePair(chosen, rejected) : null;
    }

    /// <summary>
    /// <see cref="EncodePreference"/> for many rows on all cores, each row with its pair (null when it gives none), in the
    /// rows' order. The rows are read on the calling thread.
    /// </summary>
    public IEnumerable<(JsonObject Row, PreferencePair? Pair)> EncodePreferences(IEnumerable<JsonObject> rows, int maxLength,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(rows);
        return rows.AsParallel().AsOrdered().WithCancellation(cancellationToken).Select(row => (row, EncodePreference(row, maxLength)));
    }

    /// <summary>
    /// A dataset row as training sequences: a conversation (<c>{"messages": [...], "tools": [...]}</c>, see
    /// <see cref="ChatTranscript.FromJson"/>) trains the assistant's turns; a text row (<c>{"text": ...}</c>) trains every
    /// token. Other rows give nothing.
    /// </summary>
    public IEnumerable<TrainingSequence> EncodeRow(JsonObject row, int maxLength)
    {
        if (row.ContainsKey("messages") || row.ContainsKey("conversations"))
        {
            if (Encode(ChatTranscript.FromJson(row), maxLength) is { } sequence)
            {
                yield return sequence;
            }
        }
        else if (row["text"] is JsonValue v && v.TryGetValue<string>(out var text) && text.Length > 0)
        {
            foreach (var sequence in EncodeText(text, maxLength))
            {
                yield return sequence;
            }
        }
    }
}

/// <summary>Settings for <see cref="FineTuner"/>.</summary>
public sealed record FineTuningOptions
{
    /// <summary>LoRA rank (the adapters' inner size).</summary>
    public int Rank { get; init; } = 16;

    /// <summary>LoRA alpha: the adapters' output is scaled by alpha / rank.</summary>
    public float Alpha { get; init; } = 32f;

    /// <summary>
    /// DoRA adapters (<see cref="DoraAdapter"/>: LoRA on the weight's direction plus a trainable magnitude per output)
    /// instead of LoRA. They train with the same settings, saved in the PEFT format with <c>use_dora</c>; their products run
    /// without the fused LoRA kernels (and the norm of each adapted weight is computed every step), so steps are slower.
    /// </summary>
    public bool Dora { get; init; }

    /// <summary>
    /// Layers that get adapters, by name: q, k, v, o (attention), gate, up, down (feed-forward), head. Null (the default):
    /// the model's own (<see cref="PretrainedModel.NetworkParts"/>' adapter targets: q, k, v, o, gate, up and down for the
    /// library's decoders).
    /// </summary>
    public IReadOnlyList<string>? Targets { get; init; }

    /// <summary>Peak learning rate (reached after the warm-up, then cosine decay to <see cref="MinLearningRate"/>).</summary>
    public float LearningRate { get; init; } = 2e-4f;

    /// <summary>Learning rate at the end of training.</summary>
    public float MinLearningRate { get; init; }

    /// <summary>Fraction of the optimizer steps spent warming up linearly.</summary>
    public float WarmupFraction { get; init; } = 0.03f;

    /// <summary>AdamW weight decay on the adapters.</summary>
    public float WeightDecay { get; init; }

    /// <summary>Passes over the training data.</summary>
    public int Epochs { get; init; } = 1;

    /// <summary>Longest sequence in tokens (longer transcripts are cut).</summary>
    public int MaxLength { get; init; } = 2048;

    /// <summary>
    /// Token budget of one batch: with <see cref="Packing"/>, the positions of each batch (rows of this many positions,
    /// or of the longest sequence when it is longer); otherwise sequences of similar length padded to the longest.
    /// </summary>
    public int BatchTokens { get; init; } = 4096;

    /// <summary>
    /// Packs several sequences into each row of <see cref="BatchTokens"/> positions (each position attends only within
    /// its own sequence), so batches carry no padding between sequences and every step has the same shape. Used when the
    /// model and device support it (<see cref="PackedSequences.Supports"/>); otherwise batches are padded.
    /// </summary>
    public bool Packing { get; init; } = true;

    /// <summary>
    /// On a device that records graphs (CUDA, Vulkan), with packed batches and no gradient accumulation: records one
    /// training step's forward and backward pass as a graph (after two ordinary steps) and replays it for every later
    /// batch of the same shape, with the batch's tokens copied into the recorded buffers. The thousands of operations of a
    /// step then cost one launch of host work instead of one each. Falls back to ordinary steps when recording fails.
    /// </summary>
    public bool CudaGraphs { get; init; } = true;

    /// <summary>
    /// Forward products of the frozen base weights on FP8 (e4m3) tensor cores (compute capability 8.9 and newer): each
    /// weight is quantized once per output column, activations per row as they are read; the backward pass stays in
    /// bfloat16. Checked first: the loss of a sample of the training data with FP8 must be within
    /// <see cref="Float8Tolerance"/> of the bfloat16 loss, otherwise training continues in bfloat16.
    /// </summary>
    public bool Float8 { get; init; }

    /// <summary>Largest relative difference between the FP8 and the bfloat16 loss that <see cref="Float8"/> accepts.</summary>
    public float Float8Tolerance { get; init; } = 0.02f;

    /// <summary>Batches whose gradients are added before each optimizer step.</summary>
    public int GradientAccumulation { get; init; } = 1;

    /// <summary>Gradients are scaled down to at most this norm (0: no clipping).</summary>
    public float MaxGradientNorm { get; init; } = 1f;

    /// <summary>
    /// Optimizer steps in a row whose loss is not a finite number (NaN or infinite) that are skipped (their gradients
    /// dropped, the parameters left as they were) before the run stops with an error naming the step; 0 stops at the first.
    /// </summary>
    public int NonFiniteSteps { get; init; } = 8;

    /// <summary>Rows of the output head computed at once by the loss (see <see cref="Losses.TokenCrossEntropy"/>).</summary>
    public int LossChunkRows { get; init; } = 512;

    /// <summary>Evaluate every this many optimizer steps (0: after each epoch).</summary>
    public int EvaluateEvery { get; init; }

    /// <summary>Save the adapters every this many optimizer steps (0: only at the end).</summary>
    public int SaveEvery { get; init; }

    /// <summary>
    /// A checkpoint a stopped cluster run wrote to go on from (plan 23, C8b: a run steered from outside, a dashboard's stop,
    /// writes <c>checkpoint-STEP</c> in the output folder with the adapters, the optimizer's state and the step before every
    /// rank stops): its adapters and optimizer state are loaded and the steps it finished are skipped, so the run ends as the
    /// uninterrupted one. The run's data, options and seed must be the stopped run's. Null: from the start.
    /// </summary>
    public string? ResumeFrom { get; init; }

    /// <summary>
    /// Recompute each decoder block's activations in the backward pass instead of storing them (about 20× less
    /// activation memory for about a third more compute): needed for long sequences, large batches or large models on
    /// small devices. Null (the default): off while the activations fit, turned on for the rest of training the first time
    /// a step runs out of device memory (that step is run again); on when <see cref="ComputeResources.OffloadToHostMemory"/>
    /// is set (a full device then spills to system memory instead of failing, which is slower than recomputing).
    /// </summary>
    public bool? Checkpointing { get; init; }

    /// <summary>
    /// Recompute each block's feed-forward activation in the backward pass instead of keeping it (about a fifth less
    /// activation memory for one element-wise kernel per block; see <see cref="ActivationMemory"/>). Null (the default):
    /// off, and the first thing turned on when a step runs out of device memory, before <see cref="Checkpointing"/>. With
    /// all three of these null, the first setting that fits is timed over a step against checkpointing, and the faster one
    /// kept for the rest of training.
    /// </summary>
    public bool? RecomputeFeedForward { get; init; }

    /// <summary>
    /// Hold the activations the backward pass reads as bfloat16 between the passes (half their memory, one pack and one
    /// unpack pass each; the inputs of tensor-core products lose nothing, see <see cref="ActivationMemory.CompressToBFloat16"/>).
    /// Null (the default): off, and turned on when a step runs out of device memory with <see cref="RecomputeFeedForward"/>
    /// on already, before <see cref="Checkpointing"/>.
    /// </summary>
    public bool? BFloat16Activations { get; init; }

    /// <summary>
    /// Before the first step of each batch shape (rows × positions), estimate from the model's dimensions the device memory
    /// each rung of the activation ladder needs (activations held, recomputed feed-forward, also bfloat16, checkpointing;
    /// only the rungs the three settings above leave automatic) and start at the first whose estimate fits the memory the
    /// device reports available (within its configured limit), instead of finding it by running out of memory. A step that
    /// still runs out steps down as before and says the estimate was short. True by default; false starts from the settings
    /// as given, the out-of-memory ladder alone. IDRAK_ACTIVATION_ESTIMATE=0 sets the default to false.
    /// </summary>
    public bool EstimateActivations { get; init; } = ActivationEstimate.FromEnvironment();

    /// <summary>
    /// Run AdamW's update on the CPU, with its state (two moments per trained value) in system memory: only gradients and
    /// weights cross PCIe each step (see <see cref="Optimizers.HostOptimizer"/>). For training many parameters (large
    /// adapters, or unfrozen layers) on a GPU too small for the optimizer state; slower per step. Recorded graphs are not
    /// used with it.
    /// </summary>
    public bool HostOptimizer { get; init; }

    /// <summary>Seed for the adapters' initial values and the batch order.</summary>
    public int Seed { get; init; }

    /// <summary>
    /// For models with experts: the weight of their load-balancing loss (<see cref="MixtureOfExperts.LoadBalancingLoss"/>,
    /// averaged over the layers) added to the training loss, which pushes the router towards using the experts evenly
    /// (transformers' router_aux_loss_coef; 0.001 to 0.02 is usual). 0 (the default): none. Blocks recomputed by
    /// <see cref="Checkpointing"/> contribute none; the reported loss includes it.
    /// </summary>
    public float LoadBalancingWeight { get; init; }

    /// <summary>
    /// Creates the optimizer from the trainable parameters (the adapters), e.g. <c>ps =&gt; new Sgd(ps, 0.05f, momentum: 0.9f)</c>
    /// or one by name, <see cref="OptimizerKinds.Factory"/>. Null (the default): AdamW with <see cref="LearningRate"/> and
    /// <see cref="WeightDecay"/>. The rate the optimizer starts with is the peak the schedule works from. With
    /// <see cref="HostOptimizer"/> it receives the CPU copies of the parameters (one each). Recorded steps
    /// (<see cref="CudaGraphs"/>) record only the forward and backward pass, so they work with any optimizer that keeps the
    /// gradient buffers between steps (every built-in one except <see cref="Optimizers.HostOptimizer"/>); with one that
    /// does not, training runs ordinary steps.
    /// </summary>
    public Func<IReadOnlyList<Tensor>, Optimizer>? Optimizer { get; init; }

    /// <summary>
    /// Creates the learning-rate schedule from the optimizer and the number of optimizer steps planned; it is stepped once
    /// per optimizer step (by name: <see cref="LearningRateSchedules.Factory"/>). Null (the default): a linear warm-up over
    /// <see cref="WarmupFraction"/> of the steps, then cosine decay to <see cref="MinLearningRate"/>.
    /// </summary>
    public Func<Optimizer, int, ILearningRateScheduler>? Scheduler { get; init; }

    /// <summary>
    /// The loss of a batch from its trained tokens' log-probabilities (built-in: <see cref="FineTuningLosses"/>; DPO, ORPO
    /// and SimPO need <see cref="PreferencePair"/> data). Null (the default): the mean token cross-entropy per trained token,
    /// computed in one fused pass that steps can record as a graph; with a loss of one's own, steps run as ordinary ones
    /// and the output head runs twice over the trained rows (once for the log-probabilities, once for their gradients).
    /// Evaluation losses are this loss over the whole evaluation set. With a <see cref="Teacher"/> and no loss,
    /// <see cref="FineTuningLosses.Distillation"/> with its defaults.
    /// </summary>
    public IFineTuningLoss? Loss { get; init; }

    /// <summary>
    /// The teacher for knowledge distillation (<see cref="DistillationTeachers.FromModel(PretrainedModel, int, int)"/> or
    /// <see cref="DistillationTeachers.FromFile"/>), whose distributions a loss reads through
    /// <see cref="FineTuningLossInput.TeacherDivergence(float)"/> (see <see cref="FineTuningLosses.Distillation"/>). It must share the
    /// student's vocabulary (<see cref="IDistillationTeacher.Check"/>, which training runs first). Null: no teacher. Not
    /// disposed by training.
    /// </summary>
    public IDistillationTeacher? Teacher { get; init; }

    /// <summary>
    /// The image side for sequences with images (<see cref="TrainingSequence.Images"/>, a vision-language model): the
    /// family's encoder, the preparation, the vision parts that train beside the adapters and the feature cache
    /// (<see cref="TuningVision.Create"/>; the one the sequences were encoded with). Null: made from the model when the
    /// sequences hold images (the language model trains alone, a memory cache), and disposed after training. Not disposed by
    /// training when given. Steps with images pack like the others where the device's packed attention serves, and also
    /// where every sequence holds images attended by a rule that is not causal and the device's span kernels take the head
    /// size (<c>PackedSequences.SupportsImages</c>: every position then attends by its key ranges); else they run padded.
    /// While no vision part trains they are recorded as graphs, their features copied in (one graph per rows and length,
    /// recorded for the most image tokens of any batch, so it serves them all); the out-of-memory ladder covers them as any
    /// step. The
    /// trained vision parts are saved with the adapters (PEFT's <c>modules_to_save</c>) and the preparation beside them
    /// (<see cref="TuningImages.FileName"/>).
    /// </summary>
    public TuningVision? Vision { get; init; }

    /// <summary>
    /// The sources of the sequences' soft parts other than images (an audio part's: <see cref="ChatTranscriptEncoder.Parts"/>),
    /// one per kind: each step encodes the batch's parts through them (each distinct part once) and lays their features
    /// out at the soft tokens, as images'. A trainable source (<see cref="ITrainableSoftParts"/>) is read with gradients
    /// and its trained parameters (<see cref="ITrainableSoftParts.Parameters"/>) join the optimizer's. Steps with parts
    /// run as ordinary steps (not recorded as graphs). Empty (the default): a sequence with such parts is refused.
    /// </summary>
    public IReadOnlyList<ISoftPartSource> Parts { get; init; } = [];

    /// <summary>
    /// Answers generated on held-out conversations and scored by a metric (<see cref="TuningAnswerScorer"/>: character or
    /// word error rate), every <see cref="TuningAnswerScorer.Every"/> steps (at the end of each epoch when 0), with their
    /// images through <see cref="Vision"/>; each report comes with that step's <see cref="FineTuningProgress.Answers"/>.
    /// Null: none.
    /// </summary>
    public TuningAnswerScorer? Answers { get; init; }

    /// <summary>
    /// Data-parallel fine-tuning (plan 23): this rank's collectives, or null (the default) to train in this process alone,
    /// exactly as without it. Every rank runs <see cref="FineTuner.Train(PretrainedModel, IReadOnlyList{TrainingSequence}, IReadOnlyList{TrainingSequence}?, FineTuningOptions, string?, IProgress{FineTuningProgress}?, CancellationToken, Action{string}?)"/>
    /// with its own model on its rank's device and the same sequences and seed: each step covers
    /// <see cref="GradientAccumulation"/> batches per rank, rank r taking the step's batches k·W + r (or a share by the
    /// ranks' measured speeds, <see cref="ClusterOptions.RebalanceSteps"/>), so W ranks train as one process with W times
    /// the accumulation; the gradients are summed after the last batch's backward pass, weighted by each rank's trained
    /// tokens (a loss of one's own sees the whole step's token and sequence counts); the reported and evaluated losses
    /// are over every rank's; rank 0 writes the adapters and checkpoints; dropout masks differ by rank.
    /// </summary>
    public ICollectives? Cluster { get; init; }

    /// <summary>The cluster settings a <see cref="Cluster"/> run uses, or null: <see cref="Abstraction.Cluster.ClusterOptions.FromEnvironment"/>.</summary>
    public ClusterOptions? ClusterOptions { get; init; }
}

/// <summary>What <see cref="FineTuner.Profile"/> measured.</summary>
/// <param name="Kernels">GPU time per kernel over the profiled steps, most first (empty on the CPU).</param>
/// <param name="ProfiledSteps">Steps profiled.</param>
/// <param name="ProfiledTokens">Tokens in the profiled steps.</param>
/// <param name="SecondsPerStep">Wall time per step, measured without the profiler.</param>
/// <param name="TokensPerStep">Tokens per step in the timed steps.</param>
public sealed record FineTuningProfile(IReadOnlyList<GpuProfileEntry> Kernels, int ProfiledSteps, long ProfiledTokens, double SecondsPerStep, double TokensPerStep)
{
    /// <summary>Tokens per second without the profiler.</summary>
    public double TokensPerSecond => SecondsPerStep > 0 ? TokensPerStep / SecondsPerStep : 0;

    /// <summary>GPU time per profiled step, in milliseconds.</summary>
    public double GpuMillisecondsPerStep => Kernels.Sum(k => k.Milliseconds) / ProfiledSteps;

    /// <summary>
    /// Kernel time grouped into what it does (matrix products, attention, normalization, activations, loss, optimizer,
    /// copies and conversions, other), per step, most first.
    /// </summary>
    public IReadOnlyList<(string Group, double Milliseconds, long Calls)> Groups => [.. Kernels
        .GroupBy(k => GroupOf(k.Name))
        .Select(g => (g.Key, g.Sum(k => k.Milliseconds) / ProfiledSteps, g.Sum(k => k.Calls) / ProfiledSteps))
        .OrderByDescending(g => g.Item2)];

    private static string GroupOf(string kernel)
    {
        // Ignoring case matches as lowercasing does for ASCII names (kernel names are); other names are lowercased.
        bool ascii = System.Text.Ascii.IsValid(kernel);
        string k = ascii ? kernel : kernel.ToLowerInvariant();
        var comparison = ascii ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        bool Has(string part) => k.Contains(part, comparison);
        return k switch
        {
            _ when Has("gemm") || Has("matmul") || Has("gemv") => "matrix products",
            _ when Has("attention") || Has("flash") || Has("softmax_rows") && !Has("cross") => "attention",
            _ when Has("cross_entropy") || Has("crossentropy") => "loss (output layer softmax)",
            _ when Has("norm") => "normalization",
            _ when Has("rope") || Has("rotary") => "rotary embedding",
            _ when Has("silu") || Has("gelu") || Has("gated") || Has("act") => "activations",
            _ when Has("adam") || Has("sumsq") || Has("clip") => "optimizer and clipping",
            _ when Has("transpose") || Has("copy") || Has("convert") || Has("bf16") || Has("fill") || Has("quant") => "copies, conversions, zeroing",
            _ when Has("add") || Has("mul") || Has("axpy") || Has("scale") || Has("sum") => "element-wise and reductions",
            _ => "other",
        };
    }
}

/// <summary>Where a fine-tuning run is.</summary>
/// <param name="Step">Optimizer steps done.</param>
/// <param name="TotalSteps">Optimizer steps planned.</param>
/// <param name="Epoch">The current epoch (from 1).</param>
/// <param name="Loss">The last step's loss: the mean loss per trained token, or <see cref="FineTuningOptions.Loss"/> when set.</param>
/// <param name="LearningRate">The learning rate of the last step.</param>
/// <param name="TokensPerSecond">Tokens (including prompts) processed per second over the last step.</param>
/// <param name="EvaluationLoss">The latest evaluation loss, when one was computed at this step.</param>
public sealed record FineTuningProgress(int Step, int TotalSteps, int Epoch, float Loss, float LearningRate, double TokensPerSecond, float? EvaluationLoss)
{
    /// <summary>The generated answers' score, when <see cref="FineTuningOptions.Answers"/> scored them at this step.</summary>
    public TuningAnswerReport? Answers { get; init; }
}

/// <summary>
/// Fine-tunes a pretrained model with LoRA adapters: the base weights stay as loaded (int4, int8 or bfloat16 for QLoRA,
/// or float32) and frozen; only the adapters train, on the assistant tokens of chat transcripts (see
/// <see cref="ChatTranscriptEncoder"/>), or on preference pairs. By default AdamW with a linear warm-up and cosine decay,
/// gradient accumulation and clipping, and a token loss that never stores the full logits
/// (<see cref="Losses.TokenCrossEntropy"/>); the optimizer, the schedule and the loss can be replaced
/// (<see cref="FineTuningOptions.Optimizer"/>, <see cref="FineTuningOptions.Scheduler"/>, <see cref="FineTuningOptions.Loss"/>).
/// </summary>
public static class FineTuner
{
    /// <summary>
    /// Evens out how often each answer is trained when the answers come from a short list (labels, yes / no, one of a few
    /// choices): sequences are grouped by their trained tokens, and a group smaller than the largest is repeated, at most
    /// <paramref name="maxRepeats"/> times in all, towards the largest one's size. A rare answer then weighs more in the
    /// loss (its recall rises, usually at some cost to its precision). With more than <paramref name="maxAnswers"/>
    /// distinct answers (free text) the sequences are returned as they are. Deterministic for a <paramref name="seed"/>.
    /// </summary>
    /// <returns>The sequences (repeats included, all in a seeded random order) and each answer's count before and after, by its tokens.</returns>
    public static (List<TrainingSequence> Sequences, IReadOnlyList<(int[] Answer, int Before, int After)> Answers) BalanceAnswers(
        IReadOnlyList<TrainingSequence> sequences, int maxRepeats = 4, int maxAnswers = 64, int seed = 0)
    {
        ArgumentNullException.ThrowIfNull(sequences);
        static int[] Answer(TrainingSequence s) => [.. s.Tokens.Where((_, i) => s.Trained[i])];
        var groups = sequences.GroupBy(s => string.Join(',', Answer(s))).Select(g => g.ToList()).ToList();
        var result = new List<TrainingSequence>(sequences);
        if (groups.Count < 2 || groups.Count > maxAnswers || maxRepeats <= 1)
        {
            return (result, [.. groups.Select(g => (Answer(g[0]), g.Count, g.Count))]);
        }

        int largest = groups.Max(g => g.Count);
        var random = new Random(seed);
        var answers = new List<(int[] Answer, int Before, int After)>();
        foreach (var group in groups)
        {
            // Target: the largest group's size, at most maxRepeats times this one's: whole copies, then a seeded sample.
            int target = (int)Math.Min(largest, (long)group.Count * maxRepeats);
            for (int added = group.Count; added < target;)
            {
                int take = Math.Min(group.Count, target - added);
                result.AddRange(take == group.Count ? group : group.OrderBy(_ => random.Next()).Take(take));
                added += take;
            }

            answers.Add((Answer(group[0]), group.Count, Math.Max(group.Count, target)));
        }

        random.Shuffle(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(result));   // repeats spread among the originals
        return (result, answers);
    }

    /// <summary>
    /// Adds adapters to <paramref name="model"/> (unless it already has some) and trains them on <paramref name="train"/>;
    /// evaluates on <paramref name="evaluation"/> when given, and writes the adapters to <paramref name="outputFolder"/>
    /// (PEFT format, see <see cref="PretrainedModel.SaveAdapter"/>) when given. Returns the evaluation losses.
    /// </summary>
    /// <param name="model">The model (adapters are added unless it has some).</param>
    /// <param name="train">Training sequences.</param>
    /// <param name="evaluation">Sequences to evaluate on, or null.</param>
    /// <param name="options">Settings.</param>
    /// <param name="outputFolder">Where to write the adapters (and checkpoints), or null.</param>
    /// <param name="progress">Receives one report per optimizer step.</param>
    /// <param name="cancellationToken">Stops between batches.</param>
    /// <param name="trace">Receives a line per batch (its shape and time) and per evaluation, as they finish.</param>
    public static IReadOnlyList<float> Train(PretrainedModel model, IReadOnlyList<TrainingSequence> train, IReadOnlyList<TrainingSequence>? evaluation,
        FineTuningOptions options, string? outputFolder = null, IProgress<FineTuningProgress>? progress = null, CancellationToken cancellationToken = default,
        Action<string>? trace = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(train);
        ArgumentNullException.ThrowIfNull(options);
        if (train.Count == 0)
        {
            throw new ArgumentException("There is nothing to train on.", nameof(train));
        }

        return TrainCore(model, train, evaluation, pairs: false, Custom(model, options, pairs: false), options, outputFolder, progress, cancellationToken, trace);
    }

    /// <summary>
    /// Adds adapters to <paramref name="model"/> (unless it already has some) and trains them on preference pairs with a
    /// preference loss (<see cref="FineTuningOptions.Loss"/>: <see cref="FineTuningLosses.Dpo"/> when not set, or
    /// <see cref="FineTuningLosses.Orpo"/>, <see cref="FineTuningLosses.SimPo"/> or a loss of one's own that reads
    /// <see cref="FineTuningLossInput.Pairs"/>). Each batch holds whole pairs, padded to the longest answer (no packing),
    /// so the steps run as ordinary ones. DPO's reference is the model with its adapters disabled (see
    /// <see cref="FineTuningLossInput.ReferenceLogProbabilities"/>). Evaluates on <paramref name="evaluation"/> (the loss
    /// over all its pairs) and writes the adapters as <see cref="Train(PretrainedModel, IReadOnlyList{TrainingSequence}, IReadOnlyList{TrainingSequence}?, FineTuningOptions, string?, IProgress{FineTuningProgress}?, CancellationToken, Action{string}?)"/>
    /// does. Returns the evaluation losses.
    /// </summary>
    /// <param name="model">The model (adapters are added unless it has some).</param>
    /// <param name="train">Training pairs (see <see cref="ChatTranscriptEncoder.EncodePreference"/>).</param>
    /// <param name="evaluation">Pairs to evaluate on, or null.</param>
    /// <param name="options">Settings.</param>
    /// <param name="outputFolder">Where to write the adapters (and checkpoints), or null.</param>
    /// <param name="progress">Receives one report per optimizer step.</param>
    /// <param name="cancellationToken">Stops between batches.</param>
    /// <param name="trace">Receives a line per batch (its shape and time) and per evaluation, as they finish.</param>
    public static IReadOnlyList<float> Train(PretrainedModel model, IReadOnlyList<PreferencePair> train, IReadOnlyList<PreferencePair>? evaluation,
        FineTuningOptions options, string? outputFolder = null, IProgress<FineTuningProgress>? progress = null, CancellationToken cancellationToken = default,
        Action<string>? trace = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(train);
        ArgumentNullException.ThrowIfNull(options);
        if (train.Count == 0)
        {
            throw new ArgumentException("There is nothing to train on.", nameof(train));
        }

        var custom = Custom(model, options with { Loss = options.Loss ?? FineTuningLosses.Dpo() }, pairs: true);
        return TrainCore(model, Flatten(train), evaluation is null ? null : Flatten(evaluation), pairs: true, custom, options, outputFolder, progress,
            cancellationToken, trace);
    }

    // The loss of one's own (or distillation, with a teacher and no loss), or null for the fused token cross-entropy. A
    // teacher is checked against the model first; a loss that needs pairs or a teacher it is not given is refused here.
    internal static CustomLoss? Custom(PretrainedModel model, FineTuningOptions options, bool pairs)
    {
        options.Teacher?.Check(model.Tokenizer ?? throw new InvalidOperationException("The student has no tokenizer: its vocabulary cannot be compared with the teacher's."));
        var loss = options.Loss ?? (options.Teacher is not null ? FineTuningLosses.Distillation() : null);
        return loss is null ? null : new CustomLoss(loss, pairs, options.Teacher);
    }

    // Pair k as sequences 2k (chosen) and 2k + 1 (rejected).
    internal static List<TrainingSequence> Flatten(IReadOnlyList<PreferencePair> pairs) =>
        [.. pairs.SelectMany(p => new[] { p.Chosen ?? throw new ArgumentException("A pair has no chosen answer."), p.Rejected ?? throw new ArgumentException("A pair has no rejected answer.") })];

    // A run over a session (FineTuningSession): the session holds what the run keeps across its steps (the adapters, the
    // optimizer and its schedule, FP8 copies, the cluster's sync, the recorded step graphs); the run makes each epoch's
    // batches, steps the session through them, evaluates, scores answers and saves.
    private static IReadOnlyList<float> TrainCore(PretrainedModel model, IReadOnlyList<TrainingSequence> train, IReadOnlyList<TrainingSequence>? evaluation,
        bool pairs, CustomLoss? custom, FineTuningOptions options, string? outputFolder, IProgress<FineTuningProgress>? progress, CancellationToken cancellationToken,
        Action<string>? trace)
    {
        using var session = new FineTuningSession(model, options, train, evaluation, pairs, custom, trace);
        options = session.Options;
        var vision = options.Vision;
        var network = model.Network;
        var sync = session.Sync;
        var sharded = session.Sharded;
        long resumeStep = session.ResumedSteps;
        bool lead = sync is null || sync.Rank == 0;
        var random = new Random(options.Seed);
        var epochBatches = Enumerable.Range(0, options.Epochs)
            .Select(_ => pairs ? PairBatches(train, options.BatchTokens, random) : MakeBatches(model, train, options, random)).ToList();
        int accumulation = Math.Max(1, options.GradientAccumulation);
        int perStep = accumulation * (sync?.WorldSize ?? 1);
        int totalSteps = epochBatches.Sum(b => (b.Count + perStep - 1) / perStep);
        session.Begin(totalSteps, train, MostTrained(epochBatches.SelectMany(b => b), train), Largest(epochBatches.SelectMany(b => b)),
            MostImageTokens(epochBatches.SelectMany(b => b), train));
        var runner = session.Runner;
        var evaluations = new List<float>();
        int step = 0;
        // On a cluster a cancelled token asks the ranks to stop and they agree at a step boundary (Plan 27, R7): the work
        // inside a step runs to its end on every rank (its token is cancelled only when this rank gives up waiting).
        var agreement = session.Stop;
        agreement?.Reset();
        using var asks = agreement?.Watch(cancellationToken) ?? default;
        var inner = agreement?.GiveUpToken ?? cancellationToken;
        try
        {
            for (int epoch = 0; epoch < options.Epochs; epoch++)
            {
                var batches = epochBatches[epoch];
                for (int first = 0; first < batches.Count; first += perStep)
                {
                    if (agreement is null)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                    }

                    var global = batches.Skip(first).Take(perStep).ToList();
                    if (step < resumeStep)
                    {
                        session.Skip();                                        // a step the stopped run finished (ResumeFrom)
                        step = session.Steps;
                        continue;
                    }

                    bool present;
                    try
                    {
                        if (agreement is not null && agreement.Agree(step))
                        {
                            throw agreement.Stopped(step);                     // every rank at this same boundary
                        }

                        present = sync is null || sync.BeginStep(step);
                    }
                    catch (ClusterRunStoppedException) when (outputFolder is not null && step > resumeStep)
                    {
                        // Stopped at this step boundary (a dashboard's stop, plan 23 C9; written since C8b; a stop the ranks
                        // agreed on, Plan 27 R7): the last finished step's adapters, optimizer state (a sharded run's joined) and
                        // step, as a saving step writes them.
                        string stop = Path.Combine(outputFolder, $"checkpoint-{step}");
                        session.Checkpoint(stop);
                        trace?.Invoke($"stopped: checkpoint {stop}");
                        throw;
                    }

                    if (!present)
                    {
                        // This rank sits the step out (a simulated late joiner before it joins, every rank past a replay's last
                        // step): the ranks that take part share the step's batches; the schedule moves on alike.
                        session.Skip();
                        step = session.Steps;
                        continue;
                    }

                    bool writes = sync is null || sync.Rank == 0;            // the step's group's first rank writes and scores
                    var group = sync is null ? global : Share(global, train, sync, runner, trace);
                    var watch = Stopwatch.StartNew();
                    if (vision is not null)
                    {
                        // The next step's images made on workers while this one runs (its own were started the step before).
                        var coming = first + perStep < batches.Count ? batches.Skip(first + perStep).Take(perStep)
                            : epoch + 1 < options.Epochs ? epochBatches[epoch + 1].Take(perStep) : [];
                        vision.Prefetch(coming.SelectMany(b => b.Sequences.SelectMany(i => train[i].Images)));
                    }

                    var (loss, tokens, rate) = session.Run(group, null, inner,
                        b => $"step {step + 1}/{totalSteps}, batch {first + b + 1}/{batches.Count} of epoch {epoch + 1}");
                    step = session.Steps;
                    float? evaluationLoss = null;
                    bool lastOfEpoch = first + perStep >= batches.Count;
                    bool evaluate = evaluation is { Count: > 0 } && (options.EvaluateEvery > 0 ? step % options.EvaluateEvery == 0 : lastOfEpoch);
                    bool saving = outputFolder is not null && options.SaveEvery > 0 && step % options.SaveEvery == 0;
                    // A sharded run's weights whole on every rank while they are read (plan 23, C6; nothing to do otherwise).
                    using var whole = evaluate || saving || options.Answers is { } due && due.IsDue(step, lastOfEpoch) ? sharded?.Gather() : null;
                    if (evaluate && evaluation is not null)
                    {
                        evaluationLoss = custom is not null ? EvaluateCustom(model, evaluation, pairs, custom, options, trace)
                            : sync is null ? Evaluate(model, evaluation, options.BatchTokens, options.LossChunkRows, trace, vision)
                            : EvaluateAcross(model, evaluation, sync, options.BatchTokens, options.LossChunkRows, trace, vision);
                        evaluations.Add(evaluationLoss.Value);
                    }

                    // Generated answers are scored on rank 0 alone (every rank holds the same weights; the scores train nothing).
                    var answers = writes && options.Answers is { } scorer && scorer.IsDue(step, lastOfEpoch) ? scorer.Score(model, vision, cancellationToken: inner) : null;
                    if (answers is not null)
                    {
                        trace?.Invoke($"answers at step {step}: {answers} ({answers.MeanTokens:F0} tokens each, {answers.Duration.TotalSeconds:F1} s)");
                    }

                    if (saving && outputFolder is not null)
                    {
                        if (writes)
                        {
                            session.Save(Path.Combine(outputFolder, $"checkpoint-{step}"));
                        }

                        sync?.Barrier();                                       // rank 0 has written it
                    }

                    progress?.Report(new FineTuningProgress(step, totalSteps, epoch + 1, loss, rate, tokens / Math.Max(1e-9, watch.Elapsed.TotalSeconds), evaluationLoss) { Answers = answers });
                }
            }
        }
        catch (OperationCanceledException e) when (agreement.GaveUpError(e) is { } gaveUp)
        {
            throw gaveUp;                                                  // this rank left its step on its own, loudly
        }

        sharded?.End();                                                    // a sharded run's weights whole on every rank again
        network.Eval();
        if (outputFolder is not null)
        {
            if (lead)
            {
                session.Save(outputFolder);
            }

            sync?.Barrier();
        }

        return evaluations;
    }

    internal const string StopStateFile = "tune-state.json", StopOptimizerFile = "optimizer.safetensors";

    // A stop's optimizer state and step beside its adapters (plan 23, C8b).
    // rung: the activation ladder's rung the run settled on, which a resumed run starts at (null: none written).
    internal static void WriteStopState(string folder, IReadOnlyDictionary<string, OptimizerStateValue> state, Optimizer optimizer, int step, ActivationRung? rung = null)
    {
        Directory.CreateDirectory(folder);
        SafeTensorsWriter.Write(Path.Combine(folder, StopOptimizerFile), state.Select(e => (e.Key, e.Value.Shape, e.Value.Values)), SafeTensorType.F32,
            new Dictionary<string, string> { ["format"] = "idrak-optimizer", ["optimizer"] = optimizer.GetType().Name });
        var json = new System.Text.Json.Nodes.JsonObject { ["format"] = "idrak-tune-state/1", ["step"] = step };
        if (rung is { } settled)
        {
            json["activations"] = settled.ToJson();
        }

        File.WriteAllText(Path.Combine(folder, StopStateFile), json.ToJsonString() + "\n");
    }

    // The activation rung a stop state holds, or null (none written, or no state).
    internal static ActivationRung? ReadStopRung(string folder)
    {
        string path = Path.Combine(folder, StopStateFile);
        return File.Exists(path) ? ActivationRung.FromJson(System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(path))?["activations"]) : null;
    }

    internal static long ReadStopState(string folder)
    {
        string path = Path.Combine(folder, StopStateFile);
        return File.Exists(path)
            ? (long)System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(path))!["step"]!
            : throw new FileNotFoundException($"{folder} holds no stopped tune's state ({StopStateFile}); give a checkpoint-STEP folder a stop wrote.", path);
    }

    internal static Dictionary<string, OptimizerStateValue> ReadOptimizerFile(string folder)
    {
        using var reader = SafeTensorsReader.Open(Path.Combine(folder, StopOptimizerFile));
        ITensorStore store = reader;
        return store.Names.ToDictionary(n => n, n => new OptimizerStateValue(store.ShapeOf(n), store.Read(n)), StringComparer.Ordinal);
    }

    // This rank's share of a step's batches (by the ranks' shares: draws k·W + r while they are equal), and the step's
    // totals over every rank for a loss of one's own; a change of the shares is traced.
    internal static List<Batch> Share(List<Batch> global, IReadOnlyList<TrainingSequence> train, IGradientSync sync, StepRunner runner, Action<string>? trace)
    {
        var sizes = global.Select(b => (long)b.Sequences.Sum(i => train[i].TrainedTokens)).ToArray();
        var owners = DrawShares.Owners(sizes, sync.Shares);
        runner.StepTotals = ((float)Math.Max(1, sizes.Sum()), global.Sum(b => b.Sequences.Count()));
        runner.TraceShares(trace);
        return [.. global.Where((_, i) => owners[i] == sync.Rank)];
    }

    // The mean loss per trained token of `sequences` over every rank: each rank evaluates the sequences k·W + r, and the
    // ranks' token-weighted sums are added.
    internal static float EvaluateAcross(PretrainedModel model, IReadOnlyList<TrainingSequence> sequences, IGradientSync sync, int batchTokens, int chunkRows,
        Action<string>? trace, TuningVision? vision = null, IReadOnlyList<ISoftPartSource>? parts = null)
    {
        var mine = sequences.Where((_, i) => i % sync.WorldSize == sync.Rank).ToList();
        long trained = mine.Sum(q => (long)q.TrainedTokens);
        float mean = mine.Count > 0 ? Evaluate(model, mine, batchTokens, chunkRows, trace, vision, parts) : 0f;
        float[] totals = [(float)(mean * (double)trained), trained];
        sync.Sum(totals);
        return totals[1] > 0 ? totals[0] / totals[1] : 0f;
    }

    // The trained parameters of the trainable part sources, for the optimizer.
    internal static IEnumerable<Tensor> TrainedParts(IReadOnlyList<ISoftPartSource> parts) => parts.OfType<ITrainableSoftParts>().SelectMany(p => p.Parameters);

    // The image side of a run: the one given (checked against the model), else one made from the model when the sequences
    // hold images (returned as Owned too: the caller disposes it), else none.
    internal static (TuningVision? Vision, TuningVision? Owned) VisionFor(PretrainedModel model, IEnumerable<TrainingSequence> sequences, TuningVision? given)
    {
        if (given is not null)
        {
            if (!ReferenceEquals(given.Model, model))
            {
                throw new ArgumentException("FineTuningOptions.Vision was made for another model (TuningVision.Create with the model being tuned).");
            }

            return (given, null);
        }

        var preparations = sequences.SelectMany(s => s.Images).Select(i => i.Preparation).Distinct().Take(2).ToList();
        if (preparations.Count == 0)
        {
            return (null, null);
        }

        if (preparations.Count > 1)
        {
            throw new ArgumentException("The sequences' images are prepared in different ways (transforms or vision options); give FineTuningOptions.Vision.");
        }

        var made = TuningVision.Create(model, preparations[0]);
        return (made, made);
    }

    // Runs optimizer steps: ordinary ones, or replays of a recorded graph once one is recorded (see
    // FineTuningOptions.CudaGraphs).
    // lossRows: the most trained tokens of any batch the runner will see (the recorded loss's capacity).
    // custom: a loss of one's own (FineTuningOptions.Loss), or null for the token cross-entropy.
    // largest: the most positions any batch of the run holds (the first step's measured bytes per position say whether it fits).
    // imageTokens: the most image tokens any batch holds (a recorded step with images is made for them).
    // sync: data parallel (plan 23): the gradients are summed across the ranks after the step's backward passes, before
    // the update; null in one process.
    internal sealed class StepRunner(PretrainedModel model, IReadOnlyList<TrainingSequence> train, Optimizer optimizer, FineTuningOptions options,
        int lossRows, CustomLoss? custom, Action<string>? trace, long largest, int imageTokens, IGradientSync? sync = null) : IDisposable
    {
        private readonly IShardedGradientSync? _sharded = sync as IShardedGradientSync;   // asked once: it makes the update
        private IReadOnlyList<double>? _tracedShares;

        /// <summary>
        /// The loss of one's own the steps run with (null: the fused token cross-entropy, which may be recorded as a graph).
        /// A run keeps the one it was made with; a session (<see cref="FineTuningSession.Step"/>) sets each step's.
        /// </summary>
        public CustomLoss? Custom { get; set; } = custom;

        /// <summary>
        /// The step's trained tokens and sequences over every rank (data parallel): a loss of one's own divides by them, so
        /// each rank's loss is its share of the step's and the ranks' gradients add up; null in one process.
        /// </summary>
        public (float Tokens, int Sequences)? StepTotals { get; set; }

        /// <summary>Traces the ranks' shares of a step when they changed (measured from their speeds).</summary>
        public void TraceShares(Action<string>? trace)
        {
            if (sync is null || ReferenceEquals(_tracedShares, sync.Shares) || _tracedShares is not null && _tracedShares.SequenceEqual(sync.Shares))
            {
                _tracedShares = sync?.Shares;
                return;
            }

            bool first = _tracedShares is null;
            _tracedShares = sync.Shares;
            if (!first || sync.Shares.Distinct().Count() > 1)
            {
                trace?.Invoke("rows split by the ranks' measured speeds: "
                              + string.Join(", ", sync.Shares.Select((share, r) => string.Create(System.Globalization.CultureInfo.InvariantCulture, $"rank {r} {share * 100:F0}%"))));
            }
        }

        // Sums the step's gradients across the ranks (nothing in one process) and gives the step's loss over every rank.
        private (float Loss, long Tokens) Synced(float loss, long tokens, double trained, string? lost = null)
        {
            if (sync is null)
            {
                return (loss, tokens);
            }

            // A step with no backward pass here (no rows on this rank, a recorded step) still opens the overlap, so this rank
            // sends its buckets and the header in the order the ranks that overlapped do (plan 23, C4; found in C9).
            bool normalized = Custom is not null && StepTotals is not null;
            sync.Overlap(trained, normalized)?.Dispose();
            if (lost is not null)
            {
                sync.OverlapLost(lost);
            }

            var reduced = sync.Reduce(trained, loss, normalized: normalized);
            return ((float)reduced.Loss, (long)reduced.Tokens);
        }

        // The sequences the batches of the current step index (the run's, or one step's when sequences stream in, Run's
        // overload with them).
        private IReadOnlyList<TrainingSequence> _train = train;

        /// <summary>Runs one optimizer step over <paramref name="group"/>, whose batches index <paramref name="sequences"/> (this step's own).</summary>
        public (float Loss, long Tokens) Run(IReadOnlyList<Batch> group, IReadOnlyList<TrainingSequence> sequences, CancellationToken cancellationToken, Func<int, string> label)
        {
            _train = sequences;
            return Run(group, cancellationToken, label);
        }

        // Not with offloading (tensors move between steps, a graph holds their addresses) or the CPU update. The recorded
        // pass is the token cross-entropy's; a loss of one's own runs ordinary steps. Experts' routing is read back to the
        // host on every pass, which a recording cannot.
        private readonly bool _graphsBase = options.CudaGraphs && options.GradientAccumulation <= 1
            && !ComputeResources.OffloadToHostMemory && !options.HostOptimizer
            && model.Device.Backend.SupportsGraphs
            && Recordable(optimizer) && sync is not IShardedGradientSync   // a sharded update gathers and sends, which a recording cannot
            && !model.Network.Descendants().Any(m => m is Dropout { Probability: > 0f } or MixtureOfExperts);   // a recorded pass would reuse dropout masks

        // Graphs for the token cross-entropy only: a loss of one's own runs ordinary steps.
        private bool _graphsAllowed => _graphsBase && Custom is null;

        // An optimizer of one's own (FineTuningOptions.Optimizer): the recorded backward pass writes into the gradient
        // buffers the parameters had when it was recorded, so they must stay the parameters' between steps. Checked before
        // recording (each trained parameter still has its gradient after an update) and before every replay (the same
        // buffers); an optimizer that updates in place (IInPlaceOptimizer: the built-in AdamW) keeps them by construction.
        private readonly bool _checkGradients = options.Optimizer is not null && optimizer is not IInPlaceOptimizer;
        private Tensor?[] _gradients = [];
        private readonly bool _automatic = options.Checkpointing is null && !ComputeResources.OffloadToHostMemory;
        private FineTuningOptions _options = options with { Checkpointing = options.Checkpointing ?? ComputeResources.OffloadToHostMemory };

        // The recorded steps: one graph per batch shape (rows and positions, and whether it holds images: a graph with images
        // holds the features of up to the run's most image tokens, so one serves every batch of its rows and positions), each
        // recorded once its shape ran twice as an ordinary step; another only while the device's measured free memory
        // leaves room for one more of the largest so far.
        private readonly List<TrainingGraph> _recorded = [];
        private readonly Dictionary<(int Rows, int Length, int ImageTokens), int> _seen = [];
        private bool? _graphs;
        private bool _toldImages;

        private readonly bool _automaticRecompute = options.RecomputeFeedForward is null && options.Checkpointing is null && !ComputeResources.OffloadToHostMemory;
        private readonly bool _automaticBFloat16 = options.BFloat16Activations is null && options.Checkpointing is null && !ComputeResources.OffloadToHostMemory;

        /// <summary>Whether steps run with activation checkpointing (it can turn on during training, see <see cref="FineTuningOptions.Checkpointing"/>).</summary>
        public bool Checkpointing => _options.Checkpointing == true;

        /// <summary>The memory settings in effect and the recorded step graphs, as a log line says them.</summary>
        public string Settings => (Checkpointing ? "checkpointing" : _options.RecomputeFeedForward == true ? Describe(_options) : "activations held")
            + (_recorded.Count > 0 ? $", {_recorded.Count} recorded step graph{(_recorded.Count == 1 ? "" : "s")}" : "");

        // Choosing by speed after the first out-of-memory step: the lightest setting that fits (recomputed feed-forward
        // activations, then also bfloat16 ones) is timed over a step, then checkpointing is, and the faster is kept.
        private readonly bool _choose = options.Checkpointing is null && options.RecomputeFeedForward is null && options.BFloat16Activations is null
            && !ComputeResources.OffloadToHostMemory;
        private int _probe;                         // 0: not measuring; 1: the fitting setting; 2: checkpointing; 3: chosen
        private int _probeSteps;
        private double _lightSeconds;
        private FineTuningOptions? _light;

        public (float Loss, long Tokens) Run(IReadOnlyList<Batch> group, CancellationToken cancellationToken, Func<int, string> label, bool graphs = true)
        {
            if (group.Count == 0)
            {
                // Data parallel: this rank has none of the step's rows; it adds nothing to the sum and updates with the others.
                optimizer.ZeroGrad();
                var (loss, tokens) = Synced(0f, 0, 0);
                Update(model, optimizer, _options, loss);
                return (loss, tokens);
            }

            // Every step with images whose vision side trains runs ordinarily (its features need gradients); said once.
            if (!_toldImages && options.Vision is { Parameters.Count: > 0 } && _graphsAllowed && group.Any(b => b.HasImages(_train)))
            {
                _toldImages = true;
                trace?.Invoke("steps with images run without graphs while a vision part trains (the features need gradients)");
            }

            // Out of device memory: first recompute the feed-forward activations (cheap), then hold activations as
            // bfloat16, then checkpoint every block (a third more compute); the step is run again each time and the graph,
            // recorded without them, recorded again. Which of the setting that fits and checkpointing is faster is measured.
            // The first step also measures its bytes per position, so a run whose largest batch would not fit takes the
            // first rung before it fails (FirstRung). Before that, a shape not seen yet starts at the rung its estimate
            // fits (Estimate).
            if (options.EstimateActivations && _probe is 0 or 3)
            {
                Estimate(group);
            }

            while (true)
            {
                try
                {
                    bool measure = !_firstMeasured && _automaticRecompute && _options.RecomputeFeedForward != true && !Checkpointing && _probe == 0;
                    long held = 0;
                    if (measure)
                    {
                        model.Device.Backend.ResetPeakMemoryUsage();
                        held = model.Device.Backend.GetMemoryUsage().InUse;
                    }

                    long started = Stopwatch.GetTimestamp();
                    var result = RunOnce(group, cancellationToken, label, graphs && _probe is 0 or 3);
                    _estimate = null;                                            // it held
                    Measured(Stopwatch.GetElapsedTime(started).TotalSeconds / Math.Max(1L, group.Sum(b => (long)b.Rows.Length * b.Length)));
                    if (measure)
                    {
                        _firstMeasured = true;
                        FirstRung(group, held);
                    }

                    return result;
                }
                catch (ResourceLimitExceededException ex) when (_probe == 2)
                {
                    Choose(checkpointing: false, $"checkpointing ran out of device memory ({ex.Message.Split(':')[0]}{Short()})");
                }
                catch (ResourceLimitExceededException ex) when (_automaticRecompute && _options.RecomputeFeedForward != true && !Checkpointing)
                {
                    _options = _options with { RecomputeFeedForward = true };
                    Reset();
                    StartProbe();
                    trace?.Invoke($"out of device memory ({ex.Message.Split(':')[0]}{Short()}): feed-forward activations are recomputed in the backward pass "
                                  + "from this step (RecomputeFeedForward / --recompute to start with it)");
                }
                catch (ResourceLimitExceededException ex) when (_automaticBFloat16 && _options.BFloat16Activations != true && !Checkpointing)
                {
                    _options = _options with { BFloat16Activations = true };
                    Reset();
                    StartProbe();
                    trace?.Invoke($"out of device memory ({ex.Message.Split(':')[0]}{Short()}): activations are held as bfloat16 between the passes "
                                  + "from this step (BFloat16Activations / --bf16-activations to start with it)");
                }
                catch (ResourceLimitExceededException ex) when (_automatic && !Checkpointing)
                {
                    _options = _options with { Checkpointing = true };
                    Reset();
                    _probe = 3;                                                  // nothing lighter fits
                    trace?.Invoke($"out of device memory without activation checkpointing ({ex.Message.Split(':')[0]}{Short()}): checkpointing is on from this step "
                                  + "(set Checkpointing / --checkpointing to start with it)");
                }
            }
        }

        private bool _firstMeasured;

        // The shapes (rows × positions) estimated, whether the rung was said, and the estimate of the step about to run
        // (said by the out-of-memory line when it was short).
        private readonly HashSet<(int Rows, int Length)> _estimated = [];
        private bool _toldRung;
        private (long Bytes, long Room)? _estimate;

        /// <summary>The rung of the activation ladder the steps run on.</summary>
        internal ActivationRung Rung => ActivationRung.Of(_options);

        // Before the first step of a shape: the estimate of each rung from the one in effect down, against what the device
        // can give; the first that fits is taken (measured against checkpointing as after a failed step when lighter), the
        // lightest when none does. Nothing changes when the device reports no free memory or no limit.
        private void Estimate(IReadOnlyList<Batch> group)
        {
            var batch = group.MaxBy(b => (long)b.Rows.Length * b.Length)!;
            int rows = batch.Rows.Length, length = batch.Length;
            var backend = model.Device.Backend;
            if (!_estimated.Add((rows, length)) || Room() is not { } room)
            {
                return;
            }

            long pending = 0;
            foreach (var parameter in optimizer.Parameters)
            {
                pending += parameter.RequiresGrad && parameter.Grad is null ? parameter.Size : 0;
            }

            long Bytes(ActivationRung r) => ActivationEstimate.Bytes(model.Spec, backend.Capabilities, rows, length, r, _options.LossChunkRows, pending);
            var start = ActivationRung.Of(_options);
            var (rung, bytes, without) = (start, Bytes(start), 0L);
            while (bytes > room && Next(rung) is { } next)
            {
                (rung, bytes, without) = (next, Bytes(next), bytes);
            }

            _estimate = (bytes, room);
            string shape = $"{rows} × {length} positions";
            if (rung == start)
            {
                if (!_toldRung)
                {
                    trace?.Invoke($"activations: {start} (estimated {ActivationEstimate.Size(bytes)} for {shape}, {ActivationEstimate.Size(room)} free)");
                }

                _toldRung = true;
                return;
            }

            _options = Apply(rung);
            Reset();
            if (rung.Checkpointing)
            {
                _probe = 3;                                                      // nothing lighter fits
            }
            else
            {
                StartProbe();
            }

            trace?.Invoke($"activations: {rung} from {(_toldRung ? "this step" : "the start")} (estimated {ActivationEstimate.Size(without)} without, "
                          + $"{ActivationEstimate.Size(bytes)} with, {ActivationEstimate.Size(room)} free for {shape}"
                          + (bytes > room ? "; no lighter rung's estimate fits" : "")
                          + "; EstimateActivations / IDRAK_ACTIVATION_ESTIMATE=0 to start from the settings as given)");
            _toldRung = true;
        }

        // The rung the out-of-memory ladder takes after `rung` (the catches in Run, in their order), or null at its end.
        private ActivationRung? Next(ActivationRung rung) =>
            rung.Checkpointing ? null
            : _automaticRecompute && !rung.Recompute ? rung with { Recompute = true }
            : _automaticBFloat16 && !rung.BFloat16 ? rung with { BFloat16 = true }
            : _automatic ? rung with { Checkpointing = true }
            : null;

        private FineTuningOptions Apply(ActivationRung rung) => _options with
        {
            RecomputeFeedForward = rung.Recompute ? true : _options.RecomputeFeedForward,
            BFloat16Activations = rung.BFloat16 ? true : _options.BFloat16Activations,
            Checkpointing = rung.Checkpointing ? true : _options.Checkpointing,
        };

        /// <summary>
        /// Starts at the rung a run settled on before (a checkpoint's), as far as the settings leave the ladder automatic, and
        /// keeps it (no measuring between rungs). A lighter rung than the one in effect changes nothing.
        /// </summary>
        internal void StartAt(ActivationRung saved, string source)
        {
            var rung = ActivationRung.Of(_options);
            var target = new ActivationRung(rung.Recompute || saved.Recompute && _automaticRecompute, rung.BFloat16 || saved.BFloat16 && _automaticBFloat16,
                rung.Checkpointing || saved.Checkpointing && _automatic);
            if (target == rung)
            {
                return;
            }

            _options = Apply(target);
            Reset();
            _probe = 3;
            _toldRung = true;
            trace?.Invoke($"activations: {target} from the start (the rung {source} settled on)");
        }

        // What the estimate is held to: a step that fails short of it says so.
        private string Short()
        {
            if (_estimate is not { } estimate)
            {
                return "";
            }

            _estimate = null;
            return $"; the estimate, {ActivationEstimate.Size(estimate.Bytes)} of {ActivationEstimate.Size(estimate.Room)} free, was short";
        }

        // What the device can still give: its reported free memory, within a configured limit (a hard cap: a device that
        // reports its free memory without the limit, the CPU's system memory, is held to it); null when it reports neither.
        private long? Room()
        {
            var backend = model.Device.Backend;
            var usage = backend.GetMemoryUsage();
            long? reported = backend.AvailableMemory(), capped = usage.Limit is { } limit ? limit - usage.InUse : null;
            return reported is { } r && capped is { } c ? Math.Min(r, c) : reported ?? capped;
        }

        // After the first step: its bytes per position (the peak above what stays held after it, the optimizer's state and
        // gradients made by that step aside, over its largest batch's positions) times the run's largest batch, against
        // what the device can still give (its reported free memory, within the configured limit). When that batch would
        // not fit, the ladder's first rung (recomputed feed-forward activations) is taken now, and measured against
        // checkpointing as after a failed step, instead of first failing part-way through a large step. Nothing changes
        // when the device reports neither, or the run's batches all fit.
        private void FirstRung(IReadOnlyList<Batch> group, long held)
        {
            var backend = model.Device.Backend;
            long positions = group.Max(b => (long)b.Rows.Length * b.Length);
            var usage = backend.GetMemoryUsage();
            long stepBytes = usage.Peak - Math.Max(held, usage.InUse);
            if (positions <= 0 || largest <= positions || stepBytes <= 0 || Room() is not { } room)
            {
                return;
            }

            double perPosition = (double)stepBytes / positions, needed = perPosition * largest;
            if (needed <= room)
            {
                return;
            }

            _options = _options with { RecomputeFeedForward = true };
            Reset();
            StartProbe();
            trace?.Invoke(string.Create(System.Globalization.CultureInfo.InvariantCulture,
                $"the largest batch ({largest} positions) needs about {needed / (1 << 20):F0} MiB at the first step's {perPosition / 1024:F1} KiB per position, ")
                + string.Create(System.Globalization.CultureInfo.InvariantCulture,
                $"{room / (1 << 20)} MiB can be had: feed-forward activations are recomputed in the backward pass from the next step (RecomputeFeedForward / --recompute to start with it)"));
        }

        private void StartProbe()
        {
            if (_choose && _automatic && _probe != 3)
            {
                (_probe, _probeSteps) = (1, 0);
            }
        }

        // Seconds per position of a step that finished: the second step of each measured setting counts (the first one
        // allocates its memory).
        private void Measured(double secondsPerPosition)
        {
            if (_probe is not (1 or 2) || ++_probeSteps < 2)
            {
                return;
            }

            if (_probe == 1)
            {
                (_light, _lightSeconds) = (_options, secondsPerPosition);
                _options = _options with { Checkpointing = true, RecomputeFeedForward = false, BFloat16Activations = false };
                Reset();
                (_probe, _probeSteps) = (2, 0);
                trace?.Invoke($"measuring: {Describe(_light)} took {secondsPerPosition * 1e6:F2} µs per position; timing checkpointing next");
                return;
            }

            Choose(checkpointing: secondsPerPosition < _lightSeconds,
                $"{Describe(_light!)} {_lightSeconds * 1e6:F2} µs per position, checkpointing {secondsPerPosition * 1e6:F2} µs");
        }

        private void Choose(bool checkpointing, string why)
        {
            if (!checkpointing)
            {
                _options = _light!;
            }

            Reset();
            _probe = 3;
            trace?.Invoke($"keeping {(checkpointing ? "checkpointing" : Describe(_light!))} for the rest of training ({why})");
        }

        private static string Describe(FineTuningOptions options) =>
            options.BFloat16Activations == true ? "recomputed feed-forward and bfloat16 activations" : "recomputed feed-forward activations";

        /// <summary>How many times the settings changed (out of memory, or a measured choice).</summary>
        public int Changes { get; private set; }

        /// <summary>Whether the settings are final (no measurement between two settings is running).</summary>
        public bool Settled => _probe is 0 or 3;

        private void Reset()
        {
            Changes++;
            DropGraphs();
            _graphs = null;
            _seen.Clear();
        }

        private void DropGraphs()
        {
            _recorded.ForEach(g => g.Dispose());
            _recorded.Clear();
        }

        private (float Loss, long Tokens) RunOnce(IReadOnlyList<Batch> group, CancellationToken cancellationToken, Func<int, string> label, bool graphs)
        {
            var options = _options;
            _graphs ??= _graphsAllowed;
            var batch = group[0];
            var shape = (batch.Rows.Length, batch.Length, batch.HasImages(_train) ? Math.Max(imageTokens, batch.ImageTokens(_train)) : 0);

            // A batch with images replays only when its features are values (no vision part trains): they are copied in.
            if (graphs && _graphs == true && Custom is null && group.Count == 1 && batch.Packed && (shape.Item3 == 0 || options.Vision is { Parameters.Count: 0 }) && !batch.HasParts(_train))
            {
                var graph = _recorded.Where(g => g.Fits(batch, _train)).MinBy(g => g.ImageTokens);
                if (graph is null && _seen.GetValueOrDefault(shape) >= 2 && RoomForAnother(shape))
                {
                    if (_checkGradients && optimizer.Parameters.Any(p => p.RequiresGrad && p.Grad is null))
                    {
                        trace?.Invoke("graph not used (the optimizer does not keep the gradient buffers between steps); ordinary steps continue");
                        _graphs = false;
                        return Ordinary(group, shape, cancellationToken, label);
                    }

                    graph = TrainingGraph.Record(model, _train, optimizer, options, batch, (Math.Max(1, lossRows) + 63) / 64 * 64, shape.Item3, trace);
                    if (graph is null)
                    {
                        _graphs = _recorded.Count > 0;                       // the first failed: none; a later one: not this shape
                        _seen[shape] = int.MinValue;
                    }
                    else
                    {
                        if (_recorded.Count == 0)
                        {
                            _gradients = [.. optimizer.Parameters.Select(p => p.Grad)];
                        }

                        _recorded.Add(graph);
                    }
                }

                if (_recorded.Count > 0 && _checkGradients && optimizer.Parameters.Where((p, i) => !ReferenceEquals(p.Grad, _gradients[i])).Any())
                {
                    trace?.Invoke("graph dropped (the optimizer replaced the gradient buffers the recorded step writes); ordinary steps continue");
                    DropGraphs();
                    (graph, _graphs) = (null, false);
                }

                if (graph is not null)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var watch = Stopwatch.StartNew();
                    model.Network.Train();
                    var (loss, tokens) = graph.Run(_train, batch);
                    trace?.Invoke($"{label(0)}: {batch.Describe(_train)}, replayed: loss {loss:F4}, {watch.Elapsed.TotalSeconds:F2} s");
                    (loss, tokens) = Synced(loss, tokens, batch.Sequences.Sum(i => _train[i].TrainedTokens),
                        "a recorded step: the graph runs no hook, so its gradients are reduced after the replay");
                    Update(model, optimizer, options, loss);
                    return (loss, tokens);
                }
            }

            return Ordinary(group, shape, cancellationToken, label);
        }

        private (float Loss, long Tokens) Ordinary(IReadOnlyList<Batch> group, (int, int, int) shape, CancellationToken cancellationToken, Func<int, string> label)
        {
            _seen[shape] = _seen.GetValueOrDefault(shape) + 1;
            return RunStep(model, _train, group, optimizer, _options, Custom, cancellationToken, trace, label, sync, _sharded, StepTotals);
        }

        // Whether another graph may be recorded: always the first; another while the device's free memory, measured now,
        // holds twice the largest recorded graph (the new one, and the same again for the steps that still run
        // ordinarily). A device that reports no free memory keeps one graph. Said once per shape refused.
        private bool RoomForAnother((int Rows, int Length, int ImageTokens) shape)
        {
            if (_recorded.Count == 0)
            {
                return true;
            }

            long largest = _recorded.Max(g => g.Bytes);
            long? free = model.Device.Backend.AvailableMemory();
            if (free is long bytes && bytes >= 2 * largest)
            {
                return true;
            }

            _seen[shape] = int.MinValue;
            trace?.Invoke($"no graph for {shape.Rows} × {shape.Length} positions with {shape.ImageTokens} image tokens: "
                          + (free is long f ? $"{f / 1048576.0:F0} MB free on the device, " : "the device reports no free memory, ")
                          + $"{_recorded.Count} recorded graphs hold up to {largest / 1048576.0:F0} MB each; that shape runs ordinary steps");
            return false;
        }

        public void Dispose() => DropGraphs();

        // Whether the optimizer updates the parameters where they are: one that says so (IInPlaceOptimizer), a group of
        // such (IGroupedOptimizer), or one of one's own that does not bring the gradients to the host (IHostGradients: the CPU update, which
        // releases them); the gradient buffers of one that does not say so are checked before recording and replaying.
        private static bool Recordable(IOptimizer optimizer) => optimizer switch
        {
            IInPlaceOptimizer => true,
            IGroupedOptimizer grouped => grouped.Groups.All(Recordable),
            _ => optimizer is not IHostGradients,
        };
    }

    // FP8 copies of the frozen weights of the layers with adapters (options.Float8), kept only when a sample's loss with
    // them is within the tolerance of its bfloat16 loss. Disposing the result removes them again.
    internal static IDisposable? PrepareFloat8(PretrainedModel model, IReadOnlyList<TrainingSequence> train, FineTuningOptions options, Action<string>? trace)
    {
        if (!options.Float8)
        {
            return null;
        }

        var layers = model.Network.Descendants().OfType<Linear>().Where(l => l.Adapter is not null).ToList();
        var removal = new Float8Removal(layers);
        int count = Math.Min(16, train.Count);
        var sample = Enumerable.Range(0, count).Select(i => train[(int)((long)i * train.Count / count)]).ToList();
        float reference = Evaluate(model, sample, options.BatchTokens, options.LossChunkRows, vision: options.Vision);
        int attached = layers.Count(l => l.AttachFloat8());
        if (attached == 0)
        {
            trace?.Invoke("FP8 products are not available here (CUDA, compute capability 8.9 or newer); training stays in bfloat16");
            return removal;
        }

        float quantized = Evaluate(model, sample, options.BatchTokens, options.LossChunkRows, vision: options.Vision);
        double difference = Math.Abs(quantized - reference) / Math.Max(1e-6, Math.Abs(reference));
        trace?.Invoke($"FP8 check on {count} training sequences: loss {quantized:F4} with FP8 products of the frozen weights, {reference:F4} in bfloat16 "
                      + $"({difference:P2} apart, {options.Float8Tolerance:P0} allowed); {attached} of {layers.Count} layers");
        if (!(difference <= options.Float8Tolerance))
        {
            removal.Dispose();
            trace?.Invoke("FP8 loss too far from bfloat16's: training stays in bfloat16");
        }

        return removal;
    }

    private sealed class Float8Removal(List<Linear> layers) : IDisposable
    {
        public void Dispose() => layers.ForEach(l => l.DetachFloat8());
    }

    private static int MostTrained(IEnumerable<Batch> batches, IReadOnlyList<TrainingSequence> train) =>
        batches.Select(b => b.Sequences.Sum(i => train[i].TrainedTokens)).DefaultIfEmpty(0).Max();

    // The most positions any batch of the run holds (rows × length): the largest pass the device must hold.
    private static long Largest(IEnumerable<Batch> batches) => batches.Select(b => (long)b.Rows.Length * b.Length).DefaultIfEmpty(0).Max();

    // The most image tokens any batch of the run holds: what a recorded step with images is made for, so it serves every batch.
    private static int MostImageTokens(IEnumerable<Batch> batches, IReadOnlyList<TrainingSequence> train) =>
        batches.Select(b => b.ImageTokens(train)).DefaultIfEmpty(0).Max();

    // AdamW (or FineTuningOptions.Optimizer) on the device, or on the CPU with its state in system memory
    // (FineTuningOptions.HostOptimizer).
    internal static Optimizer CreateOptimizer(List<Tensor> parameters, FineTuningOptions options)
    {
        Func<IReadOnlyList<Tensor>, Optimizer> create = options.Optimizer is { } custom
            ? ps => custom(ps) ?? throw new InvalidOperationException("FineTuningOptions.Optimizer returned no optimizer.")
            : ps => new AdamW(ps, options.LearningRate, weightDecay: options.WeightDecay);
        return options.HostOptimizer ? new HostOptimizer(parameters, create) : create(parameters);
    }

    // Clipping and the optimizer's update, after the gradients of a step whose loss (read already) is finite; a NaN or
    // infinite loss drops the step's gradients instead, so one bad step leaves the parameters as they were.
    private static void Update(PretrainedModel model, Optimizer optimizer, FineTuningOptions options, float loss, IShardedGradientSync? sharded = null)
    {
        if (!float.IsFinite(loss))
        {
            optimizer.ZeroGrad();
            return;
        }

        if (sharded is not null)
        {
            sharded.Update(options.MaxGradientNorm);                    // a sharded run: each rank steps its own share (plan 23, C6)
            return;
        }

        // Clipping and the update of every adapter matrix in a few device passes (which also zero the gradients for the next
        // step), with no read of the norm and no wait: the next step's first read of the device orders everything, so the
        // host prepares the next batch while the device finishes this step.
        optimizer.ClipAndStep(options.MaxGradientNorm);
    }

    // One optimizer step over a group of batches (gradient accumulation): forward, backward, clipping, update. Returns
    // the mean loss per trained token and the tokens covered.
    // With a cluster (sync) the gradients are summed across the ranks before the update: weighted by each rank's trained
    // tokens for the token loss; a loss of one's own divides by the whole step's counts (stepTotals) and is summed.
    private static (float Loss, long Tokens) RunStep(PretrainedModel model, IReadOnlyList<TrainingSequence> train, IReadOnlyList<Batch> group, Optimizer optimizer,
        FineTuningOptions options, CustomLoss? custom, CancellationToken cancellationToken, Action<string>? trace, Func<int, string> label,
        IGradientSync? sync = null, IShardedGradientSync? sharded = null, (float Tokens, int Sequences)? stepTotals = null)
    {
        var network = model.Network;
        int trained = group.Sum(b => b.Sequences.Sum(i => train[i].TrainedTokens));
        bool whole = custom is not null && sync is not null && stepTotals is not null;
        float normalizer = whole ? stepTotals!.Value.Tokens : Math.Max(1, trained);
        int sequences = whole ? stepTotals!.Value.Sequences : group.Sum(b => b.Sequences.Count());
        float loss = 0f;
        long tokens = 0;
        network.Train();
        optimizer.ZeroGrad();
        for (int b = 0; b < group.Count; b++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var batch = group[b];
            var batchWatch = Stopwatch.StartNew();
            using var scope = new TensorScope();
            trace?.Invoke($"{label(b)}: {batch.Describe(train)}…");
            using var packing = batch.Packed ? PackedSequences.Create([.. batch.Rows.Select(r => r.Select(i => train[i].Tokens.Length - 1).ToArray())], batch.Length, model.Device) : null;
            using var packed = packing?.Use();                                // forward and backward (checkpointed blocks run again)
            using var recompute = options.RecomputeFeedForward == true ? ActivationMemory.Recompute() : (ActivationMemory.Scope?)null;
            using var compress = options.BFloat16Activations == true ? ActivationMemory.CompressToBFloat16() : (ActivationMemory.Scope?)null;
            long hits = options.Vision?.CacheHits ?? 0, encoded = options.Vision?.Encoded ?? 0;
            var (lossTensor, count) = BatchLoss(model, train, batch, normalizer, options.LossChunkRows, options.Checkpointing == true, custom, sequences, options.Vision, options.Parts);
            if (options.LoadBalancingWeight > 0f && LoadBalancing(network) is { } balance)
            {
                lossTensor += (balance * (options.LoadBalancingWeight / group.Count)).Reshape(lossTensor.Shape);
            }

            // queued behind the forward pass, no wait between; the last batch's gradients go across the ranks while it runs
            using (b == group.Count - 1 ? sync?.Overlap(trained, whole) : null)
            {
                lossTensor.Backward();
            }

            float batchLoss = lossTensor.Item();                             // waits for the batch's forward and backward
            loss += batchLoss;
            tokens += count;
            float shown = custom is null ? batchLoss * normalizer / Math.Max(1, batch.Sequences.Sum(i => train[i].TrainedTokens)) : batchLoss;
            string imageCounts = options.Vision is { } v && batch.HasImages(train)
                ? $", image features: {v.CacheHits - hits} from the cache, {v.Encoded - encoded} encoded, {v.LastFeatures.Images} images in {v.LastFeatures.Passes} "
                  + $"pass{(v.LastFeatures.Passes == 1 ? "" : "es")}, {v.LastFeatures.Milliseconds:F1} ms"
                : "";
            trace?.Invoke($"  forward and backward {batchWatch.Elapsed.TotalSeconds:F2} s, loss {shown:F4}{imageCounts}");
        }

        if (sync is not null)
        {
            var reduced = sync.Reduce(trained, loss, normalized: whole);
            (loss, tokens) = ((float)reduced.Loss, (long)reduced.Tokens);
        }

        Update(model, optimizer, options, loss, sharded);
        return (loss, tokens);
    }

    // The mean of the experts' load-balancing losses of the pass just run (null without experts, or when checkpointed
    // blocks recorded none).
    private static Tensor? LoadBalancing(Module network)
    {
        var losses = network.Descendants().OfType<MixtureOfExperts>().Select(m => m.LoadBalancingLoss).OfType<Tensor>().ToList();
        if (losses.Count == 0)
        {
            return null;
        }

        var sum = losses[0];
        for (int i = 1; i < losses.Count; i++)
        {
            sum += losses[i];
        }

        return sum * (1f / losses.Count);
    }

    /// <summary>
    /// Measures training steps without a full run: <paramref name="warmup"/> steps first (the first pays one-time
    /// setup), then <paramref name="timed"/> steps timed by the clock, then <paramref name="profiled"/> steps with every
    /// GPU kernel timed (<see cref="GpuProfiler"/>; each kernel is waited for, so these steps run slower). The adapters
    /// train during the measurement; discard the model's adapters afterwards (or reload the model).
    /// </summary>
    public static FineTuningProfile Profile(PretrainedModel model, IReadOnlyList<TrainingSequence> train, FineTuningOptions options, int warmup = 3,
        int timed = 3, int profiled = 3, Action<string>? trace = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        var (vision, ownedVision) = VisionFor(model, train, options.Vision);
        using var disposeVision = ownedVision;
        options = options with { Vision = vision };
        if (!model.Network.Descendants().OfType<IAdaptableLinear>().Any(l => l.Adapter is not null))
        {
            model.AddAdapters(options.Rank, options.Alpha, options.Targets, options.Seed, options.Dora);
        }

        using var float8 = PrepareFloat8(model, train, options, trace);
        using var optimizer = CreateOptimizer(model.Network.TrainableParameters().Concat(vision?.Parameters ?? []).Concat(TrainedParts(options.Parts)).ToList(), options);
        var batches = MakeBatches(model, train, options, new Random(options.Seed));
        int accumulation = Math.Max(1, options.GradientAccumulation);
        int next = 0;
        using var runner = new StepRunner(model, train, optimizer, options, MostTrained(batches, train), Custom(model, options, pairs: false), trace, Largest(batches),
            MostImageTokens(batches, train));
        (float Loss, long Tokens, double Seconds) Step(string phase)
        {
            var group = Enumerable.Range(0, accumulation).Select(i => batches[(next + i) % batches.Count]).ToList();
            next += accumulation;
            var watch = Stopwatch.StartNew();
            // The profiled steps run ordinary launches (each kernel timed); the others replay a graph when one is used.
            var (loss, tokens) = runner.Run(group, CancellationToken.None, b => $"{phase} step, batch {b + 1}", graphs: phase != "profiled");
            return (loss, tokens, watch.Elapsed.TotalSeconds);
        }

        // Warm-up again after the runner changed settings (out of memory, or a measured choice): the timed steps run the
        // settings training keeps.
        for (int round = 0, changes = -1; round < 5 && changes != runner.Changes; round++)
        {
            changes = runner.Changes;
            for (int i = 0; i < warmup || !runner.Settled && i < 8; i++)
            {
                Step("warm-up");
            }
        }

        long timedTokens = 0;
        double timedSeconds = 0;
        for (int i = 0; i < timed; i++)
        {
            var (_, tokens, seconds) = Step("timed");
            timedTokens += tokens;
            timedSeconds += seconds;
        }

        GpuProfiler.Start(model.Device);
        long profiledTokens = 0;
        for (int i = 0; i < profiled; i++)
        {
            profiledTokens += Step("profiled").Tokens;
        }

        var kernels = GpuProfiler.Stop(model.Device);
        model.Network.Eval();
        return new FineTuningProfile(kernels, Math.Max(1, profiled), profiledTokens, timedSeconds / Math.Max(1, timed), timedTokens / (double)Math.Max(1, timed));
    }

    /// <summary>
    /// Mean loss per trained token of <paramref name="sequences"/> (no gradients). Sequences with images read their
    /// features through <paramref name="vision"/> (one made from the model for this call when null).
    /// </summary>
    public static float Evaluate(PretrainedModel model, IReadOnlyList<TrainingSequence> sequences, int batchTokens = 4096, int chunkRows = 512,
        Action<string>? trace = null, TuningVision? vision = null, IReadOnlyList<ISoftPartSource>? parts = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(sequences);
        var (images, owned) = VisionFor(model, sequences, vision);
        using var disposeVision = owned;
        model.Network.Eval();
        double total = 0;
        long trained = 0;
        using (Autograd.NoGrad())
        {
            var batches = Batches(sequences, batchTokens, random: null);
            for (int b = 0; b < batches.Count; b++)
            {
                var batch = batches[b];
                var watch = Stopwatch.StartNew();
                using var scope = new TensorScope();
                int count = batch.Sum(i => sequences[i].TrainedTokens);
                var (loss, _) = BatchLoss(model, sequences, Batch.Padded(batch, sequences), 1f, chunkRows, vision: images, parts: parts);
                total += loss.Item();
                trained += count;
                trace?.Invoke($"evaluation batch {b + 1}/{batches.Count}: {batch.Length} sequences × {batch.Max(i => sequences[i].Tokens.Length) - 1} tokens, {watch.Elapsed.TotalSeconds:F2} s");
            }
        }

        return (float)(total / Math.Max(1, trained));
    }

    // The loss of one's own (FineTuningOptions.Loss) over all of `sequences` (batches of whole pairs when `pairs`), without
    // gradients: the step totals are the whole set's, so the batches' losses add up to the set's loss.
    private static float EvaluateCustom(PretrainedModel model, IReadOnlyList<TrainingSequence> sequences, bool pairs, CustomLoss custom, FineTuningOptions options,
        Action<string>? trace)
    {
        model.Network.Eval();
        var batches = pairs ? PairBatches(sequences, options.BatchTokens, random: null)
            : [.. Batches(sequences, options.BatchTokens, random: null).Select(b => Batch.Padded(b, sequences))];
        int tokens = batches.Sum(b => b.Sequences.Sum(i => sequences[i].TrainedTokens));
        int count = batches.Sum(b => b.Sequences.Count());
        double total = 0;
        using (Autograd.NoGrad())
        {
            for (int b = 0; b < batches.Count; b++)
            {
                var watch = Stopwatch.StartNew();
                using var scope = new TensorScope();
                var (loss, _) = BatchLoss(model, sequences, batches[b], tokens, options.LossChunkRows, custom: custom, stepSequences: count, vision: options.Vision, parts: options.Parts);
                total += loss.Item();
                trace?.Invoke($"evaluation batch {b + 1}/{batches.Count}: {batches[b].Describe(sequences)}, {watch.Elapsed.TotalSeconds:F2} s");
            }
        }

        return (float)total;
    }

    // Preference pairs (sequences 2k and 2k + 1) in padded batches of whole pairs, pairs of similar length together, at most
    // batchTokens padded positions each (one pair when a pair alone exceeds it); the batch order is shuffled when `random`
    // is given.
    internal static List<Batch> PairBatches(IReadOnlyList<TrainingSequence> sequences, int batchTokens, Random? random)
    {
        int Length(int pair) => Math.Max(sequences[2 * pair].Tokens.Length, sequences[2 * pair + 1].Tokens.Length) - 1;
        var tie = TieOrder(sequences.Count / 2, random);
        var order = Enumerable.Range(0, sequences.Count / 2).Where(p => sequences[2 * p].Tokens.Length > 1 && sequences[2 * p + 1].Tokens.Length > 1)
            .OrderBy(Length).ThenBy(p => tie[p]).ToList();
        var batches = new List<Batch>();
        var current = new List<int>();
        int longest = 0;
        foreach (int pair in order)
        {
            if (current.Count > 0 && (current.Count + 2) * Math.Max(longest, Length(pair)) > batchTokens)
            {
                batches.Add(Batch.Padded([.. current], sequences));
                current.Clear();
                longest = 0;
            }

            current.Add(2 * pair);
            current.Add(2 * pair + 1);
            longest = Math.Max(longest, Length(pair));
        }

        if (current.Count > 0)
        {
            batches.Add(Batch.Padded([.. current], sequences));
        }

        if (random is not null)
        {
            for (int i = batches.Count - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                (batches[i], batches[j]) = (batches[j], batches[i]);
            }
        }

        return batches;
    }

    // A loss of one's own (FineTuningOptions.Loss) for each batch: the log-probabilities of the trained tokens are computed
    // without gradients, the loss and its gradient with respect to them on the CPU, and that gradient becomes the weights
    // of a token cross-entropy over the same rows (d loss / d θ = Σ_t g_t · d log p_t / d θ, and the cross-entropy is
    // −log p_t), whose backward pass is the network's. The output head runs twice over the trained rows; the full logits
    // are never stored. Reference log-probabilities (adapters disabled) are kept per sequence for the whole run.
    // A teacher's divergences (FineTuningLossInput.TeacherDivergence) go the same way: their values first, without
    // gradients, then the divergence over the same rows weighted by their gradient g (d loss / d θ = Σ_t g_t · d KL_t / d θ).
    internal sealed class CustomLoss(IFineTuningLoss loss, bool pairs, IDistillationTeacher? teacher = null)
    {
        private readonly Dictionary<TrainingSequence, float[]> _reference = [];
        private readonly IFineTuningLoss _loss = Checked(loss, pairs, teacher);   // what the loss needs, checked once before any step

        // The loss, refused when it needs pairs or a teacher it is not given.
        private static IFineTuningLoss Checked(IFineTuningLoss loss, bool pairs, IDistillationTeacher? teacher)
        {
            ArgumentNullException.ThrowIfNull(loss);
            if (loss.NeedsPairs && !pairs)
            {
                throw new InvalidOperationException(FineTuningLosses.NeedsPairsMessage(loss.Name));
            }

            return !loss.NeedsTeacher || teacher is not null ? loss : throw new InvalidOperationException(FineTuningLossInput.NoTeacher);
        }

        public Tensor BatchLoss(PretrainedModel model, Tensor tokens, Tensor hidden, OutputHead head, IReadOnlyList<TrainingSequence> sequences, BatchData data,
            int stepTokens, int stepSequences, int chunkRows, StepImages? images = null)
        {
            int[] targets = [.. data.Targets.Select(t => (int)t)];
            var values = Losses.TokenLogProbabilities(hidden, h => head.Forward(h), data.Trained, targets, chunkRows);
            bool training = Autograd.IsEnabled;
            var probabilities = Tensor.From(values, [values.Length], Device.Cpu, requiresGrad: training);
            ITeacherDistributions? distributions = null;
            Func<int, int, Tensor> Teacher(float temperature) => (start, count) =>
            {
                distributions ??= teacher!.Distributions(sequences);
                return distributions.Probabilities(start, count, temperature, head.OutFeatures, hidden.Device);
            };

            try
            {
                Func<float, ITeacherDivergence?, float, float[]>? divergence = teacher is null ? null
                    : (temperature, kind, beta) => Tensor.TokenDivergences(hidden, h => head.Forward(h), data.Trained, Teacher(temperature), temperature, chunkRows,
                        TeacherDivergences.Rule(kind, beta));
                var input = new FineTuningLossInput(probabilities, [.. sequences.Select(s => s.TrainedTokens)], sequences,
                    () => Reference(model, tokens, sequences, data, targets, chunkRows, images), pairs, stepTokens, stepSequences, divergence);
                var value = _loss.Compute(input) ?? throw new InvalidOperationException("FineTuningOptions.Loss returned no tensor.");
                if (value.Size != 1)
                {
                    throw new InvalidOperationException($"FineTuningOptions.Loss must return one value, not {Tensor.FormatShape(value.Shape)}.");
                }

                float result = value.Item();
                if (!training || !value.RequiresGrad)
                {
                    return Tensor.From([result], [1], hidden.Device);
                }

                value.Backward();
                var gradient = probabilities.Grad?.ToArray() ?? new float[values.Length];
                var weights = new float[values.Length];
                double surrogate = 0;
                for (int t = 0; t < values.Length; t++)
                {
                    weights[t] = -gradient[t];
                    surrogate += (double)gradient[t] * values[t];
                }

                // Each divergence weighted by its gradient: the value Σ g_t KL_t, the loss's gradient through it.
                Tensor? total = null;
                foreach (var (divergences, temperature, kind, beta) in input.Divergences)
                {
                    var g = divergences.Grad?.ToArray() ?? new float[values.Length];
                    var kl = divergences.ToArray();
                    for (int t = 0; t < g.Length; t++)
                    {
                        surrogate += (double)g[t] * kl[t];
                    }

                    var term = Tensor.TokenDivergenceRows(hidden, h => head.Forward(h), data.Trained, g, Teacher(temperature), temperature, chunkRows,
                        TeacherDivergences.Rule(kind, beta));
                    total = total is null ? term : total + term;
                }

                // The cross-entropy weighted by −g has the loss's gradient and the value Σ g_t log p_t (skipped when the loss
                // reads only divergences); shifted to the loss's value.
                if (total is null || weights.Any(w => w != 0f))
                {
                    var weighted = Losses.TokenCrossEntropyRows(hidden, h => head.Forward(h), data.Trained, data.Targets, weights, 1f, chunkRows);
                    total = total is null ? weighted : total + weighted;
                }

                return total + (float)(result - surrogate);
            }
            finally
            {
                distributions?.Dispose();
            }
        }

        // A session's step (FineTuningSession.Step): the references kept for sequences this step does not train are dropped,
        // so a run that brings new sequences every step holds one step's references, not every step's.
        internal void Keep(IReadOnlyList<TrainingSequence> sequences)
        {
            if (_reference.Count == 0)
            {
                return;
            }

            var kept = new HashSet<TrainingSequence>(sequences);
            foreach (var sequence in _reference.Keys.Where(s => !kept.Contains(s)).ToList())
            {
                _reference.Remove(sequence);
            }
        }

        // The trained tokens' log-probabilities under the base model (adapters disabled, no gradients, dropout off), one
        // forward pass of the batch the first time its sequences are seen.
        private float[] Reference(PretrainedModel model, Tensor tokens, IReadOnlyList<TrainingSequence> sequences, BatchData data, int[] targets, int chunkRows,
            StepImages? images)
        {
            if (sequences.All(_reference.ContainsKey))
            {
                return [.. sequences.SelectMany(s => _reference[s])];
            }

            var network = model.Network;
            bool wasTraining = network.IsTraining;
            float[] values;
            using (Autograd.NoGrad())
            using (network.DisableAdapters())
            {
                network.Eval();
                try
                {
                    values = NetworkLoss(model, tokens, (hidden, head) => Losses.TokenLogProbabilities(hidden, h => head.Forward(h), data.Trained, targets, chunkRows),
                        checkpointing: false, images);
                }
                finally
                {
                    network.Train(wasTraining);
                }
            }

            for (int s = 0, at = 0; s < sequences.Count; at += sequences[s].TrainedTokens, s++)
            {
                _reference[sequences[s]] = values[at..(at + sequences[s].TrainedTokens)];
            }

            return values;
        }
    }

    // A batch: rows of sequences (one per row when padded, several when packed), each row Length positions.
    internal sealed record Batch(int[][] Rows, int Length, bool Packed)
    {
        public IEnumerable<int> Sequences => Rows.SelectMany(r => r);

        public static Batch Padded(int[] sequences, IReadOnlyList<TrainingSequence> all) =>
            new([.. sequences.Select(i => new[] { i })], sequences.Max(i => all[i].Tokens.Length) - 1, false);

        public string Describe(IReadOnlyList<TrainingSequence> all) => (Packed
            ? $"{Rows.Sum(r => r.Length)} sequences packed in {Rows.Length} × {Length} positions ({Rows.Sum(r => r.Sum(i => all[i].Tokens.Length - 1)) * 100.0 / (Rows.Length * Length):F0}% filled)"
            : $"{Rows.Length} sequences × {Length} tokens") + DescribeImages(all);

        // Whether a sequence of the batch holds images.
        public bool HasImages(IReadOnlyList<TrainingSequence> all) => Rows.Any(r => r.Any(i => all[i].Images.Count > 0));

        // The image tokens of the batch's sequences.
        public int ImageTokens(IReadOnlyList<TrainingSequence> all) => Sequences.Sum(i => all[i].Images.Sum(image => image.Tokens));

        // Whether a sequence of the batch holds soft parts other than images (an audio part).
        public bool HasParts(IReadOnlyList<TrainingSequence> all) => Rows.Any(r => r.Any(i => all[i].Parts.Count > 0));

        private string DescribeImages(IReadOnlyList<TrainingSequence> all)
        {
            if (!HasImages(all))
            {
                return "";
            }

            var images = Sequences.SelectMany(i => all[i].Images).ToList();
            return $", {images.Count} images ({images.Sum(i => i.Tokens)} image tokens{(Packed ? ", packed with the others" : "")})";
        }
    }

    // The training batches: packed when asked and supported, else padded. Sequences with images pack like the others (each
    // keeps its images at its own positions and attends within itself by the family's rule); when every sequence holds
    // images under a rule that is not causal, every position attends by key ranges, so the span kernels decide.
    internal static List<Batch> MakeBatches(PretrainedModel model, IReadOnlyList<TrainingSequence> train, FineTuningOptions options, Random? random)
    {
        bool spans = options.Vision is { Attention.Causal: false } && train.Count > 0 && train.All(s => s.Images.Count > 0)
                     && PackedSequences.SupportsImages(model.Network);
        if (options.Packing && (PackedSequences.Supports(model.Network) || spans))
        {
            int longest = train.Max(s => s.Tokens.Length - 1);
            int context = model.Network.Descendants().Where(m => m is IAttentionCacheInfo).OfType<IPositionLimit>().Min(a => a.MaxPositions);   // the attention layers' (not a position table's)
            int length = Math.Max(longest, Math.Min(options.BatchTokens, context));
            int rows = Math.Max(1, options.BatchTokens / length);
            return [.. PackedBatches(train, length, rows, random).Select(b => new Batch(b, length, true))];
        }

        return [.. Batches(train, options.BatchTokens, random).Select(b => Batch.Padded(b, train))];
    }

    /// <summary>
    /// Packs sequences into batches of <paramref name="rows"/> rows of <paramref name="length"/> positions each (first fit,
    /// longest first; a row holds sequences whose token counts minus one, the positions they fill, add up to at most
    /// <paramref name="length"/>). Returns each batch's rows of sequence indices; the batch order is shuffled when
    /// <paramref name="random"/> is given. Sequences longer than a row are left out.
    /// </summary>
    public static List<int[][]> PackedBatches(IReadOnlyList<TrainingSequence> sequences, int length, int rows, Random? random)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        // Sequences of one length in a random order (not the list's): neighbours in the list (a repeated answer, one
        // source file) would otherwise fill whole rows, and a batch of one kind teaches that kind, not the distinction.
        var tie = TieOrder(sequences.Count, random);
        var order = Enumerable.Range(0, sequences.Count).Where(i => sequences[i].Tokens.Length - 1 is >= 1 and var n && n <= length)
            .OrderByDescending(i => sequences[i].Tokens.Length).ThenBy(i => tie[i]).ToList();
        var bins = new List<List<int>>();
        var free = new List<int>();
        // First fit: the first row with room (rows are kept in creation order; `fullest` skips rows too full for
        // anything, which grow with the list, so the search stays short).
        int fullest = 0;
        int shortest = order.Count == 0 ? 1 : sequences[order[^1]].Tokens.Length - 1;
        foreach (int index in order)
        {
            int n = sequences[index].Tokens.Length - 1;
            while (fullest < bins.Count && free[fullest] < shortest)
            {
                fullest++;
            }

            int bin = -1;
            for (int b = fullest; b < bins.Count; b++)
            {
                if (free[b] >= n)
                {
                    bin = b;
                    break;
                }
            }

            if (bin < 0)
            {
                bins.Add([]);
                free.Add(length);
                bin = bins.Count - 1;
            }

            bins[bin].Add(index);
            free[bin] -= n;
        }

        if (random is not null)
        {
            for (int i = bins.Count - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                (bins[i], bins[j]) = (bins[j], bins[i]);
            }
        }

        var batches = new List<int[][]>();
        for (int first = 0; first < bins.Count; first += rows)
        {
            batches.Add([.. bins.Skip(first).Take(rows).Select(b => b.ToArray())]);
        }

        return batches;
    }

    /// <summary>
    /// Groups sequences of similar length into batches of at most <paramref name="batchTokens"/> padded tokens (one
    /// sequence per batch when a sequence alone exceeds it); the batch order is shuffled when <paramref name="random"/> is given.
    /// </summary>
    public static List<int[]> Batches(IReadOnlyList<TrainingSequence> sequences, int batchTokens, Random? random)
    {
        var tie = TieOrder(sequences.Count, random);
        var order = Enumerable.Range(0, sequences.Count).OrderBy(i => sequences[i].Tokens.Length).ThenBy(i => tie[i]).ToList();
        var batches = new List<int[]>();
        var current = new List<int>();
        int longest = 0;
        foreach (int index in order)
        {
            int length = sequences[index].Tokens.Length - 1;
            if (length < 1)
            {
                continue;
            }

            if (current.Count > 0 && (current.Count + 1) * Math.Max(longest, length) > batchTokens)
            {
                batches.Add([.. current]);
                current.Clear();
                longest = 0;
            }

            current.Add(index);
            longest = Math.Max(longest, length);
        }

        if (current.Count > 0)
        {
            batches.Add([.. current]);
        }

        if (random is not null)
        {
            for (int i = batches.Count - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                (batches[i], batches[j]) = (batches[j], batches[i]);
            }
        }

        return batches;
    }

    // The rank of each sequence among those of its length: random when a random source is given (see PackedBatches), else
    // the list's order.
    private static int[] TieOrder(int count, Random? random)
    {
        var rank = Enumerable.Range(0, count).ToArray();
        if (random is not null)
        {
            random.Shuffle(rank);
        }

        return rank;
    }

    // The summed weighted loss of one batch divided by normalizer (padding and untrained positions weigh 0), and the
    // number of tokens it covers. The network runs up to its final normalization; the head runs inside the loss.
    // The host side of a batch: inputs [rows · length], and the trained positions with their targets and weights.
    internal sealed record BatchData(float[] Inputs, int[] Trained, float[] Targets, float[] Weights, long Tokens);

    internal static BatchData Prepare(IReadOnlyList<TrainingSequence> sequences, Batch batch, float weight = 1f)
    {
        int length = batch.Length;
        int rows = batch.Rows.Length;
        var inputs = new float[rows * length];
        var trained = new List<int>();
        var targets = new List<float>();
        long tokens = 0;
        for (int b = 0; b < rows; b++)
        {
            int offset = b * length;
            foreach (int index in batch.Rows[b])
            {
                var sequence = sequences[index];
                for (int t = 0; t + 1 < sequence.Tokens.Length; t++)
                {
                    inputs[offset + t] = sequence.Tokens[t];
                    if (sequence.Trained[t + 1])
                    {
                        // The output layer and softmax run only on the trained positions (the assistant's tokens): prompts
                        // and padding would only cost a vocabulary-wide product each.
                        trained.Add(offset + t);
                        targets.Add(sequence.Tokens[t + 1]);
                    }

                    tokens++;
                }

                offset += sequence.Tokens.Length - 1;
            }
        }

        return new BatchData(inputs, [.. trained], [.. targets], [.. Enumerable.Repeat(weight, trained.Count)], tokens);
    }

    // The summed weighted loss of one batch divided by normalizer (untrained positions weigh 0), and the number of tokens
    // it covers. Packed batches run under their PackedSequences (the caller's), padded ones as they are.
    // With a loss of one's own (custom), normalizer is the step's trained tokens and stepSequences its sequences.
    // With images, their features come through `vision` (made in the caller's tensor scope, released with the batch).
    private static (Tensor Loss, long Tokens) BatchLoss(PretrainedModel model, IReadOnlyList<TrainingSequence> sequences, Batch batch, float normalizer,
        int chunkRows, bool checkpointing = false, CustomLoss? custom = null, int stepSequences = 0, TuningVision? vision = null, IReadOnlyList<ISoftPartSource>? parts = null)
    {
        var data = Prepare(sequences, batch);
        var images = StepImages.Of(sequences, batch, vision, checkpointing, parts);
        var tokens = Tensor.From(data.Inputs, [batch.Rows.Length, batch.Length], model.Device);
        var loss = custom is null
            ? NetworkLoss(model, tokens, (hidden, head) => Losses.TokenCrossEntropyRows(hidden, h => head.Forward(h), data.Trained, data.Targets,
                data.Weights, normalizer, chunkRows), checkpointing, images)
            : NetworkLoss(model, tokens, (hidden, head) => custom.BatchLoss(model, tokens, hidden, head, [.. batch.Sequences.Select(i => sequences[i])], data,
                (int)normalizer, stepSequences, chunkRows, images), checkpointing, images);
        return (loss, data.Tokens);
    }

    // log p of every trained token of a batch under the network as it is (a session's LogProbabilities: the caller holds
    // the batch's packing, gradients off and the adapters as asked), in the batch's order (its rows, each row's sequences).
    internal static float[] TrainedLogProbabilities(PretrainedModel model, IReadOnlyList<TrainingSequence> sequences, Batch batch, FineTuningOptions options)
    {
        var data = Prepare(sequences, batch);
        var images = StepImages.Of(sequences, batch, options.Vision, checkpointing: false, options.Parts);
        var tokens = Tensor.From(data.Inputs, [batch.Rows.Length, batch.Length], model.Device);
        int[] targets = [.. data.Targets.Select(t => (int)t)];
        return NetworkLoss(model, tokens, (hidden, head) => Losses.TokenLogProbabilities(hidden, h => head.Forward(h), data.Trained, targets, options.LossChunkRows),
            checkpointing: false, images);
    }

    // The images of a batch for the decoder's pass: each image's features (once per distinct image and preparation in the
    // batch, all of them through the vision side together and its cache) at its blocks' positions in its row (after the
    // sequences before it, in a packed row), and the family's attention rule. With fixed inputs (a recorded step) the pass
    // reads the features from them.
    internal sealed record StepImages(IReadOnlyList<PromptImage> Images, IImageAttentionRule Attention)
    {
        public ImagePrefillInputs? Inputs { get; init; }

        public static StepImages? Of(IReadOnlyList<TrainingSequence> sequences, Batch batch, TuningVision? vision, bool checkpointing = false,
            IReadOnlyList<ISoftPartSource>? parts = null)
        {
            bool hasImages = batch.HasImages(sequences), hasParts = batch.HasParts(sequences);
            if (!hasImages && !hasParts)
            {
                return null;
            }

            var images = new List<PromptImage>();
            IImageAttentionRule? attention = null;
            if (hasImages)
            {
                if (vision is null)
                {
                    throw new InvalidOperationException("The sequences hold images and no vision side is given (FineTuningOptions.Vision, TuningVision.Create).");
                }

                PlaceImages(sequences, batch, vision, checkpointing, images);
                attention = vision.Attention;
            }

            if (hasParts)
            {
                attention = PlaceParts(sequences, batch, parts ?? [], checkpointing, images, attention);
                images.Sort((x, y) => x.Sequence != y.Sequence ? x.Sequence.CompareTo(y.Sequence) : x.Position.CompareTo(y.Position));   // the prompt's order, images and parts together
            }

            return new StepImages(images, attention!);
        }

        // Each image's features (once per distinct image and preparation) at its blocks' positions.
        private static void PlaceImages(IReadOnlyList<TrainingSequence> sequences, Batch batch, TuningVision vision, bool checkpointing, List<PromptImage> images)
        {
            var distinct = new List<TrainingImage>();
            var index = new Dictionary<(string, TuningImages), int>();
            foreach (var image in batch.Sequences.SelectMany(i => sequences[i].Images))
            {
                if (index.TryAdd((image.Image.Hash, image.Preparation), distinct.Count))
                {
                    distinct.Add(image);
                }
            }

            var features = vision.Features(distinct, checkpointing);
            for (int row = 0; row < batch.Rows.Length; row++)
            {
                int offset = 0;
                foreach (int sequence in batch.Rows[row])
                {
                    foreach (var image in sequences[sequence].Images)
                    {
                        var all = features[index[(image.Image.Hash, image.Preparation)]];
                        int at = 0, b = 0;
                        foreach (var (position, count) in image.Blocks)
                        {
                            var block = at == 0 && count == all.Shape[0] ? all : all.Narrow(0, at, count);
                            var axes = image.AxisPositions?[b];          // a multi-axis family's positions, read on the host
                            images.Add(new PromptImage(offset + position, block)
                            {
                                Sequence = row, Positions = axes is null ? null : Tensor.From(axes, [axes.Length / count, count], Device.Cpu),
                            });
                            at += count;
                            b++;
                        }
                    }

                    offset += sequences[sequence].Tokens.Length - 1;
                }
            }
        }

        // The soft parts other than images (an audio part's): each kind's distinct parts encoded together through its source
        // (with gradients when the source trains), each block's features at its position. The parts and the images must
        // attend by one rule; it is returned.
        private static IImageAttentionRule PlaceParts(IReadOnlyList<TrainingSequence> sequences, Batch batch, IReadOnlyList<ISoftPartSource> sources, bool checkpointing,
            List<PromptImage> images, IImageAttentionRule? attention)
        {
            var kinds = new Dictionary<string, (ISoftPartSource Source, List<ChatPart> Distinct, Dictionary<ChatPart, int> Index)>(StringComparer.Ordinal);
            foreach (int i in batch.Sequences)
            {
                foreach (var part in sequences[i].Parts)
                {
                    if (!kinds.TryGetValue(part.Kind, out var entry))
                    {
                        var source = sources.FirstOrDefault(s => s.Kind == part.Kind) ?? throw new InvalidOperationException(
                            $"Sequence {i} holds a '{part.Kind}' part and the step has no source for it: give it in FineTuningOptions.Parts (the Pretrainer opens the model's parts itself).");
                        entry = (source, [], []);
                        kinds[part.Kind] = entry;
                    }

                    if (entry.Index.TryAdd(part.Part, entry.Distinct.Count))
                    {
                        entry.Distinct.Add(part.Part);
                    }
                }
            }

            var features = new Dictionary<(string Kind, int Part), List<Tensor>>();
            foreach (var (kind, (source, distinct, _)) in kinds)
            {
                if (attention is not null && attention.Name != source.Attention.Name)
                {
                    throw new NotSupportedException($"The batch's '{kind}' parts attend by '{source.Attention.Name}' and its other soft tokens by '{attention.Name}': one rule per step.");
                }

                attention ??= source.Attention;
                var encoded = source is ITrainableSoftParts { Parameters.Count: > 0 } trainable ? trainable.EncodeTraining(distinct, checkpointing) : source.Encode(distinct);
                int at = 0;
                for (int p = 0; p < distinct.Count; p++)
                {
                    int blocks = source.Blocks(distinct[p]).Count;
                    features[(kind, p)] = [.. encoded.Skip(at).Take(blocks).Select(f => f.Features)];
                    at += blocks;
                }

                if (at != encoded.Count)
                {
                    throw new InvalidOperationException($"The '{kind}' source gave {encoded.Count} blocks of features for {distinct.Count} parts of {at} blocks.");
                }
            }

            for (int row = 0; row < batch.Rows.Length; row++)
            {
                int offset = 0;
                foreach (int sequence in batch.Rows[row])
                {
                    foreach (var part in sequences[sequence].Parts)
                    {
                        var blocks = features[(part.Kind, kinds[part.Kind].Index[part.Part])];
                        if (blocks.Count != part.Blocks.Count)
                        {
                            throw new InvalidOperationException($"A '{part.Kind}' part was laid out in {part.Blocks.Count} blocks and its source encodes {blocks.Count}.");
                        }

                        for (int b = 0; b < blocks.Count; b++)
                        {
                            var (position, count) = part.Blocks[b];
                            if (blocks[b].Shape[0] != count)
                            {
                                throw new InvalidOperationException($"A '{part.Kind}' part's block {b} takes {count} soft tokens and its source gave {blocks[b].Shape[0]} rows of features.");
                            }

                            images.Add(new PromptImage(offset + position, blocks[b]) { Sequence = row });
                        }
                    }

                    offset += sequences[sequence].Tokens.Length - 1;
                }
            }

            return attention!;
        }
    }

    // The network's output head as the losses read it: the module (its forward pass) and its sizes (ILinearLayer), found
    // once a pass by NetworkLoss.
    internal readonly record struct OutputHead(Module Module, ILinearLayer Linear)
    {
        public Tensor Forward(Tensor hidden) => Module.Forward(hidden);

        public int OutFeatures => Linear.OutFeatures;
    }

    // The network up to its final normalization on tokens [rows, length], then `loss` of the hidden states [rows · length,
    // dim] with the output head (which the loss runs itself, on the rows it needs). With images, the decoder attends by the
    // family's rule and the first decoder block reads the embeddings with the images' features in (checkpointed blocks
    // recompute with the same image blocks in the backward pass).
    private static T NetworkLoss<T>(PretrainedModel model, Tensor tokens, Func<Tensor, OutputHead, T> loss, bool checkpointing, StepImages? images = null)
    {
        var modules = model.Network.ToList();
        if (modules[^1] is not ILinearLayer linear)
        {
            throw new InvalidOperationException("The network does not end with its output head (a linear layer, ILinearLayer).");
        }

        var head = new OutputHead(modules[^1], linear);

        using var imageScope = images is null ? null : images.Inputs is { } inputs ? inputs.Begin() : model.Network.Begin(tokens, images.Images, images.Attention);
        var hidden = tokens;
        for (int i = 0; i < modules.Count - 1; i++)
        {
            if (imageScope is not null && i == imageScope.FirstBlock)
            {
                var substituted = imageScope.Substitute(hidden);
                if (!Autograd.IsEnabled && !ReferenceEquals(hidden, tokens))
                {
                    hidden.Dispose();                                        // evaluation frees each activation once read
                }

                hidden = substituted;
            }

            if (checkpointing && modules[i] is ICheckpointable && Autograd.IsEnabled)
            {
                hidden = modules[i].ForwardCheckpointed(hidden);                // keeps only the block's output
            }
            else if (!Autograd.IsEnabled)
            {
                // Evaluation: each module's intermediate results, and the previous activation, are freed at once.
                Tensor next;
                using (var scope = new TensorScope())
                {
                    next = scope.Keep(modules[i].Forward(hidden));
                }

                if (!ReferenceEquals(next, hidden) && !ReferenceEquals(hidden, tokens))
                {
                    hidden.Dispose();
                }

                hidden = next;
            }
            else
            {
                var input = hidden;
                hidden = modules[i].Forward(hidden);

                // A block's input is read by no backward step once the block has run when its first norm is an RMS norm
                // (which keeps its own normalized values), nor the final norm's input: released now, not at the step's end.
                if (i > 0 && !ReferenceEquals(input, hidden) && modules[i] is DecoderBlock { AttentionNorm: RMSNorm } or RMSNorm)
                {
                    ActivationMemory.Release(input);
                }
            }
        }

        var flat = hidden.Reshape(tokens.Size, hidden.Shape[^1]);
        var result = loss(flat, head);
        if (!checkpointing)
        {
            ActivationMemory.Release(flat);                                  // the loss read its rows; its backward scatters only
        }

        return result;
    }

    // One training step's forward and backward pass over a packed batch, recorded as a graph: its inputs (tokens,
    // the packing's layout, the trained positions with targets and weights, and with images their features, where they go
    // and each row's key ranges) live in fixed buffers that each step overwrites before replaying the graph. The loss
    // weights carry 1 / trained tokens, so the recorded pass needs no per-batch constant. Parameters' gradients are the
    // ones the graph zeroes and fills. A graph with images serves batches of its shape holding up to its image tokens;
    // their features are computed (or taken from the cache) before each replay, without gradients.
    private sealed class TrainingGraph : IDisposable
    {
        private readonly PretrainedModel _model;
        private readonly Tensor _tokens, _rows, _targets, _weights, _loss;
        private readonly PackedSequences _packing;
        private ImagePrefillInputs? _images;
        private TuningVision? _vision;
        private Batch? _imagesOf;                                         // the batch whose features _images holds
        private IntPtr _executable, _graph;
        private Idrak.Abstraction.Devices.IGraphCaptureBackend? _graphs;      // the device's recording, once recorded
        private List<Idrak.Abstraction.Devices.Storage> _owned = [];

        private TrainingGraph(PretrainedModel model, Batch batch, IReadOnlyList<TrainingSequence> train, int lossRows)
        {
            _model = model;
            var device = model.Device;
            _tokens = Tensor.Persistent(new float[batch.Rows.Length * batch.Length], [batch.Rows.Length, batch.Length], device, requiresGrad: false);
            _rows = Tensor.Persistent(new float[lossRows], [lossRows], device, requiresGrad: false);
            _targets = Tensor.Persistent(new float[lossRows], [lossRows], device, requiresGrad: false);
            _weights = Tensor.Persistent(new float[lossRows], [lossRows], device, requiresGrad: false);
            _loss = Tensor.Persistent([0f], [1], device, requiresGrad: false);
            _packing = PackedSequences.Create(Lengths(batch, train), batch.Length, device);
            LossRows = lossRows;
        }

        public int LossRows { get; }

        // The image tokens the graph holds features for (0: a graph without images).
        public int ImageTokens => _images?.ImageTokens ?? 0;

        // The device memory the graph keeps: what it allocated while recording, and its image inputs.
        public long Bytes => _owned.Sum(s => 4L * s.Length) + (_images?.Bytes ?? 0);

        private static int[][] Lengths(Batch batch, IReadOnlyList<TrainingSequence> train) =>
            [.. batch.Rows.Select(r => r.Select(i => train[i].Tokens.Length - 1).ToArray())];

        // Records the pass (nothing runs yet); null, with the reason traced, when the device cannot.
        // imageTokens: the image tokens the recorded step holds features for (at least the batch's; a later batch with no more replays it).
        public static TrainingGraph? Record(PretrainedModel model, IReadOnlyList<TrainingSequence> train, Optimizer optimizer, FineTuningOptions options, Batch batch,
            int lossRows, int imageTokens, Action<string>? trace)
        {
            var graph = new TrainingGraph(model, batch, train, lossRows);
            var backend = model.Device.Backend;
            StepImages? images = null;
            if (batch.HasImages(train))
            {
                // The features are values (no vision part trains): computed now, copied into the fixed inputs the recorded
                // pass reads, as before every replay.
                try
                {
                    var vision = options.Vision!;
                    graph._vision = vision;
                    graph._images = ImagePrefillInputs.Create(model.Network, batch.Rows.Length, batch.Length, Math.Max(imageTokens, batch.ImageTokens(train)),
                        vision.Attention);
                    using var scope = new TensorScope();
                    using (Autograd.NoGrad())
                    {
                        var step = StepImages.Of(train, batch, vision)!;
                        graph._images.Load(step.Images, graph._packing);
                        graph._imagesOf = batch;
                        images = step with { Inputs = graph._images };
                    }
                }
                catch (Exception ex) when (ex is not (OutOfMemoryException or ResourceLimitExceededException))
                {
                    graph.Dispose();
                    Failed(model, ex, trace);
                    return null;
                }
            }

            if (backend is not Idrak.Abstraction.Devices.IGraphCaptureBackend graphs)
            {
                graph.Dispose();
                trace?.Invoke($"graph not used ({backend.Name} records no graphs); ordinary steps continue");
                return null;
            }

            model.Device.Synchronize();
            model.Network.Train();
            // Nothing may be freed while a graph records, so blocks the library keeps cached cannot make room for it: a
            // recording out of device memory with blocks cached gives them back and records once more.
            for (int attempt = 0; ; attempt++)
            {
                try
                {
                    graphs.BeginCapture();
                }
                catch (Exception ex) when (ex is not OutOfMemoryException)
                {
                    graph.Dispose();
                    Failed(model, ex, trace);
                    return null;
                }

                try
                {
                    optimizer.ZeroGrad();
                    using (var scope = new TensorScope())
                    using (graph._packing.Use())
                    using (options.RecomputeFeedForward == true ? ActivationMemory.Recompute() : (ActivationMemory.Scope?)null)
                    using (options.BFloat16Activations == true ? ActivationMemory.CompressToBFloat16() : (ActivationMemory.Scope?)null)
                    {
                        var loss = NetworkLoss(model, graph._tokens, (hidden, head) => Tensor.TokenCrossEntropyRows(hidden, h => head.Forward(h),
                            graph._rows, graph._targets, graph._weights, 1f, options.LossChunkRows), options.Checkpointing == true, images);
                        loss.Backward();
                        backend.Copy(loss.Storage, graph._loss.Storage, 1);
                    }

                    (graph._executable, graph._graph, graph._owned) = graphs.EndCapture();
                    graph._graphs = graphs;
                    trace?.Invoke($"recorded one training step as a graph ({batch.Rows.Length} × {batch.Length} positions, up to {lossRows} trained"
                                  + (graph._images is { } inputs ? $", up to {inputs.ImageTokens} image tokens" : "") + $", {graph.Bytes / 1048576.0:F0} MB); later steps of that shape replay it");
                    return graph;
                }
                catch (ResourceLimitExceededException) when (attempt == 0 && backend.GetMemoryUsage().Cached is long cached && cached > 0)
                {
                    foreach (var block in graphs.AbortCapture())
                    {
                        block.Release();
                    }

                    ComputeResources.ReleaseCachedMemory(model.Device);
                    trace?.Invoke($"recording the step ran out of device memory with {cached / 1048576.0:F0} MB of freed blocks cached; they go back to the "
                                  + "device and the step is recorded again");
                }
                catch (Exception ex) when (ex is not OutOfMemoryException)
                {
                    graph._owned = graphs.AbortCapture();
                    graph.Dispose();
                    Failed(model, ex, trace);
                    return null;
                }
            }
        }

        // A recording that failed: traced, and published (GraphRecordingFailed) so --verbose and --profile show it with its reason.
        private static void Failed(PretrainedModel model, Exception ex, Action<string>? trace)
        {
            trace?.Invoke($"graph not used ({ex.Message}); ordinary steps continue");
            Telemetry.GraphRecordingFailed(new GraphRecordingFailed(model.Device.Backend.Kind, "a training step", ex.Message));
        }

        // Same shape, trained positions within the recorded capacity, and images (or none) as recorded, their tokens within
        // the recorded features' buffer.
        public bool Fits(Batch batch, IReadOnlyList<TrainingSequence> train) =>
            batch.Packed && batch.Rows.Length == _packing.Rows && batch.Length == _packing.Length
            && batch.Sequences.Sum(i => train[i].TrainedTokens) <= LossRows
            && (batch.HasImages(train) ? _images is { } inputs && inputs.Fits(batch.Rows.Length, batch.Length, batch.ImageTokens(train)) : _images is null);

        // Copies the batch in and replays the pass: the parameters' gradients are then set; returns the mean loss per
        // trained token and the tokens covered.
        public (float Loss, long Tokens) Run(IReadOnlyList<TrainingSequence> train, Batch batch)
        {
            int trained = batch.Sequences.Sum(i => train[i].TrainedTokens);
            var data = Prepare(train, batch, 1f / Math.Max(1, trained));
            var rows = new float[LossRows];
            var targets = new float[LossRows];
            var weights = new float[LossRows];
            for (int i = 0; i < data.Trained.Length; i++)
            {
                rows[i] = data.Trained[i];
                targets[i] = data.Targets[i];
                weights[i] = data.Weights[i];
            }

            _tokens.Load(data.Inputs);
            _rows.Load(rows);
            _targets.Load(targets);
            _weights.Load(weights);
            _packing.Update(Lengths(batch, train));
            // The features again only for another batch: the step right after recording, and a batch replayed again before
            // any other, find theirs in place (no part of the image side trains in a recorded step, so they are the same).
            if (_images is not null && !ReferenceEquals(_imagesOf, batch))
            {
                using var scope = new TensorScope();
                using (Autograd.NoGrad())
                {
                    _images.Load(StepImages.Of(train, batch, _vision)!.Images, _packing);
                }

                _imagesOf = batch;
            }

            _graphs!.ReplayGraph(_executable);
            return (_loss.Item(), data.Tokens);
        }

        public void Dispose()
        {
            if (_executable != IntPtr.Zero)
            {
                _model.Device.Synchronize();
                _graphs!.DestroyGraph(_executable, _graph);
                _executable = IntPtr.Zero;
            }

            foreach (var block in _owned)
            {
                block.Release();
            }

            _owned = [];
            _tokens.Dispose();
            _rows.Dispose();
            _targets.Dispose();
            _weights.Dispose();
            _loss.Dispose();
            _packing.Dispose();
            _images?.Dispose();
        }
    }
}
