// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Datasets;
using Idrak.Diagnostics;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Optimizers;

namespace Idrak.LanguageModels;

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
    /// (content as a string or a list of text parts; reasoning in reasoning_content, thinking or reasoning; tool calls as
    /// [{"function": {"name", "arguments"}}] with arguments as an object or a JSON string; tool results with role "tool"
    /// and a name or a tool_call_id), or ShareGPT's <c>{"conversations": [{"from": "human" | "gpt" | "system", "value"}]}</c>.
    /// Optional "enable_thinking" / "think" sets <see cref="Think"/>.
    /// </summary>
    public static ChatTranscript FromJson(JsonObject json)
    {
        var messages = new List<ChatMessage>();
        var callNames = new Dictionary<string, string>(StringComparer.Ordinal);
        if (json["messages"] is JsonArray list)
        {
            foreach (var node in list)
            {
                messages.Add(Message(node as JsonObject ?? throw new InvalidDataException("A message is not an object."), callNames));
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
                messages.Add(new ChatMessage(role, (string?)node?["value"] ?? ""));
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

    private static ChatMessage Message(JsonObject m, Dictionary<string, string> callNames)
    {
        string role = (string?)m["role"] ?? throw new InvalidDataException("A message has no role.");
        string content = m["content"] switch
        {
            null => "",
            JsonValue v => (string?)v ?? "",
            JsonArray parts => string.Concat(parts.Select(p => p is JsonValue pv ? (string?)pv : (string?)p?["text"] ?? "")),
            _ => throw new InvalidDataException($"Unsupported content in a {role} message."),
        };
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
        int number = 0;
        foreach (string line in File.ReadLines(path))
        {
            number++;
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            ChatTranscript transcript;
            try
            {
                transcript = FromJson(JsonNode.Parse(line) as JsonObject ?? throw new InvalidDataException("The line is not a JSON object."));
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
/// One tokenized training sequence: its token ids and, for each token, whether the model learns to produce it (the
/// assistant's turns: reasoning, text, tool calls and the end-of-turn marker; not the system, user or tool messages).
/// </summary>
public sealed record TrainingSequence(int[] Tokens, bool[] Trained)
{
    private (bool[]? Of, int Count) _trained;

    /// <summary>Tokens the loss is computed on (a token is predicted from the ones before it, so the first never is).</summary>
    public int TrainedTokens
    {
        get
        {
            // Counted once per Trained array (batching asks for it many times per step).
            var cached = _trained;
            if (!ReferenceEquals(cached.Of, Trained))
            {
                int count = 0;
                for (int i = 1; i < Trained.Length; i++)
                {
                    count += Trained[i] ? 1 : 0;
                }

                _trained = cached = (Trained, count);
            }

            return cached.Count;
        }
    }

    /// <summary>Equal when both hold the same token and trained arrays (the cached count is not compared).</summary>
    public bool Equals(TrainingSequence? other) =>
        other is not null && ReferenceEquals(Tokens, other.Tokens) && ReferenceEquals(Trained, other.Trained);

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Tokens, Trained);
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

    /// <summary>The rendered transcript and the character ranges [start, end) of the assistant's turns in it.</summary>
    public (string Text, IReadOnlyList<(int Start, int End)> Spans) Render(ChatTranscript transcript)
    {
        string text = Template.Render(transcript.Messages, transcript.Tools, transcript.Think, addGenerationPrompt: false);
        var spans = new List<(int, int)>();
        for (int at = text.IndexOf(AssistantHeader, StringComparison.Ordinal); at >= 0; at = text.IndexOf(AssistantHeader, at, StringComparison.Ordinal))
        {
            int start = at + AssistantHeader.Length;
            int end = AssistantEnd.Length == 0 ? -1 : text.IndexOf(AssistantEnd, start, StringComparison.Ordinal);
            end = end < 0 ? text.Length : end + AssistantEnd.Length;
            spans.Add((start, end));
            at = end;
        }

        return (text, spans);
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
    public TrainingSequence? Encode(ChatTranscript transcript, int maxLength)
    {
        var (tokens, trained) = Tokens(transcript);
        if (tokens.Count > maxLength + 1 && ShortenToFit && Shortened(transcript, maxLength) is { } fitted)
        {
            return fitted;
        }

        int keep = Math.Min(tokens.Count, maxLength + 1);
        var sequence = new TrainingSequence([.. tokens.Take(keep)], [.. trained.Take(keep)]);
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
        if (Tokens(transcript).Tokens.Count <= maxLength + 1)
        {
            return transcript;
        }

        return ShortenedTranscript(transcript, maxLength).Transcript;
    }

    private TrainingSequence? Shortened(ChatTranscript transcript, int maxLength)
    {
        var (fitted, tokens, trained) = ShortenedTranscript(transcript, maxLength);
        if (fitted is null)
        {
            return null;
        }

        var sequence = new TrainingSequence([.. tokens], [.. trained]);
        return sequence.TrainedTokens > 0 ? sequence : null;
    }

    // Cuts the end of the last user message before the last assistant turn until everything fits: a first cut from the
    // tokens over, then 10% less each time the estimate falls short (tokens do not map to characters exactly).
    private (ChatTranscript? Transcript, List<int> Tokens, List<bool> Trained) ShortenedTranscript(ChatTranscript transcript, int maxLength)
    {
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
            return (null, [], []);
        }

        string content = messages[user].Content;
        int total = Tokens(transcript).Tokens.Count;
        int contentTokens = Math.Max(1, Tokenizer.Encode(content).Count);
        int keepTokens = contentTokens - (total - (maxLength + 1)) - 2;
        int length = keepTokens <= 0 ? 0 : (int)((long)content.Length * keepTokens / contentTokens);
        while (true)
        {
            var shorter = messages.ToArray();
            shorter[user] = shorter[user] with { Content = content[..length] };
            var candidate = transcript with { Messages = shorter };
            var (tokens, trained) = Tokens(candidate);
            if (tokens.Count <= maxLength + 1)
            {
                return (candidate, tokens, trained);
            }

            if (length == 0)
            {
                return (null, [], []);
            }

            length = length * 9 / 10;
        }
    }

    // Every token of the rendered transcript, the assistant's turns marked as trained (only the last `answerTurns` of them
    // when given).
    private (List<int> Tokens, List<bool> Trained) Tokens(ChatTranscript transcript, int answerTurns = int.MaxValue)
    {
        var (text, spans) = Render(transcript);
        int firstTrained = Math.Max(0, spans.Count - answerTurns);
        var tokens = new List<int>();
        var trained = new List<bool>();
        void Add(int from, int to, bool train)
        {
            if (to > from)
            {
                int count = tokens.Count;
                if (Tokenizer is BpeTokenizer bpe)
                {
                    bpe.EncodeRange(text, from, to - from, tokens);                // no copy of the span's text
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
        return (tokens, trained);
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

            var (tokens, trained) = Tokens(transcript, turns);
            int keep = Math.Min(tokens.Count, maxLength + 1);
            var sequence = new TrainingSequence([.. tokens.Take(keep)], [.. trained.Take(keep)]);
            return sequence.TrainedTokens > 0 ? sequence : null;
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

    /// <summary>Layers that get adapters, by name: q, k, v, o (attention), gate, up, down (feed-forward), head.</summary>
    public IReadOnlyList<string> Targets { get; init; } = ["q", "k", "v", "o", "gate", "up", "down"];

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

    /// <summary>Rows of the output head computed at once by the loss (see <see cref="Losses.TokenCrossEntropy"/>).</summary>
    public int LossChunkRows { get; init; } = 512;

    /// <summary>Evaluate every this many optimizer steps (0: after each epoch).</summary>
    public int EvaluateEvery { get; init; }

    /// <summary>Save the adapters every this many optimizer steps (0: only at the end).</summary>
    public int SaveEvery { get; init; }

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
    /// or <see cref="FineTuningOptimizers.Create"/>. Null (the default): AdamW with <see cref="LearningRate"/> and
    /// <see cref="WeightDecay"/>. The rate the optimizer starts with is the peak the schedule works from. With
    /// <see cref="HostOptimizer"/> it receives the CPU copies of the parameters (one each). Recorded steps
    /// (<see cref="CudaGraphs"/>) record only the forward and backward pass, so they work with any optimizer that keeps the
    /// gradient buffers between steps (every built-in one except <see cref="Optimizers.HostOptimizer"/>); with one that
    /// does not, training runs ordinary steps.
    /// </summary>
    public Func<IReadOnlyList<Tensor>, Optimizer>? Optimizer { get; init; }

    /// <summary>
    /// Creates the learning-rate schedule from the optimizer and the number of optimizer steps planned; it is stepped once
    /// per optimizer step (built-in: <see cref="FineTuningSchedules"/>). Null (the default): a linear warm-up over
    /// <see cref="WarmupFraction"/> of the steps, then cosine decay to <see cref="MinLearningRate"/>.
    /// </summary>
    public Func<Optimizer, int, LearningRateScheduler>? Scheduler { get; init; }

    /// <summary>
    /// The loss of a batch from its trained tokens' log-probabilities (built-in: <see cref="FineTuningLosses"/>; DPO, ORPO
    /// and SimPO need <see cref="PreferencePair"/> data). Null (the default): the mean token cross-entropy per trained token,
    /// computed in one fused pass that steps can record as a graph; with a loss of one's own, steps run as ordinary ones
    /// and the output head runs twice over the trained rows (once for the log-probabilities, once for their gradients).
    /// Evaluation losses are this loss over the whole evaluation set. With a <see cref="Teacher"/> and no loss,
    /// <see cref="FineTuningLosses.Distillation"/> with its defaults.
    /// </summary>
    public FineTuningLoss? Loss { get; init; }

    /// <summary>
    /// The teacher for knowledge distillation (<see cref="DistillationTeacher.FromModel"/> or
    /// <see cref="DistillationTeacher.FromFile"/>), whose distributions a loss reads through
    /// <see cref="FineTuningLossInput.TeacherDivergence"/> (see <see cref="FineTuningLosses.Distillation"/>). It must share the
    /// student's vocabulary (<see cref="DistillationTeacher.Check"/>, which training runs first). Null: no teacher. Not
    /// disposed by training.
    /// </summary>
    public DistillationTeacher? Teacher { get; init; }
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
public sealed record FineTuningProgress(int Step, int TotalSteps, int Epoch, float Loss, float LearningRate, double TokensPerSecond, float? EvaluationLoss);

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
    // teacher is checked against the model first.
    private static CustomLoss? Custom(PretrainedModel model, FineTuningOptions options, bool pairs)
    {
        options.Teacher?.Check(model);
        var loss = options.Loss ?? (options.Teacher is not null ? FineTuningLosses.Distillation() : null);
        return loss is null ? null : new CustomLoss(loss, pairs, options.Teacher);
    }

    // Pair k as sequences 2k (chosen) and 2k + 1 (rejected).
    private static List<TrainingSequence> Flatten(IReadOnlyList<PreferencePair> pairs) =>
        [.. pairs.SelectMany(p => new[] { p.Chosen ?? throw new ArgumentException("A pair has no chosen answer."), p.Rejected ?? throw new ArgumentException("A pair has no rejected answer.") })];

    private static IReadOnlyList<float> TrainCore(PretrainedModel model, IReadOnlyList<TrainingSequence> train, IReadOnlyList<TrainingSequence>? evaluation,
        bool pairs, CustomLoss? custom, FineTuningOptions options, string? outputFolder, IProgress<FineTuningProgress>? progress, CancellationToken cancellationToken,
        Action<string>? trace)
    {
        var network = model.Network;
        if (!network.Descendants().OfType<Linear>().Any(l => l.Adapter is not null))
        {
            model.AddAdapters(options.Rank, options.Alpha, options.Targets, options.Seed, options.Dora);
        }

        var parameters = network.TrainableParameters().ToList();
        using var float8 = PrepareFloat8(model, train, options, trace);
        using var optimizer = CreateOptimizer(parameters, options);
        var random = new Random(options.Seed);
        var epochBatches = Enumerable.Range(0, options.Epochs)
            .Select(_ => pairs ? PairBatches(train, options.BatchTokens, random) : MakeBatches(model, train, options, random)).ToList();
        int accumulation = Math.Max(1, options.GradientAccumulation);
        int totalSteps = epochBatches.Sum(b => (b.Count + accumulation - 1) / accumulation);
        var schedule = (options.Scheduler ?? FineTuningSchedules.Cosine(options.WarmupFraction, options.MinLearningRate))(optimizer, Math.Max(1, totalSteps))
            ?? throw new InvalidOperationException("FineTuningOptions.Scheduler returned no schedule.");
        var evaluations = new List<float>();
        int step = 0;
        using var runner = new StepRunner(model, train, optimizer, options, MostTrained(epochBatches.SelectMany(b => b), train), custom, trace);
        for (int epoch = 0; epoch < options.Epochs; epoch++)
        {
            var batches = epochBatches[epoch];
            for (int first = 0; first < batches.Count; first += accumulation)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var group = batches.Skip(first).Take(accumulation).ToList();
                var watch = Stopwatch.StartNew();
                var (loss, tokens) = runner.Run(group, cancellationToken,
                    b => $"step {step + 1}/{totalSteps}, batch {first + b + 1}/{batches.Count} of epoch {epoch + 1}");
                float rate = optimizer.LearningRate;                     // the rate RunStep's update used
                schedule.Step();
                step++;
                float? evaluationLoss = null;
                bool lastOfEpoch = first + accumulation >= batches.Count;
                if (evaluation is { Count: > 0 } && (options.EvaluateEvery > 0 ? step % options.EvaluateEvery == 0 : lastOfEpoch))
                {
                    evaluationLoss = custom is null ? Evaluate(model, evaluation, options.BatchTokens, options.LossChunkRows, trace)
                        : EvaluateCustom(model, evaluation, pairs, custom, options, trace);
                    evaluations.Add(evaluationLoss.Value);
                }

                if (outputFolder is not null && options.SaveEvery > 0 && step % options.SaveEvery == 0)
                {
                    model.SaveAdapter(Path.Combine(outputFolder, $"checkpoint-{step}"));
                }

                progress?.Report(new FineTuningProgress(step, totalSteps, epoch + 1, loss, rate, tokens / Math.Max(1e-9, watch.Elapsed.TotalSeconds), evaluationLoss));
            }
        }

        network.Eval();
        if (outputFolder is not null)
        {
            model.SaveAdapter(outputFolder);
        }

        return evaluations;
    }

    // Runs optimizer steps: ordinary ones, or replays of a recorded graph once one is recorded (see
    // FineTuningOptions.CudaGraphs).
    // lossRows: the most trained tokens of any batch the runner will see (the recorded loss's capacity).
    // custom: a loss of one's own (FineTuningOptions.Loss), or null for the token cross-entropy.
    private sealed class StepRunner(PretrainedModel model, IReadOnlyList<TrainingSequence> train, Optimizer optimizer, FineTuningOptions options,
        int lossRows, CustomLoss? custom, Action<string>? trace) : IDisposable
    {
        // Not with offloading (tensors move between steps, a graph holds their addresses) or the CPU update. The recorded
        // pass is the token cross-entropy's; a loss of one's own runs ordinary steps. Experts' routing is read back to the
        // host on every pass, which a recording cannot.
        private readonly bool _graphsAllowed = options.CudaGraphs && options.GradientAccumulation <= 1
            && !ComputeResources.OffloadToHostMemory && !options.HostOptimizer
            && model.Device.Backend.SupportsGraphs
            && custom is null && Recordable(optimizer)
            && !model.Network.Descendants().Any(m => m is Dropout { Probability: > 0f } or MixtureOfExperts);   // a recorded pass would reuse dropout masks

        // An optimizer of one's own (FineTuningOptions.Optimizer): the recorded backward pass writes into the gradient
        // buffers the parameters had when it was recorded, so they must stay the parameters' between steps. Checked before
        // recording (each trained parameter still has its gradient after an update) and before every replay (the same
        // buffers); the built-in AdamW keeps them by construction.
        private readonly bool _checkGradients = options.Optimizer is not null;
        private Tensor?[] _gradients = [];
        private readonly bool _automatic = options.Checkpointing is null && !ComputeResources.OffloadToHostMemory;
        private FineTuningOptions _options = options with { Checkpointing = options.Checkpointing ?? ComputeResources.OffloadToHostMemory };
        private TrainingGraph? _graph;
        private bool? _graphs;
        private int _ordinary;

        private readonly bool _automaticRecompute = options.RecomputeFeedForward is null && options.Checkpointing is null && !ComputeResources.OffloadToHostMemory;
        private readonly bool _automaticBFloat16 = options.BFloat16Activations is null && options.Checkpointing is null && !ComputeResources.OffloadToHostMemory;

        /// <summary>Whether steps run with activation checkpointing (it can turn on during training, see <see cref="FineTuningOptions.Checkpointing"/>).</summary>
        public bool Checkpointing => _options.Checkpointing == true;

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
            // Out of device memory: first recompute the feed-forward activations (cheap), then hold activations as
            // bfloat16, then checkpoint every block (a third more compute); the step is run again each time and the graph,
            // recorded without them, recorded again. Which of the setting that fits and checkpointing is faster is measured.
            while (true)
            {
                try
                {
                    long started = Stopwatch.GetTimestamp();
                    var result = RunOnce(group, cancellationToken, label, graphs && _probe is 0 or 3);
                    Measured(Stopwatch.GetElapsedTime(started).TotalSeconds / Math.Max(1L, group.Sum(b => (long)b.Rows.Length * b.Length)));
                    return result;
                }
                catch (ResourceLimitExceededException ex) when (_probe == 2)
                {
                    Choose(checkpointing: false, $"checkpointing ran out of device memory ({ex.Message.Split(':')[0]})");
                }
                catch (ResourceLimitExceededException ex) when (_automaticRecompute && _options.RecomputeFeedForward != true && !Checkpointing)
                {
                    _options = _options with { RecomputeFeedForward = true };
                    Reset();
                    StartProbe();
                    trace?.Invoke($"out of device memory ({ex.Message.Split(':')[0]}): feed-forward activations are recomputed in the backward pass "
                                  + "from this step (RecomputeFeedForward / --recompute to start with it)");
                }
                catch (ResourceLimitExceededException ex) when (_automaticBFloat16 && _options.BFloat16Activations != true && !Checkpointing)
                {
                    _options = _options with { BFloat16Activations = true };
                    Reset();
                    StartProbe();
                    trace?.Invoke($"out of device memory ({ex.Message.Split(':')[0]}): activations are held as bfloat16 between the passes "
                                  + "from this step (BFloat16Activations / --bf16-activations to start with it)");
                }
                catch (ResourceLimitExceededException ex) when (_automatic && !Checkpointing)
                {
                    _options = _options with { Checkpointing = true };
                    Reset();
                    _probe = 3;                                                  // nothing lighter fits
                    trace?.Invoke($"out of device memory without activation checkpointing ({ex.Message.Split(':')[0]}): checkpointing is on from this step "
                                  + "(set Checkpointing / --checkpointing to start with it)");
                }
            }
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
            _graph?.Dispose();
            (_graph, _graphs, _ordinary) = (null, null, 0);
        }

        private (float Loss, long Tokens) RunOnce(IReadOnlyList<Batch> group, CancellationToken cancellationToken, Func<int, string> label, bool graphs)
        {
            var options = _options;
            _graphs ??= _graphsAllowed;
            var batch = group[0];
            if (graphs && _graphs == true && group.Count == 1 && batch.Packed)
            {
                if (_graph is null && _ordinary >= 2)
                {
                    if (_checkGradients && optimizer.Parameters.Any(p => p.RequiresGrad && p.Grad is null))
                    {
                        trace?.Invoke("graph not used (the optimizer does not keep the gradient buffers between steps); ordinary steps continue");
                        _graphs = false;
                        _ordinary++;
                        return RunStep(model, train, group, optimizer, options, custom, cancellationToken, trace, label);
                    }

                    _graph = TrainingGraph.Record(model, train, optimizer, options, batch, (Math.Max(1, lossRows) + 63) / 64 * 64, trace);
                    _graphs = _graph is not null;
                    _gradients = [.. optimizer.Parameters.Select(p => p.Grad)];
                }

                if (_graph is not null && _checkGradients && optimizer.Parameters.Where((p, i) => !ReferenceEquals(p.Grad, _gradients[i])).Any())
                {
                    trace?.Invoke("graph dropped (the optimizer replaced the gradient buffers the recorded step writes); ordinary steps continue");
                    _graph.Dispose();
                    (_graph, _graphs) = (null, false);
                }

                if (_graph is not null && _graph.Fits(batch, train))
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var watch = Stopwatch.StartNew();
                    model.Network.Train();
                    var (loss, tokens) = _graph.Run(train, batch);
                    trace?.Invoke($"{label(0)}: {batch.Describe(train)}, replayed: loss {loss:F4}, {watch.Elapsed.TotalSeconds:F2} s");
                    Update(model, optimizer, options);
                    return (loss, tokens);
                }
            }

            _ordinary++;
            return RunStep(model, train, group, optimizer, options, custom, cancellationToken, trace, label);
        }

        public void Dispose() => _graph?.Dispose();

        // Whether the optimizer updates the parameters where they are: not the CPU update, which releases the gradients.
        private static bool Recordable(Optimizer optimizer) =>
            optimizer is not HostOptimizer && (optimizer is not GroupedOptimizer grouped || grouped.Groups.All(Recordable));
    }

    // FP8 copies of the frozen weights of the layers with adapters (options.Float8), kept only when a sample's loss with
    // them is within the tolerance of its bfloat16 loss. Disposing the result removes them again.
    private static IDisposable? PrepareFloat8(PretrainedModel model, IReadOnlyList<TrainingSequence> train, FineTuningOptions options, Action<string>? trace)
    {
        if (!options.Float8)
        {
            return null;
        }

        var layers = model.Network.Descendants().OfType<Linear>().Where(l => l.Adapter is not null).ToList();
        var removal = new Float8Removal(layers);
        int count = Math.Min(16, train.Count);
        var sample = Enumerable.Range(0, count).Select(i => train[(int)((long)i * train.Count / count)]).ToList();
        float reference = Evaluate(model, sample, options.BatchTokens, options.LossChunkRows);
        int attached = layers.Count(l => l.AttachFloat8());
        if (attached == 0)
        {
            trace?.Invoke("FP8 products are not available here (CUDA, compute capability 8.9 or newer); training stays in bfloat16");
            return removal;
        }

        float quantized = Evaluate(model, sample, options.BatchTokens, options.LossChunkRows);
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

    // AdamW (or FineTuningOptions.Optimizer) on the device, or on the CPU with its state in system memory
    // (FineTuningOptions.HostOptimizer).
    private static Optimizer CreateOptimizer(List<Tensor> parameters, FineTuningOptions options)
    {
        Func<IReadOnlyList<Tensor>, Optimizer> create = options.Optimizer is { } custom
            ? ps => custom(ps) ?? throw new InvalidOperationException("FineTuningOptions.Optimizer returned no optimizer.")
            : ps => new AdamW(ps, options.LearningRate, weightDecay: options.WeightDecay);
        return options.HostOptimizer ? new HostOptimizer(parameters, create) : create(parameters);
    }

    // Clipping and the optimizer's update, after the gradients of a step.
    private static void Update(PretrainedModel model, Optimizer optimizer, FineTuningOptions options)
    {
        // Clipping and the update of every adapter matrix in a few device passes (which also zero the gradients for the next
        // step), with no read of the norm and no wait: the next step's first read of the device orders everything, so the
        // host prepares the next batch while the device finishes this step.
        optimizer.ClipAndStep(options.MaxGradientNorm);
    }

    // One optimizer step over a group of batches (gradient accumulation): forward, backward, clipping, update. Returns
    // the mean loss per trained token and the tokens covered.
    private static (float Loss, long Tokens) RunStep(PretrainedModel model, IReadOnlyList<TrainingSequence> train, IReadOnlyList<Batch> group, Optimizer optimizer,
        FineTuningOptions options, CustomLoss? custom, CancellationToken cancellationToken, Action<string>? trace, Func<int, string> label)
    {
        var network = model.Network;
        float normalizer = Math.Max(1, group.Sum(b => b.Sequences.Sum(i => train[i].TrainedTokens)));
        int sequences = group.Sum(b => b.Sequences.Count());
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
            var (lossTensor, count) = BatchLoss(model, train, batch, normalizer, options.LossChunkRows, options.Checkpointing == true, custom, sequences);
            if (options.LoadBalancingWeight > 0f && LoadBalancing(network) is { } balance)
            {
                lossTensor += (balance * (options.LoadBalancingWeight / group.Count)).Reshape(lossTensor.Shape);
            }

            lossTensor.Backward();                                           // queued behind the forward pass, no wait between
            float batchLoss = lossTensor.Item();                             // waits for the batch's forward and backward
            loss += batchLoss;
            tokens += count;
            float shown = custom is null ? batchLoss * normalizer / Math.Max(1, batch.Sequences.Sum(i => train[i].TrainedTokens)) : batchLoss;
            trace?.Invoke($"  forward and backward {batchWatch.Elapsed.TotalSeconds:F2} s, loss {shown:F4}");
        }

        Update(model, optimizer, options);
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
        if (!model.Network.Descendants().OfType<Linear>().Any(l => l.Adapter is not null))
        {
            model.AddAdapters(options.Rank, options.Alpha, options.Targets, options.Seed, options.Dora);
        }

        using var float8 = PrepareFloat8(model, train, options, trace);
        using var optimizer = CreateOptimizer(model.Network.TrainableParameters().ToList(), options);
        var batches = MakeBatches(model, train, options, new Random(options.Seed));
        int accumulation = Math.Max(1, options.GradientAccumulation);
        int next = 0;
        using var runner = new StepRunner(model, train, optimizer, options, MostTrained(batches, train), Custom(model, options, pairs: false), trace);
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

    /// <summary>Mean loss per trained token of <paramref name="sequences"/> (no gradients).</summary>
    public static float Evaluate(PretrainedModel model, IReadOnlyList<TrainingSequence> sequences, int batchTokens = 4096, int chunkRows = 512,
        Action<string>? trace = null)
    {
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
                var (loss, _) = BatchLoss(model, sequences, Batch.Padded(batch, sequences), 1f, chunkRows);
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
                var (loss, _) = BatchLoss(model, sequences, batches[b], tokens, options.LossChunkRows, custom: custom, stepSequences: count);
                total += loss.Item();
                trace?.Invoke($"evaluation batch {b + 1}/{batches.Count}: {batches[b].Describe(sequences)}, {watch.Elapsed.TotalSeconds:F2} s");
            }
        }

        return (float)total;
    }

    // Preference pairs (sequences 2k and 2k + 1) in padded batches of whole pairs, pairs of similar length together, at most
    // batchTokens padded positions each (one pair when a pair alone exceeds it); the batch order is shuffled when `random`
    // is given.
    private static List<Batch> PairBatches(IReadOnlyList<TrainingSequence> sequences, int batchTokens, Random? random)
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
    private sealed class CustomLoss(FineTuningLoss loss, bool pairs, DistillationTeacher? teacher = null)
    {
        private readonly Dictionary<TrainingSequence, float[]> _reference = [];

        public Tensor BatchLoss(PretrainedModel model, Tensor tokens, Tensor hidden, Linear head, IReadOnlyList<TrainingSequence> sequences, BatchData data,
            int stepTokens, int stepSequences, int chunkRows)
        {
            int[] targets = [.. data.Targets.Select(t => (int)t)];
            var values = Losses.TokenLogProbabilities(hidden, h => head.Forward(h), data.Trained, targets, chunkRows);
            bool training = Autograd.IsEnabled;
            var probabilities = Tensor.From(values, [values.Length], Device.Cpu, requiresGrad: training);
            TeacherDistributions? distributions = null;
            Func<int, int, Tensor> Teacher(float temperature) => (start, count) =>
            {
                distributions ??= teacher!.Distributions(sequences);
                return distributions.Probabilities(start, count, temperature, head.OutFeatures, hidden.Device);
            };

            try
            {
                Func<float, float[]>? divergence = teacher is null ? null
                    : temperature => Tensor.TokenDivergences(hidden, h => head.Forward(h), data.Trained, Teacher(temperature), temperature, chunkRows);
                var input = new FineTuningLossInput(probabilities, [.. sequences.Select(s => s.TrainedTokens)], sequences,
                    () => Reference(model, tokens, sequences, data, targets, chunkRows), pairs, stepTokens, stepSequences, divergence);
                var value = loss(input) ?? throw new InvalidOperationException("FineTuningOptions.Loss returned no tensor.");
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
                foreach (var (divergences, temperature) in input.Divergences)
                {
                    var g = divergences.Grad?.ToArray() ?? new float[values.Length];
                    var kl = divergences.ToArray();
                    for (int t = 0; t < g.Length; t++)
                    {
                        surrogate += (double)g[t] * kl[t];
                    }

                    var term = Tensor.TokenDivergenceRows(hidden, h => head.Forward(h), data.Trained, g, Teacher(temperature), temperature, chunkRows);
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

        // The trained tokens' log-probabilities under the base model (adapters disabled, no gradients, dropout off), one
        // forward pass of the batch the first time its sequences are seen.
        private float[] Reference(PretrainedModel model, Tensor tokens, IReadOnlyList<TrainingSequence> sequences, BatchData data, int[] targets, int chunkRows)
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
                        checkpointing: false);
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
    private sealed record Batch(int[][] Rows, int Length, bool Packed)
    {
        public IEnumerable<int> Sequences => Rows.SelectMany(r => r);

        public static Batch Padded(int[] sequences, IReadOnlyList<TrainingSequence> all) =>
            new([.. sequences.Select(i => new[] { i })], sequences.Max(i => all[i].Tokens.Length) - 1, false);

        public string Describe(IReadOnlyList<TrainingSequence> all) => Packed
            ? $"{Rows.Sum(r => r.Length)} sequences packed in {Rows.Length} × {Length} positions ({Rows.Sum(r => r.Sum(i => all[i].Tokens.Length - 1)) * 100.0 / (Rows.Length * Length):F0}% filled)"
            : $"{Rows.Length} sequences × {Length} tokens";
    }

    // The training batches: packed when asked and supported, else padded.
    private static List<Batch> MakeBatches(PretrainedModel model, IReadOnlyList<TrainingSequence> train, FineTuningOptions options, Random random)
    {
        if (options.Packing && PackedSequences.Supports(model.Network))
        {
            int longest = train.Max(s => s.Tokens.Length - 1);
            int context = model.Network.Descendants().OfType<CausalSelfAttention>().Min(a => a.MaxPositions);
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
    private sealed record BatchData(float[] Inputs, int[] Trained, float[] Targets, float[] Weights, long Tokens);

    private static BatchData Prepare(IReadOnlyList<TrainingSequence> sequences, Batch batch, float weight = 1f)
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
    private static (Tensor Loss, long Tokens) BatchLoss(PretrainedModel model, IReadOnlyList<TrainingSequence> sequences, Batch batch, float normalizer,
        int chunkRows, bool checkpointing = false, CustomLoss? custom = null, int stepSequences = 0)
    {
        var data = Prepare(sequences, batch);
        var tokens = Tensor.From(data.Inputs, [batch.Rows.Length, batch.Length], model.Device);
        var loss = custom is null
            ? NetworkLoss(model, tokens, (hidden, head) => Losses.TokenCrossEntropyRows(hidden, h => head.Forward(h), data.Trained, data.Targets,
                data.Weights, normalizer, chunkRows), checkpointing)
            : NetworkLoss(model, tokens, (hidden, head) => custom.BatchLoss(model, tokens, hidden, head, [.. batch.Sequences.Select(i => sequences[i])], data,
                (int)normalizer, stepSequences, chunkRows), checkpointing);
        return (loss, data.Tokens);
    }

    // The network up to its final normalization on tokens [rows, length], then `loss` of the hidden states [rows · length,
    // dim] with the output head (which the loss runs itself, on the rows it needs).
    private static T NetworkLoss<T>(PretrainedModel model, Tensor tokens, Func<Tensor, Linear, T> loss, bool checkpointing)
    {
        var modules = model.Network.ToList();
        if (modules[^1] is not Linear head)
        {
            throw new InvalidOperationException("The network does not end with its output head (a Linear layer).");
        }

        var hidden = tokens;
        for (int i = 0; i < modules.Count - 1; i++)
        {
            if (checkpointing && modules[i] is DecoderBlock && Autograd.IsEnabled)
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
    // the packing's layout, the trained positions with targets and weights) live in fixed buffers that each step
    // overwrites before replaying the graph. The loss weights carry 1 / trained tokens, so the recorded pass needs no
    // per-batch constant. Parameters' gradients are the ones the graph zeroes and fills.
    private sealed class TrainingGraph : IDisposable
    {
        private readonly PretrainedModel _model;
        private readonly Tensor _tokens, _rows, _targets, _weights, _loss;
        private readonly PackedSequences _packing;
        private IntPtr _executable, _graph;
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

        private static int[][] Lengths(Batch batch, IReadOnlyList<TrainingSequence> train) =>
            [.. batch.Rows.Select(r => r.Select(i => train[i].Tokens.Length - 1).ToArray())];

        // Records the pass (nothing runs yet); null, with the reason traced, when the device cannot.
        public static TrainingGraph? Record(PretrainedModel model, IReadOnlyList<TrainingSequence> train, Optimizer optimizer, FineTuningOptions options, Batch batch,
            int lossRows, Action<string>? trace)
        {
            var graph = new TrainingGraph(model, batch, train, lossRows);
            var backend = model.Device.Backend;
            model.Device.Synchronize();
            model.Network.Train();
            try
            {
                backend.BeginCapture();
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                graph.Dispose();
                trace?.Invoke($"graph not used ({ex.Message}); ordinary steps continue");
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
                        graph._rows, graph._targets, graph._weights, 1f, options.LossChunkRows), options.Checkpointing == true);
                    loss.Backward();
                    backend.Copy(loss.Storage, graph._loss.Storage, 1);
                }

                (graph._executable, graph._graph, graph._owned) = backend.EndCapture();
                trace?.Invoke($"recorded one training step as a graph ({batch.Rows.Length} × {batch.Length} positions, up to {lossRows} trained); later steps replay it");
                return graph;
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                graph._owned = backend.AbortCapture();
                graph.Dispose();
                trace?.Invoke($"graph not used ({ex.Message}); ordinary steps continue");
                return null;
            }
        }

        // Same shape, and trained positions within the recorded capacity.
        public bool Fits(Batch batch, IReadOnlyList<TrainingSequence> train) =>
            batch.Packed && batch.Rows.Length == _packing.Rows && batch.Length == _packing.Length
            && batch.Sequences.Sum(i => train[i].TrainedTokens) <= LossRows;

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
            _model.Device.Backend.ReplayGraph(_executable);
            return (_loss.Item(), data.Tokens);
        }

        public void Dispose()
        {
            if (_executable != IntPtr.Zero)
            {
                _model.Device.Synchronize();
                _model.Device.Backend.DestroyGraph(_executable, _graph);
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
        }
    }
}
