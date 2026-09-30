using System.Runtime.CompilerServices;
using System.Text;
using System.Text.RegularExpressions;
using Idrak.Generation;

namespace Idrak.Retrieval;

/// <summary>A passage shown to the model, with the number it is cited by.</summary>
/// <param name="Number">The citation number ([1], [2], …).</param>
/// <param name="Label">Where it came from, as shown to the model and the reader.</param>
/// <param name="Chunk">The passage.</param>
/// <param name="Score">Its retrieval (or re-ranking) score.</param>
public sealed record Citation(int Number, string Label, Chunk Chunk, double Score);

/// <summary>The outcome of <see cref="RagPipeline.AskAsync"/>.</summary>
/// <param name="Text">The answer.</param>
/// <param name="Cited">The passages the answer cites ([n] markers that match a passage), in order of first citation.</param>
/// <param name="Passages">Every passage the model was shown.</param>
/// <param name="Message">The model's full message (reasoning, tool calls).</param>
/// <param name="DoneReason">"stop" or "length".</param>
public sealed record RagAnswer(string Text, IReadOnlyList<Citation> Cited, IReadOnlyList<Citation> Passages, ChatMessage Message, string? DoneReason);

/// <summary>Retrieval-augmented generation: find passages for a question, show them to a chat model, and answer with citations.</summary>
public static class Rag
{
    /// <summary>Starts a pipeline that answers with <paramref name="model"/> (a <see cref="ChatGenerator"/>, an engine model, or a fake for tests).</summary>
    public static RagBuilder For(IChatModel model) => new(model);

    /// <summary>
    /// The prompt used unless <see cref="RagBuilder.Prompt"/> sets another: an instruction to answer from the passages
    /// and cite them as [n], the numbered passages with their labels, and the question.
    /// </summary>
    public static string DefaultPrompt(string question, IReadOnlyList<Citation> passages)
    {
        var text = new StringBuilder("Answer the question using only the passages below. Cite the passages you use as [1], [2], … ")
            .Append("If the passages do not contain the answer, say that you do not know.\n\n");
        foreach (var p in passages)
        {
            text.Append('[').Append(p.Number).Append("] (").Append(p.Label).Append(") ").Append(p.Chunk.Text).Append('\n');
        }

        return text.Append("\nQuestion: ").Append(question).ToString();
    }
}

/// <summary>
/// Configures a <see cref="RagPipeline"/>: <see cref="Retrieve"/> is required; <see cref="Rerank"/>, <see cref="Label"/>,
/// <see cref="Prompt"/>, <see cref="System"/>, <see cref="Think"/> and <see cref="Options"/> are optional.
/// </summary>
public sealed class RagBuilder
{
    private readonly IChatModel _model;
    private IRetriever? _retriever;
    private int _top;
    private IReranker? _reranker;
    private int _keep;
    private Func<Chunk, string> _label = c => c.DocumentId;
    private Func<string, IReadOnlyList<Citation>, string> _prompt = Rag.DefaultPrompt;
    private string? _system;
    private bool? _think;
    private GenerationOptions? _options;

    internal RagBuilder(IChatModel model) => _model = model;

    /// <summary>
    /// Asks <paramref name="retriever"/> (a <see cref="RetrievalIndex"/> or any <see cref="IRetriever"/>) for the
    /// <paramref name="top"/> best chunks (the candidates, when re-ranking).
    /// </summary>
    public RagBuilder Retrieve(IRetriever retriever, int top)
    {
        ArgumentNullException.ThrowIfNull(retriever);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(top);
        _retriever = retriever;
        _top = top;
        return this;
    }

    /// <summary>
    /// Re-orders the retrieved chunks with <paramref name="reranker"/> (a <see cref="CrossEncoder"/> or any
    /// <see cref="IReranker"/>) and shows the model the best <paramref name="keep"/>.
    /// </summary>
    public RagBuilder Rerank(IReranker reranker, int keep)
    {
        ArgumentNullException.ThrowIfNull(reranker);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(keep);
        _reranker = reranker;
        _keep = keep;
        return this;
    }

    /// <summary>How a passage's source is named in the prompt and in <see cref="Citation.Label"/> (the chunk's document id unless set).</summary>
    public RagBuilder Label(Func<Chunk, string> label)
    {
        _label = label;
        return this;
    }

    /// <summary>Builds the user message from the question and the numbered passages (<see cref="Rag.DefaultPrompt"/> unless set).</summary>
    public RagBuilder Prompt(Func<string, IReadOnlyList<Citation>, string> prompt)
    {
        _prompt = prompt;
        return this;
    }

    /// <summary>A system message sent before the prompt.</summary>
    public RagBuilder System(string text)
    {
        _system = text;
        return this;
    }

    /// <summary>The reasoning mode passed to the model.</summary>
    public RagBuilder Think(bool? think)
    {
        _think = think;
        return this;
    }

    /// <summary>Generation options passed to the model.</summary>
    public RagBuilder Options(GenerationOptions options)
    {
        _options = options;
        return this;
    }

    /// <summary>Creates the pipeline.</summary>
    public RagPipeline Build()
    {
        if (_retriever is null)
        {
            throw new InvalidOperationException("Call Retrieve(index, top) to choose where passages come from.");
        }

        if (_reranker is not null && _keep > _top)
        {
            throw new InvalidOperationException($"Rerank keeps {_keep} passages but Retrieve finds only {_top} candidates.");
        }

        return new RagPipeline(_model, _retriever, _top, _reranker, _keep, _label, _prompt, _system, _think, _options);
    }
}

/// <summary>Answers questions from a retriever's passages with citations; create one with <see cref="Rag.For"/>.</summary>
public sealed partial class RagPipeline
{
    private readonly IChatModel _model;
    private readonly int _top;
    private readonly IReranker? _reranker;
    private readonly int _keep;
    private readonly Func<Chunk, string> _label;
    private readonly Func<string, IReadOnlyList<Citation>, string> _prompt;
    private readonly string? _system;
    private readonly bool? _think;
    private readonly GenerationOptions? _options;

    internal RagPipeline(IChatModel model, IRetriever retriever, int top, IReranker? reranker, int keep, Func<Chunk, string> label,
        Func<string, IReadOnlyList<Citation>, string> prompt, string? system, bool? think, GenerationOptions? options)
    {
        _model = model;
        Retriever = retriever;
        _top = top;
        _reranker = reranker;
        _keep = keep;
        _label = label;
        _prompt = prompt;
        _system = system;
        _think = think;
        _options = options;
    }

    /// <summary>Where passages come from.</summary>
    public IRetriever Retriever { get; }

    /// <summary>The index passages come from, when the retriever is a <see cref="RetrievalIndex"/> (null for another retriever).</summary>
    public RetrievalIndex? Index => Retriever as RetrievalIndex;

    /// <summary>The outcome of the most recent <see cref="StreamAsync"/>, once its stream has ended.</summary>
    public RagAnswer? LastAnswer { get; private set; }

    /// <summary>The numbered passages the model would be shown for <paramref name="question"/> (waits for <see cref="RetrieveAsync"/>).</summary>
    public IReadOnlyList<Citation> Retrieve(string question) => RetrieveAsync(question).AsTask().GetAwaiter().GetResult();

    /// <summary>The numbered passages the model would be shown for <paramref name="question"/>.</summary>
    public async ValueTask<IReadOnlyList<Citation>> RetrieveAsync(string question, CancellationToken cancellationToken = default)
    {
        var found = await Retriever.RetrieveAsync(question, _top, cancellationToken).ConfigureAwait(false);
        if (_reranker is not null)
        {
            found = await _reranker.RerankAsync(question, found, _keep, cancellationToken).ConfigureAwait(false);
        }

        return [.. found.Select((r, i) => new Citation(i + 1, _label(r.Chunk), r.Chunk, r.Score))];
    }

    /// <summary>The chat messages sent to the model for <paramref name="question"/> and <paramref name="passages"/>.</summary>
    public IReadOnlyList<ChatMessage> Messages(string question, IReadOnlyList<Citation> passages) =>
        _system is null
            ? [new ChatMessage("user", _prompt(question, passages))]
            : [new ChatMessage("system", _system), new ChatMessage("user", _prompt(question, passages))];

    /// <summary>Retrieves passages, asks the model and returns the answer with the passages it cites.</summary>
    public async Task<RagAnswer> AskAsync(string question, CancellationToken cancellationToken = default)
    {
        RagAnswer? answer = null;
        await foreach (var _ in RunAsync(question, a => answer = a, cancellationToken).ConfigureAwait(false))
        {
        }

        return answer!;
    }

    /// <summary>Retrieves passages and streams the answer; when the stream ends, <see cref="LastAnswer"/> holds the outcome.</summary>
    public IAsyncEnumerable<ChatDelta> StreamAsync(string question, CancellationToken cancellationToken = default) =>
        RunAsync(question, a => LastAnswer = a, cancellationToken);

    /// <summary>The passages of <paramref name="passages"/> that <paramref name="text"/> cites as [n] (spaces inside the brackets allowed, as word-level tokenizers write "[ 2 ]"), in order of first citation.</summary>
    public static IReadOnlyList<Citation> CitedIn(string text, IReadOnlyList<Citation> passages)
    {
        var byNumber = passages.ToDictionary(p => p.Number);
        return [.. CitationMarker().Matches(text).Select(m => int.Parse(m.Groups[1].ValueSpan, provider: null)).Distinct()
            .Where(byNumber.ContainsKey).Select(n => byNumber[n])];
    }

    private async IAsyncEnumerable<ChatDelta> RunAsync(string question, Action<RagAnswer> done, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var passages = await RetrieveAsync(question, cancellationToken).ConfigureAwait(false);
        var request = new ChatRequest(Messages(question, passages), null, _think, _options);
        ChatChunk? final = null;
        await foreach (var chunk in _model.StreamAsync(request, cancellationToken).ConfigureAwait(false))
        {
            if (!chunk.Delta.IsEmpty)
            {
                yield return chunk.Delta;
            }

            if (chunk.Done)
            {
                final = chunk;
            }
        }

        if (final?.Message is not { } message)
        {
            throw new InvalidOperationException("The model ended without a final message.");
        }

        done(new RagAnswer(message.Content, CitedIn(message.Content, passages), passages, message, final.DoneReason));
    }

    [GeneratedRegex(@"\[\s*(\d{1,4})\s*\]")]
    private static partial Regex CitationMarker();
}

/// <summary>Tools that let a chat model search an index itself (for agents that decide when and what to look up).</summary>
public static class RetrievalTools
{
    /// <summary>
    /// A tool with one string argument, "query", that returns the <paramref name="top"/> best chunks of
    /// <paramref name="index"/>, one per line as "[n] (label) text", or "No results." when nothing matches.
    /// </summary>
    /// <param name="index">What to search: a <see cref="RetrievalIndex"/> or any <see cref="IRetriever"/>.</param>
    /// <param name="top">Chunks per search.</param>
    /// <param name="name">The tool's name, for example "search_documents".</param>
    /// <param name="description">What the model is told the tool searches.</param>
    /// <param name="label">Names a chunk's source (its document id when null).</param>
    public static Tool Search(IRetriever index, int top, string name, string description, Func<Chunk, string>? label = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(top);
        label ??= c => c.DocumentId;
        var schema = new System.Text.Json.Nodes.JsonObject
        {
            ["type"] = "object",
            ["properties"] = new System.Text.Json.Nodes.JsonObject
            {
                ["query"] = new System.Text.Json.Nodes.JsonObject { ["type"] = "string", ["description"] = "What to look for." },
            },
            ["required"] = new System.Text.Json.Nodes.JsonArray("query"),
        };
        return Tool.Create(name, description, schema, async (arguments, cancellationToken) =>
        {
            var hits = await index.RetrieveAsync((string?)arguments["query"] ?? "", top, cancellationToken).ConfigureAwait(false);
            return hits.Count == 0
                ? "No results."
                : string.Join('\n', hits.Select((h, i) => $"[{i + 1}] ({label(h.Chunk)}) {h.Chunk.Text}"));
        });
    }
}
