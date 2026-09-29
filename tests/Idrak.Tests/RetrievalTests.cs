using System.IO.Pipelines;
using System.Text.Json.Nodes;
using ModelContextProtocol.Protocol;
using ModelContextProtocol.Server;
using Idrak;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Mcp;
using Idrak.Optimizers;
using Idrak.Retrieval;

// Retrieval (chunking, BM25, vectors, hybrid fusion, re-ranking, RAG) and MCP tools.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] Retrieval =
    [
        ("retrieval: chunking by words and sentences with overlap", Chunking),
        ("retrieval: BM25 scores match the formula; vector index search and save/load", KeywordAndVectorSearch),
        ("retrieval: text encoder pools real tokens and normalizes; contrastive training retrieves the pairs", TextEncoderPoolingAndTraining),
        ("retrieval: index with keywords, vectors, reciprocal rank fusion; save/load; builder rules", HybridIndex),
        ("retrieval: cross-encoder re-ranking, RAG prompt and citations, search tool", RerankAndRag),
        ("mcp: serve a tool registry over MCP and call it from a client (prefix, rules, errors)", d => { if (d == Device.Cpu) McpRoundTrip(); }),
    ];

    private static readonly Document[] Towns =
    [
        new("armor", "Armor is a town by the river. About 4000 people live in Armor. The winters in Armor are cold."),
        new("belle", "Belle sits on a hill. Belle has 900 residents. Summers in Belle are hot and dry."),
        new("corin", "Corin is a fishing port. The harbour of Corin is busy. Corin is home to 12000 people."),
    ];

    private static void Chunking(Device device)
    {
        var words = Chunker.Split([new Document("d", "a b c d e f g")], ChunkUnit.Words, size: 3, overlap: 1);
        Check(words.Select(c => c.Text).SequenceEqual(["a b c", "c d e", "e f g"]), string.Join(" | ", words.Select(c => c.Text)));
        Check(words.Select(c => (c.Id, c.Position)).SequenceEqual([(0, 0), (1, 1), (2, 2)]), "ids and positions");
        var sentences = Chunker.Split(Towns, ChunkUnit.Sentences, size: 2, overlap: 0);
        Check(sentences.Count == 6 && sentences[1].Text == "The winters in Armor are cold." && sentences[2].DocumentId == "belle" && sentences[2].Position == 0,
            string.Join(" | ", sentences.Select(c => $"{c.DocumentId}:{c.Text}")));
        Check(sentences.Select(c => c.Id).SequenceEqual(Enumerable.Range(0, 6)), "ids across documents");
        Check(Chunker.Split([new Document("e", "   ")], ChunkUnit.Words, 3, 0).Count == 0, "empty document");
        Throws<ArgumentOutOfRangeException>(() => Chunker.Split(Towns, ChunkUnit.Words, 3, 3), "overlap must be smaller than size");
    }

    private static void KeywordAndVectorSearch(Device device)
    {
        string[] texts = ["the cat sat", "the dog sat on the cat", "birds fly"];
        var bm25 = new Bm25Index(texts, k1: 1.5, b: 0.75);
        var hits = bm25.Search("cat", 10);
        Check(hits.Select(h => h.Id).SequenceEqual([0, 1]), "documents without the word are left out");
        double idf = Math.Log(1 + (3 - 2 + 0.5) / (2 + 0.5)), average = (3 + 6 + 2) / 3.0;
        double expected0 = idf * 1 * 2.5 / (1 + 1.5 * (1 - 0.75 + 0.75 * 3 / average));
        Check(Math.Abs(hits[0].Score - expected0) < 1e-12, $"bm25 {hits[0].Score} vs {expected0}");

        var cosine = new VectorIndex(3, VectorMetric.Cosine);
        cosine.Add([1, 0, 0]);
        cosine.Add([0, 2, 0]);
        cosine.Add([1, 1, 0]);
        var near = cosine.Search([0, 5, 0.1f], 2);
        Check(near[0].Id == 1 && near[1].Id == 2, "cosine order");
        Check(Math.Abs(cosine[1][1] - 1) < 1e-6, "cosine vectors are stored at unit length");
        var dot = new VectorIndex(2, VectorMetric.Dot);
        dot.AddRange([[1, 0], [3, 0]]);
        Check(dot.Search([1, 0], 1)[0] is { Id: 1, Score: 3 }, "dot keeps magnitudes");
        using var stream = new MemoryStream();
        cosine.Save(stream);
        stream.Position = 0;
        var loaded = VectorIndex.Load(stream);
        Check(loaded.Count == 3 && loaded.Metric == VectorMetric.Cosine && loaded.Search([0, 5, 0.1f], 3).SequenceEqual(cosine.Search([0, 5, 0.1f], 3)), "save/load");
        Throws<ArgumentException>(() => cosine.Add([1, 2]), "wrong dimensions");
    }

    private static (WordTokenizer Words, Module Model) SmallEncoder(Device device, IEnumerable<string> texts, int dim, int seed)
    {
        var words = new WordTokenizer(["<pad>", "<unk>", .. texts.SelectMany(t => WordTokenizer.Split(t)).Distinct().Order()]);
        return (words, new Embedding(words.VocabularySize, dim, device, new Random(seed)));
    }

    private static void TextEncoderPoolingAndTraining(Device device)
    {
        var (words, model) = SmallEncoder(device, ["red green blue cyan"], 4, 1);
        using var _ = model;
        var encoder = new TextEncoder(model, words, maxLength: 6, padId: words["<pad>"]);
        var embedding = ((Embedding)model).Weight.ToArray();
        float[] Row(string w) => embedding.AsSpan(words[w] * 4, 4).ToArray();
        var expected = Row("red").Zip(Row("blue"), (a, b) => (a + b) / 2).ToArray();
        float norm = MathF.Sqrt(expected.Sum(v => v * v));
        var single = encoder.Encode("red blue");
        Check(single.Zip(expected).All(p => MathF.Abs(p.First - p.Second / norm) < 1e-4f), "mean of the real tokens, scaled to length 1");
        var batch = encoder.Encode(["red blue", "green cyan red blue green"]);
        Check(batch[0].Zip(single).All(p => MathF.Abs(p.First - p.Second) < 1e-5f), "padding does not change the vector");
        Check(Math.Abs(batch[1].Sum(v => v * v) - 1) < 1e-4, "unit length");

        // Pairs share one topic word; after training, each query's own passage is its nearest.
        string[] topics = ["apple", "river", "engine", "violin", "desert", "glacier", "harbor", "tulip"];
        var pairs = topics.Select(t => ($"about {t} please", $"notes on the {t} here")).ToList();
        var (vocab, net) = SmallEncoder(device, pairs.SelectMany(p => new[] { p.Item1, p.Item2 }), 16, 2);
        using var __ = net;
        var trainable = new TextEncoder(net, vocab, maxLength: 8, padId: vocab["<pad>"]);
        var losses = trainable.Train(pairs, epochs: 60, batchSize: 8, p => new Adam(p, learningRate: 0.05f), temperature: 0.1f, seed: 3);
        Check(losses[^1] < losses[0] / 4, $"loss {losses[0]:F3} -> {losses[^1]:F3}");
        var index = new VectorIndex(16, VectorMetric.Dot);
        index.AddRange(trainable.Encode([.. pairs.Select(p => p.Item2)]));
        int correct = pairs.Select((p, i) => index.Search(trainable.Encode(p.Item1), 1)[0].Id == i ? 1 : 0).Sum();
        Check(correct == pairs.Count, $"{correct}/{pairs.Count} queries find their passage");
    }

    private static void HybridIndex(Device device)
    {
        var keyword = RetrievalIndex.Create().Documents(Towns, ChunkUnit.Sentences, 1, 0).Bm25().Build();
        var reference = new Bm25Index(keyword.Chunks.Select(c => c.Text));
        var found = keyword.Search("how many people live in corin", 3);
        var expected = reference.Search("how many people live in corin", 3);
        Check(found.Select(r => (r.Chunk.Id, r.Score)).SequenceEqual(expected.Select(h => (h.Id, h.Score))), "keyword search is BM25");
        Check(found.Select(r => r.KeywordRank).SequenceEqual([1, 2, 3]) && found.All(r => r.VectorRank is null), "ranks");

        var (words, model) = SmallEncoder(device, Towns.Select(t => t.Text), 8, 4);
        using var _ = model;
        var encoder = new TextEncoder(model, words, 16, words["<pad>"]);
        var hybrid = RetrievalIndex.Create().Documents(Towns, ChunkUnit.Sentences, 1, 0).Bm25().Embeddings(encoder, batchSize: 4).Fusion(k: 60, depth: 5).Build();
        const string query = "winters in armor";
        var k = hybrid.Keywords!.Search(query, 5);
        var v = hybrid.Vectors!.Search(encoder.Encode(query), 5);
        var rrf = new Dictionary<int, double>();
        foreach (var list in new[] { k, v })
        {
            for (int i = 0; i < list.Count; i++)
            {
                rrf[list[i].Id] = rrf.GetValueOrDefault(list[i].Id) + 1.0 / (60 + i + 1);
            }
        }

        var fused = hybrid.Search(query, 4);
        var manual = rrf.OrderByDescending(p => p.Value).ThenBy(p => p.Key).Take(4).ToList();
        Check(fused.Select(r => r.Chunk.Id).SequenceEqual(manual.Select(p => p.Key)), "reciprocal rank fusion order");
        Check(fused.Zip(manual).All(p => Math.Abs(p.First.Score - p.Second.Value) < 1e-12), "fused scores");
        Check(fused.All(r => r.KeywordRank is not null || r.VectorRank is not null), "each result carries its source ranks");

        using var stream = new MemoryStream();
        hybrid.Save(stream);
        stream.Position = 0;
        var loaded = RetrievalIndex.Load(stream, encoder);
        Check(loaded.Search(query, 4).SequenceEqual(fused), "save/load gives the same results");
        stream.Position = 0;
        Throws<ArgumentNullException>(() => RetrievalIndex.Load(stream, null), "vectors need the encoder");

        Throws<InvalidOperationException>(() => RetrievalIndex.Create().Documents(Towns, ChunkUnit.Words, 5, 0).Build(), "no search chosen");
        Throws<InvalidOperationException>(() => RetrievalIndex.Create().Bm25().Embeddings(encoder).Build(), "both need fusion");
        Throws<InvalidOperationException>(() => RetrievalIndex.Create().Bm25().Fusion(60, 5).Build(), "fusion needs both");
        Check(RetrievalIndex.Create().Bm25().Build().Search("anything", 3).Count == 0, "empty index");
    }

    private static void RerankAndRag(Device device)
    {
        var index = RetrievalIndex.Create().Documents(Towns, ChunkUnit.Sentences, 1, 0).Bm25().Build();
        // A "model" that scores a pair by how many query words the passage contains (the pair is encoded as that count).
        using var identity = new Lambda(x => x.Reshape(-1), "Score");
        var reranker = new CrossEncoder(identity, (q, p) =>
            [WordTokenizer.Split(q).Intersect(WordTokenizer.Split(p)).Count()], [1]);
        var scores = reranker.Score("people in corin", ["Corin is home to 12000 people.", "Belle sits on a hill."]);
        Check(scores.SequenceEqual([2f, 0f]), string.Join(",", scores));
        var candidates = index.Search("corin people", 4);
        var reranked = reranker.Rerank("corin people", candidates, 2);
        Check(reranked.Count == 2 && reranked[0].Chunk.Text == "Corin is home to 12000 people." && reranked[0].Score == 2, reranked[0].Chunk.Text);

        var fake = FakeChatModel.Script(FakeChatModel.Answer("Corin has 12000 people [1]. See also [9] and [1]."));
        var rag = Rag.For(fake).Retrieve(index, 4).Rerank(reranker, 2).Label(c => $"{c.DocumentId}#{c.Position}").System("Be brief.").Build();
        var answer = rag.AskAsync("corin people").GetAwaiter().GetResult();
        Check(answer.Passages.Count == 2 && answer.Passages[0].Label == "corin#2", string.Join(",", answer.Passages.Select(p => p.Label)));
        Check(answer.Cited.Count == 1 && answer.Cited[0].Number == 1, "only [n] markers that match a passage are cited, once each");
        var sent = fake.Requests.Single().Messages;
        Check(sent[0].Role == "system" && sent[1].Content.Contains("[1] (corin#2) Corin is home to 12000 people.") && sent[1].Content.EndsWith("Question: corin people"),
            sent[1].Content);
        Throws<InvalidOperationException>(() => Rag.For(fake).Build(), "Retrieve is required");
        Throws<InvalidOperationException>(() => Rag.For(fake).Retrieve(index, 2).Rerank(reranker, 3).Build(), "keep ≤ top");

        var tools = ToolRegistry.Create().Add(RetrievalTools.Search(index, 2, "search_towns", "Searches facts about towns.")).Build();
        var result = tools.InvokeAsync(new ToolCall("search_towns", new JsonObject { ["query"] = "harbour of corin" })).GetAwaiter().GetResult();
        Check(result.Succeeded && result.Content.StartsWith("[1] (corin) The harbour of Corin is busy."), result.Content);
        var none = tools.InvokeAsync(new ToolCall("search_towns", new JsonObject { ["query"] = "zebra" })).GetAwaiter().GetResult();
        Check(none.Content == "No results.", none.Content);
    }

    private static void McpRoundTrip() => McpRoundTripAsync().GetAwaiter().GetResult();

    private static async Task McpRoundTripAsync()
    {
        var served = ToolRegistry.Create()
            .Add("add", "Adds two integers.", new JsonObject
            {
                ["type"] = "object",
                ["properties"] = new JsonObject { ["a"] = new JsonObject { ["type"] = "integer" }, ["b"] = new JsonObject { ["type"] = "integer" } },
                ["required"] = new JsonArray("a", "b"),
            }, (args, _) => Task.FromResult(((int)args["a"]! + (int)args["b"]!).ToString()))
            .Add("fail", "Always fails.", new JsonObject { ["type"] = "object", ["properties"] = new JsonObject() },
                (_, _) => throw new InvalidOperationException("boom"))
            .Allow("add", args => (int)args["a"]! >= 0)
            .Build();

        var toServer = new Pipe();
        var toClient = new Pipe();
        var options = new McpServerOptions { ServerInfo = new Implementation { Name = "test-server", Version = "1.0" }, ToolCollection = [] };
        foreach (var tool in McpTools.ServerTools(served))
        {
            options.ToolCollection.Add(tool);
        }

        await using var server = McpServer.Create(new StreamServerTransport(toServer.Reader.AsStream(), toClient.Writer.AsStream(), "test-server"), options);
        using var stop = new CancellationTokenSource();
        var running = server.RunAsync(stop.Token);
        await using (var source = await McpTools.ConnectAsync(new StreamClientTransport(toServer.Writer.AsStream(), toClient.Reader.AsStream())))
        {
            Check(source.ServerName == "test-server", $"server name {source.ServerName}");
            var tools = await source.ListToolsAsync(prefix: "calc_");
            Check(tools.Select(t => t.Definition.Name).Order().SequenceEqual(["calc_add", "calc_fail"]), string.Join(",", tools.Select(t => t.Definition.Name)));
            Check(tools.First(t => t.Definition.Name == "calc_add").Definition.Parameters!["required"]!.AsArray().Count == 2, "schema carried over");

            var registry = ToolRegistry.Create().Add(tools).Build();
            var sum = await registry.InvokeAsync(new ToolCall("calc_add", new JsonObject { ["a"] = 2, ["b"] = 40 }));
            Check(sum.Succeeded && sum.Content == "42", sum.Content);
            var invalid = await registry.InvokeAsync(new ToolCall("calc_add", new JsonObject { ["a"] = 2 }));
            Check(!invalid.Succeeded && invalid.Error!.Contains("'b' is required"), invalid.Content);
            var denied = await registry.InvokeAsync(new ToolCall("calc_add", new JsonObject { ["a"] = -1, ["b"] = 1 }));
            Check(!denied.Succeeded && denied.Error!.Contains("not allowed"), $"the server's rules apply: {denied.Content}");
            var failed = await registry.InvokeAsync(new ToolCall("calc_fail", []));
            Check(!failed.Succeeded && failed.Error!.Contains("boom"), failed.Content);
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

    private static void Throws<T>(Action action, string message) where T : Exception
    {
        try
        {
            action();
        }
        catch (T)
        {
            return;
        }

        throw new InvalidOperationException($"expected {typeof(T).Name}: {message}");
    }
}
