// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Data;
using Idrak.Diagnostics;
using Idrak.Generation;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Training.Abstractions;
using Idrak.Training;
using Idrak;

// Each simplified API must behave exactly like the code it replaces: same layers, same weights, same training results.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] Simplified =
    [
        ("builder: same layers, shapes and weights as new Sequential { ... }", BuilderMatchesSequential),
        ("sequential: inference frees each layer's intermediates; ForwardFirst runs the first layers", SequentialInferenceFreesLayers),
        ("builder: CNN, RNN and GPT shapes; JSON round trip; clear shape errors", BuilderShapesAndJson),
        ("data: Split/Standardize/Batches equal the manual steps", DataExtensionsMatchManual),
        ("data: Deduplicate keeps the first of exact duplicates in order; targets, conflicts, near duplicates, report", Deduplication),
        ("training: factory Trainer and TrainingRun give the same history as the manual trainer", TrainingRunMatchesTrainer),
        ("training: FitAsync progress and TrainAsync early exit", TrainingAsync),
        ("losses: CrossEntropy and SparseCrossEntropy pass as a Loss (method group) and equal the smoothing overload at 0", LossMethodGroups),
        ("fine-tuning: freeze, trainable-only save/load, grouped optimizer", FreezingAndGroups),
        ("fine-tuning: LoRA starts neutral, trains only adapters, merges exactly", Lora),
        ("telemetry: Configure().Console/JsonLines/Record.Start() subscribes, disposes and flushes", TelemetryBuilderSession),
        ("predictor: equals the manual scale/predict/unscale steps; typed input and output", PredictorMatchesManual),
        ("predictor: softmax and class names; save and load round trip", PredictorClassesAndPackage),
        ("package: weights, scalers, tokenizers, JSON, text, trainable weights, text generator", PackageRoundTrip),
        ("generation: StreamAsync gives the same chunks as Stream", StreamAsyncMatches),
        ("tools: schema from delegates and [Tool] methods; validation, allowlist, approval, timeout, parallel", ToolRules),
        ("conversation: history, tool loop with a scripted model, MaxToolRounds required and enforced", ConversationLoop),
        ("engine: predictors from a factory and a package; batches, PredictManyAsync, micro-batching, warm-up, stats", EnginePredictors),
        ("engine: instances, queue limit, timeout, keep-alive with a manual clock, load on first use, telemetry", EngineLifecycle),
        ("engine: text generation and chat conversations through the engine", EngineGeneration),
        ("engine: a model kind of one's own (EngineModel): lent copies, keep-alive, micro-batching, status, description, contracts, dispose", EngineOwnKind),
    ];

    private static void Deduplication(Device device)
    {
        // rows: 0 [1,2]->a, 1 [3,4]->b, 2 [1,2]->a (exact), 3 [1,2]->b (conflict), 4 [5,6]->a, 5 [3,4]->b (exact), 6 [1,2]->b
        float[] x = [1, 2, 3, 4, 1, 2, 1, 2, 5, 6, 3, 4, 1, 2];
        float[] y = [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1];                          // one-hot: [1,0] = a, [0,1] = b
        var data = Dataset.FromFlat(x, y, 7, ["x0", "x1"], ["a", "b"]);
        int[] Rows(Dataset d) => [.. Enumerable.Range(0, d.Count).Select(i => (int)d.GetFeatures(i)[0] * 10 + (d.GetTargets(i)[0] == 1 ? 0 : 1))];

        var first = data.DeduplicateWithReport();
        Check(first.Kept.SequenceEqual([0, 1, 4]) && first.ExactDuplicates == 4 && first.Removed == 4, $"features: first of each ({string.Join(",", first.Kept)})");
        Check(Rows(first.Data).SequenceEqual([10, 31, 50]), "features: rows and targets of the first rows, in order");
        Check(data.Count == 7, "the original dataset is not changed");

        var withTargets = data.DeduplicateWithReport(new DeduplicationOptions { Match = DuplicateMatch.FeaturesAndTargets });
        Check(withTargets.Kept.SequenceEqual([0, 1, 3, 4]) && withTargets.ExactDuplicates == 3, "features and targets: [1,2]->b is its own row");

        var vote = data.DeduplicateWithReport(new DeduplicationOptions { Conflicts = DuplicateConflicts.KeepMostFrequent });
        Check(vote.Kept.SequenceEqual([0, 1, 4]), $"most frequent: [1,2] is a twice and b twice, the tie goes to the earlier ({string.Join(",", vote.Kept)})");
        var majority = Dataset.FromFlat([7, 7, 7], [1, 0, 0, 1, 0, 1], 3, ["x"], ["a", "b"]).DeduplicateWithReport(new DeduplicationOptions { Conflicts = DuplicateConflicts.KeepMostFrequent });
        Check(majority.Kept.SequenceEqual([1]) && majority.Data.GetTargets(0)[1] == 1, "most frequent: [7] is a once and b twice, so the first b row is kept");
        var drop = data.DeduplicateWithReport(new DeduplicationOptions { Conflicts = DuplicateConflicts.DropAll });
        Check(drop.Kept.SequenceEqual([1, 4]) && drop.Conflicting == 4 && drop.ExactDuplicates == 1, "drop all: every [1,2] row goes");

        // Near duplicates from an embedding the caller computes: row 4 points the same way as row 0 ([5,6] ~ [1,2] here).
        float[][] embeddings = [[1, 0], [0, 1], [1, 0], [1, 0], [0.99f, 0.05f], [0, 1], [1, 0]];
        var near = data.DeduplicateWithReport(new DeduplicationOptions { Embedding = i => embeddings[i], SimilarityThreshold = 0.95f });
        Check(near.Kept.SequenceEqual([0, 1]) && near.NearDuplicates == 1, "near duplicates: dropped after the row they resemble");
        var strict = data.Deduplicate(new DeduplicationOptions { Embedding = i => embeddings[i], SimilarityThreshold = 0.9999f });
        Check(strict.Count == 3, "near duplicates: a higher threshold keeps them");

        // Blocks and tiles: 3,000 rows (several blocks of 512 and tiles of 1,024), a third of them near copies of earlier
        // rows, give the same rows as the plain greedy pass (each row against every row kept before it).
        var random = new Random(11);
        int many = 3000, dim = 24;
        var vectors = new float[many][];
        for (int i = 0; i < many; i++)
        {
            vectors[i] = i % 3 == 2
                ? [.. vectors[random.Next(i)].Select(v => v + ((float)random.NextDouble() - 0.5f) * 0.02f)]
                : [.. Enumerable.Range(0, dim).Select(_ => (float)random.NextDouble() - 0.5f)];
        }

        var rows = Dataset.FromFlat([.. Enumerable.Range(0, many).Select(i => (float)i)], [.. Enumerable.Range(0, many).Select(i => (float)(i % 2))], many, ["x"], ["y"]);
        foreach (bool matchTargets in new[] { false, true })
        {
            var options = new DeduplicationOptions
            {
                Embedding = i => vectors[i], SimilarityThreshold = 0.97f,
                Match = matchTargets ? DuplicateMatch.FeaturesAndTargets : DuplicateMatch.Features,
            };
            var expected = new List<int>();
            foreach (int i in Enumerable.Range(0, many))
            {
                if (!expected.Any(k => Cosine(vectors[i], vectors[k]) >= 0.97f && (!matchTargets || i % 2 == k % 2)))
                {
                    expected.Add(i);
                }
            }

            var result = rows.DeduplicateWithReport(options);
            Check(result.Kept.SequenceEqual(expected) && result.NearDuplicates == many - expected.Count,
                $"near duplicates in blocks = the plain greedy pass (targets {matchTargets}: kept {result.Kept.Length}, expected {expected.Count})");
        }

        var unique = Dataset.FromFlat([1, 2, 3], [0, 1, 0], 3, ["x"], ["y"]);
        Check(unique.Deduplicate().Count == 3, "no duplicates: every row kept");
        Check(Throws(() => data.Deduplicate(new DeduplicationOptions { Embedding = i => embeddings[i], SimilarityThreshold = 1.5f })), "threshold checked");
        var (train, test) = data.Deduplicate().Split(0.67, seed: 1);
        Check(train.Count + test.Count == 3, "Deduplicate().Split(...) splits the clean rows");
        Check(ReferenceEquals(unique.Deduplicate(), unique), "no duplicates: the same dataset, not a copy");

        // Split deduplicates by default (the default rules, or the ones given); removeDuplicates: false keeps every row.
        var (defaultTrain, defaultTest) = data.Split(0.5, seed: 1);
        Check(defaultTrain.Count + defaultTest.Count == 3, $"Split: 7 rows, deduplicated to 3 ({defaultTrain.Count + defaultTest.Count})");
        var seen = new HashSet<string>();
        foreach (var part in new[] { defaultTrain, defaultTest })
        {
            for (int i = 0; i < part.Count; i++)
            {
                Check(seen.Add(string.Join(",", part.GetFeatures(i).ToArray())), "Split: no features in both parts");
            }
        }

        var (ruleTrain, ruleTest) = data.Split(0.5, seed: 1, duplicates: new DeduplicationOptions { Match = DuplicateMatch.FeaturesAndTargets });
        Check(ruleTrain.Count + ruleTest.Count == 4, "Split with rules: features and targets");
        var (keepTrain, keepTest) = data.Split(0.5, seed: 1, removeDuplicates: false);
        Check(keepTrain.Count + keepTest.Count == 7, "Split(removeDuplicates: false): every row kept, identical copies included");
        var (sameTrain, _) = unique.Split(0.67, seed: 4);
        var (plainTrain, _) = unique.Split(0.67, seed: 4, removeDuplicates: false);
        Check(sameTrain.Features.SequenceEqual(plainTrain.Features), "no duplicates: the same split either way");
    }

    private static float Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            (dot, na, nb) = (dot + a[i] * b[i], na + a[i] * a[i], nb + b[i] * b[i]);
        }

        return (float)(dot / Math.Sqrt(na * nb));
    }

    private static bool Throws(Action action)
    {
        try
        {
            action();
            return false;
        }
        catch (ArgumentException)
        {
            return true;
        }
    }

    private static void LossMethodGroups(Device device)
    {
        Func<Tensor, Tensor, Tensor> dense = Losses.CrossEntropy, sparse = Losses.SparseCrossEntropy;
        using var scope = new TensorScope();
        var logits = Tensor.From([0.2f, -1.1f, 0.7f, 1.5f, 0.3f, -0.4f], [2, 3], device);
        var oneHot = Tensor.From([0f, 0f, 1f, 1f, 0f, 0f], [2, 3], device);
        var labels = Tensor.From([2f, 0f], [2], device);
        AssertClose([dense(logits, oneHot).Item()], [Losses.CrossEntropy(logits, oneHot, 0f).Item()], 0f, "CrossEntropy(a, b) == CrossEntropy(a, b, 0)");
        AssertClose([sparse(logits, labels).Item()], [Losses.SparseCrossEntropy(logits, labels, 0f).Item()], 0f, "SparseCrossEntropy(a, b) == (a, b, 0)");
        AssertClose([dense(logits, oneHot).Item()], [sparse(logits, labels).Item()], 1e-6f, "sparse == dense");
        AssertClose([Losses.CrossEntropy(logits, oneHot, labelSmoothing: 0.1f).Item()], [Losses.CrossEntropy(logits, oneHot, 0.1f).Item()], 0f, "named smoothing still binds");

        using var model = Network.Input(3).OnDevice(device).Seed(1).Linear(3).Build();
        var run = new TrainingRun
        {
            Model = model, Loss = Losses.CrossEntropy, Optimizer = p => new Sgd(p, 0.1f),
            Train = Dataset.FromFlat([1f, 0f, 0f, 0f, 1f, 0f], [1f, 0f, 0f, 0f, 1f, 0f], 2, ["a", "b", "c"], ["x", "y", "z"]).Batches(2, device: device),
            Epochs = 2,
        };
        var history = run.Fit();
        Check(history.Epochs.Count == 2 && history.Epochs[1].Loss < history.Epochs[0].Loss, "TrainingRun with Loss = Losses.CrossEntropy trains");
    }

    private static Sequential ManualMlp(Device device, Random r) => new()
    {
        new Linear(9, 16, device: device, random: r), new ReLU(), new Dropout(0.1f, r),
        new Linear(16, 8, device: device, random: r), new Tanh(),
        new Linear(8, 1, device: device, random: r),
    };

    private static NetworkBuilder BuiltMlp(Device device) =>
        Network.Input(9).OnDevice(device).Seed(7).Linear(16).ReLU().Dropout(0.1f).Linear(8).Tanh().Linear(1);

    private static void SequentialInferenceFreesLayers(Device device)
    {
        var builder = Network.Input(128).OnDevice(device).Seed(3);
        for (int i = 0; i < 6; i++)
        {
            builder = builder.Linear(128).ReLU();
        }

        using var net = builder.Linear(8).Build();
        var random = new Random(5);
        var data = Enumerable.Range(0, 256 * 128).Select(_ => (float)random.NextDouble() - 0.5f).ToArray();
        using var x = Tensor.From(data, [256, 128], device);

        float[] expected, firstFour;
        using (var scope = new TensorScope())
        {
            expected = net.Forward(x).ToArray();                                          // autograd on: every result kept
            var h = x;
            for (int i = 0; i < 4; i++)
            {
                h = net[i].Forward(h);
            }

            firstFour = h.ToArray();
        }

        using (Autograd.NoGrad())
        using (var scope = new TensorScope())
        {
            long before = ComputeResources.GetMemoryUsage(device).InUse;
            var y = net.Forward(x);
            long held = ComputeResources.GetMemoryUsage(device).InUse - before;
            Check(y.ToArray().SequenceEqual(expected), "same outputs without autograd");
            Check(held < 256 * 128 * sizeof(float), $"only the output is held after the pass ({held:N0} bytes; one hidden layer is {256 * 128 * 4:N0})");
            Check(x.ToArray().SequenceEqual(data), "the input is left alone");
            Check(net.ForwardFirst(x, 4).ToArray().SequenceEqual(firstFour), "ForwardFirst matches the first layers run one by one");
        }
    }

    private static void BuilderMatchesSequential(Device device)
    {
        using var manual = ManualMlp(device, new Random(7));
        using var built = BuiltMlp(device).Build();
        Check(built.Count == manual.Count, "layer count");
        for (int i = 0; i < manual.Count; i++)
        {
            Check(built[i].ToString() == manual[i].ToString(), $"layer {i}: {built[i]} vs {manual[i]}");
        }

        var a = manual.Parameters().SelectMany(p => p.ToArray()).ToArray();
        var b = built.Parameters().SelectMany(p => p.ToArray()).ToArray();
        Check(a.SequenceEqual(b), "same seed must give identical initial weights");

        var x = new float[4, 9];
        for (int i = 0; i < 36; i++) x[i / 9, i % 9] = i * 0.1f - 1.5f;
        Check(manual.Predict(x).Cast<float>().SequenceEqual(built.Predict(x).Cast<float>()), "same outputs");
    }

    private static void BuilderShapesAndJson(Device device)
    {
        var cnn = Architectures.Cnn(1, 16, 16, [16, 32], 3, 4).OnDevice(device).Seed(1);
        Check(cnn.CurrentShape.SequenceEqual([4]), $"cnn output {string.Join(",", cnn.CurrentShape)}");
        using (var model = cnn.Build())
        using (var input = Tensor.Zeros([2, 1, 16, 16], device))
        using (var output = model.Predict(input))
        {
            Check(output.Shape.SequenceEqual([2, 4]), "cnn forward shape");
            Check(model.OfType<Flatten>().Any(), "flatten present");
        }

        var gpt = Architectures.Gpt(vocabulary: 20, context: 12, dim: 16, heads: 4, layers: 2, ffDim: 32, dropout: 0f).OnDevice(device).Seed(3);
        var json = gpt.ToJson();
        var replayed = Network.FromJson(json).OnDevice(device);
        Check(replayed.ToJson().ToJsonString() == json.ToJsonString(), "JSON round trip");
        using (var a = gpt.Build())
        using (var b = replayed.Build())
        {
            Check(a.Parameters().SelectMany(p => p.ToArray()).SequenceEqual(b.Parameters().SelectMany(p => p.ToArray())), "replayed weights");
            Check(Network.ArchitectureOf(a)?.ToJsonString() == json.ToJsonString(), "architecture registered");
            using var ids = Tensor.Zeros([2, 12], device);
            using var logits = a.Predict(ids);
            Check(logits.Shape.SequenceEqual([2, 12, 20]), "gpt forward shape");
        }

        var rnn = Architectures.Rnn(RecurrentCell.GRU, vocabulary: 10, length: 6, embed: 8, hidden: 12, outputs: 2);
        Check(rnn.CurrentShape.SequenceEqual([2]), "rnn output");
        bool threw = false;
        try { Network.Input(4).Conv2d(8, 3); } catch (InvalidOperationException) { threw = true; }
        Check(threw, "Conv2d after a feature input must fail with a clear error");
        threw = false;
        try { Network.Input(4).Linear(8, inputs: 5); } catch (InvalidOperationException) { threw = true; }
        Check(threw, "explicit inputs are checked");
        var custom = Network.Input(4).Add(new Linear(4, 3), [3]);
        Check(!custom.IsDescribable, "a custom module makes the builder non-describable");
    }

    private static Dataset SmallRegression(int count, int seed)
    {
        var r = new Random(seed);
        var x = new float[count, 3];
        var y = new float[count, 1];
        for (int i = 0; i < count; i++)
        {
            for (int j = 0; j < 3; j++) x[i, j] = r.NextSingle() * 10 + j * 5;
            y[i, 0] = 2 * x[i, 0] - x[i, 1] + 0.5f * x[i, 2] + r.NextSingle();
        }

        return Dataset.FromArrays(x, y);
    }

    private static void DataExtensionsMatchManual(Device device)
    {
        var data = SmallRegression(100, 1);
        var (train, test) = data.Split(0.8, seed: 2);
        var fs = StandardScaler.FitFeatures(train);
        var ts = StandardScaler.FitTargets(train);
        var trainScaled = train.Scale(fs, ts);
        var testScaled = test.Scale(fs, ts);

        var split = data.Split(0.8, seed: 2).StandardizeFeatures().StandardizeTargets();
        Check(split.Train.Features.SequenceEqual(trainScaled.Features) && split.Train.Targets.SequenceEqual(trainScaled.Targets), "train part");
        Check(split.Test.Features.SequenceEqual(testScaled.Features) && split.Test.Targets.SequenceEqual(testScaled.Targets), "test part");
        Check(split.FeatureScaler is StandardScaler f && f.Mean.SequenceEqual(fs.Mean), "feature scaler kept");

        var loader = split.Train.Batches(16, shuffle: true, device: device, seed: 4);
        Check(loader.BatchSize == 16 && loader.Shuffle && loader.Device == device && loader.BatchCount == 5, "Batches = DataLoader");
        var normalized = data.Split(0.8, seed: 2).NormalizeFeatures();
        Check(normalized.Train.Features.ToArray().All(v => v >= -1e-6f && v <= 1 + 1e-6f), "min-max scaled into [0, 1]");

        var path = Path.GetTempFileName();
        ((MinMaxScaler)normalized.FeatureScaler!).Save(path);
        var loaded = MinMaxScaler.Load(path);
        File.Delete(path);
        Check(loaded.Min.SequenceEqual(((MinMaxScaler)normalized.FeatureScaler).Min), "MinMaxScaler save/load");
    }

    private static TrainingHistory ManualFit(Device device, DataSplit split)
    {
        using var model = BuiltMlpFor3(device);
        using var optimizer = new AdamW(model.Parameters(), 0.01f, weightDecay: 1e-4f);
        var trainer = new Trainer(model, optimizer, Losses.MeanSquaredError)
        {
            Metrics = { Metric.MeanAbsoluteError },
            Scheduler = new CosineAnnealing(optimizer, 6, warmupEpochs: 1),
            MaxGradientNorm = 1f,
        };
        return trainer.Fit(new DataLoader(split.Train, 16, shuffle: true, device: device, seed: 5), 6,
            validation: new DataLoader(split.Test, 64, device: device));
    }

    private static Sequential BuiltMlpFor3(Device device) =>
        Network.Input(3).OnDevice(device).Seed(11).Linear(16).ReLU().Linear(1).Build();

    private static void TrainingRunMatchesTrainer(Device device)
    {
        var split = SmallRegression(200, 3).Split(0.8, seed: 1).StandardizeFeatures().StandardizeTargets();
        var manual = ManualFit(device, split);

        using var model = BuiltMlpFor3(device);
        var history = new TrainingRun
        {
            Model = model,
            Loss = Losses.MeanSquaredError,
            Optimizer = p => new AdamW(p, 0.01f, weightDecay: 1e-4f),
            Scheduler = o => new CosineAnnealing(o, 6, warmupEpochs: 1),
            Train = split.Train.Batches(16, shuffle: true, device: device, seed: 5),
            Validation = split.Test.Batches(64, device: device),
            Epochs = 6,
            Metrics = [Metric.MeanAbsoluteError],
            MaxGradientNorm = 1f,
        }.Fit();

        Check(history.Epochs.Count == manual.Epochs.Count, "epoch count");
        for (int i = 0; i < manual.Epochs.Count; i++)
        {
            Check(Math.Abs(history.Epochs[i].Loss - manual.Epochs[i].Loss) < 1e-9 &&
                  Math.Abs(history.Epochs[i].ValidationLoss!.Value - manual.Epochs[i].ValidationLoss!.Value) < 1e-9 &&
                  history.Epochs[i].LearningRate == manual.Epochs[i].LearningRate,
                $"epoch {i + 1}: {history.Epochs[i].Loss} vs {manual.Epochs[i].Loss}");
        }

        using var model2 = BuiltMlpFor3(device);
        using var trainer = new Trainer(model2, Losses.MeanSquaredError, p => new AdamW(p, 0.01f, weightDecay: 1e-4f),
            o => new CosineAnnealing(o, 6, warmupEpochs: 1)) { MaxGradientNorm = 1f, Metrics = { Metric.MeanAbsoluteError } };
        var h2 = trainer.Fit(split.Train.Batches(16, shuffle: true, device: device, seed: 5), 6, split.Test.Batches(64, device: device));
        Check(Math.Abs(h2.Epochs[^1].Loss - manual.Epochs[^1].Loss) < 1e-9, "factory constructor");
    }

    private static void TrainingAsync(Device device)
    {
        var split = SmallRegression(120, 5).Split(0.8, seed: 1).StandardizeFeatures().StandardizeTargets();
        using var model = BuiltMlpFor3(device);
        var run = new TrainingRun
        {
            Model = model, Loss = Losses.MeanSquaredError, Optimizer = p => new Adam(p, 0.01f),
            Train = split.Train.Batches(16, device: device), Epochs = 5,
        };
        var seen = new List<int>();
        var progress = new SyncProgress<EpochCompleted>(e => seen.Add(e.Epoch));
        var history = run.FitAsync(progress).GetAwaiter().GetResult();
        Check(history.Epochs.Count == 5 && seen.SequenceEqual([1, 2, 3, 4, 5]), $"progress {string.Join(",", seen)}");

        using var model2 = BuiltMlpFor3(device);
        var epochs = new List<int>();
        Task.Run(async () =>
        {
            await foreach (var e in (run with { Model = model2, Epochs = 50 }).TrainAsync())
            {
                epochs.Add(e.Epoch);
                if (e.Epoch == 3) break;
            }
        }).GetAwaiter().GetResult();
        Check(epochs.SequenceEqual([1, 2, 3]), $"early exit {string.Join(",", epochs)}");
    }

    private sealed class SyncProgress<T>(Action<T> report) : IProgress<T>
    {
        public void Report(T value) => report(value);
    }

    private static void FreezingAndGroups(Device device)
    {
        using var model = Network.Input(3).OnDevice(device).Seed(2).Linear(8).ReLU().Linear(8).ReLU().Linear(1).Build();
        model.Freeze(0..^1);
        Check(model.TrainableParameters().Count() == 2 && model[^1].Parameters().All(p => p.RequiresGrad), "only the head trains");
        var before = model[0].Parameters().First().ToArray();
        var split = SmallRegression(64, 9).Split(0.75, seed: 1).StandardizeFeatures().StandardizeTargets();
        using (var trainer = new Trainer(model, Losses.MeanSquaredError, p => new Adam(p, 0.05f)))
        {
            Check(trainer.Optimizer.Parameters.Count == 2, "factory receives trainable parameters only");
            trainer.Fit(split.Train.Batches(16, device: device), 3);
        }

        Check(model[0].Parameters().First().ToArray().SequenceEqual(before), "frozen layer unchanged");
        var path = Path.GetTempFileName();
        model.SaveTrainable(path);
        using var other = Network.Input(3).OnDevice(device).Seed(99).Linear(8).ReLU().Linear(8).ReLU().Linear(1).Build();
        other.Freeze(0..^1);
        other.LoadTrainable(path);
        Check(other[^1].Parameters().First().ToArray().SequenceEqual(model[^1].Parameters().First().ToArray()), "head restored");
        Check(new FileInfo(path).Length < 200, "only the head was saved");
        File.Delete(path);
        model.Unfreeze();
        Check(model.TrainableParameters().Count() == 6, "unfreeze");

        using var body = Network.Input(3).OnDevice(device).Seed(4).Linear(4).Build();
        using var head = Network.Input(4).OnDevice(device).Seed(5).Linear(1).Build();
        using var grouped = new GroupedOptimizer(new Sgd(body.Parameters(), 0.01f), new Sgd(head.Parameters(), 0.1f));
        grouped.LearningRate = 0.005f;                 // a schedule halving the rate
        grouped.Step();
        Check(Math.Abs(grouped.Groups[0].LearningRate - 0.005f) < 1e-9 && Math.Abs(grouped.Groups[1].LearningRate - 0.05f) < 1e-7, "ratio kept");
    }

    private static void Lora(Device device)
    {
        using var model = Architectures.Gpt(vocabulary: 12, context: 8, dim: 16, heads: 2, layers: 1, ffDim: 32, dropout: 0f).OnDevice(device).Seed(8).Build();
        var ids = new float[2 * 8];
        for (int i = 0; i < ids.Length; i++) ids[i] = i % 12;
        using var x = Tensor.From(ids, [2, 8], device);
        float[] before;
        using (var y = model.Predict(x)) before = y.ToArray();

        int added = model.AddLora(rank: 4, alpha: 8, targets: _ => true, freezeBase: true, random: new Random(1));
        Check(added == 5, $"adapters on qkv, output, ff1, ff2 and head: {added}");
        using (var y = model.Predict(x)) Check(y.ToArray().SequenceEqual(before), "B = 0: outputs unchanged");
        Check(model.TrainableParameters().Count() == 10, "only A and B train");

        var targets = new float[2 * 8];
        for (int i = 0; i < targets.Length; i++) targets[i] = (i + 1) % 12;
        using var t = Tensor.From(targets, [2, 8], device);
        using (var trainer = new Trainer(model, (p, y) => Losses.SparseCrossEntropy(p, y), p => new Adam(p, 0.05f)))
        {
            for (int step = 0; step < 5; step++)
            {
                using var scope = new TensorScope();
                trainer.Optimizer.ZeroGrad();
                Losses.SparseCrossEntropy(model.Forward(x), t).Backward();
                trainer.Optimizer.Step();
            }
        }

        float[] adapted;
        using (var y = model.Predict(x)) adapted = y.ToArray();
        Check(!adapted.SequenceEqual(before), "training changed the outputs");
        int merged = model.MergeLora();
        using (var y = model.Predict(x))
        {
            var after = y.ToArray();
            Check(merged == 5 && after.Zip(adapted).All(p => MathF.Abs(p.First - p.Second) < 1e-4f), "merge keeps outputs");
        }

        Check(model.Parameters().Count() == model.Descendants().OfType<Linear>().Sum(l => l.Bias is null ? 1 : 2)
              + model.Descendants().OfType<LayerNorm>().Count() * 2 + 1, "adapters removed");
    }

    private static void TelemetryBuilderSession(Device device)
    {
        var levelsBefore = Telemetry.ActiveLevels;
        var console = new StringWriter();
        string path = Path.GetTempFileName();
        MetricsRecorder recorder;
        using (var session = Telemetry.Configure()
                   .Console(TelemetryLevel.Training, output: console)
                   .JsonLines(path, TelemetryLevel.Training)
                   .Record(out recorder)
                   .Start())
        {
            Check(session.HookCount == 3 && Telemetry.IsEnabled(TelemetryLevel.Training), "subscribed");
            var split = SmallRegression(64, 2).Split(0.75, seed: 1).StandardizeFeatures().StandardizeTargets();
            using var model = BuiltMlpFor3(device);
            new TrainingRun { Model = model, Loss = Losses.MeanSquaredError, Optimizer = p => new Adam(p, 0.01f),
                Train = split.Train.Batches(16, device: device), Epochs = 3 }.Fit();
        }

        Check(Telemetry.ActiveLevels == levelsBefore, "unsubscribed on dispose");
        Check(recorder.Epochs.Count == 3, "recorder got every epoch");
        Check(console.ToString().Contains("Epoch 3/3"), "console output");
        var lines = File.ReadAllLines(path);
        File.Delete(path);
        Check(lines.Count(l => l.Contains("\"event\":\"epoch\"")) == 3, $"json lines flushed ({lines.Length} lines)");
    }

    private sealed record Point(float A, float B, float C);

    private static void PredictorMatchesManual(Device device)
    {
        var split = SmallRegression(200, 7).Split(0.8, seed: 1).StandardizeFeatures().StandardizeTargets();
        using var model = BuiltMlpFor3(device);
        new TrainingRun { Model = model, Loss = Losses.MeanSquaredError, Optimizer = p => new Adam(p, 0.01f),
            Train = split.Train.Batches(16, device: device), Epochs = 5 }.Fit();

        float[,] raw = { { 1, 6, 11 }, { 9, 12, 17 }, { 4, 8, 13 } };
        float[] flat = [.. raw.Cast<float>()];
        split.FeatureScaler!.Transform(flat, 3);
        using var x = Tensor.From(flat, [3, 3], device);
        using var y = model.Predict(x);
        float[] manual = y.ToArray();
        split.TargetScaler!.InverseTransform(manual, 1);

        var predictor = Predictor.For(model).ScaleInputs(split.FeatureScaler).UnscaleOutputs(split.TargetScaler!).Build();
        var rows = predictor.Predict([[1f, 6, 11], [9f, 12, 17], [4f, 8, 13]]);
        Check(rows.Select(r => r[0]).SequenceEqual(manual), "untyped rows equal the manual pipeline");

        var typed = Predictor.For(model).Input<Point>(p => [p.A, p.B, p.C]).ScaleInputs(split.FeatureScaler)
            .UnscaleOutputs(split.TargetScaler!).Output(v => v[0]).WarmUp(new Point(1, 2, 3)).BatchSize(2).Build();
        Check(typed.Predict(new Point(9, 12, 17)) == manual[1], "typed single prediction");
        Check(typed.Predict([new Point(1, 6, 11), new Point(9, 12, 17), new Point(4, 8, 13)]).SequenceEqual(manual), "typed batch (2 + 1 rows)");
        Check(typed.PredictAsync(new Point(4, 8, 13)).AsTask().Result == manual[2], "async");

        bool threw = false;
        try { Predictor.For(model).Batching(8, TimeSpan.FromMilliseconds(1)).Build(); } catch (InvalidOperationException) { threw = true; }
        Check(threw, "engine-only settings are rejected outside the engine");

        // Without BatchSize, large inputs still go through in batches of DefaultBatchSize (a whole data set as one batch
        // exceeded CUDA's grid limits in a convolution).
        var sizes = new List<int>();
        using var recorder = Network.Input(3).OnDevice(device).Lambda(t => { sizes.Add(t.Shape[0]); return t; }, "record", [3]).Build();
        int count = 2 * Predictor<float[], float[]>.DefaultBatchSize + 7;
        var data = Dataset.FromArrays(new float[count, 3], new float[count, 1]);
        var unbatched = Predictor.For(recorder).Build();
        Check(unbatched.Predict(data).Count == count, "every row predicted");
        Check(sizes.SequenceEqual([Predictor<float[], float[]>.DefaultBatchSize, Predictor<float[], float[]>.DefaultBatchSize, 7]),
            $"Predict(Dataset) batches: {string.Join(", ", sizes)}");
        sizes.Clear();
        unbatched.Predict([.. Enumerable.Range(0, count).Select(_ => new float[3])]);
        Check(sizes.Max() == Predictor<float[], float[]>.DefaultBatchSize, $"Predict(list) batches: {string.Join(", ", sizes)}");
    }

    private static void PredictorClassesAndPackage(Device device)
    {
        var network = Architectures.Mlp(3, [8], 3, Activation.Tanh).OnDevice(device).Seed(21);
        using var model = network.Build();
        var scaler = StandardScaler.Fit([1f, 2, 3, 4, 5, 6, 7, 8, 9], 3);
        var classifier = Predictor.For(model).Input<Point>(p => [p.A, p.B, p.C]).ScaleInputs(scaler).Softmax()
            .Classes(["low", "mid", "high"]).Build();
        var answer = classifier.Predict(new Point(2, 5, 7));

        float[] flat = [2, 5, 7];
        scaler.Transform(flat, 3);
        using var x = Tensor.From(flat, [1, 3], device);
        using var logits = model.Predict(x);
        using var probabilities = logits.Softmax();
        var p = probabilities.ToArray();
        int best = Array.IndexOf(p, p.Max());
        Check(answer.Index == best && answer.Class == new[] { "low", "mid", "high" }[best], "argmax class");
        Check(MathF.Abs(answer.Probability - p[best]) < 1e-6f && MathF.Abs(answer.Scores.Sum(s => s.Score) - 1f) < 1e-5f, "probabilities");

        string path = Path.Combine(Path.GetTempPath(), $"predictor-{Guid.NewGuid():N}.ikm");
        classifier.Save(path);
        var loadedBuilder = Predictor.Load(path, device);
        Check(loadedBuilder.StoredClasses!.SequenceEqual(["low", "mid", "high"]), "classes stored");
        using var loaded = loadedBuilder.Input<Point>(q => [q.A, q.B, q.C]).Classes(loadedBuilder.StoredClasses!).Build();
        var again = loaded.Predict(new Point(2, 5, 7));
        Check(again.Class == answer.Class && again.Scores.Select(s => s.Score).SequenceEqual(answer.Scores.Select(s => s.Score)), "loaded predictor answers the same");

        using var manualModel = new Sequential { new Linear(3, 8, device: device), new Tanh(), new Linear(8, 3, device: device) };
        var intoManual = Predictor.Load(path, manualModel).Classes(["low", "mid", "high"]).Build();
        Check(intoManual.Predict([2f, 5, 7]).Class == answer.Class, "load into a hand-built model");
        File.Delete(path);
    }

    private static void PackageRoundTrip(Device device)
    {
        string path = Path.Combine(Path.GetTempPath(), $"package-{Guid.NewGuid():N}.ikm");
        var words = WordTokenizer.FromTexts(["the cat sat on the mat ."], ["<pad>"]);
        var gpt = Architectures.Gpt(vocabulary: words.VocabularySize, context: 10, dim: 16, heads: 2, layers: 1, ffDim: 32, dropout: 0f).OnDevice(device).Seed(5);
        using var model = gpt.Build();
        var chars = new CharTokenizer("abc ", ' ');
        var standard = StandardScaler.Fit([1f, 2, 3, 4], 2);
        var minMax = MinMaxScaler.Fit([1f, 5, 3, 9], 2);
        using var head = Network.Input(2).OnDevice(device).Seed(1).Linear(2).Build();

        ModelPackage.Create(path)
            .Architecture(gpt).Weights(model)
            .Tokenizer("words", words).Tokenizer("chars", chars)
            .Scaler("standard", standard).Scaler("minmax", minMax)
            .Json("settings", new System.Text.Json.Nodes.JsonObject { ["threshold"] = 0.42 })
            .Text("notes", "trained for a test")
            .TrainableWeights("head", head)
            .Save();

        using var package = ModelPackage.Open(path);
        Check(package.Entries.Count == 9, $"{package.Entries.Count} entries");
        using (var rebuilt = package.BuildNetwork(device: device))
        {
            Check(rebuilt.Parameters().SelectMany(t => t.ToArray()).SequenceEqual(model.Parameters().SelectMany(t => t.ToArray())), "rebuilt weights");
        }

        Check(package.WordTokenizer("words").Vocabulary.SequenceEqual(words.Vocabulary), "word tokenizer");
        Check(package.CharTokenizer("chars").Vocabulary == "abc " && package.CharTokenizer("chars").UnknownCharacter == ' ', "char tokenizer");
        Check(package.StandardScaler("standard").Mean.SequenceEqual(standard.Mean), "standard scaler");
        Check(package.Scaler("minmax") is MinMaxScaler m && m.Range.SequenceEqual(minMax.Range), "min-max scaler");
        Check((double)package.Json("settings")["threshold"]! == 0.42 && package.Text("notes") == "trained for a test", "json and text");
        using (var head2 = Network.Input(2).OnDevice(device).Seed(9).Linear(2).Build())
        {
            package.LoadTrainableWeights(head2, "head");
            Check(head2.Parameters().First().ToArray().SequenceEqual(head.Parameters().First().ToArray()), "trainable weights");
        }

        var generator = package.TextGenerator("words", device: device);
        var manual = new TextGenerator(model, words, 10);
        model.Eval();
        var options = new GenerationOptions { Temperature = 0f, TopK = 1, RepeatPenalty = 1f, NumPredict = 6 };
        Check(generator.Generate("the cat", options).Text == manual.Generate("the cat", options).Text, "package text generator");
        Check(generator.ContextLength == 10, "context from the architecture");
        generator.Model.Dispose();

        bool threw = false;
        try { package.StandardScaler("nope"); } catch (KeyNotFoundException) { threw = true; }
        Check(threw, "missing entries name what exists");
        package.Dispose();
        File.Delete(path);
    }

    private static void StreamAsyncMatches(Device device)
    {
        var (model, tokenizer) = TinyLanguageModel(device);
        using var _ = model;
        var generator = new TextGenerator(model, tokenizer, 32);
        var options = new GenerationOptions { Seed = 5, NumPredict = 20, ChunkSize = 4 };
        var sync = generator.Stream("abc", options).Select(c => c.Text).ToList();
        var async = new List<string>();
        Task.Run(async () => { await foreach (var c in generator.StreamAsync("abc", options)) async.Add(c.Text); }).GetAwaiter().GetResult();
        Check(sync.SequenceEqual(async), "same chunks");
        var chat = new ChatGenerator(generator);
        var request = new ChatRequest([new ChatMessage("user", "hi")], Options: options);
        var final = chat.ChatAsync(request).GetAwaiter().GetResult();
        Check(final.Done && final.Message is not null && final.Message.Content == chat.Chat(request).Message!.Content, "ChatAsync = Chat");
    }

    private sealed class Calculator
    {
        public int Calls;

        [Tool("add", "Adds two integers.")]
        public int Add([System.ComponentModel.Description("first")] int a, int b) { Calls++; return a + b; }

        [Tool("shout", "Upper-cases text.")]
        public static Task<string> Shout(string text, bool exclaim = false) => Task.FromResult(text.ToUpperInvariant() + (exclaim ? "!" : ""));
    }

    private static ToolCall Call(string name, string json) => new(name, System.Text.Json.Nodes.JsonNode.Parse(json)!.AsObject());

    private static void ToolRules(Device device)
    {
        _ = device;
        var calculator = new Calculator();
        int slowCalls = 0;
        var tools = ToolRegistry.Create()
            .Add(calculator)
            .Add("pick", "Picks a colour.", (Colour colour, CancellationToken token) => colour.ToString())
            .Add("slow", "Takes a while.", async (CancellationToken token) => { slowCalls++; await Task.Delay(5000, token); return "late"; })
            .Allow("add", args => (int)args["a"]! < 100)
            .RequireApproval("shout", (call, _) => ValueTask.FromResult((string?)call.Arguments["text"] != "no"))
            .Timeout(TimeSpan.FromMilliseconds(100))
            .Parallel()
            .Build();

        var add = tools.Definitions.Single(d => d.Name == "add").Parameters!.ToJsonString();
        Check(add.Contains("\"a\":{\"type\":\"integer\",\"description\":\"first\"}") && add.Contains("\"required\":[\"a\",\"b\"]"), $"schema {add}");
        var shout = tools.Definitions.Single(d => d.Name == "shout").Parameters!.ToJsonString();
        Check(shout.Contains("\"required\":[\"text\"]"), "defaulted parameter is optional");
        Check(tools.Definitions.Single(d => d.Name == "pick").Parameters!.ToJsonString().Contains("\"enum\":[\"Red\",\"Green\"]"), "enum schema");

        ToolResult Run(string name, string json) => tools.InvokeAsync(Call(name, json)).GetAwaiter().GetResult();
        Check(Run("add", "{\"a\":2,\"b\":3}") is { Succeeded: true, Content: "5" }, "add runs");
        Check(Run("add", "{\"a\":2}").Error!.Contains("'b' is required"), "missing argument");
        Check(Run("add", "{\"a\":\"two\",\"b\":3}").Error!.Contains("'a' must be an integer"), "wrong type");
        Check(Run("add", "{\"a\":500,\"b\":3}").Error!.Contains("not allowed"), "allowlist");
        Check(calculator.Calls == 1, "rejected calls never reach the method");
        Check(Run("shout", "{\"text\":\"hi\",\"exclaim\":true}").Content == "HI!", "static tool, awaited");
        Check(Run("shout", "{\"text\":\"no\"}").Error!.Contains("not approved"), "approval");
        Check(Run("pick", "{\"colour\":\"Blue\"}").Error!.Contains("one of Red, Green"), "enum validation");
        Check(Run("pick", "{\"colour\":\"Green\"}").Content == "Green", "enum argument");
        Check(Run("slow", "{}").Error!.Contains("timed out"), "timeout");
        Check(Run("missing", "{}").Error!.Contains("unknown tool"), "unknown tool");
        Check(Run("add", "{\"a\":1,\"b\":1}").ToMessage() is { Role: "tool", ToolName: "add", Content: "2" }, "tool message");

        var clock = System.Diagnostics.Stopwatch.StartNew();
        var both = tools.InvokeAsync([Call("slow", "{}"), Call("slow", "{}")]).GetAwaiter().GetResult();
        Check(both.Count == 2 && clock.ElapsedMilliseconds < 190 && slowCalls == 3, $"parallel ({clock.ElapsedMilliseconds} ms)");

        using var http = new HttpClient(new FakeHttp("<html><script>x()</script><p>Ollama&nbsp;0.12.3   is  out</p></html>"));
        var web = ToolRegistry.Create().Add(WebTools.Fetch(http, url => url.Host == "ollama.com", maxCharacters: 12)).Build();
        var page = web.InvokeAsync(Call("web_fetch", "{\"url\":\"https://ollama.com/releases\"}")).GetAwaiter().GetResult();
        Check(page.Content == "Ollama 0.12.", $"page text '{page.Content}'");
        Check(web.InvokeAsync(Call("web_fetch", "{\"url\":\"https://evil.example/\"}")).GetAwaiter().GetResult().Error!.Contains("allowlist"), "web allowlist");
    }

    private enum Colour { Red, Green }

    private sealed class FakeHttp(string body) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
            Task.FromResult(new HttpResponseMessage(System.Net.HttpStatusCode.OK) { Content = new StringContent(body) });
    }

    private static void ConversationLoop(Device device)
    {
        _ = device;
        var fake = FakeChatModel.Script(
            FakeChatModel.ToolCall("web_fetch", new System.Text.Json.Nodes.JsonObject { ["url"] = "https://ollama.com/releases" }),
            FakeChatModel.Answer("The latest Ollama version is 0.12.3 [1].", thinking: "The page says 0.12.3."));
        string? fetched = null;
        var conversation = Conversation.For(fake)
            .System("You are a helpful assistant.")
            .Think(true)
            .Tool("web_fetch", "Fetch a page.", (string url) => { fetched = url; return "[1] Ollama 0.12.3 is the latest release."; })
            .MaxToolRounds(3)
            .Build();
        var reply = conversation.SendAsync("What is the latest Ollama version?").GetAwaiter().GetResult();
        Check(reply.Message.Content.Contains("0.12.3") && reply.Rounds == 2 && !reply.ToolLimitReached, "answer after one tool round");
        Check(fetched == "https://ollama.com/releases" && reply.ToolResults.Single().Succeeded, "tool ran");
        Check(conversation.Messages.Select(m => m.Role).SequenceEqual(["system", "user", "assistant", "tool", "assistant"]), "history");
        Check(fake.Requests.Count == 2 && fake.Requests[1].Messages.Count == 4 && fake.Requests[0].Think == true && fake.Requests[0].Tools!.Count == 1, "requests");

        bool threw = false;
        try { Conversation.For(fake).Tool("x", "y", () => "z").Build(); } catch (InvalidOperationException) { threw = true; }
        Check(threw, "MaxToolRounds is required with tools");

        var looping = FakeChatModel.Script(
            FakeChatModel.ToolCall("ping", []), FakeChatModel.ToolCall("ping", []), FakeChatModel.ToolCall("ping", []));
        var limited = Conversation.For(looping).Tool("ping", "Ping.", () => "pong").MaxToolRounds(2).Build()
            .SendAsync("go").GetAwaiter().GetResult();
        Check(limited.ToolLimitReached && limited.Rounds == 3 && limited.ToolResults.Count == 2, "limit enforced");

        var noTools = Conversation.For(FakeChatModel.Script(FakeChatModel.ToolCall("web_fetch", []))).Build();
        var clientSide = noTools.SendAsync("hi").GetAwaiter().GetResult();
        Check(clientSide.Message.ToolCalls!.Count == 1 && !clientSide.ToolLimitReached, "without tools the calls are returned");
    }

    private static Predictor<Point, float> DirectPredictor(Sequential model, DataSplit split) =>
        Predictor.For(model).Input<Point>(p => [p.A, p.B, p.C]).ScaleInputs(split.FeatureScaler!).UnscaleOutputs(split.TargetScaler!).Output(v => v[0]).Build();

    private static void EnginePredictors(Device device)
    {
        var split = SmallRegression(200, 7).Split(0.8, seed: 1).StandardizeFeatures().StandardizeTargets();
        using var model = BuiltMlpFor3(device);
        new TrainingRun { Model = model, Loss = Losses.MeanSquaredError, Optimizer = p => new Adam(p, 0.01f),
            Train = split.Train.Batches(16, device: device), Epochs = 3 }.Fit();
        var direct = DirectPredictor(model, split);
        var points = Enumerable.Range(0, 37).Select(i => new Point(i % 10, i % 7 + 5, i % 5 + 10)).ToList();
        var expected = direct.Predict(points);

        string weights = Path.GetTempFileName();
        model.Save(weights);
        string package = Path.Combine(Path.GetTempPath(), $"engine-{Guid.NewGuid():N}.ikm");
        direct.Save(package);
        int warmups = 0;
        var engine = InferenceEngine.Create()
            .Predictor<Point, float>("factory", () => { var m = BuiltMlpFor3(device); m.Load(weights); return m; }, p => p
                .Input<Point>(q => { if (q.A < 0) warmups++; return [q.A, q.B, q.C]; })
                .ScaleInputs(split.FeatureScaler!).UnscaleOutputs(split.TargetScaler!).Output(v => v[0])
                .WarmUp(new Point(-1, 0, 0)))
            .Predictor<Point, float>("package", package, p => p.OnDevice(device).Input<Point>(q => [q.A, q.B, q.C]).Output(v => v[0]))
            .Predictor<Point, float>("batched", model, p => p.Input<Point>(q => [q.A, q.B, q.C])
                .ScaleInputs(split.FeatureScaler!).UnscaleOutputs(split.TargetScaler!).Output(v => v[0])
                .Batching(maxBatch: 8, maxWait: TimeSpan.FromMilliseconds(40)))
            .BuildAsync().GetAwaiter().GetResult();
        try
        {
            Check(warmups == 1, "warm-up ran once at load");
            Check(engine.PredictAsync<Point, float>("factory", points[3]).AsTask().Result == expected[3], "single");
            Check(engine.PredictAsync<Point, float>("package", points).AsTask().Result.SequenceEqual(expected), "package source, batch");
            var many = new List<float>();
            Task.Run(async () => { await foreach (var v in engine.PredictManyAsync<Point, float>("factory", points, batchSize: 10)) many.Add(v); }).Wait();
            Check(many.SequenceEqual(expected), "PredictManyAsync keeps order");

            var concurrent = Task.WhenAll(points.Take(24).Select(p => engine.PredictAsync<Point, float>("batched", p).AsTask())).Result;
            Check(concurrent.SequenceEqual(expected.Take(24)), "micro-batched results");
            var stats = engine.Stats("batched");
            Check(stats.Requests >= 3 && stats.AverageBatchSize > 1 && stats.Rows == 24, $"batching stats: {stats}");
            Check(engine.Stats("factory").Requests == 5, $"factory stats {engine.Stats("factory").Requests}");
            Check(engine.Models.All(m => m.Loaded && m.Kind == "predictor"), "status");

            bool threw = false;
            try { engine.PredictAsync<string, float>("factory", "x").AsTask().Wait(); } catch (InvalidOperationException) { threw = true; }
            Check(threw, "wrong types give a clear error");
            threw = false;
            try { InferenceEngine.Create().Predictor<Point, float>("x", model, p => p.Input<Point>(q => [q.A]).Output(v => v[0]).Instances(2)); }
            catch (InvalidOperationException) { threw = true; }
            Check(threw, "copies of a fixed model object are refused");
        }
        finally
        {
            engine.DisposeAsync().AsTask().Wait();
            File.Delete(weights);
            File.Delete(package);
        }
    }

    private sealed class TimerClock : TimeProvider
    {
        private readonly List<ManualTimer> _timers = [];
        private DateTimeOffset _now = new(2026, 1, 1, 0, 0, 0, TimeSpan.Zero);

        public override DateTimeOffset GetUtcNow() => _now;

        public override ITimer CreateTimer(TimerCallback callback, object? state, TimeSpan dueTime, TimeSpan period)
        {
            var timer = new ManualTimer(this, callback, state, _now + dueTime);
            lock (_timers) _timers.Add(timer);
            return timer;
        }

        public void Advance(TimeSpan by)
        {
            _now += by;
            List<ManualTimer> due;
            lock (_timers) due = _timers.Where(t => t.Due <= _now).ToList();
            foreach (var t in due)
            {
                lock (_timers) _timers.Remove(t);
                t.Fire();
            }
        }

        private sealed class ManualTimer(TimerClock clock, TimerCallback callback, object? state, DateTimeOffset due) : ITimer
        {
            public DateTimeOffset Due { get; private set; } = due;
            public void Fire() => callback(state);
            public bool Change(TimeSpan dueTime, TimeSpan period) { Due = clock._now + dueTime; return true; }
            public void Dispose() { lock (clock._timers) clock._timers.Remove(this); }
            public ValueTask DisposeAsync() { Dispose(); return ValueTask.CompletedTask; }
        }
    }

    private static void EngineLifecycle(Device device)
    {
        int loads = 0;
        TextGenerator Load()
        {
            loads++;
            var (m, t) = TinyLanguageModel(device);
            return new TextGenerator(m, t, 32);
        }

        var slow = new GenerationOptions { Seed = 1, NumPredict = 200, ChunkSize = 1 };
        var clock = new TimerClock();
        var events = new List<EngineEvent>();
        using var hook = Telemetry.Configure().Hook(new EngineRecorder(events)).Start();
        var engine = InferenceEngine.Create()
            .TextModel("one", Load, m => m.Instances(1).QueueLimit(1).Timeout(TimeSpan.FromSeconds(30)))
            .TextModel("two", Load, m => m.Instances(2))
            .TextModel("idle", Load, m => m.KeepAlive(TimeSpan.FromMinutes(5)))
            .TextModel("each", Load, m => m.KeepAlive(TimeSpan.Zero))
            .TimeProvider(clock)
            .LoadOnFirstUse()
            .Telemetry()
            .BuildAsync().GetAwaiter().GetResult();
        try
        {
            Check(loads == 0 && engine.Models.All(m => !m.Loaded), "nothing loaded before first use");

            // One copy busy: a second request queues, a third is rejected.
            var first = engine.Model<ITextModel>("one").StreamAsync("abc", slow).GetAsyncEnumerator();
            Check(first.MoveNextAsync().AsTask().Result, "first request running");
            var second = engine.Model<ITextModel>("one").StreamAsync("abc", slow).GetAsyncEnumerator();
            var secondStarted = second.MoveNextAsync().AsTask();
            SpinWait.SpinUntil(() => engine.Models.First(m => m.Name == "one").Queued == 1, 2000);
            bool rejected = false;
            try { engine.Model<ITextModel>("one").GenerateAsync("abc", slow).Wait(); }
            catch (AggregateException ex) when (ex.InnerException is InferenceQueueFullException) { rejected = true; }
            Check(rejected && engine.Stats("one").Rejected == 1, "queue limit");
            Check(!secondStarted.Wait(50), "second waits for the copy");
            first.DisposeAsync().AsTask().Wait();
            Check(secondStarted.Result, "second runs after the first ends");
            second.DisposeAsync().AsTask().Wait();

            // Two copies: two streams at once, none queued.
            var a = engine.Model<ITextModel>("two").StreamAsync("abc", slow).GetAsyncEnumerator();
            var b = engine.Model<ITextModel>("two").StreamAsync("abc", slow).GetAsyncEnumerator();
            Check(a.MoveNextAsync().AsTask().Result && b.MoveNextAsync().AsTask().Wait(2000), "two copies run together");
            a.DisposeAsync().AsTask().Wait();
            b.DisposeAsync().AsTask().Wait();
            Check(loads == 3, $"copies loaded on first use ({loads})");

            // Keep-alive: unloaded after 5 idle minutes, reloaded on the next request.
            engine.Model<ITextModel>("idle").GenerateAsync("abc", new GenerationOptions { Seed = 1, NumPredict = 3 }).Wait();
            var status = engine.Models.First(m => m.Name == "idle");
            Check(status.Loaded && status.ExpiresAt == clock.GetUtcNow() + TimeSpan.FromMinutes(5), "expiry scheduled");
            clock.Advance(TimeSpan.FromMinutes(4));
            Check(engine.Models.First(m => m.Name == "idle").Loaded, "still loaded after 4 minutes");
            clock.Advance(TimeSpan.FromMinutes(1));
            SpinWait.SpinUntil(() => !engine.Models.First(m => m.Name == "idle").Loaded, 2000);
            Check(!engine.Models.First(m => m.Name == "idle").Loaded, "unloaded after 5 minutes");
            engine.Model<ITextModel>("idle").GenerateAsync("abc", new GenerationOptions { Seed = 1, NumPredict = 3 }).Wait();
            Check(loads == 5, $"reloaded ({loads})");

            engine.Model<ITextModel>("each").GenerateAsync("abc", new GenerationOptions { Seed = 1, NumPredict = 3 }).Wait();
            SpinWait.SpinUntil(() => !engine.Models.First(m => m.Name == "each").Loaded, 2000);
            Check(!engine.Models.First(m => m.Name == "each").Loaded, "keep-alive zero unloads at once");
            Check(events.Any(e => e.Kind == EngineEventKind.ModelLoaded) && events.Any(e => e.Kind == EngineEventKind.ModelUnloaded)
                  && events.Any(e => e.Kind == EngineEventKind.RequestCompleted) && events.Any(e => e.Kind == EngineEventKind.RequestRejected), "telemetry");
        }
        finally
        {
            engine.DisposeAsync().AsTask().Wait();
        }

        var timed = InferenceEngine.Create().TextModel("t", Load, m => m.Timeout(TimeSpan.FromMilliseconds(30))).BuildAsync().GetAwaiter().GetResult();
        try
        {
            var holder = timed.Model<ITextModel>("t").StreamAsync("abc", slow).GetAsyncEnumerator();
            holder.MoveNextAsync().AsTask().Wait();
            bool timedOut = false;
            try { timed.Model<ITextModel>("t").GenerateAsync("abc", slow).Wait(); }
            catch (AggregateException ex) when (ex.InnerException is TimeoutException) { timedOut = true; }
            Check(timedOut && timed.Stats("t").Failed == 1, "timeout while waiting for a copy");
            holder.DisposeAsync().AsTask().Wait();
        }
        finally
        {
            timed.DisposeAsync().AsTask().Wait();
        }
    }

    private sealed class EngineRecorder(List<EngineEvent> events) : ITelemetryHook
    {
        public TelemetryLevel Levels => TelemetryLevel.Engine;
        public void OnEngine(in EngineEvent e) { lock (events) events.Add(e); }
    }

    private static void EngineGeneration(Device device)
    {
        var (model, tokenizer) = TinyLanguageModel(device);
        var direct = new TextGenerator(model, tokenizer, 32);
        var options = new GenerationOptions { Seed = 9, NumPredict = 12 };
        string expected = direct.Generate("abc", options).Text;
        var engine = InferenceEngine.Create()
            .TextModel("text", direct)
            .ChatModel("chat", () => { var (m, t) = TinyLanguageModel(device); return new TextGenerator(m, t, 32); }, c => c.WarmUp("a"))
            .BuildAsync().GetAwaiter().GetResult();
        try
        {
            Check(engine.Model<ITextModel>("text").GenerateAsync("abc", options).Result.Text == expected, "engine text = direct text");
            var reply = Conversation.For(engine.Model<IChatModel>("chat")).System("be brief").Options(options).Build().SendAsync("hi").Result;
            Check(reply.Rounds == 1 && reply.Message.Role == "assistant", "conversation through the engine");
            var chunk = engine.Model<IChatModel>("chat").ChatAsync(new ChatRequest([new ChatMessage("user", "hi")], Options: options)).Result;
            Check(chunk.Done && chunk.Message is not null, "chat request");
            Check(engine.DescribeAsync("chat").Result is { Kind: "chat", ContextLength: 32 }, "context length");
            bool threw = false;
            try { engine.Model<IChatModel>("text"); } catch (InvalidOperationException) { threw = true; }
            Check(threw, "chat on a text model is refused");
        }
        finally
        {
            engine.DisposeAsync().AsTask().Wait();
            model.Dispose();
        }
    }

    // A model kind as a domain package would write one: the engine lends it copies and batches its requests.
    private sealed class EchoModel(EngineHosting hosting, int maxBatch = 0) : EngineModel<EchoModel.Copy>("echo", hosting)
    {
        private int _loads;
        private IEngineBatcher<string, string>? _batcher;

        public sealed record Copy(int Id);

        public List<Copy> Unloaded { get; } = [];

        public List<int> Batches { get; } = [];

        public bool Disposed { get; private set; }

        protected override void OnAttached()
        {
            if (maxBatch > 0)
            {
                _batcher = Host.Batcher<string, string>(maxBatch, TimeSpan.FromMilliseconds(500), RunAsync);
            }
        }

        public override Copy LoadCopy() => new(Interlocked.Increment(ref _loads));

        public override void UnloadCopy(Copy copy) { lock (Unloaded) Unloaded.Add(copy); }

        public override ModelDescription Describe(Copy copy) => new(Name, Kind, 0, Device.Cpu, null);

        public async Task<string> SayAsync(string text, CancellationToken token = default) =>
            _batcher is { } batcher ? await batcher.SubmitAsync(text, token) : (await RunAsync([text], token))[0];

        private async Task<IReadOnlyList<string>> RunAsync(IReadOnlyList<string> texts, CancellationToken token)
        {
            using var lease = await Host.AcquireAsync(token);
            lock (Batches) Batches.Add(texts.Count);
            lease.Completed(texts.Count, texts.Sum(t => t.Length), TimeSpan.FromMilliseconds(1));
            return [.. texts.Select(t => $"{lease.Copy.Id}:{t}")];
        }

        public override ValueTask DisposeAsync()
        {
            Disposed = true;
            return base.DisposeAsync();
        }
    }

    private static void EngineOwnKind(Device device)
    {
        if (device != Device.Cpu)
        {
            return;     // no tensors: once is enough
        }

        var exclusive = new EchoModel(new EngineHosting { Instances = 2, KeepAlive = TimeSpan.Zero });
        var batched = new EchoModel(new EngineHosting { SharedCopies = true }, maxBatch: 4);
        var engine = InferenceEngine.Create().Add("echo", exclusive).Add("batched", batched).LoadOnFirstUse().BuildAsync().GetAwaiter().GetResult();
        try
        {
            Check(engine.Models.All(m => m.Kind == "echo" && !m.Loaded), "the kind's name in the status; nothing loaded before first use");
            Check(engine.Model<EchoModel>("echo").SayAsync("hi").Result is "1:hi" or "2:hi", "a request on a lent copy");
            SpinWait.SpinUntil(() => !engine.Models.First(m => m.Name == "echo").Loaded, 2000);
            Check(exclusive.Unloaded.Count == 2, "keep-alive zero unloads both copies through UnloadCopy");

            var answers = Task.WhenAll(Enumerable.Range(0, 4).Select(i => batched.SayAsync($"x{i}"))).Result;
            Check(answers.SequenceEqual(Enumerable.Range(0, 4).Select(i => $"1:x{i}")), "batched answers in order");
            Check(batched.Batches.Sum() == 4 && batched.Batches.Max() > 1, $"micro-batched by the engine ({string.Join(", ", batched.Batches)})");
            var stats = engine.Stats("batched");
            Check(stats.Rows == 8 && stats.Requests == batched.Batches.Count, $"statistics from the lease: {stats}");
            Check(engine.DescribeAsync("batched").Result is { Name: "batched", Kind: "echo" }, "description");

            bool threw = false;
            try { engine.Model<IChatModel>("echo"); } catch (InvalidOperationException) { threw = true; }
            Check(threw && !engine.TryGetModel<IPredictor<float[], float[]>>("echo", out _), "a kind answers only the contracts it implements");
            threw = false;
            try { InferenceEngine.Create().Add("again", exclusive).BuildAsync().GetAwaiter().GetResult(); } catch (InvalidOperationException) { threw = true; }
            Check(threw, "a model object belongs to one engine");
            threw = false;
            try { InferenceEngine.Create().Add("fixed", new EchoModel(new EngineHosting { Reloadable = false, Instances = 2 })); }
            catch (InvalidOperationException) { threw = true; }
            Check(threw, "copies of a model the engine cannot reload are refused");
        }
        finally
        {
            engine.DisposeAsync().AsTask().Wait();
        }

        Check(exclusive.Disposed && batched.Disposed && batched.Unloaded.Count == 1, "the engine disposes its models and unloads their copies");
    }
}
