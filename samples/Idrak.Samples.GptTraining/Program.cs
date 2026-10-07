// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Character-level GPT trainer: the Idrak counterpart of train_char_gpt.py (same options, data files, schedule and
// model), on Idrak's own CUDA kernels.
//
//   train (default)  --bin data/final_corpus.bin --vocab-file data/final_corpus.vocab --out checkpoints/x
//                    or --corpus text.txt [--skip-clean] (the text is encoded once to <out>/corpus.bin)
//                    --block 384 --batch 32 --dmodel 768 --heads 12 --layers 12 --dropout 0.1
//                    --lr 3e-4 --min-lr 3e-5 --warmup 1000 --steps 80000 --eval-every 500 --eval-steps 40
//                    --save-every 1000 --seed 42 --resume <out>/last.ikw --grad-checkpoint --optim8bit
//                    Idrak only: --fp32 (no bfloat16 tensor cores), --fp8 (FP8 tensor-core products, compute
//                    capability 8.9+: faster, coarser; compare the loss curve), --cpu, --gpu-memory GiB, --offload,
//                    --log-every N (a new progress line every N steps; default: one line redrawn in place),
//                    --profile (time every kernel of one step, after three warm-up steps, and exit; without a corpus
//                    it times random tokens), --loss-log file.csv (step, loss, lr and, at evaluations, val per step),
//                    --no-save (no checkpoints: comparison runs)
//   compare          a.csv b.csv [c.csv …] [--window 200]: loss logs side by side (the mean loss of each window of
//                    steps, each run's difference from the first) and their validation losses, e.g. --fp8 against
//                    bfloat16 from the same --seed and schedule
//   generate         --out <checkpoint dir> [--checkpoint best|last] --prompt "text" --tokens 500
//                    --temperature 0.7 --top-k 0 --top-p 0.9 --repeat-penalty 1.15
//
// Differences from the PyTorch script: GELU uses the tanh approximation, attention weights are not dropped (embedding
// and residual dropout are), batches are drawn with .NET's random generator, and checkpoints are Idrak files
// (<out>/last.ikw and best.ikw with a .json beside each). Text cleaning (clean_corpus.py) is not reproduced: pass the
// cleaned text the script wrote (--clean-out) with --skip-clean.

using System.Diagnostics;
using System.IO.MemoryMappedFiles;
using System.Text.Json.Nodes;
using System.Text.Json;
using System.Text;
using Idrak.Diagnostics;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models.Abstractions;
using Idrak.Optimizers;
using Idrak;

var options = new Dictionary<string, string>(StringComparer.Ordinal);
var flags = new HashSet<string>(StringComparer.Ordinal);
string command = "train";
if (args is ["compare", .. var logs])
{
    return CompareLogs(logs);
}

string[] flagNames = ["--skip-clean", "--grad-checkpoint", "--optim8bit", "--fp32", "--cpu", "--cuda", "--offload", "--profile", "--fp8", "--no-save"];
string[] valueNames = ["--corpus", "--clean-out", "--out", "--block", "--batch", "--dmodel", "--heads", "--layers", "--dropout", "--lr", "--min-lr",
    "--warmup", "--steps", "--eval-every", "--eval-steps", "--save-every", "--seed", "--resume", "--bin", "--vocab-file", "--gpu-memory",
    "--log-every", "--loss-log", "--checkpoint", "--prompt", "--tokens", "--temperature", "--top-k", "--top-p", "--repeat-penalty"];
for (int i = 0; i < args.Length; i++)
{
    if (i == 0 && args[i] is "train" or "generate")
    {
        command = args[i];
    }
    else if (flagNames.Contains(args[i]))
    {
        flags.Add(args[i]);
    }
    else if (valueNames.Contains(args[i]) && i + 1 < args.Length)
    {
        options[args[i]] = args[++i];
    }
    else
    {
        Console.Error.WriteLine($"unknown option '{args[i]}'. Options: {string.Join(' ', valueNames.Concat(flagNames))}");
        return 2;
    }
}

string Get(string name, string fallback) => options.TryGetValue(name, out var v) ? v : fallback;
int Int(string name, int fallback) => int.Parse(Get(name, fallback.ToString(System.Globalization.CultureInfo.InvariantCulture)), System.Globalization.CultureInfo.InvariantCulture);
float Float(string name, float fallback) => float.Parse(Get(name, fallback.ToString("R", System.Globalization.CultureInfo.InvariantCulture)), System.Globalization.CultureInfo.InvariantCulture);

if (options.TryGetValue("--gpu-memory", out var gib))
{
    ComputeResources.GpuMemoryLimit = (long)(double.Parse(gib, System.Globalization.CultureInfo.InvariantCulture) * (1L << 30));
}

if (flags.Contains("--offload"))
{
    ComputeResources.OffloadToHostMemory = true;
}

var device = flags.Contains("--cpu") || !Device.IsCudaAvailable ? Device.Cpu : Device.Cuda();
Console.WriteLine($"device={device}  {device.Name}");
return command == "generate" ? Generate() : Train();

int Train()
{
    string outDir = Get("--out", "checkpoints/idrak-outlets-768");
    int block = Int("--block", 384), batch = Int("--batch", 32), dim = Int("--dmodel", 768), heads = Int("--heads", 12), layers = Int("--layers", 12);
    int steps = Int("--steps", 80000), warmup = Int("--warmup", 1000), evalEvery = Int("--eval-every", 500), evalSteps = Int("--eval-steps", 40);
    int saveEvery = Int("--save-every", 1000), seed = Int("--seed", 42), logEvery = Int("--log-every", 0);
    float dropout = Float("--dropout", 0.1f), baseLr = Float("--lr", 3e-4f), minLr = Float("--min-lr", 3e-5f);
    bool checkpointing = flags.Contains("--grad-checkpoint"), eightBit = flags.Contains("--optim8bit");
    var precision = flags.Contains("--fp32") ? MatMulPrecision.Float32 : flags.Contains("--fp8") ? MatMulPrecision.Float8 : MatMulPrecision.BFloat16;
    Directory.CreateDirectory(outDir);

    // ---------------------------------------------------------------- data
    bool profiling = flags.Contains("--profile"), synthetic = false;
    string binPath;
    List<string> itos;
    if (options.TryGetValue("--bin", out var bin))
    {
        binPath = bin;
        itos = LoadVocab(options.TryGetValue("--vocab-file", out var vf) ? vf : throw new ArgumentException("--vocab-file is required when --bin is set"));
    }
    else
    {
        string corpus = Get("--corpus", "data/outlets.txt");
        string encoded = Path.Combine(outDir, "corpus.bin"), encodedVocab = Path.Combine(outDir, "vocab.txt");
        if (File.Exists(corpus))
        {
            if (!flags.Contains("--skip-clean"))
            {
                Console.WriteLine($"note: clean_corpus.py's cleaning is not reproduced; training on {corpus} as it is "
                                  + "(pass the cleaned file the Python script wrote, with --skip-clean, to match it).");
            }

            binPath = encoded;
            itos = EncodeCorpus(corpus, binPath);
        }
        else if (File.Exists(encoded) && File.Exists(encodedVocab))
        {
            Console.WriteLine($"note: {corpus} not found; using {encoded} and {encodedVocab}, encoded by an earlier run.");
            binPath = encoded;
            itos = LoadVocab(encodedVocab);
        }
        else if (profiling)
        {
            // Timing needs no real text: random ids over a character-sized vocabulary.
            const int SyntheticVocabulary = 256;
            Console.WriteLine($"note: {corpus} not found; profiling on random tokens (vocabulary {SyntheticVocabulary}).");
            binPath = Path.Combine(outDir, "profile-tokens.bin");
            var ids = new byte[2 * (4 << 20)];
            var fill = new Random(seed);
            for (int i = 0; i < ids.Length; i += 2)
            {
                ids[i] = (byte)fill.Next(SyntheticVocabulary);
            }

            File.WriteAllBytes(binPath, ids);
            itos = [.. Enumerable.Range(0, SyntheticVocabulary).Select(i => char.ConvertFromUtf32(0x100 + i))];
            synthetic = true;
        }
        else
        {
            throw new FileNotFoundException($"{corpus} not found: pass --corpus <text file> (or --bin <ids> with --vocab-file <vocab>).", corpus);
        }
    }

    using var tokens = new TokenFile(binPath);
    long split = (long)(tokens.Length * 0.98);
    var train = (Start: 0L, Length: split);
    var val = (Start: split, Length: tokens.Length - split);
    Console.WriteLine($"memmap  bin={binPath}  vocab={itos.Count}  train={train.Length:N0}  val={val.Length:N0}");
    if (!synthetic)
    {
        SaveVocab(Path.Combine(outDir, "vocab.txt"), itos);
    }

    // ---------------------------------------------------------------- model
    var spec = GptSpec(itos.Count, dim, heads, layers, block, dropout);
    using var model = spec.Build(options: new DecoderBuildOptions { Device = device, Seed = seed, InitStd = 0.02f });
    int startStep = 0;
    double best = double.PositiveInfinity;
    if (options.TryGetValue("--resume", out var resume) && File.Exists(resume))
    {
        model.Load(resume);
        var info = ReadInfo(resume);
        startStep = (int?)info?["step"] ?? 0;
        best = (double?)info?["val"] ?? best;
        Console.WriteLine($"resumed {resume}  step={startStep}  val={best}");
    }

    long parameters = model.Parameters().Sum(p => (long)p.Size);
    int tokensPerStep = batch * block;
    Console.WriteLine($"params={parameters:N0}  block={block}  batch={batch}  steps={steps}  tok/step={tokensPerStep:N0}  "
                      + $"seen≈{(double)steps * tokensPerStep / 1e6:F1}M  grad_checkpoint={checkpointing}  optim8bit={eightBit}  "
                      + $"matmul={(precision == MatMulPrecision.Float32 ? "float32" : MixedPrecision.TensorCoresUnavailable(device) is { } why ? $"float32 ({why})" : precision == MatMulPrecision.Float8 ? "fp8 tensor cores" : "bfloat16 tensor cores")}");

    var trainable = model.Parameters().ToList();
    using Optimizer optimizer = eightBit
        ? new AdamW8Bit(trainable, baseLr, 0.9f, 0.95f, 1e-8f, weightDecay: 0.1f)
        : new AdamW(trainable, baseLr, 0.9f, 0.95f, 1e-8f, weightDecay: 0.1f);
    if (eightBit)
    {
        Console.WriteLine("using Idrak AdamW8Bit");
    }

    var head = (Linear)model[^1];
    var body = model.Take(model.Count - 1).ToList();
    using var ones = Tensor.Ones([tokensPerStep], device);
    var random = new Random(seed + startStep);
    var xs = new float[tokensPerStep];
    var ys = new float[tokensPerStep];

    // One batch through the model: the mean token cross-entropy (the output head runs in chunks with a fused kernel,
    // so the [tokens, vocabulary] logits never exist at once).
    Tensor Loss(bool training)
    {
        var x = Tensor.From(xs, [batch, block], device);
        var targets = Tensor.From(ys, [tokensPerStep], device);
        var h = x;
        foreach (var module in body)
        {
            h = training && checkpointing && module is DecoderBlock b ? b.ForwardCheckpointed(h) : module.Forward(h);
        }

        return Losses.TokenCrossEntropy(h.Reshape(tokensPerStep, dim), head.Forward, targets, ones, tokensPerStep, chunkRows: 2048);
    }

    void Sample((long Start, long Length) range)
    {
        for (int b = 0; b < batch; b++)
        {
            long offset = range.Start + random.NextInt64(range.Length - block - 1);
            tokens.Read(offset, xs.AsSpan(b * block, block), ys.AsSpan(b * block, block));
        }
    }

    double Evaluate()
    {
        model.Eval();
        double sum = 0;
        var range = val.Length > block + 1 ? val : train;
        using (Autograd.NoGrad())
        using (MixedPrecision.Use(precision))
        {
            for (int s = 0; s < evalSteps; s++)
            {
                using var scope = new TensorScope();
                Sample(range);
                sum += Loss(training: false).Item();
            }
        }

        model.Train();
        return sum / evalSteps;
    }

    void Save(string name, int step, double value)
    {
        if (flags.Contains("--no-save"))
        {
            return;
        }

        string path = Path.Combine(outDir, name + ".ikw");
        model.Save(path);
        var info = new JsonObject
        {
            ["step"] = step, ["val"] = double.IsFinite(value) ? value : null, ["vocab_size"] = itos.Count, ["spec"] = spec.ToJson(),
            ["config"] = new JsonObject(options.Select(o => KeyValuePair.Create(o.Key.TrimStart('-'), (JsonNode?)JsonValue.Create(o.Value)))
                .Concat(flags.Select(f => KeyValuePair.Create(f.TrimStart('-'), (JsonNode?)JsonValue.Create(true))))),
        };
        File.WriteAllText(Path.ChangeExtension(path, ".json"), info.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
    }

    int stopRequests = 0;
    Console.CancelKeyPress += (_, e) =>
    {
        if (Interlocked.Increment(ref stopRequests) == 1)
        {
            e.Cancel = true;
            Console.WriteLine("\nstopping after this step and saving last.ikw (Ctrl+C again to quit now)…");
        }
    };

    model.Train();
    using var lossLog = options.TryGetValue("--loss-log", out var lossLogPath) ? new StreamWriter(lossLogPath, append: startStep > 0) : null;
    if (lossLog is not null && startStep == 0)
    {
        lossLog.WriteLine("step,loss,lr,val");
    }

    var progress = new ProgressLine(startStep, steps, logEvery);
    double lossSum = 0, lastVal = double.IsFinite(best) ? best : double.NaN;
    var watch = Stopwatch.StartNew();
    var stepWatch = new Stopwatch();
    for (int step = startStep + 1; step <= steps; step++)
    {
        if (profiling && step - startStep == 4)
        {
            device.Synchronize();
            progress.Write($"last step {stepWatch.Elapsed.TotalMilliseconds:F0} ms wall; profiling the next one (every kernel waited for)…");
            GpuProfiler.Start(device);
        }

        device.Synchronize();
        stepWatch.Restart();
        float lr = CosineLr(step, steps, baseLr, minLr, warmup);
        optimizer.LearningRate = lr;
        Sample(train);
        float loss;
        using (MixedPrecision.Use(precision))
        using (var scope = new TensorScope())
        {
            optimizer.ZeroGrad();
            var l = Loss(training: true);
            l.Backward();
            optimizer.ClipGradientNorm(1f);
            optimizer.Step();
            loss = l.Item();
        }

        lossSum += loss;
        int done = step - startStep;
        if (profiling && done == 4)
        {
            var entries = GpuProfiler.Stop(device);
            progress.Write(GpuProfiler.Format(entries, rows: 60));
            return 0;
        }

        double tokPerSecond = done * (double)tokensPerStep / Math.Max(watch.Elapsed.TotalSeconds, 1e-6);
        progress.Update(step, $"loss={loss:F4}, avg={lossSum / done:F4}, lr={lr:0.00e+00}, tok/s={tokPerSecond:N0}"
            + (double.IsNaN(lastVal) ? "" : $", val={lastVal:F4}") + $", {ComputeResources.GetMemoryUsage(device)}");

        bool stopping = Volatile.Read(ref stopRequests) > 0;
        double? evaluated = null;
        if (step % evalEvery == 0 || step == steps)
        {
            double v = Evaluate();
            evaluated = v;
            lastVal = v;
            Save("last", step, v);
            string tag = "";
            if (v < best)
            {
                best = v;
                Save("best", step, v);
                tag = "  best";
            }

            progress.Write($"  val {v:F4}  ppl {Math.Exp(Math.Min(v, 20)):F2}  lr {lr:0.00e+00}{tag}");
        }
        else if (step % saveEvery == 0 || stopping)
        {
            Save("last", step, lastVal);
        }

        if (lossLog is not null)
        {
            var invariant = System.Globalization.CultureInfo.InvariantCulture;
            lossLog.WriteLine(string.Create(invariant, $"{step},{loss:R},{lr:R},{(evaluated is { } e ? e.ToString("R", invariant) : "")}"));
            if (step % 50 == 0 || evaluated is not null)
            {
                lossLog.Flush();
            }
        }

        if (stopping)
        {
            progress.Write($"stopped at step {step}; saved {Path.Combine(outDir, "last.ikw")} (resume with --resume)");
            return 0;
        }
    }

    progress.Write($"done. best val={best:F4}  dir={outDir}");
    return 0;
}

int Generate()
{
    string outDir = Get("--out", "checkpoints/idrak-outlets-768");
    string path = Path.Combine(outDir, Get("--checkpoint", "best") + ".ikw");
    var info = ReadInfo(path) ?? throw new FileNotFoundException($"{Path.ChangeExtension(path, ".json")} not found");
    var spec = DecoderSpec.FromJson(info["spec"]!.AsObject()) with { Dropout = 0f };
    var itos = LoadVocab(Path.Combine(outDir, "vocab.txt"));
    using var model = spec.Build(options: new DecoderBuildOptions { Device = device });
    model.Load(path);
    var generator = new TextGenerator(model, new RuneTokenizer(itos), spec.MaxPositions);
    var generation = new GenerationOptions
    {
        NumPredict = Int("--tokens", 500), Temperature = Float("--temperature", 0.7f), TopK = Int("--top-k", 0), TopP = Float("--top-p", 0.9f),
        RepeatPenalty = Float("--repeat-penalty", 1.15f), NumCtx = spec.MaxPositions,
    };
    string prompt = Get("--prompt", "\n");
    Console.Write(prompt);
    GenerationStats? stats = null;
    foreach (var chunk in generator.Stream(prompt, generation))
    {
        Console.Write(chunk.Text);
        stats = chunk.Stats ?? stats;
    }

    Console.WriteLine($"\n[{stats?.GeneratedTokens} tokens, {stats?.TokensPerSecond:F1} tok/s]");
    return 0;
}

static DecoderSpec GptSpec(int vocabulary, int dim, int heads, int layers, int block, float dropout) => new()
{
    Vocabulary = vocabulary, Dim = dim, Layers = layers, Heads = heads, KvHeads = heads, HeadDim = dim / heads, FfDim = 4 * dim,
    MaxPositions = block, Norm = DecoderNorm.Layer, NormEpsilon = 1e-5f, Gated = false, Activation = FeedForwardActivation.Gelu,
    Rope = null, LearnedPositions = true, FeedForwardBias = true, TieEmbeddings = true, Dropout = dropout,
};

static float CosineLr(int step, int total, float baseLr, float minLr, int warmup)
{
    if (step < warmup)
    {
        return baseLr * step / Math.Max(1, warmup);
    }

    double t = Math.Clamp((step - warmup) / (double)Math.Max(1, total - warmup), 0, 1);
    return (float)(minLr + 0.5 * (baseLr - minLr) * (1 + Math.Cos(Math.PI * t)));
}

static JsonObject? ReadInfo(string checkpoint)
{
    string json = Path.ChangeExtension(checkpoint, ".json");
    return File.Exists(json) ? JsonNode.Parse(File.ReadAllText(json))?.AsObject() : null;
}

// The vocabulary file of train_char_gpt.py: one symbol per line, with \n, \r, \t and \\ escaped. Lines are split as
// Python's str.splitlines() splits them, so a file the script wrote reads back identically.
static List<string> LoadVocab(string path)
{
    string text = File.ReadAllText(path, Encoding.UTF8);
    var itos = new List<string>();
    int start = 0;
    for (int i = 0; i < text.Length; i++)
    {
        char c = text[i];
        if (c is '\n' or '\r' or '\v' or '\f' or '\x1c' or '\x1d' or '\x1e' or '\x85' or '\u2028' or '\u2029')
        {
            itos.Add(Unescape(text[start..i]));
            if (c == '\r' && i + 1 < text.Length && text[i + 1] == '\n')
            {
                i++;
            }

            start = i + 1;
        }
    }

    if (start < text.Length)
    {
        itos.Add(Unescape(text[start..]));
    }

    return itos;

    static string Unescape(string raw) => raw switch { "\\n" => "\n", "\\r" => "\r", "\\t" => "\t", "\\\\" => "\\", _ => raw };
}

// Loss logs (--loss-log) side by side: the mean training loss of each window of steps in every run, each run's relative
// difference from the first, then the validation losses at the steps they share and a summary over the last fifth.
static int CompareLogs(string[] arguments)
{
    int window = 200;
    var files = new List<string>();
    for (int i = 0; i < arguments.Length; i++)
    {
        if (arguments[i] == "--window" && i + 1 < arguments.Length)
        {
            window = int.Parse(arguments[++i], System.Globalization.CultureInfo.InvariantCulture);
        }
        else
        {
            files.Add(arguments[i]);
        }
    }

    if (files.Count < 2)
    {
        Console.Error.WriteLine("compare needs two or more loss logs (--loss-log files).");
        return 2;
    }

    var runs = files.Select(f =>
    {
        var losses = new SortedDictionary<int, double>();
        var vals = new SortedDictionary<int, double>();
        foreach (var line in File.ReadLines(f).Skip(1))
        {
            var parts = line.Split(',');
            if (parts.Length < 4 || !int.TryParse(parts[0], System.Globalization.CultureInfo.InvariantCulture, out int step))
            {
                continue;
            }

            losses[step] = double.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture);
            if (parts[3].Length > 0)
            {
                vals[step] = double.Parse(parts[3], System.Globalization.CultureInfo.InvariantCulture);
            }
        }

        return (Name: Path.GetFileNameWithoutExtension(f), Losses: losses, Vals: vals);
    }).ToList();

    int last = runs.Min(r => r.Losses.Count == 0 ? 0 : r.Losses.Keys.Max());
    string Header(string first) => $"{first,-16}" + string.Concat(runs.Select((r, i) => $"{r.Name,16}" + (i == 0 ? "" : $"{"diff",9}")));
    Console.WriteLine($"training loss, mean of each {window} steps (diff: relative to {runs[0].Name}; steps 1..{last}, shared by all runs)");
    Console.WriteLine(Header("steps"));
    var tail = new double[runs.Count];
    int tailWindows = 0;
    for (int start = 1; start <= last; start += window)
    {
        int end = Math.Min(start + window - 1, last);
        var means = runs.Select(r => r.Losses.Where(p => p.Key >= start && p.Key <= end).Select(p => p.Value).DefaultIfEmpty(double.NaN).Average()).ToList();
        Console.WriteLine($"{$"{start}-{end}",-16}" + string.Concat(means.Select((m, i) => $"{m,16:F4}" + (i == 0 ? "" : $"{(m - means[0]) / means[0],9:P2}"))));
        if (end > last * 4 / 5)
        {
            for (int i = 0; i < runs.Count; i++)
            {
                tail[i] += means[i];
            }

            tailWindows++;
        }
    }

    var shared = runs[0].Vals.Keys.Where(step => runs.All(r => r.Vals.ContainsKey(step))).ToList();
    if (shared.Count > 0)
    {
        Console.WriteLine();
        Console.WriteLine("validation loss");
        Console.WriteLine(Header("step"));
        foreach (int step in shared)
        {
            double first = runs[0].Vals[step];
            Console.WriteLine($"{step,-16}" + string.Concat(runs.Select((r, i) => $"{r.Vals[step],16:F4}" + (i == 0 ? "" : $"{(r.Vals[step] - first) / first,9:P2}"))));
        }
    }

    if (tailWindows > 0)
    {
        Console.WriteLine();
        Console.WriteLine($"last fifth of the steps: mean training loss " + string.Join(", ", runs.Select((r, i) =>
            $"{r.Name} {tail[i] / tailWindows:F4}" + (i == 0 ? "" : $" ({(tail[i] - tail[0]) / tail[0]:+0.00%;-0.00%})"))));
    }

    return 0;
}

static void SaveVocab(string path, List<string> itos)
{
    var lines = itos.Select(ch => ch switch { "\n" => "\\n", "\r" => "\\r", "\t" => "\\t", "\\" => "\\\\", _ => ch });
    File.WriteAllText(path, string.Join("\n", lines) + "\n", new UTF8Encoding(false));
}

// Encodes a text file as uint16 ids (one per Unicode code point, as Python's str iterates), streaming, so corpora of
// any size work: the vocabulary is the sorted set of code points (the first pass), unknown symbols map to a space.
static List<string> EncodeCorpus(string corpus, string binPath)
{
    var watch = Stopwatch.StartNew();
    var seen = new HashSet<int>();
    ForEachCodePoint(corpus, (span) =>
    {
        foreach (int cp in span)
        {
            seen.Add(cp);
        }
    });
    var sorted = seen.Order().ToList();
    if (sorted.Count > ushort.MaxValue)
    {
        throw new InvalidDataException($"{sorted.Count} distinct symbols do not fit 16-bit ids.");
    }

    var stoi = new Dictionary<int, ushort>();
    for (int i = 0; i < sorted.Count; i++)
    {
        stoi[sorted[i]] = (ushort)i;
    }

    ushort unknown = stoi.TryGetValue(' ', out var space) ? space : (ushort)0;
    long count = 0;
    using (var output = new BufferedStream(File.Create(binPath), 1 << 20))
    {
        var buffer = new byte[1 << 16];
        ForEachCodePoint(corpus, span =>
        {
            int n = 0;
            foreach (int cp in span)
            {
                ushort id = stoi.TryGetValue(cp, out var v) ? v : unknown;
                buffer[n++] = (byte)id;
                buffer[n++] = (byte)(id >> 8);
                if (n == buffer.Length)
                {
                    output.Write(buffer, 0, n);
                    n = 0;
                }
            }

            output.Write(buffer, 0, n);
            count += span.Length;
        });
    }

    Console.WriteLine($"encoded {corpus}: {count:N0} symbols, vocabulary {sorted.Count}, in {watch.Elapsed.TotalSeconds:F0} s → {binPath}");
    return [.. sorted.Select(char.ConvertFromUtf32)];
}

// Calls `chunk` with the file's code points, a block at a time (surrogate pairs joined across blocks).
static void ForEachCodePoint(string path, Action<ReadOnlySpan<int>> chunk)
{
    using var reader = new StreamReader(path, Encoding.UTF8, detectEncodingFromByteOrderMarks: true, bufferSize: 1 << 20);
    var chars = new char[1 << 20];
    var points = new int[chars.Length + 1];
    char pending = '\0';
    int read;
    while ((read = reader.Read(chars, 0, chars.Length)) > 0)
    {
        int n = 0, i = 0;
        if (pending != '\0')
        {
            points[n++] = char.IsLowSurrogate(chars[0]) ? char.ConvertToUtf32(pending, chars[i++]) : pending;
            pending = '\0';
        }

        for (; i < read; i++)
        {
            char c = chars[i];
            if (char.IsHighSurrogate(c))
            {
                if (i + 1 == read)
                {
                    pending = c;
                    break;
                }

                if (char.IsLowSurrogate(chars[i + 1]))
                {
                    points[n++] = char.ConvertToUtf32(c, chars[++i]);
                    continue;
                }
            }

            points[n++] = c;
        }

        chunk(points.AsSpan(0, n));
    }

    if (pending != '\0')
    {
        chunk([pending]);
    }
}

// uint16 token ids read from a memory-mapped file: only the windows a batch touches are paged in.
internal sealed unsafe class TokenFile : IDisposable
{
    private readonly MemoryMappedFile _file;
    private readonly MemoryMappedViewAccessor _view;
    private readonly byte* _base;

    public TokenFile(string path)
    {
        long bytes = new FileInfo(path).Length;
        Length = bytes / 2;
        _file = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        _view = _file.CreateViewAccessor(0, bytes, MemoryMappedFileAccess.Read);
        byte* pointer = null;
        _view.SafeMemoryMappedViewHandle.AcquirePointer(ref pointer);
        _base = pointer + _view.PointerOffset;
    }

    public long Length { get; }

    /// <summary>x = ids[offset .. offset + n), y = ids[offset + 1 .. offset + n + 1).</summary>
    public void Read(long offset, Span<float> x, Span<float> y)
    {
        var ids = new ReadOnlySpan<ushort>(_base + offset * 2, x.Length + 1);
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = ids[i];
            y[i] = ids[i + 1];
        }
    }

    public void Dispose()
    {
        _view.SafeMemoryMappedViewHandle.ReleasePointer();
        _view.Dispose();
        _file.Dispose();
    }
}

// Characters (Unicode code points) ↔ ids for generation.
internal sealed class RuneTokenizer(List<string> itos) : ITokenizer
{
    private readonly Dictionary<string, int> _stoi = itos.Select((s, i) => (s, i)).GroupBy(p => p.s).ToDictionary(g => g.Key, g => g.First().i);

    public int VocabularySize => itos.Count;

    public IReadOnlyList<int> Encode(string text)
    {
        int unknown = _stoi.TryGetValue(" ", out int space) ? space : 0;
        var ids = new List<int>(text.Length);
        foreach (var rune in text.EnumerateRunes())
        {
            ids.Add(_stoi.TryGetValue(rune.ToString(), out int id) ? id : unknown);
        }

        return ids;
    }

    public string Decode(IEnumerable<int> ids)
    {
        var sb = new StringBuilder();
        foreach (int id in ids)
        {
            if ((uint)id < (uint)itos.Count)
            {
                sb.Append(itos[id]);
            }
        }

        return sb.ToString();
    }
}

// A tqdm-like progress line: redrawn in place on a console, or a new line every `every` steps (always when redirected).
internal sealed class ProgressLine(int start, int total, int every)
{
    private readonly Stopwatch _watch = Stopwatch.StartNew();
    private readonly bool _inPlace = every <= 0 && !Console.IsOutputRedirected;
    private readonly int _every = every > 0 ? every : 10;
    private double _lastDraw = double.NegativeInfinity;             // milliseconds
    private int _width;

    public void Update(int step, string stats)
    {
        if (!_inPlace && step % _every != 0 && step != total)
        {
            return;
        }

        if (_inPlace && _watch.Elapsed.TotalMilliseconds - _lastDraw < 200 && step != total)
        {
            return;
        }

        _lastDraw = _watch.Elapsed.TotalMilliseconds;
        int done = step - start;
        double rate = done / Math.Max(_watch.Elapsed.TotalSeconds, 1e-6);
        var remaining = TimeSpan.FromSeconds(rate > 0 ? (total - step) / rate : 0);
        double fraction = total > 0 ? step / (double)total : 1;
        const int BarWidth = 30;
        int filled = (int)(fraction * BarWidth);
        string line = $"{fraction * 100,5:F1}%|{new string('█', filled)}{new string(' ', BarWidth - filled)}| {step}/{total} "
                      + $"[{Format(_watch.Elapsed)}<{Format(remaining)}, {rate:F2}step/s, {stats}]";
        if (_inPlace)
        {
            Console.Write("\r" + line.PadRight(_width));
            _width = line.Length;
        }
        else
        {
            Console.WriteLine(line);
        }
    }

    public void Write(string message)
    {
        if (_inPlace && _width > 0)
        {
            Console.Write("\r" + new string(' ', _width) + "\r");
            _width = 0;
        }

        Console.WriteLine(message);
    }

    private static string Format(TimeSpan t) => t.TotalHours >= 1 ? $"{(int)t.TotalHours}:{t.Minutes:00}:{t.Seconds:00}" : $"{t.Minutes:00}:{t.Seconds:00}";
}
