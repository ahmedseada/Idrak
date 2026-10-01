// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Datasets;
using Idrak.Diagnostics;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.LanguageModels;

const string Usage = """
    idrak-tune: fine-tune pretrained language models (LoRA / QLoRA) on any dataset, evaluate, chat and export.

      idrak-tune train <model> <data…> --out <dir>    tune adapters on the data and write them (PEFT format) with
                                                      idrak-tuning.json (base model, system prompt, max length)
      idrak-tune evaluate <model> <data…>             loss and answer scores on held-out conversations (greedy answers against
                                                      the references); with --adapter DIR, the base model and the adapter side by side
      idrak-tune chat <model> [message…]              chat, streamed (the messages given, else one per line typed)
      idrak-tune export <model> --out <dir>           merge the adapter into the weights and write a Hugging Face checkpoint
      idrak-tune download <model id…>                 download models and print their folders
      idrak-tune info <model>                         what the loader made of a model

    <model>: any chat or base model the library loads (its own tokenizer and chat template): a Hugging Face id (owner/name,
    found in a cache or downloaded once; HF_TOKEN for gated models), a model folder, a .gguf file, ollama:name, or an
    adapter folder written by train (its base model and the adapter).

    <data…>: files, folders, hf:, github:, kaggle:, zenodo: or URLs (JSON Lines, JSON, CSV, Parquet, text), or a recipe
    (.json with "sources"), as idrak-data reads them. Conversations train the assistant's turns; text rows every token. Map
    columns into a conversation with user / assistant / system templates after "?" (joined with "&"):
      idrak-tune train owner/model "data.csv?user={question}&assistant={answer}" --out adapters/qa

    Data:      --eval F|spec (evaluation data), --eval-fraction F (hold out a fraction; evaluate with the same data and
               fraction scores that part), --system S (added to conversations without one), --kind auto|chat|text,
               --max-rows N, --seed N, --min-chars N, --max-chars N, --no-dedup, --mix, --no-shuffle, --balance N (answers from a
               short list, such as labels: rarer answers repeated, up to N times, towards the most common one's count)
    Training:  --rank 16, --alpha 32, --lr 2e-4, --epochs 1, --max-length 2048, --batch-tokens 4096, --accumulate 1,
               --targets q,k,v,o,gate,up,down, --save-every N, --eval-every N, --checkpointing | --no-checkpointing
               (default: off, turned on if a step runs out of device memory even with --recompute), --recompute (recompute the
               feed-forward activations in the backward pass: a fifth less activation memory for one kernel per block;
               default: turned on first when a step runs out of device memory), --bf16-activations (hold the activations
               as bfloat16 between the passes: half their memory; default: turned on second), --no-packing,
               --no-graphs, --fp8 (the frozen base's forward products in FP8, checked against bfloat16 first),
               --adapter DIR (continue training an adapter), --profile (time a few steps instead of training),
               --cpu-optimizer (AdamW's update on the CPU, its state in system memory)
    Evaluate:  --adapter DIR, --samples 100 (0: all), --batch 8, --max-new 512, --metric auto|number|exact|contains|f1, --out F.jsonl,
               --choices a,b,c | auto (the answer is one of these: each is scored as the model's answer, the most likely one
               taken, nothing generated; auto: the distinct answers in the data; accuracy and recall per answer)
    Chat:      --system S, --max-new N, --temperature T (0: greedy)
    Model:     --cuda | --cpu | --device cuda:N (default: the first GPU when there is one; any CUDA GPU works, the faster
               kernels load where the GPU has them), --int8 | --int4 | --bf16 (base weights), --context N,
               --kv8 | --kv16, --no-think, --matmul fp32|bf16|fp8, --offload, --gpu-memory GiB
    """;

Console.OutputEncoding = System.Text.Encoding.UTF8;                       // "…" and emoji on Windows consoles
var positional = new List<string>();
bool int8 = false, bf16 = false, int4 = false, kv8 = false, kv16 = false, noThink = false, profileTraining = false;
bool shuffleRows = true, dedupRows = true, mixByWeight = false;
int context = 4096, samples = 100, maxNew = 512, evaluationBatch = 8, seed = 0, minChars = 0, maxChars = 0;
long maxRows = 0;
int balance = 0;
double evalFraction = 0;
string? output = null, evalFile = null, adapterFolder = null, systemPrompt = null, choices = null;
float? temperature = null;
var metric = AnswerMetric.Auto;
var rowKind = RowKind.Auto;
MatMulPrecision? matmul = null;
var tuning = new FineTuningOptions();
Device device = Device.IsCudaAvailable ? Device.Cuda() : Device.Cpu;
try
{
    for (int i = 0; i < args.Length; i++)
    {
        string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value.");
        int NextInt() => int.Parse(Next(), CultureInfo.InvariantCulture);
        float NextFloat() => float.Parse(Next(), CultureInfo.InvariantCulture);
        switch (args[i])
        {
            case "--cuda" or "--gpu": device = Device.Cuda(); break;
            case "--device":
                device = Next() switch
                {
                    "cpu" => Device.Cpu,
                    "cuda" or "gpu" => Device.Cuda(),
                    ['c', 'u', 'd', 'a', ':', .. var ordinal] => Device.Cuda(int.Parse(ordinal, CultureInfo.InvariantCulture)),
                    var other => throw new ArgumentException($"--device {other}: use cpu, cuda or cuda:N."),
                };
                break;
            case "--cpu": device = Device.Cpu; break;
            case "--int8": int8 = true; break;
            case "--bf16": bf16 = true; break;
            case "--int4": int4 = true; break;
            case "--kv8": kv8 = true; break;
            case "--kv16": kv16 = true; break;
            case "--no-think": noThink = true; break;
            case "--context": context = NextInt(); break;
            case "--out" or "-o": output = Next(); break;
            case "--eval": evalFile = Next(); break;
            case "--adapter": adapterFolder = Next(); break;
            case "--rank": tuning = tuning with { Rank = NextInt() }; break;
            case "--alpha": tuning = tuning with { Alpha = NextFloat() }; break;
            case "--lr": tuning = tuning with { LearningRate = NextFloat() }; break;
            case "--epochs": tuning = tuning with { Epochs = NextInt() }; break;
            case "--max-length": tuning = tuning with { MaxLength = NextInt() }; break;
            case "--batch-tokens": tuning = tuning with { BatchTokens = NextInt() }; break;
            case "--accumulate": tuning = tuning with { GradientAccumulation = NextInt() }; break;
            case "--save-every": tuning = tuning with { SaveEvery = NextInt() }; break;
            case "--eval-every": tuning = tuning with { EvaluateEvery = NextInt() }; break;
            case "--targets": tuning = tuning with { Targets = Next().Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries) }; break;
            case "--no-checkpointing": tuning = tuning with { Checkpointing = false }; break;
            case "--checkpointing": tuning = tuning with { Checkpointing = true }; break;
            case "--recompute": tuning = tuning with { RecomputeFeedForward = true }; break;
            case "--bf16-activations": tuning = tuning with { BFloat16Activations = true }; break;
            case "--no-packing": tuning = tuning with { Packing = false }; break;
            case "--no-graphs": tuning = tuning with { CudaGraphs = false }; break;
            case "--fp8": tuning = tuning with { Float8 = true }; break;
            case "--profile": profileTraining = true; break;
            case "--offload": ComputeResources.OffloadToHostMemory = true; break;
            case "--cpu-optimizer": tuning = tuning with { HostOptimizer = true }; break;
            case "--gpu-memory": ComputeResources.GpuMemoryLimit = (long)(double.Parse(Next(), CultureInfo.InvariantCulture) * (1L << 30)); break;
            case "--matmul":
                matmul = Next() switch
                {
                    "fp32" or "float32" => MatMulPrecision.Float32,
                    "bf16" or "bfloat16" => MatMulPrecision.BFloat16,
                    "fp8" or "float8" => MatMulPrecision.Float8,
                    var other => throw new ArgumentException($"--matmul {other}: use fp32, bf16 or fp8."),
                };
                break;
            case "--samples": samples = NextInt(); break;
            case "--batch": evaluationBatch = NextInt(); break;
            case "--max-new": maxNew = NextInt(); break;
            case "--metric": metric = Enum.Parse<AnswerMetric>(Next(), ignoreCase: true); break;
            case "--choices": choices = Next(); break;
            case "--temperature": temperature = NextFloat(); break;
            case "--eval-fraction": evalFraction = double.Parse(Next(), CultureInfo.InvariantCulture); break;
            case "--system": systemPrompt = Next(); break;
            case "--kind": rowKind = Enum.Parse<RowKind>(Next(), ignoreCase: true); break;
            case "--max-rows": maxRows = long.Parse(Next(), CultureInfo.InvariantCulture); break;
            case "--seed": seed = NextInt(); tuning = tuning with { Seed = seed }; break;
            case "--min-chars": minChars = NextInt(); break;
            case "--max-chars": maxChars = NextInt(); break;
            case "--no-shuffle": shuffleRows = false; break;
            case "--no-dedup": dedupRows = false; break;
            case "--balance": balance = NextInt(); break;
            case "--mix": mixByWeight = true; break;
            case "-h" or "--help":
                Console.WriteLine(Usage);
                return 0;
            case ['-', '-', ..]:
                throw new ArgumentException($"Unknown option {args[i]}.");
            default: positional.Add(args[i]); break;
        }
    }
}
catch (Exception ex) when (ex is ArgumentException or FormatException)
{
    Console.Error.WriteLine($"error: {ex.Message}");
    return 1;
}

string command = positional.Count > 0 ? positional[0] : "";
if (command is not ("train" or "evaluate" or "chat" or "export" or "download" or "info") || positional.Count < 2
    || command is "train" or "evaluate" && positional.Count < 3 || command is "train" or "export" && output is null)
{
    Console.WriteLine(Usage);
    return 1;
}

// Tensor cores (bfloat16) for the larger products unless asked otherwise.
MixedPrecision.Default = matmul ?? MatMulPrecision.BFloat16;
Device.Default = device;
var cacheFormat = kv8 ? KeyValueFormat.Int8 : kv16 ? KeyValueFormat.BFloat16 : KeyValueFormat.Float32;
var status = new ConsoleStatus();
var downloads = status.CreateDownloader();
var readable = new JsonSerializerOptions { Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping };

// <model> may be an adapter folder from train: its base model, with the adapter (and its system prompt as the default).
string baseModel = positional[1];
TuningManifest? manifest = command != "download" && Directory.Exists(baseModel) ? TuningManifest.Read(baseModel) : null;
if (manifest is not null)
{
    adapterFolder ??= baseModel;
    baseModel = manifest.BaseModel;
    systemPrompt ??= manifest.System;
    Console.WriteLine($"adapter {adapterFolder}: tuned from {baseModel}{(manifest.System is null ? "" : $", system prompt \"{manifest.System}\"")}");
}

try
{
    switch (command)
    {
        case "train":
            return Train();
        case "evaluate":
            return Evaluate();
        case "chat":
            return Chat();
        case "export":
        {
            if (int8 || int4 || bf16)
            {
                Console.Error.WriteLine("export merges into float weights: leave out --int8, --int4 and --bf16.");
                return 1;
            }

            if (adapterFolder is null)
            {
                Console.Error.WriteLine("export needs an adapter: an adapter folder as <model>, or --adapter DIR.");
                return 1;
            }

            using var model = Load(merge: false);
            model.SaveHuggingFace(output!);
            Console.WriteLine($"merged model written to {output} (bfloat16 safetensors, config and tokenizer files)");
            return 0;
        }

        case "download":
            foreach (var id in positional.Skip(1))
            {
                string folder = ModelSource.Resolve(id, downloader: downloads);
                Console.WriteLine($"{id}: {folder}");
                foreach (var file in Directory.GetFiles(folder).Order(StringComparer.Ordinal))
                {
                    Console.WriteLine($"  {Downloader.Size(new FileInfo(file).Length),10}  {Path.GetFileName(file)}");
                }
            }

            return 0;

        default:
        {
            using var model = Load(merge: true);
            Console.WriteLine(model.Spec.ToJson());
            Console.WriteLine(model.ChatTemplate is { } t ? $"chat template: {t.Source.Length} characters, stops [{string.Join(", ", t.StopSequences)}]" : "no chat template");
            return 0;
        }
    }
}
catch (Exception ex) when (ex is IOException or InvalidDataException or InvalidOperationException or ArgumentException or NotSupportedException
                               or ResourceLimitExceededException or HttpRequestException)
{
    status.Clear();
    Console.Error.WriteLine($"error: {ex.Message}");
    return 1;
}

int Train()
{
    // Training keeps the adapter separate (to continue training it); a new adapter is added otherwise.
    using var model = Load(merge: false);
    if (tuning.MaxLength >= model.MaxPositions)
    {
        Console.WriteLine($"--max-length {tuning.MaxLength} is beyond the model's context of {model.MaxPositions} positions; using {model.MaxPositions - 1}");
        tuning = tuning with { MaxLength = model.MaxPositions - 1 };
    }

    var encoder = new ChatTranscriptEncoder(model.ChatTemplate ?? throw new InvalidOperationException("The model has no chat template."),
        model.Tokenizer ?? throw new InvalidOperationException("The model has no tokenizer."));
    var recipe = Recipe(positional.Skip(2).ToList(), forTraining: true);
    var counts = new RecipeCounts();
    var (trainRows, heldOut) = recipe.Build(downloads, counts);
    var train = ReadSequences(encoder, trainRows, "training");
    Console.WriteLine($"data: {counts.Rows:N0} rows"
                      + (recipe.Deduplicate ? $", {counts.Duplicates:N0} repeats dropped (--no-dedup keeps them: repeats weigh what is common)" : "")
                      + (counts.OutsideLengths > 0 ? $", {counts.OutsideLengths:N0} outside the length limits" : "")
                      + (heldOut is not null ? $"; prompts held out for evaluation: {recipe.EvaluationFraction:P1}" : ""));
    if (train.Count == 0)
    {
        Console.Error.WriteLine("error: nothing to train on (no conversation with an assistant turn and no text).");
        return 1;
    }

    if (balance > 1)
    {
        var (balanced, answers) = FineTuner.BalanceAnswers(train, balance, seed: seed);
        if (answers.Count is > 1 and <= 64)
        {
            Console.WriteLine($"balanced {answers.Count} answers (at most {balance}×): "
                              + string.Join(", ", answers.OrderByDescending(a => a.Before).Select(a =>
                                  $"{JsonValue.Create(model.Tokenizer!.Decode(a.Answer).Replace(encoder.AssistantEnd, "", StringComparison.Ordinal).Trim()).ToJsonString(readable)} {a.Before:N0}→{a.After:N0}")));
            train = balanced;
        }
        else
        {
            Console.WriteLine($"--balance: {answers.Count:N0} distinct answers (free text, or one): the data is used as it is");
        }
    }

    if (profileTraining)
    {
        return Profile(model, train);
    }

    var evaluationRows = evalFile is not null ? (Recipe([evalFile], forTraining: true) with { EvaluationFraction = 0 }).Build(downloads).Train : heldOut;
    var evaluation = evaluationRows is null ? null : ReadSequences(encoder, evaluationRows, "evaluation");
    Console.WriteLine($"assistant turns start with {JsonValue.Create(encoder.AssistantHeader).ToJsonString(readable)} and end with {JsonValue.Create(encoder.AssistantEnd).ToJsonString(readable)}");
    if (evaluation is { Count: > 0 })
    {
        Console.WriteLine($"evaluation loss before training: {FineTuner.Evaluate(model, evaluation, tuning.BatchTokens, tuning.LossChunkRows):F4}");
    }

    // What the adapter folder records: the base model as named here, so it loads without being told again.
    var record = new TuningManifest { BaseModel = baseModel, System = systemPrompt, MaxLength = tuning.MaxLength };
    using var cancel = new CancellationTokenSource();
    Console.CancelKeyPress += (_, e) =>
    {
        if (cancel.IsCancellationRequested)
        {
            return;                                                           // a second Ctrl+C quits at once
        }

        e.Cancel = true;
        cancel.Cancel();
        status.Log("stopping after the current step and saving the adapter (Ctrl+C again to quit now)…");
    };

    Console.WriteLine($"training on {device}…");
    var clock = Stopwatch.StartNew();
    float? lastEvaluation = null;
    var progress = new Reporter<FineTuningProgress>(p =>
    {
        if (p.EvaluationLoss is { } e)
        {
            lastEvaluation = e;
            status.Log($"  step {p.Step}: evaluation loss {e:F4}");
        }

        status.Progress("training", p.Step, p.TotalSteps, clock.Elapsed,
            $"epoch {p.Epoch}/{tuning.Epochs}, loss {p.Loss:F4}, lr {p.LearningRate:G3}, {p.TokensPerSecond:N0} tokens/s");
    });

    // The trace's notable lines (graph recording, FP8, checkpoints); not the line of every batch.
    void Trace(string line)
    {
        if (!line.StartsWith("step ", StringComparison.Ordinal) && !line.StartsWith("evaluation batch", StringComparison.Ordinal)
            && !line.StartsWith("  forward ", StringComparison.Ordinal))
        {
            status.Log("  " + line);
        }
    }

    try
    {
        FineTuner.Train(model, train, evaluation, tuning, output, progress, cancel.Token, Trace);
        record.Save(output!);
        status.Finish();
        Console.WriteLine($"trained in {Elapsed(clock.Elapsed)}{(lastEvaluation is { } l ? $", evaluation loss {l:F4}" : "")}; adapter written to {Path.GetFullPath(output!)}");
    }
    catch (OperationCanceledException)
    {
        model.SaveAdapter(output!);
        record.Save(output!);
        status.Clear();
        Console.WriteLine($"stopped after {Elapsed(clock.Elapsed)}; the adapter so far written to {Path.GetFullPath(output!)}");
    }

    return 0;
}

int Profile(PretrainedModel model, List<TrainingSequence> train)
{
    Console.WriteLine("profiling: 3 warm-up steps, 3 timed steps, 3 steps with every GPU kernel timed…");
    var measured = FineTuner.Profile(model, train, tuning);
    Console.Write(GpuProfiler.Format(measured.Kernels, rows: 40));
    double gpu = measured.GpuMillisecondsPerStep, wall = measured.SecondsPerStep * 1000;
    if (device.Type != DeviceType.Cuda)
    {
        Console.WriteLine($"per step: {measured.TokensPerStep:N0} tokens, {wall:F0} ms ({measured.TokensPerSecond:N0} tok/s); kernel times need a CUDA device (--cuda)");
        return 0;
    }

    Console.WriteLine($"\nper step: {measured.TokensPerStep:N0} tokens, {wall:F0} ms wall ({measured.TokensPerSecond:N0} tok/s), "
                      + $"{gpu:F0} ms of GPU kernels, {Math.Max(0, wall - gpu):F0} ms of host overhead and waiting ({Math.Max(0, wall - gpu) / Math.Max(wall, 1e-9):P0})");
    Console.WriteLine($"{"kind of work",-32} {"ms/step",9} {"share",7} {"launches",9}");
    foreach (var (group, ms, calls) in measured.Groups)
    {
        Console.WriteLine($"{group,-32} {ms,9:F1} {ms / Math.Max(gpu, 1e-9),7:P1} {calls,9}");
    }

    Console.WriteLine($"GPU memory {ComputeResources.GetMemoryUsage(device)}");
    return 0;
}

int Evaluate()
{
    // The rows: the held-out part when --eval-fraction splits the data as train did, else the data itself.
    var recipe = Recipe(positional.Skip(2).ToList(), forTraining: true) with { Kind = RowKind.Chat };
    var (allRows, heldOut) = recipe.Build(downloads);
    var rows = status.Track(heldOut ?? allRows, "reading").Take(samples > 0 ? samples : int.MaxValue).ToList();
    Console.WriteLine($"{rows.Count} conversations from {(heldOut is null ? "the data" : $"the {recipe.EvaluationFraction:P1} held out of the data")}");
    if (choices is not null)
    {
        return EvaluateChoices(rows);
    }

    var runs = adapterFolder is null ? new[] { (string?)null } : [null, adapterFolder];
    var reports = new List<(string Name, EvaluationReport Report)>();
    string? adapter = adapterFolder;
    foreach (var run in runs)
    {
        adapterFolder = run;
        string name = run is null ? "base model" : $"adapter {Path.GetFileName(Path.TrimEndingDirectorySeparator(run))}";
        Console.WriteLine($"\n{name}:");
        using var model = Load(merge: true);
        var encoder = new ChatTranscriptEncoder(model.ChatTemplate ?? throw new InvalidOperationException("The model has no chat template."), model.Tokenizer!);
        var sequences = rows.SelectMany(r => encoder.EncodeRow((JsonObject)r.DeepClone(), Math.Min(tuning.MaxLength, model.MaxPositions - 1))).ToList();
        double loss = sequences.Count > 0 ? FineTuner.Evaluate(model, sequences, tuning.BatchTokens) : double.NaN;
        var chat = model.CreateChat(cacheFormat, context);
        int done = 0;
        double sum = 0;
        var clock = Stopwatch.StartNew();
        var report = ChatEvaluation.Run(chat, rows.Select(r => (JsonObject)r.DeepClone()), metric, maxNew, noThink ? false : null,
            new Reporter<EvaluatedAnswer>(a =>
            {
                done++;
                sum += a.Score;
                status.Progress("answering", done, rows.Count, clock.Elapsed, $"score {sum / done:P1}", "answers");
            }), context, batchSize: evaluationBatch);
        status.Finish();
        report = report with { Loss = loss };
        reports.Add((name, report));
        Console.WriteLine($"  loss {report.Loss:F4}, {report.Metric.ToString().ToLowerInvariant()} score {report.Score:P1} on {report.Answers.Count} answers, "
                          + $"{report.MeanTokens:F0} tokens per answer, {report.TokensPerSecond:F0} tok/s ({report.Duration.TotalSeconds:F0} s)");
    }

    adapterFolder = adapter;
    if (reports.Count == 2)
    {
        var (b, a) = (reports[0].Report, reports[1].Report);
        int fixedCount = b.Answers.Zip(a.Answers).Count(p => p.First.Score < 0.5 && p.Second.Score >= 0.5);
        int broken = b.Answers.Zip(a.Answers).Count(p => p.First.Score >= 0.5 && p.Second.Score < 0.5);
        Console.WriteLine($"\n{"",-22}{"loss",10}{"score",10}{"tokens",9}");
        foreach (var (name, r) in reports)
        {
            Console.WriteLine($"{name,-22}{r.Loss,10:F4}{r.Score,10:P1}{r.MeanTokens,9:F0}");
        }

        Console.WriteLine($"the adapter answers {fixedCount} questions right that the base model got wrong, and {broken} the other way round");
    }

    if (output is not null)
    {
        using var writer = new StreamWriter(output);
        for (int i = 0; i < reports[0].Report.Answers.Count; i++)
        {
            var line = new JsonObject
            {
                ["prompt"] = reports[0].Report.Answers[i].Prompt[^1].Content,
                ["reference"] = reports[0].Report.Answers[i].Reference,
            };
            foreach (var (name, r) in reports)
            {
                line[name] = new JsonObject { ["answer"] = r.Answers[i].Answer, ["score"] = r.Answers[i].Score };
            }

            writer.WriteLine(line.ToJsonString(readable));
        }

        Console.WriteLine($"answers written to {output}");
    }

    return 0;
}

// --choices: every conversation's answer is one of a known list. Each is scored as the model's answer after the prompt
// (AnswerScorer: the probability of its tokens), the most likely one is the prediction; accuracy and recall per answer.
int EvaluateChoices(List<JsonObject> rows)
{
    var items = rows.Select(ChatTranscript.FromJson)
        .Where(t => t.Messages.Count > 1 && t.Messages[^1].Role == "assistant")
        .Select(t => (Prompt: (IReadOnlyList<ChatMessage>)[.. t.Messages.Take(t.Messages.Count - 1)], Reference: t.Messages[^1].Content.Trim()))
        .ToList();
    string[] answers = choices == "auto"
        ? [.. items.Select(i => i.Reference).Where(r => r.Length > 0).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal)]
        : [.. choices!.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Distinct(StringComparer.Ordinal)];
    if (answers.Length < 2)
    {
        Console.Error.WriteLine($"--choices needs at least two answers (got {answers.Length}).");
        return 1;
    }

    int outside = items.Count(i => !answers.Contains(i.Reference, StringComparer.Ordinal));
    Console.WriteLine($"{items.Count} answers to score among {answers.Length}: {string.Join(", ", answers)}"
                      + (outside > 0 ? $" ({outside} references are none of them and count as wrong)" : ""));
    var runs = adapterFolder is null ? new[] { (string?)null } : [null, adapterFolder];
    var results = new List<(string Name, AnswerChoice[] Choices)>();
    string? adapter = adapterFolder;
    foreach (var run in runs)
    {
        adapterFolder = run;
        string name = run is null ? "base model" : $"adapter {Path.GetFileName(Path.TrimEndingDirectorySeparator(run))}";
        using var model = Load(merge: true);
        var scorer = new AnswerScorer(model, Math.Min(tuning.MaxLength, model.MaxPositions - 1));
        var clock = Stopwatch.StartNew();
        var chosen = scorer.Choose([.. items.Select(i => i.Prompt)], answers,
            done => status.Progress($"scoring ({name})", done, items.Count, clock.Elapsed, $"{done / Math.Max(1e-9, clock.Elapsed.TotalSeconds):F1} answers/s", "answers"));
        status.Finish();
        results.Add((name, chosen));
        int right = items.Zip(chosen).Count(p => answers[p.Second.Best] == p.First.Reference);
        Console.WriteLine($"\n{name}: accuracy {right / (double)Math.Max(1, items.Count):P1} ({right} of {items.Count}) in {clock.Elapsed.TotalSeconds:F1} s"
                          + $" ({items.Count / Math.Max(1e-9, clock.Elapsed.TotalSeconds):F0} answers/s)");
        Console.WriteLine($"  {"answer",-24}{"references",12}{"predicted",11}{"recall",9}");
        foreach (string answer in answers)
        {
            int references = items.Count(i => i.Reference == answer);
            int predicted = chosen.Count(c => answers[c.Best] == answer);
            int hits = items.Zip(chosen).Count(p => p.First.Reference == answer && answers[p.Second.Best] == answer);
            Console.WriteLine($"  {answer,-24}{references,12}{predicted,11}{(references > 0 ? hits / (double)references : double.NaN),9:P1}");
        }
    }

    adapterFolder = adapter;
    if (results.Count == 2)
    {
        int fixedCount = items.Select((item, i) => (item, i)).Count(p => answers[results[0].Choices[p.i].Best] != p.item.Reference && answers[results[1].Choices[p.i].Best] == p.item.Reference);
        int broken = items.Select((item, i) => (item, i)).Count(p => answers[results[0].Choices[p.i].Best] == p.item.Reference && answers[results[1].Choices[p.i].Best] != p.item.Reference);
        Console.WriteLine($"\nthe adapter answers {fixedCount} right that the base model got wrong, and {broken} the other way round");
    }

    if (output is not null)
    {
        // One line per conversation: the prompt's last message, the reference, and each run's answer with every probability.
        using var writer = new StreamWriter(output);
        for (int i = 0; i < items.Count; i++)
        {
            var line = new JsonObject { ["prompt"] = items[i].Prompt[^1].Content, ["reference"] = items[i].Reference };
            foreach (var (name, chosen) in results)
            {
                var probabilities = new JsonObject();
                for (int a = 0; a < answers.Length; a++)
                {
                    probabilities[answers[a]] = Math.Round(chosen[i].Probabilities[a], 6);
                }

                line[name] = new JsonObject { ["answer"] = answers[chosen[i].Best], ["probabilities"] = probabilities };
            }

            writer.WriteLine(line.ToJsonString(readable));
        }

        Console.WriteLine($"answers written to {output}");
    }

    return 0;
}

int Chat()
{
    using var model = Load(merge: true);
    var chat = model.CreateChat(cacheFormat, context);
    var messages = new List<ChatMessage>();
    if (systemPrompt is not null)
    {
        messages.Add(new ChatMessage("system", systemPrompt));
    }

    var given = positional.Skip(2).ToList();
    bool interactive = given.Count == 0;
    if (interactive)
    {
        Console.WriteLine("Type a message (empty line to quit, /system <text> to set the system prompt, /reset to start over).");
    }

    var queue = new Queue<string>(given);
    while ((interactive ? Console.ReadLine() : queue.Count > 0 ? queue.Dequeue() : null) is { Length: > 0 } line)
    {
        if (line.StartsWith("/system ", StringComparison.Ordinal))
        {
            messages.RemoveAll(m => m.Role == "system");
            messages.Insert(0, new ChatMessage("system", line[8..]));
            continue;
        }

        if (line == "/reset")
        {
            messages.RemoveAll(m => m.Role != "system");
            continue;
        }

        if (!interactive)
        {
            Console.WriteLine($"> {line}");
            messages.RemoveAll(m => m.Role != "system");                     // each message given is its own conversation
        }

        messages.Add(new ChatMessage("user", line));
        var options = temperature is 0f
            ? new GenerationOptions { Temperature = 0f, TopK = 1, TopP = 1f, RepeatPenalty = 1f, NumPredict = maxNew, NumCtx = context, ChunkSize = 1 }
            : new GenerationOptions { Temperature = temperature ?? 0.6f, TopK = 20, TopP = 0.95f, RepeatPenalty = 1f, NumPredict = maxNew, NumCtx = context, ChunkSize = 1 };
        ChatMessage? reply = null;
        bool thinking = false;
        foreach (var chunk in chat.Stream(new ChatRequest(messages, Think: noThink ? false : null, Options: options)))
        {
            if (chunk.Delta.Thinking.Length > 0)
            {
                if (!thinking)
                {
                    Console.ForegroundColor = ConsoleColor.DarkGray;
                    thinking = true;
                }

                Console.Write(chunk.Delta.Thinking);
            }

            if (chunk.Delta.Content.Length > 0)
            {
                if (thinking)
                {
                    Console.ResetColor();
                    Console.WriteLine();
                    thinking = false;
                }

                Console.Write(chunk.Delta.Content);
            }

            foreach (var call in chunk.Delta.ToolCalls)
            {
                Console.Write($"\n[tool call] {call.Name}({call.Arguments.ToJsonString()})");
            }

            if (chunk.Done)
            {
                reply = chunk.Message;
                Console.ResetColor();
                Console.WriteLine($"\n[{chunk.Stats?.GeneratedTokens} tokens, {chunk.Stats?.TokensPerSecond:F1} tok/s]");
            }
        }

        if (reply is not null)
        {
            messages.Add(reply);
        }
    }

    return 0;
}

// The model: the base, with the adapter merged as the weights are read (full speed) or kept separate (to train it).
PretrainedModel Load(bool merge)
{
    string folder = ModelSource.Resolve(baseModel, downloader: downloads);
    var watch = Stopwatch.StartNew();
    var model = PretrainedModel.Load(folder, new PretrainedOptions
    {
        Device = device, Int8 = int8, BFloat16 = bf16, Int4 = int4, MaxPositions = context, MergeAdapter = merge ? adapterFolder : null,
    });
    Console.WriteLine($"loaded {model.Config["architectures"]?[0]} from {folder} in {watch.Elapsed.TotalSeconds:F1} s on {device}{(int8 ? ", int8 weights" : int4 ? ", int4 weights" : bf16 ? ", bf16 weights" : "")}");
    Console.WriteLine($"  {model.Spec.ParameterCount / 1e6:F0}M parameters, {model.Spec.Layers} layers, dim {model.Spec.Dim}, heads {model.Spec.Heads}/{model.Spec.KvHeads}, "
                      + $"vocabulary {model.Spec.Vocabulary}, context {model.MaxPositions}");
    foreach (var note in model.Notes)
    {
        Console.WriteLine($"  note: {note}");
    }

    if (adapterFolder is not null && !merge)
    {
        Console.WriteLine($"  adapter: {model.LoadAdapter(adapterFolder)} layers from {adapterFolder}");
    }

    return model;
}

// The data from the command line: one recipe file, or sources with the recipe options given as flags.
DatasetRecipe Recipe(IReadOnlyList<string> specs, bool forTraining)
{
    if (specs is [var single] && single.EndsWith(".json", StringComparison.OrdinalIgnoreCase) && File.Exists(single)
        && JsonNode.Parse(File.ReadAllText(single)) is JsonObject json && json.ContainsKey("sources"))
    {
        var loaded = DatasetRecipe.Load(single);
        return evalFraction > 0 ? loaded with { EvaluationFraction = evalFraction } : loaded;
    }

    return new DatasetRecipe
    {
        Sources = [.. specs.Select(DatasetSpec.Parse)],
        Kind = rowKind,
        System = systemPrompt,
        MixByWeight = mixByWeight ? true : null,
        Seed = seed,
        Shuffle = shuffleRows && !forTraining,          // fine-tuning orders its batches itself
        Deduplicate = dedupRows,
        MinCharacters = minChars,
        MaxCharacters = maxChars,
        MaxRows = maxRows,
        EvaluationFraction = evalFraction,
    };
}

// Rows tokenized into training sequences, with a count of what was read, shortened and skipped.
List<TrainingSequence> ReadSequences(ChatTranscriptEncoder encoder, Dataset rows, string what)
{
    var watch = Stopwatch.StartNew();
    var sequences = new List<TrainingSequence>();
    long read = 0, skipped = 0, chats = 0, texts = 0, tokens = 0;
    Console.WriteLine($"{what}: reading and tokenizing {rows.Name}");
    foreach (var (row, encoded) in encoder.EncodeRows(status.Track(rows, what, extra: () => $"  {Interlocked.Read(ref tokens):N0} tokens"), tuning.MaxLength))
    {
        read++;
        int before = sequences.Count;
        foreach (var sequence in encoded)
        {
            Interlocked.Add(ref tokens, sequence.Tokens.Length);
            sequences.Add(sequence);
        }

        if (sequences.Count == before)
        {
            skipped++;
        }
        else if (row.ContainsKey("messages"))
        {
            chats++;
        }
        else
        {
            texts++;
        }
    }

    Console.WriteLine($"{what}: {read:N0} rows ({chats:N0} conversations, {texts:N0} texts) → {sequences.Count:N0} sequences, "
                      + $"{sequences.Sum(q => (long)q.Tokens.Length):N0} tokens, {sequences.Sum(q => (long)q.TrainedTokens):N0} trained; "
                      + $"{skipped:N0} rows without trainable tokens skipped ({watch.Elapsed.TotalSeconds:F1} s)");
    return sequences;
}

static string Elapsed(TimeSpan time) =>
    time.TotalHours >= 1 ? $"{(int)time.TotalHours}:{time.Minutes:D2}:{time.Seconds:D2}" : time.TotalMinutes >= 1 ? $"{time.TotalMinutes:F1} min" : $"{time.TotalSeconds:F1} s";

// Reports on the calling thread (Progress<T> posts to the thread pool, so its reports can arrive late).
internal sealed class Reporter<T>(Action<T> report) : IProgress<T>
{
    public void Report(T value) => report(value);
}
