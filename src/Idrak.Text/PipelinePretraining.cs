// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Cluster;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Text.Abstractions;
using Idrak.Training;

namespace Idrak.Text;

public static partial class Pretrainer
{
    private const string PipelineKey = "pipeline";
    private const int GatherTag = 7 << 20;

    // Pipeline placement (plan 23, phase C7): each rank trains its stage of the model; the step's micro-batches run through
    // the stages in the schedule's order (Idrak's PipelineStage), each pipeline's ranks take their share of the step's draws
    // as data-parallel replicas when the stages are fewer than the ranks, and rank 0 writes whole checkpoints, every stage's
    // weights and optimizer state gathered to it. Token sources train the decoder; what single placement adds beside it
    // (conversations with images or parts, a teacher, the host optimizer) is refused by name.
    private static PretrainingResult TrainPipelined(PretrainedModel model, PretrainingOptions options, ClusterOptions cluster, string outputFolder,
        PretrainingCheckpoint? resume, IProgress<PretrainingProgress>? progress, CancellationToken cancellationToken, Action<string>? trace)
    {
        var collectives = options.Cluster!;
        string? refused = options.Teacher is not null ? "a teacher's distillation"
            : options.HostOptimizer ? "the host optimizer (PretrainingOptions.HostOptimizer)"
            : options.Stages.SelectMany(s => s.Sources).Concat(options.Evaluation).Any(s => s.Kind.Equals(PretrainingSources.Records, StringComparison.OrdinalIgnoreCase))
                ? "conversation records (their images and parts are read beside the decoder)"
            : options.Stages.Any(s => s.Trained.Any(t => t.Equals(VisionTuningParts.Tower, StringComparison.OrdinalIgnoreCase) || t.Equals(VisionTuningParts.Projector, StringComparison.OrdinalIgnoreCase)))
                ? "training the vision part" : null;
        if (refused is not null)
        {
            throw new NotSupportedException($"Pipeline placement (IDRAK_CLUSTER_PLACEMENT=pipeline) trains a decoder's stages on token sources; {refused} runs with single placement.");
        }

        if (resume is not null && ReadPipeline(resume.Folder) is null)
        {
            throw new NotSupportedException($"The checkpoint {resume.Folder} was written with single placement: its optimizer state is the whole model's, not a stage's; resume it with single placement.");
        }

        Directory.CreateDirectory(outputFolder);
        var started = Stopwatch.StartNew();
        bool lead = collectives.Rank == 0;
        using var pipe = PipelineStage.Place(model.Network, collectives, cluster, keepOthers: lead);
        if (pipe.IsLast && pipe.Head is not Linear)
        {
            throw new InvalidOperationException("The network does not end with its output head (a Linear layer): the last stage's loss runs it.");
        }

        if (resume is not null && ReadPipeline(resume.Folder) is { } saved && !saved.SequenceEqual(pipe.Plan.Stages.Select(s => s.Start)))
        {
            throw new NotSupportedException($"The checkpoint {resume.Folder} was written by a pipeline cut at [{string.Join(", ", saved)}], this one is cut at "
                + $"[{string.Join(", ", pipe.Plan.Stages.Select(s => s.Start))}]: each stage's optimizer state belongs to its cut; set IDRAK_CLUSTER_PIPELINE_CUTS to the checkpoint's.");
        }

        string settings = string.Create(CultureInfo.InvariantCulture,
            $"pipeline {pipe.Layout.Stages} stage(s) × {pipe.Layout.Replicas} replica(s), {pipe.Schedule.Name}, cut by {pipe.Plan.Basis}: {pipe.Plan}");
        trace?.Invoke($"rank {pipe.Rank}: stage {pipe.Stage}{(pipe.Layout.Replicas > 1 ? $" of replica {pipe.Replica}" : "")}, {pipe.Modules.Count} module(s), {pipe.Bytes / 1024.0:F1} KiB; {settings}");
        var losses = new List<float>(resume?.Losses ?? []);
        var evaluations = new List<(int, float)>(resume?.EvaluationLosses ?? []);
        int totalSteps = options.Stages.Sum(s => s.Steps);
        int step = resume?.Step ?? 0;
        string? last = resume?.Folder;
        double secondsPerStep = 0;
        int timedSteps = 0;
        bool startedTelemetry = false;
        using var seeds = Dropout.SeedsForRank(collectives.Rank);

        for (int stageIndex = resume?.Stage ?? 0; stageIndex < options.Stages.Count; stageIndex++)
        {
            var stage = options.Stages[stageIndex];
            int stageStep = resume is { } r && r.Stage == stageIndex ? r.StageStep : 0;
            if (stageStep >= stage.Steps)
            {
                continue;
            }

            var context = new PretrainingContext(model, options.MaxLength, null);
            var readers = new List<SourceState>();
            try
            {
                for (int s = 0; s < stage.Sources.Count; s++)
                {
                    var source = stage.Sources[s];
                    var reader = PretrainingSources.Get(source.Kind).Open(source, context);
                    var position = resume is { } rr && rr.Stage == stageIndex && s < rr.Sources.Count ? rr.Sources[s] : (0L, 0L, 0L);
                    readers.Add(new SourceState(reader, source.Weight, options.Seed, stageIndex, s) { Pass = position.Item1, Next = position.Item2, Drawn = position.Item3 });
                }

                if (readers.All(rd => rd.Reader.Count == 0))
                {
                    throw new ArgumentException($"Stage {stage.Name}'s sources hold no sequences.");
                }

                var evaluation = ReadEvaluation(options, context);
                var parameters = StageTrainable(model, stage, pipe);
                using var optimizer = CreateOptimizer(parameters, options, stage);
                if (resume is { } rs && rs.Stage == stageIndex)
                {
                    // This stage's entries alone are read from the checkpoint (plan 23, C8b).
                    string prefix = $"{PipelineKey}.{pipe.Stage}.";
                    optimizer.LoadState(ReadOptimizerState(rs.Folder, name => name.StartsWith(prefix, StringComparison.Ordinal))
                        .ToDictionary(e => e.Key[prefix.Length..], e => e.Value, StringComparer.Ordinal));
                }

                // The ranks holding this stage (its replicas) sum its gradients; with one replica there is nothing to sum.
                using var sync = pipe.Replicas is { } replicas ? GradientSyncs.Create(replicas, optimizer, cluster) : null;
                sync?.Start();
                var schedule = FineTuningSchedules.Create(stage.Schedule, stage.WarmupFraction, stage.MinLearningRate)(optimizer, stage.Steps);
                for (int i = 0; i < stageStep; i++)
                {
                    schedule.Step();
                }

                if (!startedTelemetry && lead)
                {
                    startedTelemetry = true;
                    Telemetry.TrainingStarted(new TrainingStarted((string?)model.Config["architectures"]?[0] ?? "model", options.Optimizer, pipe.Device, options.Stages.Count,
                        (int)Math.Min(int.MaxValue, readers.Sum(rd => rd.Reader.Count)), evaluation.Count, Rows(options), stage.Steps,
                        parameters.Sum(p => (long)p.Size), stage.LearningRate, Environment.ProcessorCount));
                }

                (long, long, long)[] finished = [.. readers.Select(rd => (rd.Pass, rd.Next, rd.Drawn))];
                var next = Task.Run(() => Draw(readers, options, model));
                while (stageStep < stage.Steps)
                {
                    var (drawn, positions) = next.GetAwaiter().GetResult();
                    next = stageStep + 1 < stage.Steps ? Task.Run(() => Draw(readers, options, model)) : Task.FromResult<(List<TrainingSequence>, (long, long, long)[])>(([], []));
                    bool present;
                    try
                    {
                        present = collectives is not IStepGroup group || group.BeginStep(step);
                    }
                    catch (ClusterRunStoppedException) when (step > 0 && last != Path.Combine(outputFolder, $"checkpoint-{step}"))
                    {
                        // Stopped at this step boundary (a dashboard's stop, plan 23 C9; written since C8b): every rank stops
                        // here alike, so every stage gives its part of the last finished step's checkpoint as a saving step does.
                        last = SavePipelined(model, pipe, collectives, optimizer, options, cluster, outputFolder, stageIndex, stageStep, step, finished, losses, evaluations);
                        trace?.Invoke($"stopped: checkpoint {last}");
                        throw;
                    }

                    if (!present)
                    {
                        if (cluster.Simulation.ReplayStep is not { } replayed || step <= replayed)
                        {
                            throw new NotSupportedException($"Rank {collectives.Rank} sits out step {step} (a late join): a pipeline needs every stage in every step; "
                                + "run pipeline placement without the late-join failure model (IDRAK_CLUSTER_FAILURES).");
                        }

                        schedule.Step();                                     // every rank sits out past a replay's last step alike
                        finished = positions;
                        step++;
                        stageStep++;
                        continue;
                    }

                    cancellationToken.ThrowIfCancellationRequested();
                    var watch = Stopwatch.StartNew();
                    float rate = optimizer.LearningRate;
                    var owners = sync is null ? null : DrawShares.Owners([.. drawn.Select(q => (long)q.TrainedTokens)], sync.Shares);
                    var mine = owners is null ? drawn : [.. drawn.Where((_, i) => owners[i] == pipe.Replica)];
                    var batches = MicroBatches(model, mine, options, cluster);
                    var (loss, tokens) = PipelinedStep(pipe, optimizer, mine, batches, options, step, trace, $"{stage.Name} step {stageStep + 1}/{stage.Steps}");
                    if (sync is not null)
                    {
                        var reduced = sync.Reduce(mine.Sum(q => q.TrainedTokens), loss);
                        (loss, tokens) = ((float)reduced.Loss, (long)reduced.Tokens);
                    }

                    if (float.IsFinite(loss))
                    {
                        pipe.ClipAndStep(optimizer, options.MaxGradientNorm);
                    }
                    else
                    {
                        optimizer.ZeroGrad();
                        if (losses.Skip(Math.Max(0, losses.Count - options.NonFiniteSteps)).Count(l => !float.IsFinite(l)) >= options.NonFiniteSteps)
                        {
                            throw new InvalidOperationException($"The loss of step {step + 1} is {loss}, the last {options.NonFiniteSteps + 1} steps not finite ({settings}); "
                                + "the run stops (PretrainingOptions.NonFiniteSteps).");
                        }
                    }

                    schedule.Step();
                    finished = positions;
                    losses.Add(loss);
                    step++;
                    stageStep++;
                    sync?.Stepped(step);
                    double seconds = watch.Elapsed.TotalSeconds;
                    if (step > (resume?.Step ?? 0) + 1)
                    {
                        secondsPerStep = (secondsPerStep * timedSteps + seconds) / ++timedSteps;
                    }

                    float? evaluationLoss = null;
                    bool stageEnd = stageStep == stage.Steps;
                    if (evaluation.Count > 0 && (options.EvaluateEvery > 0 ? step % options.EvaluateEvery == 0 : stageEnd))
                    {
                        evaluationLoss = PipelinedEvaluation(pipe, sync, evaluation, options);
                        evaluations.Add((step, evaluationLoss.Value));
                    }

                    if (lead && Telemetry.IsEnabled(TelemetryLevel.Batches))
                    {
                        Telemetry.BatchCompleted(new BatchCompleted(stageIndex + 1, stageStep, stage.Steps, step, drawn.Count, loss, rate, null, TimeSpan.Zero, watch.Elapsed));
                    }

                    if (stageEnd || options.SaveEvery > 0 && step % options.SaveEvery == 0)
                    {
                        last = SavePipelined(model, pipe, collectives, optimizer, options, cluster, outputFolder, stageIndex, stageStep, step, positions, losses, evaluations);
                        trace?.Invoke($"checkpoint {last}");
                    }

                    long peak = pipe.Device.Backend.GetMemoryUsage().Peak;
                    progress?.Report(new PretrainingProgress(stage.Name, stageIndex, stageStep, stage.Steps, step, totalSteps, loss, rate,
                        tokens / Math.Max(1e-9, seconds), 0, peak, TimeSpan.FromSeconds(secondsPerStep * (totalSteps - step)), evaluationLoss));
                }

                _ = finished;
            }
            finally
            {
                readers.ForEach(rd => rd.Reader.Dispose());
            }

            resume = null;
        }

        foreach (var module in pipe.Modules)
        {
            module.Eval();
        }

        if (lead)
        {
            Telemetry.TrainingCompleted(new TrainingCompleted(options.Stages.Count, started.Elapsed, losses.Count > 0 ? losses[^1] : double.NaN,
                options.Stages.Count, evaluations.Count > 0 ? evaluations.Min(e => e.Item2) : double.NaN, false, false));
        }

        return new PretrainingResult(step, losses, evaluations, last, started.Elapsed, settings);
    }

    // The stage's parameters the pretraining stage trains (its parts: embeddings, decoder, head), in the stage's order.
    private static List<Tensor> StageTrainable(PretrainedModel model, PretrainingStage stage, PipelineStage pipe)
    {
        bool all = stage.Trained.Contains("all", StringComparer.OrdinalIgnoreCase);
        bool Trains(string part) => all || stage.Trained.Contains(part, StringComparer.OrdinalIgnoreCase);
        var held = new HashSet<Tensor>(pipe.Parameters, ReferenceEqualityComparer.Instance);
        var part = new Dictionary<Tensor, string>(ReferenceEqualityComparer.Instance);
        foreach (var (path, module) in model.NamedModules())
        {
            if (module is Linear { TiedTo: not null })
            {
                continue;
            }

            string top = path.Split('.')[0];
            string name = top is "embed" or "pos" ? "embeddings" : top == "head" ? "head" : "decoder";
            foreach (var p in module.Parameters().Except(module.Children().SelectMany(c => c.Parameters()), ReferenceEqualityComparer.Instance).Cast<Tensor>())
            {
                if (held.Contains(p))
                {
                    part.TryAdd(p, name);
                }
            }
        }

        var chosen = new List<Tensor>();
        foreach (var p in pipe.Parameters)
        {
            p.RequiresGrad = part.TryGetValue(p, out string? name) && Trains(name);
            if (p.RequiresGrad)
            {
                chosen.Add(p);
            }
        }

        // A stage may hold nothing the training stage trains (a frozen part); its optimizer then steps over nothing.
        return chosen;
    }

    // The step's micro-batches over this pipeline's sequences: IDRAK_CLUSTER_MICROBATCHES of them (one sequence a row, the
    // sequences shared out in order), or the pretrainer's own batches (packed rows of MaxLength where the model packs).
    private static List<FineTuner.Batch> MicroBatches(PretrainedModel model, List<TrainingSequence> sequences, PretrainingOptions options, ClusterOptions cluster)
    {
        if (cluster.MicroBatches is not { } count)
        {
            return Batches(model, sequences, options);
        }

        int m = Math.Min(count, Math.Max(1, sequences.Count));
        var batches = new List<FineTuner.Batch>(m);
        for (int b = 0, first = 0; b < m; b++)
        {
            int size = (sequences.Count / m) + (b < sequences.Count % m ? 1 : 0);
            if (size > 0)
            {
                batches.Add(FineTuner.Batch.Padded([.. Enumerable.Range(first, size)], sequences));
            }

            first += size;
        }

        return batches;
    }

    // One step through this rank's stage: the micro-batches' forward and backward passes in the schedule's order, each
    // micro-batch's token loss divided by the pipeline's trained tokens, so their sum is the step's mean loss.
    private static (float Loss, long Tokens) PipelinedStep(PipelineStage pipe, Optimizer optimizer, List<TrainingSequence> sequences, List<FineTuner.Batch> batches,
        PretrainingOptions options, int step, Action<string>? trace, string label)
    {
        float normalizer = Math.Max(1, sequences.Sum(q => q.TrainedTokens));
        var data = batches.Select(b => FineTuner.Prepare(sequences, b)).ToList();
        var packings = batches.Select(b => b.Packed
            ? PackedSequences.Create([.. b.Rows.Select(r => r.Select(i => sequences[i].Tokens.Length - 1).ToArray())], b.Length, pipe.Device) : null).ToList();
        try
        {
            foreach (var module in pipe.Modules)
            {
                module.Train();
            }

            optimizer.ZeroGrad();
            var watch = Stopwatch.StartNew();
            using var recompute = options.RecomputeFeedForward == true ? ActivationMemory.Recompute() : (ActivationMemory.Scope?)null;
            using var compress = options.BFloat16Activations == true ? ActivationMemory.CompressToBFloat16() : (ActivationMemory.Scope?)null;
            var result = pipe.Step(Math.Max(1, batches.Count),
                m => Tensor.From(data[m].Inputs, [batches[m].Rows.Length, batches[m].Length], pipe.Device),
                (m, hidden) =>
                {
                    var flat = hidden.Reshape(hidden.Size / hidden.Shape[^1], hidden.Shape[^1]);
                    return Losses.TokenCrossEntropyRows(flat, h => pipe.Head.Forward(h), data[m].Trained, data[m].Targets, data[m].Weights, normalizer, options.LossChunkRows);
                },
                new PipelineStepOptions { HeadInLoss = true, Checkpointing = options.Checkpointing == true, Around = m => packings[m]?.Use(), Step = step });
            trace?.Invoke(string.Create(CultureInfo.InvariantCulture,
                $"{label}: stage {pipe.Stage}, {batches.Count} micro-batch(es), loss {result.Loss:F4}, {watch.Elapsed.TotalSeconds:F2} s, blocked {result.Blocked.TotalMilliseconds:F1} ms (bubble {result.Bubble:P0}, {pipe.Schedule.Name}'s {result.IdealBubble:P0})"));
            return ((float)result.Loss, data.Sum(d => d.Tokens));
        }
        finally
        {
            packings.ForEach(p => p?.Dispose());
        }
    }

    // The mean loss per trained token of the evaluation sequences through the pipeline (each replica its share, summed).
    private static float PipelinedEvaluation(PipelineStage pipe, IGradientSync? sync, List<TrainingSequence> evaluation, PretrainingOptions options)
    {
        foreach (var module in pipe.Modules)
        {
            module.Eval();
        }

        int replicas = pipe.Layout.Replicas;
        var mine = evaluation.Where((_, i) => i % replicas == pipe.Replica).ToList();
        var batches = FineTuner.Batches(mine, options.BatchTokens, random: null).Select(b => FineTuner.Batch.Padded(b, mine)).ToList();
        var data = batches.Select(b => FineTuner.Prepare(mine, b)).ToList();
        double total = batches.Count == 0 ? 0 : pipe.Evaluate(batches.Count,
            m => Tensor.From(data[m].Inputs, [batches[m].Rows.Length, batches[m].Length], pipe.Device),
            (m, hidden) =>
            {
                var flat = hidden.Reshape(hidden.Size / hidden.Shape[^1], hidden.Shape[^1]);
                return Losses.TokenCrossEntropyRows(flat, h => pipe.Head.Forward(h), data[m].Trained, data[m].Targets, data[m].Weights, 1f, options.LossChunkRows);
            });
        Span<float> sums = [(float)total, mine.Sum(q => q.TrainedTokens)];
        sync?.Sum(sums);
        return sums[0] / Math.Max(1, sums[1]);
    }

    // A whole checkpoint from a pipeline: replica 0 of every stage sends its parameters and its optimizer's state to rank 0
    // (stage 0), which loads them into the copies it kept of the other stages, writes the model, every stage's optimizer
    // state (under "pipeline.<stage>.") and the run's state with the cut; then every rank meets.
    private static string SavePipelined(PretrainedModel model, PipelineStage pipe, ICollectives collectives, Optimizer optimizer, PretrainingOptions options,
        ClusterOptions cluster, string outputFolder, int stage, int stageStep, int step, (long, long, long)[] positions, List<float> losses, List<(int, float)> evaluations)
    {
        string folder = Path.Combine(outputFolder, $"checkpoint-{step}");
        if (cluster.StreamedSave)
        {
            return SavePipelinedStreamed(model, pipe, collectives, optimizer, options, folder, outputFolder, stage, stageStep, step, positions, losses, evaluations);
        }

        if (pipe.Replica == 0)
        {
            if (pipe.Stage != 0)
            {
                int owner = pipe.Layout.RankOf(0, 0);
                var owned = OwnedParameters(pipe);
                var indices = new byte[sizeof(int) * (owned.Count + 1)];
                BinaryPrimitives.WriteInt32LittleEndian(indices, owned.Count);
                for (int i = 0; i < owned.Count; i++)
                {
                    BinaryPrimitives.WriteInt32LittleEndian(indices.AsSpan(sizeof(int) * (i + 1)), owned[i].Global);
                }

                SendBlock(pipe.Transport, owner, indices, GatherTag);
                foreach (var (_, tensor) in owned)
                {
                    SendBlock(pipe.Transport, owner, MemoryMarshal.AsBytes(tensor.ToArray().AsSpan()).ToArray(), GatherTag + 2);
                }

                SendBlock(pipe.Transport, owner, PackState(optimizer.SaveState()), GatherTag + 4);
            }
            else
            {
                var everything = model.Network.Parameters().Distinct(ReferenceEqualityComparer.Instance).Cast<Tensor>().ToList();
                var state = new Dictionary<string, OptimizerStateValue>(StringComparer.Ordinal);
                foreach (var (name, value) in optimizer.SaveState())
                {
                    state[$"{PipelineKey}.0.{name}"] = value;
                }

                for (int s = 1; s < pipe.Layout.Stages; s++)
                {
                    int from = pipe.Layout.RankOf(s, 0);
                    var indices = ReceiveBlock(pipe.Transport, from, GatherTag);
                    int count = BinaryPrimitives.ReadInt32LittleEndian(indices);
                    for (int i = 0; i < count; i++)
                    {
                        int global = BinaryPrimitives.ReadInt32LittleEndian(indices.AsSpan(sizeof(int) * (i + 1)));
                        var values = MemoryMarshal.Cast<byte, float>(ReceiveBlock(pipe.Transport, from, GatherTag + 2).AsSpan());
                        everything[global].Load(values);
                    }

                    foreach (var (name, value) in UnpackState(ReceiveBlock(pipe.Transport, from, GatherTag + 4)))
                    {
                        state[$"{PipelineKey}.{s}.{name}"] = value;
                    }
                }

                model.SaveHuggingFace(folder, SafeTensorType.F32);
                SafeTensorsWriter.Write(Path.Combine(folder, OptimizerFile), state.Select(p => (p.Key, p.Value.Shape, p.Value.Values)), SafeTensorType.F32,
                    new Dictionary<string, string> { ["format"] = "idrak-optimizer", ["optimizer"] = optimizer.GetType().Name, ["pipeline"] = pipe.Layout.Stages.ToString(CultureInfo.InvariantCulture) });
                WritePipelineState(pipe, options, folder, outputFolder, stage, stageStep, step, positions, losses, evaluations);
            }
        }

        collectives.Barrier();
        return folder;
    }

    // The checkpoint streamed to rank 0 (plan 23, C8b): replica 0 of every other stage answers rank 0's requests a tensor at a
    // time (its weights, its optimizer entries) while rank 0 writes the files one process writes (the stage's entries under
    // "pipeline.<stage>.", in stage order), so no rank holds another stage's state, and rank 0 never more than one tensor of it.
    private static string SavePipelinedStreamed(PretrainedModel model, PipelineStage pipe, ICollectives collectives, Optimizer optimizer, PretrainingOptions options,
        string folder, string outputFolder, int stage, int stageStep, int step, (long, long, long)[] positions, List<float> losses, List<(int, float)> evaluations)
    {
        var transport = pipe.Transport;
        int writer = pipe.Layout.RankOf(0, 0);
        if (pipe.Replica == 0 && pipe.Stage != 0)
        {
            var mine = new Dictionary<int, Tensor>();
            for (int i = 0; i < pipe.Parameters.Count; i++)
            {
                mine[pipe.ParameterIndices[i]] = pipe.Parameters[i];
            }

            StreamedCheckpoint.Serve(transport, writer, global => mine[global].ToArray(), optimizer.SaveState());
        }
        else if (pipe.Replica == 0)
        {
            var global = new Dictionary<Tensor, int>(ReferenceEqualityComparer.Instance);
            foreach (var p in model.Network.Parameters())
            {
                global.TryAdd(p, global.Count);
            }

            var owners = Enumerable.Range(1, pipe.Layout.Stages - 1).Select(s => pipe.Layout.RankOf(s, 0)).ToList();
            var entries = owners.Select(rank => StreamedCheckpoint.AskEntries(transport, rank)).ToList();
            model.SaveHuggingFace(folder, SafeTensorType.F32, null, stored =>
            {
                int index = global[stored.Tensor];
                int holder = pipe.HolderOf(index);
                return holder == 0 ? stored.Tensor.ToArray() : StreamedCheckpoint.AskWeight(transport, pipe.Layout.RankOf(holder, 0), index);
            });

            var own = optimizer.SaveState();
            var tensors = own.Select(e => ($"{PipelineKey}.0.{e.Key}", e.Value.Shape, (Func<float[]>)(() => e.Value.Values))).ToList();
            for (int o = 0; o < owners.Count; o++)
            {
                int rank = owners[o], s = o + 1;
                tensors.AddRange(entries[o].Select((e, i) => ($"{PipelineKey}.{s}.{e.Name}", e.Shape, (Func<float[]>)(() => StreamedCheckpoint.AskEntry(transport, rank, i)))));
            }

            SafeTensorsWriter.Write(Path.Combine(folder, OptimizerFile), tensors, SafeTensorType.F32,
                new Dictionary<string, string> { ["format"] = "idrak-optimizer", ["optimizer"] = optimizer.GetType().Name, ["pipeline"] = pipe.Layout.Stages.ToString(CultureInfo.InvariantCulture) });
            owners.ForEach(rank => StreamedCheckpoint.Finish(transport, rank));
            WritePipelineState(pipe, options, folder, outputFolder, stage, stageStep, step, positions, losses, evaluations);
        }

        collectives.Barrier();
        return folder;
    }

    // The run's state with the cut (rank 0, last: a checkpoint counts once its state file is there), and old checkpoints gone.
    private static void WritePipelineState(PipelineStage pipe, PretrainingOptions options, string folder, string outputFolder, int stage, int stageStep, int step,
        (long, long, long)[] positions, List<float> losses, List<(int, float)> evaluations)
    {
        var json = new JsonObject
        {
            ["format"] = StateFormat,
            ["stage"] = stage,
            ["stage_step"] = stageStep,
            ["step"] = step,
            ["sources"] = new JsonArray([.. positions.Select(p => (JsonNode)new JsonObject { ["pass"] = p.Item1, ["next"] = p.Item2, ["drawn"] = p.Item3 })]),
            ["losses"] = new JsonArray([.. losses.Select(l => (JsonNode)(float.IsFinite(l) ? l : float.NaN.ToString(CultureInfo.InvariantCulture)))]),
            ["evaluations"] = new JsonArray([.. evaluations.Select(e => (JsonNode)new JsonObject { ["step"] = e.Item1, ["loss"] = e.Item2 })]),
            ["options"] = options.ToJson(),
            [PipelineKey] = new JsonObject { ["stages"] = pipe.Layout.Stages, ["starts"] = new JsonArray([.. pipe.Plan.Stages.Select(s => (JsonNode)s.Start)]) },
        };
        File.WriteAllText(Path.Combine(folder, StateFile), json.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }) + "\n");
        if (options.KeepCheckpoints > 0)
        {
            foreach (var old in Checkpoints(outputFolder).SkipLast(options.KeepCheckpoints))
            {
                Directory.Delete(old.Folder, recursive: true);
            }
        }
    }

    // The parameters this stage writes into a whole checkpoint: each once, a table several stages hold by the first of them.
    private static List<(int Global, Tensor Tensor)> OwnedParameters(PipelineStage pipe) =>
        [.. Enumerable.Range(0, pipe.Parameters.Count).Where(pipe.Owns).Select(i => (pipe.ParameterIndices[i], pipe.Parameters[i]))];

    private static void SendBlock(ITransport transport, int to, byte[] bytes, int tag)
    {
        Span<byte> length = stackalloc byte[sizeof(int)];
        BinaryPrimitives.WriteInt32LittleEndian(length, bytes.Length);
        transport.Send(to, length, tag);
        transport.Send(to, bytes, tag + 1);
    }

    private static byte[] ReceiveBlock(ITransport transport, int from, int tag)
    {
        Span<byte> length = stackalloc byte[sizeof(int)];
        transport.Receive(from, length, tag);
        var bytes = new byte[BinaryPrimitives.ReadInt32LittleEndian(length)];
        transport.Receive(from, bytes, tag + 1);
        return bytes;
    }

    // An optimizer's state as bytes: per entry its name (UTF-8), shape and float values, little-endian.
    private static byte[] PackState(IReadOnlyDictionary<string, OptimizerStateValue> state)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8);
        writer.Write(state.Count);
        foreach (var (name, value) in state)
        {
            writer.Write(name);
            writer.Write(value.Shape.Length);
            foreach (int d in value.Shape)
            {
                writer.Write(d);
            }

            writer.Write(value.Values.Length);
            writer.Write(MemoryMarshal.AsBytes(value.Values.AsSpan()));
        }

        writer.Flush();
        return stream.ToArray();
    }

    private static Dictionary<string, OptimizerStateValue> UnpackState(byte[] bytes)
    {
        using var reader = new BinaryReader(new MemoryStream(bytes), Encoding.UTF8);
        int count = reader.ReadInt32();
        var state = new Dictionary<string, OptimizerStateValue>(count, StringComparer.Ordinal);
        for (int i = 0; i < count; i++)
        {
            string name = reader.ReadString();
            var shape = new int[reader.ReadInt32()];
            for (int d = 0; d < shape.Length; d++)
            {
                shape[d] = reader.ReadInt32();
            }

            var values = new float[reader.ReadInt32()];
            reader.BaseStream.ReadExactly(MemoryMarshal.AsBytes(values.AsSpan()));
            state[name] = new OptimizerStateValue(shape, values);
        }

        return state;
    }

    // The cut a pipelined checkpoint was written by (each stage's first module), or null for one of single placement.
    private static int[]? ReadPipeline(string folder)
    {
        string path = Path.Combine(folder, StateFile);
        return File.Exists(path) && JsonNode.Parse(File.ReadAllText(path))?[PipelineKey]?["starts"] is JsonArray starts
            ? [.. starts.Select(s => (int)s!)]
            : null;
    }
}
