// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Abstraction.Reinforcement;
using Idrak.Abstraction.Testing;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Reinforcement;
using Idrak.Text;
using Idrak.Text.Abstractions;
using Idrak.Text.Generation;
using Idrak.Text.Reinforcement;

// Plan 27, R3: the language-model policy in Idrak.Text, on tiny decoders made by the tests (no download). Idrak.Text does
// not reference Idrak.Reinforcement; since R4 the policy is trained here by R2's trainer, estimators and losses (grpo,
// ppo) through Idrak.Abstraction's registries, with a test-only wrapper of a library loss that records what it was given.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] ReinforcementTextGroup =
    [
        ("reinforcement text: FineTuner.Train is a loop over a session: its losses step by step and its adapters equal, bit for bit, a session stepped by hand (the fused loss, a loss of one's own, one given per step)", SessionMatchesTrain),
        ("reinforcement text: a session's log-probabilities follow its adapters (off: the base model), a checkpoint resumes in a new session and restores into a fresh one, and the generator samples the live adapters", SessionStateAndCheckpoints),
        ("reinforcement text: sampled tokens: off, no chunk carries ids; on, the last chunk of Chat, Stream, ChatBatch and StreamBatch carries the prompt as fed and the ids and log-probabilities the sampler drew, up to the token completing a stop; the texts are the same", SampledTokensRecorded),
        ("reinforcement text: multi-turn: a prefix-stable episode trains as one sequence, its replies alone trained; a template that drops earlier reasoning gives a sequence per turn; auto picks on the ids, episode is refused on unstable ids, and a two-turn run of the policy lays out what its ids say", MultiTurnLayouts),
        ("reinforcement text: a reply ends where its context window does (no slide, its ids following its prompt as fed) and ends its episode, truncated, alone and in a batch (a float32 and a bfloat16 cache), with room for a token or with the prompt cut to its most recent part; the episode trains", ContextWindowEndsEpisode),
        ("reinforcement text: a policy loss through FineTuningLosses.Policy gets the token values, behaviour and reference log-probabilities a step's sequences carry (the reference only when the loss reads it: IPolicyLoss.NeedsReference), and the input gives them all or none", PolicyLossAdapter),
        ("reinforcement text: R2's trainer with grpo (group advantages, a k3 KL to the reference, two epochs a rollout) drives the policy on a prompts task (reward metric, exact) and raises its pass rate on a tiny decoder; the first update's ratios are exactly 1; trained ids are the sampled ids; held device memory does not grow over 40 iterations", PolicyRaisesPassRate),
        ("reinforcement text: R2's trainer with ppo (returns less their mean, mini-batches, no KL so no reference pass) drives the policy on the prompts task and raises its pass rate", PolicyPpoRaisesPassRate),
        ("reinforcement text: on-policy distillation: forward, reverse and jsd match a direct computation on fixed logits (values and gradients through the chunked head); the library's forward by name is the distillation's bit for bit; the registry", DivergenceKinds),
        ("reinforcement text: on-policy distillation: a teacher equal to the student gives divergences of 0 and leaves the weights; a stronger teacher's reverse KL on the student's samples falls alone, with a mix of data answers and as a teacher term beside grpo under R2's trainer; a teacher file and another vocabulary are refused", OnPolicyDistillationLearns),
        ("reinforcement text: the prompts environment passes the environment suite and reads a transcript file; the text rewards (metric, regex, json, length, judge, sum) pass the reward suite and are the library's in RewardFunctions", PromptsAndRewards),
    ];

    private const string ChatMlTemplate =
        "{%- for m in messages %}{{ '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}{%- endfor %}{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{%- endif %}";

    private static FineTuningOptions SessionOptions(float rate = 5e-3f) => new()
    {
        Rank = 4, Alpha = 8, LearningRate = rate, BatchTokens = 512, Seed = 7, CudaGraphs = false, Checkpointing = false, RecomputeFeedForward = false,
        BFloat16Activations = false, Scheduler = LearningRateSchedules.Factory(LearningRateSchedules.Constant, new LearningRateScheduleSettings()),
    };

    private static float[] TrainedValues(PretrainedModel model) => [.. model.Network.TrainableParameters().SelectMany(p => p.ToArray())];

    private static bool SameBits(float[] a, float[] b) => a.Length == b.Length && a.Zip(b).All(p => BitConverter.SingleToInt32Bits(p.First) == BitConverter.SingleToInt32Bits(p.Second));

    private static void SessionMatchesTrain(Device device)
    {
        string folder = WriteChatModel(TinyChatSpec());
        try
        {
            var sequences = RandomSequences(20, 72);
            // One optimizer step per epoch (the accumulation covers every batch): Train's batches are then the session's,
            // drawn from the same seed in the same order.
            var options = new FineTuningOptions
            {
                Rank = 4, Alpha = 8, LearningRate = 3e-3f, BatchTokens = 200, Epochs = 3, GradientAccumulation = 64, Seed = 4, CudaGraphs = false,
                Checkpointing = false, RecomputeFeedForward = false, BFloat16Activations = false,
            };
            foreach (var (name, run, given) in new (string, FineTuningOptions, FineTuningLoss?)[]
                     {
                         ("the fused token cross-entropy", options, null),
                         ("a loss of one's own (FineTuningOptions.Loss)", options with { Loss = FineTuningLosses.TokenCrossEntropy }, null),
                         ("a loss given to each step", options, FineTuningLosses.TokenCrossEntropy),
                     })
            {
                var trained = new List<float>();
                float[] trainedAdapters;
                using (var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
                {
                    FineTuner.Train(model, sequences, null, run with { Loss = run.Loss ?? given }, progress: new SynchronousProgress<FineTuningProgress>(p => trained.Add(p.Loss)));
                    trainedAdapters = TrainedValues(model);
                }

                var stepped = new List<float>();
                float[] steppedAdapters;
                using (var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
                {
                    using var session = new FineTuningSession(model, run, totalSteps: run.Epochs, sample: sequences);
                    for (int epoch = 0; epoch < run.Epochs; epoch++)
                    {
                        var stats = session.Step(sequences, given);
                        stepped.Add(stats.Loss);
                        Check(stats.Step == epoch + 1 && session.Steps == epoch + 1, $"{name}: the step count {stats.Step}");
                    }

                    steppedAdapters = TrainedValues(model);
                }

                Check(trained.Count == run.Epochs && SameBits([.. trained], [.. stepped]),
                    $"{name}: Train's losses {string.Join(", ", trained)} and the session's {string.Join(", ", stepped)} bit for bit");
                Check(SameBits(trainedAdapters, steppedAdapters), $"{name}: the adapters bit for bit");
            }
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    private static void SessionStateAndCheckpoints(Device device)
    {
        string folder = WriteChatModel(TinyChatSpec()), checkpoint = Path.Combine(folder, "checkpoint");
        try
        {
            var sequences = RandomSequences(12, 31);
            float[] before, afterOn, afterOff, resumedLosses, continuedLosses;
            float continuedRate;
            using (var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
            {
                using var session = new FineTuningSession(model, SessionOptions(), totalSteps: 8);
                before = [.. session.LogProbabilities(sequences).SelectMany(v => v)];
                float evaluated = FineTuner.Evaluate(model, sequences);
                Check(Math.Abs(-before.Average() - evaluated) < 1e-4f, $"log-probabilities agree with Evaluate's mean loss: {-before.Average()} against {evaluated}");
                Check(session.LogProbabilities(sequences).Select((v, i) => v.Length == sequences[i].TrainedTokens).All(x => x), "a value per trained token");
                session.Step(sequences);
                session.Step(sequences);
                session.Checkpoint(checkpoint);
                afterOn = [.. session.LogProbabilities(sequences).SelectMany(v => v)];
                afterOff = [.. session.LogProbabilities(sequences, adapters: false).SelectMany(v => v)];
                continuedLosses = [session.Step(sequences).Loss, session.Step(sequences).Loss];
                continuedRate = session.LearningRate;

                // The generator samples the adapters as they are: a step changes what greedy decoding gives the scores of.
                var chat = session.Generator();
                var reply = chat.Chat(new ChatRequest([new ChatMessage("user", "hi")], Options: new GenerationOptions { Temperature = 0f, TopK = 1, NumPredict = 4, ReturnTokens = true }));
                Check(reply.Tokens is { Reply.Length: > 0 }, "the session's generator answers with ids");
            }

            Check(before.Zip(afterOff).All(p => Math.Abs(p.First - p.Second) < 1e-5f), "with the adapters off, the base model's values (the adapters start as the identity)");
            Check(before.Zip(afterOn).Any(p => Math.Abs(p.First - p.Second) > 1e-4f), "with the adapters on, the trained values");

            // A new session resumed from the checkpoint goes on where it was: the adapters, the optimizer state, the steps.
            using (var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
            {
                using var session = new FineTuningSession(model, SessionOptions() with { ResumeFrom = checkpoint }, totalSteps: 8);
                Check(session.Steps == 2 && session.ResumedSteps == 2, $"resumed at step {session.Steps}");
                var values = session.LogProbabilities(sequences).SelectMany(v => v).ToArray();
                Check(SameBits(values, afterOn), "the resumed adapters are the checkpoint's");
                resumedLosses = [session.Step(sequences).Loss, session.Step(sequences).Loss];
                Check(session.LearningRate == continuedRate, "the schedule where it was");
            }

            CloseByNorm(continuedLosses, resumedLosses, 1e-4f, "a resumed session's losses (its batches packed afresh)");

            // Restore into a fresh session of the same model; one that went past the checkpoint is refused.
            using (var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
            {
                using var session = new FineTuningSession(model, SessionOptions(), totalSteps: 8);
                session.Restore(checkpoint);
                Check(session.Steps == 2 && SameBits([.. session.LogProbabilities(sequences).SelectMany(v => v)], afterOn), "restored adapters and steps");
                session.Step(sequences);
                session.Step(sequences);
                session.Step(sequences);
                ThrowsNaming<InvalidOperationException>(() => session.Restore(checkpoint), "a schedule only moves forward", "ResumeFrom");
                ThrowsNaming<ArgumentException>(() => session.Step([]), "an empty step", "at least one sequence");
            }
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    // A sampler that records the ids and probabilities it gave, by step and row.
    private sealed class IdRecordingSampler(ITokenSampler inner) : ITokenSampler
    {
        public readonly SortedDictionary<int, SampledToken[]> Steps = [];

        public int Rows => inner.Rows;

        public int Vocabulary => inner.Vocabulary;

        public Tensor Ids => inner.Ids;

        public bool Recordable => inner.Recordable;

        public void Sample(Tensor logits) => inner.Sample(logits);

        public void SetHistory(IReadOnlyList<int> tokens) => inner.SetHistory(tokens);

        public void Reset() => inner.Reset();

        public SampledToken[][] Read(int fromStep, int toStep)
        {
            var read = inner.Read(fromStep, toStep);
            for (int s = 0; s < read.Length; s++)
            {
                Steps[fromStep + s] = read[s];
            }

            return read;
        }

        public void Dispose() => inner.Dispose();

        public int[] Row(int row) => [.. Steps.Values.Select(s => s[row].Id)];

        public float[] Logs(int row) => [.. Steps.Values.Select(s => MathF.Log(s[row].Probability))];

        public static Func<SamplerRequest, ITokenSampler> Into(List<IdRecordingSampler> made) => request =>
        {
            var sampler = new IdRecordingSampler(TokenSamplers.Create(request));
            made.Add(sampler);
            return sampler;
        };
    }

    private static void SampledTokensRecorded(Device device)
    {
        string folder = WriteChatModelWith(TinyChatSpec(), ChatMlTemplate);
        try
        {
            using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
            var chat = model.CreateChat();
            chat.Generator.KeepCache = false;
            var made = new List<IdRecordingSampler>();
            chat.Generator.CreateSampler = IdRecordingSampler.Into(made);
            var tokenizer = model.Tokenizer!;
            string[] vowels = ["a", "e", "i", "o", "u", "s", "t"];                  // frequent stops, so some replies stop early
            var off = new GenerationOptions { Temperature = 1f, TopK = 0, TopP = 1f, RepeatPenalty = 1f, Seed = 5, NumPredict = 24, Stop = vowels };
            var on = off with { ReturnTokens = true, ReturnLogProbabilities = true };
            var requests = Enumerable.Range(0, 4).Select(i => new ChatRequest([new ChatMessage("user", $"question {i}")], Options: on)).ToList();
            var offRequests = requests.Select(r => r with { Options = off }).ToList();
            int stops = 0;

            void Verify(ChatChunk reply, ChatRequest request, IdRecordingSampler sampler, int row, string what)
            {
                var tokens = reply.Tokens ?? throw new InvalidOperationException($"{what}: no ids");
                Check(tokens.Prompt.SequenceEqual(tokenizer.Encode(chat.RenderPrompt(request))), $"{what}: the prompt ids as fed");
                int[] drawn = sampler.Row(row);
                Check(tokens.Reply.Length >= 1 && tokens.Reply.Length <= drawn.Length && drawn.AsSpan(0, tokens.Reply.Length).SequenceEqual(tokens.Reply),
                    $"{what}: the reply ids are the sampler's ({tokens.Reply.Length} of {drawn.Length})");
                Check(tokens.LogProbabilities is { } logs && logs.SequenceEqual(sampler.Logs(row).Take(tokens.Reply.Length)), $"{what}: the sampler's log-probabilities");
                if (reply.DoneReason == "stop")
                {
                    stops++;
                    string text = tokenizer.Decode(tokens.Reply), shorter = tokenizer.Decode(tokens.Reply[..^1]);
                    var all = vowels.Concat(chat.Template.StopSequences).ToArray();
                    Check(all.Any(s => text.Contains(s, StringComparison.Ordinal)) && !all.Any(s => shorter.Contains(s, StringComparison.Ordinal)),
                        $"{what}: the ids end with the token completing the stop: '{text}'");
                }
            }

            // Chat and Stream (one prompt).
            made.Clear();
            var single = chat.Chat(requests[0]);
            Verify(single, requests[0], made[^1], 0, "Chat");
            made.Clear();
            var streamed = chat.Stream(requests[0]).ToList();
            Check(streamed.Take(streamed.Count - 1).All(c => c.Tokens is null) && streamed[^1].Tokens is not null, "Stream: the ids on the last chunk only");
            Verify(streamed[^1], requests[0], made[^1], 0, "Stream");
            var plain = chat.Chat(offRequests[0]);
            Check(plain.Tokens is null && plain.Message!.Content == single.Message!.Content, "off: no ids, the same text");

            // ChatBatch and StreamBatch (prompts together).
            made.Clear();
            var batch = chat.ChatBatch(requests);
            for (int i = 0; i < requests.Count; i++)
            {
                Verify(batch[i], requests[i], made[^1], i, $"ChatBatch {i}");
            }

            var plainBatch = chat.ChatBatch(offRequests);
            Check(plainBatch.All(c => c.Tokens is null) && plainBatch.Select(c => c.Message!.Content).SequenceEqual(batch.Select(c => c.Message!.Content)),
                "ChatBatch off: no ids, the same texts");
            made.Clear();
            var finals = chat.StreamBatch(requests).Where(c => c.Chunk.Done).OrderBy(c => c.Index).ToList();
            for (int i = 0; i < requests.Count; i++)
            {
                Verify(finals[i].Chunk, requests[i], made[^1], i, $"StreamBatch {i}");
            }

            Check(stops > 0, "some replies stopped at a stop sequence (the ids' end checked)");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    private static ActionRecord Turn(int[] prompt, int[] reply) => new(reply, [.. reply.Select(_ => -1f)], float.NaN, prompt);

    // An environment of two turns: a question, then the same question again after the first reply; rewarded at the end.
    private sealed class TwoTurns : IEnvironment<ChatObservation, ChatMessage>
    {
        private List<ChatMessage> _messages = [];
        private int _turn = -1;

        public Space ObservationSpace => Space.Conversation;

        public Space ActionSpace => Space.Conversation;

        public ChatObservation Reset(int seed)
        {
            _messages = [new ChatMessage("user", "Q")];
            _turn = 0;
            return new([.. _messages]);
        }

        public ValueTask<EnvironmentStep<ChatObservation>> StepAsync(ChatMessage action, CancellationToken cancellationToken = default)
        {
            if (_turn < 0)
            {
                throw new InvalidOperationException("reset first");
            }

            _messages.Add(action);
            if (++_turn == 1)
            {
                _messages.Add(new ChatMessage("user", "Q"));
                return ValueTask.FromResult(new EnvironmentStep<ChatObservation>(new([.. _messages]), 0f, false, false));
            }

            _turn = -1;
            return ValueTask.FromResult(new EnvironmentStep<ChatObservation>(new([.. _messages]), action.Content == "yes" ? 1f : 0f, true, false));
        }

        public void Dispose()
        {
        }
    }

    private static void MultiTurnLayouts(Device device)
    {
        string folder = WriteChatModelWith(TinyChatSpec(), ChatMlTemplate);
        try
        {
            using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
            var tokenizer = model.Tokenizer!;
            int[] Ids(string text) => [.. tokenizer.Encode(text)];

            // The ids of two turns under a template: turn 1's prompt and sampled reply, then turn 2's prompt rendered from the
            // conversation with the reply parsed back (its reasoning apart).
            IReadOnlyList<ActionRecord> Episode(IChatTemplate template)
            {
                string replyText = "<think>\nr\n</think>\n\nA<|im_end|>";
                var first = new List<ChatMessage> { new("user", "Q") };
                var second = new List<ChatMessage>(first) { new("assistant", "A", Thinking: "r"), new("user", "Q2") };
                return [Turn(Ids(template.Render(first, [], null)), Ids(replyText)), Turn(Ids(template.Render(second, [], null)), Ids("B<|im_end|>"))];
            }

            // ChatML renders every turn as it was written: prefix-stable when the reply's text is what the template writes back.
            var chatMl = model.ChatTemplate!;
            var stableTurns = new List<ChatMessage> { new("user", "Q") };
            int[] p1 = Ids(chatMl.Render(stableTurns, [], null)), r1 = Ids("A<|im_end|>");
            stableTurns.Add(new ChatMessage("assistant", "A"));
            stableTurns.Add(new ChatMessage("user", "Q2"));
            int[] p2 = Ids(chatMl.Render(stableTurns, [], null)), r2 = Ids("B<|im_end|>");
            IReadOnlyList<ActionRecord> stable = [Turn(p1, r1), Turn(p2, r2)];
            Check(EpisodeTurns.PrefixStable(stable), "ChatML: the second prompt begins with the first prompt and reply");
            var one = EpisodeTurns.Lay(stable);
            Check(one.Mode == EpisodeTurns.Episode && one.Sequences.Count == 1, $"auto picks episode: {one.Mode}");
            var sequence = one.Sequences[0];
            int[] trained = [.. sequence.Tokens.Where((_, i) => sequence.Trained[i])];
            Check(sequence.Tokens.SequenceEqual(p2.Concat(r2)) && trained.SequenceEqual(r1.Concat(r2)) && sequence.TrainedTokens == r1.Length + r2.Length,
                "one sequence: the last prompt and reply, the two replies' ids alone trained");
            var perTurn = EpisodeTurns.Lay(stable, "TURN");
            Check(perTurn.Mode == EpisodeTurns.Turn && perTurn.Sequences.Count == 2 && perTurn.Sequences[0].Tokens.SequenceEqual(p1.Concat(r1)), "turn asked for: a sequence per turn");

            // Qwen3's template drops an earlier turn's reasoning: not prefix-stable, so a sequence per turn.
            var qwen = new JinjaChatTemplate(Qwen3Template, ["<|im_end|>"]);
            var dropping = Episode(qwen);
            Check(!EpisodeTurns.PrefixStable(dropping), "Qwen3: the second prompt drops the first reply's reasoning");
            var layout = EpisodeTurns.Lay(dropping);
            Check(layout.Mode == EpisodeTurns.Turn && layout.Sequences.Count == 2
                  && layout.Sequences.Select((s, i) => s.Tokens.Where((_, t) => s.Trained[t]).SequenceEqual(dropping[i].Tokens)).All(x => x), "auto picks turn, each reply trained on its own prompt");
            ThrowsNaming<InvalidOperationException>(() => EpisodeTurns.Lay(dropping, EpisodeTurns.Episode), "episode on unstable ids", "not prefix-stable", "turn");
            ThrowsNaming<ArgumentException>(() => EpisodeTurns.Lay(stable, "every"), "an unknown layout", "auto, episode, turn", EpisodeTurns.EnvironmentVariable);

            // The policy over two turns of an environment: the layout follows its ids, the replies alone are trained.
            using var policy = new LanguageModelPolicy(new FineTuningSession(model, SessionOptions(), totalSteps: 4), new LanguageModelPolicyOptions { MaxNewTokens = 6 });
            var environments = Enumerable.Range(0, 4).Select(_ => new TwoTurns()).ToList();
            var observations = environments.Select((e, i) => e.Reset(i)).ToList();
            var history = observations.Select(o => new List<ChatObservation> { o }).ToList();
            var records = environments.Select(_ => new List<ActionRecord>()).ToList();
            var actions = environments.Select(_ => new List<ChatMessage>()).ToList();
            var rewards = environments.Select(_ => new List<float>()).ToList();
            for (int turn = 0; turn < 2; turn++)
            {
                var acted = policy.ActAsync(observations, new ActionSampling { Seed = turn }).AsTask().GetAwaiter().GetResult();
                for (int e = 0; e < environments.Count; e++)
                {
                    var step = environments[e].StepAsync(acted.Actions[e]).AsTask().GetAwaiter().GetResult();
                    observations[e] = step.Observation;
                    history[e].Add(step.Observation);
                    records[e].Add(acted.Records[e]);
                    actions[e].Add(acted.Actions[e]);
                    rewards[e].Add(step.Reward);
                }
            }

            using var batch = new RolloutBatch<ChatObservation, ChatMessage>(Enumerable.Range(0, environments.Count)
                .Select(e => new Trajectory<ChatObservation, ChatMessage>(e, e, history[e], actions[e], records[e], rewards[e], true, false)));
            var layouts = policy.Lay(batch);
            for (int e = 0; e < layouts.Count; e++)
            {
                bool expected = EpisodeTurns.PrefixStable(records[e]);
                Check(layouts[e].Mode == (expected ? EpisodeTurns.Episode : EpisodeTurns.Turn), $"episode {e}: {layouts[e].Mode} for ids that are {(expected ? "" : "not ")}prefix-stable");
                var trainedIds = layouts[e].Sequences.SelectMany(s => s.Tokens.Where((_, t) => s.Trained[t]));
                Check(trainedIds.SequenceEqual(records[e].SelectMany(r => r.Tokens)), $"episode {e}: the replies' sampled ids alone are trained");
            }

            var stats = policy.Update(batch, PolicyLosses.Get("grpo"));
            Check(float.IsFinite(stats.Loss) && stats.Episodes == 4 && stats.Units == batch.Trajectories.Sum(t => t.Units), $"an update over the two-turn episodes: {stats}");
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    // TwoTurns as an environment kind, for the rollout collector.
    private sealed class TwoTurnsKind : IEnvironmentKind
    {
        public string Name => "reinforcement-text-test-two-turns";

        public string Summary => "A question, then the same question again; rewarded at the end.";

        public IReadOnlySet<string> Keys => OptionKeys.None;

        public Type ObservationType => typeof(ChatObservation);

        public Type ActionType => typeof(ChatMessage);

        public IEnvironment Create(EnvironmentRequest request) => new TwoTurns();
    }

    private static void ContextWindowEndsEpisode(Device device)
    {
        string folder = WriteChatModelWith(TinyChatSpec(), ChatMlTemplate);
        try
        {
            IReadOnlyList<ChatMessage> question = [new ChatMessage("user", "Q")];
            int[] full;
            using (var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
            using (var probe = new LanguageModelPolicy(new FineTuningSession(model, SessionOptions(), totalSteps: 1), new LanguageModelPolicyOptions()))
            {
                full = [.. model.Tokenizer!.Encode(probe.Chat.RenderPrompt(new ChatRequest(question)))];
            }

            // Room for a token of reply after the first prompt, then a window the prompt no longer fits (its most recent part
            // kept); an episode alone (the generator's single path) and four together (its batch path).
            foreach (int window in new[] { full.Length + 1, full.Length - 2 })
            {
                foreach (var (episodes, cache) in new[] { (1, "float32"), (4, "float32"), (4, "bfloat16") })
                {
                    using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
                    using var policy = new LanguageModelPolicy(new FineTuningSession(model, SessionOptions(), totalSteps: 1),
                        new LanguageModelPolicyOptions { MaxNewTokens = 6, ContextLength = window, CacheFormat = cache });
                    using var collector = new RolloutCollector<ChatObservation, ChatMessage>(new TwoTurnsKind(), environments: 1, parallelSteps: false);
                    using var batch = collector.CollectAsync(policy, [.. Enumerable.Range(0, episodes).Select(k => new EpisodeStart(k, k, k))], new ActionSampling { Seed = 3 })
                        .AsTask().GetAwaiter().GetResult();
                    string label = $"a {window}-token window, a {full.Length}-token prompt, {episodes} episode{(episodes == 1 ? "" : "s")}, a {cache} cache";
                    int[] kept = full.Length < window ? full : full[^(window - 1)..];
                    foreach (var trajectory in batch.Trajectories)
                    {
                        Check(trajectory.Records.Count == 1 && trajectory.Truncated && !trajectory.Terminated,
                            $"{label}: the reply that filled the window ended its episode, truncated ({trajectory.Records.Count} turns, truncated {trajectory.Truncated}, terminated {trajectory.Terminated})");
                        var record = trajectory.Records[0];
                        Check(record.EndsEpisode && record.Tokens.Length == 1 && record.Context!.SequenceEqual(kept),
                            $"{label}: the prompt as fed ({record.Context!.Length} ids, {kept.Length} expected) and a token's reply ({record.Tokens.Length}) fill the window");
                    }

                    var stats = policy.Update(batch, PolicyLosses.Get("grpo"));
                    Check(float.IsFinite(stats.Loss) && stats.Units == batch.Trajectories.Sum(t => t.Units), $"{label}: the truncated episodes train: {stats}");
                }
            }
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    // A library policy loss under a test name, recording what it was given: the largest |log p − behaviour| of its first
    // call, and whether any call had the reference's log-probabilities. It reads the reference when the library's does.
    private sealed class RecordingLoss(string name, IPolicyLoss inner) : IPolicyLoss
    {
        public float? FirstGap { get; private set; }

        public bool SawReference { get; private set; }

        public string Name => name;

        public string Summary => $"{inner.Name}, recorded (the reinforcement text tests)";

        public IReadOnlySet<string> Keys => inner.Keys;

        public bool NeedsReference(IReadOnlyDictionary<string, string>? options = null) => inner.NeedsReference(options);

        public Tensor Loss(PolicyLossInput input, IReadOnlyDictionary<string, string>? options = null)
        {
            FirstGap ??= input.LogProbabilities.ToArray().Zip(input.BehaviorLogProbabilities.ToArray()).Max(p => Math.Abs(p.First - p.Second));
            SawReference |= input.ReferenceLogProbabilities is not null;
            return inner.Loss(input, options);
        }
    }

    // A loss that does not say whether it reads the reference: it is given it (IPolicyLoss.NeedsReference's default).
    private sealed class SilentLoss : IPolicyLoss
    {
        public string Name => "reinforcement-text-test-silent";

        public string Summary => "a loss that does not answer NeedsReference";

        public IReadOnlySet<string> Keys => OptionKeys.None;

        public Tensor Loss(PolicyLossInput input, IReadOnlyDictionary<string, string>? options = null) => input.LogProbabilities.Sum();
    }

    private static void PolicyLossAdapter(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                              // the losses run on the CPU on every device
        }

        int[] counts = [3, 2, 4];
        var random = new Random(3);
        float[] Values(int n, float scale, float shift) => [.. Enumerable.Range(0, n).Select(_ => shift + scale * (random.NextSingle() - 0.5f))];
        float[] logp = Values(9, 1f, -1.5f), behavior = Values(9, 1f, -1.5f), reference = Values(9, 1f, -1.5f), advantages = Values(9, 2f, 0f);
        var grpo = PolicyLosses.Get("grpo");
        var kl = new Dictionary<string, string> { ["kl"] = "0.1" };

        // Whether a loss reads the reference: grpo and ppo when their KL is weighted, reinforce never, a loss that does not say always.
        Check(grpo.NeedsReference(kl) && !grpo.NeedsReference() && !grpo.NeedsReference(new Dictionary<string, string> { ["kl"] = "0" })
              && PolicyLosses.Get("ppo").NeedsReference(new Dictionary<string, string> { ["kl"] = "0.04" }) && !PolicyLosses.Get("ppo").NeedsReference()
              && !PolicyLosses.Get("reinforce").NeedsReference(), "NeedsReference: the KL's weight decides");
        Check(((IPolicyLoss)new SilentLoss()).NeedsReference(), "a loss that does not answer is given the reference");

        // The adapter's loss and gradient equal the policy loss's on the same values.
        var leaf = Tensor.From(logp, [9], Device.Cpu, requiresGrad: true);
        var input = new FineTuningLossInput(leaf, counts, Tensor.From(reference, [9], Device.Cpu), stepTokens: 12, tokenValues: Tensor.From(advantages, [9], Device.Cpu),
            behaviorLogProbabilities: Tensor.From(behavior, [9], Device.Cpu));
        var adapted = FineTuningLosses.Policy(grpo, kl)(input);
        adapted.Backward();
        var direct = Tensor.From(logp, [9], Device.Cpu, requiresGrad: true);
        var expected = grpo.Loss(new PolicyLossInput(direct, counts, Tensor.From(advantages, [9], Device.Cpu), Tensor.From(behavior, [9], Device.Cpu),
            Tensor.From(reference, [9], Device.Cpu), stepUnits: 12), kl);
        expected.Backward();
        AssertClose([expected.Item()], [adapted.Item()], 1e-6f, "the policy loss through the adapter");
        AssertClose(direct.Grad!.ToArray(), leaf.Grad!.ToArray(), 1e-6f, "its gradient");

        // A loss that does not read the reference (grpo without a KL) is not given it, though the input has one.
        var recorder = new RecordingLoss("reinforcement-text-test-adapter", grpo);
        var noReference = FineTuningLosses.Policy(recorder)(new FineTuningLossInput(Tensor.From(logp, [9], Device.Cpu), counts, Tensor.From(reference, [9], Device.Cpu),
            tokenValues: Tensor.From(advantages, [9], Device.Cpu)));
        var expectedNoReference = grpo.Loss(new PolicyLossInput(Tensor.From(logp, [9], Device.Cpu), counts, Tensor.From(advantages, [9], Device.Cpu)));
        AssertClose([expectedNoReference.Item()], [noReference.Item()], 1e-6f, "on-policy, no reference");
        Check(!recorder.SawReference, "the reference is left out for a loss that does not read it");
        _ = FineTuningLosses.Policy(recorder, kl)(new FineTuningLossInput(Tensor.From(logp, [9], Device.Cpu), counts, Tensor.From(reference, [9], Device.Cpu),
            tokenValues: Tensor.From(advantages, [9], Device.Cpu)));
        Check(recorder.SawReference, "and given to one that reads it");

        // From the sequences: a value each (repeated over its tokens) or token values, behaviour log-probabilities all or none.
        TrainingSequence Sequence(int trained, float? value = null, float[]? tokens = null, float[]? behaviour = null) =>
            new([.. Enumerable.Range(0, trained + 1)], [false, .. Enumerable.Repeat(true, trained)]) { Value = value, TokenValues = tokens, BehaviorLogProbabilities = behaviour };
        var values = new FineTuningLossInput(Tensor.From(logp[..5], [5], Device.Cpu), [3, 2], [Sequence(3, 0.5f), Sequence(2, tokens: [1f, 2f])], null, false, 0, 0);
        Check(values.TokenValues.ToArray().SequenceEqual([0.5f, 0.5f, 0.5f, 1f, 2f]) && !values.HasBehavior, "a value repeated, token values as given, no behaviour");
        ThrowsNaming<InvalidOperationException>(() => _ = values.SequenceValues, "a sequence without a value", "TrainingSequence.Value");
        var mixed = new FineTuningLossInput(Tensor.From(logp[..5], [5], Device.Cpu), [3, 2], [Sequence(3, 1f, behaviour: [-1f, -1f, -1f]), Sequence(2, 1f)], null, false, 0, 0);
        ThrowsNaming<InvalidOperationException>(() => _ = mixed.BehaviorLogProbabilities, "behaviour for some sequences only", "every sequence or none");
        var both = new FineTuningLossInput(Tensor.From(logp[..5], [5], Device.Cpu), [3, 2], [Sequence(3, 1f, behaviour: [-1f, -2f, -3f]), Sequence(2, 2f, behaviour: [-4f, -5f])], null, false, 0, 0);
        Check(both.BehaviorLogProbabilities!.ToArray().SequenceEqual([-1f, -2f, -3f, -4f, -5f]) && both.SequenceValues.ToArray().SequenceEqual([1f, 2f]), "behaviour and sequence values joined in order");
    }

    // One prompt, its sequences for answer `answer` (the prompt as the template renders it, the answer then the end of turn).
    private static TrainingSequence AnswerSequence(PretrainedModel model, IReadOnlyList<ChatMessage> prompt, string answer)
    {
        int[] p = [.. model.Tokenizer!.Encode(model.ChatTemplate!.Render(prompt, [], null))], a = [.. model.Tokenizer.Encode(answer + "<|im_end|>")];
        return new TrainingSequence([.. p, .. a], [.. p.Select(_ => false), .. a.Select(_ => true)]);
    }

    private static void PolicyRaisesPassRate(Device device) => TrainerRaisesPassRate(device, "grpo", seed: 5, iterations: 40);

    private static void PolicyPpoRaisesPassRate(Device device) => TrainerRaisesPassRate(device, "ppo", seed: 9, iterations: 24);

    // R2's trainer driving the policy on a one-prompt task (exact match with "yes"), the loss a library one under a test
    // name (so what it is given is recorded): grpo with group advantages, a k3 KL of 0.01 to the reference and two epochs a
    // rollout; ppo with returns less their mean, two epochs of two mini-batches, and no KL (so no reference pass). The
    // pass rate sampled from 16 episodes is noisy, so it is averaged over 5 iterations. Measured on the CPU over seeds 1 to
    // 8 (R4): grpo in 40 iterations from 0.35-0.45 to 0.57-0.69, ppo in 24 from 0.31-0.42 to 0.54-0.69; the bar, a rise
    // of more than 0.1 to at least 0.5, held on all 16. (The probability of the exact string "yes" is no measure: one ppo
    // run passing half its episodes had moved to another spelling the metric's normalization accepts.)
    private static void TrainerRaisesPassRate(Device device, string algorithm, int seed, int iterations)
    {
        string folder = WriteChatModelWith(TinyChatSpec(), ChatMlTemplate);
        bool grpo = algorithm == "grpo";
        var loss = new RecordingLoss($"reinforcement-text-test-{algorithm}", PolicyLosses.Get(algorithm));
        PolicyLosses.Register(loss);
        try
        {
            using var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
            var session = new FineTuningSession(model, SessionOptions(1e-2f), totalSteps: 200);
            using var policy = new LanguageModelPolicy(session, new LanguageModelPolicyOptions { MaxNewTokens = 6 });
            IReadOnlyList<ChatMessage> prompt = [new ChatMessage("user", "Q")];

            // A warm-up on both answers, so the model says yes or no about as often: the reward then has something to choose.
            var warm = Enumerable.Range(0, 8).Select(i => AnswerSequence(model, prompt, i % 2 == 0 ? "yes" : "no")).ToList();
            for (int i = 0; i < 40; i++)
            {
                session.Step(warm);
            }

            var made = new List<IdRecordingSampler>();
            policy.Chat.Generator.CreateSampler = IdRecordingSampler.Into(made);
            var kind = new PromptsEnvironmentKind([new PromptTask(prompt, "yes")], name: "reinforcement-text-test-prompts");
            const int Group = 16;
            var options = new ReinforcementOptions
            {
                Algorithm = loss.Name, Estimator = grpo ? "group" : "returns", GroupSize = grpo ? Group : 1, EpisodesPerIteration = Group, Iterations = iterations,
                Epochs = 2, MiniBatch = grpo ? 0 : Group / 2, KlWeight = grpo ? 0.01f : 0f, Seed = seed, Environments = 1, ParallelSteps = false,
                EnvironmentOptions = new Dictionary<string, string> { ["reward"] = "metric", ["metric.metric"] = "exact" },
                EstimatorOptions = grpo ? null : new Dictionary<string, string> { ["baseline"] = "mean" },
            };
            long heldAfterSecond = 0, heldAtEnd = 0;
            var trainer = new ReinforcementTrainer<ChatObservation, ChatMessage>(kind, policy, options)
            {
                OnBatch = (iteration, batch) =>
                {
                    if (iteration != 0)
                    {
                        return;
                    }

                    // Trained on exactly the ids sampled: the prompt as fed and the sampler's ids.
                    var layouts = policy.Lay(batch);
                    for (int i = 0; i < batch.Episodes; i++)
                    {
                        var trajectory = batch.Trajectories[i];
                        var record = trajectory.Records[0];
                        Check(record.Context!.SequenceEqual(model.Tokenizer!.Encode(policy.Chat.RenderPrompt(new ChatRequest(trajectory.Observations[0].Messages)))),
                            $"episode {i}: the prompt ids as rendered");
                        Check(made[^1].Row(i).Take(record.Tokens.Length).SequenceEqual(record.Tokens), $"episode {i}: the sampler's ids");
                        Check(layouts[i].Sequences.Single().Tokens.SequenceEqual(record.Context!.Concat(record.Tokens)), $"episode {i}: the sequence trained is the prompt and the sampled ids");
                    }
                },
                OnIteration = summary =>
                {
                    long held = model.Device.Backend.GetMemoryUsage().InUse;
                    if (summary.Iteration == 2)
                    {
                        heldAfterSecond = held;
                    }

                    heldAtEnd = held;
                },
            };
            var result = trainer.Train();
            Check(loss.FirstGap == 0f, $"the first update's ratios are exactly 1 (behaviour recomputed by the training forward): largest |log p − behaviour| {loss.FirstGap}");
            Check(loss.SawReference == grpo, $"the reference's log-probabilities {(grpo ? "given to grpo's KL" : "not computed for ppo without a KL")}");
            var passRates = result.History.Select(h => h.RewardMean).ToList();
            Check(passRates.Count == iterations && result.History.All(h => h.Updates == (grpo ? 2 : 4)), $"{passRates.Count} iterations of {result.History[0].Updates} updates");
            double first = passRates.Take(5).Average(), last = passRates.TakeLast(5).Average();
            Check(last >= 0.5 && last > first + 0.1, string.Create(CultureInfo.InvariantCulture,
                $"{algorithm}: the pass rate rises ({first:F2} to {last:F2} over five iterations): {string.Join(", ", passRates.Select(p => p.ToString("F2", CultureInfo.InvariantCulture)))}"));
            Check(heldAtEnd <= heldAfterSecond + (1L << 20), $"device memory held after iteration 2 ({heldAfterSecond} bytes) does not grow by iteration {iterations} ({heldAtEnd} bytes)");
        }
        finally
        {
            PolicyLosses.Unregister(loss.Name);
            Directory.Delete(folder, true);
        }
    }

    private static void DivergenceKinds(Device device)
    {
        const int Positions = 7, Dim = 10, Vocabulary = 17;
        var random = new Random(51);
        float[] Values(int n, float scale) => [.. Enumerable.Range(0, n).Select(_ => (random.NextSingle() * 2f - 1f) * scale)];
        float[] h = Values(Positions * Dim, 1f), w = Values(Dim * Vocabulary, 0.8f);
        int[] rows = [0, 2, 3, 5, 6, 2];
        float[] teacherLogits = Values(rows.Length * Vocabulary, 3f), weights = Values(rows.Length, 1f);
        const float Temperature = 1.5f;
        var q = new double[rows.Length * Vocabulary];
        for (int r = 0; r < rows.Length; r++)
        {
            double max = Enumerable.Range(0, Vocabulary).Max(v => teacherLogits[r * Vocabulary + v] / Temperature);
            var e = Enumerable.Range(0, Vocabulary).Select(v => Math.Exp(teacherLogits[r * Vocabulary + v] / Temperature - max)).ToArray();
            q[r * Vocabulary + 3] = 0;                                         // a zero of q, read as 1e-30 inside the logarithms
            for (int v = 0; v < Vocabulary; v++)
            {
                q[r * Vocabulary + v] = v == 3 ? 0 : e[v] / (e.Sum() - e[3]);
            }
        }

        // The divergence of one row from its logits z, in double precision, as the registry's documentation writes it.
        static double Divergence(string kind, double[] z, double[] qr, double temperature, double beta)
        {
            double max = z.Max() / temperature, sum = z.Sum(x => Math.Exp(x / temperature - max));
            double[] logp = [.. z.Select(x => x / temperature - max - Math.Log(sum))];
            double Log(double x) => Math.Log(Math.Max(x, 1e-30));
            double d = 0;
            for (int v = 0; v < z.Length; v++)
            {
                double p = Math.Exp(logp[v]), m = beta * qr[v] + (1 - beta) * p;
                d += kind switch
                {
                    "forward" => qr[v] * (Log(qr[v]) - logp[v]),
                    "reverse" => p * (logp[v] - Log(qr[v])),
                    _ => beta * qr[v] * (Log(qr[v]) - Log(m)) + (1 - beta) * p * (logp[v] - Log(m)),
                };
            }

            return d;
        }

        foreach (var (kind, beta) in new[] { ("forward", 0f), ("reverse", 0f), ("jsd", 0.3f) })
        {
            var expected = new float[rows.Length];
            var dz = new double[rows.Length * Vocabulary];
            for (int r = 0; r < rows.Length; r++)
            {
                var z = new double[Vocabulary];
                for (int v = 0; v < Vocabulary; v++)
                {
                    for (int d = 0; d < Dim; d++)
                    {
                        z[v] += h[rows[r] * Dim + d] * w[d * Vocabulary + v];
                    }
                }

                double[] qr = q[(r * Vocabulary)..((r + 1) * Vocabulary)];
                expected[r] = (float)Divergence(kind, z, qr, Temperature, beta);
                for (int v = 0; v < Vocabulary; v++)
                {
                    // Central differences: the gradient checked against the divergence itself, not against its formula.
                    double saved = z[v];
                    z[v] = saved + 1e-5;
                    double up = Divergence(kind, z, qr, Temperature, beta);
                    z[v] = saved - 1e-5;
                    double down = Divergence(kind, z, qr, Temperature, beta);
                    z[v] = saved;
                    dz[r * Vocabulary + v] = weights[r] * (up - down) / 2e-5;
                }
            }

            var expectedHidden = new float[Positions * Dim];
            for (int r = 0; r < rows.Length; r++)
            {
                for (int d = 0; d < Dim; d++)
                {
                    for (int v = 0; v < Vocabulary; v++)
                    {
                        expectedHidden[rows[r] * Dim + d] += (float)(dz[r * Vocabulary + v] * w[d * Vocabulary + v]);
                    }
                }
            }

            using var scope = new TensorScope();
            var hidden = Tensor.From(h, [Positions, Dim], device, requiresGrad: true);
            var weight = Tensor.From(w, [Dim, Vocabulary], device);
            float[] qf = [.. q.Select(v => (float)v)];
            Tensor Teacher(int start, int count) => Tensor.From(qf.AsSpan(start * Vocabulary, count * Vocabulary), [count, Vocabulary], device);
            var divergence = TeacherDivergences.Get(kind);
            TokenDivergenceRule rule = (s, t, g) => divergence.Rows(s, t, beta, g);
            var values = Tensor.TokenDivergences(hidden, x => x.MatMul(weight), rows, Teacher, Temperature, 4, rule);
            AssertClose(expected, values, 1e-4f, $"{kind}: the divergences");
            var lossTensor = Tensor.TokenDivergenceRows(hidden, x => x.MatMul(weight), rows, weights, Teacher, Temperature, 4, rule);
            AssertClose([(float)expected.Select((v, i) => (double)v * weights[i]).Sum()], [lossTensor.Item()], 1e-4f, $"{kind}: the weighted sum");
            lossTensor.Backward();
            CloseByNorm(expectedHidden, hidden.Grad!.ToArray(), 1e-3f, $"{kind}: the hidden gradient against central differences");
            if (kind == "forward")
            {
                var built = Tensor.TokenDivergences(hidden, x => x.MatMul(weight), rows, Teacher, Temperature, 4);
                Check(SameBits(built, values), "the library's forward KL through the registry is the distillation's own, bit for bit");
            }
        }

        // The registry: the library's three, refusals naming what is registered, an app's over a library name and back.
        Check(TeacherDivergences.Names.SequenceEqual(["forward", "reverse", "jsd"]) && TeacherDivergences.Origin("REVERSE") == Idrak.Abstraction.Overrides.Library,
            $"the library's divergences: {string.Join(", ", TeacherDivergences.Names)}");
        ThrowsNaming<NotSupportedException>(() => TeacherDivergences.Get("hellinger"), "an unknown divergence", "'hellinger' is not registered", "forward, reverse, jsd");
        ThrowsNaming<ArgumentOutOfRangeException>(() => FineTuningLosses.OnPolicyDistillation("jsd", beta: 1f), "jsd's beta", "(0, 1)");
        var app = new AppDivergence();
        TeacherDivergences.Register(app);
        try
        {
            Check(TeacherDivergences.Find("reverse") is { } found && found.Summary == app.Summary && TeacherDivergences.Default("reverse") is { } library && library != found,
                "an app's divergence over the library's, the library's behind it");
        }
        finally
        {
            TeacherDivergences.Unregister("reverse");
        }

        Check(TeacherDivergences.Find("reverse") == TeacherDivergences.Default("reverse"), "unregistered: the library's again");
    }

    private sealed class AppDivergence : ITeacherDivergence
    {
        public string Name => "reverse";

        public string Summary => "the app's reverse KL (the reinforcement text tests)";

        public bool UsesBeta => false;

        public (Tensor Values, Tensor? Gradient) Rows(Tensor studentLogProbabilities, Tensor teacherProbabilities, float beta, bool gradient) =>
            TeacherDivergences.Default("reverse")!.Rows(studentLogProbabilities, teacherProbabilities, beta, gradient);
    }

    // A teacher of another vocabulary (its fingerprint differs).
    private sealed class OtherVocabularyTeacher : DistillationTeacher
    {
        public override int Vocabulary => 260;

        public override int TopK => 0;

        public override string? Fingerprint => "17-0000000000000000";

        public override TeacherDistributions Distributions(IReadOnlyList<TrainingSequence> sequences) => throw new NotSupportedException();
    }

    // The student's reverse KL on its own samples (a Monte Carlo estimate: the mean over replies of Σ log p − log q at
    // the ids it sampled), with the teacher's values from a session on the teacher.
    private static double ReverseKlOnSamples(LanguageModelPolicy student, FineTuningSession teacher, IReadOnlyList<ChatMessage> prompt, int seed)
    {
        var acted = student.ActAsync([.. Enumerable.Range(0, 16).Select(_ => new ChatObservation(prompt))], new ActionSampling { Seed = seed }).AsTask().GetAwaiter().GetResult();
        var sequences = acted.Records.Select(r => EpisodeTurns.Lay([r]).Sequences[0]).ToList();
        var p = student.Session.LogProbabilities(sequences);
        var q = teacher.LogProbabilities(sequences);
        return sequences.Select((_, i) => p[i].Zip(q[i]).Sum(x => (double)x.First - x.Second)).Average();
    }

    private static void OnPolicyDistillationLearns(Device device)
    {
        string folder = WriteChatModelWith(TinyChatSpec(), ChatMlTemplate);
        try
        {
            IReadOnlyList<ChatMessage> prompt = [new ChatMessage("user", "Q")];
            var data = new List<ChatTranscript> { new([.. prompt, new ChatMessage("assistant", "yes")], []) };

            // A teacher equal to the student: every divergence 0, and (with plain gradient descent) the weights stay put.
            using (var student = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
            using (var same = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
            using (var teacher = DistillationTeachers.FromModel(same))
            {
                using var session = new FineTuningSession(student, SessionOptions() with { Teacher = teacher, Optimizer = ps => new Idrak.Abstraction.Training.Sgd(ps, 0.1f) }, totalSteps: 3);
                var start = TrainedValues(student);
                foreach (string kind in new[] { "forward", "reverse", "jsd" })
                {
                    var losses = OnPolicyDistiller.Train(session, data, new OnPolicyDistillationOptions { Steps = 1, PromptsPerStep = 4, MaxNewTokens = 6, Divergence = kind, Seed = 1 });
                    Check(losses.All(l => Math.Abs(l) < 1e-6f), $"{kind} from an equal teacher: {string.Join(", ", losses)}");
                }

                float moved = start.Zip(TrainedValues(student)).Max(p => Math.Abs(p.First - p.Second));
                Check(moved < 1e-6f, $"the weights do not move ({moved})");
            }

            // A stronger teacher, tuned here to answer yes.
            using var teacherModel = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
            var answers = Enumerable.Range(0, 8).Select(_ => AnswerSequence(teacherModel, prompt, "yes")).ToList();
            FineTuner.Train(teacherModel, answers, null, SessionOptions(1e-2f) with { Epochs = 30 });
            using var teacher2 = DistillationTeachers.FromModel(teacherModel);
            using var teacherSession = new FineTuningSession(teacherModel, SessionOptions(), totalSteps: 1);   // its log-probabilities only

            foreach (var (what, mix) in new[] { ("alone", 0f), ("with half the prompts from the data's answers", 0.5f) })
            {
                using var studentModel = PretrainedModel.Load(folder, new PretrainedOptions { Device = device });
                var session = new FineTuningSession(studentModel, SessionOptions(1e-2f) with { Teacher = teacher2 }, totalSteps: 30);
                using var student = new LanguageModelPolicy(session, new LanguageModelPolicyOptions { MaxNewTokens = 6 });
                double before = ReverseKlOnSamples(student, teacherSession, prompt, 99);
                int sampled = 0, fromData = 0;
                OnPolicyDistiller.Train(session, data, new OnPolicyDistillationOptions { Steps = 25, PromptsPerStep = 8, MaxNewTokens = 6, Mix = mix, Seed = 2 },
                    new SynchronousProgress<OnPolicyDistillationProgress>(p => (sampled, fromData) = (sampled + p.Sampled, fromData + p.FromData)));
                double after = ReverseKlOnSamples(student, teacherSession, prompt, 99);
                Check(after < 0.7 * before, $"on-policy distillation {what}: the reverse KL on the student's samples falls from {before:F3} to {after:F3}");
                Check(mix == 0f ? fromData == 0 : fromData > 0 && sampled > 0, $"{what}: {sampled} sampled, {fromData} from the data");
            }

            // Beside a reward: R2's grpo and a teacher term, under R2's trainer.
            using (var studentModel = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
            {
                var session = new FineTuningSession(studentModel, SessionOptions(1e-2f) with { Teacher = teacher2 }, totalSteps: 30);
                using var student = new LanguageModelPolicy(session, new LanguageModelPolicyOptions { MaxNewTokens = 6, TeacherWeight = 1f });
                var kind = new PromptsEnvironmentKind([new PromptTask(prompt, "yes")], name: "reinforcement-text-test-prompts");
                double before = ReverseKlOnSamples(student, teacherSession, prompt, 98);
                var result = ReinforcementTrainer.Train(kind, student, new ReinforcementOptions
                {
                    Algorithm = "grpo", EpisodesPerIteration = 8, GroupSize = 8, Iterations = 20, Seed = 500, Environments = 1, ParallelSteps = false,
                });
                double after = ReverseKlOnSamples(student, teacherSession, prompt, 98);
                Check(result.Iterations == 20 && result.History.All(h => float.IsFinite((float)h.Loss)), $"20 iterations of grpo with a teacher term: {result.Iterations}");
                Check(after < 0.7 * before, $"a teacher term beside a reward: the reverse KL on the student's samples falls from {before:F3} to {after:F3}");
            }

            // A teacher file cannot cover new samples; a teacher of another vocabulary is refused naming the way across.
            string file = Path.Combine(folder, "teacher.idtl");
            TeacherLogitsWriter.Write(file, teacherModel, answers, 4);
            using var stored = DistillationTeachers.FromFile(file);
            using (var studentModel = PretrainedModel.Load(folder, new PretrainedOptions { Device = device }))
            {
                ThrowsNaming<NotSupportedException>(() => DistillationTeachers.CheckOnPolicy(stored, studentModel.Tokenizer!), "a teacher file", "live teacher", "new samples");
                ThrowsNaming<InvalidOperationException>(() => DistillationTeachers.CheckOnPolicy(new OtherVocabularyTeacher(), studentModel.Tokenizer!), "another vocabulary",
                    "share a vocabulary", "idrak distill --generate");
                using var session = new FineTuningSession(studentModel, SessionOptions() with { Teacher = stored }, totalSteps: 1);
                ThrowsNaming<NotSupportedException>(() => new LanguageModelPolicy(session, new LanguageModelPolicyOptions { TeacherWeight = 1f }, ownsSession: false), "a policy's teacher file",
                    "live teacher");
                ThrowsNaming<NotSupportedException>(() => OnPolicyDistiller.Train(session, data), "distilling from a file", "live teacher");
            }
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    private static void PromptsAndRewards(Device device)
    {
        // The library's registrations.
        Check(EnvironmentKinds.Origin("prompts") == Idrak.Abstraction.Overrides.Library && EnvironmentKinds.Get("PROMPTS") is PromptsEnvironmentKind, "the prompts kind is the library's");
        foreach (string name in new[] { "metric", "regex", "json", "length", "judge", "sum" })
        {
            Check(RewardFunctions.Origin(name) == Idrak.Abstraction.Overrides.Library, $"the reward {name} is the library's");
        }

        // The environment suite on a kind of tasks, and the library's kind on a transcript file.
        var tasks = new[] { new PromptTask([new ChatMessage("user", "2 + 2?")], "4"), new PromptTask([new ChatMessage("user", "The capital of France?")], "Paris") };
        ChatMessage Act(Random random, ChatObservation _) => new("assistant", random.Next(3) switch { 0 => "4", 1 => "Paris", _ => "no idea" });
        Conformance.CheckEnvironmentKind<ChatObservation, ChatMessage>(new PromptsEnvironmentKind(tasks, name: "reinforcement-text-tasks"), Act).ThrowIfFailed();
        Conformance.CheckEnvironmentKind<ChatObservation, ChatMessage>(new PromptsEnvironmentKind(tasks, name: "reinforcement-text-tasks"), Act,
            new Dictionary<string, string> { ["reward"] = "exact:1,length:0.5", ["length.unit"] = "characters", ["length.budget"] = "3" }).ThrowIfFailed();
        string folder = TempFolder();
        try
        {
            string data = Path.Combine(folder, "prompts.jsonl");
            File.WriteAllLines(data, tasks.Select(t => new JsonObject
            {
                ["messages"] = new JsonArray(new JsonObject { ["role"] = "user", ["content"] = t.Prompt[0].Content }, new JsonObject { ["role"] = "assistant", ["content"] = t.Reference }),
            }.ToJsonString()));
            using var environment = EnvironmentKinds.Create<ChatObservation, ChatMessage>("prompts", new Dictionary<string, string> { ["data"] = data, ["reward"] = "exact" });
            var observation = environment.Reset(3);
            var answer = observation.Messages[^1].Content.Contains("France", StringComparison.Ordinal) ? "Paris." : "4";
            var step = environment.StepAsync(new ChatMessage("assistant", answer)).AsTask().GetAwaiter().GetResult();
            Check(step is { Reward: 1f, Terminated: true } && step.Info!["rounds"] == "1", $"a task from the file, answered: {step.Reward}");
            ThrowsNaming<ArgumentException>(() => EnvironmentKinds.Get("prompts").Create(new EnvironmentRequest()), "no data", "data");
            ThrowsNaming<ArgumentException>(() => EnvironmentKinds.Get("prompts").Create(new EnvironmentRequest(new Dictionary<string, string> { ["data"] = data, ["reward"] = "exact", ["length.unit"] = "words" })),
                "an option of a term not in the expression", "length.unit");
            ThrowsNaming<NotSupportedException>(() => EnvironmentKinds.Get("prompts").Create(new EnvironmentRequest(new Dictionary<string, string> { ["data"] = data, ["reward"] = "nonsense" })),
                "an unknown reward", "neither a reward", "metric");
        }
        finally
        {
            Directory.Delete(folder, true);
        }

        // The rewards on their suite, each against known scores.
        RewardInput[] samples =
        [
            new("2 + 2?", "4", "4"),
            new("The capital of France?", "Paris, which is the capital", "Paris"),
            new("Give JSON", "{\"a\": 1}", "x", new Dictionary<string, string> { ["tokens"] = "12" }),
            new("Give JSON", "```json\n{'a': 1,}\n```", "x", new Dictionary<string, string> { ["tokens"] = "300" }),
        ];
        float[] Score(string name, Dictionary<string, string>? options) => RewardFunctions.Get(name).ScoreAsync(samples, options).AsTask().GetAwaiter().GetResult();
        foreach (var (name, options, expected) in new (string, Dictionary<string, string>?, float[]?)[]
                 {
                     ("metric", null, [1f, 0f, 0f, 0f]),
                     ("metric", new() { ["metric"] = "contains" }, [1f, 1f, 0f, 0f]),
                     ("metric", new() { ["metric"] = "cer" }, null),
                     ("regex", new() { ["pattern"] = "^[0-9]+$" }, [1f, 0f, 0f, 0f]),
                     ("regex", new() { ["pattern"] = "paris", ["ignore_case"] = "true" }, [0f, 1f, 0f, 0f]),
                     ("json", null, [0f, 0f, 1f, 0.5f]),
                     ("length", new() { ["unit"] = "words", ["budget"] = "2", ["penalty"] = "0.5" }, [0f, -1.5f, 0f, -1f]),
                     ("sum", new() { ["terms"] = "exact:2,json:1" }, [2f, 0f, 1f, 0.5f]),
                 })
        {
            Conformance.CheckRewardFunction(RewardFunctions.Get(name), samples, options).ThrowIfFailed();
            if (expected is not null)
            {
                Check(Score(name, options).SequenceEqual(expected), $"{name} ({string.Join(",", options?.Select(o => $"{o.Key}={o.Value}") ?? [])}): {string.Join(", ", Score(name, options))}");
            }
        }

        var tokensOnly = samples[2..];
        Check(RewardFunctions.Get("length").ScoreAsync(tokensOnly, new Dictionary<string, string> { ["budget"] = "100" }).AsTask().GetAwaiter().GetResult()
            .SequenceEqual([0f, -2f]), "length in tokens from the episode's info");
        ThrowsNaming<ArgumentException>(() => Score("length", null), "length in tokens without the count", "tokens", "unit=words");
        ThrowsNaming<ArgumentException>(() => Score("metric", new() { ["strip_diacritics"] = "true" }), "an option the metric does not take", "strip_diacritics");
        ThrowsNaming<ArgumentException>(() => Score("regex", null), "a pattern", "pattern");
        ThrowsNaming<NotSupportedException>(() => Score("sum", new() { ["terms"] = "exact:1,nonsense:1" }), "an unknown term", "nonsense");

        // The judge: a tiny model's yes against no, loaded from its folder by the library's reward, or given as a scorer.
        string judgeFolder = WriteChatModelWith(TinyChatSpec(), ChatMlTemplate);
        try
        {
            var judgeOptions = new Dictionary<string, string> { ["model"] = judgeFolder, ["question"] = "Correct?", ["yes"] = "yes", ["no"] = "no" };
            Conformance.CheckRewardFunction(RewardFunctions.Get("judge"), samples, judgeOptions).ThrowIfFailed();
            var scores = RewardFunctions.Get("judge").ScoreAsync(samples, judgeOptions).AsTask().GetAwaiter().GetResult();
            using var judgeModel = PretrainedModel.Load(judgeFolder, new PretrainedOptions { Device = Device.Cpu });
            var scorer = new AnswerScorer(judgeModel);
            var own = new JudgeReward(scorer, "reinforcement-text-judge");
            var direct = scorer.Choose([.. samples.Select(s => (IReadOnlyList<ChatMessage>)[new ChatMessage("system", "Correct?"), new ChatMessage("user", $"{s.Prompt}\n\n{s.Answer}\n\n{s.Reference}")])], ["yes", "no"]);
            var given = own.ScoreAsync(samples, new Dictionary<string, string> { ["question"] = "Correct?", ["yes"] = "yes", ["no"] = "no" }).AsTask().GetAwaiter().GetResult();
            AssertClose([.. direct.Select(c => (float)c.Probabilities[0])], given, 1e-6f, "the judge's yes");
            AssertClose(given, scores, 1e-5f, "the library's judge loads the same model");
            ThrowsNaming<ArgumentException>(() => own.ScoreAsync(samples).AsTask().GetAwaiter().GetResult(), "the judge's question", "question");
            ((JudgeReward)RewardFunctions.Default("judge")!).Release();
        }
        finally
        {
            Directory.Delete(judgeFolder, true);
        }
    }
}
