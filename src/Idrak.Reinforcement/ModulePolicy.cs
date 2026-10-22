// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Reinforcement;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Reinforcement.Abstractions;

namespace Idrak.Reinforcement;

/// <summary>
/// The settings of a <see cref="ModulePolicy{TAction}"/>: its optimizer, made by name through <c>OptimizerKinds</c>, and the
/// networks <see cref="PolicyNetworks"/> builds for it (hidden widths, activation, a value head).
/// </summary>
public sealed record ModulePolicyOptions
{
    /// <summary>The optimizer, by name in <c>OptimizerKinds</c> (adam, adamw, adamw8bit, sgd, or an app's). <c>IDRAK_RL_OPTIMIZER</c>.</summary>
    public string Optimizer { get; init; } = "adam";

    /// <summary>The optimizer's learning rate, the base of a trainer's schedule. <c>IDRAK_RL_LEARNING_RATE</c>.</summary>
    public float LearningRate { get; init; } = 3e-4f;

    /// <summary>The optimizer's weight decay.</summary>
    public float WeightDecay { get; init; }

    /// <summary>The largest global gradient norm an update applies (0: no clip); a trainer sets its own (<see cref="ReinforcementOptions.MaxGradientNorm"/>).</summary>
    public float MaxGradientNorm { get; init; }

    /// <summary>The Gaussian policy's starting log standard deviation of every action element (a learned parameter).</summary>
    public float InitialLogStd { get; init; }

    /// <summary>
    /// The hidden layers' widths of a network <see cref="PolicyNetworks"/> builds from these options (an actor, a critic, a
    /// Q-network); empty for one linear layer. <c>IDRAK_RL_HIDDEN</c> ("64,64"; "none" for none).
    /// </summary>
    public IReadOnlyList<int> Hidden { get; init; } = DefaultHidden;

    /// <summary>The activation after each hidden layer of such a network: one of <see cref="PolicyNetworks.Activations"/>. <c>IDRAK_RL_ACTIVATION</c>.</summary>
    public string Activation { get; init; } = PolicyNetworks.Tanh;

    /// <summary>
    /// Whether a policy <see cref="PolicyNetworks"/> builds has a value head (a critic); null for automatic
    /// (<see cref="ValueHeadFor"/>). <c>IDRAK_RL_VALUE_HEAD</c> (on, off, auto).
    /// </summary>
    public bool? ValueHead { get; init; }

    // One array for the default, so the defaults compare equal (a record compares a list by reference).
    private static readonly int[] DefaultHidden = [64, 64];

    /// <summary>
    /// Whether a policy trained by <paramref name="run"/> gets a value head: <see cref="ValueHead"/> when set; else
    /// automatic, a critic where the run's advantages use one (ppo with its own estimator, gae, or a value baseline of
    /// returns) and none where they do not (grpo's group, reinforce's returns), so no unused network is trained.
    /// </summary>
    public bool ValueHeadFor(ReinforcementOptions run)
    {
        ArgumentNullException.ThrowIfNull(run);
        if (ValueHead is { } set)
        {
            return set;
        }

        string? estimator = run.Estimator;
        if (estimator is null)
        {
            return string.Equals(run.Algorithm, ReinforcementNames.Ppo, StringComparison.OrdinalIgnoreCase);
        }

        return string.Equals(estimator, ReinforcementNames.Gae, StringComparison.OrdinalIgnoreCase)
               || (string.Equals(estimator, ReinforcementNames.Returns, StringComparison.OrdinalIgnoreCase) && run.EstimatorOptions is { } options
                   && options.TryGetValue(ReinforcementNames.BaselineKey, out string? baseline) && string.Equals(baseline, "value", StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>The hidden widths in <paramref name="text"/>: whole numbers above 0 separated by commas ("64,64"), or "none" (or empty) for no hidden layer.</summary>
    /// <exception cref="ArgumentException">A width that is not a whole number above 0.</exception>
    public static IReadOnlyList<int> ParseHidden(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        string trimmed = text.Trim();
        if (trimmed.Length == 0 || trimmed.Equals("none", StringComparison.OrdinalIgnoreCase))
        {
            return [];
        }

        var widths = new List<int>();
        foreach (string part in trimmed.Split(','))
        {
            widths.Add(int.TryParse(part.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out int width) && width > 0 ? width
                : throw new ArgumentException($"'{text}' is not a list of hidden widths: whole numbers above 0 separated by commas (64,64), or none."));
        }

        return widths;
    }

    /// <summary>
    /// These options with each one an <c>IDRAK_RL_*</c> variable sets in its place (<c>IDRAK_RL_OPTIMIZER</c>,
    /// <c>IDRAK_RL_LEARNING_RATE</c>, <c>IDRAK_RL_HIDDEN</c>, <c>IDRAK_RL_ACTIVATION</c>, <c>IDRAK_RL_VALUE_HEAD</c>); the
    /// variables not set keep <paramref name="start"/>'s (the defaults when null).
    /// </summary>
    /// <exception cref="ArgumentException">A variable's value is not one the option takes (the message names the variable).</exception>
    public static ModulePolicyOptions FromEnvironment(ModulePolicyOptions? start = null)
    {
        var options = start ?? new ModulePolicyOptions();
        if (Variables.Text("IDRAK_RL_OPTIMIZER") is { } optimizer)
        {
            options = options with { Optimizer = optimizer };
        }

        if (Variables.Float("IDRAK_RL_LEARNING_RATE", min: 0f) is { } rate)
        {
            options = options with { LearningRate = rate };
        }

        if (Variables.Text("IDRAK_RL_HIDDEN") is { } hidden)
        {
            try
            {
                options = options with { Hidden = ParseHidden(hidden) };
            }
            catch (ArgumentException)
            {
                throw new ArgumentException($"IDRAK_RL_HIDDEN={hidden} is not a list of hidden widths: whole numbers above 0 separated by commas (64,64), or none.");
            }
        }

        if (Variables.Text("IDRAK_RL_ACTIVATION") is { } activation)
        {
            options = options with
            {
                Activation = PolicyNetworks.Activations.FirstOrDefault(a => a.Equals(activation, StringComparison.OrdinalIgnoreCase))
                             ?? throw new ArgumentException($"IDRAK_RL_ACTIVATION={activation} is not {string.Join(", ", PolicyNetworks.Activations)}."),
            };
        }

        if (Variables.Switch("IDRAK_RL_VALUE_HEAD") is var (valueSet, valueHead) && valueSet)
        {
            options = options with { ValueHead = valueHead };
        }

        return options;
    }
}

/// <summary>
/// Policies over a <see cref="Module"/> (<see cref="ModulePolicy{TAction}"/>): the general policy for environments whose
/// observations are floats (a <see cref="SpaceKind.Box"/>), a categorical one for a <see cref="SpaceKind.Discrete"/> action
/// space and a Gaussian one for a box of actions, with a critic (a value head) or not.
/// </summary>
public static class ModulePolicy
{
    /// <summary>
    /// A categorical policy: <paramref name="actor"/> maps observations [batch, observation size] to logits [batch, n] over
    /// the n choices of <paramref name="actionSpace"/>; π(a | s) = softmax(logits)_a.
    /// </summary>
    /// <param name="observationSpace">The environment's observations: a box (its size is a row of the actor's input).</param>
    /// <param name="actionSpace">The environment's actions: discrete.</param>
    /// <param name="actor">Observations to logits; trained.</param>
    /// <param name="critic">Observations to values [batch, 1] (a value head for GAE, PPO's value loss, a learned baseline); null for none.</param>
    /// <param name="reference">A module of the actor's architecture that takes the actor's starting weights and stays frozen, the reference of a KL penalty; null for none.</param>
    /// <param name="options">The optimizer and its settings; null for <see cref="ModulePolicyOptions.FromEnvironment"/>.</param>
    /// <exception cref="ArgumentException">An observation space that is not a box, or an action space that is not discrete.</exception>
    public static ModulePolicy<int> Categorical(Space observationSpace, Space actionSpace, Module actor, Module? critic = null, Module? reference = null,
        ModulePolicyOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(actionSpace);
        if (actionSpace.Kind != SpaceKind.Discrete)
        {
            throw new ArgumentException($"A categorical policy chooses among discrete actions, not {actionSpace}; use ModulePolicy.Gaussian for a box.", nameof(actionSpace));
        }

        return new ModulePolicy<int>(observationSpace, actionSpace, actor, critic, reference, options ?? ModulePolicyOptions.FromEnvironment());
    }

    /// <summary>
    /// A Gaussian policy: <paramref name="actor"/> maps observations [batch, observation size] to the means [batch, d] of the
    /// d elements of <paramref name="actionSpace"/>; each element is drawn with a learned standard deviation of its own
    /// (state-independent, from <see cref="ModulePolicyOptions.InitialLogStd"/>) and clipped to the box's bounds, its
    /// log-probability taken at the clipped value both when acting and when updating.
    /// </summary>
    /// <param name="observationSpace">The environment's observations: a box.</param>
    /// <param name="actionSpace">The environment's actions: a box.</param>
    /// <param name="actor">Observations to means; trained.</param>
    /// <param name="critic">Observations to values [batch, 1]; null for none.</param>
    /// <param name="reference">A module of the actor's architecture that takes the actor's starting weights and stays frozen; null for none.</param>
    /// <param name="options">The optimizer and its settings; null for <see cref="ModulePolicyOptions.FromEnvironment"/>.</param>
    /// <exception cref="ArgumentException">An observation or action space that is not a box.</exception>
    public static ModulePolicy<float[]> Gaussian(Space observationSpace, Space actionSpace, Module actor, Module? critic = null, Module? reference = null,
        ModulePolicyOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(actionSpace);
        if (actionSpace.Kind != SpaceKind.Box)
        {
            throw new ArgumentException($"A Gaussian policy acts in a box, not {actionSpace}; use ModulePolicy.Categorical for discrete actions.", nameof(actionSpace));
        }

        return new ModulePolicy<float[]>(observationSpace, actionSpace, actor, critic, reference, options ?? ModulePolicyOptions.FromEnvironment());
    }

    /// <summary>
    /// A small fully connected network: <paramref name="inputs"/> to each of <paramref name="hidden"/> in turn (each followed
    /// by tanh) to <paramref name="outputs"/>, its weights drawn from <paramref name="random"/>; for an actor or a critic.
    /// </summary>
    /// <param name="inputs">The input features (the observation size).</param>
    /// <param name="outputs">The outputs (the choices, the action size, or 1 for a critic).</param>
    /// <param name="hidden">The hidden layers' widths; empty for one linear layer.</param>
    /// <param name="device">Where the weights live; null for the default device.</param>
    /// <param name="random">The weights' draws; pass a seeded one for a reproducible run.</param>
    public static Sequential Perceptron(int inputs, int outputs, IReadOnlyList<int> hidden, Device? device = null, Random? random = null)
    {
        ArgumentNullException.ThrowIfNull(hidden);
        var network = new Sequential();
        int width = inputs;
        foreach (int size in hidden)
        {
            network.Add(new Linear(width, size, device: device, random: random));
            network.Add(new Tanh());
            width = size;
        }

        network.Add(new Linear(width, outputs, device: device, random: random));
        return network;
    }
}

/// <summary>
/// A policy over a <see cref="Module"/> (<see cref="ModulePolicy.Categorical"/>, <see cref="ModulePolicy.Gaussian"/>):
/// observations are float arrays, a batch of them a [batch, size] input of the actor; an action is one unit (a choice, or a
/// whole box of floats) with its log-probability under the policy that drew it; the critic's value of the observation is
/// recorded when there is one. <see cref="Update"/> is autograd plus an optimizer made through <c>OptimizerKinds</c>: the
/// mini-batch's log-probabilities, entropies and values under the current weights (the reference's too when there is one)
/// go to the policy loss, its gradient is clipped to <see cref="MaxGradientNorm"/> and stepped. The same observations, seed
/// and weights give the same actions; every draw comes from the call's seed, row by row in order.
/// </summary>
/// <typeparam name="TAction"><see cref="int"/> (a choice) for a categorical policy, a <see cref="float"/> array for a Gaussian one.</typeparam>
public sealed class ModulePolicy<TAction> : IPolicy<float[], TAction>, IScheduledPolicy
{
    private const string Format = "idrak-module-policy/1", StateFile = "policy.json", ActorFile = "actor.ikw", CriticFile = "critic.ikw",
        SpreadFile = "spread.ikw", ReferenceFile = "reference.ikw", OptimizerFile = "optimizer.safetensors";

    private static readonly double HalfLogTwoPi = 0.5 * Math.Log(2 * Math.PI);

    private readonly Module _actor;
    private readonly Module? _critic, _reference;
    private readonly LogSpread? _spread;
    private readonly int _observationSize, _actionSize;
    private readonly bool _gaussian;
    private bool _disposed;

    internal ModulePolicy(Space observationSpace, Space actionSpace, Module actor, Module? critic, Module? reference, ModulePolicyOptions options)
    {
        ArgumentNullException.ThrowIfNull(observationSpace);
        ArgumentNullException.ThrowIfNull(actor);
        ArgumentNullException.ThrowIfNull(options);
        if (observationSpace.Kind != SpaceKind.Box)
        {
            throw new ArgumentException($"A module policy reads float observations (a box), not {observationSpace}.", nameof(observationSpace));
        }

        _gaussian = actionSpace.Kind == SpaceKind.Box;
        if (_gaussian ? typeof(TAction) != typeof(float[]) : typeof(TAction) != typeof(int))
        {
            throw new ArgumentException($"A module policy over {actionSpace} acts with {(_gaussian ? "float[]" : "int")}, not {typeof(TAction).Name}.", nameof(actionSpace));
        }

        ObservationSpace = observationSpace;
        ActionSpace = actionSpace;
        _actor = actor;
        _critic = critic;
        _observationSize = observationSpace.Size;
        _actionSize = _gaussian ? actionSpace.Size : actionSpace.Count;
        Device = actor.WeightsDevice ?? Device.Default;
        Options = options;
        if (_gaussian)
        {
            _spread = new LogSpread(_actionSize, options.InitialLogStd, Device);
        }

        if (reference is not null)
        {
            using var copy = new MemoryStream();
            actor.Save(copy);
            copy.Position = 0;
            reference.Load(copy);
            reference.Freeze();
            _reference = reference;
        }

        List<Tensor> trained = [.. actor.Parameters()];
        if (critic is not null)
        {
            trained.AddRange(critic.Parameters());
        }

        if (_spread is not null)
        {
            trained.AddRange(_spread.Parameters());
        }

        Optimizer = OptimizerKinds.Factory(options.Optimizer, new OptimizerSettings(options.LearningRate, options.WeightDecay))(trained);
        MaxGradientNorm = options.MaxGradientNorm;
    }

    /// <summary>The observations the policy reads.</summary>
    public Space ObservationSpace { get; }

    /// <summary>The actions it takes.</summary>
    public Space ActionSpace { get; }

    /// <summary>The settings it was made with.</summary>
    public ModulePolicyOptions Options { get; }

    /// <summary>The network from observations to logits or means.</summary>
    public Module Actor => _actor;

    /// <summary>The value head, or null.</summary>
    public Module? Critic => _critic;

    /// <summary>Whether it keeps a frozen reference (for a KL penalty).</summary>
    public bool HasReference => _reference is not null;

    /// <inheritdoc />
    public Optimizer Optimizer { get; }

    /// <inheritdoc />
    public float MaxGradientNorm
    {
        get;
        set => field = value >= 0 && float.IsFinite(value) ? value : throw new ArgumentOutOfRangeException(nameof(value), value, "The gradient-norm clip is 0 (none) or positive.");
    }

    /// <inheritdoc />
    public Device Device { get; }

    /// <summary>Every tensor the policy trains: the actor's, the critic's and the Gaussian's spread, in the optimizer's order.</summary>
    public IReadOnlyList<Tensor> Parameters => Optimizer.Parameters;

    /// <inheritdoc />
    public ValueTask<PolicyActions<TAction>> ActAsync(IReadOnlyList<float[]> observations, ActionSampling sampling, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(observations);
        ArgumentNullException.ThrowIfNull(sampling);
        cancellationToken.ThrowIfCancellationRequested();
        int rows = observations.Count;
        if (rows == 0)
        {
            return ValueTask.FromResult(new PolicyActions<TAction>(Array.Empty<TAction>(), Array.Empty<ActionRecord>()));
        }

        float[] input = ArrayPool<float>.Shared.Rent(rows * _observationSize);
        try
        {
            Fill(observations, input);
            float[] outputs, values;
            using (var scope = new TensorScope())
            using (Autograd.NoGrad())
            {
                var x = Tensor.From(input.AsSpan(0, rows * _observationSize), [rows, _observationSize], Device);
                outputs = _actor.Forward(x).ToArray();
                values = _critic is null ? [] : _critic.Forward(x).ToArray();
            }

            CheckWidth(outputs.Length, rows, "acted");
            var random = new Random(sampling.Seed);
            var records = new ActionRecord[rows];
            float value(int row) => values.Length == 0 ? float.NaN : values[row];
            if (_gaussian)
            {
                var actions = new float[rows][];
                var logStd = _spread!.Values();
                for (int row = 0; row < rows; row++)
                {
                    var mean = outputs.AsSpan(row * _actionSize, _actionSize);
                    var action = new float[_actionSize];
                    double logp = 0;
                    for (int j = 0; j < _actionSize; j++)
                    {
                        // The spread sampled with: σ · √temperature (greedy: the mean, its density under σ).
                        double sigma = Math.Exp(logStd[j]) * (sampling.Greedy ? 1 : Math.Sqrt(sampling.Temperature));
                        double drawn = sampling.Greedy ? mean[j] : mean[j] + sigma * random.NextGaussian();
                        float clipped = Math.Clamp((float)drawn, ActionSpace.Low[j], ActionSpace.High[j]);
                        action[j] = clipped;
                        double z = (clipped - mean[j]) / sigma;
                        logp += -0.5 * z * z - Math.Log(sigma) - HalfLogTwoPi;
                    }

                    actions[row] = action;
                    records[row] = new ActionRecord([], [(float)logp], value(row));
                }

                return ValueTask.FromResult(new PolicyActions<TAction>((TAction[])(object)actions, records));
            }

            var choices = new int[rows];
            for (int row = 0; row < rows; row++)
            {
                var logits = outputs.AsSpan(row * _actionSize, _actionSize);
                int choice = sampling.Greedy ? ArgMax(logits) : Sample(logits, sampling.Temperature, random);
                choices[row] = choice;
                records[row] = new ActionRecord([choice], [(float)LogProbability(logits, choice, sampling.Greedy ? 1f : sampling.Temperature)], value(row));
            }

            return ValueTask.FromResult(new PolicyActions<TAction>((TAction[])(object)choices, records));
        }
        finally
        {
            ArrayPool<float>.Shared.Return(input);
        }
    }

    /// <inheritdoc />
    public UpdateStats Update(RolloutBatch<float[], TAction> batch, IPolicyLoss loss, IReadOnlyList<int>? episodes = null, IReadOnlyDictionary<string, string>? lossOptions = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(batch);
        ArgumentNullException.ThrowIfNull(loss);
        int count = episodes?.Count ?? batch.Episodes;
        if (count == 0)
        {
            throw new ArgumentException("An update trains on at least one episode.", nameof(episodes));
        }

        int units = 0;
        for (int i = 0; i < count; i++)
        {
            int e = episodes?[i] ?? i;
            if ((uint)e >= (uint)batch.Episodes)
            {
                throw new ArgumentOutOfRangeException(nameof(episodes), e, $"The batch holds {batch.Episodes} episodes.");
            }

            units += batch.Trajectories[e].Steps;
        }

        float[] observations = ArrayPool<float>.Shared.Rent(units * _observationSize);
        float[] actions = ArrayPool<float>.Shared.Rent(units * (_gaussian ? _actionSize : 1));
        float[] unitValues = ArrayPool<float>.Shared.Rent(3 * units);                                 // behaviour, advantages, returns
        int[] counts = ArrayPool<int>.Shared.Rent(count);
        try
        {
            Gather(batch, episodes, count, observations, actions, unitValues, counts, units);
            using var scope = new TensorScope();
            var x = Tensor.From(observations.AsSpan(0, units * _observationSize), [units, _observationSize], Device);
            var (logp, entropy) = LogProbabilities(_actor, x, actions, units);
            Tensor? reference = null;
            if (_reference is not null)
            {
                using (Autograd.NoGrad())
                {
                    reference = LogProbabilities(_reference, x, actions, units).LogProbabilities;
                }
            }

            var behavior = Tensor.From(unitValues.AsSpan(0, units), [units], Device);
            var advantages = Tensor.From(unitValues.AsSpan(units, units), [units], Device);
            Tensor? values = null, returns = null;
            if (_critic is not null)
            {
                values = _critic.Forward(x).Reshape(units);
                returns = Tensor.From(unitValues.AsSpan(2 * units, units), [units], Device);
            }

            var input = new PolicyLossInput(logp, new ArraySegment<int>(counts, 0, count), advantages, behavior, reference, values, returns, entropy);
            var total = loss.Loss(input, lossOptions);
            if (total.Size != 1)
            {
                throw new InvalidOperationException($"The policy loss {loss.Name} gave {total.Size} values: a loss is one element.");
            }

            Optimizer.ZeroGrad();
            total.Backward();
            double norm = MaxGradientNorm > 0 ? Optimizer.ClipGradientNorm(MaxGradientNorm) : Optimizer.GradientNorm();
            Optimizer.Step();
            float lossValue = total.Item();
            var metrics = Metrics(logp.ToArray(), unitValues.AsSpan(0, units), entropy.ToArray(), reference?.ToArray(), Clip(lossOptions), norm);
            return new UpdateStats(lossValue, count, units, metrics);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(observations);
            ArrayPool<float>.Shared.Return(actions);
            ArrayPool<float>.Shared.Return(unitValues);
            ArrayPool<int>.Shared.Return(counts);
        }
    }

    /// <summary>
    /// Saves the policy into the folder <paramref name="path"/> (made when missing): the actor's, critic's, spread's and
    /// reference's weights (.ikw), the optimizer's state (its moments and step, a safetensors file), and what kind of policy
    /// it is, so <see cref="Load"/> resumes it where it stopped.
    /// </summary>
    public void Save(string path)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        Directory.CreateDirectory(path);
        _actor.Save(Path.Combine(path, ActorFile));
        _critic?.Save(Path.Combine(path, CriticFile));
        _spread?.Save(Path.Combine(path, SpreadFile));
        _reference?.Save(Path.Combine(path, ReferenceFile));
        var state = Optimizer.SaveState();
        SafeTensorsWriter.Write(Path.Combine(path, OptimizerFile), state.Select(e => (e.Key, e.Value.Shape, e.Value.Values)), SafeTensorType.F32,
            new Dictionary<string, string> { ["format"] = "idrak-optimizer", ["optimizer"] = Optimizer.GetType().Name });
        File.WriteAllText(Path.Combine(path, StateFile), new JsonObject
        {
            ["format"] = Format,
            ["kind"] = _gaussian ? "gaussian" : "categorical",
            ["observations"] = _observationSize,
            ["actions"] = _actionSize,
            ["critic"] = _critic is not null,
            ["reference"] = _reference is not null,
        }.ToJsonString() + "\n");
    }

    /// <summary>Loads what <see cref="Save"/> wrote in the folder <paramref name="path"/> into this policy (of the same kind and sizes).</summary>
    /// <exception cref="InvalidDataException">The folder holds another kind of policy, or other sizes.</exception>
    public void Load(string path)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        string statePath = Path.Combine(path, StateFile);
        if (!File.Exists(statePath))
        {
            throw new FileNotFoundException($"{path} holds no module policy ({StateFile}); give a folder ModulePolicy.Save wrote.", statePath);
        }

        var state = JsonNode.Parse(File.ReadAllText(statePath)) as JsonObject;
        string kind = (string?)state?["kind"] ?? "", expected = _gaussian ? "gaussian" : "categorical";
        if ((string?)state?["format"] != Format || kind != expected || (int?)state?["observations"] != _observationSize || (int?)state?["actions"] != _actionSize
            || (bool?)state?["critic"] != (_critic is not null))
        {
            throw new InvalidDataException(string.Create(CultureInfo.InvariantCulture,
                $"{path} holds a {kind} policy of {(int?)state?["observations"]} observations and {(int?)state?["actions"]} actions (critic: {(bool?)state?["critic"]}), "
                + $"not a {expected} one of {_observationSize} and {_actionSize} (critic: {_critic is not null})."));
        }

        _actor.Load(Path.Combine(path, ActorFile));
        _critic?.Load(Path.Combine(path, CriticFile));
        _spread?.Load(Path.Combine(path, SpreadFile));
        if (_reference is not null && File.Exists(Path.Combine(path, ReferenceFile)))
        {
            _reference.Load(Path.Combine(path, ReferenceFile));
        }

        using var reader = SafeTensorsReader.Open(Path.Combine(path, OptimizerFile));
        ITensorStore store = reader;
        Optimizer.LoadState(store.Names.ToDictionary(n => n, n => new OptimizerStateValue(store.ShapeOf(n), store.Read(n)), StringComparer.Ordinal));
    }

    /// <summary>Disposes the optimizer and the weights (the actor's, the critic's, the reference's).</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        Optimizer.Dispose();
        _actor.Dispose();
        _critic?.Dispose();
        _spread?.Dispose();
        _reference?.Dispose();
    }

    // The observations, checked, into rows of `input`.
    private void Fill(IReadOnlyList<float[]> observations, float[] input)
    {
        for (int row = 0; row < observations.Count; row++)
        {
            var observation = observations[row];
            if (observation is null || observation.Length != _observationSize)
            {
                throw new ArgumentException($"Observation {row} holds {observation?.Length ?? 0} floats; the policy reads {_observationSize} ({ObservationSpace}).", nameof(observations));
            }

            observation.CopyTo(input, row * _observationSize);
        }
    }

    private void CheckWidth(int outputs, int rows, string what)
    {
        if (outputs != rows * _actionSize)
        {
            throw new InvalidOperationException(string.Create(CultureInfo.InvariantCulture,
                $"The actor {what} with {outputs / Math.Max(1, rows)} outputs a row; the policy needs {_actionSize} ({(_gaussian ? "the means of" : "the logits of")} {ActionSpace})."));
        }
    }

    // The units of the episodes asked for, in order: observations, actions, behaviour log-probabilities, advantages, returns
    // and each episode's steps.
    private void Gather(RolloutBatch<float[], TAction> batch, IReadOnlyList<int>? episodes, int count, float[] observations, float[] actions, float[] unitValues, int[] counts, int units)
    {
        var advantages = batch.Advantages;
        var returns = batch.Returns;
        int at = 0;
        for (int i = 0; i < count; i++)
        {
            int e = episodes?[i] ?? i;
            var trajectory = batch.Trajectories[e];
            int offset = batch.Offsets[e];
            counts[i] = trajectory.Steps;
            for (int t = 0; t < trajectory.Steps; t++, at++)
            {
                var observation = trajectory.Observations[t];
                if (observation.Length != _observationSize)
                {
                    throw new ArgumentException($"Episode {trajectory.Index}'s observation {t} holds {observation.Length} floats; the policy reads {_observationSize}.", nameof(batch));
                }

                observation.CopyTo(observations, at * _observationSize);
                switch (trajectory.Actions[t])
                {
                    case int choice:
                        actions[at] = choice;
                        break;
                    case float[] values:
                        values.AsSpan(0, _actionSize).CopyTo(actions.AsSpan(at * _actionSize));
                        break;
                }

                unitValues[at] = trajectory.Records[t].LogProbabilities[0];
                unitValues[units + at] = advantages[offset + t];
                unitValues[2 * units + at] = returns[offset + t];
            }
        }
    }

    // The log-probability [units] of each unit's action under `network`, and the policy's entropy at each unit [units].
    private (Tensor LogProbabilities, Tensor Entropies) LogProbabilities(Module network, Tensor x, float[] actions, int units)
    {
        var outputs = network.Forward(x);
        CheckWidth(outputs.Size, units, "updated");
        if (_gaussian)
        {
            var taken = Tensor.From(actions.AsSpan(0, units * _actionSize), [units, _actionSize], Device);
            var logStd = Tensor.Zeros([units, _actionSize], Device) + _spread!.Parameter;   // [d] over every row
            var z = (taken - outputs) * (-logStd).Exp();
            var perElement = z.Square() * -0.5f - logStd - (float)HalfLogTwoPi;
            var entropy = logStd.Sum(1) + _actionSize * (float)(0.5 + HalfLogTwoPi);
            return (perElement.Sum(1), entropy);
        }

        var logSoftmax = outputs.LogSoftmax();
        var chosen = Tensor.OneHot(Tensor.From(actions.AsSpan(0, units), [units], Device), _actionSize);    // the choices as exact floats, as Tensor.Indices holds them
        var logp = (logSoftmax * chosen).Sum(1);
        var entropies = -(logSoftmax.Exp() * logSoftmax).Sum(1);
        return (logp, entropies);
    }

    // What the update measured, from the host copies (sums in unit order): the KL of the behaviour policy to the updated
    // one (k3), the share of ratios outside the clip, the mean entropy, the KL to the reference and the gradient norm.
    private static Dictionary<string, double> Metrics(float[] logp, ReadOnlySpan<float> behavior, float[] entropy, float[]? reference, float clip, double norm)
    {
        double kl = 0, entropySum = 0, referenceKl = 0;
        int clipped = 0;
        for (int u = 0; u < logp.Length; u++)
        {
            double d = (double)logp[u] - behavior[u];
            double ratio = Math.Exp(d);
            kl += ratio - 1 - d;
            clipped += Math.Abs(ratio - 1) > clip ? 1 : 0;
            entropySum += entropy[u];
            if (reference is not null)
            {
                double r = (double)reference[u] - logp[u];
                referenceKl += Math.Exp(r) - r - 1;
            }
        }

        int n = logp.Length;
        var metrics = new Dictionary<string, double>(StringComparer.Ordinal)
        {
            ["kl"] = kl / n,
            ["clip-fraction"] = (double)clipped / n,
            ["entropy"] = entropySum / n,
            ["gradient-norm"] = norm,
        };
        if (reference is not null)
        {
            metrics["kl-reference"] = referenceKl / n;
        }

        return metrics;
    }

    // The clip the loss's options give (for the clipped share); PPO's 0.2 when they give none.
    private static float Clip(IReadOnlyDictionary<string, string>? options) =>
        options is not null && options.TryGetValue(ReinforcementNames.ClipKey, out string? text)
        && float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out float clip) ? clip : 0.2f;

    private static int ArgMax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > logits[best])
            {
                best = i;
            }
        }

        return best;
    }

    // A draw from softmax(logits / temperature): one uniform draw against the cumulative probabilities, in order.
    private static int Sample(ReadOnlySpan<float> logits, float temperature, Random random)
    {
        double max = double.NegativeInfinity;
        foreach (float logit in logits)
        {
            max = Math.Max(max, logit / (double)temperature);
        }

        double total = 0;
        foreach (float logit in logits)
        {
            total += Math.Exp(logit / (double)temperature - max);
        }

        double u = random.NextDouble() * total, cumulative = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            cumulative += Math.Exp(logits[i] / (double)temperature - max);
            if (u < cumulative)
            {
                return i;
            }
        }

        return logits.Length - 1;
    }

    // log softmax(logits / temperature)[choice].
    private static double LogProbability(ReadOnlySpan<float> logits, int choice, float temperature)
    {
        double max = double.NegativeInfinity;
        foreach (float logit in logits)
        {
            max = Math.Max(max, logit / (double)temperature);
        }

        double total = 0;
        foreach (float logit in logits)
        {
            total += Math.Exp(logit / (double)temperature - max);
        }

        return logits[choice] / (double)temperature - max - Math.Log(total);
    }

    // The Gaussian's log standard deviation per action element: a trained parameter of its own, saved with the policy.
    private sealed class LogSpread : Module
    {
        public LogSpread(int size, float initial, Device device)
        {
            var values = new float[size];
            values.AsSpan().Fill(initial);
            Parameter = CreateParameter(values, [size], device);
        }

        public Tensor Parameter { get; }

        public float[] Values() => Parameter.ToArray();

        public override IEnumerable<Tensor> Parameters() => [Parameter];

        protected override Tensor ForwardCore(Tensor input) => throw new NotSupportedException("The spread is a parameter, not a layer.");
    }
}
