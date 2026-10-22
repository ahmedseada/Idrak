// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Training;

/// <summary>
/// What an optimizer is made with, by name (<see cref="OptimizerKinds"/>): for command lines, settings files and trainers
/// that name their optimizer. A kind reads what it uses and leaves the rest.
/// </summary>
/// <param name="LearningRate">The step size the optimizer starts with (the peak a schedule works from).</param>
/// <param name="WeightDecay">Weight decay: decoupled for AdamW and 8-bit AdamW, an L2 term on the gradients for Adam and SGD.</param>
/// <param name="Momentum">SGD's momentum.</param>
public sealed record OptimizerSettings(float LearningRate, float WeightDecay = 0f, float Momentum = 0.9f)
{
    /// <summary>The first moment's decay (the Adam kinds); null for the kind's own (Adam's 0.9).</summary>
    public float? Beta1 { get; init; }

    /// <summary>The second moment's decay (the Adam kinds); null for the kind's own (Adam's 0.999).</summary>
    public float? Beta2 { get; init; }

    /// <summary>The term added to the second moment's root (the Adam kinds); null for the kind's own (Adam's 1e-8).</summary>
    public float? Epsilon { get; init; }
}

/// <summary>
/// A kind of optimizer, made by name from the parameters it updates and <see cref="OptimizerSettings"/>. Registered in
/// <see cref="OptimizerKinds"/> under <see cref="Name"/>. Thread-safe: one kind makes the optimizers of several runs.
/// </summary>
public interface IOptimizerKind
{
    /// <summary>The kind's name in <see cref="OptimizerKinds"/> ("adamw", "sgd").</summary>
    string Name { get; }

    /// <summary>One line for help and listings.</summary>
    string Summary { get; }

    /// <summary>A new optimizer over <paramref name="parameters"/>, starting at <see cref="OptimizerSettings.LearningRate"/>.</summary>
    /// <param name="parameters">The tensors it updates, in the order its state is saved.</param>
    /// <param name="settings">The rate, decay and the kind's own settings.</param>
    /// <exception cref="ArgumentException">A setting the kind cannot use (a negative rate, say).</exception>
    Optimizer Create(IReadOnlyList<Tensor> parameters, OptimizerSettings settings);
}

/// <summary>
/// The optimizers trainers make by name (ignoring case): fine-tuning, pretraining, the speech trainers, the command-line
/// tools, and any trainer that takes an optimizer's name. The library's:
/// <list type="bullet">
/// <item>"adamw": <see cref="AdamW"/>, decoupled weight decay;</item>
/// <item>"adam": <see cref="Adam"/>, weight decay as an L2 term;</item>
/// <item>"adamw8bit": AdamW with its moments in 8 bits (Idrak's <c>AdamW8Bit</c>, registered by the Idrak package);</item>
/// <item>"sgd": <see cref="Sgd"/>, with <see cref="OptimizerSettings.Momentum"/> and an L2 term.</item>
/// </list>
/// Register another with <see cref="Register"/>; one under a library name takes the library's place, which
/// <see cref="Unregister"/> brings back. An optimizer holds its moments from the first step to the last and a checkpoint
/// saves them under its own names, so another kind cannot take over a run: the registry has no fallback.
/// </summary>
public static class OptimizerKinds
{
    // In the order adamw, adam, adamw8bit, sgd: core registers its 8-bit AdamW, as a library default, between the two it
    // builds on and SGD (Idrak.Abstraction cannot name it).
    private static readonly SlotTable<string, IOptimizerKind> Table = new(nameof(OptimizerKinds), comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "an optimizer holds its moments from the first step to the last and a checkpoint saves them under its own names, so another kind cannot take over a run");

    static OptimizerKinds()
    {
        Table.RegisterDefault("adamw", new BuiltInKind("adamw", "AdamW: Adam with decoupled weight decay (beta1, beta2, epsilon)",
            (ps, s) => new AdamW(ps, s.LearningRate, s.Beta1 ?? 0.9f, s.Beta2 ?? 0.999f, s.Epsilon ?? 1e-8f, s.WeightDecay)));
        Table.RegisterDefault("adam", new BuiltInKind("adam", "Adam: weight decay as an L2 term on the gradients (beta1, beta2, epsilon)",
            (ps, s) => new Adam(ps, s.LearningRate, s.Beta1 ?? 0.9f, s.Beta2 ?? 0.999f, s.Epsilon ?? 1e-8f, s.WeightDecay)));
        LibraryDefaults.Ensure(typeof(OptimizerKinds));
        Table.RegisterDefault("sgd", new BuiltInKind("sgd", "SGD with momentum and weight decay as an L2 term",
            (ps, s) => new Sgd(ps, s.LearningRate, s.Momentum, s.WeightDecay)));
    }

    /// <summary>Registers <paramref name="kind"/> under its <see cref="IOptimizerKind.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IOptimizerKind kind)
    {
        ArgumentNullException.ThrowIfNull(kind);
        ArgumentException.ThrowIfNullOrWhiteSpace(kind.Name);
        Table.Register(kind.Name, kind, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's kind <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in order.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The kind registered as <paramref name="name"/>, or null.</summary>
    public static IOptimizerKind? Find(string name) => Table.Find(name);

    /// <summary>The kind registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names the registered ones.</exception>
    public static IOptimizerKind Get(string name) => Table.Find(name)
        ?? throw new NotSupportedException($"Optimizer '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with OptimizerKinds.Register.");

    /// <summary>The library's kind <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IOptimizerKind? Default(string name) => Table.Default(name);

    /// <summary>Who registered the kind <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered kind as "name: summary", one per line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(k => $"{k.Name}: {k.Summary}"));

    /// <summary>
    /// What makes the optimizer <paramref name="name"/> with <paramref name="settings"/> from a list of parameters (the
    /// form <c>FineTuningOptions.Optimizer</c> and a host optimizer take). The kind is looked up now, so an unknown name
    /// fails here, not when training starts.
    /// </summary>
    /// <exception cref="NotSupportedException">The kind is not registered.</exception>
    public static Func<IReadOnlyList<Tensor>, Optimizer> Factory(string name, OptimizerSettings settings)
    {
        ArgumentNullException.ThrowIfNull(settings);
        var kind = Get(name);
        return parameters => kind.Create(parameters, settings) ?? throw new InvalidOperationException($"The optimizer kind '{kind.Name}' made no optimizer.");
    }

    private sealed class BuiltInKind(string name, string summary, Func<IReadOnlyList<Tensor>, OptimizerSettings, Optimizer> create) : IOptimizerKind
    {
        public string Name => name;

        public string Summary => summary;

        public Optimizer Create(IReadOnlyList<Tensor> parameters, OptimizerSettings settings)
        {
            ArgumentNullException.ThrowIfNull(parameters);
            ArgumentNullException.ThrowIfNull(settings);
            return create(parameters, settings);
        }

        public override string ToString() => name;
    }
}
