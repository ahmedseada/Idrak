// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.CompilerServices;

namespace Idrak.Text.Abstractions;

/// <summary>
/// A preference loss by name (<see cref="PreferenceLosses"/>): what <see cref="Create"/> makes is an
/// <see cref="IFineTuningLoss"/> that needs <see cref="PreferencePair"/>s (DPO, ORPO, SimPO), with its options declared once
/// (<see cref="IOptionSchema.Options"/>), so an app, a setting or <c>idrak reinforce --loss</c> chooses one by name.
/// </summary>
public interface IPreferenceLoss : IConfigurableKind, IOptionSchema
{
    /// <summary>The loss with <paramref name="options"/> (null: its defaults).</summary>
    /// <exception cref="ArgumentException">An option it does not take (named, with its keys) or a bad value.</exception>
    IFineTuningLoss Create(IReadOnlyDictionary<string, string>? options = null);
}

/// <summary>
/// The preference losses by name (ignoring case): "dpo", "orpo" and "simpo" are the library's, over
/// <see cref="FineTuningLosses.Dpo"/>, <see cref="FineTuningLosses.Orpo"/> and <see cref="FineTuningLosses.SimPo"/> (the
/// factories stay); an app registers its own with <see cref="Register"/>, one under a library name shadowing the
/// library's until <see cref="Unregister"/>. A name nobody registered fails "not registered". Unguarded: the loss is the
/// objective a whole run trains on, so another cannot take over half-way.
/// </summary>
public static class PreferenceLosses
{
    /// <summary>Direct preference optimization against the reference: "dpo".</summary>
    public const string Dpo = "dpo";

    /// <summary>Odds ratio preference optimization (no reference): "orpo".</summary>
    public const string Orpo = "orpo";

    /// <summary>Simple preference optimization (no reference): "simpo".</summary>
    public const string SimPo = "simpo";

    private static readonly KindTable<IPreferenceLoss> Table = BuiltIn();

    private static KindTable<IPreferenceLoss> BuiltIn()
    {
        var table = new KindTable<IPreferenceLoss>(nameof(PreferenceLosses), "Preference loss", "add it with PreferenceLosses.Register",
            comparer: StringComparer.OrdinalIgnoreCase, unguarded: "a preference loss is the objective a whole run trains on, so another cannot take over half-way");
        var beta = OptionSpec.Number("beta", 0.1f, 1e-6f, takes: "a positive number", summary: "how far the model may move from the reference (larger: closer to it)");
        var smoothing = new OptionSpec<float>("label-smoothing", 0f, "a number from 0 up to (not including) 0.5", "the probability a preference is wrong (0: plain DPO)",
            (ReadOnlySpan<char> text, out float value) => float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out value) && value is >= 0f and < 0.5f, "0");
        table.RegisterDefault(new Builtin(Dpo, "direct preference optimization against the reference (the previous adapter, or the base model)",
            new OptionSchema("preference loss dpo", beta, smoothing), (schema, v) => FineTuningLosses.Dpo(v.Get(beta), v.Get(smoothing))));
        var lambda = OptionSpec.Number("lambda", 0.1f, 0f, takes: "a number of at least 0", summary: "the weight of the odds-ratio term");
        table.RegisterDefault(new Builtin(Orpo, "odds ratio preference optimization: the chosen answer's likelihood and an odds-ratio term (no reference)",
            new OptionSchema("preference loss orpo", lambda), (schema, v) => FineTuningLosses.Orpo(v.Get(lambda))));
        var simBeta = OptionSpec.Number("beta", 2f, 1e-6f, takes: "a positive number", summary: "the reward scale");
        var gamma = OptionSpec.Number("gamma", 1f, takes: "a number", summary: "the target reward margin");
        table.RegisterDefault(new Builtin(SimPo, "simple preference optimization: length-normalized rewards with a target margin (no reference)",
            new OptionSchema("preference loss simpo", simBeta, gamma), (schema, v) => FineTuningLosses.SimPo(v.Get(simBeta), v.Get(gamma))));
        return table;
    }

    /// <summary>Registers <paramref name="loss"/> under its <see cref="IKind.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IPreferenceLoss loss) => Table.Register(loss, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's loss <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Names;

    /// <summary>The loss registered as <paramref name="name"/>, or null.</summary>
    public static IPreferenceLoss? Find(ReadOnlySpan<char> name) => Table.Find(name);

    /// <summary>The loss registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotRegisteredException">None is: the message names the registered ones.</exception>
    public static IPreferenceLoss Get(ReadOnlySpan<char> name) => Table.Get(name);

    /// <summary>The library's loss <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IPreferenceLoss? Default(string name) => Table.Default(name);

    /// <summary>Who registered the loss <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered loss as "name: summary", one per line.</summary>
    public static string Describe() => Table.Describe();

    // A library loss: its schema and how its parsed options make the loss.
    private sealed class Builtin(string name, string summary, OptionSchema options, Func<OptionSchema, OptionValues, IFineTuningLoss> make) : IPreferenceLoss
    {
        public string Name => name;

        public string Summary => summary;

        public OptionSchema Options => options;

        public IReadOnlySet<string> Keys => options.Keys;

        public IFineTuningLoss Create(IReadOnlyDictionary<string, string>? options = null) => make(Options, Options.Parse(options));
    }
}
