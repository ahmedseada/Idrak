// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;

namespace Idrak.Layers.Abstractions;

/// <summary>One size or choice a network recipe reads (see <see cref="INetworkRecipe.Parameters"/>).</summary>
/// <param name="Name">Its key in the recipe's arguments ("vocabulary", "dim").</param>
/// <param name="Meaning">What it sets.</param>
/// <param name="Default">The value used when it is left out, as text ("4 · dim"); null when it must be given.</param>
public sealed record NetworkRecipeParameter(string Name, string Meaning, string? Default = null);

/// <summary>
/// The arguments of a network recipe: a JSON object of sizes and choices (<c>{"vocabulary": 65, "dim": 96}</c>), read with
/// the typed helpers. A required argument that is missing fails naming the recipe and the argument.
/// </summary>
public sealed class NetworkRecipeArguments
{
    /// <summary>The arguments <paramref name="json"/> of the recipe <paramref name="recipe"/>.</summary>
    public NetworkRecipeArguments(string recipe, JsonObject json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(recipe);
        ArgumentNullException.ThrowIfNull(json);
        (Recipe, Json) = (recipe, json);
    }

    /// <summary>The recipe they are for.</summary>
    public string Recipe { get; }

    /// <summary>The arguments as given, for what the helpers do not cover.</summary>
    public JsonObject Json { get; }

    /// <summary>Whether the argument <paramref name="key"/> is given.</summary>
    public bool Has(string key) => Json[key] is not null;

    /// <summary>The integer argument <paramref name="key"/>; it must be given.</summary>
    public int Int(string key) => (int)Required(key);

    /// <summary>The integer argument <paramref name="key"/>, or <paramref name="fallback"/> when it is left out.</summary>
    public int Int(string key, int fallback) => Json[key] is { } value ? (int)value : fallback;

    /// <summary>The number argument <paramref name="key"/>; it must be given.</summary>
    public float Float(string key) => (float)Required(key);

    /// <summary>The number argument <paramref name="key"/>, or <paramref name="fallback"/> when it is left out.</summary>
    public float Float(string key, float fallback) => Json[key] is { } value ? (float)value : fallback;

    /// <summary>The integer array argument <paramref name="key"/> (hidden sizes, filter counts); it must be given.</summary>
    public int[] Ints(string key) => [.. Required(key).AsArray().Select(v => (int)v!)];

    /// <summary>The text argument <paramref name="key"/>, or <paramref name="fallback"/> when it is left out.</summary>
    public string Text(string key, string fallback) => Json[key] is { } value ? (string)value! : fallback;

    /// <summary>The argument <paramref name="key"/> as a value of <typeparamref name="T"/> by name (ignoring case), or <paramref name="fallback"/>.</summary>
    /// <exception cref="InvalidDataException">A name that is not one of the type's.</exception>
    public T Choice<T>(string key, T fallback) where T : struct, Enum =>
        Json[key] is not { } value ? fallback
        : Enum.TryParse<T>((string)value!, ignoreCase: true, out var parsed) && Enum.IsDefined(parsed) ? parsed
        : throw new InvalidDataException($"The network recipe '{Recipe}' takes {key} as one of {string.Join(", ", Enum.GetNames<T>()).ToLowerInvariant()}, not '{value}'.");

    private JsonNode Required(string key) => Json[key] ?? throw new InvalidDataException($"The network recipe '{Recipe}' needs the argument '{key}'.");
}

/// <summary>
/// A ready-made network (the Director of the builder pattern): from its sizes and choices, a fixed, documented sequence of
/// network builder steps. It returns the builder, so the device, the seed and the name are still set before
/// <see cref="NetworkBuilder.Build"/>. Register one with <see cref="NetworkRecipes.Register"/>.
/// </summary>
public interface INetworkRecipe
{
    /// <summary>Its name in <see cref="NetworkRecipes"/> ("gpt", "mlp").</summary>
    string Name { get; }

    /// <summary>One line: the layers it builds.</summary>
    string Summary { get; }

    /// <summary>The arguments it reads.</summary>
    IReadOnlyList<NetworkRecipeParameter> Parameters { get; }

    /// <summary>The builder with the recipe's steps for <paramref name="arguments"/>.</summary>
    NetworkBuilder Build(NetworkRecipeArguments arguments);
}

/// <summary>
/// The network recipes, by name (ignoring case). The library's are the recipes of <see cref="Architectures"/>, which stay
/// their implementations: "mlp", "cnn", "rnn", "transformer-classifier" and "gpt" (a decoder-only language model). Register
/// another with <see cref="Register"/>; one under a library name shadows the library's (guarded by its slot's policy, see
/// <see cref="SetPolicy"/>), which <see cref="Unregister"/> brings back.
/// </summary>
/// <example>
/// <code>
/// var model = NetworkRecipes.Build("gpt", new JsonObject { ["vocabulary"] = 65, ["context"] = 64, ["dim"] = 96, ["heads"] = 4, ["layers"] = 3 })
///     .OnDevice(device).Seed(2).Build();
/// </code>
/// </example>
public static class NetworkRecipes
{
    private static readonly SlotTable<string, INetworkRecipe> Table = new(nameof(NetworkRecipes),
        (slot, app, library) => new GuardedRecipe(slot, app, library), StringComparer.OrdinalIgnoreCase);

    static NetworkRecipes() => Overrides.AsLibraryDefaults(() =>
    {
        foreach (var recipe in LibraryRecipes())
        {
            Register(recipe);
        }
    });

    /// <summary>Registers <paramref name="recipe"/> under its <see cref="INetworkRecipe.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(INetworkRecipe recipe)
    {
        ArgumentNullException.ThrowIfNull(recipe);
        ArgumentException.ThrowIfNullOrWhiteSpace(recipe.Name);
        Table.Register(recipe.Name, recipe, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's recipe <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The recipe registered as <paramref name="name"/>, or null.</summary>
    public static INetworkRecipe? Find(string name) => Table.Find(name);

    /// <summary>The recipe registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names the registered ones.</exception>
    public static INetworkRecipe Get(string name) => Table.Find(name)
        ?? throw new NotSupportedException($"Network recipe '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with NetworkRecipes.Register.");

    /// <summary>The library's recipe <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static INetworkRecipe? Default(string name) => Table.Default(name);

    /// <summary>Who registered the recipe <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>
    /// What happens when the app's recipe <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/> builds the library's), or whether it only runs beside the library's
    /// (<see cref="SlotPolicy.Shadow"/>: the library's builder is used, the two compared by their steps).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary>Each registered recipe as "name(arguments): summary", one per line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(r =>
        $"{r.Name}({string.Join(", ", r.Parameters.Select(p => p.Default is null ? p.Name : $"{p.Name} = {p.Default}"))}): {r.Summary}"));

    /// <summary>The builder of the recipe <paramref name="name"/> for <paramref name="arguments"/>.</summary>
    /// <exception cref="NotSupportedException">No recipe has that name.</exception>
    /// <exception cref="InvalidDataException">A required argument is missing or a choice is unknown.</exception>
    public static NetworkBuilder Build(string name, JsonObject arguments)
    {
        ArgumentNullException.ThrowIfNull(arguments);
        return Get(name).Build(new NetworkRecipeArguments(name, arguments));
    }

    // The recipes of Architectures, by name: their static methods are the implementations.
    private static IEnumerable<INetworkRecipe> LibraryRecipes()
    {
        static NetworkRecipeParameter P(string name, string meaning, string? fallback = null) => new(name, meaning, fallback);
        yield return new LibraryRecipe("mlp", "Input(inputs) \u2192 for each hidden size: Linear, activation (and Dropout when dropout is given) \u2192 Linear(outputs)",
            [P("inputs", "features of a row"), P("hidden", "the hidden layers' sizes, in order"), P("outputs", "the last layer's size"),
                P("activation", "relu, tanh, sigmoid or gelu", "relu"), P("dropout", "the dropout after each activation", "none")],
            a => a.Has("dropout")
                ? Architectures.Mlp(a.Int("inputs"), a.Ints("hidden"), a.Int("outputs"), a.Choice("activation", Activation.ReLU), a.Float("dropout"))
                : Architectures.Mlp(a.Int("inputs"), a.Ints("hidden"), a.Int("outputs"), a.Choice("activation", Activation.ReLU)));
        yield return new LibraryRecipe("cnn", "Image(channels, height, width) \u2192 for each filter count: Conv2d, BatchNorm, ReLU, MaxPool2d(2) \u2192 Flatten \u2192 Linear(outputs)",
            [P("channels", "image channels"), P("height", "image height"), P("width", "image width"), P("filters", "each convolution's filters, in order"),
                P("kernelSize", "the convolutions' kernel side", "3"), P("outputs", "the last layer's size")],
            a => Architectures.Cnn(a.Int("channels"), a.Int("height"), a.Int("width"), a.Ints("filters"), a.Int("kernelSize", 3), a.Int("outputs")));
        yield return new LibraryRecipe("rnn", "Tokens(length) \u2192 Embedding(vocabulary, embed) \u2192 LSTM or GRU(hidden) \u2192 Linear(outputs)",
            [P("cell", "lstm or gru", "lstm"), P("vocabulary", "token ids"), P("length", "tokens a sequence"), P("embed", "the embedding's width"),
                P("hidden", "the recurrent layer's width"), P("outputs", "the last layer's size")],
            a => Architectures.Rnn(a.Choice("cell", RecurrentCell.LSTM), a.Int("vocabulary"), a.Int("length"), a.Int("embed"), a.Int("hidden"), a.Int("outputs")));
        yield return new LibraryRecipe("transformer-classifier",
            "Tokens(length) \u2192 Embedding \u2192 PositionalEncoding \u2192 layers \u00D7 TransformerEncoderLayer \u2192 LayerNorm \u2192 MeanOverTime \u2192 Linear(outputs)",
            [P("vocabulary", "token ids"), P("length", "tokens a sequence"), P("dim", "the model's width"), P("heads", "attention heads"), P("layers", "encoder layers"),
                P("ffDim", "the feed-forward width", "4 \u00B7 dim"), P("dropout", "the layers' dropout", "0.1"), P("outputs", "classes")],
            a => Architectures.TransformerClassifier(a.Int("vocabulary"), a.Int("length"), a.Int("dim"), a.Int("heads"), a.Int("layers"),
                a.Int("ffDim", 4 * a.Int("dim")), a.Float("dropout", 0.1f), a.Int("outputs")));
        yield return new LibraryRecipe("gpt",
            "a decoder-only language model: Tokens(context) \u2192 Embedding(vocabulary, dim) \u2192 PositionalEncoding \u2192 layers \u00D7 causal TransformerEncoderLayer \u2192 LayerNorm \u2192 Linear(vocabulary)",
            [P("vocabulary", "token ids"), P("context", "tokens the model sees"), P("dim", "the model's width"), P("heads", "attention heads"), P("layers", "decoder layers"),
                P("ffDim", "the feed-forward width", "4 \u00B7 dim"), P("dropout", "the layers' dropout", "0.1")],
            a => Architectures.Gpt(a.Int("vocabulary"), a.Int("context"), a.Int("dim"), a.Int("heads"), a.Int("layers"), a.Int("ffDim", 4 * a.Int("dim")), a.Float("dropout", 0.1f)));
    }

    private sealed class LibraryRecipe(string name, string summary, NetworkRecipeParameter[] parameters, Func<NetworkRecipeArguments, NetworkBuilder> build) : INetworkRecipe
    {
        public string Name => name;

        public string Summary => summary;

        public IReadOnlyList<NetworkRecipeParameter> Parameters => parameters;

        public NetworkBuilder Build(NetworkRecipeArguments arguments)
        {
            ArgumentNullException.ThrowIfNull(arguments);
            return build(arguments);
        }
    }

    // An app's recipe over a library one: a failing build falls back on its own, or the two are compared by their steps.
    private sealed class GuardedRecipe(Slot slot, INetworkRecipe app, INetworkRecipe library) : INetworkRecipe
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public IReadOnlyList<NetworkRecipeParameter> Parameters => app.Parameters;

        public NetworkBuilder Build(NetworkRecipeArguments arguments) =>
            slot.Call(() => app.Build(arguments), () => library.Build(arguments), static (a, b) =>
                !a.IsDescribable || !b.IsDescribable ? null
                : JsonNode.DeepEquals(a.ToJson(), b.ToJson()) ? null
                : string.Create(CultureInfo.InvariantCulture, $"{a.Count} steps against the library's {b.Count}: {a.ToJson().ToJsonString()} against {b.ToJson().ToJsonString()}"));
    }
}
