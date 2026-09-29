using System.Diagnostics.CodeAnalysis;
using System.IO.Compression;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization.Metadata;
using Idrak.Data;
using Idrak.Generation;
using Idrak.Layers;

namespace Idrak.Inference;

/// <summary>
/// One file holding everything a model needs: a standard .zip whose entries use the library's existing formats
/// (weights as written by <see cref="Module.Save(Stream)"/>, scalers as written by <see cref="StandardScaler.Save(TextWriter)"/>,
/// tokenizers as JSON) plus a manifest. Only what you add goes in.
/// </summary>
/// <example>
/// <code>
/// ModelPackage.Create("house-price.ikm")
///     .Architecture(network)                  // a NetworkBuilder, so loading can rebuild the layers
///     .Weights(model)
///     .Scaler("features", featureScaler)
///     .Scaler("price", priceScaler)
///     .Save();
///
/// using var package = ModelPackage.Open("house-price.ikm");
/// using var model = package.BuildNetwork();   // or: build it yourself, then package.LoadWeights(model)
/// var featureScaler = package.StandardScaler("features");
/// </code>
/// </example>
public static class ModelPackage
{
    internal const string Format = "idrak-package/1";

    /// <summary>The name used by the overloads without a name (<see cref="ModelPackageWriter.Weights(Module)"/> and others).</summary>
    public const string DefaultModelName = "model";

    /// <summary>Starts a package that <see cref="ModelPackageWriter.Save"/> writes to <paramref name="path"/>.</summary>
    public static ModelPackageWriter Create(string path) => new(path);

    /// <summary>Opens a package for reading.</summary>
    public static ModelPackageReader Open(string path) => new(ZipFile.OpenRead(path), path);
}

/// <summary>The kind of an entry in a model package.</summary>
public enum PackageEntryKind
{
    /// <summary>All parameters and buffers of a module (<see cref="Module.Save(Stream)"/>).</summary>
    Weights,

    /// <summary>Only trainable parameters (<see cref="ModuleExtensions.SaveTrainable(Module, Stream)"/>).</summary>
    TrainableWeights,

    /// <summary>A network description (<see cref="NetworkBuilder.ToJson"/>).</summary>
    Architecture,

    /// <summary>A <see cref="Data.StandardScaler"/>.</summary>
    StandardScaler,

    /// <summary>A <see cref="Data.MinMaxScaler"/>.</summary>
    MinMaxScaler,

    /// <summary>A <see cref="Generation.CharTokenizer"/> or <see cref="Generation.WordTokenizer"/>.</summary>
    Tokenizer,

    /// <summary>Any JSON document.</summary>
    Json,

    /// <summary>Any text.</summary>
    Text,
}

/// <summary>An entry of a package: its kind and name.</summary>
public sealed record PackageEntry(PackageEntryKind Kind, string Name);

/// <summary>Collects the entries of a package; <see cref="Save"/> writes the file. Nothing is written until then.</summary>
public sealed class ModelPackageWriter
{
    private readonly string _path;
    private readonly List<(PackageEntry Entry, Action<Stream> Write)> _entries = [];

    internal ModelPackageWriter(string path) => _path = path;

    /// <summary>The entries added so far.</summary>
    public IReadOnlyList<PackageEntry> Entries => [.. _entries.Select(e => e.Entry)];

    /// <summary>The model's weights under the name "model" (<see cref="Module.Save(Stream)"/>; written at <see cref="Save"/>).</summary>
    public ModelPackageWriter Weights(Module model) => Weights(ModelPackage.DefaultModelName, model);

    /// <summary>A module's weights under <paramref name="name"/>.</summary>
    public ModelPackageWriter Weights(string name, Module model) => Add(PackageEntryKind.Weights, name, model.Save);

    /// <summary>A module's weights under <paramref name="name"/> in <paramref name="format"/> (Float16/BFloat16 halve them; int8 layers stay int8).</summary>
    public ModelPackageWriter Weights(string name, Module model, WeightFormat format) => Add(PackageEntryKind.Weights, name, s => model.Save(s, format));

    /// <summary>Only the trainable parameters of <paramref name="model"/> (for example LoRA adapters or a new head).</summary>
    public ModelPackageWriter TrainableWeights(string name, Module model) => Add(PackageEntryKind.TrainableWeights, name, model.SaveTrainable);

    /// <summary>The builder's description under the name "model", so <see cref="ModelPackageReader.BuildNetwork"/> can rebuild the layers.</summary>
    public ModelPackageWriter Architecture(NetworkBuilder network) => Architecture(ModelPackage.DefaultModelName, network.ToJson());

    /// <summary>A network description under <paramref name="name"/> (from <see cref="NetworkBuilder.ToJson"/> or <see cref="Network.ArchitectureOf"/>).</summary>
    public ModelPackageWriter Architecture(string name, JsonObject description) =>
        Add(PackageEntryKind.Architecture, name, s => WriteJson(s, description));

    /// <summary>A <see cref="Data.StandardScaler"/> or <see cref="Data.MinMaxScaler"/> in its text format.</summary>
    public ModelPackageWriter Scaler(string name, IScaler scaler) => scaler switch
    {
        StandardScaler standard => Add(PackageEntryKind.StandardScaler, name, s => WriteText(s, standard.Save)),
        MinMaxScaler minMax => Add(PackageEntryKind.MinMaxScaler, name, s => WriteText(s, minMax.Save)),
        _ => throw new NotSupportedException($"{scaler.GetType().Name} cannot be stored; StandardScaler and MinMaxScaler can."),
    };

    /// <summary>A <see cref="Generation.CharTokenizer"/> or <see cref="Generation.WordTokenizer"/>.</summary>
    public ModelPackageWriter Tokenizer(string name, ITokenizer tokenizer)
    {
        if (tokenizer is not (CharTokenizer or WordTokenizer))
        {
            throw new NotSupportedException($"{tokenizer.GetType().Name} cannot be stored; CharTokenizer and WordTokenizer can.");
        }

        return Add(PackageEntryKind.Tokenizer, name, s => Tokenizers.Save(tokenizer, s));
    }

    /// <summary>A JSON document (settings, class names, a threshold, training notes).</summary>
    public ModelPackageWriter Json(string name, JsonNode json) => Add(PackageEntryKind.Json, name, s => WriteJson(s, json));

    /// <summary>A JSON object (an exact overload, so it is not taken by the reflection-based <c>Json&lt;T&gt;</c>).</summary>
    public ModelPackageWriter Json(string name, JsonObject json) => Json(name, (JsonNode)json);

    /// <summary>A JSON array (an exact overload, so it is not taken by the reflection-based <c>Json&lt;T&gt;</c>).</summary>
    public ModelPackageWriter Json(string name, JsonArray json) => Json(name, (JsonNode)json);

    /// <summary>A value serialized with <paramref name="typeInfo"/> (trimming and AOT safe).</summary>
    public ModelPackageWriter Json<T>(string name, T value, JsonTypeInfo<T> typeInfo) =>
        Add(PackageEntryKind.Json, name, s => JsonSerializer.Serialize(s, value, typeInfo));

    /// <summary>A value serialized with reflection (any object, including anonymous ones).</summary>
    [RequiresUnreferencedCode("Reflection-based JSON serialization; use the JsonTypeInfo overload for trimmed or AOT apps.")]
    [RequiresDynamicCode("Reflection-based JSON serialization; use the JsonTypeInfo overload for trimmed or AOT apps.")]
    public ModelPackageWriter Json<T>(string name, T value) =>
        Add(PackageEntryKind.Json, name, s => JsonSerializer.Serialize(s, value, new JsonSerializerOptions { WriteIndented = true }));

    /// <summary>Any text.</summary>
    public ModelPackageWriter Text(string name, string text) => Add(PackageEntryKind.Text, name, s => WriteText(s, w => w.Write(text)));

    /// <summary>Writes the package file (replacing an existing one).</summary>
    public void Save()
    {
        var manifest = new JsonObject
        {
            ["format"] = ModelPackage.Format,
            ["created"] = DateTimeOffset.UtcNow.ToString("O"),
            ["entries"] = new JsonArray([.. _entries.Select(e => (JsonNode)new JsonObject
            {
                ["kind"] = e.Entry.Kind.ToString(),
                ["name"] = e.Entry.Name,
                ["file"] = ModelPackageReader.FileName(e.Entry),
            })]),
        };

        string temporary = _path + ".tmp";
        using (var file = File.Create(temporary))
        using (var zip = new ZipArchive(file, ZipArchiveMode.Create))
        {
            using (var stream = zip.CreateEntry("manifest.json").Open())
            {
                WriteJson(stream, manifest);
            }

            foreach (var (entry, write) in _entries)
            {
                using var stream = zip.CreateEntry(ModelPackageReader.FileName(entry)).Open();
                write(stream);
            }
        }

        File.Move(temporary, _path, overwrite: true);
    }

    private ModelPackageWriter Add(PackageEntryKind kind, string name, Action<Stream> write)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        if (name.IndexOfAny(['/', '\\']) >= 0)
        {
            throw new ArgumentException("Entry names cannot contain '/' or '\\'.", nameof(name));
        }

        var entry = new PackageEntry(kind, name);
        if (_entries.Any(e => ModelPackageReader.FileName(e.Entry) == ModelPackageReader.FileName(entry)))
        {
            throw new ArgumentException($"The package already has a {kind} entry named '{name}'.", nameof(name));
        }

        _entries.Add((entry, write));
        return this;
    }

    private static void WriteJson(Stream stream, JsonNode json)
    {
        using var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true });
        json.WriteTo(writer);
    }

    private static void WriteText(Stream stream, Action<TextWriter> write)
    {
        using var writer = new StreamWriter(stream, new UTF8Encoding(false), leaveOpen: true);
        write(writer);
    }
}

/// <summary>Reads a package written by <see cref="ModelPackageWriter"/>. Dispose it to close the file.</summary>
public sealed class ModelPackageReader : IDisposable
{
    private readonly ZipArchive _zip;
    private readonly string _path;

    internal ModelPackageReader(ZipArchive zip, string path)
    {
        _zip = zip;
        _path = path;
        var manifest = zip.GetEntry("manifest.json") is { } m
            ? ReadJson(m) : throw new InvalidDataException($"{path} is not a Idrak package (no manifest.json).");
        if ((string?)manifest["format"] != ModelPackage.Format)
        {
            throw new InvalidDataException($"{path}: unsupported package format '{(string?)manifest["format"]}'.");
        }

        Entries = [.. manifest["entries"]!.AsArray().Select(e => new PackageEntry(Enum.Parse<PackageEntryKind>((string)e!["kind"]!), (string)e["name"]!))];
    }

    /// <summary>Everything in the package.</summary>
    public IReadOnlyList<PackageEntry> Entries { get; }

    /// <summary>Whether an entry of <paramref name="kind"/> named <paramref name="name"/> exists.</summary>
    public bool Contains(PackageEntryKind kind, string name) => Entries.Contains(new PackageEntry(kind, name));

    /// <summary>Loads the weights named "model" into <paramref name="model"/> (<see cref="Module.Load(Stream)"/>; the architecture must match).</summary>
    public void LoadWeights(Module model) => LoadWeights(model, ModelPackage.DefaultModelName);

    /// <summary>Loads the weights named <paramref name="name"/> into <paramref name="model"/>.</summary>
    public void LoadWeights(Module model, string name)
    {
        using var stream = OpenSeekable(PackageEntryKind.Weights, name);
        model.Load(stream);
    }

    /// <summary>Loads trainable parameters stored with <see cref="ModelPackageWriter.TrainableWeights"/>.</summary>
    public void LoadTrainableWeights(Module model, string name)
    {
        using var stream = OpenSeekable(PackageEntryKind.TrainableWeights, name);
        model.LoadTrainable(stream);
    }

    /// <summary>The network description named <paramref name="name"/>.</summary>
    public JsonObject Architecture(string name = ModelPackage.DefaultModelName) => (JsonObject)ReadJson(Find(PackageEntryKind.Architecture, name));

    /// <summary>A builder replaying the stored description (add settings such as <see cref="NetworkBuilder.OnDevice"/>, then build).</summary>
    public NetworkBuilder Network(string name = ModelPackage.DefaultModelName) => Layers.Network.FromJson(Architecture(name));

    /// <summary>Rebuilds the network from its stored description on <paramref name="device"/> and loads its weights.</summary>
    public Sequential BuildNetwork(string name = ModelPackage.DefaultModelName, Device? device = null)
    {
        var builder = Network(name);
        if (device is not null)
        {
            builder.OnDevice(device);
        }

        var model = builder.Build();
        try
        {
            LoadWeights(model, name);
            return model;
        }
        catch
        {
            model.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Rebuilds the model named <paramref name="name"/> on <paramref name="device"/> and loads its weights, whether its
    /// architecture was written by the network builder (a <see cref="Sequential"/>), is a <see cref="GraphModule"/> or a
    /// <see cref="DecoderSpec"/>.
    /// </summary>
    public Module BuildModel(string name = ModelPackage.DefaultModelName, Device? device = null)
    {
        var architecture = Architecture(name);
        if (DecoderSpec.IsDescription(architecture))
        {
            var decoder = DecoderSpec.FromJson(architecture).Build(options: new DecoderBuildOptions { Device = device });
            try
            {
                LoadWeights(decoder, name);
                return decoder;
            }
            catch
            {
                decoder.Dispose();
                throw;
            }
        }

        if (!GraphModule.IsDescription(architecture))
        {
            return BuildNetwork(name, device);
        }

        var graph = GraphModule.FromJson(architecture, device);
        try
        {
            LoadWeights(graph, name);
            return graph;
        }
        catch
        {
            graph.Dispose();
            throw;
        }
    }

    /// <summary>The <see cref="Data.StandardScaler"/> named <paramref name="name"/>.</summary>
    public StandardScaler StandardScaler(string name)
    {
        using var reader = new StreamReader(Find(PackageEntryKind.StandardScaler, name).Open());
        return Data.StandardScaler.Load(reader);
    }

    /// <summary>The <see cref="Data.MinMaxScaler"/> named <paramref name="name"/>.</summary>
    public MinMaxScaler MinMaxScaler(string name)
    {
        using var reader = new StreamReader(Find(PackageEntryKind.MinMaxScaler, name).Open());
        return Data.MinMaxScaler.Load(reader);
    }

    /// <summary>The scaler named <paramref name="name"/>, whichever kind it is.</summary>
    public IScaler Scaler(string name) =>
        Contains(PackageEntryKind.StandardScaler, name) ? StandardScaler(name)
        : Contains(PackageEntryKind.MinMaxScaler, name) ? MinMaxScaler(name)
        : throw new KeyNotFoundException($"{_path} has no scaler named '{name}'.");

    /// <summary>The tokenizer named <paramref name="name"/>.</summary>
    public ITokenizer Tokenizer(string name)
    {
        using var stream = OpenSeekable(PackageEntryKind.Tokenizer, name);
        return Tokenizers.Load(stream);
    }

    /// <summary>The <see cref="Generation.WordTokenizer"/> named <paramref name="name"/>.</summary>
    public WordTokenizer WordTokenizer(string name) => Tokenizer(name) as WordTokenizer
        ?? throw new InvalidDataException($"The tokenizer '{name}' is not a WordTokenizer.");

    /// <summary>The <see cref="Generation.CharTokenizer"/> named <paramref name="name"/>.</summary>
    public CharTokenizer CharTokenizer(string name) => Tokenizer(name) as CharTokenizer
        ?? throw new InvalidDataException($"The tokenizer '{name}' is not a CharTokenizer.");

    /// <summary>The JSON document named <paramref name="name"/>.</summary>
    public JsonNode Json(string name) => ReadJson(Find(PackageEntryKind.Json, name));

    /// <summary>The JSON document named <paramref name="name"/> as <typeparamref name="T"/> (trimming and AOT safe).</summary>
    public T Json<T>(string name, JsonTypeInfo<T> typeInfo)
    {
        using var stream = Find(PackageEntryKind.Json, name).Open();
        return JsonSerializer.Deserialize(stream, typeInfo)!;
    }

    /// <summary>The JSON document named <paramref name="name"/> as <typeparamref name="T"/>, using reflection.</summary>
    [RequiresUnreferencedCode("Reflection-based JSON deserialization; use the JsonTypeInfo overload for trimmed or AOT apps.")]
    [RequiresDynamicCode("Reflection-based JSON deserialization; use the JsonTypeInfo overload for trimmed or AOT apps.")]
    public T Json<T>(string name)
    {
        using var stream = Find(PackageEntryKind.Json, name).Open();
        return JsonSerializer.Deserialize<T>(stream)!;
    }

    /// <summary>The text named <paramref name="name"/>.</summary>
    public string Text(string name)
    {
        using var reader = new StreamReader(Find(PackageEntryKind.Text, name).Open());
        return reader.ReadToEnd();
    }

    /// <summary>
    /// A <see cref="Generation.TextGenerator"/> for a language model stored with its architecture: the network named
    /// <paramref name="model"/> is rebuilt with its weights, the tokenizer is <paramref name="tokenizer"/>, and the
    /// context length is the architecture's token length (<c>Network.Tokens(length)</c>).
    /// </summary>
    public TextGenerator TextGenerator(string tokenizer, string model = ModelPackage.DefaultModelName, Device? device = null)
    {
        if (DecoderSpec.IsDescription(Architecture(model)))
        {
            var spec = DecoderSpec.FromJson(Architecture(model));
            var decoder = (Sequential)BuildModel(model, device);
            decoder.Eval();
            return new TextGenerator(decoder, Tokenizer(tokenizer), spec.MaxPositions);
        }

        var network = Network(model);
        if (network.InputKind != InputKind.Tokens)
        {
            throw new InvalidOperationException($"The network '{model}' does not read tokens (it starts with {network.InputKind}).");
        }

        var tokens = Tokenizer(tokenizer);
        var built = BuildNetwork(model, device);
        built.Eval();
        return new TextGenerator(built, tokens, network.InputShape[0]);
    }

    /// <inheritdoc />
    public void Dispose() => _zip.Dispose();

    internal static string FileName(PackageEntry e) => e.Kind switch
    {
        PackageEntryKind.Weights => $"weights/{e.Name}.ikw",
        PackageEntryKind.TrainableWeights => $"trainable/{e.Name}.nsp",
        PackageEntryKind.Architecture => $"architecture/{e.Name}.json",
        PackageEntryKind.StandardScaler or PackageEntryKind.MinMaxScaler => $"scalers/{e.Name}.txt",
        PackageEntryKind.Tokenizer => $"tokenizers/{e.Name}.json",
        PackageEntryKind.Json => $"json/{e.Name}.json",
        _ => $"text/{e.Name}.txt",
    };

    private ZipArchiveEntry Find(PackageEntryKind kind, string name) =>
        Contains(kind, name) && _zip.GetEntry(FileName(new PackageEntry(kind, name))) is { } entry
            ? entry
            : throw new KeyNotFoundException($"{_path} has no {kind} entry named '{name}'. It contains: {string.Join(", ", Entries.Select(e => $"{e.Kind} '{e.Name}'"))}.");

    // Zip entry streams cannot seek; the binary readers only read forward, but copying keeps them independent of that.
    private MemoryStream OpenSeekable(PackageEntryKind kind, string name)
    {
        var buffer = new MemoryStream();
        using (var stream = Find(kind, name).Open())
        {
            stream.CopyTo(buffer);
        }

        buffer.Position = 0;
        return buffer;
    }

    private static JsonNode ReadJson(ZipArchiveEntry entry)
    {
        using var stream = entry.Open();
        return JsonNode.Parse(stream) ?? throw new InvalidDataException($"{entry.FullName} is empty.");
    }
}
