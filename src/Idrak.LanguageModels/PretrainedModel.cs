using System.Text.Json.Nodes;
using Idrak.Generation;
using Idrak.Layers;

namespace Idrak.LanguageModels;

/// <summary>Settings for <see cref="PretrainedModel.Load"/>.</summary>
public sealed record PretrainedOptions
{
    /// <summary>Where the model is created (the default device when null).</summary>
    public Device? Device { get; init; }

    /// <summary>Store the projections as int8 (quantized as they are read; about a quarter of the float32 memory).</summary>
    public bool Int8 { get; init; }

    /// <summary>Store the projections as bfloat16: half the float32 memory, exact for bfloat16 checkpoints (most are).</summary>
    public bool BFloat16 { get; init; }

    /// <summary>Store the projections as 4-bit weights (one scale per 32 rows and column; about 5 bits per weight).</summary>
    public bool Int4 { get; init; }

    /// <summary>Longest sequence to support (sizes the rotary tables); the model's maximum when null.</summary>
    public int? MaxPositions { get; init; }

    /// <summary>The architecture to use instead of the one named in config.json.</summary>
    public string? Architecture { get; init; }

    /// <summary>
    /// A folder holding a PEFT LoRA adapter (from <see cref="PretrainedModel.SaveAdapter"/> or peft) to merge into the
    /// weights as they are read: the model then runs as fast as the base model, and int8 / int4 / bfloat16 weights are
    /// packed with the update included. Use <see cref="PretrainedModel.LoadAdapter"/> instead to keep the adapter separate
    /// (to train it further, or export it).
    /// </summary>
    public string? MergeAdapter { get; init; }
}

/// <summary>
/// A pretrained decoder-only language model read from a folder in the Hugging Face layout (config.json, safetensors
/// weights, tokenizer.json, tokenizer_config.json): the <see cref="Network"/> built from its <see cref="Spec"/>, its
/// <see cref="Tokenizer"/> and <see cref="ChatTemplate"/>. The network is an ordinary Idrak model: generate with
/// <see cref="CreateGenerator"/>, chat with <see cref="CreateChat"/>, fine-tune with LoRA, quantize, save as a package.
/// </summary>
public sealed class PretrainedModel : IDisposable
{
    private PretrainedModel(string folder, JsonObject config, DecoderSpec spec, Sequential network, ITokenizer? tokenizer, JinjaChatTemplate? template,
        IReadOnlyList<string> notes, int maxPositions, PretrainedArchitecture architecture, Device device)
    {
        Architecture = architecture;
        Device = device;
        Folder = folder;
        Config = config;
        Spec = spec;
        Network = network;
        Tokenizer = tokenizer;
        ChatTemplate = template;
        Notes = notes;
        MaxPositions = maxPositions;
    }

    /// <summary>The folder the model was read from.</summary>
    public string Folder { get; }

    /// <summary>The model's config.json.</summary>
    public JsonObject Config { get; }

    /// <summary>The architecture as Idrak describes it.</summary>
    public DecoderSpec Spec { get; }

    /// <summary>The model.</summary>
    public Sequential Network { get; }

    /// <summary>The tokenizer (from tokenizer.json), or null when the folder has none.</summary>
    public ITokenizer? Tokenizer { get; }

    /// <summary>The chat template (from tokenizer_config.json), or null when the model has none.</summary>
    public JinjaChatTemplate? ChatTemplate { get; }

    /// <summary>Anything approximated while reading the model (see <see cref="PretrainedArchitectures.CommonSpec"/>).</summary>
    public IReadOnlyList<string> Notes { get; }

    /// <summary>The longest sequence the loaded model supports.</summary>
    public int MaxPositions { get; }

    /// <summary>How the model's checkpoint names map to Idrak's (used to save adapters and weights back).</summary>
    public PretrainedArchitecture Architecture { get; }

    /// <summary>Where the model lives.</summary>
    public Device Device { get; }

    /// <summary>
    /// Reads the model in <paramref name="folder"/>. Every weight is read from disk one tensor at a time and (with
    /// <see cref="PretrainedOptions.Int8"/>) quantized on the host, so the model is never held twice.
    /// </summary>
    public static PretrainedModel Load(string folder, PretrainedOptions? options = null)
    {
        options ??= new PretrainedOptions();
        if (File.Exists(folder) && folder.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
        {
            folder = GgufModel.Prepare(folder);                     // config, tokenizer and template from the file's metadata
        }

        var config = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "config.json")))!.AsObject();
        string name = options.Architecture ?? (string?)config["architectures"]?[0]
            ?? throw new InvalidDataException("config.json names no architecture; pass PretrainedOptions.Architecture.");
        var architecture = PretrainedArchitectures.Get(name);
        var notes = new List<string>();
        var spec = architecture.Spec(config, notes);
        int maxPositions = Math.Min(options.MaxPositions ?? spec.MaxPositions, spec.MaxPositions);
        using ITensorStore reader = GgufModel.IsPrepared(folder) ? GgufModel.OpenTensors(folder) : SafeTensorsReader.Open(folder);
        using var adapter = options.MergeAdapter is { } adapterFolder ? new AdapterMerge(adapterFolder) : null;
        var weights = new CheckpointWeights(reader, architecture) { Adapter = adapter };
        var network = spec.Build(weights, new DecoderBuildOptions { Device = options.Device, Int8 = options.Int8, BFloat16 = options.BFloat16, Int4 = options.Int4, MaxPositions = maxPositions });
        var unused = reader.Names.Where(k => !weights.Used.Contains(k) && !k.EndsWith("rotary_emb.inv_freq", StringComparison.Ordinal)).ToList();
        if (unused.Count > 0)
        {
            notes.Add($"{unused.Count} checkpoint tensors were not used (for example {string.Join(", ", unused.Take(3))}).");
        }

        if (adapter is not null)
        {
            notes.Add($"adapter {options.MergeAdapter} merged into {adapter.Merged} weights.");
            if (adapter.Merged == 0)
            {
                throw new InvalidDataException($"The adapter in {options.MergeAdapter} matches none of the model's weights.");
            }
        }

        if (GgufModel.IsPrepared(folder))
        {
            notes.Insert(0, $"weights read from {GgufModel.SourceOf(folder)}");
            notes.AddRange(GgufModel.NotesOf(folder));
        }

        var tokenizer = File.Exists(Path.Combine(folder, "tokenizer.json")) ? BpeTokenizer.Load(folder) : null;
        tokenizer?.PadVocabulary(spec.Vocabulary);
        var template = JinjaChatTemplate.Load(folder, tokenizer);
        return new PretrainedModel(folder, config, spec, network, tokenizer, template, notes, maxPositions, architecture,
            options.Device ?? Idrak.Device.Default);
    }

    /// <summary>A text generator for the model (int8 KV cache with <paramref name="cacheFormat"/>).</summary>
    public TextGenerator CreateGenerator(KeyValueFormat cacheFormat = KeyValueFormat.Float32, int? contextLength = null)
    {
        Network.Eval();
        return new TextGenerator(Network, Tokenizer ?? throw new InvalidOperationException("The model has no tokenizer."),
            Math.Min(contextLength ?? MaxPositions, MaxPositions)) { CacheFormat = cacheFormat };
    }

    /// <summary>A chat model using the model's own chat template (and its tool-call format).</summary>
    public ChatGenerator CreateChat(KeyValueFormat cacheFormat = KeyValueFormat.Float32, int? contextLength = null) =>
        new(CreateGenerator(cacheFormat, contextLength), ChatTemplate ?? throw new InvalidOperationException("The model has no chat template."));

    /// <summary>
    /// Adds LoRA adapters (rank <paramref name="rank"/>, scale alpha / rank) to the projections named in
    /// <paramref name="targets"/> (q, k, v, o, gate, up, down, head) and freezes everything else. Returns how many were added.
    /// </summary>
    public int AddAdapters(int rank, float alpha, IEnumerable<string> targets, int seed = 0)
    {
        var names = targets.ToHashSet(StringComparer.Ordinal);
        return Network.AddLora(rank, alpha, l => l.Name is { } name && names.Contains(name) && l.TiedTo is null, freezeBase: true, new Random(seed));
    }

    /// <summary>The model's modules with their Idrak paths (layers.3.attn.q, norm, …), which name their weights.</summary>
    public IEnumerable<(string Path, Module Module)> NamedModules()
    {
        IEnumerable<(string, Module)> Walk(Module module, string path)
        {
            yield return (path, module);
            foreach (var child in module.Children())
            {
                foreach (var item in Walk(child, $"{path}.{child.Name}"))
                {
                    yield return item;
                }
            }
        }

        return Network.Children().SelectMany(child => Walk(child, child.Name ?? ""));
    }

    // The checkpoint name of a Idrak weight name.
    private string CheckpointName(string name) =>
        Architecture.TensorName(name) ?? throw new InvalidOperationException($"The architecture has no checkpoint name for '{name}'.");

    /// <summary>
    /// Writes the LoRA adapters in the PEFT layout (adapter_model.safetensors with base_model.model.… names, A as
    /// [rank, in] and B as [out, rank]; adapter_config.json), which transformers / peft / vLLM load on top of the
    /// original checkpoint, and <see cref="LoadAdapter"/> reads back.
    /// </summary>
    public void SaveAdapter(string folder)
    {
        Directory.CreateDirectory(folder);
        var tensors = new List<(string, int[], float[])>();
        var modules = new SortedSet<string>(StringComparer.Ordinal);
        int rank = 0;
        float alpha = 0f;
        foreach (var (path, module) in NamedModules())
        {
            if (module is not Linear { Adapter: { } adapter } linear)
            {
                continue;
            }

            string weight = CheckpointName($"{path}.weight");
            string prefix = "base_model.model." + weight[..^".weight".Length];
            modules.Add(weight.Split('.')[^2]);
            (rank, alpha) = (adapter.Rank, adapter.Scale * adapter.Rank);
            tensors.Add(($"{prefix}.lora_A.weight", [adapter.Rank, linear.InFeatures], HostParallel.Transpose(adapter.A.ToArray(), linear.InFeatures, adapter.Rank)));
            tensors.Add(($"{prefix}.lora_B.weight", [linear.OutFeatures, adapter.Rank], HostParallel.Transpose(adapter.B.ToArray(), adapter.Rank, linear.OutFeatures)));
        }

        if (tensors.Count == 0)
        {
            throw new InvalidOperationException("The model has no LoRA adapters to save.");
        }

        SafeTensorsWriter.Write(Path.Combine(folder, "adapter_model.safetensors"), tensors, SafeTensorType.F32,
            new Dictionary<string, string> { ["format"] = "pt" });
        var config = new JsonObject
        {
            ["peft_type"] = "LORA", ["task_type"] = "CAUSAL_LM", ["r"] = rank, ["lora_alpha"] = alpha, ["lora_dropout"] = 0.0,
            ["bias"] = "none", ["fan_in_fan_out"] = false, ["inference_mode"] = true,
            ["target_modules"] = new JsonArray([.. modules.Select(m => (JsonNode)m)]),
            ["base_model_name_or_path"] = (string?)Config["_name_or_path"] ?? Path.GetFileName(Path.TrimEndingDirectorySeparator(Folder)),
        };
        File.WriteAllText(Path.Combine(folder, "adapter_config.json"), config.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
    }

    /// <summary>
    /// Reads LoRA adapters in the PEFT layout (from <see cref="SaveAdapter"/>, or trained with peft on the same base
    /// model) and attaches them to the matching projections. Returns how many layers received one.
    /// </summary>
    public int LoadAdapter(string folder)
    {
        var config = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "adapter_config.json")))!.AsObject();
        int rank = (int?)config["r"] ?? throw new InvalidDataException("adapter_config.json has no r.");
        float alpha = (float?)config["lora_alpha"] ?? rank;
        using var reader = SafeTensorsReader.Open(Path.Combine(folder, "adapter_model.safetensors"));
        int loaded = 0;
        foreach (var (path, module) in NamedModules().ToList())
        {
            if (module is not Linear linear || Architecture.TensorName($"{path}.weight") is not { } weight)
            {
                continue;
            }

            string prefix = "base_model.model." + weight[..^".weight".Length];
            string a = $"{prefix}.lora_A.weight", b = $"{prefix}.lora_B.weight";
            if (!reader.Contains(a))
            {
                (a, b) = ($"{prefix}.lora_A.default.weight", $"{prefix}.lora_B.default.weight");
                if (!reader.Contains(a))
                {
                    continue;
                }
            }

            if (linear.Adapter is null)
            {
                linear.AddLora(rank, alpha, l => ReferenceEquals(l, linear), freezeBase: false);
            }

            var adapter = linear.Adapter!;
            adapter.A.Load(HostParallel.Transpose(reader.Read(a), adapter.Rank, linear.InFeatures));
            adapter.B.Load(HostParallel.Transpose(reader.Read(b), linear.OutFeatures, adapter.Rank));
            loaded++;
        }

        return loaded;
    }

    /// <summary>
    /// Writes the model as a Hugging Face checkpoint (model.safetensors in <paramref name="type"/>, with config.json,
    /// the tokenizer and chat template files copied from the original folder): LoRA adapters are merged into the
    /// weights first. The model must hold float32 weights (load it without Int8 / Int4 / BFloat16 to export).
    /// </summary>
    public void SaveHuggingFace(string folder, SafeTensorType type = SafeTensorType.BF16)
    {
        if (Network.Descendants().OfType<Linear>().Any(l => l.Packed) || Network.Descendants().OfType<Embedding>().Any(e => e.BFloat16 is not null))
        {
            throw new InvalidOperationException("Exporting needs float32 weights: load the model without Int8, Int4 or BFloat16 (then LoadAdapter) to merge and save.");
        }

        Network.MergeLora();
        Directory.CreateDirectory(folder);
        var tensors = new List<(string, int[], float[])>();
        void Add(string name, Tensor tensor)
        {
            string stored = CheckpointName(name);
            var values = tensor.ToArray();
            if (Architecture.Transposed(name) && tensor.Rank == 2)
            {
                tensors.Add((stored, [tensor.Shape[1], tensor.Shape[0]], HostParallel.Transpose(values, tensor.Shape[0], tensor.Shape[1])));
            }
            else
            {
                tensors.Add((stored, [.. tensor.Shape], values));
            }
        }

        foreach (var (path, module) in NamedModules())
        {
            switch (module)
            {
                case Linear { TiedTo: not null }:
                    break;
                case Linear linear:
                    Add($"{path}.weight", linear.Weight);
                    if (linear.Bias is { } bias)
                    {
                        Add($"{path}.bias", bias);
                    }

                    break;
                case Embedding embedding:
                    Add($"{path}.weight", embedding.Weight);
                    break;
                case RMSNorm norm:
                    Add($"{path}.weight", norm.Gain);
                    break;
                case LayerNorm:
                    throw new NotSupportedException("Exporting LayerNorm models is not supported yet.");
            }
        }

        SafeTensorsWriter.Write(Path.Combine(folder, "model.safetensors"), tensors, type, new Dictionary<string, string> { ["format"] = "pt" });
        foreach (string file in Directory.GetFiles(Folder))
        {
            string name = Path.GetFileName(file);
            bool metadata = name.EndsWith(".json", StringComparison.OrdinalIgnoreCase) && name != "model.safetensors.index.json"
                || name.EndsWith(".jinja", StringComparison.OrdinalIgnoreCase) || name.EndsWith(".model", StringComparison.OrdinalIgnoreCase)
                || name is "merges.txt" or "vocab.txt";
            if (metadata && !name.StartsWith("adapter_", StringComparison.Ordinal))
            {
                File.Copy(file, Path.Combine(folder, name), overwrite: true);
            }
        }
    }

    /// <inheritdoc />
    public void Dispose() => Network.Dispose();
}
