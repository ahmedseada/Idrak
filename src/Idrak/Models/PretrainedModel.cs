// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Devices;
using Idrak.Layers;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

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

    /// <summary>
    /// Store the projections in the packed format registered under this name (<see cref="PackedWeight.FormatNames"/>; a
    /// format of your own is added with <see cref="PackedWeight.Register"/>). Takes precedence over Int8, Int4 and BFloat16.
    /// </summary>
    public string? PackedFormatName { get; init; }

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
/// <see cref="Tokenizer"/> and <see cref="ChatTemplate"/>. The network is an ordinary Idrak model: generate and chat with
/// it (Idrak.Nlp's <c>CreateGenerator</c> and <c>CreateChat</c>), fine-tune it with LoRA, quantize it, save it as a package.
/// </summary>
public sealed class PretrainedModel : IDisposable
{
    private PretrainedModel(string folder, JsonObject config, DecoderSpec spec, Sequential network, ITokenizer? tokenizer, ChatTemplate? template,
        IReadOnlyList<string> notes, int maxPositions, PretrainedArchitecture architecture, Device device, PretrainedVision? vision)
    {
        Architecture = architecture;
        Vision = vision;
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

    /// <summary>
    /// The model's own chat template (from tokenizer_config.json, read by a reader registered with
    /// <see cref="ChatTemplates"/>: Idrak.Nlp's Jinja templates), or null when the model has none or no reader reads it.
    /// </summary>
    public ChatTemplate? ChatTemplate { get; }

    /// <summary>Anything approximated while reading the model (see <see cref="PretrainedFamilies.CommonSpec"/>).</summary>
    public IReadOnlyList<string> Notes { get; }

    /// <summary>The longest sequence the loaded model supports.</summary>
    public int MaxPositions { get; }

    /// <summary>How the model's checkpoint names map to Idrak's (used to save adapters and weights back).</summary>
    public PretrainedArchitecture Architecture { get; }

    /// <summary>Where the model lives.</summary>
    public Device Device { get; }

    /// <summary>
    /// The vision part of a vision-language model as its family's registration read it (<see cref="VisionFamilies"/>: the
    /// encoder it builds, its prompt format and attention rule; read and checked but not built), or null for a text-only
    /// model. <see cref="Network"/> is the text decoder alone.
    /// </summary>
    public PretrainedVision? Vision { get; }

    /// <summary>
    /// Reads the model in <paramref name="folder"/> (a model folder, a .gguf file, or any path a format registered with
    /// <see cref="CheckpointFormats"/> reads). Every weight is read from disk one tensor at a time and (with
    /// <see cref="PretrainedOptions.Int8"/>) quantized on the host, so the model is never held twice.
    /// </summary>
    public static PretrainedModel Load(string folder, PretrainedOptions? options = null)
    {
        options ??= new PretrainedOptions();
        var format = CheckpointFormats.For(folder);
        folder = format.Prepare(folder);                            // a .gguf file: config, tokenizer and template from its metadata

        var config = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "config.json")))!.AsObject();
        string name = options.Architecture ?? (string?)config["architectures"]?[0]
            ?? throw new InvalidDataException("config.json names no architecture; pass PretrainedOptions.Architecture.");
        var architecture = PretrainedArchitectures.Get(name);
        // A vision-language checkpoint is read by its own vision family, by the same name; none registered is an error
        // (families are applications of the contracts, registered by whoever brings them; nothing falls back to another).
        var visionFamily = VisionFamilies.For(name, config);

        using var reader = format.Open(folder);
        if (architecture.ForCheckpoint is { } fit)
        {
            architecture = fit(reader.Names.ToHashSet(StringComparer.Ordinal));     // the naming this checkpoint uses
        }

        var notes = new List<string>();
        var spec = architecture.Spec(config, notes);
        int maxPositions = Math.Min(options.MaxPositions ?? spec.MaxPositions, spec.MaxPositions);
        using var adapter = options.MergeAdapter is { } adapterFolder ? new AdapterMerge(adapterFolder) : null;
        var weights = new CheckpointWeights(reader, architecture) { Adapter = adapter };
        var buildOptions = new DecoderBuildOptions { Device = options.Device, Int8 = options.Int8, BFloat16 = options.BFloat16, Int4 = options.Int4, PackedFormatName = options.PackedFormatName, MaxPositions = maxPositions };
        var network = architecture.Build is { } build
            ? build(new PretrainedBuildContext(config, spec, weights, reader, buildOptions, notes)) switch
            {
                null => throw new InvalidOperationException($"The architecture '{name}' built no network."),
                Sequential built => built,
                var layers => new Sequential(layers) { Name = "decoder" },
            }
            : spec.Build(weights, buildOptions);
        var vision = visionFamily?.Read(new VisionCheckpoint(name, config, reader, folder, () => format.Open(folder), notes));
        if (vision is not null)
        {
            weights.Used.UnionWith(vision.StoredTensors);
        }

        var unused = reader.Names.Where(k => !weights.Used.Contains(k) && !k.EndsWith("rotary_emb.inv_freq", StringComparison.Ordinal)).ToList();
        if (unused.Count > 0)
        {
            notes.Add($"{unused.Count} checkpoint tensors were not used (for example {string.Join(", ", unused.Take(3))}).");
        }

        // A vision part's trained modules (modules_to_save) are kept beside the merged weights, for CreateVisionEncoder.
        var visionTensors = adapter?.VisionTensors(vision?.StoredTensors);
        if (adapter is not null)
        {
            notes.Add($"adapter {options.MergeAdapter} merged into {adapter.Merged} weights{(visionTensors is { Count: > 0 } ? $", {visionTensors.Count} trained vision tensors kept" : "")}.");
            if (adapter.Merged == 0)
            {
                throw new InvalidDataException($"The adapter in {options.MergeAdapter} matches none of the model's weights.");
            }

            adapter.CheckAllUsed();
        }

        notes.InsertRange(0, format.Notes(folder));                 // where the weights come from, first

        var tokenizer = File.Exists(Path.Combine(folder, "tokenizer.json")) ? BpeTokenizer.Load(folder) : null;
        tokenizer?.PadVocabulary(spec.Vocabulary);
        var template = ChatTemplates.Load(folder, tokenizer);
        var model = new PretrainedModel(folder, config, spec, network, tokenizer, template, notes, maxPositions, architecture,
            options.Device ?? Idrak.Abstraction.Device.Default, vision);
        foreach (var (stored, value) in visionTensors ?? [])
        {
            model._vision[stored] = value;
        }

        return model;
    }

    // The trained vision tensors (a vision part's fine-tuned modules), by their checkpoint names, in the checkpoint's layout.
    private readonly Dictionary<string, (int[] Shape, float[] Values)> _vision = new(StringComparer.Ordinal);

    /// <summary>
    /// The names (as the checkpoint stores them) of the trained vision tensors the model carries: those an adapter folder
    /// brought (<see cref="LoadAdapter"/> or <see cref="PretrainedOptions.MergeAdapter"/>, PEFT's <c>modules_to_save</c>)
    /// or a fine-tune kept (<see cref="KeepTrainedVision"/>). <see cref="CreateVisionEncoder"/> puts them into each
    /// encoder it builds, and <see cref="SaveAdapter"/> writes them. Empty for most models.
    /// </summary>
    public IReadOnlyCollection<string> TrainedVisionTensors => _vision.Keys;

    /// <summary>
    /// The family's vision encoder (<see cref="PretrainedVision.CreateEncoder"/>) on <paramref name="options"/>' device
    /// (the model's when not given), with the model's trained vision tensors put in (<see cref="TrainedVisionTensors"/>,
    /// through the family's <see cref="IVisionTuningPart.Import"/>). An encoder whose trained part would hold packed
    /// weights is built with float32 weights instead (only then). Dispose it when done.
    /// </summary>
    /// <exception cref="InvalidOperationException">The model has no vision part.</exception>
    /// <exception cref="NotSupportedException">The vision family is no longer registered (the message names the registry).</exception>
    /// <exception cref="InvalidDataException">A trained vision tensor fits no part the family offers.</exception>
    public IVisionEncoder CreateVisionEncoder(VisionEncoderOptions? options = null)
    {
        var vision = Vision ?? throw new InvalidOperationException("The model has no vision part (a text-only checkpoint): it reads no images.");
        _ = VisionFamilies.Get(vision.Family);                                  // unregistered since the model was read: the registry's message
        options ??= new VisionEncoderOptions();
        if (options.Device is null)
        {
            options = options with { Device = Device };
        }

        var encoder = vision.CreateEncoder(options);
        if (_vision.Count == 0)
        {
            return encoder;
        }

        try
        {
            var tuning = vision as IVisionTuningPart
                ?? throw new InvalidDataException($"The model carries {_vision.Count} trained vision tensors, but the vision family {vision.Family} offers no trainable parts.");
            try
            {
                ImportVision(tuning, encoder);
            }
            catch (InvalidOperationException) when (options.Weights != EncoderWeights.Float32)
            {
                encoder.Dispose();
                encoder = vision.CreateEncoder(options with { Weights = EncoderWeights.Float32 });   // trained values need float32 weights
                ImportVision(tuning, encoder);
            }

            return encoder;
        }
        catch
        {
            encoder.Dispose();
            throw;
        }
    }

    private void ImportVision(IVisionTuningPart tuning, IVisionEncoder encoder)
    {
        var tensors = _vision.ToDictionary(p => p.Key, p => Tensor.From(p.Value.Values, p.Value.Shape, Idrak.Abstraction.Device.Cpu), StringComparer.Ordinal);
        try
        {
            var taken = tuning.Import(encoder, tensors).ToHashSet(StringComparer.Ordinal);
            var left = _vision.Keys.Where(k => !taken.Contains(k)).Order(StringComparer.Ordinal).ToList();
            if (left.Count > 0)
            {
                throw new InvalidDataException($"{left.Count} trained vision tensors fit no part the vision family {Vision!.Family} trains (for example {string.Join(", ", left.Take(3))}).");
            }
        }
        finally
        {
            foreach (var tensor in tensors.Values)
            {
                tensor.Dispose();
            }
        }
    }

    /// <summary>
    /// Keeps the values of the vision parts <paramref name="parts"/> of <paramref name="encoder"/> (a fine-tune's trained
    /// projector) as the model's trained vision tensors (<see cref="TrainedVisionTensors"/>, copied to the host in the
    /// checkpoint's names and layout through the family's <see cref="IVisionTuningPart.Export"/>), replacing those of the
    /// same names: <see cref="SaveAdapter"/> then writes them beside the adapters. Returns how many tensors were kept.
    /// </summary>
    /// <exception cref="InvalidOperationException">The model has no vision part, or its family offers no trainable parts.</exception>
    public int KeepTrainedVision(IVisionEncoder encoder, IEnumerable<string> parts)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(parts);
        var vision = Vision ?? throw new InvalidOperationException("The model has no vision part (a text-only checkpoint).");
        var asked = parts.Select(p => p?.Trim() ?? "").Where(p => p.Length > 0).Distinct(StringComparer.OrdinalIgnoreCase).ToList();
        var tuning = VisionTuningParts.For(vision, asked);
        int kept = 0;
        foreach (string part in asked)
        {
            foreach (var (name, tensor) in tuning!.Export(encoder, part))
            {
                using (tensor)
                {
                    _vision[name] = ([.. tensor.Shape], tensor.ToArray());
                }

                kept++;
            }
        }

        return kept;
    }

    /// <summary>
    /// Adds LoRA adapters (rank <paramref name="rank"/>, scale alpha / rank), or DoRA adapters with <paramref name="dora"/>
    /// (see <see cref="DoraAdapter"/>), to the projections named in <paramref name="targets"/> (q, k, v, o, gate, up,
    /// down, head) and freezes everything else. Returns how many were added.
    /// </summary>
    public int AddAdapters(int rank, float alpha, IEnumerable<string> targets, int seed = 0, bool dora = false)
    {
        var names = targets.ToHashSet(StringComparer.Ordinal);
        Func<Linear, bool> chosen = l => l.Name is { } name && names.Contains(name) && l.TiedTo is null;
        return dora ? Network.AddDora(rank, alpha, chosen, freezeBase: true, new Random(seed))
            : Network.AddLora(rank, alpha, chosen, freezeBase: true, new Random(seed));
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
    /// original checkpoint, and <see cref="LoadAdapter"/> reads back. DoRA adapters add their magnitude vectors
    /// (lora_magnitude_vector, [out]) and <c>use_dora: true</c>; a model cannot mix LoRA and DoRA adapters in one folder.
    /// The trained vision tensors (<see cref="TrainedVisionTensors"/>) are written as peft writes <c>modules_to_save</c>:
    /// base_model.model. + their checkpoint names, as the checkpoint lays them out, and the module they lie under listed.
    /// </summary>
    public void SaveAdapter(string folder)
    {
        Directory.CreateDirectory(folder);
        var tensors = new List<(string, int[], float[])>();
        var modules = new SortedSet<string>(StringComparer.Ordinal);
        int rank = 0;
        float alpha = 0f;
        bool? dora = null;
        foreach (var (path, module) in NamedModules())
        {
            if (module is not Linear { Adapter: { } any } linear)
            {
                continue;
            }

            var (a, b, magnitude, adapterRank, scale) = any switch
            {
                LoraAdapter l => (l.A, l.B, (Tensor?)null, l.Rank, l.Scale),
                DoraAdapter d => (d.A, d.B, d.Magnitude, d.Rank, d.Scale),
                _ => throw new InvalidOperationException($"{linear}: a {any.GetType().Name} has no PEFT layout; only LoRA and DoRA adapters are saved."),
            };
            if (dora is { } kind && kind != magnitude is not null)
            {
                throw new InvalidOperationException("The model has both LoRA and DoRA adapters; PEFT stores one kind per folder.");
            }

            dora = magnitude is not null;
            string weight = CheckpointName($"{path}.weight");
            string prefix = "base_model.model." + weight[..^".weight".Length];
            modules.Add(weight.Split('.')[^2]);
            (rank, alpha) = (adapterRank, scale * adapterRank);
            tensors.Add(($"{prefix}.lora_A.weight", [adapterRank, linear.InFeatures], HostParallel.Transpose(a.ToArray(), linear.InFeatures, adapterRank)));
            tensors.Add(($"{prefix}.lora_B.weight", [linear.OutFeatures, adapterRank], HostParallel.Transpose(b.ToArray(), adapterRank, linear.OutFeatures)));
            if (magnitude is not null)
            {
                tensors.Add(($"{prefix}.lora_magnitude_vector", [linear.OutFeatures], magnitude.ToArray()));
            }
        }

        if (tensors.Count == 0)
        {
            throw new InvalidOperationException("The model has no LoRA adapters to save.");
        }

        // The trained vision tensors as PEFT writes modules_to_save: base_model.model. + the checkpoint's name, as stored.
        foreach (var (name, (shape, values)) in _vision.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            tensors.Add(($"base_model.model.{name}", shape, values));
        }

        SafeTensorsWriter.Write(Path.Combine(folder, "adapter_model.safetensors"), tensors, SafeTensorType.F32,
            new Dictionary<string, string> { ["format"] = "pt" });
        var config = new JsonObject
        {
            ["peft_type"] = "LORA", ["task_type"] = "CAUSAL_LM", ["r"] = rank, ["lora_alpha"] = alpha, ["lora_dropout"] = 0.0,
            ["bias"] = "none", ["fan_in_fan_out"] = false, ["inference_mode"] = true, ["use_dora"] = dora == true,
            ["target_modules"] = new JsonArray([.. modules.Select(m => (JsonNode)m)]),
            ["base_model_name_or_path"] = (string?)Config["_name_or_path"] ?? Path.GetFileName(Path.TrimEndingDirectorySeparator(Folder)),
        };
        if (_vision.Count > 0)
        {
            config["modules_to_save"] = new JsonArray([.. SavedModules(_vision.Keys).Select(m => (JsonNode)m)]);
        }

        File.WriteAllText(Path.Combine(folder, "adapter_config.json"), config.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
    }

    // The modules PEFT's modules_to_save names for these tensors: the module all of them lie under (its last name, which
    // peft matches as a suffix: "multi_modal_projector"), else each tensor's own module.
    private static IEnumerable<string> SavedModules(IEnumerable<string> names)
    {
        var paths = names.Select(n => n.Split('.')[..^1]).ToList();
        int common = 0;
        while (paths.All(p => p.Length > common) && paths.Select(p => p[common]).Distinct(StringComparer.Ordinal).Count() == 1)
        {
            common++;
        }

        return common > 0 ? [paths[0][common - 1]] : paths.Where(p => p.Length > 0).Select(p => p[^1]).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal);
    }

    /// <summary>
    /// Reads LoRA adapters in the PEFT layout (from <see cref="SaveAdapter"/>, or trained with peft on the same base
    /// model) and attaches them to the matching projections. Returns how many layers received one. The configuration is
    /// checked first: another peft_type, or an option Idrak does not apply (per-module ranks, trained biases, modules to
    /// save of the language model), is refused with an error, as is a file holding tensors for layers the model does not
    /// adapt; use_rslora sets the scale to alpha / √r; use_dora loads DoRA adapters (<see cref="DoraAdapter"/>) with their
    /// magnitude vectors. A vision part's trained modules (modules_to_save named as its checkpoint tensors) become the
    /// model's <see cref="TrainedVisionTensors"/>, which <see cref="CreateVisionEncoder"/> puts into its encoders.
    /// </summary>
    public int LoadAdapter(string folder)
    {
        var config = PeftAdapterConfig.Read(folder);
        config.CheckModulesToSave(Vision?.StoredTensors);
        using var reader = SafeTensorsReader.Open(Path.Combine(folder, "adapter_model.safetensors"));
        var found = new List<(Linear Layer, string A, string B, string? Magnitude)>();
        var used = new HashSet<string>(StringComparer.Ordinal);
        var vision = PeftAdapterConfig.VisionTensors(reader, Vision?.StoredTensors, used);
        foreach (var (path, module) in NamedModules())
        {
            if (module is Linear linear && Architecture.TensorName($"{path}.weight") is { } weight
                && config.Names(reader, weight[..^".weight".Length]) is var (a, b, magnitude))
            {
                found.Add((linear, a, b, magnitude));
                used.UnionWith(new[] { a, b, magnitude }.OfType<string>());
            }
        }

        PeftAdapterConfig.CheckAllUsed(reader, used, folder);                   // before any layer changes
        foreach (var (linear, a, b, magnitude) in found)
        {
            int rank = reader.Tensors[a].Shape[0];
            bool fits = linear.Adapter switch
            {
                null => true,
                LoraAdapter l => magnitude is null && l.Rank == rank,
                DoraAdapter d => magnitude is not null && d.Rank == rank,
                _ => false,
            };
            if (!fits)
            {
                throw new InvalidDataException($"{linear} has an adapter of another kind or rank than the rank-{rank} {(magnitude is null ? "LoRA" : "DoRA")} adapter in {folder}.");
            }

            if (linear.Adapter is null)
            {
                if (magnitude is null)
                {
                    linear.AddLora(rank, config.Alpha, l => ReferenceEquals(l, linear), freezeBase: false);
                }
                else
                {
                    linear.AddDora(rank, config.Alpha, l => ReferenceEquals(l, linear), freezeBase: false);
                }
            }

            (Tensor down, Tensor up) = linear.Adapter switch
            {
                LoraAdapter l => (l.A, l.B),
                DoraAdapter d => (d.A, d.B),
                _ => throw new InvalidOperationException($"{linear} has no LoRA or DoRA adapter."),
            };
            if (linear.Adapter is LoraAdapter lora)
            {
                linear.Adapter = lora with { Scale = config.Scale };
            }
            else if (linear.Adapter is DoraAdapter old)
            {
                old.Magnitude.Load(reader.Read(magnitude!));
                linear.Adapter = new DoraAdapter(old.A, old.B, old.Magnitude, old.Rank, config.Scale);
                old.Dispose();
            }

            down.Load(HostParallel.Transpose(reader.Read(a), rank, linear.InFeatures));
            up.Load(HostParallel.Transpose(reader.Read(b), linear.OutFeatures, rank));
        }

        foreach (var (name, value) in vision)
        {
            _vision[name] = value;
        }

        return found.Count;
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
        // The header is written from the shapes; each tensor is copied to the host (and transposed) only when it is written.
        var tensors = new List<(string, int[], Func<float[]>)>();
        void Add(string name, Tensor tensor)
        {
            string stored = CheckpointName(name);
            if (Architecture.Transposed(name) && tensor.Rank == 2)
            {
                tensors.Add((stored, [tensor.Shape[1], tensor.Shape[0]], () => HostParallel.Transpose(tensor.ToArray(), tensor.Shape[0], tensor.Shape[1])));
            }
            else
            {
                tensors.Add((stored, [.. tensor.Shape], tensor.ToArray));
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
