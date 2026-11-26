// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Formats;
using Idrak.Layers;
using Idrak.Layers.Abstractions;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// The files a model made from a spec carries beside its weights (<see cref="PretrainedModel.Create"/>): its tokenizer,
/// chat template and the tokens that start, end and pad text.
/// </summary>
public sealed record ModelFiles
{
    /// <summary>
    /// The template created models get when none is given: ChatML turns (<c>&lt;|im_start|&gt;role\n … &lt;|im_end|&gt;\n</c>),
    /// content as a string or a list of parts, an image part written as <c>&lt;image&gt;</c> and an audio part as <c>&lt;audio&gt;</c> (a family's
    /// marker), and the assistant's header when a generation prompt is asked for. The tokenizer needs
    /// <c>&lt;|im_start|&gt;</c> and <c>&lt;|im_end|&gt;</c> as tokens.
    /// </summary>
    public const string ChatMLWithParts =
        "{%- for message in messages -%}\n"
        + "{{- '<|im_start|>' + message['role'] + '\\n' -}}\n"
        + "{%- if message['content'] is string -%}\n"
        + "{{- message['content'] -}}\n"
        + "{%- else -%}\n"
        + "{%- for item in message['content'] -%}\n"
        + "{%- if item['type'] == 'image' -%}\n"
        + "{{- '<image>' -}}\n"
        + "{%- elif item['type'] == 'audio' -%}\n"
        + "{{- '<audio>' -}}\n"
        + "{%- elif item['type'] == 'text' -%}\n"
        + "{{- item['text'] -}}\n"
        + "{%- endif -%}\n"
        + "{%- endfor -%}\n"
        + "{%- endif -%}\n"
        + "{{- '<|im_end|>\\n' -}}\n"
        + "{%- endfor -%}\n"
        + "{%- if add_generation_prompt -%}\n"
        + "{{- '<|im_start|>assistant\\n' -}}\n"
        + "{%- endif -%}\n";

    /// <summary>The tokenizer (one <see cref="PretrainedTokenizers.Load"/> read, or one a <see cref="TokenizerTrainers">trainer</see> learned).</summary>
    public required IPretrainedTokenizer Tokenizer { get; init; }

    /// <summary>The Jinja chat template, or null for <see cref="ChatMLWithParts"/>.</summary>
    public string? ChatTemplate { get; init; }

    /// <summary>The token that begins a text (written as bos), or null for none.</summary>
    public string? BosToken { get; init; }

    /// <summary>The token that ends a turn: generation stops at it.</summary>
    public string EosToken { get; init; } = "<|im_end|>";

    /// <summary>More tokens generation stops at (a document's end, for example "&lt;/s&gt;"), when the tokenizer has them.</summary>
    public IReadOnlyList<string> StopTokens { get; init; } = [];

    /// <summary>The padding token, or null for none.</summary>
    public string? PadToken { get; init; }

    /// <summary>Whether the template's text is preceded by the bos token when encoded (tokenizer_config.json's add_bos_token).</summary>
    public bool AddBosToken { get; init; }

    /// <summary>More files to copy into the model folder, by the name they get there (preprocessor_config.json, for example).</summary>
    public IReadOnlyDictionary<string, string> Extra { get; init; } = new Dictionary<string, string>();
}

/// <summary>Settings for <see cref="PretrainedModel.Create"/>.</summary>
public sealed record CreateOptions
{
    /// <summary>Where the created model is loaded (the default device when null); its weights are drawn on the CPU and written first.</summary>
    public Device? Device { get; init; }

    /// <summary>The seed the weights are drawn from.</summary>
    public int Seed { get; init; }

    /// <summary>
    /// The standard deviation of the normal draw of every weight matrix and embedding table (biases zero, norm gains one),
    /// as GPT-2 and nanoGPT initialize; null for Xavier-uniform matrices and ±0.02 embeddings.
    /// </summary>
    public float? InitStd { get; init; } = 0.02f;

    /// <summary>Write into a folder that already holds files (they are replaced); otherwise such a folder is refused.</summary>
    public bool Overwrite { get; init; }

    /// <summary>
    /// A vision part, for a vision-language model: the description the architecture's vision family reads
    /// (<see cref="IVisionFamily.Create"/>); null for a text model.
    /// </summary>
    public JsonObject? Vision { get; init; }

    /// <summary>
    /// A model folder whose decoder weights the new model starts from instead of drawn ones (a text model pretrained
    /// first, given to a vision-language model made around it); its spec must be the new one (the context length may
    /// differ). Null draws every weight.
    /// </summary>
    public string? Decoder { get; init; }

    /// <summary>
    /// Parts stored beside the decoder, by kind ("audio"): each description read by the part of that kind
    /// (<see cref="IModelPart.Create"/>; picking it by its kind is not on the new design yet), its tensors, config keys and
    /// files written with the model's; null or empty for none.
    /// </summary>
    public IReadOnlyDictionary<string, JsonObject>? Parts { get; init; }
}

/// <summary>
/// How far <see cref="PretrainedModel.Load"/> is (Craft's status bar, 2026-11-01: a 7B load took over 100 s with no sense
/// of how far it was): the checkpoint's tensors begun of all of them, and their values (each tensor's size, so a large
/// projection counts for what it holds).
/// </summary>
/// <param name="Tensors">The tensors read (or begun) so far.</param>
/// <param name="TotalTensors">The checkpoint's tensors.</param>
/// <param name="Values">The values of the tensors read so far.</param>
/// <param name="TotalValues">The values of every tensor of the checkpoint.</param>
public sealed record PretrainedLoadProgress(int Tensors, int TotalTensors, long Values, long TotalValues)
{
    /// <summary>The share read, by values, from 0 to 1 (by tensors when the sizes are not known).</summary>
    public double Fraction => TotalValues > 0 ? Math.Clamp((double)Values / TotalValues, 0, 1) : TotalTensors > 0 ? Math.Clamp((double)Tensors / TotalTensors, 0, 1) : 1;
}

/// <summary>Settings for <see cref="PretrainedModel.Load"/>.</summary>
public sealed record PretrainedOptions
{
    /// <summary>
    /// Told how far the load is: once when the checkpoint's header is read (nothing read yet), as each tensor is read, and
    /// once at the end (every tensor); on the loading thread. Null for none (no work is done for it then).
    /// </summary>
    public IProgress<PretrainedLoadProgress>? Progress { get; init; }

    /// <summary>Where the model is created (the default device when null).</summary>
    public Device? Device { get; init; }

    /// <summary>Longest sequence to support (sizes the rotary tables); the model's maximum when null.</summary>
    public int? MaxPositions { get; init; }

    /// <summary>The architecture's name to use instead of the one config.json names (matched against <see cref="Families"/>' own names).</summary>
    public string? Architecture { get; init; }

    /// <summary>
    /// The architecture that reads the checkpoint (its configuration keys and tensor names), handed over directly (a
    /// family's own, such as Idrak.Text.Plugins' <c>QwenArchitectures.Qwen3()</c>); null: the family of
    /// <see cref="Families"/> whose own names (<see cref="IPretrainedArchitecture.Names"/>) hold the name the checkpoint
    /// calls itself by (config.json's "architectures").
    /// </summary>
    public IPretrainedArchitecture? Family { get; init; }

    /// <summary>
    /// The families a checkpoint's own name is matched against where it comes in, when no <see cref="Family"/> is handed
    /// over (Idrak.Text.Plugins' <c>BuiltInFamilies.All</c>, or a package's own list): walked in order, each family's own
    /// names compared; null: none (give <see cref="Family"/>).
    /// </summary>
    public IFamilyProvider? Families { get; init; }

    /// <summary>
    /// The format the checkpoint is read by, handed over directly; null: the library's safetensors format
    /// (<see cref="SafeTensorsCheckpointFormat"/>) for a folder it opens. Each tensor is read as the type its format names, a
    /// projection's weight kept as stored (never widened).
    /// </summary>
    public ICheckpointFormat? Format { get; init; }

    /// <summary>
    /// The type every projection's weight and the token table are written as, by that type's encoder (a model's weights
    /// written to any type: <c>Idrak.Types.Q4_K.Type</c>, a type of one's own); null keeps each as its checkpoint stores it.
    /// A weight whose rows are not whole groups of the type's blocks keeps its stored type.
    /// </summary>
    public IElementType? Weights { get; init; }

    /// <summary>How the tokenizer is read (which file, its piece cache); null takes the variables, then the defaults.</summary>
    public TokenizerOptions? Tokenizer { get; init; }


    /// <summary>
    /// A folder holding a PEFT LoRA adapter (from <see cref="PretrainedModel.SaveAdapter"/> or peft) to merge into the
    /// weights as they are read: the model then runs as fast as the base model. Use <see cref="PretrainedModel.LoadAdapter"/> instead to keep the adapter separate
    /// (to train it further, or export it).
    /// </summary>
    public string? MergeAdapter { get; init; }

    /// <summary>
    /// Read no weight now (plan 23, C8b): the network is built on the deferred device (<see cref="DeferredTensors"/>), each
    /// weight checked against the checkpoint's header and given its tensor's place in the file, and a weight's bytes are
    /// read when its module moves to a device (<see cref="Module.To(Device)"/>), one tensor at a time. A pipeline stage
    /// (<c>PipelineStage.Place</c>) or a parameters shard (the pretrainer) so reads only its own tensors, and no process
    /// builds or holds the whole model. The weights are float32 (no merged adapter); a family whose
    /// weights the checkpoint names do not cover (<see cref="PretrainedModel.StoredTensors"/>) is refused naming the tensor.
    /// Move the network to a device before running it (<see cref="PretrainedModel.Place"/>).
    /// </summary>
    public bool Deferred { get; init; }

    /// <summary>
    /// The tokens the weight and expert placements keep key/value room for (<see cref="WeightPlacement"/>,
    /// <see cref="ExpertPlacement"/>): null takes <see cref="MaxPositions"/> when given, else
    /// <see cref="Models.AutoWeights.DefaultContext"/> (both within the model's).
    /// </summary>
    public int? AutoWeightsContext { get; init; }

    /// <summary>The share of the weights' bytes the placements keep free beside the working memory (<see cref="Models.AutoWeights.DefaultHeadroom"/>, 0.1, by default).</summary>
    public double AutoWeightsHeadroom { get; init; } = Models.AutoWeights.DefaultHeadroom;

    /// <summary>The bytes the placements may take instead of the memory the device reports free (a budget of one's own); null measures the device.</summary>
    public long? AutoWeightsBudget { get; init; }

    /// <summary>
    /// Where a decoder's layers go on a device other than the CPU (<see cref="WeightPlacements"/>, plan 37 W1), by name:
    /// "device", "split", "stream" or "auto". Picking a placement by its name is not on the new design yet; null (the
    /// default) places none, so every layer goes on the device. The free memory is <see cref="AutoWeightsBudget"/> when
    /// given, else what the device reports. <see cref="PretrainedModel.Placement"/> says what was chosen.
    /// </summary>
    public string? WeightPlacement { get; init; }

    /// <summary>
    /// The decoder layers on the device for a split (<see cref="WeightPlacement"/> split or auto): a count, or null for
    /// <see cref="WeightPlacements.DefaultDeviceLayers"/> (IDRAK_DEVICE_LAYERS; sized from the measured free memory when not set).
    /// </summary>
    public int? DeviceLayers { get; init; }

    /// <summary>
    /// Where a mixture-of-experts model's experts live (<see cref="ExpertPlacements"/>, plan 37 R1), by name: "resident",
    /// "offload" or "auto". Picking a placement by its name is not on the new design yet; null (the default) places none,
    /// so every expert stays where its layer is. With <see cref="LazyLayers"/>, each expert is read the first time the
    /// router picks it. <see cref="PretrainedModel.ExpertPlacement"/> says what was chosen.
    /// </summary>
    public string? ExpertPlacement { get; init; }

    /// <summary>
    /// The bytes of experts an offloading <see cref="ExpertPlacement"/> keeps on the device, or null for
    /// <see cref="ExpertPlacements.DefaultCacheBytes"/> (IDRAK_MOE_DEVICE_EXPERTS_MB; sized from the measured free memory
    /// when not set).
    /// </summary>
    public long? DeviceExpertBytes { get; init; }

    /// <summary>
    /// Whether an offloaded cold expert is computed by the CPU from system memory instead of copied to the device when
    /// the router picks it; null takes <see cref="ExpertPlacements.DefaultColdOnCpu"/> (IDRAK_MOE_COLD; copy).
    /// </summary>
    public bool? ColdExpertsOnCpu { get; init; }

    /// <summary>
    /// Whether a split or auto placement moves layers between the device and the CPU at a run's step boundaries as the
    /// device's measured free memory changes (more on the device when memory frees, fewer under pressure; plan 37 W2):
    /// null takes <see cref="WeightPlacements.DefaultGrow"/> (IDRAK_WEIGHT_GROW; on).
    /// </summary>
    public bool? FollowMemory { get; init; }

    /// <summary>
    /// Build each decoder layer on its first use (plan 37 W2): the load reads the token table, the final norm, the head and
    /// each layer's attention norm, and each layer's weights are read when it first computes, so a model of any size starts
    /// at once; the checkpoint stays open until the model is disposed. Not with <see cref="Deferred"/>, a merged adapter,
    /// or a family that builds its own network (those load whole). Null takes <see cref="WeightPlacements.DefaultLazy"/>
    /// (IDRAK_LAZY_LOAD; off). <see cref="PretrainedModel.LayersBuilt"/> counts the layers built so far.
    /// </summary>
    public bool? LazyLayers { get; init; }
}

/// <summary>A weight of a model as its checkpoint stores it (<see cref="PretrainedModel.StoredTensors"/>).</summary>
/// <param name="Name">Its name in the checkpoint.</param>
/// <param name="Shape">Its shape in the checkpoint (the tensor's transpose when <paramref name="Transposed"/>).</param>
/// <param name="Tensor">The model's tensor.</param>
/// <param name="Transposed">Whether the checkpoint stores the tensor's transpose (a projection [in, out] stored [out, in]).</param>
public sealed record StoredTensor(string Name, int[] Shape, Tensor Tensor, bool Transposed);

/// <summary>
/// Where the time of <see cref="PretrainedModel.Load"/> went, by stage: wall time on the loading thread (work spread over
/// the cores counts once). Measured on every load: a few clock reads per tensor.
/// </summary>
/// <param name="Open">Reading config.json and the checkpoint's headers, and opening a merged adapter.</param>
/// <param name="Read">Reading the weights' bytes from storage (zero when the store does not measure it, <c>ITensorStore.ReadTime</c>: then part of <paramref name="Convert"/>).</param>
/// <param name="Convert">Turning the bytes read into the layers' layout: transposing.</param>
/// <param name="Merge">Adding a merged adapter's updates to the weights (<see cref="PretrainedOptions.MergeAdapter"/>).</param>
/// <param name="Build">The rest of building the network: the weights' memory and uploads, the layers.</param>
/// <param name="Vision">Reading the vision part of a vision-language checkpoint (its encoder is built on first use, not here).</param>
/// <param name="Tokenizer">Reading the tokenizer and the chat template.</param>
/// <param name="Total">The whole load.</param>
public sealed record PretrainedLoadTimes(TimeSpan Open, TimeSpan Read, TimeSpan Convert, TimeSpan Merge, TimeSpan Build,
    TimeSpan Vision, TimeSpan Tokenizer, TimeSpan Total)
{
    /// <summary>
    /// The stages in order with their names ("open", "read", "convert", "merge", "build", "vision", "tokenizer",
    /// "total"), for a log line or a profile's stages (<see cref="ProfileRun.Stage(string, TimeSpan)"/>).
    /// </summary>
    public IReadOnlyList<(string Stage, TimeSpan Elapsed)> Stages =>
        [("open", Open), ("read", Read), ("convert", Convert), ("merge", Merge), ("build", Build), ("vision", Vision), ("tokenizer", Tokenizer), ("total", Total)];

    /// <summary>"open 0.01 s, read 1.20 s, …, total 4.90 s" (invariant culture).</summary>
    public override string ToString() =>
        string.Join(", ", Stages.Select(s => string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{s.Stage} {s.Elapsed.TotalSeconds:F2} s")));
}

/// <summary>
/// A pretrained decoder-only language model read from a folder in the Hugging Face layout (config.json, safetensors
/// weights, tokenizer.json, tokenizer_config.json): the <see cref="Network"/> built from its <see cref="Spec"/>, its
/// <see cref="Tokenizer"/> and <see cref="ChatTemplate"/>. The network is an ordinary Idrak model: generate and chat with
/// it (Idrak.Text's <c>CreateGenerator</c> and <c>CreateChat</c>), fine-tune it with LoRA, save it as a package.
/// </summary>
public sealed class PretrainedModel : IDisposable
{
    private PretrainedModel(string folder, JsonObject config, DecoderSpec spec, Sequential network, ITokenizer? tokenizer, IChatTemplate? template,
        IReadOnlyList<string> notes, int maxPositions, IPretrainedArchitecture architecture, Device device, IPretrainedVision? vision)
    {
        Architecture = architecture;
        NetworkParts = architecture as INetworkParts ?? Abstraction.Formats.NetworkParts.Decoder;   // asked once, at load
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
    /// The model's own chat template (from tokenizer_config.json), or null when the model has none or it is not read
    /// (reading a folder's own template is not on the new design yet).
    /// </summary>
    public IChatTemplate? ChatTemplate { get; }

    /// <summary>Anything approximated while reading the model (see <see cref="PretrainedFamilies.CommonSpec"/>).</summary>
    public IReadOnlyList<string> Notes { get; }

    /// <summary>The longest sequence the loaded model supports.</summary>
    public int MaxPositions { get; }

    /// <summary>How the model's checkpoint names map to Idrak's (used to save adapters and weights back).</summary>
    public IPretrainedArchitecture Architecture { get; }

    /// <summary>
    /// The parts of <see cref="Network"/> training speaks of (embeddings, decoder, head) and the projections adapters go on
    /// by default: the architecture's (<see cref="INetworkParts"/>), else the library's decoders' (<see cref="Abstraction.Formats.NetworkParts.Decoder"/>).
    /// </summary>
    public INetworkParts NetworkParts { get; }

    /// <summary>Where the model lives.</summary>
    public Device Device { get; }

    /// <summary>
    /// The vision part of a vision-language model as its family read it (<see cref="IVisionFamily"/>: the
    /// encoder it builds, its prompt format and attention rule; read and checked but not built), or null for a text-only
    /// model. <see cref="Network"/> is the text decoder alone.
    /// </summary>
    public IPretrainedVision? Vision { get; }

    /// <summary>
    /// The family of <paramref name="families"/> that states <paramref name="name"/> (the name a checkpoint calls itself by)
    /// among its own names: the list walked in order, each family's names compared, where the checkpoint comes in.
    /// </summary>
    private static IPretrainedArchitecture FamilyNamed(IFamilyProvider? families, string name)
    {
        foreach (var family in families?.Families ?? [])
        {
            foreach (string own in family.Names)
            {
                if (string.Equals(own, name, StringComparison.Ordinal))
                {
                    return family;
                }
            }
        }

        throw new NotSupportedException(families is null
            ? $"The checkpoint is a '{name}'; no family was handed over to read it (give PretrainedOptions.Family, or PretrainedOptions.Families to match its name against)."
            : $"The checkpoint is a '{name}'; none of the families handed over states that name ({string.Join(", ", families.Families.SelectMany(f => f.Names))}).");
    }

    /// <summary>
    /// The library's checkpoint format that opens <paramref name="path"/>, each asked in turn (its own test of the path):
    /// a .gguf file or the folder prepared from one (read through <paramref name="families"/>' GGUF mappings), else a model
    /// folder of safetensors.
    /// </summary>
    private static ICheckpointFormat FormatOf(string path, IFamilyProvider? families)
    {
        foreach (var format in (ReadOnlySpan<ICheckpointFormat>)[new GgufCheckpointFormat(families), SafeTensorsCheckpointFormat.Format])
        {
            if (format.CanOpen(path))
            {
                return format;
            }
        }

        throw new FileNotFoundException($"{path} is neither a .gguf file nor a model folder.", path);
    }

    /// <summary>
    /// Reads the model in <paramref name="folder"/> (a model folder, a .gguf file). Every weight is read from disk one
    /// tensor at a time, so the model is never held twice. Its architecture is <see cref="PretrainedOptions.Family"/>, or
    /// the family of <see cref="PretrainedOptions.Families"/> that states the name config.json gives; picking the folder's
    /// checkpoint format and its vision family by name is not on the new design yet.
    /// </summary>
    public static PretrainedModel Load(string folder, PretrainedOptions? options = null)
    {
        long started = Stopwatch.GetTimestamp();
        options ??= new PretrainedOptions();
        if (options.Deferred && options.MergeAdapter is not null)
        {
            throw new ArgumentException("A deferred load (PretrainedOptions.Deferred) reads float32 weights where they are placed: load without a merged adapter.", nameof(options));
        }

        var format = options.Format ?? FormatOf(folder, options.Families);
        folder = format.Prepare(folder);                            // a .gguf file: config, tokenizer and template from its metadata

        var config = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "config.json")))!.AsObject();
        string name = options.Architecture ?? (string?)config["architectures"]?[0]
            ?? throw new InvalidDataException("config.json names no architecture; pass PretrainedOptions.Architecture.");
        var architecture = options.Family ?? FamilyNamed(options.Families, name);
        // A vision-language checkpoint (one with a vision_config) is read by its own vision family, by the same name.
        var visionFamily = config["vision_config"] is JsonObject ? NotOnTheNewDesign.Yet<IVisionFamily>($"Picking the vision family '{name}' by its name") : null;

        using var reader = format.Open(folder);
        if (architecture is ICheckpointFittedArchitecture fitted)
        {
            architecture = fitted.ForCheckpoint(reader.Names.ToHashSet(StringComparer.Ordinal));     // the naming this checkpoint uses
        }

        var notes = new List<string>();
        var spec = architecture.Spec(config, notes);
        int maxPositions = Math.Min(options.MaxPositions ?? spec.MaxPositions, spec.MaxPositions);

        // The weight placement (plan 37, W1): on a device other than the CPU, where the layers go when the model may not fit.
        var placementTarget = options.Device ?? Idrak.Abstraction.Device.Default;
        var placement = !options.Deferred && placementTarget.Type != DeviceType.Cpu && options.WeightPlacement is { } placementName
            ? NotOnTheNewDesign.Yet<IWeightPlacement>($"Picking the weight placement '{placementName}' by its name") : null;
        bool mayLeaveDevice = placement is not null and not WeightPlacements.DevicePlacement;

        // The experts' placement (plan 37, R1): where a mixture-of-experts model's experts live, decided before the layers'
        // placement, which then counts only the experts kept on the device.
        ExpertPlan? expertPlan = null;
        int loadContext = Math.Min(options.AutoWeightsContext ?? options.MaxPositions ?? AutoWeights.DefaultContext, spec.MaxPositions);
        ExpertPlacementRequest? expertRequest = null;
        if (spec.Experts > 0 && !options.Deferred && architecture is not INetworkBuildingArchitecture && options.ExpertPlacement is { } expertPlacementName)
        {
            expertRequest = AutoWeights.ExpertRequest(spec, loadContext, placementTarget, options.AutoWeightsBudget ?? placementTarget.Backend.AvailableMemory(),
                options.DeviceExpertBytes ?? ExpertPlacements.DefaultCacheBytes, options.ColdExpertsOnCpu ?? ExpertPlacements.DefaultColdOnCpu, options.AutoWeightsHeadroom);
            expertPlan = NotOnTheNewDesign.Yet<IExpertPlacement>($"Picking the expert placement '{expertPlacementName}' by its name").Place(expertRequest);
            if (expertPlan.Offload && placementTarget.Type == DeviceType.Cpu)
            {
                expertPlan = expertPlan with { Offload = false, DeviceExperts = spec.Experts, Reason = $"every expert on {placementTarget} (the CPU's memory is the system's)" };
            }

            if (expertPlan.Offload)
            {
                notes.Add($"experts {expertPlan}.");
            }
        }

        WeightPlan? plan = null;
        WeightPlacementRequest? placedRequest = null;
        if (placement is not null)
        {
            int context = loadContext;
            var request = AutoWeights.PlacementRequest(spec, context, placementTarget, options.AutoWeightsBudget ?? placementTarget.Backend.AvailableMemory(),
                options.DeviceLayers ?? WeightPlacements.DefaultDeviceLayers, options.AutoWeightsHeadroom);
            if (expertPlan is { Offload: true } && expertRequest is not null)
            {
                // A layer holds only the experts kept on the device: the others live in system memory.
                long apart = (long)expertRequest.ExpertLayers * (spec.Experts - expertPlan.DeviceExperts) * expertRequest.ExpertBytes / Math.Max(1, spec.Layers);
                request = request with { LayerBytes = Math.Max(0, request.LayerBytes - apart) };
            }

            plan = placement.Place(request);
            placedRequest = request;
            if (plan.Split && architecture is INetworkBuildingArchitecture)
            {
                notes.Add($"placement {plan.Placement}: {name} builds its own network, which is not split; every layer stays on {placementTarget}.");
                plan = plan with { DeviceLayers = plan.Layers, Reason = $"every layer on {placementTarget} ({name} builds its own network)" };
            }
            else if (plan.Split || plan.Stream)
            {
                notes.Add($"placement {plan}.");
            }

            if (plan.Stream)
            {
                ComputeResources.OffloadToHostMemory = true;                  // today's staging (IMemoryOffload) holds what does not fit
            }
        }

        using var adapter = options.MergeAdapter is { } adapterFolder ? new AdapterMerge(adapterFolder) : null;
        var weights = new CheckpointWeights(reader, architecture) { Adapter = adapter, Deferred = options.Deferred, Progress = LoadProgress.Of(reader, options.Progress), Writes = options.Weights };
        weights.Progress?.Begin();
        var buildDevice = options.Deferred ? NotOnTheNewDesign.Yet<Device>("The deferred device (picked by its name)") : options.Device;
        // W2: layers built on first use (the checkpoint kept open for them), and layers following the memory at step boundaries.
        bool buildsOwnNetwork = architecture is INetworkBuildingArchitecture;
        bool lazy = (options.LazyLayers ?? WeightPlacements.DefaultLazy) && !options.Deferred && options.MergeAdapter is null && !buildsOwnNetwork;
        if ((options.LazyLayers ?? WeightPlacements.DefaultLazy) && !lazy)
        {
            notes.Add("lazy layers: this load reads every layer now (a deferred load, a merged adapter, or a family that builds its own network).");
        }

        bool follow = placement is not null && placedRequest is not null && plan is { Stream: false } && mayLeaveDevice && !buildsOwnNetwork
            && placedRequest.DeviceLayers is null && (options.FollowMemory ?? WeightPlacements.DefaultGrow);
        var lazyReader = lazy ? format.Open(folder) : null;
        var buildWeights = lazyReader is not null ? new CheckpointWeights(lazyReader, architecture) { Writes = options.Weights } : weights;
        var buildOptions = new DecoderBuildOptions { Device = buildDevice, DeviceLayers = plan is { Split: true } ? plan.DeviceLayers : null,
            LazyLayers = lazy, Experts = expertPlan, LazyExperts = lazy, FollowMemory = follow ? (placement!, placedRequest!) : null, MaxPositions = maxPositions };
        long building = Stopwatch.GetTimestamp();
        var fileTime = reader.ReadTime;
        Sequential network;
        using ((buildDevice ?? Idrak.Abstraction.Device.Default).Backend.LongLivedAllocations())
        using (options.Deferred ? DeferredTensors.Build() : null)
        {
            // The weights live as long as the model: the device may place them together (Backend.LongLivedAllocations).
            network = architecture is INetworkBuildingArchitecture ownNetwork
                ? ownNetwork.Build(new PretrainedBuildContext(config, spec, weights, reader, buildOptions, notes)) switch
                {
                    null => throw new InvalidOperationException($"The architecture '{name}' built no network."),
                    Sequential built => built,
                    var layers => new Sequential(layers) { Name = "decoder" },
                }
                : spec.Build(buildWeights, buildOptions);
        }

        long buildEnded = Stopwatch.GetTimestamp();
        fileTime = reader.ReadTime - fileTime;

        var vision = visionFamily?.Read(new VisionCheckpoint(name, config, reader, folder, () => format.Open(folder), notes));
        long visionRead = Stopwatch.GetTimestamp();
        if (vision is not null)
        {
            weights.Used.UnionWith(vision.StoredTensors);
        }

        // The parts stored beside the decoder (an audio encoder): finding them is not on the new design yet, so none is
        // read.
        var partTensors = new Dictionary<string, IReadOnlySet<string>>(StringComparer.Ordinal);

        var unused = lazy ? new List<string>() : reader.Names.Where(k => !weights.Used.Contains(k) && !k.EndsWith("rotary_emb.inv_freq", StringComparison.Ordinal)).ToList();
        if (unused.Count > 0)
        {
            notes.Add($"{unused.Count} checkpoint tensors were not used (for example {string.Join(", ", unused.Take(3))}).");
        }

        // A vision part's and the other parts' trained modules (modules_to_save) are kept beside the merged weights, for
        // CreateVisionEncoder and OpenPart.
        var partStored = partTensors.Values.SelectMany(t => t).ToHashSet(StringComparer.Ordinal);
        adapter?.CheckModulesToSave((vision?.StoredTensors ?? Enumerable.Empty<string>()).Concat(partStored).ToHashSet(StringComparer.Ordinal));
        var visionTensors = adapter?.Tensors(vision?.StoredTensors);
        var partTrained = adapter?.Tensors(partStored);
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

        long tokenizing = Stopwatch.GetTimestamp();
        var tokenizer = PretrainedTokenizers.TryLoad(folder, options.Tokenizer);
        tokenizer?.PadVocabulary(spec.Vocabulary);
        IChatTemplate? template = null;                                 // reading the folder's own template: not on the new design yet
        long tokenized = Stopwatch.GetTimestamp();
        var model = new PretrainedModel(folder, config, spec, network, tokenizer, template, notes, maxPositions, architecture,
            options.Device ?? Idrak.Abstraction.Device.Default, vision);
        foreach (var (stored, value) in visionTensors ?? [])
        {
            model._vision[stored] = value;
        }

        foreach (var (kind, owned) in partTensors)
        {
            model._partTensors[kind] = owned;
        }

        foreach (var (stored, value) in partTrained ?? [])
        {
            model._parts[stored] = value;
        }

        model._loadBytes = reader.BytesRead;
        model.Placement = plan;
        model.ExpertPlacement = expertPlan;
        model._lazyReader = lazyReader;
        try
        {
            model.RecommendedSampling = Abstraction.Generation.SamplingSettings.Read(folder);
        }
        catch (InvalidDataException e)
        {
            notes.Add($"the recommended sampling was not read: {e.Message}");
        }
        if (options.Deferred)
        {
            try
            {
                model.BindDeferred(format.Open(folder));            // a store of its own, kept while the model lives
            }
            catch
            {
                model.Dispose();
                throw;
            }
        }

        // The store's file time (when it measures it) apart from the rest of its reads; what the build did besides reading,
        // and merging is the weights' uploads and the layers.
        var read = fileTime ?? TimeSpan.Zero;
        var others = Stopwatch.GetElapsedTime(building, buildEnded) - weights.ReadTime - weights.MergeTime;
        model.LoadTimes = new PretrainedLoadTimes(Stopwatch.GetElapsedTime(started, building), read, Max(weights.ReadTime - read), weights.MergeTime,
            Max(others), Stopwatch.GetElapsedTime(buildEnded, visionRead), Stopwatch.GetElapsedTime(tokenizing, tokenized),
            Stopwatch.GetElapsedTime(started));
        weights.Progress?.End();
        return model;

        static TimeSpan Max(TimeSpan time) => time > TimeSpan.Zero ? time : TimeSpan.Zero;
    }

    /// <summary>
    /// Makes a model that did not exist: the network of <paramref name="spec"/> with weights drawn from
    /// <see cref="CreateOptions.Seed"/>, written to <paramref name="folder"/> as a checkpoint of
    /// <paramref name="architecture"/> (config.json in the family's keys, through its
    /// <see cref="IConfigWritingArchitecture.Config"/>, with the spec kept as "idrak_spec"; model.safetensors; the tokenizer,
    /// template and special-token files of <paramref name="files"/>), then loaded from there, so it is a model like any
    /// other: it runs, chats, serves, tunes and trains (plan 19).
    /// </summary>
    /// <exception cref="NotSupportedException">Picking the architecture by its name is not on the new design yet; or it cannot be written from a spec, or cannot hold this spec.</exception>
    /// <exception cref="ArgumentException">The tokenizer is larger than the vocabulary, or a named token is not in it.</exception>
    /// <exception cref="IOException">The folder holds files and <see cref="CreateOptions.Overwrite"/> is off.</exception>
    public static PretrainedModel Create(string folder, DecoderSpec spec, string architecture, ModelFiles files, CreateOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(folder);
        ArgumentNullException.ThrowIfNull(spec);
        ArgumentNullException.ThrowIfNull(files);
        options ??= new CreateOptions();
        var family = NotOnTheNewDesign.Yet<IPretrainedArchitecture>($"Picking the architecture '{architecture}' by its name");
        var write = family is IConfigWritingArchitecture writing ? writing : throw new NotSupportedException(
            $"Architecture '{architecture}' cannot be written from a spec (it reads checkpoints only); choose one that can.");
        var tokenizer = files.Tokenizer;
        if (tokenizer.VocabularySize > spec.Vocabulary)
        {
            throw new ArgumentException($"The tokenizer has {tokenizer.VocabularySize} tokens, more than the spec's vocabulary of {spec.Vocabulary}.", nameof(files));
        }

        int Id(string token, string what) => tokenizer.IdOf(token)
            ?? throw new ArgumentException($"The tokenizer has no {what} token '{token}'.", nameof(files));
        int eos = Id(files.EosToken, "end-of-turn");
        int? bos = files.BosToken is { } b ? Id(b, "begin") : null;
        int? pad = files.PadToken is { } p ? Id(p, "padding") : null;
        var stops = new List<int> { eos };
        stops.AddRange(files.StopTokens.Select(t => Id(t, "stop")).Where(t => !stops.Contains(t)));
        string template = files.ChatTemplate ?? ModelFiles.ChatMLWithParts;
        if (files.ChatTemplate is null)
        {
            _ = Id("<|im_start|>", "ChatML");
            _ = Id("<|im_end|>", "ChatML");
        }

        var config = write.Config(spec);
        var parts = new List<CreatedModelPart>();
        foreach (var (kind, partDescription) in options.Parts ?? new Dictionary<string, JsonObject>())
        {
            var made = NotOnTheNewDesign.Yet<IModelPart>($"Picking the model part '{kind}' by its name")
                .Create(new ModelPartRequest(architecture, partDescription, spec.Dim, tokenizer, options.Seed));
            foreach (var (key, value) in made.Config)
            {
                config[key] = value?.DeepClone();
            }

            parts.Add(made);
        }

        CreatedVision? vision = null;
        if (options.Vision is { } description)
        {
            var visionFamily = NotOnTheNewDesign.Yet<IVisionFamily>($"Picking the vision family '{architecture}' by its name");
            vision = visionFamily.Create(description, spec.Dim, tokenizer, options.Seed);
            foreach (var (key, value) in vision.Config)
            {
                config[key] = value?.DeepClone();
            }
        }

        config["architectures"] = new JsonArray((JsonNode)architecture);
        config["bos_token_id"] = bos;
        config["eos_token_id"] = stops.Count == 1 ? eos : new JsonArray([.. stops.Select(i => (JsonNode)i)]);
        config["pad_token_id"] = pad;
        config["idrak_spec"] = spec.ToJson();

        if (Directory.Exists(folder) && Directory.EnumerateFileSystemEntries(folder).Any() && !options.Overwrite)
        {
            throw new IOException($"{folder} already holds files; choose an empty folder or overwrite.");
        }

        Directory.CreateDirectory(folder);
        var cpu = Idrak.Abstraction.Device.Cpu;
        Sequential network;
        if (options.Decoder is { } decoderFolder)
        {
            // The decoder of a model that exists (its spec the new one's but for the context length), read as Load reads it.
            var format = NotOnTheNewDesign.Yet<ICheckpointFormat>($"Picking the checkpoint format of {decoderFolder}");
            decoderFolder = format.Prepare(decoderFolder);
            var decoderConfig = JsonNode.Parse(File.ReadAllText(Path.Combine(decoderFolder, "config.json")))!.AsObject();
            string decoderArchitecture = (string?)decoderConfig["architectures"]?[0] ?? throw new InvalidDataException($"{decoderFolder}/config.json names no architecture.");
            var decoderFamily = NotOnTheNewDesign.Yet<IPretrainedArchitecture>($"Picking the architecture '{decoderArchitecture}' by its name");
            var decoderSpec = decoderFamily.Spec(decoderConfig, []);
            var mine = (spec with { MaxPositions = 0, Dropout = 0 }).ToJson();
            var theirs = (decoderSpec with { MaxPositions = 0, Dropout = 0 }).ToJson();
            if (!JsonNode.DeepEquals(mine, theirs))
            {
                throw new ArgumentException($"The decoder in {decoderFolder} is not this spec's: "
                    + string.Join(", ", mine.Select(p => p.Key).Where(k => !JsonNode.DeepEquals(mine[k], theirs[k]))) + " differ.", nameof(options));
            }

            using var reader = format.Open(decoderFolder);
            var fitted = decoderFamily is ICheckpointFittedArchitecture fit ? fit.ForCheckpoint(reader.Names.ToHashSet(StringComparer.Ordinal)) : decoderFamily;
            network = spec.Build(new CheckpointWeights(reader, fitted), new DecoderBuildOptions { Device = cpu });
        }
        else
        {
            network = spec.Build(options: new DecoderBuildOptions { Device = cpu, Seed = options.Seed, InitStd = options.InitStd });
        }

        using (var drawn = new PretrainedModel(folder, config, spec, network, tokenizer, null, [], spec.MaxPositions, family, cpu, null))
        {
            var extra = (vision?.Tensors ?? []).Concat(parts.SelectMany(p => p.Tensors));
            drawn.WriteTensors(folder, extra.Select(t => (t.Name, t.Shape, (Func<float[]>)(() => t.Values))));
        }

        foreach (var (name, text) in (vision?.Files ?? new Dictionary<string, string>()).Concat(parts.SelectMany(p => p.Files)))
        {
            File.WriteAllText(Path.Combine(folder, name), text);
        }

        var indented = new System.Text.Json.JsonSerializerOptions { WriteIndented = true, Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping };
        void Json(string name, JsonObject value) => File.WriteAllText(Path.Combine(folder, name), value.ToJsonString(indented) + "\n");
        Json("config.json", config);
        tokenizer.Save(Path.Combine(folder, "tokenizer.json"));
        Json("tokenizer_config.json", new JsonObject
        {
            ["add_bos_token"] = files.AddBosToken,
            ["bos_token"] = files.BosToken,
            ["eos_token"] = files.EosToken,
            ["pad_token"] = files.PadToken,
            ["chat_template"] = template,
            ["model_max_length"] = spec.MaxPositions,
            ["tokenizer_class"] = "PreTrainedTokenizerFast",
        });
        var map = new JsonObject { ["eos_token"] = files.EosToken };
        if (files.BosToken is not null)
        {
            map["bos_token"] = files.BosToken;
        }

        if (files.PadToken is not null)
        {
            map["pad_token"] = files.PadToken;
        }

        Json("special_tokens_map.json", map);
        Json("generation_config.json", new JsonObject
        {
            ["bos_token_id"] = bos,
            ["eos_token_id"] = new JsonArray([.. stops.Select(i => (JsonNode)i)]),
            ["pad_token_id"] = pad,
        });
        foreach (var (name, source) in files.Extra)
        {
            File.Copy(source, Path.Combine(folder, name), overwrite: true);
        }

        return Load(folder, new PretrainedOptions { Device = options.Device });
    }

    /// <summary>
    /// The sampling the model's makers recommend (its generation_config.json: temperature, top_p, top_k, min_p,
    /// repetition_penalty, do_sample; a GGUF file's general.sampling keys, written there when the file is prepared), or
    /// null when the folder has none. The generators made from the model carry it (<c>TextGenerator.RecommendedSampling</c>);
    /// an app lays it under its explicit values with <see cref="Abstraction.Generation.SamplingSettings.Resolve"/>.
    /// </summary>
    public Abstraction.Generation.SamplingSettings? RecommendedSampling { get; private set; }

    /// <summary>
    /// Where the load placed the decoder's layers (<see cref="PretrainedOptions.WeightPlacement"/>): on the device, split with
    /// the CPU (which layers compute where) or streamed from system memory, with what was measured; null on the CPU.
    /// </summary>
    public WeightPlan? Placement { get; private set; }

    /// <summary>
    /// Where the load placed a mixture-of-experts model's experts (<see cref="PretrainedOptions.ExpertPlacement"/>, plan 37
    /// R1): every one on the device, or the most used in a device cache and the rest in system memory, with what was
    /// measured; null for a model without experts. Each layer's cache is its <c>MixtureOfExperts.Cache</c>, and what they
    /// did is <see cref="Layers.ExpertTraffic"/>.
    /// </summary>
    public ExpertPlan? ExpertPlacement { get; private set; }

    /// <summary>Where the time of <see cref="Load"/> went, by stage (read, convert, upload, adapter merge, tokenizer, …).</summary>
    public PretrainedLoadTimes LoadTimes { get; private set; } = new(default, default, default, default, default, default, default, default);

    // The trained vision tensors (a vision part's fine-tuned modules), by their checkpoint names, in the checkpoint's layout.
    private readonly Dictionary<string, (int[] Shape, float[] Values)> _vision = new(StringComparer.Ordinal);

    // The parts stored beside the decoder (ModelParts), by kind: the checkpoint tensors each owns.
    private readonly Dictionary<string, IReadOnlySet<string>> _partTensors = new(StringComparer.Ordinal);

    // The parts' trained tensors (a training run's kept encoder and projector), by their checkpoint names and layout.
    private readonly Dictionary<string, (int[] Shape, float[] Values)> _parts = new(StringComparer.Ordinal);

    /// <summary>
    /// The kinds of the parts stored beside the decoder that an <see cref="IModelPart"/> read ("audio"): their
    /// tensors count as used, every save carries them, and <see cref="OpenPart"/> opens each as a soft-part source.
    /// </summary>
    public IReadOnlyCollection<string> Parts => _partTensors.Keys;

    /// <summary>The names (as the checkpoint stores them) of the parts' trained tensors the model carries (<see cref="KeepTrainedParts"/>).</summary>
    public IReadOnlyCollection<string> TrainedPartTensors => _parts.Keys;

    /// <summary>
    /// Keeps <paramref name="tensors"/> (a part's trained tensors, from <see cref="ITrainableSoftParts.Export"/>) as the
    /// model's, replacing those of the same names: <see cref="OpenPart"/> puts them into every source it opens and every
    /// save writes them in place of the checkpoint's. Returns how many were kept.
    /// </summary>
    /// <exception cref="ArgumentException">A tensor is not one of the model's parts' (<see cref="Parts"/>).</exception>
    public int KeepTrainedParts(IEnumerable<(string Name, int[] Shape, float[] Values)> tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        int kept = 0;
        foreach (var (name, shape, values) in tensors)
        {
            if (!_partTensors.Values.Any(owned => owned.Contains(name)))
            {
                throw new ArgumentException($"'{name}' is not a tensor of the model's parts ({(_partTensors.Count == 0 ? "it has none" : string.Join(", ", _partTensors.Keys))}).", nameof(tensors));
            }

            _parts[name] = ([.. shape], values);
            kept++;
        }

        return kept;
    }

    /// <summary>
    /// The part <paramref name="kind"/> (one of <see cref="Parts"/>) as a soft-part source on <paramref name="device"/>
    /// (the model's when null), read from the model's checkpoint with its trained tensors put in
    /// (<see cref="TrainedPartTensors"/>). Dispose it when done.
    /// </summary>
    /// <exception cref="InvalidOperationException">The model has no such part.</exception>
    /// <exception cref="NotSupportedException">Picking the part's kind by its name is not on the new design yet.</exception>
    public ITrainableSoftParts OpenPart(string kind, Device? device = null, IFeatureCache? cache = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(kind);
        if (!_partTensors.ContainsKey(kind))
        {
            throw new InvalidOperationException($"The model has no '{kind}' part ({(_partTensors.Count == 0 ? "none" : string.Join(", ", _partTensors.Keys))}).");
        }

        var part = NotOnTheNewDesign.Yet<IModelPart>($"Picking the model part '{kind}' by its name");
        var format = NotOnTheNewDesign.Yet<ICheckpointFormat>($"Picking the checkpoint format of {Folder}");
        using var tensors = format.Open(Folder);
        string architecture = (string?)Config["architectures"]?[0] ?? throw new InvalidDataException($"{Folder}/config.json names no architecture.");
        return part.Open(new ModelPartCheckpoint(architecture, Config, tensors, Folder, () => format.Open(Folder), []),
            new ModelPartOptions { Device = device ?? Device, Trained = _parts, Cache = cache });
    }

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
    /// through the family's <see cref="IVisionTuningPart.Import"/>). The parts those tensors belong to
    /// (<see cref="IVisionTuningPart.PartsOf"/>) are named in <see cref="VisionEncoderOptions.TrainedParts"/>. Dispose it
    /// when done.
    /// </summary>
    /// <exception cref="InvalidOperationException">The model has no vision part.</exception>

    /// <exception cref="InvalidDataException">A trained vision tensor fits no part the family offers.</exception>
    public IVisionEncoder CreateVisionEncoder(VisionEncoderOptions? options = null)
    {
        var vision = Vision ?? throw new InvalidOperationException("The model has no vision part (a text-only checkpoint): it reads no images.");
        options ??= new VisionEncoderOptions();
        if (options.Device is null)
        {
            options = options with { Device = Device };
        }

        // The parts the trained tensors belong to.
        if (_vision.Count > 0 && vision is IVisionTuningPart parts)
        {
            options = options with { TrainedParts = [.. options.TrainedParts.Union(parts.PartsOf(_vision.Keys), StringComparer.OrdinalIgnoreCase)] };
        }

        IVisionEncoder encoder;
        using (options.Device!.Backend.LongLivedAllocations())                  // the encoder's weights live as long as it
        {
            encoder = vision.CreateEncoder(options);
        }

        if (_vision.Count == 0)
        {
            return encoder;
        }

        try
        {
            var tuning = vision as IVisionTuningPart
                ?? throw new InvalidDataException($"The model carries {_vision.Count} trained vision tensors, but the vision family {vision.Family} offers no trainable parts.");
            ImportVision(tuning, encoder);
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
    /// down, head; null: the network's <see cref="INetworkParts.AdapterTargets"/>) and freezes everything else. Returns how
    /// many were added.
    /// </summary>
    public int AddAdapters(int rank, float alpha, IEnumerable<string>? targets, int seed = 0, bool dora = false)
    {
        var names = (targets ?? NetworkParts.AdapterTargets).ToHashSet(StringComparer.Ordinal);
        Func<IAdaptableLinear, bool> chosen = l => l.Name is { } name && names.Contains(name) && l is not ITiedWeights { WeightsOf: not null };
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

    // The store a deferred model's tensors are read from (plan 23, C8b), and the bytes the load itself read.
    private ITensorStore? _source;
    private long? _loadBytes;

    /// <summary>
    /// The bytes of weight values read from the checkpoint so far: by the load, and for a deferred model
    /// (<see cref="PretrainedOptions.Deferred"/>) by each tensor read as it was placed; null when the store does not count them.
    /// </summary>
    public long? BytesRead => _loadBytes is { } loaded ? loaded + (_source?.BytesRead ?? 0) : _source?.BytesRead;

    /// <summary>Whether some of the network's tensors are still on the deferred device (not read yet, <see cref="PretrainedOptions.Deferred"/>).</summary>
    public bool IsDeferred => Network.Parameters().Concat(Network.Buffers()).Any(DeferredTensors.IsDeferred);

    /// <summary>
    /// Moves a deferred network to <see cref="Device"/>, reading every tensor still unread, one at a time (nothing when the
    /// model was loaded whole). A rank that holds only part of the model places that part itself instead.
    /// </summary>
    public void Place()
    {
        if (IsDeferred)
        {
            Network.To(Device);
        }
    }

    /// <summary>
    /// The model's weights as its checkpoint stores them, in the order a save writes them: each one's checkpoint name, its
    /// stored shape and the model's tensor (a tied head's table once, under the embedding's name). The names are the
    /// architecture's (<see cref="IPretrainedArchitecture.TensorName"/>), so this holds for any family.
    /// </summary>
    public IReadOnlyList<StoredTensor> StoredTensors()
    {
        var tensors = new List<StoredTensor>();
        void Add(string name, Tensor tensor)
        {
            string stored = CheckpointName(name);
            bool transposed = Architecture.Transposed(name) && tensor.Rank == 2;
            tensors.Add(new StoredTensor(stored, transposed ? [tensor.Shape[1], tensor.Shape[0]] : [.. tensor.Shape], tensor, transposed));
        }

        foreach (var (path, module) in NamedModules())
        {
            if (module is INamedWeights named)
            {
                foreach (var (name, tensor) in named.NamedWeights())
                {
                    Add($"{path}.{name}", tensor);
                }
            }
        }

        return tensors;
    }

    // A deferred build's tensors given their places in the checkpoint: every parameter must have one.
    private void BindDeferred(ITensorStore store)
    {
        _source = store;
        foreach (var stored in StoredTensors())
        {
            if (!DeferredTensors.IsDeferred(stored.Tensor) || DeferredTensors.SourceOf(stored.Tensor) is not null)
            {
                continue;
            }

            string name = stored.Name;
            bool transposed = stored.Transposed;
            int count = stored.Tensor.Size;
            if (!store.Contains(name) || store.ShapeOf(name).Aggregate(1L, (a, b) => a * b) != count)
            {
                throw new InvalidDataException($"A deferred load reads '{name}' for a tensor of {count} values, which the checkpoint does not hold as such; load the model whole.");
            }

            DeferredTensors.SetSource(stored.Tensor, name, values => (transposed ? store.ReadTransposed(name) : store.Read(name)).AsSpan(0, values.Length).CopyTo(values));
        }

        if (Network.Parameters().FirstOrDefault(p => DeferredTensors.IsDeferred(p) && DeferredTensors.SourceOf(p) is null) is { } unread)
        {
            var owner = NamedModules().FirstOrDefault(m => m.Module.Parameters().Any(q => ReferenceEquals(q, unread)));
            throw new InvalidDataException($"A deferred load cannot place the parameter {Tensor.FormatShape(unread.Shape)} of '{owner.Path}' ({owner.Module?.GetType().Name}): "
                + "the architecture names no checkpoint tensor for it (PretrainedModel.StoredTensors). Load the model whole (PretrainedOptions.Deferred off, IDRAK_CLUSTER_OWN_TENSORS=0).");
        }
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
        string? peftType = null;
        foreach (var (path, module) in NamedModules())
        {
            if (module is not IAdaptableLinear { Adapter: { } any } linear)
            {
                continue;
            }

            if (any is not IPeftAdapter peft)
            {
                throw new InvalidOperationException($"{linear}: a {any.GetType().Name} has no PEFT layout; only adapters in peft's layout (IPeftAdapter: LoRA, DoRA) are saved.");
            }

            var (a, b, magnitude, adapterRank, scale) = (peft.A, peft.B, peft.Magnitude, peft.Rank, peft.Scale);
            if (peftType is not null && !string.Equals(peftType, peft.PeftType, StringComparison.Ordinal))
            {
                throw new InvalidOperationException($"The model has {peftType} and {peft.PeftType} adapters; PEFT stores one type per folder.");
            }

            peftType = peft.PeftType;
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

        // The trained vision and part tensors as PEFT writes modules_to_save: base_model.model. + the checkpoint's name, as stored.
        foreach (var (name, (shape, values)) in _vision.Concat(_parts).OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            tensors.Add(($"base_model.model.{name}", shape, values));
        }

        SafeTensorsWriter.Write(Path.Combine(folder, "adapter_model.safetensors"), tensors,
            new Dictionary<string, string> { ["format"] = "pt" });
        var config = new JsonObject
        {
            ["peft_type"] = peftType, ["task_type"] = "CAUSAL_LM", ["r"] = rank, ["lora_alpha"] = alpha, ["lora_dropout"] = 0.0,
            ["bias"] = "none", ["fan_in_fan_out"] = false, ["inference_mode"] = true, ["use_dora"] = dora == true,
            ["target_modules"] = new JsonArray([.. modules.Select(m => (JsonNode)m)]),
            ["base_model_name_or_path"] = (string?)Config["_name_or_path"] ?? Path.GetFileName(Path.TrimEndingDirectorySeparator(Folder)),
        };
        if (_vision.Count > 0 || _parts.Count > 0)
        {
            var saved = (_vision.Count > 0 ? SavedModules(_vision.Keys) : []).Concat(_parts.Count > 0 ? SavedModules(_parts.Keys) : []);
            config["modules_to_save"] = new JsonArray([.. saved.Distinct(StringComparer.Ordinal).Select(m => (JsonNode)m)]);
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
        var partStored = _partTensors.Values.SelectMany(t => t).ToHashSet(StringComparer.Ordinal);
        config.CheckModulesToSave((Vision?.StoredTensors ?? Enumerable.Empty<string>()).Concat(partStored).ToHashSet(StringComparer.Ordinal));
        using var reader = SafeTensorsReader.Open(Path.Combine(folder, "adapter_model.safetensors"));
        var found = new List<(Module Module, IAdaptableLinear Layer, string A, string B, string? Magnitude)>();
        var used = new HashSet<string>(StringComparer.Ordinal);
        var vision = PeftAdapterConfig.VisionTensors(reader, Vision?.StoredTensors, used);
        foreach (var (name, value) in PeftAdapterConfig.VisionTensors(reader, partStored, used))
        {
            _parts[name] = value;                                             // a part's trained modules (an audio projector)
        }
        foreach (var (path, module) in NamedModules())
        {
            if (module is IAdaptableLinear linear && Architecture.TensorName($"{path}.weight") is { } weight
                && config.Names(reader, weight[..^".weight".Length]) is var (a, b, magnitude))
            {
                found.Add((module, linear, a, b, magnitude));
                used.UnionWith(new[] { a, b, magnitude }.OfType<string>());
            }
        }

        PeftAdapterConfig.CheckAllUsed(reader, used, folder);                   // before any layer changes
        // A model that had no adapter gets its base frozen, as AddAdapters leaves it: the loaded weights come trainable, so a
        // tune going on from this adapter (a FineTuningSession, which adds none when the model has some) trained every
        // weight of the model, not the adapter (Craft's learning plug-in, its second round: the whole model's optimizer state).
        if (found.Count > 0 && !Network.Descendants().OfType<IAdaptableLinear>().Any(l => l.Adapter is not null))
        {
            Network.Freeze();
        }

        foreach (var (module, linear, a, b, magnitude) in found)
        {
            int rank = reader.Tensors[a].Shape[0];
            bool fits = linear.Adapter switch
            {
                null => true,
                IPeftAdapter p => (p.Magnitude is null) == (magnitude is null) && p.Rank == rank,
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
                    module.AddLora(rank, config.Alpha, l => ReferenceEquals(l, linear), freezeBase: false);
                }
                else
                {
                    module.AddDora(rank, config.Alpha, l => ReferenceEquals(l, linear), freezeBase: false);
                }
            }

            var peft = linear.Adapter as IPeftAdapter ?? throw new InvalidOperationException($"{linear} has no adapter in peft's layout (IPeftAdapter: LoRA, DoRA).");
            var (down, up) = (peft.A, peft.B);
            if (peft.Magnitude is { } stored)
            {
                stored.Load(reader.Read(magnitude!));
            }

            linear.Adapter = peft.WithScale(config.Scale);
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
    /// Writes the model as a Hugging Face checkpoint (model.safetensors of float32 tensors, with config.json, the tokenizer
    /// and chat template files copied from the original folder): LoRA adapters are merged into the weights first.
    /// </summary>
    public void SaveHuggingFace(string folder) => SaveHuggingFace(folder, null);

    /// <summary>
    /// <see cref="SaveHuggingFace(string)"/> with the vision part's tensors: each of
    /// <see cref="PretrainedVision.StoredTensors"/> from <paramref name="vision"/> when it holds that name (a trained
    /// part's, from <see cref="IVisionTuningPart.Export"/>), else the model's trained vision tensors
    /// (<see cref="TrainedVisionTensors"/>), else the checkpoint's own, as stored.
    /// </summary>
    public void SaveHuggingFace(string folder, IReadOnlyDictionary<string, Tensor>? vision) => SaveHuggingFace(folder, vision, null);

    /// <summary>
    /// <see cref="SaveHuggingFace(string, IReadOnlyDictionary{string, Tensor})"/> with each weight's values
    /// from <paramref name="values"/> (in the model's layout, asked for one tensor at a time as it is written, in the order
    /// of <see cref="StoredTensors"/>), or from the tensor itself when null: a checkpoint streamed from the ranks holding a
    /// model's parts (plan 23, C8b) writes the file one process writes, holding one tensor at a time.
    /// </summary>
    public void SaveHuggingFace(string folder, IReadOnlyDictionary<string, Tensor>? vision, Func<StoredTensor, float[]>? values)
    {
        using var source = Vision is null && _partTensors.Count == 0 ? null : NotOnTheNewDesign.Yet<ICheckpointFormat>($"Picking the checkpoint format of {Folder}").Open(Folder);
        var visionTensors = Vision is null ? [] : Vision.StoredTensors.Order(StringComparer.Ordinal).Select(name =>
            vision is not null && vision.TryGetValue(name, out var trained) ? (name, [.. trained.Shape], (Func<float[]>)trained.ToArray)
            : _vision.TryGetValue(name, out var kept) ? (name, kept.Shape, () => kept.Values)
            : (name, source!.ShapeOf(name), (Func<float[]>)(() => source!.Read(name)))).ToList();
        // The parts' tensors: the kept trained ones (KeepTrainedParts), else the checkpoint's, as stored.
        var partTensors = _partTensors.Values.SelectMany(owned => owned).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal).Select(name =>
            _parts.TryGetValue(name, out var kept) ? (name, kept.Shape, (Func<float[]>)(() => kept.Values))
            : (name, source!.ShapeOf(name), (Func<float[]>)(() => source!.Read(name)))).ToList();
        WriteTensors(folder, [.. visionTensors, .. partTensors], values);
        string target = Path.GetFullPath(folder);
        foreach (string file in Directory.GetFiles(Folder))
        {
            string name = Path.GetFileName(file);
            bool metadata = name.EndsWith(".json", StringComparison.OrdinalIgnoreCase) && name != "model.safetensors.index.json"
                || name.EndsWith(".jinja", StringComparison.OrdinalIgnoreCase) || name.EndsWith(".model", StringComparison.OrdinalIgnoreCase)
                || name is "merges.txt" or "vocab.txt";
            string destination = Path.Combine(target, name);
            if (metadata && !name.StartsWith("adapter_", StringComparison.Ordinal)
                && !string.Equals(Path.GetFullPath(file), destination, StringComparison.Ordinal))
            {
                File.Copy(file, destination, overwrite: true);
            }
        }
    }

    // model.safetensors of the network (LoRA merged first), each weight under its checkpoint name and layout.
    private void WriteTensors(string folder, IEnumerable<(string Name, int[] Shape, Func<float[]> Values)>? extra = null,
        Func<StoredTensor, float[]>? values = null)
    {
        Network.MergeLora();
        Directory.CreateDirectory(folder);
        // The header is written from the shapes; each tensor is copied to the host (and transposed) only when it is written.
        var tensors = new List<(string, int[], Func<float[]>)>();
        foreach (var stored in StoredTensors())
        {
            var tensor = stored.Tensor;
            tensors.Add((stored.Name, stored.Shape, () =>
            {
                var read = values?.Invoke(stored) ?? tensor.ToArray();
                return stored.Transposed ? HostParallel.Transpose(read, tensor.Shape[0], tensor.Shape[1]) : read;
            }));
        }

        foreach (var item in extra ?? [])
        {
            tensors.Add(item);
        }

        string path2 = Path.Combine(folder, "model.safetensors");
        string temporary = path2 + ".writing";
        SafeTensorsWriter.Write(temporary, tensors, new Dictionary<string, string> { ["format"] = "pt" });
        File.Move(temporary, path2, overwrite: true);                 // a model saved over its own file reads it to the end first
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Network.Dispose();
        _source?.Dispose();
        _source = null;
        _lazyReader?.Dispose();
        _lazyReader = null;
    }

    // The checkpoint a lazy load's layers are read from on first use (PretrainedOptions.LazyLayers), open until disposed.
    private IDisposable? _lazyReader;

    /// <summary>
    /// The decoder layers built so far: every layer, unless the load is lazy (<see cref="PretrainedOptions.LazyLayers"/>),
    /// where a layer is built on its first use.
    /// </summary>
    public int LayersBuilt => Network.Descendants().Count(m => m is DecoderBlock);

    /// <summary>The decoder layers on the device now (<see cref="Placement"/> at load; a placement following the memory moves them at step boundaries), or null when not placed.</summary>
    public int? DeviceLayersNow => Network.Children().Select(m => m switch { DecoderBlock b => b.Layers, LazyLayer l => l.Layers, _ => null })
        .FirstOrDefault(l => l is not null)?.DeviceLayers;
}
