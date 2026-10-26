// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;

namespace Idrak.Audio.Plugins.RnNoise;

/// <summary>
/// RNNoise (xiph's learned noise suppressor, its classic recurrent network) as the live audio processor
/// <see cref="ProcessorName"/> ("rnnoise"), registered in <see cref="AudioProcessors"/>: a pipeline such as
/// <c>highpass=80, rnnoise, agc</c> then names it. The library registers no learned denoiser; without this plug-in
/// "rnnoise" is an unknown processor. Its model is RNNoise's model file, read when a stage opens
/// (<see cref="LocateModel"/>).
/// </summary>
public static class RnNoisePlugin
{
    /// <summary>The processor's name in a pipeline, "rnnoise".</summary>
    public const string ProcessorName = "rnnoise";

    /// <summary>The environment variable naming the model file when a step gives no <c>model=</c>: "IDRAK_RNNOISE_MODEL".</summary>
    public const string ModelVariable = "IDRAK_RNNOISE_MODEL";

    /// <summary>The step's option naming the model file (or a model's name in the cache's rnnoise folder): "model".</summary>
    public const string ModelOption = "model";

    /// <summary>The plug-in, as <see cref="IdrakPlugins"/> records it ("rnnoise"; listed in <see cref="AudioPlugins.All"/>).</summary>
    public static IIdrakPlugin Plugin { get; } = new RnNoiseRegistration();

    /// <summary>Whether the processor is registered.</summary>
    public static bool IsRegistered => Plugin.IsRegistered;

    /// <summary>
    /// Where the model is looked for when neither a step's <c>model=</c> nor <see cref="ModelVariable"/> names one:
    /// rnnoise/rnnoise.rnnn under the library's cache folder (<see cref="ComputeResources.CacheFolder"/>: <c>IDRAK_CACHE</c>,
    /// or <c>~/.cache/idrak</c>).
    /// </summary>
    public static string DefaultModelPath => Path.Combine(ModelFolder, "rnnoise.rnnn");

    private static string ModelFolder => Path.Combine(ComputeResources.CacheFolder, "rnnoise");

    /// <summary>Registers the processor (through <see cref="IdrakPlugins"/>); once: calling it again changes nothing.</summary>
    public static void Register() => IdrakPlugins.Register(Plugin);

    /// <summary>Unregisters the processor, leaving <see cref="AudioProcessors"/> as <see cref="Register"/> found it.</summary>
    public static void Unregister() => IdrakPlugins.Unregister(Plugin);

    /// <summary>
    /// The model file a step uses: <paramref name="given"/> (a step's <c>model=</c>: a file, or a name such as
    /// <c>sh</c> for rnnoise/sh.rnnn in the cache), else the file <see cref="ModelVariable"/> names, else
    /// <see cref="DefaultModelPath"/>. The file is RNNoise's model format (the text its <c>rnnoise_model_from_file</c>
    /// reads, first line "rnnoise-nu model file version 1").
    /// </summary>
    /// <exception cref="FileNotFoundException">None is there: the message says where it looked and how to get the model.</exception>
    public static string LocateModel(string? given = null)
    {
        if (given is { Length: > 0 })
        {
            if (File.Exists(given))
            {
                return given;
            }

            string? named = given.IndexOfAny(['/', '\\']) < 0 && Path.GetExtension(given).Length == 0 ? Path.Combine(ModelFolder, given + ".rnnn") : null;
            if (named is not null && File.Exists(named))
            {
                return named;
            }

            throw new FileNotFoundException($"rnnoise model={given}: no such model file{(named is null ? "" : $" (nor {named})")}. {HowToGet}", given);
        }

        if (Environment.GetEnvironmentVariable(ModelVariable) is { Length: > 0 } variable)
        {
            return File.Exists(variable) ? variable
                : throw new FileNotFoundException($"{ModelVariable}={variable}: no such model file. {HowToGet}", variable);
        }

        string path = DefaultModelPath;
        return File.Exists(path) ? path
            : throw new FileNotFoundException($"RNNoise's model was not found: the step names no model=PATH, {ModelVariable} is not set and {path} does not exist. {HowToGet}", path);
    }

    private static string HowToGet =>
        $"Get xiph's classic RNNoise weights (BSD-3-Clause) once, from an Idrak checkout (it needs Python and git): python3 tools/audio/make_rnnoise_reference.py --model-only \"{DefaultModelPath}\"; "
        + $"or copy tests/Idrak.Tests/data/audio/rnnoise/rnnoise.rnnn there. Any model in RNNoise's file format ({RnNoiseModel.Header}) can be named with rnnoise model=PATH or {ModelVariable}.";

    private sealed class RnNoiseRegistration() : IdrakPlugin("rnnoise")
    {
        protected override void RegisterParts() =>
            Part(() => AudioProcessors.Register(new RnNoiseProcessor()), () => AudioProcessors.Unregister(ProcessorName));
    }
}
