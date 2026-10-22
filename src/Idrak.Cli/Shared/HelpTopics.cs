// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using Idrak.Cli.Commands.Health;

namespace Idrak.Cli.Shared;

/// <summary>
/// Concept pages for <c>idrak help topics</c> and <c>idrak help TOPIC</c>: devices, formats, models, precision, plugins,
/// config, env and exit codes. The format and model lists are read from the registries, so plug-ins show up.
/// </summary>
internal static class HelpTopics
{
    /// <summary>The topics, in listing order, with one line each.</summary>
    public static IReadOnlyList<(string Name, string Summary)> All { get; } =
    [
        ("devices", "Device names, the default device, backends and what each device reports"),
        ("formats", "Weight, KV cache, checkpoint, dataset and tool-call formats"),
        ("models", "Model names (hub ids, folders, GGUF, .ikm, aliases) and the supported families"),
        ("precision", "float32, bfloat16 and float8 products; packed weights; KV cache formats"),
        ("plugins", "Loading assemblies that add formats, families, ops and devices"),
        ("config", "The config file, its keys and profiles"),
        ("env", "Every environment variable Idrak reads"),
        ("exit-codes", "What the tool's exit codes mean"),
        ("arabic", "Messages in Arabic (--lang ar) and how they are shown right to left in a terminal"),
        ("audio", "Sound files, speech recognition and synthesis, voices: the commands and what is registered"),
        ("reinforcement", "Reinforcement learning: idrak reinforce (language models, classic environments, DQN), what is registered"),
    ];

    /// <summary>The topic names.</summary>
    public static IEnumerable<string> Names => All.Select(t => t.Name);

    /// <summary>The page of <paramref name="name"/> (also "exit codes", "exitcodes", "environment"), or null.</summary>
    public static string? Page(string name) => PageText(name) is { } text ? text.TrimEnd('\n') + "\n" : null;

    private static string? PageText(string name) => name.ToLowerInvariant().Replace(' ', '-') switch
    {
        "topics" => Index(),
        "devices" => Devices,
        "formats" => Formats(),
        "models" => Models(),
        "precision" => Precision,
        "plugins" or "plug-ins" => Plugins,
        "config" => Config,
        "env" or "environment" => "idrak help env: every environment variable Idrak reads\n" + EnvironmentVariables.Reference(),
        "exit-codes" or "exitcodes" => ExitCodesPage,
        "arabic" or "lang" or "language" or "rtl" => Arabic,
        "audio" or "speech" or "sound" => Audio(),
        "reinforcement" or "rl" => Reinforcement(),
        _ => null,
    };

    private static string Index()
    {
        var text = new StringBuilder("idrak help TOPIC: concept pages\n\n");
        int width = All.Max(t => t.Name.Length);
        foreach (var (name, summary) in All)
        {
            text.Append("  ").Append(name.PadRight(width)).Append("  ").Append(summary).Append('\n');
        }

        return text.Append("\nExample: idrak help devices\n").ToString();
    }

    private const string Devices = """
        idrak help devices

        A device is where tensors live and their math runs: cpu, cuda:N, vulkan:N, hip:N, or a kind a plug-in
        registers. -d/--device NAME chooses it for a command; the config's "device" gives a default; otherwise the
        default device is the best GPU found (CUDA first; a Vulkan or HIP GPU only when IDRAK_VULKAN_DEFAULT=1 or
        IDRAK_HIP_DEFAULT=1), else the CPU.

        Devices a backend can drive but does not list (a software Vulkan driver, a GPU another backend already
        drives) are reached by name only. Vulkan devices are numbered by what they report (discrete, integrated,
        virtual, CPU, each by UUID), so vulkan:N names the same GPU in every run.

        Each device reports its memory, compute units, subgroup (warp, wavefront) size, matrix units and the kernel
        width chosen from its limits; nothing is chosen by a card's or vendor's name. Kernel choices that limits do
        not decide are measured on the device the first time and kept in the cache (IDRAK_AUTOTUNE=0: formulas only).

        Commands: idrak devices, idrak doctor, idrak bench, idrak report.
        Variables: IDRAK_DISABLE_CUDA, IDRAK_DISABLE_VULKAN, IDRAK_DISABLE_HIP, VK_ICD_FILENAMES (idrak help env).
        """;

    private static string Formats()
    {
        var text = new StringBuilder("""
            idrak help formats

            Weight formats (-w/--weights) pack a model's matrices: int8 and int4 (per-row scales), bfloat16, or a format
            a plug-in registers. KV cache formats (-k/--kv) store the attention cache. Checkpoint formats read and write
            weights (safetensors, GGUF, ...); dataset file formats read rows (Parquet, JSON Lines, CSV, ...); tool-call
            formats parse the calls a model writes, detected from its chat template.

            Registered now:

            """);
        foreach (var c in Registries.Snapshot().Where(c => c.Format))
        {
            text.Append("  ").Append(c.Title).Append(": ").Append(string.Join(", ", c.Names)).Append('\n');
        }

        return text.Append("\nCommands: idrak formats, idrak plugins list, idrak quantize, idrak inspect.\n").ToString();
    }

    private static string Audio()
    {
        static string Names(IEnumerable<string> names) => names.Order(StringComparer.OrdinalIgnoreCase).ToList() is { Count: > 0 } list ? string.Join(", ", list) : "none (a plug-in brings them: -P)";
        return $$$"""
            idrak help audio

            Sound files are decoded and written in C#, with no outside program: WAV, FLAC and MP3 (MP2 read too) are
            built in. Speech models are families, registered by plug-ins and never by the library: the tool registers
            Idrak.Audio.Plugins' (Whisper, VITS with uroman's Arabic romanizer, OpenVoice, IdrakAudio, and the espeak-ng
            phonemizer, which runs the espeak-ng program the user installs), another comes with -P, and a model whose
            family is not registered is refused with "not registered" and the -P remedy.

              idrak transcribe MODEL AUDIO...       speech to text: text, JSON, SRT or WebVTT
              idrak speak MODEL TEXT -o OUT         text to speech, in a speaker or an installed voice (--voice)
              idrak voice add NAME --from CLIPS     install a person's voice, with their consent (voice list, show, ...)
              idrak data audio FILES -o DIR         convert, resample and split into a folder with manifest.jsonl
              idrak data mel FILES -o DIR           a model's log-mel input features as .npy

            Registered now (with the -P plug-ins of this command):
              codecs (read): {{{Names(Idrak.Audio.Abstractions.AudioCodecs.Names)}}} ({{{string.Join(" ", Idrak.Audio.Abstractions.AudioCodecs.Extensions)}}})
              encoders (write): {{{Names(Idrak.Audio.Abstractions.AudioEncoders.Names)}}} ({{{string.Join(" ", Idrak.Audio.Abstractions.AudioEncoders.Extensions)}}})
              feature extractors: {{{Names(Idrak.Audio.Abstractions.AudioFeatureExtractors.Names)}}}
              speech recognition families: {{{Names(Idrak.Audio.Abstractions.SpeechFamilies.Names)}}}
              speech synthesis families: {{{Names(Idrak.Audio.Abstractions.SpeechSynthesisFamilies.Names)}}}
              vocoders: {{{Names(Idrak.Audio.Abstractions.Vocoders.Names)}}}
              voice converter families: {{{Names(Idrak.Audio.Abstractions.VoiceConverters.Names)}}}
              splitters (data audio --split): {{{Names(Idrak.Audio.Abstractions.AudioSplitters.Names)}}}

            Installed voices live in {{{Idrak.Audio.FolderVoiceStore.DefaultRoot}}} (IDRAK_VOICES; --voices DIR for one command).
            A voice is a person's likeness: voice add records their consent, voice show prints it; the library cannot
            verify it. The library plays no sound: open what speak writes in any player.
            Variables: IDRAK_VOICES, IDRAK_STFT_PATH, IDRAK_CACHE (idrak help env).
            """;
    }

    private static string Reinforcement()
    {
        static string Names(IEnumerable<string> names) => names.ToList() is { Count: > 0 } list ? string.Join(", ", list) : "none (a plug-in brings them: -P)";
        try
        {
            CommandLine.RegisterPluginPackages();          // the tool's plug-in packages, as every command registers them (idempotent)
        }
        catch (UsageException)
        {
            // IDRAK_PLUGIN_PACKAGES names no package: the page lists what is registered, and the command says what is wrong.
        }

        return $$$"""
            idrak help reinforcement

            A language model's adapters learn from what the model does: each iteration plays episodes in an environment
            with the model's own sampled replies, scores them with a reward, estimates each episode's advantage (against
            the others of its group: the same situation played again) and updates the adapters with a policy loss. The
            general part (rollouts, advantages, losses, the trainer, checkpoints) is Idrak.Reinforcement; the language-model
            policy, the prompts environment and the text rewards are Idrak.Text's; environments several apps use (coding)
            are plug-ins of Idrak.Reinforcement.Plugins, which the tool registers at startup. A name nobody registered is
            refused with "not registered".

            An environment whose observations are numbers (the classic plug-in's bandit, cartpole, pendulum and mountain-car,
            or a plug-in's) needs no model: idrak reinforce builds a small network from --hidden, --activation and
            --value-head and trains it on-policy (ppo, reinforce, grpo: a categorical policy for discrete actions, a
            Gaussian one for a box) or off-policy (dqn: a Q-network with a target network and a replay buffer, discrete
            actions only). DIR receives the policy or the Q-network and idrak-reinforce.json, its settings.

              idrak reinforce MODEL --env prompts --data FILE --reward exact -o DIR
                                                      prompts with references, the reply scored by a reward expression
              idrak reinforce MODEL --env coding --suite DIR -o DIR
                                                      coding tasks: the model works with the coding tools in a sandbox,
                                                      rewarded by the hidden tests
              idrak reinforce MODEL ... --teacher BIG    a teacher's distributions on the model's own replies beside the reward
              idrak reinforce MODEL --algorithm distill --teacher BIG --data FILE -o DIR
              idrak distill --teacher BIG --student MODEL --data FILE --on-policy -o DIR
                                                      on-policy distillation alone (Agarwal et al. 2024); on a
                                                      cluster each rank samples its shards' prompts (--shards)
              idrak reinforce --env cartpole --algorithm ppo --iterations 50 -o runs/cartpole
                                                      PPO on CartPole: a categorical policy of 64, 64 tanh with a value head
              idrak reinforce --env cartpole --algorithm dqn --steps 50000 -o runs/cartpole-dqn
                                                      DQN on CartPole (IDRAK_RL_DQN_* for its settings)
              idrak reinforce --env pendulum --algorithm ppo -o runs/pendulum
                                                      a Gaussian policy for a box of actions (dqn refuses a box)
              idrak reinforce --list                  every registered name below, with its options
              idrak cluster sim --topology FILE -- reinforce MODEL ... -o DIR
                                                      on a simulated cluster: each rank plays its shards' episodes (--shards,
                                                      --spread), rank 0 writes the adapter, checkpoints and log (--cluster
                                                      FILE: one rank of a real cluster)
              idrak cluster sim --topology FILE --dashboard -- reinforce MODEL ... -o DIR
                                                      steered from the cluster dashboard (idrak cluster ui starts runs too):
                                                      paused, resumed and stopped at an iteration, every rank at the same one;
                                                      its iterations, rewards and pass rate shown from DIR/log.jsonl

            Registered now (with the -P plug-ins of this command):
              environments: {{{Names(Idrak.Abstraction.Reinforcement.EnvironmentKinds.Names)}}}
              rewards: {{{Names(Idrak.Abstraction.Reinforcement.RewardFunctions.Names)}}}
              algorithms (policy losses): {{{Names(Idrak.Abstraction.Reinforcement.PolicyLosses.Names)}}}, and distill
              off-policy algorithms (Q-learning losses): {{{Names(Idrak.Reinforcement.Abstractions.QLosses.Names)}}}
              advantage estimators: {{{Names(Idrak.Abstraction.Reinforcement.AdvantageEstimators.Names)}}}
              command sandboxes: {{{Names(CommandSandboxes.Names)}}}
              teacher divergences: {{{Names(Idrak.Text.Abstractions.TeacherDivergences.Names)}}}

            A run is reproducible from its seed (--seed, IDRAK_RL_SEED) on every device and thread count; it writes
            checkpoints into DIR/checkpoints and goes on from the latest with --resume. A coding episode runs the model's
            commands on this machine: the process sandbox limits the programs, time, output and memory; container
            (Docker or Podman, IDRAK_SANDBOX=container with an image of your own) isolates them.
            Variables: IDRAK_RL_* (every run setting; the command line wins), IDRAK_RL_DQN_* (dqn's), IDRAK_SANDBOX_*
            (idrak help env).
            """;
    }

    private static string Models()
    {
        var families = Registries.Snapshot();
        return $$$"""
            idrak help models

            A model argument is, everywhere: a Hugging Face id (Qwen/Qwen3-0.6B), a local folder (config.json and
            safetensors), a GGUF file, a .ikm model package, or an alias from the config
            ("aliases": {"qwen": {"model": "Qwen/Qwen3-0.6B", "weights": "int8", "kv": "int8"}}). Hub models are
            downloaded into the cache once (idrak pull) and found there afterwards; -w/--weights and -k/--kv choose the
            weight and KV cache formats, --context the context length.

            Families (config.json "model_type"): {{{string.Join(", ", families.First(c => c.Key == "families").Names)}}}
            GGUF architectures: {{{string.Join(", ", families.First(c => c.Key == "gguf").Names)}}}
            RoPE scalings: {{{string.Join(", ", families.First(c => c.Key == "rope").Names)}}}

            Commands: idrak pull, list, show, memory, chat, run, serve, alias set.
            Variables: HF_TOKEN, HF_HOME, HF_ENDPOINT, IDRAK_CACHE (idrak help env).
            """;
    }

    private const string Precision = """
        idrak help precision

        Products run in float32 by default. IDRAK_MATMUL=bf16 (or fp8) and the library's precision scopes run them
        with bfloat16 or float8 operands and float32 sums where the device has matrix units; elsewhere float32 stays.
        Weights can be packed (-w int8, int4, bf16, or a registered format): smaller and faster to read, with a small
        loss (idrak quantize reports a perplexity check). The KV cache has its own formats (-k), which bound the memory
        a long context takes (idrak memory shows what fits where).

        Commands: idrak quantize, idrak memory, idrak perplexity, idrak bench.
        """;

    private const string Plugins = """
        idrak help plugins

        -P/--plugin PATH (repeatable), or the config's "plugins" list, loads an assembly before the command runs. Its
        module initializer runs, then any public static RegisterIdrakPlugin() method: there it registers weight and KV
        formats, model families, RoPE scalings, tool-call formats, checkpoint and dataset formats, graph ops, layer
        types, ONNX ops or device kinds through the libraries' public Register methods. Everything it adds is then
        available to every command (e.g. -w MYFORMAT).

        The tool itself registers the model families of Idrak.Text.Plugins, Idrak.Vision.Plugins and
        Idrak.Audio.Plugins, and the environment families of Idrak.Reinforcement.Plugins, first, so a -P plug-in can
        still take a family's place.

        Commands: idrak plugins list (with -P: what the assembly added, marked "+"), idrak formats, idrak new plugin.
        """;

    private const string Arabic = """
        idrak help arabic

        --lang ar (or the config's "lang", or IDRAK_LANG=ar) prints the tool's messages in Arabic: idrak help (titles,
        command summaries, the common options), usage errors, doctor, devices, questions and the chat's own lines.
        The rest of a command's help stays English. JSON, CSV and Markdown output are never translated, so scripts
        read the same keys and values in every language.

        Most terminals show characters left to right in the order they arrive: Windows' console host (cmd and
        PowerShell windows), Windows Terminal, VS Code's terminal, xterm, kitty, Alacritty, WezTerm, foot and iTerm2.
        For them the tool shapes Arabic (each letter in its joined form, lam-alef as one ligature) and reorders every
        line by the Unicode Bidirectional Algorithm, so it reads right to left while English words, numbers and paths
        inside it keep their order. Terminals that do this themselves get the text as it is: GNOME Terminal and other
        VTE terminals (0.58 and later), Konsole, mlterm, mintty (Git Bash) and macOS Terminal. Output to a file or a
        pipe is never reordered.

        --lang-render (the config's "lang-render", IDRAK_LANG_RENDER) overrides the choice:
          auto          reorder unless the terminal does it itself or the output is not a terminal (the default)
          visual        always shape and reorder (for a terminal the tool does not recognize)
          visual-right  as visual, and lines that read right to left end at the right edge of the terminal
          logical       never: the text as stored (for a terminal that applies the algorithm itself)

        Check: idrak help --lang ar should show the title line as "idrak" followed by Arabic that reads right to left
        with joined letters. Letters apart or words in reverse order: try --lang-render visual (or logical if the text
        was right before); boxes instead of letters: choose a terminal font with Arabic (Cascadia Code, Courier New,
        DejaVu Sans Mono). The Windows console host needs such a font in the window's properties.

        Examples:
          idrak help --lang ar
          idrak doctor --lang ar --lang-render visual
          idrak config set lang ar
        """;

    private const string Config = """
        idrak help config

        The config file (--config, IDRAK_CONFIG, or ~/.idrak/config.json) gives defaults; options on the command line
        win. Keys: "device", "cache", "plugins" (a list of paths), "aliases" (NAME: {"model", "weights", "kv"}),
        "lang" (en or ar) and "lang-render" (auto, visual, visual-right or logical; idrak help arabic),
        "tokens" (written by idrak login; never printed), "profile" (the profile in use) and "profiles"
        (NAME: {keys...}): a profile's keys are read before the top-level ones, so one file serves a phone and a
        desktop. IDRAK_PROFILE chooses the profile for one shell.

        Commands: idrak init, idrak config get/set/unset/list (--profile NAME), idrak alias set, idrak login.
        """;

    private const string ExitCodesPage = """
        idrak help exit-codes

          0  success
          1  the operation failed (the message says why and what to do next; IDRAK_TRACE=1 adds the stack);
             idrak doctor: a check failed; idrak report --tests: a test failed
          2  a usage error: an unknown command or option, a missing argument, or a question with no terminal to
             ask on (add --yes)

        Ctrl+C stops long commands cleanly (a second Ctrl+C ends at once).
        """;
}
