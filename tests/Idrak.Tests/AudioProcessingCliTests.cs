// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Cli;
using Idrak.Cli.Shared;

// Plan 31, P4: the audio pipeline on the command line, run in-process as the audio CLI tests do: the help of idrak audio
// process, the usage errors given before a pipeline is made, and SpeechTrainingOptions.Processing's default.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] AudioProcessingCliGroup =
    [
        ("audio processing: cli: audio process's help and Arabic summary, 'idrak audio' naming its subcommand and IDRAK_AUDIO_CLEAN in transcribe's environment", d => { CpuOnly(d); AudioProcessingCliList(); }),
        ("audio processing: cli: audio process names a missing --pipeline", d => { CpuOnly(d); AudioProcessingCliProcess(); }),
        ("audio processing: cli: data audio refuses --clean-settings beside a pipeline", d => { CpuOnly(d); AudioProcessingCliDataAudio(); }),
        ("audio processing: cli: SpeechTrainingOptions carry no processing by default (none in their JSON)", AudioProcessingCliTraining),
    ];

    private static void AudioProcessingCliList()
    {
        foreach (string name in new[] { "audio process" })
        {
            var (helpCode, help, _) = AudioCli([.. name.Split(' '), "--help"]);
            Check(helpCode == 0 && help.Contains("\nExamples:\n", StringComparison.Ordinal) && help.Contains($"idrak {name}", StringComparison.Ordinal)
                  && help.TrimEnd().Split('\n').All(l => l.Length <= 120), $"idrak {name} --help: {help}");
            Check(Messages.Arabic.ContainsKey(CommandTable.All.First(c => c.Name == name).Summary), $"idrak {name}: no Arabic summary");
        }

        var (group, _, groupError) = AudioCli("audio");
        Check(group == 2 && groupError.Contains("idrak audio process", StringComparison.Ordinal), $"'idrak audio': {groupError}");
        Check(Help.For(CommandTable.All.First(c => c.Name == "transcribe")).Contains(AudioFiles.AudioCleanVariable, StringComparison.Ordinal)
              && EnvironmentVariables.All.Any(v => v.Name == AudioFiles.AudioCleanVariable), "IDRAK_AUDIO_CLEAN in the table and transcribe's help");
        foreach (string name in new[] { "transcribe", "speak", "data audio", "voice train" })
        {
            var (helpCode, help, _) = AudioCli([.. name.Split(' '), "--help"]);
            Check(helpCode == 0 && help.Contains(name == "transcribe" || name == "speak" ? "--audio-clean" : "PIPELINE", StringComparison.Ordinal), $"idrak {name} --help names the pipeline");
        }
    }

    private static void AudioProcessingCliProcess()
    {
        string root = TempFolder("processing-cli"), wav = AudioData("wav/stereo16.wav");
        try
        {
            var missing = AudioCli("audio", "process", wav, "-o", Path.Combine(root, "x.wav"));
            Check(missing.Code == 2 && missing.Err.Contains("--pipeline", StringComparison.Ordinal), $"no pipeline: {missing.Err}");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void AudioProcessingCliDataAudio()
    {
        string root = TempFolder("processing-cli-data"), wav = AudioData("wav/pcm16.wav");
        try
        {
            const string Settings = "{\"target_decibels\": -18, \"high_pass_hertz\": 100}";
            const string Text = "highpass=80, normalize=-20 peak=0.9";
            var both = AudioCli("data", "audio", wav, "-o", Path.Combine(root, "x"), "--clean", Text, "--clean-settings", Settings);
            Check(both.Code == 2 && both.Err.Contains("--clean-settings", StringComparison.Ordinal), $"--clean-settings beside a pipeline: {both.Err}");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void AudioProcessingCliTraining(Device device)
    {
        _ = device;
        Check(SpeechTrainingOptions.FromJson(new JsonObject()).Processing.IsEmpty && new SpeechTrainingOptions().ToJson()["processing"] is null, "no processing by default");
    }
}
