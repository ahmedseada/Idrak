// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Cli;
using Idrak.Cli.Commands.Data;
using Idrak.Cli.Commands.Run;
using Idrak.Cli.Shared;
using Idrak.Models;
using Idrak.Audio.Plugins.OpenVoice;
using Idrak.Audio.Plugins.Vits;
using Idrak.Audio.Plugins.Whisper;

// Plan 21, S7: the audio commands run in-process through CommandLine.Run with captured output: their help pages, the
// audio help topic and the voice group.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] AudioCliGroup =
    [
        ("audio cli: every audio command's help has its examples and options, idrak help audio lists the registries, 'idrak voice' names its subcommands, and --format srt is transcribe's own", d => { CpuOnly(d); AudioCliHelp(); }),
    ];

    private static (int Code, string Out, string Err) AudioCli(params string[] args)
    {
        var output = new StringWriter();
        var error = new StringWriter();
        int code = StandardInput.With(new StringReader(""), () =>
            CommandLine.Run([.. args, "-C", Path.Combine(Path.GetTempPath(), "idrak-cli-audio-no-config.json")], output, error));
        return (code, output.ToString(), error.ToString());
    }

    private static JsonObject AudioCliJson(params string[] args)
    {
        var (code, output, error) = AudioCli([.. args, "-j"]);
        Check(code == 0, $"idrak {string.Join(' ', args)} -j: exit {code}\n{output}\n{error}");
        return JsonNode.Parse(output) as JsonObject ?? throw new Exception($"not one JSON object: {output}");
    }

    private static void AudioCliHelp()
    {
        string[] commands = ["transcribe", "speak", "voice add", "voice list", "voice show", "voice remove", "voice export", "voice import", "data audio", "data mel"];
        foreach (string name in commands)
        {
            var (code, help, _) = AudioCli([.. name.Split(' '), "--help"]);
            Check(code == 0 && help.Contains("\nExamples:\n", StringComparison.Ordinal) && help.Contains($"idrak {name}", StringComparison.Ordinal)
                  && help.TrimEnd().Split('\n').All(l => l.Length <= 120), $"idrak {name} --help: {help}");
            Check(Idrak.Cli.Shared.Messages.Arabic.ContainsKey(CommandTable.All.First(c => c.Name == name).Summary), $"idrak {name}: no Arabic summary");
        }

        var pageOutput = new StringWriter();
        int topic = CommandLine.Run(["help", "audio"], pageOutput, new StringWriter());       // a topic takes no options after it
        string page = pageOutput.ToString();
        Check(topic == 0 && page.Contains("idrak transcribe", StringComparison.Ordinal) && page.Contains("wav", StringComparison.Ordinal) && page.Contains("silence", StringComparison.Ordinal),
            $"idrak help audio: {page}");
        var (group, _, groupError) = AudioCli("voice");
        Check(group == 2 && groupError.Contains("idrak voice add", StringComparison.Ordinal), $"'idrak voice': {groupError}");
        Check(Help.For(CommandTable.All.First(c => c.Name == "speak")).Contains("IDRAK_VOICES", StringComparison.Ordinal)
              && Help.For(CommandTable.All.First(c => c.Name == "transcribe")).Contains("IDRAK_STFT_PATH", StringComparison.Ordinal), "the environment sections");
        var (format, _, formatError) = AudioCli("list", "--format", "srt");
        Check(format == 2 && formatError.Contains("--format", StringComparison.Ordinal), $"--format srt is transcribe's own: {formatError}");
    }
}
