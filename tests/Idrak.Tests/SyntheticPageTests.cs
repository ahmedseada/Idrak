// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Data;
using Idrak.Imaging;
using Idrak.Text;

// Plan 18, phase 5: tuning data from text: records' answers drawn as pages, scanned, written again with the page and the
// drawn text, read back by the tuning data formats.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] SyntheticDataGroup =
    [
        ("synthetic data: handwriting habits are an option, off by default (habits=none, the same bytes and labels); a rules file's habits drawn; bad rules files and settings refused", SyntheticHabits),
    ];

    private static string SyntheticFolder(string name)
    {
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-synthetic-{name}-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        return folder;
    }

    private static (int Code, string Out, string Err) SynthCli(params string[] args)
    {
        var output = new StringWriter();
        var error = new StringWriter();
        int code = Idrak.Cli.CommandLine.Run([.. args, "-C", Path.Combine(Path.GetTempPath(), "idrak-cli-synth-no-config.json")], output, error);
        return (code, output.ToString(), error.ToString());
    }
}
