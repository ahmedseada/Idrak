// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Cli.Shared;
using Idrak.Data;
using Idrak.Datasets;
using Idrak.Layers;
using Idrak.Optimizers;

// The Design group of the idrak tool (suggest, explain, viz), run in-process: the networks suggest writes for a
// table, an image folder, text and chat rows are read by Network.FromJson, build and train a step on the device;
// --search keeps its best candidate; explain counts the parameters of the built network.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] CliDesignGroup =
    [
    ];

    // Runs idrak in-process; returns the exit code and the captured output and errors.
    private static (int Code, string Output, string Error) RunIdrak(params string[] args)
    {
        var output = new StringWriter();
        var error = new StringWriter();
        int code = global::Idrak.Cli.CommandLine.Run(args, output, error);
        return (code, output.ToString(), error.ToString());
    }
}
