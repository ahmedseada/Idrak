// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// idrak-data: datasets from the command line, now also `idrak data`. Its code lives in
// src/Idrak.Cli/Commands/Data/DataTool.cs, compiled into this forwarder too, so both behave the same.
using Idrak.Cli.Commands.Data;
using Idrak.Cli.Shared;

Console.OutputEncoding = System.Text.Encoding.UTF8;                       // "…" and emoji on Windows consoles
return DataTool.RunTool(args, ToolConsole.Standard());
