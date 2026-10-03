// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// idrak-tune: the fine-tuning tool, now also `idrak tune`. Its code lives in src/Idrak.Cli/Commands/Train/TuneTool.cs,
// compiled into this forwarder too, so both behave the same.
using Idrak.Cli.Commands.Train;
using Idrak.Cli.Shared;

Console.OutputEncoding = System.Text.Encoding.UTF8;                       // "…" and emoji on Windows consoles
return TuneTool.RunTool(args, ToolConsole.Standard());
