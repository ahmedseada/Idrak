// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Cli;
using Idrak.Cli.Commands.Serve;

CommandServers.Register(new AspNetCoreServers());       // serve, ui and the cluster dashboard: the tool's ASP.NET Core servers
return CommandLine.Run(args, Console.Out, Console.Error);
