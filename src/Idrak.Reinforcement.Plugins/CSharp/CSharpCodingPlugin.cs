// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Reinforcement;
using Idrak.Reinforcement.Plugins.Coding;
using Idrak.Text.Plugins.CSharp;

namespace Idrak.Reinforcement.Plugins.CSharp;

/// <summary>
/// Agentic coding on C# projects as a plug-in: the <c>csharp</c> environment (<see cref="CSharpEnvironmentVariant"/>, the
/// coding environment with dotnet's defaults) and its rewards (<see cref="CSharpRewards"/>: csharp-tests and
/// csharp-build). It builds on the coding plug-in (<see cref="CodingPlugin"/>) and the C# coding language of
/// Idrak.Text.Plugins (<see cref="CSharpLanguagePlugin"/>) and copies neither.
/// </summary>
public static class CSharpCodingPlugin
{
}
