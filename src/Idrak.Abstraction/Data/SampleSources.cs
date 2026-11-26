// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Data;

/// <summary>
/// The sample sources tools and configuration files name their data with (picking one by its name is not on the new
/// design yet). The library's:
/// <list type="bullet">
/// <item><c>csv</c>: <c>CsvSource</c> (Idrak); options <c>target</c> (required, comma-separated), <c>ignore</c>,
/// <c>delimiter</c>, <c>header</c> (true or false).</item>
/// <item><c>images</c>: <c>ImageFolderSource</c> (Idrak.Imaging) over class folders; options <c>channels</c>, <c>height</c>,
/// <c>width</c> (each defaults to the first image's).</item>
/// <item><c>tokens</c>: <c>TokenFileSource</c> (Idrak); options <c>length</c> (required), <c>stride</c>, <c>type</c>
/// (uint16, int32 or uint32).</item>
/// <item><c>npy</c>: <c>NpySource</c> (Idrak); options <c>targets</c> (a second .npy file), <c>classes</c>.</item>
/// </list>
/// </summary>
public static class SampleSources
{
}
