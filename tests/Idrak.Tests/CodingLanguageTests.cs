// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Formats.Tar;
using System.IO.Compression;
using System.Text;
using Idrak.Data;
using Idrak.Data.Abstractions;

// Plan 27, R4b part 1: no programming language in the library: Idrak.Data skips only .git and the editors' folders
// unless told.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] CodingLanguageGroup =
    [
    ];

    private static string LanguageTemp(string name)
    {
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-languages-{name}-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        return folder;
    }
}
