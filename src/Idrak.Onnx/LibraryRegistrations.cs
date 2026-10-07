// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Onnx;

/// <summary>
/// What this assembly adds to the registries of Idrak.Abstraction, which cannot name it: the built-in ONNX import
/// operators (<see cref="OnnxImportOps"/>) and export translators (<see cref="OnnxExportOps"/>), all in
/// <see cref="OnnxBuiltIns"/>. <c>LibraryDefaults.Ensure</c> (in Idrak.Abstraction) calls <see cref="RegisterAll"/> once,
/// before either registry is first used.
/// </summary>
internal static class LibraryRegistrations
{
    private static void RegisterAll() => OnnxBuiltIns.RegisterAll();
}
