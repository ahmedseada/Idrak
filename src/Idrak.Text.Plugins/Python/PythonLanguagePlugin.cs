// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Text.Abstractions;

namespace Idrak.Text.Plugins.Python;

/// <summary>
/// Python as a coding language (<see cref="CodingLanguages"/>: "python"): a workspace with <c>pyproject.toml</c>,
/// <c>setup.py</c>, <c>setup.cfg</c>, <c>requirements.txt</c> or a <c>Pipfile</c> is Python, and its interpreter's caches
/// and virtual environment (<c>__pycache__</c>, <c>.venv</c>) are skipped (by the coding tools, and by the idrak tool's
/// data commands reading code folders). No programs, prompt lines or task commands: those are a later profile's or an app's.
/// </summary>
public static class PythonLanguagePlugin
{
    /// <summary>The language's name in <see cref="CodingLanguages"/>: "python".</summary>
    public const string LanguageName = "python";

    /// <summary>The language's profile.</summary>
    public static ICodingLanguage Language { get; } = new CodingLanguage(LanguageName, "Python: its caches and virtual environment skipped (no commands)")
    {
        Markers = ["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "Pipfile"],
        IgnoredDirectories = ["__pycache__", ".venv"],
    };
}
