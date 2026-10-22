// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;
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

    /// <summary>The profile the plug-in registers.</summary>
    public static ICodingLanguage Language { get; } = new CodingLanguage(LanguageName, "Python: its caches and virtual environment skipped (no commands)")
    {
        Markers = ["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "Pipfile"],
        IgnoredDirectories = ["__pycache__", ".venv"],
    };

    /// <summary>The plug-in, as <see cref="IdrakPlugins"/> records it ("python-language"; listed in <see cref="TextPlugins.All"/>).</summary>
    public static IIdrakPlugin Plugin { get; } = new PythonLanguageRegistration();

    /// <summary>Whether the language is registered.</summary>
    public static bool IsRegistered => Plugin.IsRegistered;

    /// <summary>Registers the language (through <see cref="IdrakPlugins"/>); once: calling it again changes nothing.</summary>
    public static void Register() => IdrakPlugins.Register(Plugin);

    /// <summary>Unregisters the language, leaving <see cref="CodingLanguages"/> as <see cref="Register"/> found it.</summary>
    public static void Unregister() => IdrakPlugins.Unregister(Plugin);

    private sealed class PythonLanguageRegistration() : IdrakPlugin("python-language")
    {
        protected override void RegisterParts() => Part(() => CodingLanguages.Register(Language), () => CodingLanguages.Unregister(LanguageName));
    }
}
