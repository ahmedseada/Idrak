// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;

namespace Idrak.Maui.Platforms.Android;

// Android's own fonts: "fonts" under the system's root directory as Android gives it (Environment.getRootDirectory, the
// system partition), read through Android's API rather than any variable.
internal sealed class AndroidSystemFonts : ISystemFonts
{
    public IReadOnlyList<string> Folders
    {
        get
        {
            string? root = global::Android.OS.Environment.RootDirectory?.AbsolutePath;
            if (string.IsNullOrEmpty(root))
            {
                return [];
            }

            string folder = Path.Combine(root, "fonts");
            return Directory.Exists(folder) ? [folder] : [];
        }
    }
}
