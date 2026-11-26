// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;

namespace Idrak.Abstraction.Data;

/// <summary>
/// The converted-font cache: a face a font map made of a legacy font (<see cref="FontMaps"/>) is written once as a
/// complete
/// font file by the writer <c>Writer</c> ("truetype": its characters, its own and drawn glyphs, and the layout
/// tables the map made) and kept in <see cref="Folder"/>; every later read of the same font opens that file, a plain
/// font
/// with its own layout tables (so the maps' detection leaves it alone), and maps and draws nothing. A kept file is
/// named
/// by a hash of the font file's bytes, the face's place in it, the map, the mapper, the drawing settings
/// (<see cref="FontMapDrawing"/>, a page's chances aside), the writer and the libraries that made it, so a change to
/// any of them makes a new one. It is written to a temporary file and then moved into place, so threads and processes that
/// share the folder never read a part-written file; it is written only where the drive has the room. A read gives back
/// the converted face whether it was just made or found, so what is drawn does not depend on the cache being warm.
/// A font no map fits is read exactly as before. <see cref="Variable"/> sets the folder or turns the cache off.
/// </summary>
public static class FontCache
{
    /// <summary>
    /// The variable that sets <see cref="Folder"/> (a folder) or turns the cache off ("0", "off", "false" or "no"; a legacy
    /// font is then mapped and drawn at every read): "IDRAK_FONT_CACHE".
    /// </summary>
    public const string Variable = "IDRAK_FONT_CACHE";

    private static string? s_folder;

    /// <summary>Whether mapped faces are converted and kept (on unless <see cref="Variable"/> turns it off).</summary>
    public static bool Enabled { get; set; } = !ComputeResources.SwitchedOff(Environment.GetEnvironmentVariable(Variable));

    /// <summary>
    /// Where converted fonts are kept: as set, else the folder <see cref="Variable"/> names, else "fonts" in
    /// <see cref="ComputeResources.CacheFolder"/> (the user's cache folder); setting null brings that back. A relative
    /// value is made absolute when read.
    /// </summary>
    /// <exception cref="ArgumentException">Set to an empty or blank folder.</exception>
    [AllowNull]
    public static string Folder
    {
        get => Volatile.Read(ref s_folder) is { } set ? Path.GetFullPath(set) : FolderFrom(Environment.GetEnvironmentVariable(Variable));
        set
        {
            if (value is not null)
            {
                ArgumentException.ThrowIfNullOrWhiteSpace(value);
            }

            Volatile.Write(ref s_folder, value);
        }
    }

    // The folder from the variable's value: a folder, or the default one when unset or a switch turned on.
    internal static string FolderFrom(string? variable) =>
        variable?.Trim() is { Length: > 0 } value && !ComputeResources.SwitchedOff(value) && value is not ("1" or "on" or "true" or "yes")
            ? Path.GetFullPath(value)
            : Path.Combine(ComputeResources.CacheFolder, "fonts");

    // The bytes free on the drive of `folder` (asked of the folder, or of its root where the system takes only a root), or
    // null when the system does not say.
    private static long? Room(string folder)
    {
        foreach (string? asked in (string?[])[folder, Path.GetPathRoot(folder)])
        {
            if (string.IsNullOrEmpty(asked))
            {
                continue;
            }

            try
            {
                return new DriveInfo(asked).AvailableFreeSpace;
            }
            catch (Exception e) when (e is IOException or ArgumentException or UnauthorizedAccessException or PlatformNotSupportedException)
            {
                // the next way of asking, or nothing known
            }
        }

        return null;
    }

    // An implementation's type and its assembly's build (a rebuilt assembly may convert differently).
    private static string Identity(object implementation)
    {
        var type = implementation.GetType();
        string build;
        try
        {
            build = type.Assembly.ManifestModule.ModuleVersionId.ToString("N");
        }
        catch (Exception e) when (e is NotSupportedException or InvalidOperationException)
        {
            build = type.Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? "";
        }

        return $"{type.FullName} {build}";
    }
}
