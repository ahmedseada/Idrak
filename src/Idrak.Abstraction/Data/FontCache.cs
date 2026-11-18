// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;

namespace Idrak.Abstraction.Data;

/// <summary>
/// The converted-font cache: a face a font map made of a legacy font (<see cref="FontMaps"/>) is written once as a complete
/// font file by the writer <see cref="Writer"/> ("truetype": its characters, its own and drawn glyphs, and the layout
/// tables the map made) and kept in <see cref="Folder"/>; every later read of the same font opens that file, a plain font
/// with its own layout tables (so the maps' detection leaves it alone), and maps and draws nothing. A kept file is named
/// by a hash of the font file's bytes, the face's place in it, the map, the mapper, the drawing settings
/// (<see cref="FontMapDrawing"/>, a page's chances aside), the writer and the libraries that made it, so a change to any
/// of them makes a new one. It is written to a temporary file and then moved into place, so threads and processes that
/// share the folder never read a part-written file; it is written only where the drive has the room. A read gives back
/// the converted face whether it was just made or found, so what is drawn does not depend on the cache being warm.
/// <see cref="FontFormats.Read(string)"/> and <see cref="FontFormats.Read(ReadOnlyMemory{byte}, string)"/> go through it;
/// a font no map fits is read exactly as before. <see cref="Variable"/> sets the folder or turns the cache off.
/// </summary>
public static class FontCache
{
    /// <summary>
    /// The variable that sets <see cref="Folder"/> (a folder) or turns the cache off ("0", "off", "false" or "no"; a legacy
    /// font is then mapped and drawn at every read): "IDRAK_FONT_CACHE".
    /// </summary>
    public const string Variable = "IDRAK_FONT_CACHE";

    private static string? s_folder;
    private static long s_hits, s_converted, s_failed;

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

    /// <summary>The <see cref="FontWriters"/> name converted fonts are written with ("truetype", the format Idrak works from).</summary>
    public static string Writer { get; set; } = "truetype";

    /// <summary>How many reads found their converted font kept.</summary>
    public static long Hits => Interlocked.Read(ref s_hits);

    /// <summary>How many faces were converted (and kept where the folder allowed).</summary>
    public static long Converted => Interlocked.Read(ref s_converted);

    /// <summary>How many faces could not be converted (no such writer, or a face the writer refused): those are used as mapped.</summary>
    public static long Failed => Interlocked.Read(ref s_failed);

    // The folder from the variable's value: a folder, or the default one when unset or a switch turned on.
    internal static string FolderFrom(string? variable) =>
        variable?.Trim() is { Length: > 0 } value && !ComputeResources.SwitchedOff(value) && value is not ("1" or "on" or "true" or "yes")
            ? Path.GetFullPath(value)
            : Path.Combine(ComputeResources.CacheFolder, "fonts");

    // The faces of a font file as the maps made them, each one a map made converted (when on); the same list when no map
    // made any.
    internal static IReadOnlyList<IFontFace> Through(ReadOnlyMemory<byte> file, string name, IReadOnlyList<IFontFace> read, IReadOnlyList<IFontFace> mapped)
    {
        if (!Enabled || ReferenceEquals(read, mapped))
        {
            return mapped;
        }

        IFontFace[]? converted = null;
        for (int i = 0; i < mapped.Count; i++)
        {
            if (!ReferenceEquals(read[i], mapped[i]) && FontMaps.MappedBy(mapped[i]) is { } map)
            {
                converted ??= [.. mapped];
                converted[i] = Convert(mapped[i], map, file, i, name);
            }
        }

        return converted ?? mapped;
    }

    // The converted face of `mapped` (face `index` of `file`, which `map` made): kept, or written now (and kept).
    internal static IFontFace Convert(IFontFace mapped, FontMap map, ReadOnlyMemory<byte> file, int index, string name)
    {
        var options = new FontWriteOptions { KeepLayout = true, KeepHinting = true };
        IFontWriter writer;
        string key;
        try
        {
            writer = FontWriters.Get(Writer);
            key = Key(file.Span, index, map, writer, options);
        }
        catch (NotRegisteredException)
        {
            Interlocked.Increment(ref s_failed);
            return mapped;
        }

        string folder = Folder;
        string path = Path.Combine(folder, key + (writer.Extensions.FirstOrDefault() ?? ".font"));
        if (Kept(path, name, mapped.GlyphCount) is { } kept)
        {
            Interlocked.Increment(ref s_hits);
            FontMaps.Remember(kept, map);
            return kept;
        }

        byte[] bytes;
        IFontFace converted;
        try
        {
            using var stream = new MemoryStream();
            writer.Write(mapped, stream, options);
            bytes = stream.ToArray();
            var faces = FontFormats.ReadFaces(bytes, name);
            converted = faces.Count == 1 && faces[0].GlyphCount == mapped.GlyphCount ? faces[0]
                : throw new InvalidDataException($"{name}: its converted font reads back as {faces.Count} faces ({string.Join(", ", faces.Select(f => f.GlyphCount))} glyphs), not one of {mapped.GlyphCount}.");
        }
        catch (Exception e) when (e is NotSupportedException or InvalidDataException or InvalidOperationException or ArgumentException)
        {
            Interlocked.Increment(ref s_failed);
            return mapped;
        }

        Store(folder, path, bytes);
        Interlocked.Increment(ref s_converted);
        FontMaps.Remember(converted, map);
        return converted;
    }

    // The face of a kept file, or null when there is none or it is not whole (it is then written again).
    private static IFontFace? Kept(string path, string name, int glyphs)
    {
        try
        {
            if (!File.Exists(path))
            {
                return null;
            }

            var faces = FontFormats.ReadFaces(File.ReadAllBytes(path), name);
            return faces.Count == 1 && faces[0].GlyphCount == glyphs ? faces[0] : null;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or InvalidDataException or NotSupportedException)
        {
            return null;
        }
    }

    // Written beside its place, then moved there: a reader sees no file or a whole one. A folder that cannot be written, or
    // a drive without the room, keeps nothing (the font is converted again on the next read).
    private static void Store(string folder, string path, byte[] bytes)
    {
        string temporary = Path.Combine(folder, $"{Path.GetFileName(path)}.{Environment.ProcessId}.{Guid.NewGuid():N}.tmp");
        try
        {
            Directory.CreateDirectory(folder);
            if (Room(folder) is { } free && free < bytes.Length)
            {
                return;
            }

            File.WriteAllBytes(temporary, bytes);
            File.Move(temporary, path, overwrite: true);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            try
            {
                File.Delete(temporary);                                         // another process holds the same file: its copy stands
            }
            catch (Exception again) when (again is IOException or UnauthorizedAccessException)
            {
                // nothing more to do: a leftover temporary file is never read
            }
        }
    }

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

    // The kept file's name: a SHA-256 of everything the converted font depends on.
    internal static string Key(ReadOnlySpan<byte> file, int index, FontMap map, IFontWriter writer, FontWriteOptions options)
    {
        var text = new StringBuilder("idrak converted font 1\n");
        text.Append(System.Convert.ToHexStringLower(SHA256.HashData(file))).Append('\n');
        text.Append(index.ToString(CultureInfo.InvariantCulture)).Append('\n');
        text.Append(map.Name).Append('\n').Append(map.Digest).Append('\n');
        var mapper = FontMappers.Get(FontMaps.Mapper);
        text.Append(FontMaps.Mapper).Append(' ').Append(Identity(mapper)).Append(' ').Append(FontMappers.Origin(FontMaps.Mapper)).Append('\n');
        text.Append(writer.Name).Append(' ').Append(Identity(writer)).Append(' ').Append(FontWriters.Origin(writer.Name)).Append('\n');
        if (FontWriters.Default(writer.Name) is { } library)
        {
            text.Append(Identity(library)).Append('\n');
        }

        if (FontMappers.Default(FontMaps.Mapper) is { } defaultMapper)
        {
            text.Append(Identity(defaultMapper)).Append('\n');
        }

        text.Append(Identity(map)).Append('\n');
        text.Append(options.CurveTolerance.ToString("R", CultureInfo.InvariantCulture)).Append(' ').Append(options.KeepHinting).Append('\n');
        var drawing = FontMaps.Drawing;
        foreach (var property in typeof(FontMapDrawing).GetProperties(BindingFlags.Public | BindingFlags.Instance).OrderBy(p => p.Name, StringComparer.Ordinal))
        {
            // A page's chances of the hand's features are not the font's: the same converted font serves every chance.
            if (property.Name is nameof(FontMapDrawing.Pairs) or nameof(FontMapDrawing.SecondStyle) or nameof(FontMapDrawing.Words))
            {
                continue;
            }

            object? value = property.GetValue(drawing);
            text.Append(property.Name).Append('=').Append(value is IFormattable formattable ? formattable.ToString(null, CultureInfo.InvariantCulture) : value?.ToString() ?? "auto").Append('\n');
        }

        return System.Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(text.ToString())));
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
