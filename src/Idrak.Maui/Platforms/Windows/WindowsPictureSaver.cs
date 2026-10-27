// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;

namespace Idrak.Maui.Platforms.Windows;

// A picture into the user's Pictures folder (the known folder, Environment.SpecialFolder.MyPictures), in the album as its
// subfolder (made when missing). The bytes go to a part file in that folder first and are then moved to the name,
// so Photos and Explorer never show a half-written file; a name already there gets " (2)", " (3)" and so on, as the
// phone's media store does, and nothing is overwritten. No permission is needed (an unpackaged app is the user's own
// program; a packaged one runs with full trust).
internal sealed class WindowsPictureSaver : IPictureSaver
{
    public bool CanSave => Pictures() is not null;

    public string Unavailable => CanSave ? "" : "Windows names no Pictures folder for this user; share the page instead";

    public async Task<string> SaveAsync(byte[] bytes, string fileName, string mediaType, string album, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(bytes);
        ArgumentException.ThrowIfNullOrWhiteSpace(fileName);
        string pictures = Pictures() ?? throw new NotSupportedException($"Pictures cannot be saved here: {Unavailable}.");
        string name = Path.GetFileName(fileName);
        if (name.Length == 0 || name.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
        {
            throw new ArgumentException($"'{fileName}' is not a file name Windows takes.", nameof(fileName));
        }

        string folder = Album(pictures, album);
        Directory.CreateDirectory(folder);
        string part = Path.Combine(folder, $".{name}.{Guid.NewGuid():N}.part");
        try
        {
            await using (var stream = new FileStream(part, FileMode.CreateNew, FileAccess.Write, FileShare.None, bufferSize: 1, FileOptions.Asynchronous))
            {
                await stream.WriteAsync(bytes, cancellationToken).ConfigureAwait(false);
            }

            string stem = Path.GetFileNameWithoutExtension(name), extension = Path.GetExtension(name);
            for (int copy = 1; ; copy++)
            {
                string target = Path.Combine(folder, copy == 1 ? name : $"{stem} ({copy}){extension}");
                if (File.Exists(target))
                {
                    continue;
                }

                try
                {
                    File.Move(part, target, overwrite: false);
                    return target;
                }
                catch (IOException) when (File.Exists(target))
                {
                    // Taken meanwhile by another save: the next number.
                }
            }
        }
        finally
        {
            if (File.Exists(part))
            {
                File.Delete(part);                                              // no half-written picture left behind
            }
        }
    }

    private static string? Pictures()
    {
        string folder = Environment.GetFolderPath(Environment.SpecialFolder.MyPictures, Environment.SpecialFolderOption.DoNotVerify);
        return folder.Length > 0 ? folder : null;
    }

    // The album as a folder under Pictures (one or more levels, as written); never a path that leaves Pictures.
    private static string Album(string pictures, string album)
    {
        if (string.IsNullOrWhiteSpace(album))
        {
            return pictures;
        }

        string folder = Path.GetFullPath(Path.Combine(pictures, album.Trim()));
        string root = Path.TrimEndingDirectorySeparator(Path.GetFullPath(pictures)) + Path.DirectorySeparatorChar;
        return Path.IsPathRooted(album.Trim()) || !folder.StartsWith(root, StringComparison.OrdinalIgnoreCase)
            ? throw new ArgumentException($"The album '{album}' is not a folder inside Pictures.", nameof(album))
            : folder;
    }
}
