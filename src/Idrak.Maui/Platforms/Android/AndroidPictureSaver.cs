// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Android.Content;
using Android.Provider;
using Idrak.Maui.Abstractions;

namespace Idrak.Maui.Platforms.Android;

// A picture into the phone's Pictures through the media store (Android 10, API 29, and later: no storage permission, the
// app's own entries only): a pending entry made under Pictures/ALBUM, its bytes written, then published, so the gallery never
// shows a half-written file. Earlier Android needs a storage permission the app does not ask for, so it says so instead.
internal sealed class AndroidPictureSaver : IPictureSaver
{
    public bool CanSave => OperatingSystem.IsAndroidVersionAtLeast(29);

    public string Unavailable => CanSave ? "" : "Android before 10 lets an app write to Pictures only with a storage permission; share the page instead";

    public async Task<string> SaveAsync(byte[] bytes, string fileName, string mediaType, string album, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(bytes);
        ArgumentException.ThrowIfNullOrWhiteSpace(fileName);
        if (!OperatingSystem.IsAndroidVersionAtLeast(29))
        {
            throw new NotSupportedException($"Pictures cannot be saved here: {Unavailable}.");
        }

        var resolver = global::Android.App.Application.Context.ContentResolver ?? throw new IOException("The phone gave no content resolver.");
        string folder = Path.Combine(global::Android.OS.Environment.DirectoryPictures ?? "Pictures", album);
        using var values = new ContentValues();
        values.Put(MediaStore.IMediaColumns.DisplayName, fileName);
        values.Put(MediaStore.IMediaColumns.MimeType, mediaType);
        values.Put(MediaStore.IMediaColumns.RelativePath, folder);
        values.Put(MediaStore.IMediaColumns.IsPending, 1);
        var collection = MediaStore.Images.Media.ExternalContentUri ?? throw new IOException("The phone has no pictures collection.");
        var entry = resolver.Insert(collection, values) ?? throw new IOException($"The phone did not make an entry for {fileName} in {folder}.");
        try
        {
            await using (var stream = resolver.OpenOutputStream(entry) ?? throw new IOException($"The phone did not open {fileName} for writing."))
            {
                await stream.WriteAsync(bytes, cancellationToken).ConfigureAwait(false);
            }

            values.Clear();
            values.Put(MediaStore.IMediaColumns.IsPending, 0);
            resolver.Update(entry, values, null, null);
        }
        catch
        {
            resolver.Delete(entry, null, null);                                  // no half-written picture left in the gallery
            throw;
        }

        return $"{folder}/{fileName}";
    }
}
