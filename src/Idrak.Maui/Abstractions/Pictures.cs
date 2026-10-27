// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Maui.Abstractions;

/// <summary>
/// Saves a picture to the phone's own Pictures (where the gallery shows it), in an album of the app's. Android's (from
/// Android 10, through the media store, no permission needed) is registered by <c>UseIdrak</c>; where the platform has no
/// way, <see cref="CanSave"/> is false and a page shares the picture instead.
/// </summary>
public interface IPictureSaver
{
    /// <summary>Whether pictures can be saved here.</summary>
    bool CanSave { get; }

    /// <summary>Why they cannot (when <see cref="CanSave"/> is false), or "".</summary>
    string Unavailable { get; }

    /// <summary>Saves <paramref name="bytes"/> (a <paramref name="mediaType"/> file) as <paramref name="fileName"/> in <paramref name="album"/>; where it went, as the gallery names it.</summary>
    /// <exception cref="NotSupportedException">Pictures cannot be saved here (<see cref="Unavailable"/> says why).</exception>
    /// <exception cref="IOException">The phone refused or failed the write.</exception>
    Task<string> SaveAsync(byte[] bytes, string fileName, string mediaType, string album, CancellationToken cancellationToken = default);
}

/// <summary>
/// The folders the platform keeps its own fonts in, for the Generate lab to list (Android's system fonts; Windows' Fonts
/// folder). Android's is registered by <c>UseIdrak</c> (Android's own API, no variable read); elsewhere the platform's
/// where .NET names it, else none. A folder set in the lab's settings goes over it.
/// </summary>
public interface ISystemFonts
{
    /// <summary>The folders that hold the platform's fonts and are there (none where the platform has none).</summary>
    IReadOnlyList<string> Folders { get; }
}
