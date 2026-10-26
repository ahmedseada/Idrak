// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Imaging;
using Idrak.Maui.Abstractions;
using Microsoft.Maui.Media;

namespace Idrak.Maui;

/// <summary>A photo the Image or OCR lab opened: its file as it came, its pixels decoded once by the library, and what that took.</summary>
/// <param name="Name">Its name as shown (the file's).</param>
/// <param name="File">Its encoded bytes, as the camera or the gallery gave them.</param>
/// <param name="Pixels">Its pixels: decoded by the library's codecs and turned upright by its EXIF orientation (8-bit planes, as the codecs give them).</param>
/// <param name="Format">The codec that read it ("jpeg", "png").</param>
/// <param name="Orientation">Its EXIF orientation (1: none, or upright already).</param>
/// <param name="DecodeTime">How long reading it took (decoding and turning it upright).</param>
public sealed record Photo(string Name, ChatImage File, ImageData Pixels, string Format, int Orientation, TimeSpan DecodeTime)
{
    /// <summary>"photo.jpg: jpeg, 4,032 x 3,024, colour, turned upright (EXIF 6), read in 120 ms".</summary>
    public string Describe() => string.Create(System.Globalization.CultureInfo.InvariantCulture,
        $"{Name}: {Format}, {Pixels.Width:N0} x {Pixels.Height:N0}, {(Pixels.Channels == 1 ? "grey" : "colour")}, {ProgressText.Bytes(File.Data.Length)}")
        + (Orientation > 1 ? $", turned upright (EXIF {Orientation})" : "")
        + $", read in {Photos.Milliseconds(DecodeTime)}";
}

/// <summary>
/// Photos for the labs: taken with the camera (the permission asked first) or picked from the gallery through MAUI's
/// media picker, their bytes kept as they came (no resizing, rotating or re-encoding by the platform: the library decodes
/// them, rule 70), decoded once by the library's own codecs (<see cref="ImageCodecs"/>) and turned upright by their EXIF
/// orientation (<see cref="ChatImageDecoder"/>); and the pictures the pages show, shrunk for the screen and encoded once
/// with the library's PNG encoder.
/// </summary>
public static class Photos
{
    // The HEIF family's brands (HEIC, HEIF, AVIF), which no codec of the library reads.
    private static readonly string[] HeifBrands = ["heic", "heix", "hevc", "hevx", "heim", "heis", "hevm", "hevs", "mif1", "msf1", "avif", "avis"];

    /// <summary>
    /// A photo from the camera (<paramref name="camera"/>: the permission asked first) or the gallery, as its bytes; null
    /// when none was taken or picked. A refusal or a failure is <see cref="PhotoRefusedException"/>, with what to tell the user.
    /// </summary>
    public static async Task<(ChatImage File, string Name)?> TakeAsync(IMediaPicker picker, IPermissionService permissions, bool camera, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(picker);
        ArgumentNullException.ThrowIfNull(permissions);
        FileResult? result;
        try
        {
            if (camera)
            {
                if (!picker.IsCaptureSupported)
                {
                    throw new PhotoRefusedException("This phone has no camera the app can take a photo with; pick one from the gallery instead.", permission: false);
                }

                var allowed = await permissions.RequestAsync(PhoneNeeds.Camera).ConfigureAwait(true);
                if (!allowed.Granted)
                {
                    throw new PhotoRefusedException(allowed.Message.Length > 0 ? allowed.Message
                        : "The app may not use the camera. Allow it in the app's settings (Permissions), then try again.", permission: true);
                }

                result = await picker.CapturePhotoAsync(new MediaPickerOptions { Title = "A photo to read" }).ConfigureAwait(true);
            }
            else
            {
                var picked = await picker.PickPhotosAsync(new MediaPickerOptions { Title = "A photo to read", SelectionLimit = 1 }).ConfigureAwait(true);
                result = picked is { Count: > 0 } ? picked[0] : null;
            }
        }
        catch (PhotoRefusedException)
        {
            throw;
        }
        catch (PermissionException e)
        {
            throw new PhotoRefusedException($"The app may not use the {(camera ? "camera" : "gallery")}: {e.Message}. Allow it in the app's settings (Permissions), then try again.", permission: true);
        }
        catch (Exception e) when (e is not OutOfMemoryException and not OperationCanceledException)
        {
            throw new PhotoRefusedException($"No photo could be {(camera ? "taken" : "picked")}: {e.Message}", permission: false);
        }

        if (result is null)
        {
            return null;
        }

        await using var stream = await result.OpenReadAsync().ConfigureAwait(true);
        return (await ReadAsync(stream, cancellationToken).ConfigureAwait(true), result.FileName);
    }

    /// <summary>A file's bytes as an image: read once into the image's own array where the length is known, else gathered once and copied.</summary>
    public static async Task<ChatImage> ReadAsync(Stream stream, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (stream.CanSeek && stream.Length - stream.Position is > 0 and var left && left <= Array.MaxLength)
        {
            int length = (int)(stream.Length - stream.Position);
            return await Task.Run(() => ChatImage.FromStream(stream, length), cancellationToken).ConfigureAwait(true);
        }

        using var gathered = new MemoryStream();
        await stream.CopyToAsync(gathered, cancellationToken).ConfigureAwait(true);
        if (gathered.Length == 0)
        {
            throw new InvalidDataException("The photo's file is empty.");
        }

        return ChatImage.FromBytes(gathered.GetBuffer().AsSpan(0, (int)gathered.Length));
    }

    /// <summary>
    /// <paramref name="file"/> decoded once by the library's codecs and turned upright by its EXIF orientation. A format no
    /// codec reads is refused naming the formats it reads; HEIC (and HEIF, AVIF), which phones often save, is named as such.
    /// </summary>
    /// <exception cref="InvalidDataException">No registered codec reads it (the message names the formats the library reads), or it is damaged.</exception>
    /// <exception cref="NotSupportedException">A variant of a format the codec does not read (a 12-bit JPEG, say).</exception>
    public static Photo Decode(ChatImage file, string name)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(name);
        var bytes = file.Data.Span;
        string? format = ChatImageDecoder.FormatOf(file);
        if (format is null)
        {
            throw new InvalidDataException(HeifBrand(bytes) is { } brand
                ? $"{name} is a HEIC/HEIF photo ('{brand}'), which the library's codecs do not read; they read {Readable()}. "
                  + "Set the camera to save JPEG pictures (its HEIF or high-efficiency option off), or pick a JPEG or PNG."
                : $"{name} is not an image the library's codecs read; they read {Readable()}.");
        }

        long start = Stopwatch.GetTimestamp();
        int orientation = ChatImageDecoder.Orientation(bytes);
        var pixels = ChatImageDecoder.Decode(bytes);
        return new Photo(name, file, pixels, format, orientation, Stopwatch.GetElapsedTime(start));
    }

    /// <summary>The formats the library's codecs read, with their extensions: "png (.png), jpeg (.jpg, .jpeg), …".</summary>
    public static string Readable() => string.Join(", ", ImageCodecs.Names.Select(n => $"{n} ({string.Join(", ", ImageCodecs.Get(n).Extensions)})"));

    /// <summary>The brand of a HEIF-family file (an ISO media "ftyp" box naming heic, heif, avif and the like), or null.</summary>
    public static string? HeifBrand(ReadOnlySpan<byte> file)
    {
        if (file.Length < 12 || !file.Slice(4, 4).SequenceEqual("ftyp"u8))
        {
            return null;
        }

        string brand = System.Text.Encoding.ASCII.GetString(file.Slice(8, 4));
        return HeifBrands.Contains(brand, StringComparer.Ordinal) ? brand : null;
    }

    /// <summary>
    /// The longest side of the phone's screen in pixels, as MAUI measures it now; null where it cannot (no screen, or no
    /// platform as in the tests): the pictures are then shown at their own size.
    /// </summary>
    public static int? ScreenPixels()
    {
        try
        {
            var display = DeviceDisplay.Current.MainDisplayInfo;
            int longest = (int)Math.Round(Math.Max(display.Width, display.Height));
            return longest > 0 ? longest : null;
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            return null;
        }
    }

    /// <summary>
    /// <paramref name="image"/> as the page shows it, encoded once with the library's PNG encoder: shrunk (only when larger)
    /// so its longest side is <paramref name="pixels"/> (the screen's when null; its own size when that cannot be measured),
    /// with <paramref name="resampling"/> (the library's when null). The image itself is not changed.
    /// </summary>
    public static PhotoPreview Preview(ImageData image, int? pixels, ImageResampling? resampling = null)
    {
        ArgumentNullException.ThrowIfNull(image);
        long start = Stopwatch.GetTimestamp();
        int? side = pixels ?? ScreenPixels();
        var shown = side is { } s && Math.Max(image.Width, image.Height) > s
            ? resampling is { } filter ? PillowImageOps.ScaleDown(image, s, s, filter) : PillowImageOps.ScaleDown(image, s, s)
            : image;
        byte[] png = ImageEncoders.Get(ImageSettings.Png).Encode(shown);
        return new PhotoPreview(png, shown.Width, shown.Height, Stopwatch.GetElapsedTime(start));
    }

    /// <summary>A time in milliseconds as the labs show it ("12.3 ms", "1,234 ms").</summary>
    public static string Milliseconds(TimeSpan time) => time.TotalMilliseconds < 100
        ? string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{time.TotalMilliseconds:F1} ms")
        : string.Create(System.Globalization.CultureInfo.InvariantCulture, $"{time.TotalMilliseconds:N0} ms");
}

/// <summary>A picture as a page shows it: PNG bytes made once (the image control decodes them once, not per frame) and its size.</summary>
/// <param name="Png">The PNG file.</param>
/// <param name="Width">Its width in pixels.</param>
/// <param name="Height">Its height in pixels.</param>
/// <param name="Time">How long shrinking and encoding it took.</param>
public sealed record PhotoPreview(byte[] Png, int Width, int Height, TimeSpan Time)
{
    /// <summary>The picture for an image control: a source over the PNG bytes (each stream it opens reads the same array).</summary>
    public ImageSource Source() => ImageSource.FromStream(() => new MemoryStream(Png, writable: false));
}

/// <summary>A photo that could not be taken or picked: what to tell the user, and whether a permission was refused.</summary>
public sealed class PhotoRefusedException : Exception
{
    /// <summary>The refusal, with <paramref name="message"/> to show.</summary>
    public PhotoRefusedException(string message, bool permission)
        : base(message)
    {
        PermissionRefused = permission;
    }

    /// <summary>Whether a permission was refused (the page offers the app's system settings).</summary>
    public bool PermissionRefused { get; }
}
