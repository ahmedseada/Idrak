// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Data.Abstractions;

/// <summary>
/// Writes one image format; register it with <see cref="ImageEncoders.Register"/>. The reading side of a format is its
/// <see cref="IImageCodec"/> in <see cref="ImageCodecs"/>.
/// </summary>
public interface IImageEncoder
{
    /// <summary>The format's name, e.g. "png" (names ignore case).</summary>
    string Name { get; }

    /// <summary>The file extensions it writes, with the dot, e.g. ".png"; <see cref="ImageEncoders.Save"/> picks an encoder by them.</summary>
    IReadOnlyCollection<string> Extensions { get; }

    /// <summary>
    /// The file's bytes for <paramref name="image"/>: values in [0, 1] become the format's 8-bit levels (rounded), grey
    /// (1 channel) or colour (3 channels).
    /// </summary>
    byte[] Encode(ImageData image);
}

/// <summary>
/// The image formats files are written in, by name (ignoring case): "png" (8-bit grey or RGB, deflated with the zlib in
/// .NET) and "netpbm" (binary PGM for grey, PPM for colour). A segmentation mask, for example, is an 8-bit grey image whose
/// level is the class (<c>idrak predict --out</c> writes them). Register another format, or replace these, with
/// <see cref="Register"/>; an app's encoder over a library one falls back on it or is compared with it by its policy.
/// </summary>
public static class ImageEncoders
{
    private static readonly SlotTable<string, IImageEncoder> Registry = BuiltIn();

    private static SlotTable<string, IImageEncoder> BuiltIn()
    {
        var table = new SlotTable<string, IImageEncoder>(nameof(ImageEncoders), (slot, app, library) => new GuardedEncoder(slot, app, library), StringComparer.OrdinalIgnoreCase);
        table.RegisterDefault("png", new PngEncoder());
        table.RegisterDefault("netpbm", new NetpbmEncoder());
        return table;
    }

    /// <summary>Registers <paramref name="encoder"/> under its name; under a library name it takes that encoder's place, which stays behind it (see <see cref="SetPolicy"/>).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IImageEncoder encoder)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentException.ThrowIfNullOrWhiteSpace(encoder.Name);
        Registry.Register(encoder.Name, encoder, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's encoder <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered encoder names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The extensions the registered encoders write (with the dot, lower case).</summary>
    public static IReadOnlyCollection<string> Extensions => [.. Registry.Values.SelectMany(e => e.Extensions).Select(e => e.ToLowerInvariant()).Distinct()];

    /// <summary>The encoder registered as <paramref name="name"/> (any case).</summary>
    /// <exception cref="NotSupportedException">None is; the message names those that are.</exception>
    public static IImageEncoder Get(string name) => Registry.Find(name)
        ?? throw new NotSupportedException($"No image encoder '{name}' is registered ({string.Join(", ", Names)}); add it with ImageEncoders.Register.");

    /// <summary>The encoder that writes files named like <paramref name="path"/> (by its extension).</summary>
    /// <exception cref="NotSupportedException">No registered encoder lists the extension; the message names the extensions there are.</exception>
    public static IImageEncoder For(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        string extension = Path.GetExtension(path);
        return Registry.Values.FirstOrDefault(e => extension.Length > 0 && e.Extensions.Contains(extension, StringComparer.OrdinalIgnoreCase))
            ?? throw new NotSupportedException($"No image encoder writes '{extension}' files ({string.Join(", ", Extensions)}); add one with ImageEncoders.Register.");
    }

    /// <summary>The library's encoder <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IImageEncoder? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the encoder <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>
    /// What happens when the app's encoder <paramref name="name"/> fails (<see cref="SlotPolicy.Throw"/> unless set;
    /// <see cref="SlotPolicy.FallBack"/> retries on the library's), or whether it only runs beside the library's
    /// (<see cref="SlotPolicy.Shadow"/>: the library's bytes are written, the two are compared).
    /// </summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Registry.SetPolicy(name, policy, shadowRate);

    /// <summary>Writes <paramref name="image"/> to <paramref name="path"/> with the encoder of its extension (the folder is created).</summary>
    /// <exception cref="NotSupportedException">No registered encoder writes the extension.</exception>
    public static void Save(string path, ImageData image)
    {
        ArgumentNullException.ThrowIfNull(image);
        var bytes = For(path).Encode(image);
        string? folder = Path.GetDirectoryName(Path.GetFullPath(path));
        if (folder is not null)
        {
            Directory.CreateDirectory(folder);
        }

        File.WriteAllBytes(path, bytes);
    }

    // An app's encoder over a library one: each call falls back on its own, or is compared with the library's.
    private sealed class GuardedEncoder(Slot slot, IImageEncoder app, IImageEncoder library) : IImageEncoder
    {
        public string Name => app.Name;

        public IReadOnlyCollection<string> Extensions => app.Extensions;

        public byte[] Encode(ImageData image) => slot.Call(() => app.Encode(image), () => library.Encode(image), (a, b) => a.AsSpan().SequenceEqual(b) ? null : $"{a.Length} and {b.Length} bytes differ");
    }
}
