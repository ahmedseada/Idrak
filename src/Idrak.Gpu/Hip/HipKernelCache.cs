// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Security.Cryptography;
using System.Text;

namespace Idrak.Gpu.Hip;

/// <summary>
/// Assembled kernels kept on disk, so Idrak's GCN assembler runs once per device architecture, driver and library
/// build rather than at every start. One code object per key, in the library's cache folder
/// (ComputeResources.CacheFolder: IDRAK_CACHE, or ~/.cache/idrak) under hip/kernels/; the key hashes everything the
/// code depends on: every kernel's GCN text (<see cref="KeyOf"/>: HIP's whole kernel set, one code object), the options
/// (the code object format), the target, the runtime and driver versions, the device's identity and the library's
/// version. No compiler version: Idrak is the compiler, so the library's version stands for it. A file of another key is
/// never read, and a file the runtime refuses (or whose metadata does not read) is deleted and assembled again. IDRAK_HIP_KERNEL_CACHE=0 keeps nothing,
/// IDRAK_HIP_KERNEL_CACHE=folder moves it.
/// </summary>
internal sealed class HipKernelCache(string? folder)
{
    /// <summary>Bumped when the way a key is formed changes.</summary>
    private const int FormatVersion = 1;

    /// <summary>The folder code objects are kept in (null: none kept).</summary>
    public string? Folder { get; } = folder;

    /// <summary>The cache the environment asks for.</summary>
    public static HipKernelCache FromEnvironment() => new(DefaultFolder());

    // Tests pass the environment read (then its IDRAK_CACHE, when given, stands for the cache folder).
    internal static string? DefaultFolder(Func<string, string?>? environment = null)
    {
        string? setting = (environment ?? Environment.GetEnvironmentVariable)("IDRAK_HIP_KERNEL_CACHE");
        if (setting is "0" or "false")
        {
            return null;
        }

        if (!string.IsNullOrEmpty(setting))
        {
            return setting;
        }

        string root = environment?.Invoke("IDRAK_CACHE") is { Length: > 0 } cache ? cache : ComputeResources.CacheFolder;
        return Path.Combine(root, "hip", "kernels");
    }

    /// <summary>The key of a compilation: a hash of every input its code object depends on.</summary>
    public static string Key(string source, IEnumerable<string> options, string architecture, string device, string versions)
    {
        var text = new StringBuilder()
            .Append(FormatVersion).Append('\n')
            .Append(typeof(HipKernelCache).Assembly.GetName().Version).Append('\n')
            .Append(architecture).Append('\n')
            .Append(device).Append('\n')
            .Append(versions).Append('\n')
            .AppendJoin(' ', options).Append('\n')
            .Append(source);
        return Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(text.ToString())))[..32];
    }

    /// <summary>
    /// The key of a kernel set's code object: <see cref="Key"/> over every kernel's name and GCN text in order (hashed as
    /// they are, without one string of them all), with <paramref name="options"/> (the code object format), the
    /// <paramref name="target"/> the device reports, the <paramref name="device"/>'s identity and the runtime and driver
    /// <paramref name="versions"/>.
    /// </summary>
    public static string KeyOf(IReadOnlyList<GcnKernelSource> kernels, IEnumerable<string> options, string target, string device, string versions)
    {
        ArgumentNullException.ThrowIfNull(kernels);
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        hash.AppendData(Encoding.UTF8.GetBytes(new StringBuilder()
            .Append(FormatVersion).Append('\n')
            .Append(typeof(HipKernelCache).Assembly.GetName().Version).Append('\n')
            .Append(target).Append('\n')
            .Append(device).Append('\n')
            .Append(versions).Append('\n')
            .AppendJoin(' ', options).Append('\n')
            .Append(kernels.Count).Append('\n').ToString()));
        foreach (var kernel in kernels)
        {
            hash.AppendData(Encoding.UTF8.GetBytes(kernel.Name + "\n" + kernel.Text.Length.ToString(System.Globalization.CultureInfo.InvariantCulture) + "\n"));
            hash.AppendData(kernel.Text);
        }

        return Convert.ToHexStringLower(hash.GetHashAndReset())[..32];
    }

    /// <summary>The file a key's code object is kept in (null when nothing is kept).</summary>
    public string? PathOf(string key, string architecture)
    {
        if (Folder is null)
        {
            return null;
        }

        var name = new StringBuilder();
        foreach (char c in architecture.Length > 0 ? architecture : "current")
        {
            name.Append(char.IsAsciiLetterOrDigit(c) || c is '-' or '.' ? c : '_');
        }

        return Path.Combine(Folder, $"{name}-{key}.co");
    }

    /// <summary>The kept code object of a key, or null.</summary>
    public byte[]? Load(string key, string architecture)
    {
        try
        {
            return PathOf(key, architecture) is { } path && File.Exists(path) ? File.ReadAllBytes(path) : null;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    /// <summary>Keeps a code object (written to a temporary file, then moved: a reader never sees half a file); errors are ignored.</summary>
    public void Store(string key, string architecture, byte[] code)
    {
        if (PathOf(key, architecture) is not { } path)
        {
            return;
        }

        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            string temporary = $"{path}.{Environment.ProcessId}.tmp";
            File.WriteAllBytes(temporary, code);
            File.Move(temporary, path, overwrite: true);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
        }
    }

    /// <summary>Forgets a key's code object (one the runtime refused).</summary>
    public void Delete(string key, string architecture)
    {
        try
        {
            if (PathOf(key, architecture) is { } path)
            {
                File.Delete(path);
            }
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
        }
    }
}
