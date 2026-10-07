// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Security.Cryptography;
using System.Text;

namespace Idrak.Gpu.Hip;

/// <summary>
/// Compiled kernels kept on disk, so hipRTC runs once per device architecture, driver and library build rather than at
/// every start. One code object per key, in IDRAK_CACHE (or ~/.cache/idrak) under hip/kernels/; the key hashes
/// everything the code depends on: the source, the compiler options (block size and target), the architecture, the
/// runtime, driver and compiler versions, the device's identity and the library's version. A file of another key is
/// never read, and a file the runtime refuses is deleted and compiled again. IDRAK_HIP_KERNEL_CACHE=0 keeps nothing,
/// IDRAK_HIP_KERNEL_CACHE=folder moves it.
/// </summary>
internal sealed class HipKernelCache(string? folder)
{
    /// <summary>Bumped when the way a key is formed changes.</summary>
    private const int FormatVersion = 1;

    /// <summary>The folder code objects are kept in (null: none kept).</summary>
    public string? Folder { get; } = folder;

    /// <summary>The cache the environment asks for.</summary>
    public static HipKernelCache FromEnvironment() => new(DefaultFolder(Environment.GetEnvironmentVariable));

    internal static string? DefaultFolder(Func<string, string?> environment)
    {
        string? setting = environment("IDRAK_HIP_KERNEL_CACHE");
        if (setting is "0" or "false")
        {
            return null;
        }

        if (!string.IsNullOrEmpty(setting))
        {
            return setting;
        }

        string root = environment("IDRAK_CACHE") is { Length: > 0 } cache ? cache
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "idrak");
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
