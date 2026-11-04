// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;

namespace Idrak.Abstraction.Devices;

/// <summary>
/// What every backend's measured kernel choices follow (the CPU's, CUDA's, HIP's, Vulkan's and a plug-in backend's): one
/// meaning of "off" for the tuning settings (<c>IDRAK_AUTOTUNE</c>, <c>IDRAK_TUNING_CACHE</c> and the backends' own cache
/// variables), and a lock across processes around a cache file's read, merge and write, so two processes measuring at
/// once both keep their choices.
/// </summary>
public static class TuningFiles
{
    /// <summary>How long <see cref="Lock"/> waits for another process's merge before giving up (10 seconds unless given).</summary>
    public static readonly TimeSpan DefaultLockTimeout = TimeSpan.FromSeconds(10);

    /// <summary>
    /// Whether a setting turns its feature off: 0, false, off or no (any case, spaces around ignored). Unset or anything
    /// else leaves it as the setting's own reading says (on, or a path).
    /// </summary>
    public static bool IsOff(string? setting) => setting?.Trim() is { Length: > 0 } value
        && (value == "0" || value.Equals("false", StringComparison.OrdinalIgnoreCase) || value.Equals("off", StringComparison.OrdinalIgnoreCase)
            || value.Equals("no", StringComparison.OrdinalIgnoreCase));

    /// <summary>
    /// Holds <paramref name="file"/>'s lock (a <c>.lock</c> file beside it, opened for this process alone) until disposed:
    /// read the file, merge, write it and move it into place while holding it, and another process's merge waits rather
    /// than overwriting the entries this one added. Waits up to <paramref name="timeout"/> (<see cref="DefaultLockTimeout"/>).
    /// </summary>
    /// <exception cref="IOException">The lock stayed held past the timeout (the caller treats it as a cache it could not write).</exception>
    /// <exception cref="UnauthorizedAccessException">The folder cannot be written.</exception>
    public static IDisposable Lock(string file, TimeSpan? timeout = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(file);
        string path = Path.GetFullPath(file) + ".lock";
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var limit = timeout ?? DefaultLockTimeout;
        long start = Stopwatch.GetTimestamp();
        while (true)
        {
            try
            {
                return new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
            }
            catch (IOException e)
            {
                if (Stopwatch.GetElapsedTime(start) >= limit)
                {
                    throw new IOException(string.Create(System.Globalization.CultureInfo.InvariantCulture,
                        $"{file}: another process held its lock ({path}) for {limit.TotalSeconds:0.#} s."), e);
                }

                Thread.Sleep(10);                                                // another process (or thread) merges now
            }
        }
    }
}
