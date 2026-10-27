// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;

namespace Idrak.Maui;

/// <summary>The phone's permissions through MAUI's Permissions (<see cref="IPermissionService"/>), asked on the main thread.</summary>
public sealed class MauiPermissionService : IPermissionService
{
    /// <inheritdoc />
    public Task<PermissionOutcome> CheckAsync(PhoneNeeds needs) => RunAsync(needs, request: false);

    /// <inheritdoc />
    public Task<PermissionOutcome> RequestAsync(PhoneNeeds needs) => RunAsync(needs, request: true);

    /// <inheritdoc />
    public void OpenSettings() => AppInfo.Current.ShowSettingsUI();

    private static Task<PermissionOutcome> RunAsync(PhoneNeeds needs, bool request) => MainThread.InvokeOnMainThreadAsync(async () =>
    {
        var refused = new List<string>();
        if ((needs & PhoneNeeds.Microphone) != 0 && await Status<Permissions.Microphone>(request) != PermissionStatus.Granted)
        {
            refused.Add("the microphone");
        }

        if ((needs & PhoneNeeds.Camera) != 0 && await Status<Permissions.Camera>(request) != PermissionStatus.Granted)
        {
            refused.Add("the camera");
        }

        return refused.Count == 0 ? new PermissionOutcome(needs, true, "")
            : new PermissionOutcome(needs, false, $"The app may not use {string.Join(" or ", refused)}. Allow it in the app's settings (Permissions), then try again.");
    });

    private static async Task<PermissionStatus> Status<TPermission>(bool request) where TPermission : Permissions.BasePermission, new()
    {
        var status = await Permissions.CheckStatusAsync<TPermission>();
        return status == PermissionStatus.Granted || !request ? status : await Permissions.RequestAsync<TPermission>();
    }
}

/// <summary>Sharing a file through MAUI's Share (<see cref="IShareService"/>).</summary>
public sealed class MauiShareService : IShareService
{
    /// <inheritdoc />
    public Task ShareFileAsync(string path, string? title = null, string? contentType = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"There is no file {path} to share.", path);
        }

        var file = contentType is null ? new ShareFile(path) : new ShareFile(path, contentType);
        return Share.Default.RequestAsync(new ShareFileRequest { Title = title ?? Path.GetFileName(path), File = file });
    }
}

/// <summary>Where the platform has no way to start the app again: the settings page says so instead of offering it.</summary>
public sealed class UnavailableAppRestarter : IAppRestarter
{
    /// <inheritdoc />
    public bool CanRestart => false;

    /// <inheritdoc />
    public string Unavailable => "this platform has no way for the app to start itself again; close it and open it again";

    /// <inheritdoc />
    public void Restart() => throw new NotSupportedException($"The app cannot restart itself here: {Unavailable}.");
}
