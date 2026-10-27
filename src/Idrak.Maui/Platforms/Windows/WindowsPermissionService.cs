// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Maui.Abstractions;

namespace Idrak.Maui.Platforms.Windows;

// The permissions on Windows: MAUI's (MauiPermissionService), whose microphone on Windows reads the system's privacy
// setting (DeviceAccessInformation for audio capture) and asks by opening the microphone once (MediaCapture), with no
// manifest entry needed for an unpackaged app (a packaged one declares the microphone device capability in its
// Package.appxmanifest); the camera takes no permission there. Windows never asks the user in a dialog for a desktop app:
// a refusal says where in Windows' Settings to allow it, and the settings button opens that page (ms-settings:privacy-microphone).
internal sealed class WindowsPermissionService : IPermissionService
{
    private readonly MauiPermissionService _maui = new();

    public async Task<PermissionOutcome> CheckAsync(PhoneNeeds needs) => Explained(await _maui.CheckAsync(needs).ConfigureAwait(false));

    public async Task<PermissionOutcome> RequestAsync(PhoneNeeds needs) => Explained(await _maui.RequestAsync(needs).ConfigureAwait(false));

    public void OpenSettings()
    {
        try
        {
            using var _ = Process.Start(new ProcessStartInfo("ms-settings:privacy-microphone") { UseShellExecute = true });
        }
        catch (Exception e) when (e is System.ComponentModel.Win32Exception or InvalidOperationException)
        {
            _maui.OpenSettings();                                               // the app's own page in Settings, as MAUI opens it
        }
    }

    private static PermissionOutcome Explained(PermissionOutcome outcome)
    {
        if (outcome.Granted)
        {
            return outcome;
        }

        string where = AppInfo.Current.PackagingModel == AppPackagingModel.Packaged
            ? $"turn on Microphone access, Let apps access your microphone and {AppInfo.Current.Name} in the list"
            : "turn on Microphone access and Let desktop apps access your microphone";
        return outcome with
        {
            Message = $"Windows does not let the app use the microphone. Allow it in Settings > Privacy & security > Microphone ({where}), "
                + "check that a microphone is connected, then try again.",
        };
    }
}
