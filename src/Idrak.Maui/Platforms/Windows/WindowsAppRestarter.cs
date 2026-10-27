// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Maui.Abstractions;

namespace Idrak.Maui.Platforms.Windows;

// Restarting the app on Windows: what the app holds is released first (the models unloaded through the model service, the
// speaker stopped, the settings already kept by their store), then the app is started again and this process leaves. A
// packaged (MSIX) run asks the Windows App SDK to restart it (Microsoft.Windows.AppLifecycle.AppInstance.Restart, which
// starts the package's app again and ends this process); should it refuse, or for an unpackaged run, the app's own .exe
// (Environment.ProcessPath) is started with the same arguments, and the app quits as when its window is closed (so the
// window's place is kept), the process ended should it still be there.
internal sealed class WindowsAppRestarter(IServiceProvider services) : IAppRestarter
{
    public bool CanRestart => Packaged || Executable() is not null;

    public string Unavailable => CanRestart ? "" : "the app's own program file was not found to start again";

    public void Restart()
    {
        string? executable = Executable();
        if (!Packaged && executable is null)
        {
            throw new NotSupportedException($"The app cannot restart itself here: {Unavailable}.");
        }

        Release(services.GetService(typeof(ISoundPlayer)) as IDisposable);
        Release(services.GetService(typeof(IModelService)) as IDisposable);
        string[] arguments = Environment.GetCommandLineArgs()[1..];
        string? refused = null;
        if (Packaged)
        {
            // Returns only when the restart was refused; otherwise this process ends in it.
            var reason = global::Microsoft.Windows.AppLifecycle.AppInstance.Restart(string.Join(' ', arguments.Select(Quote)));
            refused = reason.ToString();
            if (executable is null)
            {
                throw new NotSupportedException($"Windows did not restart the app ({refused}); close it and open it again.");
            }
        }

        var start = new ProcessStartInfo(executable!) { UseShellExecute = false, WorkingDirectory = Environment.CurrentDirectory };
        foreach (string argument in arguments)
        {
            start.ArgumentList.Add(argument);
        }

        try
        {
            using var started = Process.Start(start);
        }
        catch (Exception e) when (e is System.ComponentModel.Win32Exception or InvalidOperationException)
        {
            throw new NotSupportedException($"The app could not start itself again ({executable}: {e.Message}{(refused is null ? "" : $"; the package's restart: {refused}")}); close it and open it again.", e);
        }

        if (Application.Current is { } app)
        {
            app.Quit();
        }
        else
        {
            Environment.Exit(0);
        }
    }

    private static bool Packaged => AppInfo.Current.PackagingModel == AppPackagingModel.Packaged;

    private static string? Executable() => Environment.ProcessPath is { Length: > 0 } path && File.Exists(path) ? path : null;

    // A command-line argument as Windows reads it back: quoted when it has a space or a quote, its quotes escaped.
    private static string Quote(string argument) =>
        argument.Length > 0 && argument.AsSpan().IndexOfAny(" \t\"") < 0 ? argument : "\"" + argument.Replace("\"", "\\\"", StringComparison.Ordinal) + "\"";

    private static void Release(IDisposable? held)
    {
        try
        {
            held?.Dispose();
        }
        catch (Exception e) when (e is InvalidOperationException or ObjectDisposedException or IOException)
        {
            // What could not be released goes with the process.
        }
    }
}
