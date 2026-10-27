// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Android.Content;
using Idrak.Maui.Abstractions;

namespace Idrak.Maui.Platforms.Android;

// Restarting the app on Android: what the app holds is released first (the models unloaded through the model service, the
// speaker stopped, the settings already kept by their store), then the launcher's activity is started in a fresh task
// (Intent.makeRestartActivityTask over the package's launch intent: FLAG_ACTIVITY_NEW_TASK and CLEAR_TASK) and this
// process ends (Runtime.exit). The system starts a new process for the activity it was asked to start, so the library,
// its devices and the models load afresh, with the settings saved.
internal sealed class AndroidAppRestarter(IServiceProvider services) : IAppRestarter
{
    public bool CanRestart => Launch() is not null;

    public string Unavailable => CanRestart ? "" : "the package has no launcher activity to start again";

    public void Restart()
    {
        var context = global::Android.App.Application.Context;
        if (Launch() is not { Component: { } component })
        {
            throw new NotSupportedException($"The app cannot restart itself here: {Unavailable}.");
        }

        Release(services.GetService(typeof(ISoundPlayer)) as IDisposable);
        Release(services.GetService(typeof(IModelService)) as IDisposable);
        var restart = Intent.MakeRestartActivityTask(component);
        context.StartActivity(restart);
        Java.Lang.Runtime.GetRuntime()?.Exit(0);
    }

    private static Intent? Launch()
    {
        var context = global::Android.App.Application.Context;
        return context.PackageName is { } name ? context.PackageManager?.GetLaunchIntentForPackage(name) : null;
    }

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
