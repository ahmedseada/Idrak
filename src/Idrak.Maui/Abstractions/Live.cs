// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Maui.Abstractions;

/// <summary>
/// A clock that ticks once a display frame on the UI thread, for live drawing (the waveform, the level meters): a view
/// subscribes while it draws live and leaves when it stops, so nothing ticks while no view is live. The library's are
/// "display" (the display's own frames: Android's Choreographer, WinUI's CompositionTarget.Rendering on Windows) and
/// "timer" (a dispatcher timer at the display's refresh rate), named in <see cref="FrameClocks"/>.
/// </summary>
public interface IFrameClock
{
    /// <summary>Its name in <see cref="FrameClocks"/> ("display").</summary>
    string Name { get; }

    /// <summary>One line on how it ticks.</summary>
    string Summary { get; }

    /// <summary>Whether it can tick here (false, with <paramref name="reason"/>, where its platform is not this one).</summary>
    bool IsAvailable(out string reason);

    /// <summary>
    /// Calls <paramref name="frame"/> on the UI thread once a frame, at most <paramref name="framesPerSecond"/> a second (0:
    /// every frame the display shows), until the result is disposed.
    /// </summary>
    IDisposable Subscribe(Action frame, double framesPerSecond = 0);
}

/// <summary>
/// The frame clocks' names: "display" on Android and Windows, "timer" everywhere. <see cref="Current"/> is the one live
/// views tick on: the one chosen (<see cref="Use"/>), else the library's timer clock.
/// </summary>
public static class FrameClocks
{
    /// <summary>The display's own frames (Android's Choreographer, WinUI's CompositionTarget.Rendering).</summary>
    public const string Display = "display";

    /// <summary>A dispatcher timer at the display's refresh rate.</summary>
    public const string Timer = "timer";

    // The library's timer clock, reached directly (picking among display clocks is not on the new design yet).
    private static readonly IFrameClock TimerClock = new Idrak.Maui.Controls.TimerFrameClock();
    private static IFrameClock? _current;

    /// <summary>The clock live views tick on unless one is chosen: the library's timer clock.</summary>
    public static IFrameClock ForThisApp() => TimerClock;

    // Picking the clocks by name is not on the new design yet.
    private static IFrameClock ClockNamed(string name) =>
        throw new NotSupportedException($"The frame clock '{name}' by name is not on the new design yet.");

    /// <summary>
    /// The clock live views tick on: the one <see cref="Use"/> chose (<c>UseIdrak</c> chooses the one the options name),
    /// else <see cref="ForThisApp"/>.
    /// </summary>
    public static IFrameClock Current => _current ?? ForThisApp();

    /// <summary>The frames a second live views draw at most (0: every frame the clock gives); <c>UseIdrak</c> sets it from the options.</summary>
    public static double FramesPerSecond { get; set; }

    /// <summary>
    /// Makes the clock <paramref name="name"/> the one live views tick on ("auto", empty or null: the first that can tick
    /// here); a clock that cannot tick here is refused with its reason.
    /// </summary>
    /// <exception cref="NotSupportedException">It cannot tick here, or a clock is named: picking one by name is not on the new design yet.</exception>
    public static IFrameClock Use(string? name)
    {
        if (string.IsNullOrWhiteSpace(name) || name.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            _current = null;
            return ForThisApp();
        }

        var clock = ClockNamed(name);
        if (!clock.IsAvailable(out string reason))
        {
            throw new NotSupportedException($"The {clock.Name} frame clock cannot tick here: {reason}.");
        }

        _current = clock;
        return clock;
    }
}

/// <summary>
/// Restarts the app: what it holds released (the models unloaded, the sound stopped), its launcher activity started again
/// and this process ended, so the library, its devices and the models load afresh with the saved settings. Android's is
/// registered by <c>UseIdrak</c>, and Windows' (the app's own program started again, or the package's restart); where the
/// platform has no way to start the app again, <see cref="CanRestart"/> is false.
/// </summary>
public interface IAppRestarter
{
    /// <summary>Whether the app can restart itself here.</summary>
    bool CanRestart { get; }

    /// <summary>Why it cannot (when <see cref="CanRestart"/> is false), or "".</summary>
    string Unavailable { get; }

    /// <summary>Releases what the app holds, starts the app again and ends this process; on a platform that cannot, throws.</summary>
    /// <exception cref="NotSupportedException">The app cannot restart itself here (<see cref="Unavailable"/> says why).</exception>
    void Restart();
}
