// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Maui.Abstractions;

/// <summary>
/// A clock that ticks once a display frame on the UI thread, for live drawing (the waveform, the level meters): a view
/// subscribes while it draws live and leaves when it stops, so nothing ticks while no view is live. The library registers
/// "display" (the display's own frames: Android's Choreographer, WinUI's CompositionTarget.Rendering on Windows) and
/// "timer" (a dispatcher timer at the display's refresh rate) in <see cref="FrameClocks"/>; an app adds its own or replaces
/// one by name.
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
/// The frame clocks (a <see cref="SlotTable{TKey, TValue}"/>): "display" on Android and Windows, "timer" everywhere. <see cref="Current"/>
/// is the one live views tick on: the one <see cref="IdrakMauiOptions.FrameClock"/> names, else the first that can tick here.
/// </summary>
public static class FrameClocks
{
    /// <summary>The display's own frames (Android's Choreographer, WinUI's CompositionTarget.Rendering).</summary>
    public const string Display = "display";

    /// <summary>A dispatcher timer at the display's refresh rate.</summary>
    public const string Timer = "timer";

    private static readonly SlotTable<string, IFrameClock> Table = BuiltIn();
    private static IFrameClock? _current;

    private static SlotTable<string, IFrameClock> BuiltIn()
    {
        var table = new SlotTable<string, IFrameClock>(nameof(FrameClocks), comparer: StringComparer.OrdinalIgnoreCase,
            unguarded: "a frame clock drives a view's drawing across frames, with no call to fall back at");
#if ANDROID
        table.RegisterDefault(Display, new Idrak.Maui.Platforms.Android.ChoreographerFrameClock());
#elif WINDOWS
        table.RegisterDefault(Display, new Idrak.Maui.Platforms.Windows.CompositionFrameClock());
#endif
        table.RegisterDefault(Timer, new Idrak.Maui.Controls.TimerFrameClock());
        return table;
    }

    /// <summary>Registers <paramref name="clock"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IFrameClock clock)
    {
        ArgumentNullException.ThrowIfNull(clock);
        ArgumentException.ThrowIfNullOrWhiteSpace(clock.Name);
        Table.Register(clock.Name, clock, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's clock of that name (the library's, if any, answers again); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in registration order.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The clock of that name, or null.</summary>
    public static IFrameClock? Find(string name) => Table.Find(name);

    /// <summary>The clock of that name.</summary>
    /// <exception cref="NotRegisteredException">None is registered under that name (the message names those that are).</exception>
    public static IFrameClock Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(FrameClocks), name, Table.Keys,
            $"Frame clock '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with FrameClocks.Register.");

    /// <summary>The first registered clock that can tick here.</summary>
    public static IFrameClock ForThisApp() => Table.Values.FirstOrDefault(c => c.IsAvailable(out _)) ?? Table.Find(Timer)!;

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
    /// <exception cref="NotRegisteredException">None is registered under that name.</exception>
    /// <exception cref="NotSupportedException">It cannot tick here.</exception>
    public static IFrameClock Use(string? name)
    {
        if (string.IsNullOrWhiteSpace(name) || name.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            _current = null;
            return ForThisApp();
        }

        var clock = Get(name);
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
