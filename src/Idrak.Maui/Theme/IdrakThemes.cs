// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Maui.Theme;

/// <summary>
/// Applies a theme to the app: the colours of light or dark (<see cref="IdrakLightColors"/>, <see cref="IdrakDarkColors"/>)
/// merged into the app's resources, the one in effect swapped when the choice or (for <see cref="IdrakThemeChoice.System"/>)
/// the phone's theme changes. Pages read the colours through DynamicResource, so they follow at once. The app's own keys
/// win over these (its resources are looked up before the merged dictionaries).
/// </summary>
public static class IdrakThemes
{
    private static readonly ConditionalWeakTable<Application, State> States = new();

    /// <summary>The app theme <paramref name="choice"/> shows when the phone's is <paramref name="system"/>.</summary>
    public static AppTheme Effective(IdrakThemeChoice choice, AppTheme system) => choice switch
    {
        IdrakThemeChoice.Light => AppTheme.Light,
        IdrakThemeChoice.Dark => AppTheme.Dark,
        _ => system == AppTheme.Dark ? AppTheme.Dark : AppTheme.Light,
    };

    /// <summary>The colours of <paramref name="theme"/>.</summary>
    public static ResourceDictionary Colors(AppTheme theme) => theme == AppTheme.Dark ? new IdrakDarkColors() : new IdrakLightColors();

    /// <summary>Shows <paramref name="choice"/> on <paramref name="app"/> (again whenever it is called, as the settings page does).</summary>
    public static void Apply(Application app, IdrakThemeChoice choice)
    {
        ArgumentNullException.ThrowIfNull(app);
        var state = States.GetValue(app, a =>
        {
            var s = new State();
            a.RequestedThemeChanged += (_, _) => Swap(a, s);
            return s;
        });
        state.Choice = choice;
        app.UserAppTheme = choice switch { IdrakThemeChoice.Light => AppTheme.Light, IdrakThemeChoice.Dark => AppTheme.Dark, _ => AppTheme.Unspecified };
        Swap(app, state);
    }

    private static void Swap(Application app, State state)
    {
        var theme = Effective(state.Choice, app.PlatformAppTheme);
        if (state.Merged is not null && state.Theme == theme)
        {
            return;
        }

        var colors = Colors(theme);
        var merged = app.Resources.MergedDictionaries;
        if (state.Merged is not null)
        {
            merged.Remove(state.Merged);
        }

        merged.Add(colors);
        state.Merged = colors;
        state.Theme = theme;
    }

    private sealed class State
    {
        public IdrakThemeChoice Choice { get; set; }

        public ResourceDictionary? Merged { get; set; }

        public AppTheme Theme { get; set; }
    }
}
