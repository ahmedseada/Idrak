// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Maui.Abstractions;

/// <summary>Where a section of the app's navigation shows.</summary>
public enum SectionPlacement
{
    /// <summary>A tab: on the bottom bar of a narrow window and at the top of the rail of a wide one.</summary>
    Main = 0,

    /// <summary>At the foot of the rail of a wide window (All labs, Settings); a narrow window reaches it from Home.</summary>
    Foot = 1,
}

/// <summary>
/// One section of the app's navigation (no Shell): a tab of the bottom bar on a narrow window, an entry of the side rail on
/// a wide one. It shows a view the host keeps while the app runs (made on first use), and names the features it hosts, so
/// opening one of them (from All labs, a tile, a link) switches to it. Registered by its <c>IKind.Name</c> in
/// <see cref="AppSections"/>; the library's are Home, Chat, Voice, Vision, All labs and Settings, and an app adds its own
/// or replaces one by name.
/// </summary>
public interface IAppSection
{
    /// <summary>Its name ("voice"): what the app's settings and links name it by.</summary>
    string Name { get; }

    /// <summary>One line on it, for listings.</summary>
    string Summary { get; }

    /// <summary>Its label ("Voice").</summary>
    string Title { get; }

    /// <summary>The name of its icon among Idrak's (<c>Idrak.Maui.Theme.IdrakIcons</c>: "home", "mic").</summary>
    string Icon { get; }

    /// <summary>Its place: lower first.</summary>
    int Order { get; }

    /// <summary>Whether it is a tab or at the rail's foot.</summary>
    SectionPlacement Placement { get; }

    /// <summary>The features it hosts, by name, in its order (none for a section of its own, such as Home).</summary>
    IReadOnlyList<string> Features { get; }

    /// <summary>Its view, its services from <paramref name="services"/>; the host makes it once and keeps it.</summary>
    View CreateView(IServiceProvider services);
}

/// <summary>
/// What a view the navigation hosts hears about being shown: <see cref="Shown"/> when it comes on screen (its section
/// chosen, or the page above it closed), <see cref="Hidden"/> when it leaves (another section chosen, a page pushed over
/// it), so it starts and stops what only runs while it is seen (a timer, a recording, the carousel). A feature's page
/// calls the same when it appears and disappears.
/// </summary>
public interface IHostedView
{
    /// <summary>It came on screen.</summary>
    void Shown();

    /// <summary>It left the screen.</summary>
    void Hidden();

    /// <summary>Shows the feature <paramref name="feature"/> among its own (a segment of Voice); false when it hosts none of that name.</summary>
    bool ShowFeature(string feature) => false;

    /// <summary>
    /// Closes the sheet it shows over its content (the chat's tune sheet, the Terminal's commands pane in a narrow window),
    /// as Esc does on a desktop; false when none is open.
    /// </summary>
    bool CloseSheet() => false;
}

/// <summary>
/// The app's navigation, for views that open another place: a feature (in its section when one hosts it, else its own
/// page pushed), a section, All labs (a sheet over Home on a narrow window, the section on a wide one) or the settings.
/// </summary>
public interface IAppNavigator
{
    /// <summary>Opens the feature <paramref name="feature"/>: its section (and segment) when one hosts it, else its page.</summary>
    Task OpenFeatureAsync(string feature);

    /// <summary>Shows the section <paramref name="section"/>; false when none of that name shows.</summary>
    Task<bool> OpenSectionAsync(string section);

    /// <summary>Shows All labs.</summary>
    Task OpenLabsAsync();

    /// <summary>Shows the settings.</summary>
    Task OpenSettingsAsync();

    /// <summary>Pushes <paramref name="page"/> over the current place (back returns to it).</summary>
    Task PushAsync(Page page);
}

/// <summary>
/// The sections of the app's navigation, by name (ignoring case): "home", "chat", "voice" (Transcribe, Speak and Studio),
/// "vision" (Read text and Image), "labs" (All labs) and "settings" are the library's. Register another with
/// <c>Register</c>; one under a library name shadows the library's, which <c>Unregister</c> brings back.
/// <see cref="Visible"/> lists them in order without those whose features the settings all turn off.
/// </summary>
public static class AppSections
{
    /// <summary>The landing: the library and the app, this phone measured, the labs to try.</summary>
    public const string Home = "home";

    /// <summary>The chat.</summary>
    public const string Chat = "chat";

    /// <summary>Transcribe, Speak and Studio.</summary>
    public const string Voice = "voice";

    /// <summary>Read text and Image.</summary>
    public const string Vision = "vision";

    /// <summary>Every lab, grouped and searchable.</summary>
    public const string Labs = "labs";

    /// <summary>The settings.</summary>
    public const string Settings = "settings";

    /// <summary>Every registered section in order (<see cref="IAppSection.Order"/>, then the title).</summary>
    public static IReadOnlyList<IAppSection> All =>
        [.. Registered().OrderBy(s => s.Order).ThenBy(s => s.Title, StringComparer.CurrentCultureIgnoreCase)];

    // The sections and features were registered (AppSections.Register, MobileFeatures.Register) and listed or found by
    // name, which is not on the new design yet.
    private static IEnumerable<IAppSection> Registered() => throw new NotSupportedException("The app's registered sections are not on the new design yet.");

    private static bool IsFeatureRegistered(string name) => throw new NotSupportedException($"The feature '{name}' by name is not on the new design yet.");

    /// <summary>
    /// The sections in order, without those that host features and whose every feature <paramref name="options"/> turn off
    /// or nobody registered.
    /// </summary>
    public static IReadOnlyList<IAppSection> Visible(IdrakMauiOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return [.. All.Where(s => s.Features.Count == 0 || s.Features.Any(f => options.IsFeatureEnabled(f) && IsFeatureRegistered(f)))];
    }

    /// <summary>The visible section hosting the feature <paramref name="feature"/>, or null when none does.</summary>
    public static IAppSection? Hosting(string feature, IdrakMauiOptions options) =>
        Visible(options).FirstOrDefault(s => s.Features.Contains(feature, StringComparer.OrdinalIgnoreCase));
}
