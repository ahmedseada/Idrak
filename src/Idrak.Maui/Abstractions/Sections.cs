// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

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
/// opening one of them (from All labs, a tile, a link) switches to it. Registered by its <see cref="IKind.Name"/> in
/// <see cref="AppSections"/>; the library's are Home, Chat, Voice, Vision, All labs and Settings, and an app adds its own
/// or replaces one by name.
/// </summary>
public interface IAppSection : IKind
{
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
/// <see cref="Register"/>; one under a library name shadows the library's, which <see cref="Unregister"/> brings back.
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

    private static readonly SlotTable<string, IAppSection> Table = BuiltIn();

    private static SlotTable<string, IAppSection> BuiltIn()
    {
        var table = new SlotTable<string, IAppSection>(nameof(AppSections), comparer: StringComparer.OrdinalIgnoreCase,
            unguarded: "a section is a place the user has open, with no call to fall back at");
        foreach (var section in Idrak.Maui.Navigation.LibrarySections.All())
        {
            table.RegisterDefault(section.Name, section);
        }

        return table;
    }

    /// <summary>Registers <paramref name="section"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IAppSection section)
    {
        ArgumentNullException.ThrowIfNull(section);
        ArgumentException.ThrowIfNullOrWhiteSpace(section.Name);
        Table.Register(section.Name, section, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's section of that name (the library's, if any, answers again); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in registration order.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>Every registered section in order (<see cref="IAppSection.Order"/>, then the title).</summary>
    public static IReadOnlyList<IAppSection> All =>
        [.. Table.Values.OrderBy(s => s.Order).ThenBy(s => s.Title, StringComparer.CurrentCultureIgnoreCase)];

    /// <summary>The section of that name, or null.</summary>
    public static IAppSection? Find(string name) => Table.Find(name);

    /// <summary>The section of that name.</summary>
    /// <exception cref="NotRegisteredException">None is registered under that name (the message names those that are).</exception>
    public static IAppSection Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(AppSections), name, Table.Keys,
            $"Section '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with AppSections.Register.");

    /// <summary>The library's section of that name, or null.</summary>
    public static IAppSection? Default(string name) => Table.Default(name);

    /// <summary>Who registered the section of that name: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>
    /// The sections in order, without those that host features and whose every feature <paramref name="options"/> turn off
    /// or nobody registered.
    /// </summary>
    public static IReadOnlyList<IAppSection> Visible(IdrakMauiOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return [.. All.Where(s => s.Features.Count == 0 || s.Features.Any(f => options.IsFeatureEnabled(f) && MobileFeatures.Find(f) is not null))];
    }

    /// <summary>The visible section hosting the feature <paramref name="feature"/>, or null when none does.</summary>
    public static IAppSection? Hosting(string feature, IdrakMauiOptions options) =>
        Visible(options).FirstOrDefault(s => s.Features.Contains(feature, StringComparer.OrdinalIgnoreCase));
}
