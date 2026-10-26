// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Maui.Abstractions;

/// <summary>What a feature needs from the phone, asked for when it opens.</summary>
[Flags]
public enum PhoneNeeds
{
    /// <summary>Nothing.</summary>
    None = 0,

    /// <summary>The microphone (RECORD_AUDIO on Android), asked at first use.</summary>
    Microphone = 1,

    /// <summary>The camera (CAMERA on Android), asked at first use.</summary>
    Camera = 2,

    /// <summary>The network (a download; INTERNET on Android, granted at install).</summary>
    Network = 4,

    /// <summary>A model in the model store (pulled once, then used offline).</summary>
    Model = 8,
}

/// <summary>
/// One feature of the app: a page and its view model, listed in the app's menu. Registered by its
/// <see cref="IKind.Name"/> in <see cref="MobileFeatures"/>; the library registers its own as defaults and an app adds its
/// own or replaces one by name. A feature says what it needs from the phone (<see cref="Needs"/>) and from the library
/// (<see cref="Packages"/>, <see cref="ModelFamilies"/>), so the app asks for a permission or a model before it opens.
/// </summary>
public interface IMobileFeature : IKind
{
    /// <summary>The name in the menu and the page's title ("Devices").</summary>
    string Title { get; }

    /// <summary>A character or two drawn in the system font beside the title in the menu.</summary>
    string Glyph { get; }

    /// <summary>Where it stands in the menu: lower first, then by title.</summary>
    int Order { get; }

    /// <summary>What it needs from the phone.</summary>
    PhoneNeeds Needs { get; }

    /// <summary>The library's packages it uses ("Idrak.Gpu"), for listings.</summary>
    IReadOnlyList<string> Packages { get; }

    /// <summary>The model families it needs registered ("qwen3"); empty when none.</summary>
    IReadOnlyList<string> ModelFamilies { get; }

    /// <summary>A new view model for the page, its services taken from <paramref name="services"/>.</summary>
    object CreateViewModel(IServiceProvider services);

    /// <summary>A new page bound to <paramref name="viewModel"/> (one <see cref="CreateViewModel"/> made).</summary>
    Page CreatePage(IServiceProvider services, object viewModel);
}

/// <summary>
/// The app's features, by name (ignoring case). The library's: "devices" (the devices found, the Vulkan driver loaded in
/// the app's own process with its measured width, subgroup sizes and memory budget, CPU features, memory in use). Register
/// another with <see cref="Register"/>; one under a library name shadows the library's, which <see cref="Unregister"/>
/// brings back. <see cref="Enabled"/> lists them in menu order without those the settings turn off.
/// </summary>
public static class MobileFeatures
{
    /// <summary>The library's devices page.</summary>
    public const string Devices = "devices";

    private static readonly SlotTable<string, IMobileFeature> Table = BuiltIn();

    private static SlotTable<string, IMobileFeature> BuiltIn()
    {
        var table = new SlotTable<string, IMobileFeature>(nameof(MobileFeatures), comparer: StringComparer.OrdinalIgnoreCase,
            unguarded: "a feature is a page the user has open, with no call to fall back at");
        table.RegisterDefault(Devices, new Idrak.Maui.Views.DevicesFeature());
        return table;
    }

    /// <summary>Registers <paramref name="feature"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IMobileFeature feature)
    {
        ArgumentNullException.ThrowIfNull(feature);
        ArgumentException.ThrowIfNullOrWhiteSpace(feature.Name);
        Table.Register(feature.Name, feature, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's feature of that name (the library's, if any, answers again); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in registration order.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>Every registered feature in menu order (<see cref="IMobileFeature.Order"/>, then the title).</summary>
    public static IReadOnlyList<IMobileFeature> All =>
        [.. Table.Values.OrderBy(f => f.Order).ThenBy(f => f.Title, StringComparer.CurrentCultureIgnoreCase)];

    /// <summary>The feature of that name, or null.</summary>
    public static IMobileFeature? Find(string name) => Table.Find(name);

    /// <summary>The feature of that name.</summary>
    /// <exception cref="NotRegisteredException">None is registered under that name (the message names those that are).</exception>
    public static IMobileFeature Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(MobileFeatures), name, Table.Keys,
            $"Feature '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with MobileFeatures.Register.");

    /// <summary>The library's feature of that name, or null.</summary>
    public static IMobileFeature? Default(string name) => Table.Default(name);

    /// <summary>Who registered the feature of that name: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>The features in menu order, without those <paramref name="options"/> turn off.</summary>
    public static IReadOnlyList<IMobileFeature> Enabled(IdrakMauiOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return [.. All.Where(f => options.IsFeatureEnabled(f.Name))];
    }

    /// <summary>The page of feature <paramref name="name"/>, bound to a new view model of it.</summary>
    /// <exception cref="NotRegisteredException">No feature has that name.</exception>
    public static Page Open(string name, IServiceProvider services)
    {
        ArgumentNullException.ThrowIfNull(services);
        var feature = Get(name);
        object viewModel = feature.CreateViewModel(services);
        var page = feature.CreatePage(services, viewModel);
        page.BindingContext ??= viewModel;
        page.Title ??= feature.Title;
        return page;
    }
}
