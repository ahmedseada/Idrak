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

/// <summary>Where a feature stands: built, or placed in the menu ahead of the phase that builds it.</summary>
public enum FeatureState
{
    /// <summary>Built: its page does what <see cref="IKind.Summary"/> says.</summary>
    Ready = 0,

    /// <summary>
    /// Not built yet: it keeps its name, title and place in the menu, and its page (the library's placeholder) says what
    /// it will do (<see cref="IMobileFeature.Coming"/>) and need. Its phase replaces the registration's page and view model
    /// under the same name, so the menu does not change.
    /// </summary>
    Planned = 1,
}

/// <summary>Where a feature is listed among the labs (the All labs sheet or page): by what it works on.</summary>
public enum FeatureGroup
{
    /// <summary>The phone itself and the app: models, devices, tests, settings (the default).</summary>
    Device = 0,

    /// <summary>Language: chat, retrieval.</summary>
    Language = 1,

    /// <summary>Voice: speech into text, text into speech, sound processing.</summary>
    Voice = 2,

    /// <summary>Vision: reading a photo, image transforms.</summary>
    Vision = 3,
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

    /// <summary>
    /// Where it stands in the menu: lower first, then by title. A feature that sets none (<see cref="MobileFeatures.Unordered"/>)
    /// takes the place of the library's feature of its name when there is one (so a feature registered over a planned one
    /// keeps its place), and otherwise comes after every ordered one.
    /// </summary>
    int Order => MobileFeatures.Unordered;

    /// <summary>Whether it is built (<see cref="FeatureState.Ready"/>, the default) or planned (a placeholder page).</summary>
    FeatureState State => FeatureState.Ready;

    /// <summary>What it will do once built, one line for its placeholder page; "" for a ready feature (the default).</summary>
    string Coming => "";

    /// <summary>Where it is listed among the labs (<see cref="FeatureGroup.Device"/> unless it says).</summary>
    FeatureGroup Group => FeatureGroup.Device;

    /// <summary>A few words for a lab's card ("Live speech to text"); "" uses <see cref="IKind.Summary"/> (the default).</summary>
    string Tagline => "";

    /// <summary>The name of its icon among Idrak's (<c>Idrak.Maui.Theme.IdrakIcons</c>: "chat", "mic"); "" draws <see cref="Glyph"/> (the default).</summary>
    string Icon => "";

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

    /// <summary>
    /// Its content as a view bound to <paramref name="viewModel"/>, for a section of the app's navigation to host (a tab of
    /// the bottom bar or the rail; <see cref="AppSections"/>); null (the default) when it has none, and the section opens its
    /// page instead.
    /// </summary>
    View? CreateView(IServiceProvider services, object viewModel) => null;
}

/// <summary>
/// The app's features, by name (ignoring case). The library's: "devices" (the devices found, the Vulkan driver loaded in
/// the app's own process with its measured width, subgroup sizes and memory budget, CPU features, memory in use) and
/// "models" (pull a Hugging Face model into the app's storage with its progress, list the models with their sizes,
/// remove one, the storage's free space), "chat" (a model of the store chats, its answer streamed with its tokens per
/// second; the families are the app's to register), "transcribe" (speech into text, live), "speak" (text read aloud on the
/// speaker, played as each sentence is made), "audio-process" (a recording through a pipeline of the library's
/// processors, heard before and after), "ocr" (a photo read by a vision-language model of the store, streamed; the vision
/// families are the app's) and "image" (transforms and scan profiles on a photo, cold and warm). Register
/// another with <see cref="Register"/>; one under a library name shadows the library's, which <see cref="Unregister"/>
/// brings back. <see cref="Enabled"/> lists them in menu order without those the settings turn off.
/// </summary>
public static class MobileFeatures
{
    /// <summary>The library's devices page.</summary>
    public const string Devices = "devices";

    /// <summary>The library's models page.</summary>
    public const string Models = "models";

    private static readonly SlotTable<string, IMobileFeature> Table = BuiltIn();

    /// <summary>The <see cref="IMobileFeature.Order"/> of a feature that sets none: after every ordered one.</summary>
    public const int Unordered = int.MaxValue;

    /// <summary>The library's chat page (planned: a small model streams its answer).</summary>
    public const string Chat = "chat";

    /// <summary>The library's live transcription page (planned).</summary>
    public const string Transcribe = "transcribe";

    /// <summary>The library's speech synthesis page: text read aloud on the speaker.</summary>
    public const string Speak = "speak";

    /// <summary>The library's audio processing page: a sound through a pipeline, heard before and after.</summary>
    public const string AudioProcess = "audio-process";

    /// <summary>The library's text reading page: a photo read by a vision-language model.</summary>
    public const string Ocr = "ocr";

    /// <summary>The library's image transforms page: transforms and scan profiles on a photo.</summary>
    public const string Image = "image";

    /// <summary>Text drawn as scanned pages (the Generate lab).</summary>
    public const string Generate = "generate";

    /// <summary>The app's command runner (the Terminal page).</summary>
    public const string Terminal = "terminal";

    /// <summary>The library's retrieval page (planned).</summary>
    public const string Search = "search";

    private static SlotTable<string, IMobileFeature> BuiltIn()
    {
        var table = new SlotTable<string, IMobileFeature>(nameof(MobileFeatures), comparer: StringComparer.OrdinalIgnoreCase,
            unguarded: "a feature is a page the user has open, with no call to fall back at");
        Idrak.Maui.Views.PlannedFeatures.RegisterDefaults(table);
        table.RegisterDefault(Devices, new Idrak.Maui.Views.DevicesFeature());
        table.RegisterDefault(Models, new Idrak.Maui.Views.ModelsFeature());
        table.RegisterDefault(Chat, new Idrak.Maui.Views.ChatFeature());
        table.RegisterDefault(Transcribe, new Idrak.Maui.Views.TranscribeFeature());
        table.RegisterDefault(Speak, new Idrak.Maui.Views.SpeakFeature());
        table.RegisterDefault(AudioProcess, new Idrak.Maui.Views.AudioProcessFeature());
        table.RegisterDefault(Ocr, new Idrak.Maui.Views.OcrFeature());
        table.RegisterDefault(Image, new Idrak.Maui.Views.ImageFeature());
        table.RegisterDefault(Generate, new Idrak.Maui.Views.GenerateFeature());
        table.RegisterDefault(Terminal, new Idrak.Maui.Views.TerminalFeature());
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

    /// <summary>
    /// Every registered feature in menu order (<see cref="Place"/>, then the title): the ready and the planned
    /// (<see cref="IMobileFeature.State"/>) alike.
    /// </summary>
    public static IReadOnlyList<IMobileFeature> All =>
        [.. Table.Values.OrderBy(Place).ThenBy(f => f.Title, StringComparer.CurrentCultureIgnoreCase)];

    /// <summary>
    /// Where <paramref name="feature"/> stands in the menu: its <see cref="IMobileFeature.Order"/>, or, when it sets none,
    /// the library's feature of its name's (a ready feature registered over a planned one keeps its place), else
    /// <see cref="Unordered"/>.
    /// </summary>
    public static int Place(IMobileFeature feature)
    {
        ArgumentNullException.ThrowIfNull(feature);
        int order = feature.Order;
        return order == Unordered && Table.Default(feature.Name) is { } library ? library.Order : order;
    }

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
