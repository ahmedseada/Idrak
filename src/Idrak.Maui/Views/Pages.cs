// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.Navigation;
using Idrak.Maui.ViewModels;
using Microsoft.Extensions.DependencyInjection;

namespace Idrak.Maui.Views;

/// <summary>
/// A feature's page: its view (the same one a section of the app's navigation hosts) as the page's content, bound to its
/// view model, told when the page appears and disappears (<see cref="IHostedView"/>). Pushed when a feature has no section
/// (from All labs, a link) or opened by <see cref="MobileFeatures.Open"/>.
/// </summary>
public class LabPage : ContentPage
{
    /// <summary>A page showing <paramref name="view"/> over <paramref name="viewModel"/>, titled <paramref name="title"/>.</summary>
    public LabPage(View view, object viewModel, string title)
    {
        ArgumentNullException.ThrowIfNull(view);
        BindingContext = viewModel ?? throw new ArgumentNullException(nameof(viewModel));
        Title = title;
        Content = view;
    }

    /// <summary>The feature's view.</summary>
    public View View => Content;

    /// <inheritdoc />
    protected override void OnAppearing()
    {
        base.OnAppearing();
        (Content as IHostedView)?.Shown();
    }

    /// <inheritdoc />
    protected override void OnDisappearing()
    {
        (Content as IHostedView)?.Hidden();
        base.OnDisappearing();
    }
}

/// <summary>The Devices feature's page (<see cref="DevicesView"/>).</summary>
public sealed class DevicesPage(DevicesViewModel viewModel) : LabPage(new DevicesView(viewModel), viewModel, viewModel.Title);

/// <summary>The Models feature's page (<see cref="ModelsView"/>).</summary>
public sealed class ModelsPage(ModelsViewModel viewModel) : LabPage(new ModelsView(viewModel), viewModel, viewModel.Title);

/// <summary>The Chat feature's page (<see cref="ChatView"/>).</summary>
public sealed class ChatPage(ChatViewModel viewModel, IServiceProvider services) : LabPage(new ChatView(viewModel, services), viewModel, viewModel.Title);

/// <summary>The Transcribe feature's page (<see cref="TranscribeView"/>).</summary>
public sealed class TranscribePage(TranscribeViewModel viewModel, IServiceProvider services) : LabPage(new TranscribeView(viewModel, services), viewModel, viewModel.Title);

/// <summary>The Speak feature's page (<see cref="SpeakView"/>).</summary>
public sealed class SpeakPage(SpeakViewModel viewModel, IServiceProvider services) : LabPage(new SpeakView(viewModel, services), viewModel, viewModel.Title);

/// <summary>The Studio (audio process) feature's page (<see cref="StudioView"/>).</summary>
public sealed class AudioProcessPage(AudioProcessViewModel viewModel) : LabPage(new StudioView(viewModel), viewModel, viewModel.Title);

/// <summary>The Read text (OCR) feature's page (<see cref="OcrView"/>).</summary>
public sealed class OcrPage(OcrViewModel viewModel, IServiceProvider services) : LabPage(new OcrView(viewModel, services), viewModel, viewModel.Title);

/// <summary>The Image feature's page (<see cref="ImageView"/>).</summary>
public sealed class ImagePage(ImageViewModel viewModel) : LabPage(new ImageView(viewModel), viewModel, viewModel.Title);

/// <summary>The Generate feature's page (<see cref="GenerateView"/>).</summary>
public sealed class GeneratePage(GenerateViewModel viewModel, IServiceProvider services) : LabPage(new GenerateView(viewModel, services), viewModel, viewModel.Title);

/// <summary>The fonts of the Generate lab: every font found, searchable, chosen by a switch (<see cref="FontsView"/>).</summary>
public sealed class FontsPage(GenerateViewModel viewModel) : LabPage(new FontsView(viewModel), viewModel, "Fonts");

/// <summary>The Terminal feature's page (<see cref="TerminalView"/>).</summary>
public sealed class TerminalPage(TerminalViewModel viewModel, IServiceProvider services) : LabPage(new TerminalView(viewModel, services), viewModel, viewModel.Title);

/// <summary>A planned feature's page (<see cref="PlaceholderView"/>).</summary>
public sealed class PlaceholderPage(PlaceholderViewModel viewModel) : LabPage(new PlaceholderView(viewModel), viewModel, viewModel.Title);

/// <summary>The settings page (<see cref="SettingsView"/>).</summary>
public sealed class SettingsPage(SettingsViewModel viewModel) : LabPage(new SettingsView(viewModel), viewModel, viewModel.Title);

/// <summary>All labs as a page (where no host shows them).</summary>
public sealed class LabsPage(IServiceProvider services) : LabPage(new LabsView(services, asSheet: false), new object(), "All labs");

/// <summary>The settings page and view from the app's services.</summary>
public static class SettingsPages
{
    /// <summary>The settings page (the app's registered one, else one over the services' options and settings).</summary>
    public static Page Create(IServiceProvider services)
    {
        ArgumentNullException.ThrowIfNull(services);
        return services.GetService<SettingsPage>() ?? new SettingsPage(ViewModel(services));
    }

    /// <summary>The settings view (a section of the rail).</summary>
    public static View CreateView(IServiceProvider services)
    {
        ArgumentNullException.ThrowIfNull(services);
        return new SettingsView(services.GetService<SettingsViewModel>() ?? ViewModel(services));
    }

    private static SettingsViewModel ViewModel(IServiceProvider services)
    {
        var options = services.GetService<IdrakMauiOptions>() ?? new IdrakMauiOptions();
        var settings = services.GetService<IdrakMauiSettings>() ?? new IdrakMauiSettings(new MemorySettingsStore(), new MemorySecretStore());
        var startup = services.GetService<IdrakStartupReport>() ?? new IdrakStartupReport(ComputeResources.CacheFolder, options.Device, [], FileSystemPath());
        return new SettingsViewModel(options, settings, startup, services.GetService<ProgressCenter>(), services.GetService<IAppRestarter>());
    }

    private static string FileSystemPath()
    {
        try
        {
            return FileSystem.AppDataDirectory;
        }
        catch (Exception e) when (e is NotImplementedException or NotSupportedException)
        {
            return ComputeResources.CacheFolder;                                 // no platform (the tests)
        }
    }
}

/// <summary>What views need from the page they are on: a question with two answers, a choice among several.</summary>
internal static class Dialogs
{
    /// <summary>Asks <paramref name="title"/> on the page <paramref name="view"/> is on; false when there is no page to ask on.</summary>
    public static Task<bool> ConfirmAsync(Element view, string title, string message, string accept, string cancel) =>
        PageOf(view) is { } page ? page.DisplayAlertAsync(title, message, accept, cancel) : Task.FromResult(false);

    /// <summary>The page <paramref name="element"/> is on, else the window's.</summary>
    public static Page? PageOf(Element element)
    {
        for (var e = element; e is not null; e = e.Parent)
        {
            if (e is Page page)
            {
                return page;
            }
        }

        return Application.Current?.Windows.FirstOrDefault()?.Page;
    }

    /// <summary>The navigator of the app's services, or one that pushes pages on the window's navigation.</summary>
    public static IAppNavigator Navigator(IServiceProvider services) => services.GetService<IAppNavigator>() ?? new AppNavigator(services);
}

/// <summary>A section's stand-in when its view could not be made: what failed.</summary>
public sealed class FailedView : ContentView
{
    /// <summary>A view saying that <paramref name="title"/> could not open, and why.</summary>
    public FailedView(string title, Exception error)
    {
        ArgumentNullException.ThrowIfNull(error);
        var heading = new Label { Text = $"{title} could not open" };
        heading.SetDynamicResource(StyleProperty, "IdrakHeading");
        var why = new Label { Text = error.Message.Split('\n')[0] };
        why.SetDynamicResource(StyleProperty, "IdrakErrorText");
        var stack = new VerticalStackLayout { heading, why };
        stack.SetDynamicResource(VerticalStackLayout.SpacingProperty, "IdrakSpacingSmall");
        stack.SetDynamicResource(PaddingProperty, "IdrakPagePadding");
        Content = stack;
    }
}

/// <summary>A feature without a view of its own in a section: its title, summary and a button that opens its page.</summary>
public sealed class FeatureLinkView : ContentView
{
    /// <summary>A card for <paramref name="feature"/>.</summary>
    public FeatureLinkView(IMobileFeature feature, IServiceProvider services)
    {
        ArgumentNullException.ThrowIfNull(feature);
        ArgumentNullException.ThrowIfNull(services);
        var title = new Label { Text = feature.Title };
        title.SetDynamicResource(StyleProperty, "IdrakTitle");
        var summary = new Label { Text = feature.Summary };
        summary.SetDynamicResource(StyleProperty, "IdrakCaption");
        var open = new Button { Text = "Open" };
        open.SetDynamicResource(StyleProperty, "IdrakGradientButton");
        open.Clicked += async (_, _) => await Dialogs.Navigator(services).PushAsync(MobileFeatures.Open(feature.Name, services));
        var stack = new VerticalStackLayout { title, summary, open };
        stack.SetDynamicResource(VerticalStackLayout.SpacingProperty, "IdrakSpacing");
        stack.SetDynamicResource(PaddingProperty, "IdrakPagePadding");
        Content = stack;
    }
}
