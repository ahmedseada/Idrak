// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.Controls;
using Idrak.Maui.Views;
using Microsoft.Extensions.DependencyInjection;

namespace Idrak.Maui.Navigation;

/// <summary>
/// The app's host page: shows one section's view at a time (each made once and kept, told when it is shown and hidden),
/// cross-fading between them; measures the window on every size change into <see cref="AppLayout"/>, which switches the
/// bottom bar and the rail at once (rotation, split screen); opens All labs as a sheet on a narrow window. It is the
/// <see cref="IAppNavigator"/>'s host while shown. <see cref="CreateWindow"/> makes the app's window around it.
/// </summary>
public partial class AppHostPage : ContentPage
{
    private readonly IServiceProvider _services;
    private readonly IdrakMauiOptions _options;
    private readonly Dictionary<string, View> _views = new(StringComparer.OrdinalIgnoreCase);
    private LabsView? _sheet;
    private bool _onScreen;

    /// <summary>The host over the app's services, showing Home.</summary>
    public AppHostPage(IServiceProvider services)
    {
        _services = services ?? throw new ArgumentNullException(nameof(services));
        _options = services.GetService<IdrakMauiOptions>() ?? new IdrakMauiOptions();
        var layout = services.GetService<AppLayout>() ?? new AppLayout(_options.WideLayoutWidth);
        ViewModel = new AppHostViewModel(_options, layout);
        BindingContext = ViewModel;
        InitializeComponent();
        if (services.GetService<IAppNavigator>() is AppNavigator navigator)
        {
            navigator.Host = this;
        }

        ViewModel.SelectRequested += async (_, name) => await OpenSectionAsync(name);
        ViewModel.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == nameof(AppHostViewModel.IsSheetOpen))
            {
                OnSheet();
            }
        };
        if (services.GetService<IdrakMauiSettings>() is { } settings)
        {
            settings.Saved += (_, saved) => Dispatcher.Dispatch(() =>
            {
                layout.Breakpoint = saved.WideLayoutWidth;
                ViewModel.Refresh();
            });
        }

        SizeChanged += (_, _) => ApplySize(Width, Height);
        Show(AppSections.Get(AppSections.Home), null);
    }

    /// <summary>The host's navigation state.</summary>
    public AppHostViewModel ViewModel { get; }

    /// <summary>The view shown.</summary>
    public View? Current => Body.Content;

    /// <summary>The app's window: this host (from the app's services) in a navigation page, its own bar hidden.</summary>
    public static Window CreateWindow(IServiceProvider services)
    {
        ArgumentNullException.ThrowIfNull(services);
        var host = services.GetService<AppHostPage>() ?? new AppHostPage(services);
        return new Window(new NavigationPage(host)) { Title = "Idrak" };
    }

    /// <summary>Takes the window's measured size: its width chooses the layout (<see cref="AppLayout"/>), its height the sheet's.</summary>
    public void ApplySize(double width, double height)
    {
        if (width > 0)
        {
            ViewModel.Layout.Width = width;
        }

        if (height > 0)
        {
            SheetPanel.HeightRequest = Math.Round(height * 0.84);
        }
    }

    /// <summary>Shows the section <paramref name="name"/> (and in it the feature <paramref name="feature"/>); false when none of that name shows.</summary>
    public Task<bool> OpenSectionAsync(string name, string? feature = null)
    {
        if (AppSections.Visible(_options).FirstOrDefault(s => string.Equals(s.Name, name, StringComparison.OrdinalIgnoreCase)) is not { } section)
        {
            return Task.FromResult(false);
        }

        if (ViewModel.IsNarrow && section.Name.Equals(AppSections.Labs, StringComparison.OrdinalIgnoreCase))
        {
            ViewModel.IsSheetOpen = true;
            return Task.FromResult(true);
        }

        if (ViewModel.IsNarrow && section.Name.Equals(AppSections.Settings, StringComparison.OrdinalIgnoreCase))
        {
            ViewModel.IsSheetOpen = false;
            return Navigation.PushAsync(SettingsPages.Create(_services)).ContinueWith(_ => true, TaskScheduler.Current);
        }

        Show(section, feature);
        return Task.FromResult(true);
    }

    /// <summary>Opens the feature <paramref name="feature"/> in the section that hosts it, or pushes its page when none does.</summary>
    public async Task OpenFeatureAsync(string feature)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(feature);
        ViewModel.IsSheetOpen = false;
        if (AppSections.Hosting(feature, _options) is { } section)
        {
            await OpenSectionAsync(section.Name, feature);
            return;
        }

        await Navigation.PushAsync(MobileFeatures.Open(feature, _services));
    }

    /// <summary>Shows All labs: the sheet on a narrow window, the section on a wide one.</summary>
    public Task OpenLabsAsync() => OpenSectionAsync(AppSections.Labs);

    /// <summary>Shows the settings: pushed on a narrow window, the rail's section on a wide one.</summary>
    public Task OpenSettingsAsync() => OpenSectionAsync(AppSections.Settings);

    /// <inheritdoc />
    protected override void OnAppearing()
    {
        base.OnAppearing();
        _onScreen = true;
        (Body.Content as IHostedView)?.Shown();
        if (ViewModel.IsSheetOpen)
        {
            _sheet?.Shown();
        }
    }

    /// <inheritdoc />
    protected override void OnDisappearing()
    {
        _onScreen = false;
        (Body.Content as IHostedView)?.Hidden();
        _sheet?.Hidden();
        base.OnDisappearing();
    }

    /// <inheritdoc />
    protected override bool OnBackButtonPressed()
    {
        if (ViewModel.IsSheetOpen)
        {
            ViewModel.IsSheetOpen = false;
            return true;
        }

        if (!string.Equals(ViewModel.Selected, AppSections.Home, StringComparison.OrdinalIgnoreCase))
        {
            Show(AppSections.Get(AppSections.Home), null);
            return true;
        }

        return base.OnBackButtonPressed();
    }

    private void Show(IAppSection section, string? feature)
    {
        ViewModel.IsSheetOpen = false;
        if (!_views.TryGetValue(section.Name, out var view))
        {
            view = MakeView(section);
            _views[section.Name] = view;
        }

        if (!ReferenceEquals(Body.Content, view))
        {
            if (_onScreen)
            {
                (Body.Content as IHostedView)?.Hidden();
            }

            Body.Content = view;
            Motion.FadeIn(view);
            if (_onScreen)
            {
                (view as IHostedView)?.Shown();
            }
        }

        ViewModel.Select(section.Name);
        if (feature is not null)
        {
            (view as IHostedView)?.ShowFeature(feature);
        }
    }

    private View MakeView(IAppSection section)
    {
        try
        {
            return section.CreateView(_services);
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            return new FailedView(section.Title, e);                               // a section that cannot open says why; the rest still work
        }
    }

    private void OnSheet()
    {
        if (ViewModel.IsSheetOpen)
        {
            _sheet ??= new LabsView(_services, asSheet: true);
            SheetBody.Content ??= _sheet;
            Motion.FadeIn(SheetPanel, 180);
            if (_onScreen)
            {
                _sheet.Shown();
            }
        }
        else
        {
            _sheet?.Hidden();
        }
    }
}
