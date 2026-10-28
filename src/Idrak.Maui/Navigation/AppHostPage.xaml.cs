// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.Controls;
using Idrak.Maui.Theme;
using Idrak.Maui.Views;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Maui.Controls.PlatformConfiguration.AndroidSpecific;

namespace Idrak.Maui.Navigation;

/// <summary>
/// The app's host page: shows one section's view at a time. Each is made once and kept in the page (the one shown visible,
/// the others hidden), so a tab tapped again only shows its view: nothing is made, moved or attached again. The tap's
/// switch is made first and the view is told it is shown after (its refresh never delays the switch); a fade, when the
/// settings ask for one, runs alongside. With <see cref="IdrakMauiOptions.PrebuildSections"/> the other tabs' views are made
/// ahead, one at a time while the app is idle after Home first shows. It measures the window on every size change into
/// <see cref="AppLayout"/>, which switches the bottom bar and the rail at once (rotation, split screen); opens All labs as a
/// sheet on a narrow window. It is the <see cref="IAppNavigator"/>'s host while shown. <see cref="CreateWindow"/> makes the
/// app's window around it.
/// </summary>
public partial class AppHostPage : ContentPage
{
    private readonly IServiceProvider _services;
    private readonly IdrakMauiOptions _options;
    private readonly Dictionary<string, View> _views = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, Page> _pages = new(StringComparer.OrdinalIgnoreCase);
    private LabsView? _sheet;
    private Page? _settingsPage;
    private View? _current;
    private bool _onScreen;
    private bool _prebuilt;
    private bool _resizes;
    private bool _startupBegun;
    private readonly ISoftKeyboard? _keyboard;
    private readonly StartupRun? _startup;
    private SplashView? _splash;

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
                if (!ViewModel.ShowsSame())
                {
                    ViewModel.Refresh();                                         // the tabs listed again only when the features shown changed
                }
            });
        }

        SizeChanged += (_, _) => ApplySize(Width, Height);
        KeyboardAware(this);
        _keyboard = services.GetService<ISoftKeyboard>();
        if (_keyboard is not null)
        {
            _keyboard.VisibilityChanged += (_, visible) => Dispatcher.Dispatch(() => ViewModel.IsKeyboardOpen = visible);
        }

        Show(AppSections.Get(AppSections.Home), null);
        _startup = services.GetService<StartupRun>();
        if (_startup is not null && _options.ShowSplash && _options.Splash != SplashLength.Off)
        {
            ShowSplash(_startup, layout.Breakpoint);
        }
#if WINDOWS
        Platforms.Windows.WindowsKeyInput.WatchEscape(this);                    // Esc closes the sheet on top
#endif
    }

    /// <summary>
    /// Closes the sheet on top, as Esc does on a desktop: the sheet of a page pushed over the host (a lab's page: the
    /// Terminal's commands pane), else All labs' sheet, else the shown section's (the chat's tune sheet); false when none is
    /// open (nothing else changes).
    /// </summary>
    public bool CloseTopSheet()
    {
        if (Navigation.NavigationStack.LastOrDefault() is { } top && !ReferenceEquals(top, this))
        {
            return top is LabPage { View: IHostedView pushed } && pushed.CloseSheet();
        }

        if (ViewModel.IsSheetOpen)
        {
            ViewModel.IsSheetOpen = false;
            return true;
        }

        return _current is IHostedView shown && shown.CloseSheet();
    }

    /// <summary>
    /// Makes <paramref name="page"/>'s bottom edge follow the soft keyboard and the system's bars (<c>SafeAreaEdges</c>: the
    /// keyboard's region with the bars', where Android draws edge to edge, as from API 35 it always does), the other edges
    /// as the page has them; with the window resized for the keyboard where Android still resizes it (adjustResize, set once
    /// for the app). The host and every pushed page (<see cref="LabPage"/>) take it, so a field being typed into, and a
    /// composer under a list, sit right above the keyboard.
    /// </summary>
    public static void KeyboardAware(ContentPage page)
    {
        ArgumentNullException.ThrowIfNull(page);
        page.SafeAreaEdges = new SafeAreaEdges(SafeAreaRegions.Default, SafeAreaRegions.Default, SafeAreaRegions.Default, SafeAreaRegions.All);
    }

    /// <summary>The host's navigation state.</summary>
    public AppHostViewModel ViewModel { get; }

    /// <summary>The view shown.</summary>
    public View? Current => _current;

    /// <summary>The animated splash while it shows over the page (made underneath meanwhile), or null.</summary>
    public SplashView? Splash => _splash;

    /// <summary>Whether the tabs' views were made ahead (<see cref="PrebuildAsync"/>) or are being made.</summary>
    public bool IsPrebuilt => _prebuilt;

    /// <summary>
    /// Runs the app's start-up steps (<see cref="StartupRun.RunAsync"/>, the run from the app's services) once, with this page
    /// as their host; the window's first showing does it, before Home is told it is shown. Null when the app has no run.
    /// </summary>
    public Task? BeginStartup()
    {
        if (_startup is null)
        {
            return null;
        }

        _startupBegun = true;
        return _startup.RunAsync(new StartupContext(_services, this));
    }

    /// <summary>The sections' views made so far (shown or kept hidden), by section name.</summary>
    public IReadOnlyDictionary<string, View> Views => _views;

    /// <summary>
    /// The app's window: this host (from the app's services) in a navigation page, its own bar hidden; on a desktop its
    /// least size and its kept size and place (the app's <see cref="IWindowPlacement"/>).
    /// </summary>
    public static Window CreateWindow(IServiceProvider services)
    {
        ArgumentNullException.ThrowIfNull(services);
        var host = services.GetService<AppHostPage>() ?? new AppHostPage(services);
        var window = new Window(new NavigationPage(host)) { Title = "Idrak" };
        services.GetService<IWindowPlacement>()?.Attach(window);
        return window;
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
            if (Navigation.NavigationStack.LastOrDefault() is { } top && ReferenceEquals(top, _settingsPage))
            {
                return Task.FromResult(true);                                    // shown already
            }

            // The same settings page each time (made once, its views kept when it is closed), so opening it again makes nothing.
            if (_settingsPage is null)
            {
                _settingsPage = SettingsPages.Create(_services);
                _settingsPage.SetValue(HandlerProperties.DisconnectPolicyProperty, HandlerDisconnectPolicy.Manual);
            }

            return Navigation.PushAsync(_settingsPage).ContinueWith(_ => true, TaskScheduler.Current);
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

        // A feature's page (no section shows it: Devices, Models, Terminal) made once and kept, its views too when it closes,
        // so opening it again makes nothing; one already shown is not pushed twice.
        if (!_pages.TryGetValue(feature, out var page))
        {
            page = MobileFeatures.Open(feature, _services);
            page.SetValue(HandlerProperties.DisconnectPolicyProperty, HandlerDisconnectPolicy.Manual);
            _pages[feature] = page;
        }

        if (!Navigation.NavigationStack.Contains(page))
        {
            await Navigation.PushAsync(page);
        }
    }

    /// <summary>The features' pages made so far (kept), by feature name.</summary>
    public IReadOnlyDictionary<string, Page> Pages => _pages;

    /// <summary>Shows All labs: the sheet on a narrow window, the section on a wide one.</summary>
    public Task OpenLabsAsync() => OpenSectionAsync(AppSections.Labs);

    /// <summary>Shows the settings: pushed on a narrow window, the rail's section on a wide one.</summary>
    public Task OpenSettingsAsync() => OpenSectionAsync(AppSections.Settings);

    /// <inheritdoc />
    protected override void OnAppearing()
    {
        base.OnAppearing();
        _onScreen = true;
        if (!_resizes && Microsoft.Maui.Controls.Application.Current is { } app)
        {
            // Once for the app (it was set and restored by the chat page alone): where Android still resizes the window for
            // the keyboard (before edge to edge), it resizes it for every page; where it draws edge to edge, SafeAreaEdges pads.
            app.On<Microsoft.Maui.Controls.PlatformConfiguration.Android>().UseWindowSoftInputModeAdjust(
                Microsoft.Maui.Controls.PlatformConfiguration.AndroidSpecific.WindowSoftInputModeAdjust.Resize);
            _resizes = true;
        }

        _keyboard?.Watch();
        if (!_startupBegun)
        {
            _ = BeginStartup();                                                 // under the splash: the steps it shows are this work
        }

        (_current as IHostedView)?.Shown();
        if (ViewModel.IsSheetOpen)
        {
            _sheet?.Shown();
        }

        if (!_prebuilt && _options.PrebuildSections)
        {
            _prebuilt = true;
            Prebuild();
        }
    }

    /// <inheritdoc />
    protected override void OnDisappearing()
    {
        _onScreen = false;
        (_current as IHostedView)?.Hidden();
        _sheet?.Hidden();
        base.OnDisappearing();
    }

    /// <summary>
    /// Makes every visible section's view that is not made yet (and the labs of a tab of several), one a turn of the UI
    /// thread's queue after the frame the app shows now, so each tab's first tap only shows it. Each is made hidden; none is
    /// told it is shown. Called once after Home first shows when <see cref="IdrakMauiOptions.PrebuildSections"/> is on (the
    /// start-up's "sections" step, <see cref="PrebuildAsync"/>, where the app has a start-up run).
    /// </summary>
    public void Prebuild() => _ = PrebuildAsync();

    /// <summary>As <see cref="Prebuild"/>; the task ends when the last view is made.</summary>
    public Task PrebuildAsync()
    {
        _prebuilt = true;
        var made = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var pending = new Queue<Action>();
        foreach (var section in AppSections.Visible(_options))
        {
            if (section.Placement == SectionPlacement.Foot && ViewModel.IsNarrow)
            {
                continue;                                                        // a narrow window opens these as a sheet and a page
            }

            pending.Enqueue(() =>
            {
                var view = ViewOf(section);
                if (view is IPrebuilt prebuilt)
                {
                    foreach (var part in prebuilt.Parts())
                    {
                        pending.Enqueue(part);
                    }
                }
            });
        }

        pending.Enqueue(() =>
        {
            if (ViewModel.IsNarrow)
            {
                _sheet ??= new LabsView(_services, asSheet: true);
                SheetBody.Content ??= _sheet;                                    // in the page already, hidden with the sheet
            }
        });
        void Next()
        {
            if (pending.TryDequeue(out var make))
            {
                try
                {
                    make();
                }
                catch (Exception e) when (e is not OutOfMemoryException)
                {
                    // A section that cannot be made says why when it is opened (MakeView); the rest are still made.
                }

                Dispatcher.Dispatch(Next);                                       // one a turn: a tap between two is taken at once
            }
            else
            {
                made.TrySetResult();
            }
        }

        Dispatcher.Dispatch(Next);
        return made.Task;
    }

    /// <inheritdoc />
    protected override bool OnBackButtonPressed()
    {
        if (_splash is not null)
        {
            return true;                                                         // the splash hands over by itself; back neither closes nor skips it
        }

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

    // The splash over both columns, the page's ground the splash's meanwhile (under the system's bars too), taken away once
    // it has faded into Home. It starts where the system's splash left off: its picture's size, and its theme's ground
    // crossing to the app's where the app's theme setting forces the other one.
    private void ShowSplash(StartupRun run, double breakpoint)
    {
        var splash = new SplashView(run, _options) { Breakpoint = breakpoint };
        var system = Microsoft.Maui.Controls.Application.Current?.PlatformAppTheme ?? AppTheme.Unspecified;
        var effective = IdrakThemes.Effective(_options.Theme, system);
        var drewSystem = SystemSplashTheme(system);
        if (drewSystem is { } drew && drew != effective)
        {
            var colors = IdrakThemes.Colors(drew);
            if (colors["IdrakSplashGround"] is Color ground)
            {
                splash.CrossFrom(ground, (Color)colors["IdrakSplashTile"], (Color)colors["IdrakSplashTileStroke"], (Color)colors["IdrakSplashShadow"]);
            }
        }

        if ((_options.SplashPictureSize ?? SystemSplashPicture()) is double picture and > 0)
        {
            bool tiled = IdrakThemes.Colors(drewSystem ?? effective)["IdrakSplashTile"] is Color { Alpha: > 0 };
            splash.StartSize = tiled ? picture * Math.Clamp(_options.SplashTileScale, 0.01, 1) : picture;
        }

        Grid.SetColumnSpan(splash, 2);
        Root.Children.Add(splash);
        SetDynamicResource(BackgroundColorProperty, "IdrakSplashGround");
        splash.HandingOver += (_, _) => SetDynamicResource(BackgroundColorProperty, "IdrakPageBackground");
        splash.Finished += (_, _) =>
        {
            Root.Children.Remove(splash);
            _splash = null;
        };
        _splash = splash;
    }

    // The theme the system's own splash was drawn in: Android's follows the system's theme; elsewhere none is known.
    private static AppTheme? SystemSplashTheme(AppTheme system)
    {
#if ANDROID
        return system == AppTheme.Dark ? AppTheme.Dark : AppTheme.Light;
#else
        _ = system;
        return null;
#endif
    }

    // The size the system's splash drew its picture at: Android 12 and later its icon's size, before it the app's base size.
    private double? SystemSplashPicture()
    {
#if ANDROID
        return OperatingSystem.IsAndroidVersionAtLeast(31) ? _options.SplashSystemIconSize : _options.SplashBaseSize;
#else
        return null;
#endif
    }

    private void Show(IAppSection section, string? feature)
    {
        ViewModel.IsSheetOpen = false;
        var view = ViewOf(section);
        ViewModel.Select(section.Name);                                          // the tab marked at once
        if (!ReferenceEquals(_current, view))
        {
            var previous = _current;
            _current = view;
            view.IsVisible = true;                                               // the switch first: the view is in the page already
            if (previous is not null)
            {
                previous.IsVisible = false;
            }

            if (_options.SectionFade > TimeSpan.Zero)
            {
                Motion.FadeIn(view, (uint)Math.Min(uint.MaxValue, _options.SectionFade.TotalMilliseconds));
            }
            else
            {
                view.Opacity = 1;
            }

            if (_onScreen)
            {
                (previous as IHostedView)?.Hidden();
                Dispatcher.Dispatch(() =>
                {
                    if (_onScreen && ReferenceEquals(_current, view))
                    {
                        (view as IHostedView)?.Shown();                          // after the switch has drawn: its refresh never delays it
                    }
                });
            }
        }

        if (feature is not null)
        {
            (view as IHostedView)?.ShowFeature(feature);
        }
    }

    // The section's view, made once and kept in the page, hidden until shown.
    private View ViewOf(IAppSection section)
    {
        if (!_views.TryGetValue(section.Name, out var view))
        {
            view = MakeView(section);
            view.IsVisible = false;
            _views[section.Name] = view;
            Body.Children.Add(view);
        }

        return view;
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
            if (_options.SectionFade > TimeSpan.Zero)
            {
                Motion.FadeIn(SheetPanel, (uint)Math.Min(uint.MaxValue, _options.SectionFade.TotalMilliseconds));    // alongside: the sheet is open already
            }
            else
            {
                SheetPanel.Opacity = 1;
            }
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

/// <summary>A view with parts it can make ahead (a tab of several labs: the labs not shown yet), one a turn.</summary>
internal interface IPrebuilt
{
    /// <summary>What is left to make, each a step the host runs on its own turn of the UI thread.</summary>
    IEnumerable<Action> Parts();
}
