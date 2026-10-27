// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.Controls;
using Idrak.Maui.Navigation;
using Idrak.Maui.Theme;
using Idrak.Maui.ViewModels;
using Idrak.Maui.Views;
using Microsoft.Extensions.DependencyInjection;

namespace Idrak.Maui.Tests;

// A desktop's keyboard and mouse (plan 34, phase W): what the keys and the wheel do, through the shared paths the Windows
// handlers call (DesktopKeys.Press, ZoomView.ZoomByWheel, AppHostPage.CloseTopSheet), and the hover states of the styles.
// The WinUI handlers themselves (Platforms/Windows) build and run only on Windows.
internal static partial class MauiTests
{
    private static void DesktopChatEnter()
    {
        string folder = ChatFolder("desktop-enter");
        var model = new ScriptedChat(2, "ok", Stats(2, 1));
        var options = new IdrakMauiOptions();
        var page = new ChatViewModel(new ListedStore(Stored(folder)), new OneChat(model), options, new InlineDispatcher(), new ManualTime());
        page.RefreshAsync().GetAwaiter().GetResult();

        // The composer's Editor carries the view model's Enter (the view's XAML), so the key handler finds it there.
        using var services = AppServices();
        var view = new ChatView(services.GetRequiredService<ChatViewModel>(), services);
        var composer = view.GetVisualTreeDescendants().OfType<Editor>().SingleOrDefault(e => DesktopKeys.GetEnterCommand(e) is not null);
        Check(composer is not null && ReferenceEquals(DesktopKeys.GetEnterCommand(composer), services.GetRequiredService<ChatViewModel>().EnterCommand), "the composer's Enter");

        var editor = new Editor();
        DesktopKeys.SetEnterCommand(editor, page.EnterCommand);
        page.Draft = "hello";
        Check(DesktopKeys.Press(editor, DesktopKey.Enter) && model.Requests.Count == 0 && page.Draft == "hello", "no model chosen: Enter taken, nothing sent");
        page.SelectedModel = page.Models[0];
        page.Loading!.GetAwaiter().GetResult();
        Check(!DesktopKeys.Press(editor, DesktopKey.Enter, shift: true) && model.Requests.Count == 0, "Shift+Enter is the field's: a new line");
        Check(DesktopKeys.Press(editor, DesktopKey.Enter), "Enter sends");
        Check(SpinWait.SpinUntil(() => !page.IsStreaming && page.Messages.Count == 2, TimeSpan.FromSeconds(30)) && model.Requests.Count == 1
              && page.Messages[0].Text == "hello" && page.Draft.Length == 0, $"sent: {model.Requests.Count} requests, {page.Messages.Count} bubbles, draft '{page.Draft}'");
        Check(DesktopKeys.Press(editor, DesktopKey.Enter) && model.Requests.Count == 1, "an empty message: Enter taken, nothing sent");

        // The setting: off, Enter is the field's (a new line); kept as chat_enter_sends.
        var setting = IdrakMauiSettings.Definitions.Single(d => d.Key == "chat_enter_sends");
        Check(setting is { Group: "Chat", Kind: IdrakSettingKind.Choice, Placeholder: "on" } && page.EnterSends, $"{setting.Group} {setting.Placeholder}");
        setting.Set(options, "off");
        page.Draft = "again";
        Check(options.ChatEnterSends == false && !page.EnterSends && !DesktopKeys.Press(editor, DesktopKey.Enter) && model.Requests.Count == 1 && setting.Get(options) == "off",
            "off: Enter makes a new line");
        setting.Set(options, "");
        Check(page.EnterSends && options.ChatEnterSends is null, "empty: on again");
        Check(!DesktopKeys.Press(new Editor(), DesktopKey.Enter) && !DesktopKeys.Press(editor, DesktopKey.Up), "a field without the key's command keeps the key");
    }

    private static void DesktopTerminalKeysAndEscape()
    {
        // The Terminal's line: the arrows recall the history (the view's XAML binds them), the caret at the end; Enter runs it.
        using var services = AppServices();
        var vm = Terminal(new ScriptedRunner((_, o, _, _) =>
        {
            o.WriteLine("ran");
            return 0;
        }), out _);
        var view = new TerminalView(vm, services);
        var line = view.FindByName<Entry>("Line");
        Check(ReferenceEquals(DesktopKeys.GetUpCommand(line), vm.PreviousCommand) && ReferenceEquals(DesktopKeys.GetDownCommand(line), vm.NextCommand), "the line's arrows");
        vm.RunAsync("models list").GetAwaiter().GetResult();
        vm.RunAsync("help").GetAwaiter().GetResult();
        Check(DesktopKeys.Press(line, DesktopKey.Up) && vm.Input == "help" && line.Text == "help" && line.CursorPosition == 4, $"up: '{vm.Input}', '{line.Text}' at {line.CursorPosition}");
        Check(DesktopKeys.Press(line, DesktopKey.Up) && vm.Input == "models list" && line.CursorPosition == "models list".Length, $"up again: '{vm.Input}' at {line.CursorPosition}");
        Check(DesktopKeys.Press(line, DesktopKey.Down) && vm.Input == "help" && DesktopKeys.Press(line, DesktopKey.Down) && vm.Input.Length == 0, $"down: '{vm.Input}'");
        Check(!DesktopKeys.Press(line, DesktopKey.Enter), "Enter is the line's own (Completed runs it)");

        // Esc: the commands pane of a narrow window closes; a wide one shows it beside the output, so nothing closes.
        view.Apply(false);
        vm.ShowCommands = true;
        Check(view.CloseSheet() && !vm.ShowCommands && !view.CloseSheet(), "narrow: the commands pane closed, then nothing");
        view.Apply(true);
        vm.ShowCommands = true;
        Check(!view.CloseSheet(), "wide: the pane is not a sheet");

        // The host's path: All labs' sheet, then the shown section's (the chat's tune sheet), nothing when none is open.
        var host = new AppHostPage(services);
        host.ApplySize(412, 915);
        Check(!host.CloseTopSheet(), "nothing open");
        host.OpenLabsAsync().GetAwaiter().GetResult();
        Check(host.ViewModel.IsSheetOpen && host.CloseTopSheet() && !host.ViewModel.IsSheetOpen, "All labs' sheet closed");
        host.ViewModel.Find(AppSections.Chat)!.SelectCommand.Execute(null);
        var chat = services.GetRequiredService<ChatViewModel>();
        chat.IsTuning = true;
        Check(host.Current is ChatView && host.CloseTopSheet() && !chat.IsTuning && !host.CloseTopSheet(), $"the tune sheet closed: {host.Current?.GetType().Name}");
        host.OpenFeatureAsync(MobileFeatures.Speak).GetAwaiter().GetResult();
        Check(host.Current is SegmentedSectionView && !host.CloseTopSheet(), "a tab of several labs: its lab's sheet, none open");
    }

    private static void DesktopWheelZoom()
    {
        var zoom = new ZoomView { Content = new BoxView() };
        ((IView)zoom).Measure(400, 300);
        ((IView)zoom).Arrange(new Rect(0, 0, 400, 300));
        Check(zoom.Width == 400 && zoom.Height == 300 && zoom.Zoom == 1 && zoom.WheelStep == 1.2, $"{zoom.Width}x{zoom.Height} at {zoom.Zoom}");
        zoom.ZoomByWheel(1);
        Check(Math.Abs(zoom.Zoom - 1.2) < 1e-9 && zoom.Content!.TranslationX == 0 && zoom.Content.TranslationY == 0, $"a notch about the middle: {zoom.Zoom}");
        zoom.ZoomByWheel(2, 1, 1);
        double spareX = 400 * (zoom.Zoom - 1) / 2, spareY = 300 * (zoom.Zoom - 1) / 2;
        Check(Math.Abs(zoom.Zoom - Math.Pow(1.2, 3)) < 1e-9 && zoom.Content.TranslationX < 0 && zoom.Content.TranslationY < 0
              && zoom.Content.TranslationX >= -spareX && zoom.Content.TranslationY >= -spareY, $"about the corner: {zoom.Zoom}, {zoom.Content.TranslationX}, {zoom.Content.TranslationY}");
        zoom.ZoomByWheel(0.5);
        Check(Math.Abs(zoom.Zoom - Math.Pow(1.2, 3.5)) < 1e-9, $"half a notch (a touchpad): {zoom.Zoom}");
        zoom.ZoomByWheel(100);
        Check(zoom.Zoom == zoom.MaximumScale, $"held to the most: {zoom.Zoom}");
        zoom.ZoomByWheel(-100);
        Check(zoom.Zoom == 1 && zoom.Content.TranslationX == 0 && zoom.Content.TranslationY == 0, $"out to the whole: {zoom.Zoom}, {zoom.Content.TranslationX}");
        zoom.ZoomByWheel(double.NaN);
        zoom.WheelStep = 2;
        zoom.ZoomByWheel(1);
        Check(zoom.Zoom == 2, $"the step is the view's: {zoom.Zoom}");

        // Generate's preview is a ZoomView (the one the wheel zooms on Windows).
        string storage = Temp("desktop-wheel");
        var generate = new GenerateView(new GenerateViewModel(new FakeShare(), new MemorySettingsStore(), storage, defaults: new GenerateLabOptions { SystemFonts = Path.Combine(storage, "none") }));
        Check(generate.FindByName<ZoomView>("Zoom") is not null, "Generate's preview");
    }

    private static void DesktopHover()
    {
        // Once, in the styles: each hovered style gets its PointerOver state, its values theme keys both palettes carry.
        var styles = new IdrakStyles(pointerHover: true);
        var light = IdrakThemes.Colors(AppTheme.Light);
        var dark = IdrakThemes.Colors(AppTheme.Dark);
        Check(styles.PointerHover && new IdrakStyles().PointerHover == IdrakPlatform.IsDesktop && IdrakStyles.Hovered.Count >= 15, "hover on a desktop");
        foreach (var (key, states) in IdrakStyles.Hovered)
        {
            var style = (Style)styles[key];
            var setter = style.Setters.SingleOrDefault(s => s.Property == VisualStateManager.VisualStateGroupsProperty);
            var over = (setter?.Value as VisualStateGroupList)?.SelectMany(g => g.States).SingleOrDefault(s => s.Name == VisualStateManager.CommonStates.PointerOver);
            Check(over is { Setters.Count: > 0 } && ReferenceEquals(setter!.Value, styles[states]), $"{key}: its hover ({states})");
            foreach (var value in over!.Setters.Select(s => s.Value))
            {
                Check(value is not Microsoft.Maui.Controls.Internals.DynamicResource resource || (light.ContainsKey(resource.Key) && dark.ContainsKey(resource.Key))
                      || styles.ContainsKey(resource.Key), $"{key}: {(value as Microsoft.Maui.Controls.Internals.DynamicResource)?.Key} in both palettes");
            }
        }

        foreach (string key in IdrakStyles.Hovered.Select(h => h.Style))
        {
            Check(!((Style)new IdrakStyles(pointerHover: false)[key]).Setters.Any(s => s.Property == VisualStateManager.VisualStateGroupsProperty), $"{key}: a phone takes no hover");
        }

        // On a page: hovered, a chip's line, a card's ground and line, a gradient button's shade and the rail's pill change to
        // the theme's hover keys; back to their own after.
        foreach (var (theme, colors) in new[] { (AppTheme.Light, light), (AppTheme.Dark, dark) })
        {
            var chip = new Button { Text = "chip" };
            chip.SetDynamicResource(VisualElement.StyleProperty, "IdrakChip");
            var card = new Border();
            card.SetDynamicResource(VisualElement.StyleProperty, "IdrakLabCard");
            var accent = new Button { Text = "run" };
            accent.SetDynamicResource(VisualElement.StyleProperty, "IdrakGradientButton");
            var pill = new Border();
            pill.SetDynamicResource(VisualElement.StyleProperty, "IdrakRailPill");
            var on = new Border();
            on.SetDynamicResource(VisualElement.StyleProperty, "IdrakRailPillOn");
            var page = Themed(new VerticalStackLayout { chip, card, accent, pill, on }, theme);
            Check(page is not null && Equals(chip.BorderColor, colors["IdrakLine2"]), $"{theme}: the chip's own line {chip.BorderColor?.ToArgbHex()}");
            foreach (var view in new VisualElement[] { chip, card, accent, pill })
            {
                Check(VisualStateManager.GoToState(view, VisualStateManager.CommonStates.PointerOver), $"{theme}: {view.GetType().Name} hovers");
            }

            Check(Equals(chip.BorderColor, colors["IdrakHoverStroke"]) && card.Stroke is SolidColorBrush { Color: var stroke } && stroke.Equals(colors["IdrakHoverStroke"])
                  && card.Background is SolidColorBrush { Color: var ground } && ground.Equals(colors["IdrakSurface2"]) && SameGradient(accent.Background, colors["IdrakBrushAccentHover"])
                  && pill.Background is SolidColorBrush { Color: var pillGround } && pillGround.Equals(colors["IdrakSurface2"]),
                $"{theme}: hovered: chip {chip.BorderColor?.ToArgbHex()}, card {card.Stroke} {card.Background}, accent {accent.Background}, pill {pill.Background}");
            foreach (var view in new VisualElement[] { chip, card, accent, pill })
            {
                VisualStateManager.GoToState(view, VisualStateManager.CommonStates.Normal);
            }

            Check(Equals(chip.BorderColor, colors["IdrakLine2"]) && card.Background is SolidColorBrush { Color: var back } && back.Equals(colors["IdrakSurface"])
                  && SameGradient(accent.Background, colors["IdrakBrushAccent"]) && SameGradient(on.Background, colors["IdrakBrushPill"]) && Brush.IsNullOrEmpty(pill.Background),
                $"{theme}: back to their own: chip {chip.BorderColor?.ToArgbHex()}, card {card.Background}, accent {accent.Background}, pill {pill.Background}, on {on.Background}");
        }
    }

    // Two gradients with the same stops (a brush from a palette is the palette's, or its copy).
    private static bool SameGradient(Brush? brush, object expected) => brush is GradientBrush a && expected is GradientBrush b
        && a.GradientStops.Select(s => (s.Color.ToArgbHex(), s.Offset)).SequenceEqual(b.GradientStops.Select(s => (s.Color.ToArgbHex(), s.Offset)));
}
