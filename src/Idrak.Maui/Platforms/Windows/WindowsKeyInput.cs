// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using Idrak.Maui.Controls;
using Idrak.Maui.Navigation;
using CoreVirtualKeyStates = global::Windows.UI.Core.CoreVirtualKeyStates;
using InputKeyboardSource = Microsoft.UI.Input.InputKeyboardSource;
using KeyEventHandler = Microsoft.UI.Xaml.Input.KeyEventHandler;
using KeyRoutedEventArgs = Microsoft.UI.Xaml.Input.KeyRoutedEventArgs;
using UIElement = Microsoft.UI.Xaml.UIElement;
using VirtualKey = global::Windows.System.VirtualKey;
using WinUIWindow = Microsoft.UI.Xaml.Window;

namespace Idrak.Maui.Platforms.Windows;

// A desktop keyboard on Windows. In a text field that sets DesktopKeys' commands, the field's own WinUI text box is read
// before it acts on a key (PreviewKeyDown, which tunnels ahead of the TextBox's KeyDown): Enter without Shift runs the
// field's Enter command instead of a new line (Shift+Enter is left to the text box: a new line in a multi-line Editor), the
// arrows its Up and Down commands; a key whose command cannot run now is the text box's, as before. The field is watched
// once and hooked again when MAUI gives it a new platform view (HandlerChanged). At the window, Esc closes the sheet on
// top (AppHostPage.CloseTopSheet), whatever has the focus (the window's root sees handled keys too).
internal static class WindowsKeyInput
{
    private static readonly ConditionalWeakTable<VisualElement, object> Watched = new();
    private static readonly ConditionalWeakTable<UIElement, object> Hooked = new();

    /// <summary>Reads <paramref name="view"/>'s keys from now on (once; the attached commands are read at each key).</summary>
    public static void Watch(VisualElement view)
    {
        ArgumentNullException.ThrowIfNull(view);
        if (!Watched.TryGetValue(view, out _))
        {
            Watched.Add(view, new object());
            view.HandlerChanged += (_, _) => Hook(view);
        }

        Hook(view);
    }

    /// <summary>Esc closes <paramref name="host"/>'s sheet on top, from anywhere in its window (hooked once the window is shown).</summary>
    public static void WatchEscape(AppHostPage host)
    {
        ArgumentNullException.ThrowIfNull(host);
        host.Loaded += (_, _) =>
        {
            if (host.Window?.Handler?.PlatformView is WinUIWindow { Content: UIElement root } && !Hooked.TryGetValue(root, out _))
            {
                Hooked.Add(root, new object());
                root.AddHandler(UIElement.KeyDownEvent, new KeyEventHandler((_, e) =>
                {
                    if (e.Key == VirtualKey.Escape && host.CloseTopSheet())
                    {
                        e.Handled = true;
                    }
                }), true);                                                       // handled keys too: a text box with the focus may take Esc
            }
        };
    }

    private static void Hook(VisualElement view)
    {
        if (view.Handler?.PlatformView is not UIElement element || Hooked.TryGetValue(element, out _))
        {
            return;
        }

        Hooked.Add(element, new object());
        element.PreviewKeyDown += (_, e) => OnKey(view, e);
    }

    private static void OnKey(VisualElement view, KeyRoutedEventArgs e)
    {
        DesktopKey? key = e.Key switch
        {
            VirtualKey.Enter => DesktopKey.Enter,
            VirtualKey.Up => DesktopKey.Up,
            VirtualKey.Down => DesktopKey.Down,
            _ => null,
        };
        if (key is { } pressed && DesktopKeys.Press(view, pressed, shift: IsDown(VirtualKey.Shift)))
        {
            e.Handled = true;                                                    // the text box does not take it (no new line)
        }
    }

    private static bool IsDown(VirtualKey key) => (InputKeyboardSource.GetKeyStateForCurrentThread(key) & CoreVirtualKeyStates.Down) == CoreVirtualKeyStates.Down;
}
