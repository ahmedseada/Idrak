// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Windows.Input;

namespace Idrak.Maui.Controls;

/// <summary>
/// What a desktop keyboard's keys do in a text field, set on the field in XAML (<c>ctl:DesktopKeys.EnterCommand</c>): Enter
/// runs <see cref="EnterCommandProperty"/> (Shift+Enter still makes a new line), the Up and Down arrows run
/// <see cref="UpCommandProperty"/> and <see cref="DownCommandProperty"/> (the caret then at the end of the text). A key runs
/// its command only while the command can run (<see cref="ICommand.CanExecute"/>); otherwise the field takes the key as it
/// always does. On Windows the field's own text box reads the keys before it acts on them (Platforms/Windows,
/// <c>WindowsKeyInput</c>); a phone's soft keyboard is not changed: nothing reads these there.
/// </summary>
public static class DesktopKeys
{
    /// <summary>What Enter runs in the field (null: Enter is the field's own: a new line, or Completed).</summary>
    public static readonly BindableProperty EnterCommandProperty =
        BindableProperty.CreateAttached("EnterCommand", typeof(ICommand), typeof(DesktopKeys), null, propertyChanged: Watch);

    /// <summary>What the Up arrow runs in the field (null: the field's own).</summary>
    public static readonly BindableProperty UpCommandProperty =
        BindableProperty.CreateAttached("UpCommand", typeof(ICommand), typeof(DesktopKeys), null, propertyChanged: Watch);

    /// <summary>What the Down arrow runs in the field (null: the field's own).</summary>
    public static readonly BindableProperty DownCommandProperty =
        BindableProperty.CreateAttached("DownCommand", typeof(ICommand), typeof(DesktopKeys), null, propertyChanged: Watch);

    /// <summary>What Enter runs in <paramref name="view"/>.</summary>
    public static ICommand? GetEnterCommand(BindableObject view) => (ICommand?)(view ?? throw new ArgumentNullException(nameof(view))).GetValue(EnterCommandProperty);

    /// <summary>Sets what Enter runs in <paramref name="view"/>.</summary>
    public static void SetEnterCommand(BindableObject view, ICommand? value) => (view ?? throw new ArgumentNullException(nameof(view))).SetValue(EnterCommandProperty, value);

    /// <summary>What the Up arrow runs in <paramref name="view"/>.</summary>
    public static ICommand? GetUpCommand(BindableObject view) => (ICommand?)(view ?? throw new ArgumentNullException(nameof(view))).GetValue(UpCommandProperty);

    /// <summary>Sets what the Up arrow runs in <paramref name="view"/>.</summary>
    public static void SetUpCommand(BindableObject view, ICommand? value) => (view ?? throw new ArgumentNullException(nameof(view))).SetValue(UpCommandProperty, value);

    /// <summary>What the Down arrow runs in <paramref name="view"/>.</summary>
    public static ICommand? GetDownCommand(BindableObject view) => (ICommand?)(view ?? throw new ArgumentNullException(nameof(view))).GetValue(DownCommandProperty);

    /// <summary>Sets what the Down arrow runs in <paramref name="view"/>.</summary>
    public static void SetDownCommand(BindableObject view, ICommand? value) => (view ?? throw new ArgumentNullException(nameof(view))).SetValue(DownCommandProperty, value);

    /// <summary>
    /// Runs what <paramref name="key"/> does in <paramref name="view"/> (what a platform's key handler calls, and tests):
    /// true when a command ran, so the field must not take the key; false leaves the key to the field. Enter with Shift is
    /// the field's (a new line).
    /// </summary>
    public static bool Press(BindableObject view, DesktopKey key, bool shift = false)
    {
        ArgumentNullException.ThrowIfNull(view);
        var command = key switch
        {
            DesktopKey.Enter when !shift => GetEnterCommand(view),
            DesktopKey.Up => GetUpCommand(view),
            DesktopKey.Down => GetDownCommand(view),
            _ => null,
        };
        if (command is null || !command.CanExecute(null))
        {
            return false;
        }

        command.Execute(null);
        if (key is DesktopKey.Up or DesktopKey.Down && view is InputView { Text: { } text } field)
        {
            field.CursorPosition = text.Length;                                   // a line recalled is edited from its end
            field.SelectionLength = 0;
        }

        return true;
    }

    private static void Watch(BindableObject view, object? before, object? after)
    {
#if WINDOWS
        if (view is VisualElement element)
        {
            Platforms.Windows.WindowsKeyInput.Watch(element);
        }
#endif
    }
}

/// <summary>A key of a desktop keyboard that a text field's <see cref="DesktopKeys"/> commands answer.</summary>
public enum DesktopKey
{
    /// <summary>Enter (Return).</summary>
    Enter,

    /// <summary>The Up arrow.</summary>
    Up,

    /// <summary>The Down arrow.</summary>
    Down,
}
