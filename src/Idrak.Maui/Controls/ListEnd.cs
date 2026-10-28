// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Maui.Controls;

/// <summary>
/// A list's very end, where its last item is taller than the list: <c>ScrollTo</c> with <see cref="ScrollToPosition.End"/>
/// lines such an item's top up on Windows (it scrolls up, not to the end) and
/// reaches its old end on Android while it grows. So the platform's own scrolling is asked for its whole length.
/// </summary>
internal static class ListEnd
{
    /// <summary>Scrolls <paramref name="list"/> to the end of its last item (<paramref name="last"/>, its index).</summary>
    public static void ScrollTo(CollectionView list, int last)
    {
        ArgumentNullException.ThrowIfNull(list);
        if (last < 0 || list.Handler is null)
        {
            return;
        }
#if ANDROID
        if (list.Handler.PlatformView is AndroidX.RecyclerView.Widget.RecyclerView recycler)
        {
            recycler.ScrollToPosition(last);                                    // the last item laid out, then its whole length:
            recycler.Post(() => recycler.ScrollBy(0, 1 << 24));                 // the list stops at its end
            return;
        }
#elif WINDOWS
        if (Viewer(list) is { } viewer)
        {
            viewer.UpdateLayout();
            viewer.ChangeView(null, viewer.ScrollableHeight, null, true);
            return;
        }
#endif
        list.ScrollTo(last, position: ScrollToPosition.End, animate: false);
    }

    /// <summary>Whether <paramref name="list"/> shows its end (null where the platform does not say).</summary>
    public static bool? IsAtEnd(CollectionView list)
    {
        ArgumentNullException.ThrowIfNull(list);
#if ANDROID
        if (list.Handler?.PlatformView is AndroidX.RecyclerView.Widget.RecyclerView recycler)
        {
            return !recycler.CanScrollVertically(1);
        }
#elif WINDOWS
        if (Viewer(list) is { } viewer)
        {
            return viewer.VerticalOffset >= viewer.ScrollableHeight - 2;
        }
#endif
        return null;
    }
#if WINDOWS
    // The scroll viewer inside the list's platform view (WinUI's list draws its items in one), found once and kept.
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<CollectionView, Microsoft.UI.Xaml.Controls.ScrollViewer> Viewers = new();

    private static Microsoft.UI.Xaml.Controls.ScrollViewer? Viewer(CollectionView list)
    {
        if (Viewers.TryGetValue(list, out var kept))
        {
            return kept;
        }

        if (list.Handler?.PlatformView is not Microsoft.UI.Xaml.DependencyObject root || Descendant(root) is not { } found)
        {
            return null;
        }

        Viewers.AddOrUpdate(list, found);
        return found;
    }

    private static Microsoft.UI.Xaml.Controls.ScrollViewer? Descendant(Microsoft.UI.Xaml.DependencyObject parent)
    {
        if (parent is Microsoft.UI.Xaml.Controls.ScrollViewer viewer)
        {
            return viewer;
        }

        int count = Microsoft.UI.Xaml.Media.VisualTreeHelper.GetChildrenCount(parent);
        for (int i = 0; i < count; i++)
        {
            if (Descendant(Microsoft.UI.Xaml.Media.VisualTreeHelper.GetChild(parent, i)) is { } found)
            {
                return found;
            }
        }

        return null;
    }
#endif
}
