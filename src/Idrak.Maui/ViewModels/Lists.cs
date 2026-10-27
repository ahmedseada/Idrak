// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.ObjectModel;

namespace Idrak.Maui.ViewModels;

// A list a page binds, filled again only when what it lists changed: a page shown again (a tab tapped) reads its lists
// anew, and when nothing changed its views are kept as they are instead of made again item by item.
internal static class Lists
{
    // Makes `target` hold `items`; when it holds the same ones (by `same`) in the same order it is left alone. True when it changed.
    public static bool Replace<T>(ObservableCollection<T> target, IReadOnlyList<T> items, Func<T, T, bool> same)
    {
        if (target.Count == items.Count)
        {
            bool equal = true;
            for (int i = 0; i < items.Count && equal; i++)
            {
                equal = same(target[i], items[i]);
            }

            if (equal)
            {
                return false;
            }
        }

        target.Clear();
        foreach (var item in items)
        {
            target.Add(item);
        }

        return true;
    }
}
