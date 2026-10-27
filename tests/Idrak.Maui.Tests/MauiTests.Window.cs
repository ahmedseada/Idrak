// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Tests;

// The main window on a desktop (plan 34, phase W): its least size and its size and place kept between runs.
internal static partial class MauiTests
{
    // A store whose first write waits until released, counting the writes, so changes made during a write are seen to
    // make one write after it.
    private sealed class HeldStore : ISettingsStore
    {
        private readonly MemorySettingsStore _inner = new();
        private readonly ManualResetEventSlim _release = new(false);
        private int _writes;

        public int Writes => Volatile.Read(ref _writes);

        public void Release() => _release.Set();

        public string? Get(string key) => _inner.Get(key);

        public void Set(string key, string? value)
        {
            if (Interlocked.Increment(ref _writes) == 1)
            {
                _release.Wait(TimeSpan.FromSeconds(30));
            }

            _inner.Set(key, value);
        }
    }

    // The platform reporting the window's frame, as a desktop's window handler does when it is moved or resized.
    private static void Frame(Window window, double x, double y, double width, double height) => ((IWindow)window).FrameChanged(new Rect(x, y, width, height));

    private static void WindowPlaceKept()
    {
        var store = new MemorySettingsStore();
        var options = new IdrakMauiOptions();

        // The first run: nothing kept, so the system sizes and places the window; the least size is the options'.
        var placement = new KeptWindowPlacement(store, options);
        var window = new Window(new ContentPage());
        placement.Attach(window);
        Check(window.MinimumWidth == 360 && window.MinimumHeight == 480, $"least {window.MinimumWidth} x {window.MinimumHeight}");
        Check(placement.Kept is null && double.IsNaN(window.Width) && double.IsNaN(window.X), $"nothing kept: {placement.Kept}, {window.Width}");

        // Resized and moved by the user: each kept, off the UI thread.
        Frame(window, 120, 80, 1280, 800);
        placement.Writing.Wait();
        Check(placement.Kept == new WindowPlace(120, 80, 1280, 800), $"kept {placement.Kept}");
        Frame(window, 200, 90, 1280, 800);                                      // a move only: no SizeChanged
        placement.Writing.Wait();
        Check(placement.Kept == new WindowPlace(200, 90, 1280, 800), $"a move kept: {placement.Kept}");
        Check(store.Get(KeptWindowPlacement.Key) == "200,90,1280,800", $"as kept: {store.Get(KeptWindowPlacement.Key)}");

        // The next run opens the window where it was, at its size.
        var next = new Window(new ContentPage());
        new KeptWindowPlacement(store, options).Attach(next);
        Check(next.X == 200 && next.Y == 90 && next.Width == 1280 && next.Height == 800, $"restored {next.X},{next.Y} {next.Width} x {next.Height}");

        // A kept size below the least size opens at the least size; a place that does not read is none.
        store.Set(KeptWindowPlacement.Key, "10,10,100,100");
        var small = new Window(new ContentPage());
        new KeptWindowPlacement(store, options).Attach(small);
        Check(small.Width == 360 && small.Height == 480 && small.X == 10, $"raised to the least: {small.Width} x {small.Height}");
        Check(WindowPlace.Parse("a,b,c,d") is null && WindowPlace.Parse("1,2,3") is null && WindowPlace.Parse("1,2,3,4,5") is null
              && WindowPlace.Parse("1,2,0,4") is null && WindowPlace.Parse("1,2,NaN,4") is null && WindowPlace.Parse(null) is null
              && WindowPlace.Parse("-8.5,0,640.25,480") == new WindowPlace(-8.5, 0, 640.25, 480), "kept places read, or are none");

        // A place on no screen (minimised: the system's parking spot) is not kept.
        store.Set(KeptWindowPlacement.Key, "200,90,1280,800");
        var parked = new KeptWindowPlacement(store, options, onScreen: _ => false);
        var minimised = new Window(new ContentPage());
        parked.Attach(minimised);
        Frame(minimised, -32000, -32000, 160, 28);
        parked.Writing.Wait();
        Check(store.Get(KeptWindowPlacement.Key) == "200,90,1280,800", $"a minimised place was kept: {store.Get(KeptWindowPlacement.Key)}");

        // Keeping turned off: nothing restored, nothing written; the least size still holds.
        options.KeepWindowPlace = false;
        var off = new Window(new ContentPage());
        var unkept = new KeptWindowPlacement(store, options);
        unkept.Attach(off);
        Check(double.IsNaN(off.X) && double.IsNaN(off.Width) && off.MinimumWidth == 360, $"keeping off: restored at {off.X}, {off.Width}");
        Frame(off, 5, 5, 900, 700);
        unkept.Writing.Wait();
        Check(store.Get(KeptWindowPlacement.Key) == "200,90,1280,800", $"keeping off: written {store.Get(KeptWindowPlacement.Key)}");
        options.KeepWindowPlace = true;

        // Changes made while one is written make one write after it, with the latest place.
        var held = new HeldStore();
        var batched = new KeptWindowPlacement(held, options);
        var dragged = new Window(new ContentPage());
        batched.Attach(dragged);
        for (int i = 0; i < 200; i++)
        {
            Frame(dragged, i, i, 800 + i, 600 + i);
        }

        held.Release();
        batched.Writing.Wait();
        Check(held.Writes <= 2 && batched.Kept == new WindowPlace(199, 199, 999, 799), $"{held.Writes} writes for 200 changes, kept {batched.Kept}");

        // A phone's window is the system's: left as it is.
        var phone = new Window(new ContentPage());
        new SystemWindowPlacement().Attach(phone);
        Check(phone.MinimumWidth == 0 && double.IsNaN(phone.Width), "a phone's window untouched");
    }

    private static void WindowSettings()
    {
        // The three settings, in Layout and motion; a least size below the floor is refused, an empty one is the default.
        var keep = IdrakMauiSettings.Definitions.Single(s => s.Key == "window_keep");
        var width = IdrakMauiSettings.Definitions.Single(s => s.Key == "window_min_width_dp");
        var height = IdrakMauiSettings.Definitions.Single(s => s.Key == "window_min_height_dp");
        Check(keep.Group == "Layout and motion" && width.Group == "Layout and motion" && height.Group == "Layout and motion", $"{keep.Group}, {width.Group}, {height.Group}");
        var options = new IdrakMauiOptions();
        foreach (var (setting, text) in new[] { (width, "200"), (width, "-1"), (width, "wide"), (height, "100"), (keep, "maybe") })
        {
            bool refused = false;
            try
            {
                setting.Set(options, text);
            }
            catch (FormatException)
            {
                refused = true;
            }

            Check(refused, $"{setting.Key} took '{text}'");
        }

        Check(options.WindowMinimumWidth == 360 && options.WindowMinimumHeight == 480 && options.KeepWindowPlace, "a refused value changed nothing");
        width.Set(options, "640");
        height.Set(options, "400,5");                                           // a comma reads as the point
        keep.Set(options, "off");
        Check(options.WindowMinimumWidth == 640 && options.WindowMinimumHeight == 400.5 && !options.KeepWindowPlace,
            $"{options.WindowMinimumWidth} x {options.WindowMinimumHeight}, keep {options.KeepWindowPlace}");
        Check(KeptWindowPlacement.LeastWidth <= 360 && KeptWindowPlacement.LeastHeight <= 480, "the defaults are above the floor");

        // Kept and read back, and taken by the window.
        var settings = new IdrakMauiSettings(new MemorySettingsStore(), new MemorySecretStore());
        settings.SaveAsync(options).GetAwaiter().GetResult();
        var read = new IdrakMauiOptions();
        settings.LoadIntoAsync(read).GetAwaiter().GetResult();
        Check(read.WindowMinimumWidth == 640 && read.WindowMinimumHeight == 400.5 && !read.KeepWindowPlace, $"read {read.WindowMinimumWidth} x {read.WindowMinimumHeight}");
        var window = new Window(new ContentPage());
        new KeptWindowPlacement(new MemorySettingsStore(), read).Attach(window);
        Check(window.MinimumWidth == 640 && window.MinimumHeight == 400.5, $"the window's least {window.MinimumWidth} x {window.MinimumHeight}");
        width.Set(read, "");
        Check(read.WindowMinimumWidth == 360, "empty: the default");

        // Found by the settings' search.
        var startup = IdrakMauiStartup.Apply(new IdrakMauiOptions(), Temp("window-settings"));
        var page = new SettingsViewModel(new IdrakMauiOptions(), new IdrakMauiSettings(new MemorySettingsStore(), new MemorySecretStore()), startup);
        page.Query = "window";
        var shown = page.Entries.Where(e => e.IsShown).Select(e => e.Setting.Key).ToList();
        Check(shown.Contains("window_keep") && shown.Contains("window_min_width_dp") && shown.Contains("window_min_height_dp"), string.Join(", ", shown));
        page.Query = "least height";
        Check(page.Entries.Where(e => e.IsShown).Select(e => e.Setting.Key).SequenceEqual(["window_min_height_dp"]),
            string.Join(", ", page.Entries.Where(e => e.IsShown).Select(e => e.Setting.Key)));
    }
}
