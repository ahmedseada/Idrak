// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Maui.Abstractions;
using Idrak.Maui.Controls;
using Idrak.Maui.Navigation;
using Idrak.Maui.Theme;
using Idrak.Maui.ViewModels;
using Idrak.Maui.Views;
using Microsoft.Extensions.DependencyInjection;

namespace Idrak.Maui.Tests;

// The owner's lag notes (October 2026): the live levels handed from the capture thread and drawn once a display frame, the
// settings' search and restart, the host keeping its sections' views, the kept chats read off the UI thread.
internal static partial class MauiTests
{
    private sealed class FakeRestarter(bool can) : IAppRestarter
    {
        public int Restarts { get; private set; }

        public bool CanRestart => can;

        public string Unavailable => can ? "" : "not here";

        public void Restart() => Restarts++;
    }

    // A second of a tone that swells and fades, at 16 kHz.
    private static float[] Swell(int rate = 16_000)
    {
        var samples = new float[rate];
        for (int i = 0; i < samples.Length; i++)
        {
            double t = (double)i / rate;
            samples[i] = (float)(Math.Sin(2 * Math.PI * 440 * t) * (0.02 + 0.9 * Math.Sin(Math.PI * t)));
        }

        return samples;
    }

    private static float[] ReadAll(LiveLevels levels, long now = long.MaxValue)
    {
        long cursor = 0;
        var into = new float[1024];
        int n = levels.Read(ref cursor, into, now);
        return into[..n];
    }

    private static void LiveLevelsPaced()
    {
        var sound = Swell();
        var whole = new LiveLevels(windowSeconds: 0.05);
        whole.Add(sound, 16_000);
        var cut = new LiveLevels(windowSeconds: 0.05);
        int at = 0;
        foreach (int size in Enumerable.Repeat((int[])[37, 160, 513, 1, 999, 2048], 100).SelectMany(s => s))
        {
            if (at >= sound.Length)
            {
                break;
            }

            int take = Math.Min(size, sound.Length - at);
            cut.Add(sound.AsSpan(at, take), 16_000);
            at += take;
        }

        float[] a = ReadAll(whole), b = ReadAll(cut);
        Check(a.Length == 20 && b.Length == 20 && a.Zip(b).All(p => Math.Abs(p.First - p.Second) < 1e-4), $"{a.Length} and {b.Length} levels: one a window, however the blocks are cut");
        Check(a[10] > a[0] + 0.3 && a[10] > a[19] + 0.3 && a.All(l => l is >= 0 and <= 1), $"the swell: {a[0]:F2} {a[10]:F2} {a[19]:F2}");
        Check(Math.Abs(whole.Latest - a[^1]) < 1e-6, "the latest level");

        // Paced: each level shown when its sound plays, not when it was made.
        var played = new LiveLevels(windowSeconds: 0.05);
        long start = Stopwatch.GetTimestamp();
        played.Pace(start);
        played.Add(sound, 16_000);
        long cursor = 0;
        var into = new float[64];
        int early = played.Read(ref cursor, into, start + (long)(0.26 * Stopwatch.Frequency));
        int rest = played.Read(ref cursor, into, start + Stopwatch.Frequency);
        Check(early == 6 && rest == 14, $"paced: {early} by 0.26 s, {rest} more by 1 s");

        // A reset starts the readers again; a reader with a short buffer takes the newest.
        int generation = whole.Generation;
        whole.Reset();
        Check(whole.Generation == generation + 1 && whole.Written == 0 && ReadAll(whole).Length == 0, "a reset forgets the levels");
        whole.Add(sound, 16_000);
        long c = 0;
        var three = new float[3];
        Check(whole.Read(ref c, three, long.MaxValue) == 3 && Math.Abs(three[2] - a[^1]) < 1e-6 && c == 20, "a short buffer takes the newest");

        // The waveform takes a frame's new levels at once (one redraw for all), and starts afresh when its source does.
        var wave = new WaveformView(8) { Source = whole, IsLive = true };
        Check(wave.TakeFrame() && wave.Count == 8 && wave.Levels().Zip(a[^8..]).All(p => Math.Abs(p.First - p.Second) < 1e-6),
            $"the waveform: {string.Join(", ", wave.Levels().Select(l => l.ToString("F2", System.Globalization.CultureInfo.InvariantCulture)))}");
        Check(!wave.TakeFrame(), "nothing new: no redraw");
        whole.Reset();
        Check(wave.TakeFrame() && wave.Count == 0, "a new recording clears the bars");
        wave.Level = 0.5;
        Check(wave.Count == 0, "with a source, a bound level is not pushed");
        Check(!wave.IsTicking, "off screen (no handler): no frame clock is joined");

        // The meter eases to the latest level over its smoothing, and empties when not live.
        var feed = new LiveLevels();
        feed.AddLevel(1);
        var meter = new MeterBar { Source = feed, IsLive = true, Smoothing = 0.08 };
        Check(meter.TakeFrame(TimeSpan.FromMilliseconds(16)) && meter.Shown is > 0 and < 0.5, $"eased: {meter.Shown:F3}");
        for (int i = 0; i < 40; i++)
        {
            meter.TakeFrame(TimeSpan.FromMilliseconds(16));
        }

        Check(meter.Shown == 1, $"reached: {meter.Shown:F3}");
        meter.IsLive = false;
        Check(meter.Shown == 0, "not live: empty");

        // The frame clocks: the timer everywhere (the display's own on Android), chosen by name, an unknown one refused.
        Check(FrameClocks.Names.Contains(FrameClocks.Timer) && FrameClocks.Use("timer").Name == FrameClocks.Timer && FrameClocks.Current.Name == FrameClocks.Timer,
            string.Join(", ", FrameClocks.Names));
        FrameClocks.Use(null);
        bool refused = false;
        try
        {
            FrameClocks.Use("nowhere");
        }
        catch (NotRegisteredException)
        {
            refused = true;
        }

        Check(refused, "an unknown frame clock was taken");
        Check(IdrakMauiSettings.Definitions.Single(s => s.Key == "frame_clock").Choices.Contains(FrameClocks.Timer), "the clock is a setting");
    }

    private static void SettingsSearched()
    {
        string storage = Temp("settings-search");
        var options = new IdrakMauiOptions();
        var startup = IdrakMauiStartup.Apply(options, storage);
        var restarter = new FakeRestarter(true);
        var page = new SettingsViewModel(options, new IdrakMauiSettings(new MemorySettingsStore(), new MemorySecretStore()), startup, restarter: restarter);
        string Shown() => string.Join(", ", page.Entries.Where(e => e.IsShown).Select(e => e.Setting.Key));

        page.Query = "temperature";
        Check(page.Entries.Where(e => e.IsShown).Select(e => e.Setting.Key).SequenceEqual(["chat_temperature"]) && page.Groups.Single(g => g.IsShown).Title == "Chat"
              && !page.ShowFeatures && !page.ShowStorage && !page.ShowApp && !page.HasNoMatch && page.IsSearching, Shown());
        page.Query = "CHAT";
        Check(page.Entries.Where(e => e.IsShown).All(e => e.Group == "Chat") && page.Groups.Single(g => g.Title == "Chat").All(e => e.IsShown), Shown());
        page.Query = "int8 weights";
        Check(page.Entries.Where(e => e.IsShown).Select(e => e.Setting.Key).SequenceEqual(["chat_weights"]), $"every word, a choice's too: {Shown()}");
        options.MaxCpuThreads = 3;
        page.Entries.Single(e => e.Setting.Key == "cpu_threads").Text = "3";
        page.Query = "3";
        Check(page.Entries.Single(e => e.Setting.Key == "cpu_threads").IsShown, $"by its value: {Shown()}");
        page.Query = "transcribe";
        Check(page.ShowFeatures && page.Features.Where(f => f.IsShown).Select(f => f.Name).Contains(MobileFeatures.Transcribe), "a lab shown, by its title");
        page.Query = "restart";
        Check(page.ShowApp, "the app's restart, by its words");
        page.Query = "zzzz qqq";
        Check(page.HasNoMatch && page.Groups.All(g => !g.IsShown) && !page.ShowFeatures && !page.ShowStorage && !page.ShowApp, "no setting matches");
        page.Query = "";
        Check(page.Entries.All(e => e.IsShown) && page.Groups.All(g => g.IsShown) && page.ShowFeatures && page.ShowStorage && page.ShowApp && !page.HasNoMatch, "all again");

        // The search is quick: a thousand queries over every setting.
        var clock = Stopwatch.StartNew();
        for (int i = 0; i < 1000; i++)
        {
            page.Query = i % 2 == 0 ? "gpu memory" : "chat";
        }

        Check(clock.Elapsed < TimeSpan.FromSeconds(5), $"a thousand searches took {clock.Elapsed.TotalMilliseconds:F0} ms");
        page.Query = "";

        // A saved change that waits for the next start is marked and the restart offered; the restart asks first.
        page.Entries.Single(e => e.Setting.Key == "gpu_chunk_mb").Text = "64";
        Check(page.HasUnsavedChanges, "a change not saved");
        Check(page.SaveAsync().GetAwaiter().GetResult() && page.IsRestartPending && page.Entries.Single(e => e.Setting.Key == "gpu_chunk_mb").WaitsForRestart
              && page.Entries.Count(e => e.WaitsForRestart) == 1 && page.Message.Contains("Restart the app for", StringComparison.Ordinal), page.Message);
        Check(!page.RestartAsync().GetAwaiter().GetResult() && restarter.Restarts == 0, "no one asked: not restarted");
        page.Confirm = (_, _) => Task.FromResult(false);
        Check(!page.RestartAsync().GetAwaiter().GetResult() && restarter.Restarts == 0, "declined: not restarted");
        string asked = "";
        page.Confirm = (_, message) =>
        {
            asked = message;
            return Task.FromResult(true);
        };
        Check(page.RestartAsync().GetAwaiter().GetResult() && restarter.Restarts == 1 && !asked.Contains("not saved", StringComparison.Ordinal), asked);
        var none = new SettingsViewModel(options, new IdrakMauiSettings(new MemorySettingsStore(), new MemorySecretStore()), startup, restarter: new UnavailableAppRestarter());
        Check(!none.CanRestart && !none.RestartAsync().GetAwaiter().GetResult() && none.Message.Contains("by hand", StringComparison.Ordinal), none.Message);

        // The page makes each setting's own control only: a picker a picker, a slider a slider, a field a field.
        using var services = AppServices();
        var view = new SettingsView(page);
        Themed(view, AppTheme.Dark);
        var tree = Tree(view).ToList();
        int pickers = tree.OfType<Picker>().Count(), sliders = tree.OfType<Slider>().Count(), entries = tree.OfType<Entry>().Count(), all = tree.Count;
        int wantPickers = page.Entries.Count(e => e.IsPicker), wantSliders = page.Entries.Count(e => e.IsSlider), wantFields = page.Entries.Count(e => e.IsField);
        Check(pickers == wantPickers && sliders == wantSliders && entries == wantFields + 1, $"{pickers} pickers ({wantPickers}), {sliders} sliders ({wantSliders}), {entries} fields ({wantFields} and the search)");
        Console.WriteLine($"    settings page: {all} elements for {page.Entries.Count} settings: {pickers} pickers, {sliders} sliders, {entries} fields "
            + $"(every setting carried all four before: {page.Entries.Count} of each)");
    }

    private static void HostKeepsSections()
    {
        var options = new IdrakMauiOptions { Device = "cpu" };
        using var services = AppServices(options);
        var host = new AppHostPage(services);
        var body = host.FindByName<Grid>("Body");
        Check(host.Views.Count == 1 && body.Children.Count == 1 && host.Current is HomeView, "Home alone at first");
        host.Prebuild();
        Check(host.Views.Keys.Order(StringComparer.Ordinal).SequenceEqual([AppSections.Chat, AppSections.Home, AppSections.Vision, AppSections.Voice]) && body.Children.Count == 4
              && body.Children.Count(c => ((View)c).IsVisible) == 1 && ReferenceEquals(body.Children.Single(c => ((View)c).IsVisible), host.Current),
            $"made ahead: {string.Join(", ", host.Views.Keys)}; {body.Children.Count(c => ((View)c).IsVisible)} visible");
        var voice = (SegmentedSectionView)host.Views[AppSections.Voice];
        Check(voice.FindByName<Grid>("Body").Children.Count == 3, $"Voice's labs made ahead: {voice.FindByName<Grid>("Body").Children.Count}");
        Check(host.FindByName<ContentView>("SheetBody").Content is LabsView, "the sheet made ahead");
        var chat = host.Views[AppSections.Chat];
        host.OpenSectionAsync(AppSections.Chat).GetAwaiter().GetResult();
        Check(ReferenceEquals(host.Current, chat) && chat.IsVisible && !host.Views[AppSections.Home].IsVisible && body.Children.Count == 4, "a tab shows its view made ahead; nothing made again");
        host.OpenSectionAsync(AppSections.Home).GetAwaiter().GetResult();
        Check(host.Current is HomeView && !chat.IsVisible && host.ViewModel.ShowsSame(), "back: the same views, the tabs the same");
        options.DisabledFeatures.Add(MobileFeatures.Ocr);
        options.DisabledFeatures.Add(MobileFeatures.Image);
        Check(!host.ViewModel.ShowsSame(), "a section turned off: the tabs listed again");
        options.DisabledFeatures.Clear();
    }

    private static void ChatReadsKeptOffThread()
    {
        string folder = ChatFolder("chat-kept-reload");
        var kept = new MemoryConversationStore();
        var now = DateTimeOffset.UtcNow;
        kept.Save(new ChatConversation(ConversationText.NewId(now), "First", now, null, [new ChatMessage("user", "one"), new ChatMessage("assistant", "two")]));
        var page = new ChatViewModel(new ListedStore(Stored(folder)), new OneChat(new ScriptedChat(3, "ab", Stats(3, 1))), new IdrakMauiOptions(), new InlineDispatcher(),
            conversations: kept);
        Check(page.Conversations.Count == 0, "nothing read when the chat is made");
        page.ReloadConversationsAsync().GetAwaiter().GetResult();
        Check(page.Conversations.SelectMany(g => g).Select(c => c.Title).SequenceEqual(["First"]), "read on a worker thread and listed");
        kept.Save(new ChatConversation(ConversationText.NewId(now.AddSeconds(1)), "Second", now.AddSeconds(1), null, [new ChatMessage("user", "three")]));
        page.ReloadConversationsAsync().GetAwaiter().GetResult();
        Check(page.Conversations.SelectMany(g => g).Select(c => c.Title).SequenceEqual(["Second", "First"]), string.Join(", ", page.Conversations.SelectMany(g => g).Select(c => c.Title)));
        page.ConversationQuery = "sec";
        Check(page.Conversations.SelectMany(g => g).Select(c => c.Title).SequenceEqual(["Second"]), "the search filters what was read");
    }
}
