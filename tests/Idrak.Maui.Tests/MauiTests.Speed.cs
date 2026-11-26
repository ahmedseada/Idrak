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
// kept chats read off the UI thread.
internal static partial class MauiTests
{
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

        // The frame clock: the library's timer clock when none is chosen.
        Check(FrameClocks.Use(null).Name == FrameClocks.Timer && FrameClocks.Current.Name == FrameClocks.Timer, FrameClocks.Current.Name);
        Check(IdrakMauiSettings.Definitions.Any(s => s.Key == "frame_clock"), "the clock is a setting");
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

        // An answer kept: its chat added at the top, the list changed in place (the day's group and the other chats kept as they were).
        var today = page.Conversations.Single();
        var first = today.Single(c => c.Title == "First");
        int resets = 0;
        page.Conversations.CollectionChanged += (_, e) => resets += e.Action == System.Collections.Specialized.NotifyCollectionChangedAction.Reset ? 1 : 0;
        today.CollectionChanged += (_, e) => resets += e.Action == System.Collections.Specialized.NotifyCollectionChangedAction.Reset ? 1 : 0;
        page.RefreshAsync().GetAwaiter().GetResult();
        page.SelectedModel = page.Models[0];
        page.Loading!.GetAwaiter().GetResult();
        page.Draft = "Third question";
        Check(page.SendAsync().GetAwaiter().GetResult(), page.Error);
        Check(ReferenceEquals(page.Conversations.Single(), today) && ReferenceEquals(today.Single(c => c.Title == "First"), first) && resets == 0
              && today.Select(c => c.Title).SequenceEqual(["Second", "Third question", "First"]) && today[1].IsCurrent,   // newest first: "Second" is a second ahead
            $"kept in place: {string.Join(", ", today.Select(c => c.Title))}, resets {resets}");
        page.ConversationQuery = "sec";
        Check(page.Conversations.SelectMany(g => g).Select(c => c.Title).SequenceEqual(["Second"]), "the search filters what was read");
    }
}
