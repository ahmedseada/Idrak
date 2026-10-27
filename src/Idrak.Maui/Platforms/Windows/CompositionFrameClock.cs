// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak.Maui.Abstractions;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml.Media;

namespace Idrak.Maui.Platforms.Windows;

// The "display" frame clock on Windows: WinUI's CompositionTarget.Rendering, raised on the UI thread once for each frame
// the compositor draws, paced by the display, so a live view draws once a frame the display shows. The event is joined
// only while a subscription lives (the first subscription attaches it, the last one gone detaches it), so nothing runs
// between live drawings. WinUI may raise Rendering more than once in a frame; RenderingEventArgs.RenderingTime is the same
// for those, so a frame is given once. A rate asked for below the display's skips frames, with a quarter of the period as
// slack, as the Choreographer's clock does. CompositionTarget belongs to the UI thread: views subscribe there.
internal sealed class CompositionFrameClock : IFrameClock
{
    private readonly Lock _gate = new();
    private Subscription[] _subscriptions = [];
    private DispatcherQueue? _queue;
    private bool _attached;
    private TimeSpan _lastFrame = TimeSpan.MinValue;

    public string Name => FrameClocks.Display;

    public string Summary => "the display's own frames (WinUI's CompositionTarget.Rendering), on the UI thread";

    public bool IsAvailable(out string reason)
    {
        reason = "";
        return true;
    }

    public IDisposable Subscribe(Action frame, double framesPerSecond = 0)
    {
        ArgumentNullException.ThrowIfNull(frame);
        var queue = DispatcherQueue.GetForCurrentThread()
            ?? throw new InvalidOperationException("The display's frame clock is subscribed to from a thread that is not a UI thread; views subscribe on the UI thread.");
        var subscription = new Subscription(this, frame, framesPerSecond);
        lock (_gate)
        {
            _subscriptions = [.. _subscriptions, subscription];
            _queue ??= queue;
        }

        Attach();
        return subscription;
    }

    private void Attach()
    {
        if (!_attached)
        {
            _attached = true;
            _lastFrame = TimeSpan.MinValue;
            CompositionTarget.Rendering += OnRendering;
        }
    }

    private void Leave(Subscription subscription)
    {
        DispatcherQueue? queue;
        lock (_gate)
        {
            _subscriptions = [.. _subscriptions.Where(s => !ReferenceEquals(s, subscription))];
            if (_subscriptions.Length > 0)
            {
                return;
            }

            queue = _queue;
        }

        if (queue is null || queue.HasThreadAccess)
        {
            Detach();
        }
        else
        {
            queue.TryEnqueue(Detach);                                           // the event is the UI thread's to leave
        }
    }

    private void Detach()
    {
        lock (_gate)
        {
            if (_subscriptions.Length > 0)
            {
                return;                                                         // subscribed to again meanwhile
            }
        }

        if (_attached)
        {
            _attached = false;
            CompositionTarget.Rendering -= OnRendering;
        }
    }

    private void OnRendering(object? sender, object e)
    {
        if (e is RenderingEventArgs args)
        {
            if (args.RenderingTime == _lastFrame)
            {
                return;                                                         // raised again for the same frame
            }

            _lastFrame = args.RenderingTime;
        }

        Subscription[] subscriptions;
        lock (_gate)
        {
            subscriptions = _subscriptions;
        }

        long now = Stopwatch.GetTimestamp();
        foreach (var subscription in subscriptions)
        {
            subscription.Tick(now);
        }
    }

    private sealed class Subscription(CompositionFrameClock clock, Action frame, double framesPerSecond) : IDisposable
    {
        // A quarter of the period as slack, so a rate that divides the display's is met on every frame it asks for.
        private readonly long _period = framesPerSecond > 0 ? (long)(0.75 * Stopwatch.Frequency / framesPerSecond) : 0;
        private long _last = long.MinValue;
        private bool _stopped;

        public void Tick(long now)
        {
            if (_stopped)
            {
                return;
            }

            if (_period == 0 || _last == long.MinValue || now - _last >= _period)
            {
                _last = now;
                frame();
            }
        }

        public void Dispose()
        {
            if (!_stopped)
            {
                _stopped = true;
                clock.Leave(this);
            }
        }
    }
}
