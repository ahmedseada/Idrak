// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Android.Views;
using Idrak.Maui.Abstractions;

namespace Idrak.Maui.Platforms.Android;

// The "display" frame clock: android.view.Choreographer, the same frame callback the platform draws its views on, so a live
// view draws once a frame the display shows, in step with it (never twice a frame, never between frames). A subscription
// posts its callback for the next frame and posts it again from it while it lives; a rate asked for below the display's
// skips frames. Choreographer.getInstance belongs to the calling thread's looper: views subscribe on the UI thread.
internal sealed class ChoreographerFrameClock : IFrameClock
{
    public string Name => FrameClocks.Display;

    public string Summary => "the display's own frames (Android's Choreographer), on the UI thread";

    public bool IsAvailable(out string reason)
    {
        reason = "";
        return true;
    }

    public IDisposable Subscribe(Action frame, double framesPerSecond = 0)
    {
        ArgumentNullException.ThrowIfNull(frame);
        if (global::Android.OS.Looper.MyLooper() is null)
        {
            throw new InvalidOperationException("The display's frame clock is subscribed to from a thread without a looper; views subscribe on the UI thread.");
        }

        var subscription = new Subscription(Choreographer.Instance!, frame, framesPerSecond);
        subscription.Start();
        return subscription;
    }

    private sealed class Subscription(Choreographer choreographer, Action frame, double framesPerSecond) : Java.Lang.Object, Choreographer.IFrameCallback
    {
        // A quarter of the period as slack, so a rate that divides the display's is met on every frame it asks for.
        private readonly long _period = framesPerSecond > 0 ? (long)(0.75e9 / framesPerSecond) : 0;
        private long _last = long.MinValue;
        private bool _stopped;

        public void Start() => choreographer.PostFrameCallback(this);

        public void DoFrame(long frameTimeNanos)
        {
            if (_stopped)
            {
                return;
            }

            if (_period == 0 || _last == long.MinValue || frameTimeNanos - _last >= _period)
            {
                _last = frameTimeNanos;
                frame();
            }

            if (!_stopped)
            {
                choreographer.PostFrameCallback(this);
            }
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing && !_stopped)
            {
                _stopped = true;
                choreographer.RemoveFrameCallback(this);
            }

            base.Dispose(disposing);
        }
    }
}
