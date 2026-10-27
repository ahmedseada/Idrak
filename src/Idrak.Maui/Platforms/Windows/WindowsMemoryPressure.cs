// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using Idrak.Maui.Abstractions;
using Microsoft.Win32.SafeHandles;

namespace Idrak.Maui.Platforms.Windows;

// Windows' memory-pressure signal: the system's memory resource notifications (kernel32's
// CreateMemoryResourceNotification), two objects the system keeps signalled while the machine's available physical memory
// is low (LowMemoryResourceNotification) or plentiful (HighMemoryResourceNotification), at thresholds the system sets from
// the machine's memory. A background thread waits on them: low memory raises Low (the contract's "running low while the
// app runs", Android's TRIM_MEMORY_RUNNING_LOW); still low once that signal's handlers have given memory back, it raises
// Critical; then it waits for plenty again before the next signal, so one low spell is one signal (the low object stays
// signalled while it lasts). Raised on that thread. Where the notifications cannot be made, the manual source stands in.
internal sealed partial class WindowsMemoryPressure : IMemoryPressureSource, IDisposable
{
    private const int LowMemoryResourceNotification = 0, HighMemoryResourceNotification = 1;

    private readonly Notification _low, _high;
    private readonly ManualResetEvent _stop = new(false);
    private int _disposed;

    private WindowsMemoryPressure(SafeWaitHandle low, SafeWaitHandle high)
    {
        _low = new Notification(low);
        _high = new Notification(high);
        new Thread(Watch) { IsBackground = true, Name = "Idrak memory pressure" }.Start();
    }

    public event EventHandler<MemoryPressureSignal>? Pressure;

    // The system's notifications, or the manual source where Windows does not make them.
    public static IMemoryPressureSource Create()
    {
        var low = new SafeWaitHandle(CreateMemoryResourceNotification(LowMemoryResourceNotification), ownsHandle: true);
        var high = new SafeWaitHandle(CreateMemoryResourceNotification(HighMemoryResourceNotification), ownsHandle: true);
        if (low.IsInvalid || high.IsInvalid)
        {
            low.Dispose();
            high.Dispose();
            return new ManualMemoryPressure();
        }

        return new WindowsMemoryPressure(low, high);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _stop.Set();                                                        // the thread ends and closes the handles
        }
    }

    private void Watch()
    {
        try
        {
            WaitHandle[] low = [_stop, _low], high = [_stop, _high];
            while (WaitHandle.WaitAny(low) != 0)
            {
                Pressure?.Invoke(this, new MemoryPressureSignal(MemoryPressureLevel.Low, "LowMemoryResourceNotification"));
                if (_low.WaitOne(0))
                {
                    Pressure?.Invoke(this, new MemoryPressureSignal(MemoryPressureLevel.Critical, "LowMemoryResourceNotification (still low)"));
                }

                if (WaitHandle.WaitAny(high) == 0)
                {
                    break;
                }
            }
        }
        finally
        {
            _low.Dispose();
            _high.Dispose();
            _stop.Dispose();
        }
    }

    [LibraryImport("kernel32", SetLastError = true)]
    private static partial nint CreateMemoryResourceNotification(int notificationType);

    // A memory resource notification as a handle .NET waits on.
    private sealed class Notification : WaitHandle
    {
        public Notification(SafeWaitHandle handle) => SafeWaitHandle = handle;
    }
}
