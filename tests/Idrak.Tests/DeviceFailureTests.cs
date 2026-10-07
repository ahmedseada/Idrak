// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Reflection;
using System.Text;
using System.Text.Json;
using Idrak.Abstraction.Devices.Cpu;
using Idrak.Abstraction.Operations;

// Plan 10, decision 13: a GPU kernel that fails throws (a DeviceException, reported to telemetry with a hint naming the
// switch); with Backend.RetryOnHost the dispatcher runs the operation again on the CPU and reports that instead.
internal static partial class Tests
{
    // Plain arrays standing for device memory; Softmax throws a DeviceException while Faulty is set. Each test uses a kind
    // of its own, so the hint (which reads every live device of the kind) does not depend on the other tests' devices.
    private sealed class FaultyBackend(string kind) : Backend
    {
        public bool Faulty;

        public override BackendCapabilities Capabilities => CpuBackend.Instance.Capabilities;

        public override string Kind => kind;

        public override string Name => "faulty (Softmax fails on request)";

        public override Storage Allocate(int length, bool zeroed) => new MinimalStorage(this, new float[length], length);

        public override void Return(Storage storage)
        {
        }

        protected override void Detach(Storage storage) => ((MinimalStorage)storage).Data = null!;

        protected override void Attach(Storage storage, Storage fresh) => ((MinimalStorage)storage).Data = ((MinimalStorage)fresh).Data;

        public override MemoryUsage GetMemoryUsage() => new(0, 0, null);

        public override void ReleaseCachedMemory()
        {
        }

        public override void Upload(ReadOnlySpan<float> source, Storage destination) => source.CopyTo(((MinimalStorage)destination).Data);

        public override void Download(Storage source, Span<float> destination) => ((MinimalStorage)source).Data.AsSpan(0, destination.Length).CopyTo(destination);

        public override void DownloadRange(Storage source, int offset, Span<float> destination) =>
            ((MinimalStorage)source).Data.AsSpan(offset, destination.Length).CopyTo(destination);

        public override void Synchronize()
        {
        }

        public override void SoftmaxKernel(Storage x, Storage y, int rows, int cols, bool log)
        {
            if (Faulty)
            {
                ((MinimalStorage)y).Data.AsSpan().Fill(float.NaN);           // a partial write the retry must overwrite
                throw new DeviceException(Kind, "injected fault in Softmax");
            }

            base.SoftmaxKernel(x, y, rows, cols, log);
        }
    }

    // Keeps the DeviceFailed events of one kind of device.
    private sealed class DeviceFailures(string kind) : ITelemetryHook
    {
        private readonly List<DeviceFailed> _events = [];

        public TelemetryLevel Levels => TelemetryLevel.Devices;

        public List<DeviceFailed> Events
        {
            get
            {
                lock (_events)
                {
                    return [.. _events];
                }
            }
        }

        public void OnDeviceFailed(in DeviceFailed e)
        {
            if (e.Device == kind)
            {
                lock (_events)
                {
                    _events.Add(e);
                }
            }
        }
    }

    private static void DeviceFailureReported(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(DeviceFailureReported)))
        {
            return;
        }

        var faulty = new FaultyBackend("faulty") { RetryOnHost = false, Faulty = true };
        var x = faulty.Allocate(6, zeroed: false);
        var y = faulty.Allocate(6, zeroed: true);
        faulty.Upload([1f, 2f, 3f, -1f, 0f, 1f], x);
        var failures = new DeviceFailures("faulty");
        var printed = new StringWriter();
        using (Telemetry.Subscribe(failures))
        using (Telemetry.Subscribe(new ConsoleLogger(TelemetryLevel.Training, output: printed)))
        {
            long hostCalls = Kernels.HostCalls(faulty);
            DeviceException? thrown = null;
            try
            {
                faulty.Softmax(x, y, 2, 3, log: false);
            }
            catch (DeviceException error)
            {
                thrown = error;
            }

            Check(thrown is { Device: "faulty" } && thrown.Message.Contains("injected", StringComparison.Ordinal), "the device's error reaches the caller");
            Check(Kernels.HostCalls(faulty) == hostCalls, "retry off: nothing ran on the host");
            var events = failures.Events;
            Check(events.Count == 1, $"one DeviceFailed event, got {events.Count}");
            var e = events[0];
            Check(!e.RetriedOnHost && e.Operation is null && e.Message.Contains("injected", StringComparison.Ordinal), $"the event: {e}");
            Check(e.Hint is { } hint && hint.Contains("IDRAK_RETRY_ON_HOST", StringComparison.Ordinal) && hint.Contains("RetryOnHost", StringComparison.Ordinal)
                  && hint.Contains("faulty", StringComparison.Ordinal), $"the hint names both switches: {e.Hint}");
            Check(printed.ToString().Contains("IDRAK_RETRY_ON_HOST", StringComparison.Ordinal),
                $"a console logger subscribed for training prints device failures with the hint: {printed}");

            using (DeviceException.Handled())
            {
                _ = new DeviceException("faulty", "expected and handled");
            }

            Check(failures.Events.Count == 1, "an error inside DeviceException.Handled is not reported");
        }

        var json = new MemoryStream();
        using (var writer = new Utf8JsonWriter(json))
        {
            TelemetryJson.Write(writer, new TelemetryRecord(DateTimeOffset.UtcNow, new DeviceFailed("faulty", "Softmax", "boom", true, null)));
        }

        string line = Encoding.UTF8.GetString(json.ToArray());
        Check(line.Contains("\"event\":\"device_failed\"", StringComparison.Ordinal) && line.Contains("\"retried_on_host\":true", StringComparison.Ordinal)
              && line.Contains("\"operation\":\"Softmax\"", StringComparison.Ordinal), $"JSON Lines: {line}");
    }

    private static void DeviceFailureRetriedOnHost(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(DeviceFailureRetriedOnHost)))
        {
            return;
        }

        float[] input = [1f, 2f, 3f, -1f, 0f, 1f];
        var cpu = CpuBackend.Instance;
        var cx = cpu.Allocate(6, zeroed: false);
        var cy = cpu.Allocate(6, zeroed: true);
        cpu.Upload(input, cx);
        cpu.Softmax(cx, cy, 2, 3, log: true);
        var expected = new float[6];
        cpu.Download(cy, expected);
        cx.Release();
        cy.Release();

        var faulty = new FaultyBackend("faulty-retry") { RetryOnHost = true, Faulty = true };
        var x = faulty.Allocate(6, zeroed: false);
        var y = faulty.Allocate(6, zeroed: true);
        faulty.Upload(input, x);
        var failures = new DeviceFailures("faulty-retry");
        using (Telemetry.Subscribe(failures))
        {
            long hostCalls = Kernels.HostCalls(faulty);
            faulty.Softmax(x, y, 2, 3, log: true);
            var got = new float[6];
            faulty.Download(y, got);
            Check(got.SequenceEqual(expected), $"the retried result equals the CPU's: {string.Join(", ", got)} vs {string.Join(", ", expected)}");
            Check(Kernels.HostCalls(faulty) == hostCalls + 1, $"the retry counts as one host call ({Kernels.HostCalls(faulty) - hostCalls})");
            var events = failures.Events;
            Check(events.Count == 1 && events[0] is { RetriedOnHost: true, Operation: "Softmax", Hint: null }
                  && events[0].Message.Contains("injected", StringComparison.Ordinal),
                $"one retried event with the operation and no hint: {string.Join("; ", events)}");

            faulty.Faulty = false;
            faulty.Softmax(x, y, 2, 3, log: true);
            Check(Kernels.HostCalls(faulty) == hostCalls + 2 && failures.Events.Count == 1, "a kernel that works runs as before (its own host fallback here), with no event");
        }

        // A registered kernel is not retried: its error reaches the caller.
        OperationKernels.Softmax throwing = (b, _, _, _, _, _) => throw new DeviceException(b.Kind, "registered kernel failed");
        using (Kernels.Register(Ops.Softmax, "faulty-retry", throwing))
        {
            bool reached = false;
            try
            {
                faulty.Softmax(x, y, 2, 3, log: true);
            }
            catch (DeviceException)
            {
                reached = true;
            }

            Check(reached, "a registered kernel's error is not retried");
        }
    }

    private static void RetryOnHostSetting(Device device)
    {
        _ = device;
        foreach (var (setting, kind, on) in new (string?, string, bool)[]
                 {
                     (null, "cuda", false), ("", "cuda", false), ("0", "cuda", false), ("false", "vulkan", false), ("1", "cuda", true),
                     ("all", "hip", true), ("ALL", "vulkan", true), ("true", "cuda", true), ("1", "cpu", false), ("all", "cpu", false),
                     ("cuda,vulkan", "vulkan", true), ("cuda, vulkan", "VULKAN", true), ("cuda,vulkan", "hip", false), ("cuda", "cpu", false),
                     ("cpu", "cpu", true), (" hip ", "hip", true),
                 })
        {
            Check(Backend.RetryOnHostFrom(setting, kind) == on, $"IDRAK_RETRY_ON_HOST={setting ?? "(not set)"} for {kind}: expected {(on ? "on" : "off")}");
        }
    }

    private static void RetryOnHostLeavesFastPath(Device device)
    {
        _ = device;
        var field = typeof(Backend).GetField("_kernels", BindingFlags.NonPublic | BindingFlags.Instance)!;
        var faulty = new FaultyBackend("faulty-fast") { RetryOnHost = false };
        var y = faulty.Allocate(4, zeroed: true);
        faulty.Fill(y, 4, 1f);                                                // the first call resolves the slots
        Check(field.GetValue(faulty) is null, "nothing registered, no trace, retry off: the inlined fast path");
        faulty.RetryOnHost = true;
        faulty.Fill(y, 4, 2f);
        Check(field.GetValue(faulty) is Delegate?[] { Length: > 0 }, "retry on: calls leave the fast path, as under a trace");
        faulty.RetryOnHost = false;
        faulty.Fill(y, 4, 3f);
        Check(field.GetValue(faulty) is null, "retry off again: back on the fast path");
    }
}
