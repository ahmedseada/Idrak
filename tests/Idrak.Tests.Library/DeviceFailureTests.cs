// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Reflection;
using System.Text;
using System.Text.Json;
using Idrak.Abstraction.Devices.Cpu;
using Idrak.Abstraction.Operations;

// Plan 10, decision 13: a GPU kernel that fails throws (a DeviceException, reported to telemetry with a hint).
internal abstract partial class Tests
{
    // Plain arrays standing for device memory; Softmax throws a DeviceException while Faulty is set. Each test uses a kind
    // of its own, so the hint (which reads every live device of the kind) does not depend on the other tests' devices.
    internal sealed class FaultyBackend(string kind) : Backend
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
    internal sealed class DeviceFailures(string kind) : ITelemetryHook
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

    internal static void DeviceFailureReported(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(DeviceFailureReported)))
        {
            return;
        }

        var faulty = new FaultyBackend("faulty") { Faulty = true };
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
            Check(Kernels.HostCalls(faulty) == hostCalls, "nothing ran on the host");
            var events = failures.Events;
            Check(events.Count == 1, $"one DeviceFailed event, got {events.Count}");
            var e = events[0];
            Check(!e.RetriedOnHost && e.Operation is null && e.Message.Contains("injected", StringComparison.Ordinal), $"the event: {e}");
            Check(printed.ToString().Contains("injected", StringComparison.Ordinal),
                $"a console logger subscribed for training prints device failures: {printed}");

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
}
