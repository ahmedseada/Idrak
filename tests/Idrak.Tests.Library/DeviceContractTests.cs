// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Reflection;
using Idrak.Abstraction.Operations;
using Idrak.Gpu;
using Idrak.Gpu.Abstractions;
using Idrak.Gpu.Cuda;
using Idrak.Gpu.Hip;
using Idrak.Gpu.Vulkan;

// Plan 33, C1: the device contract as interfaces. What every device provides (IDeviceBackend), each optional capability
// an interface read once when the backend starts, a device's own kernels found without names, storages in host memory,
// peer copies, providers registered by name, the offloading hooks and the plug-in kernels' launchers.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] DeviceContractGroup =
    [
        ("device contracts: the capability interfaces a backend implements are read once when it starts (the CPU's own attention, recurrent kernels and memory report; no graphs, profiler, FP8, fused products, matrix units, offloading or peer copies), and its own kernels are the operations it overrides as reflection over its methods names them, or a capability's", DeviceCapabilitiesRead),
        ("device contracts: a host fallback takes a storage in host memory (IHostStorage) from its values in place, with no copy from its device; a copy to another backend goes straight there only where the device's IPeerCopyBackend reaches it, else through host memory", HostStorageAndPeerCopies),
        ("device contracts: every device kind is registered by its name with a summary; a kind without ISeparableDeviceProvider carves no separate backend; the library's offloading hooks are registered (ITensorOffloadingHooks)", ProvidersAndOffloadingHooks),
        ("device contracts: a plug-in's kernel runs on a backend that launches its type (IKernelLauncher, IVulkanKernelDispatcher), not on others; a Vulkan kernel in the common launch shape refuses shared memory and a storage at an offset", DeviceKernelsLaunch),
    ];

    internal static void DeviceCapabilitiesRead(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(DeviceCapabilitiesRead)))
        {
            return;
        }

        var cpu = Device.Cpu.Backend;
        Check(cpu is IDeviceBackend and ISegmentedAttentionBackend and IRowAttentionBackend and IRecurrentKernels and IMemoryReport
              && cpu is not (IGraphCaptureBackend or IKernelProfiler or IFloat8Backend or IFusedPackedKernels or IMatrixUnitBackend or IOffloadingBackend
                  or IPeerCopyBackend or ILongLivedAllocations), "the CPU's capabilities");
        Check(!cpu.SupportsGraphs && !cpu.FusedKernels && !cpu.MatrixUnits && !cpu.MatrixUnitAttention(64) && cpu.Offload is null && cpu.LongLivedAllocations() is null
              && cpu.FreeMemory() is null && cpu.AvailableMemory() > 0 && cpu.SupportsSegmentedAttention(64) && cpu.SupportsAttentionRows(64)
              && cpu.TensorCoresUnavailable() is not null, "what follows from them");

        var minimal = MinimalBackend.Instance;
        Check(minimal.OwnKernels.Count == 0 && !minimal.OwnKernels.Contains(Ops.Softmax) && minimal is IMemoryReport && minimal.AvailableMemory() > 0
              && minimal.SupportsSegmentedAttention(64) == cpu.SupportsSegmentedAttention(64), "a device with memory and copies only owns no kernel and asks the CPU");

        // The own kernels as the reflection over overrides named them before (an override declared below Backend), and a
        // capability's operations where the device implements the interface that declares them.
        var own = cpu.OwnKernels;
        var capabilities = cpu.GetType().GetInterfaces().Where(i => i.Namespace == typeof(Backend).Namespace).ToList();
        var differ = new List<string>();
        foreach (var operation in Ops.All)
        {
            var parameters = operation.KernelType.GetMethod("Invoke")!.GetParameters().Skip(1).Select(p => p.ParameterType).ToArray();
            string method = typeof(Backend).GetMethod(operation.Name + "Kernel", BindingFlags.Public | BindingFlags.Instance, parameters) is not null
                ? operation.Name
                : operation.Name.TrimEnd('0', '1', '2', '3', '4', '5', '6', '7', '8', '9');                     // an overload's number (MaxPoolBackward2)
            var kernel = typeof(Backend).GetMethod(method + "Kernel", BindingFlags.Public | BindingFlags.Instance, parameters)!;
            bool expected = kernel.IsVirtual
                ? cpu.GetType().GetMethod(kernel.Name, BindingFlags.Public | BindingFlags.Instance, parameters)!.DeclaringType != typeof(Backend)
                : capabilities.Any(i => i.GetMethod(method) is not null);
            if (expected != own.Contains(operation))
            {
                differ.Add($"{operation} (expected {expected})");
            }
        }

        Check(differ.Count == 0 && own.Count == own.Count(op => own.Contains(op)) && own.Contains(Ops.Softmax) && own.Contains(Ops.LstmCell) && !own.Contains(Ops.GemmStrided),
            $"the CPU's own kernels: {string.Join(", ", differ)}");
        Check(ReferenceEquals(cpu.OwnKernels, own) && ReferenceEquals(CpuBackend.CreateSeparate(new SeparateBackendOptions("separate", null)).OwnKernels, own),
            "built once per device type");
    }

    internal static void HostStorageAndPeerCopies(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(HostStorageAndPeerCopies)))
        {
            return;
        }

        var first = new HostArrayBackend(peers: true);
        var second = new HostArrayBackend(peers: true);
        var y = first.Allocate(4, zeroed: true);
        try
        {
            first.Fill(y, 4, 2f);                                          // the host fallback: no kernel of its own
            Check(((IHostStorage)y).HostMemory.ToArray().All(v => v == 2f) && first.Downloads == 0 && first.Uploads == 0 && Kernels.HostCalls(first) == 1,
                $"in place: {first.Downloads} downloads, {first.Uploads} uploads, {Kernels.HostCalls(first)} host calls");

            var there = second.Allocate(4, zeroed: true);
            var minimal = MinimalBackend.Instance.Allocate(4, zeroed: true);
            try
            {
                first.CopyTo(y, there, 4);
                Check(first.PeerCopies == 1 && first.CopiesDirectlyTo(second) && first.CopiesDirectlyTo(first) && ((IHostStorage)there).HostMemory[3] == 2f,
                    "to a backend it reaches: its own copy");
                first.CopyTo(y, minimal, 4);
                var values = new float[4];
                MinimalBackend.Instance.Download(minimal, values);
                Check(first.PeerCopies == 1 && !first.CopiesDirectlyTo(MinimalBackend.Instance) && values.All(v => v == 2f), "elsewhere: through host memory");
            }
            finally
            {
                there.Release();
                minimal.Release();
            }
        }
        finally
        {
            y.Release();
        }
    }

    internal static void ProvidersAndOffloadingHooks(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(ProvidersAndOffloadingHooks)))
        {
            return;
        }

        var providers = DeviceProviders.All;
        Check(providers.All(p => ReferenceEquals(DeviceProviders.Find(p.Name), p) && !string.IsNullOrWhiteSpace(p.Summary))
              && DeviceProviders.Find("cuda") is ISeparableDeviceProvider && DeviceProviders.Find("vulkan") is ISeparableDeviceProvider
              && DeviceProviders.Find(DeferredTensors.Kind) is { } deferred && deferred is not ISeparableDeviceProvider,
            $"the providers: {string.Join(", ", providers.Select(p => p.Name))}");

        bool refused = false;
        try
        {
            Device.Get(DeferredTensors.Kind, 0).CreateSeparateBackend(new SeparateBackendOptions("separate", null));
        }
        catch (NotSupportedException)
        {
            refused = true;
        }

        Check(refused, "a kind that cannot carve says so");
        Check(TensorOffloading.Hooks is { } hooks && hooks.GetType().Assembly == typeof(Idrak.Layers.Linear).Assembly, "Idrak's offloading hooks");
    }

    internal static void DeviceKernelsLaunch(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(DeviceKernelsLaunch)))
        {
            return;
        }

        const string Ptx = ".version 7.0\n.target sm_50\n.address_size 64\n.visible .entry contract_kernel(.param .u64 .ptr .global .align 4 a, .param .u32 n)\n{\n    ret;\n}\n";
        var cpu = Device.Cpu.Backend;
        var storage = cpu.Allocate(8, zeroed: true);
        try
        {
            IDeviceKernel[] kernels =
            [
                new CudaKernel(Ptx, "contract_kernel"),
                new VulkanKernel([0x07230203, 0, 0, 0, 0], bindings: 1, pushConstantBytes: 4, "contract_kernel"),
                new GcnKernel(".kernel contract_kernel\n.arg a, 8, global\n.arg n, 4, value\n  s_endpgm\n", "contract_kernel"),
            ];
            foreach (var kernel in kernels)
            {
                string? message = Refusal(() => kernel.Launch(cpu, 1, 1, 1, 64, 1, 1, [storage, 8]));
                Check(message is not null && message.Contains("runs on a", StringComparison.Ordinal) && message.Contains(cpu.Kind, StringComparison.Ordinal),
                    $"{kernel.GetType().Name} on the CPU: {message ?? "launched"}");
            }

            var vulkan = kernels[1];
            string? shared = Refusal(() => vulkan.Launch(cpu, 1, 1, 1, 64, 1, 1, [storage, 8], sharedBytes: 16));
            string? offset = Refusal(() => vulkan.Launch(cpu, 1, 1, 1, 64, 1, 1, [KernelArgument.Pointer(storage, 4), 8]));
            Check(shared is not null && shared.Contains("shared memory", StringComparison.Ordinal) && offset is not null && offset.Contains("bound at its start", StringComparison.Ordinal),
                $"the common shape on Vulkan: {shared}; {offset}");
        }
        finally
        {
            storage.Release();
        }

        static string? Refusal(Action launch)
        {
            try
            {
                launch();
                return null;
            }
            catch (ArgumentException e)
            {
                return e.Message;
            }
        }
    }

    // A device whose storages are host arrays (IHostStorage) and that copies straight to its own kind (IPeerCopyBackend),
    // counting the copies to and from it.
    internal sealed class HostArrayBackend(bool peers) : Backend, IPeerCopyBackend
    {
        private readonly MemoryAccountant _memory = new(() => null, "host-array");

        public int Downloads, Uploads, PeerCopies;

        public override BackendCapabilities Capabilities => CpuBackend.Instance.Capabilities;

        public override string Kind => "host-array";

        public override string Name => "host arrays (tests)";

        public override Storage Allocate(int length, bool zeroed) => new HostArrayStorage(this, new float[length]);

        public override void Return(Storage storage)
        {
        }

        protected override void Detach(Storage storage) => ((HostArrayStorage)storage).Data = null!;

        protected override void Attach(Storage storage, Storage fresh) => ((HostArrayStorage)storage).Data = ((HostArrayStorage)fresh).Data;

        public override MemoryUsage GetMemoryUsage() => _memory.Usage;

        public override void ReleaseCachedMemory()
        {
        }

        public override void Upload(ReadOnlySpan<float> source, Storage destination)
        {
            Uploads++;
            source.CopyTo(((HostArrayStorage)destination).Data);
        }

        public override void Download(Storage source, Span<float> destination) => DownloadRange(source, 0, destination);

        public override void DownloadRange(Storage source, int offset, Span<float> destination)
        {
            Downloads++;
            ((HostArrayStorage)source).Data.AsSpan(offset, destination.Length).CopyTo(destination);
        }

        public override void Synchronize()
        {
        }

        bool IPeerCopyBackend.CopiesDirectlyTo(Backend destination) => peers && destination is HostArrayBackend;

        void IPeerCopyBackend.CopyTo(Storage source, Storage destination, int count)
        {
            PeerCopies++;
            ((HostArrayStorage)source).Data.AsSpan(0, count).CopyTo(((HostArrayStorage)destination).Data);
        }

        private sealed class HostArrayStorage(Backend backend, float[] data) : Storage(backend, data.Length), IHostStorage
        {
            public float[] Data = data;

            public Span<float> HostMemory => Data.AsSpan(0, Length);
        }
    }
}
