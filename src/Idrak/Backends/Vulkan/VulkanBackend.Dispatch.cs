// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using static Idrak.Backends.Vulkan.VulkanDriver;

namespace Idrak.Backends.Vulkan;

// The queue: commands (dispatches, copies, fills) are recorded into the current batch's command buffer, with a barrier
// before each so they run in order; a batch is submitted when the host needs its results, before it waits, or when it
// grows long. Each batch has a fence, its command buffer and its descriptor pools, recycled when the fence signals.
// Batches are numbered: a block remembers the last batch that used it, and the host reads or writes a block only after
// that batch has finished.
internal sealed unsafe partial class VulkanBackend
{
    // Guards the queue, the batches and the pipelines (the pool has its own lock: Return runs on any thread).
    private readonly Lock _gate = new();

    private ulong _commandPool;
    private ulong _pipelineCache;
    private Batch _batch = null!;
    private readonly Queue<Batch> _inFlight = new();
    private readonly Stack<Batch> _free = new();

    // The number of the batch being recorded, and of the last batch known to have finished.
    private ulong _recording = 1;
    private ulong _completed;

    // Batches submitted but not waited for, at most; commands per batch before it is submitted on its own.
    private const int MaxInFlight = 3;
    private const int MaxBatchCommands = 256;

    // Descriptor pool sizes (sets, storage-buffer descriptors); a batch adds pools as it needs them.
    private const uint PoolSets = 256;
    private const uint PoolDescriptors = 2048;

    private readonly Dictionary<VulkanKernel, Pipeline> _pipelines = [];
    private readonly Dictionary<int, ulong> _setLayouts = [];
    private readonly Dictionary<(int Bindings, int PushBytes), ulong> _pipelineLayouts = [];

    /// <summary>Kernels dispatched (for tests and diagnostics).</summary>
    internal long Dispatches;

    /// <summary>Batches submitted to the queue (for tests and diagnostics).</summary>
    internal long Submissions;

    private sealed class Batch
    {
        public IntPtr Commands;
        public ulong Fence;
        public readonly List<DescriptorPool> Pools = [];
        public int Pool;                                                       // the pool sets come from now
        public ulong Number;
        public bool Recording;
        public int Count;
    }

    private sealed class DescriptorPool(ulong handle, uint descriptors)
    {
        public readonly ulong Handle = handle;
        public readonly uint Capacity = descriptors;
        public uint Sets = PoolSets;
        public uint Descriptors = descriptors;
    }

    private sealed record Pipeline(ulong Handle, ulong Layout);

    /// <summary>
    /// Queues <paramref name="kernel"/> over groupsX × groupsY × groupsZ workgroups: binding i of descriptor set 0 is
    /// <paramref name="storages"/>[i]; <paramref name="pushConstants"/> fills its push-constant block (exactly
    /// <see cref="VulkanKernel.PushConstantBytes"/> bytes). Dispatches run in the order queued, each after the previous
    /// one's writes; reading a storage (<see cref="Backend.Download"/>) waits for the dispatches that use it.
    /// </summary>
    public void Dispatch(VulkanKernel kernel, uint groupsX, uint groupsY, uint groupsZ, ReadOnlySpan<Storage> storages, ReadOnlySpan<byte> pushConstants)
    {
        ArgumentNullException.ThrowIfNull(kernel);
        if (storages.Length != kernel.Bindings)
        {
            throw new ArgumentException($"Vulkan kernel '{kernel.Name}' binds {kernel.Bindings} storages; {storages.Length} given.", nameof(storages));
        }

        if (pushConstants.Length != kernel.PushConstantBytes)
        {
            throw new ArgumentException($"Vulkan kernel '{kernel.Name}' takes {kernel.PushConstantBytes} bytes of push constants; {pushConstants.Length} given.", nameof(pushConstants));
        }

        var p = _physical.Properties;
        if (groupsX > p.MaxComputeWorkGroupCountX || groupsY > p.MaxComputeWorkGroupCountY || groupsZ > p.MaxComputeWorkGroupCountZ)
        {
            throw new ArgumentOutOfRangeException(nameof(groupsX),
                $"Vulkan kernel '{kernel.Name}': {groupsX} × {groupsY} × {groupsZ} workgroups exceed {Name}'s {p.MaxComputeWorkGroupCountX} × {p.MaxComputeWorkGroupCountY} × {p.MaxComputeWorkGroupCountZ}.");
        }

        if (groupsX == 0 || groupsY == 0 || groupsZ == 0)
        {
            return;
        }

        lock (_gate)
        {
            var pipeline = PipelineOf(kernel);
            int bindings = kernel.Bindings;
            var blocks = new VulkanBlock[bindings];
            var buffers = stackalloc VkDescriptorBufferInfo[Math.Max(bindings, 1)];
            for (int i = 0; i < bindings; i++)
            {
                blocks[i] = BlockOf(storages[i]);
                long bytes = BlockBytes(storages[i].Length);
                if (bytes > MaxStorageBytes)
                {
                    throw new ArgumentException(
                        $"Vulkan kernel '{kernel.Name}': storage {i} holds {bytes:N0} bytes, more than {Name} binds ({MaxStorageBytes:N0}); use the host fallback for it.", nameof(storages));
                }

                buffers[i] = new VkDescriptorBufferInfo { Buffer = blocks[i].Buffer, Offset = 0, Range = (ulong)bytes };
            }

            var commands = Record();
            vkCmdBindPipeline(commands, PipelineBindPointCompute, pipeline.Handle);
            if (bindings > 0)
            {
                ulong set = AllocateSet(SetLayout(bindings), (uint)bindings);
                var writes = stackalloc VkWriteDescriptorSet[bindings];
                for (int i = 0; i < bindings; i++)
                {
                    writes[i] = new VkWriteDescriptorSet
                    {
                        SType = StructureWriteDescriptorSet,
                        DestinationSet = set,
                        DestinationBinding = (uint)i,
                        DescriptorCount = 1,
                        DescriptorType = DescriptorStorageBuffer,
                        BufferInfo = &buffers[i],
                    };
                }

                vkUpdateDescriptorSets(_device, (uint)bindings, writes, 0, null);
                vkCmdBindDescriptorSets(commands, PipelineBindPointCompute, pipeline.Layout, 0, 1, &set, 0, null);
            }

            if (pushConstants.Length > 0)
            {
                fixed (byte* values = pushConstants)
                {
                    vkCmdPushConstants(commands, pipeline.Layout, ShaderStageCompute, 0, (uint)pushConstants.Length, values);
                }
            }

            vkCmdDispatch(commands, groupsX, groupsY, groupsZ);
            foreach (var block in blocks)
            {
                block.LastUse = _recording;
            }

            Dispatches++;
            Recorded();
        }
    }

    private void StartQueue()
    {
        var poolInfo = new VkCommandPoolCreateInfo
        {
            SType = StructureCommandPoolCreateInfo,
            Flags = CommandPoolResetCommandBuffer,
            QueueFamilyIndex = _physical.QueueFamily,
        };
        Check(vkCreateCommandPool(_device, &poolInfo, null, out _commandPool), nameof(vkCreateCommandPool));
        var cacheInfo = new VkPipelineCacheCreateInfo { SType = StructurePipelineCacheCreateInfo };
        Check(vkCreatePipelineCache(_device, &cacheInfo, null, out _pipelineCache), nameof(vkCreatePipelineCache));
        _batch = NewBatch();
    }

    private void StopQueue()
    {
        foreach (var batch in _free.Append(_batch))
        {
            vkDestroyFence(_device, batch.Fence, null);
            foreach (var pool in batch.Pools)
            {
                vkDestroyDescriptorPool(_device, pool.Handle, null);
            }
        }

        _free.Clear();
        vkDestroyCommandPool(_device, _commandPool, null);                    // frees the command buffers
        foreach (var pipeline in _pipelines.Values)
        {
            vkDestroyPipeline(_device, pipeline.Handle, null);
        }

        foreach (var layout in _pipelineLayouts.Values)
        {
            vkDestroyPipelineLayout(_device, layout, null);
        }

        foreach (var layout in _setLayouts.Values)
        {
            vkDestroyDescriptorSetLayout(_device, layout, null);
        }

        vkDestroyPipelineCache(_device, _pipelineCache, null);
        _pipelines.Clear();
        _pipelineLayouts.Clear();
        _setLayouts.Clear();
    }

    private Batch NewBatch()
    {
        var batch = new Batch();
        var allocate = new VkCommandBufferAllocateInfo
        {
            SType = StructureCommandBufferAllocateInfo,
            CommandPool = _commandPool,
            Level = CommandBufferLevelPrimary,
            CommandBufferCount = 1,
        };
        IntPtr commands;
        Check(vkAllocateCommandBuffers(_device, &allocate, &commands), nameof(vkAllocateCommandBuffers));
        batch.Commands = commands;
        var fenceInfo = new VkFenceCreateInfo { SType = StructureFenceCreateInfo };
        Check(vkCreateFence(_device, &fenceInfo, null, out batch.Fence), nameof(vkCreateFence));
        return batch;
    }

    // The current batch's command buffer, begun if needed, with a barrier: the next command runs after every earlier
    // one (in this batch or an earlier one) and sees its writes.
    private IntPtr Record()
    {
        var batch = _batch;
        if (!batch.Recording)
        {
            var begin = new VkCommandBufferBeginInfo { SType = StructureCommandBufferBeginInfo, Flags = CommandBufferOneTimeSubmit };
            Check(vkBeginCommandBuffer(batch.Commands, &begin), nameof(vkBeginCommandBuffer));
            batch.Recording = true;
            batch.Number = _recording;
        }

        var barrier = new VkMemoryBarrier
        {
            SType = StructureMemoryBarrier,
            SourceAccessMask = AccessShaderWrite | AccessTransferWrite,
            DestinationAccessMask = AccessShaderRead | AccessShaderWrite | AccessTransferRead | AccessTransferWrite,
        };
        vkCmdPipelineBarrier(batch.Commands, StageComputeShader | StageTransfer, StageComputeShader | StageTransfer, 0, 1, &barrier, 0, null, 0, null);
        return batch.Commands;
    }

    // After a command: a long batch goes to the device now, so it starts working while more is recorded.
    private void Recorded()
    {
        if (++_batch.Count >= MaxBatchCommands)
        {
            Submit();
        }
    }

    // Copies bytes between blocks in queue order.
    private void RecordCopy(VulkanBlock source, long sourceOffset, VulkanBlock destination, long destinationOffset, long bytes)
    {
        var commands = Record();
        var region = new VkBufferCopy { SourceOffset = (ulong)sourceOffset, DestinationOffset = (ulong)destinationOffset, Size = (ulong)bytes };
        vkCmdCopyBuffer(commands, source.Buffer, destination.Buffer, 1, &region);
        source.LastUse = destination.LastUse = _recording;
        Recorded();
    }

    // Zeros a block in queue order.
    private void RecordFill(VulkanBlock block)
    {
        var commands = Record();
        vkCmdFillBuffer(commands, block.Buffer, 0, WholeSize, 0);
        block.LastUse = _recording;
        Recorded();
    }

    // Submits the current batch (when it has commands), ending it with a barrier that makes its writes visible to the
    // host once its fence signals.
    private void Submit()
    {
        var batch = _batch;
        if (!batch.Recording)
        {
            return;
        }

        var barrier = new VkMemoryBarrier
        {
            SType = StructureMemoryBarrier,
            SourceAccessMask = AccessShaderWrite | AccessTransferWrite,
            DestinationAccessMask = AccessHostRead | AccessHostWrite,
        };
        vkCmdPipelineBarrier(batch.Commands, StageComputeShader | StageTransfer, StageHost, 0, 1, &barrier, 0, null, 0, null);
        Check(vkEndCommandBuffer(batch.Commands), nameof(vkEndCommandBuffer));
        IntPtr commands = batch.Commands;
        var submit = new VkSubmitInfo { SType = StructureSubmitInfo, CommandBufferCount = 1, CommandBuffers = &commands };
        Check(vkQueueSubmit(_queue, 1, &submit, batch.Fence), nameof(vkQueueSubmit));
        _inFlight.Enqueue(batch);
        Submissions++;
        _recording++;
        _batch = _free.Count > 0 ? _free.Pop() : NewBatch();
        if (_inFlight.Count > MaxInFlight)
        {
            Retire(_inFlight.Dequeue());
        }
    }

    // Waits for the submitted batches numbered up to `number`.
    private void WaitUntil(ulong number)
    {
        while (_inFlight.Count > 0 && _inFlight.Peek().Number <= number)
        {
            Retire(_inFlight.Dequeue());
        }
    }

    // Submits everything recorded and waits for all of it.
    private void SubmitAndWait()
    {
        Submit();
        WaitUntil(ulong.MaxValue);
    }

    // Waits until the work using `block` has finished, so the host may read or write it.
    private void WaitFor(VulkanBlock block)
    {
        if (block.LastUse <= _completed)
        {
            return;
        }

        if (block.LastUse == _recording)
        {
            Submit();
        }

        WaitUntil(block.LastUse);
    }

    // Waits for a submitted batch, then makes it ready to record again.
    private void Retire(Batch batch)
    {
        ulong fence = batch.Fence;
        Check(vkWaitForFences(_device, 1, &fence, 1, ulong.MaxValue), nameof(vkWaitForFences));
        Check(vkResetFences(_device, 1, &fence), nameof(vkResetFences));
        foreach (var pool in batch.Pools)
        {
            Check(vkResetDescriptorPool(_device, pool.Handle, 0), nameof(vkResetDescriptorPool));
            (pool.Sets, pool.Descriptors) = (PoolSets, pool.Capacity);
        }

        Check(vkResetCommandBuffer(batch.Commands, 0), nameof(vkResetCommandBuffer));
        (batch.Pool, batch.Recording, batch.Count) = (0, false, 0);
        _completed = Math.Max(_completed, batch.Number);
        _free.Push(batch);
    }

    // A descriptor set for this batch, from its pools (pools are counted here, so allocation never fails for lack of room).
    private ulong AllocateSet(ulong layout, uint descriptors)
    {
        var batch = _batch;
        while (true)
        {
            if (batch.Pool == batch.Pools.Count)
            {
                batch.Pools.Add(NewPool(Math.Max(PoolDescriptors, descriptors)));
            }

            var pool = batch.Pools[batch.Pool];
            if (pool.Sets == 0 || pool.Descriptors < descriptors)
            {
                batch.Pool++;
                continue;
            }

            var info = new VkDescriptorSetAllocateInfo
            {
                SType = StructureDescriptorSetAllocateInfo,
                DescriptorPool = pool.Handle,
                DescriptorSetCount = 1,
                SetLayouts = &layout,
            };
            ulong set;
            Check(vkAllocateDescriptorSets(_device, &info, &set), nameof(vkAllocateDescriptorSets));
            pool.Sets--;
            pool.Descriptors -= descriptors;
            return set;
        }
    }

    private DescriptorPool NewPool(uint descriptors)
    {
        var size = new VkDescriptorPoolSize { Type = DescriptorStorageBuffer, DescriptorCount = descriptors };
        var info = new VkDescriptorPoolCreateInfo
        {
            SType = StructureDescriptorPoolCreateInfo,
            MaxSets = PoolSets,
            PoolSizeCount = 1,
            PoolSizes = &size,
        };
        Check(vkCreateDescriptorPool(_device, &info, null, out ulong pool), nameof(vkCreateDescriptorPool));
        return new DescriptorPool(pool, descriptors);
    }

    // The descriptor set layout of `bindings` storage buffers (bindings 0 to bindings - 1), made once.
    private ulong SetLayout(int bindings)
    {
        if (_setLayouts.TryGetValue(bindings, out ulong layout))
        {
            return layout;
        }

        var entries = new VkDescriptorSetLayoutBinding[Math.Max(bindings, 1)];
        for (int i = 0; i < bindings; i++)
        {
            entries[i] = new VkDescriptorSetLayoutBinding
            {
                Binding = (uint)i,
                DescriptorType = DescriptorStorageBuffer,
                DescriptorCount = 1,
                StageFlags = ShaderStageCompute,
            };
        }

        fixed (VkDescriptorSetLayoutBinding* e = entries)
        {
            var info = new VkDescriptorSetLayoutCreateInfo { SType = StructureDescriptorSetLayoutCreateInfo, BindingCount = (uint)bindings, Bindings = e };
            Check(vkCreateDescriptorSetLayout(_device, &info, null, out layout), nameof(vkCreateDescriptorSetLayout));
        }

        _setLayouts[bindings] = layout;
        return layout;
    }

    // The pipeline of `kernel` on this device, built on its first dispatch.
    private Pipeline PipelineOf(VulkanKernel kernel)
    {
        if (_pipelines.TryGetValue(kernel, out var pipeline))
        {
            return pipeline;
        }

        var p = _physical.Properties;
        if (kernel.Bindings > p.MaxPerStageDescriptorStorageBuffers)
        {
            throw new VulkanException($"Vulkan kernel '{kernel.Name}' binds {kernel.Bindings} storages; {Name} allows {p.MaxPerStageDescriptorStorageBuffers}.");
        }

        if (kernel.PushConstantBytes > p.MaxPushConstantsSize)
        {
            throw new VulkanException($"Vulkan kernel '{kernel.Name}' takes {kernel.PushConstantBytes} bytes of push constants; {Name} allows {p.MaxPushConstantsSize}.");
        }

        var key = (kernel.Bindings, kernel.PushConstantBytes);
        if (!_pipelineLayouts.TryGetValue(key, out ulong layout))
        {
            ulong setLayout = SetLayout(kernel.Bindings);
            var range = new VkPushConstantRange { StageFlags = ShaderStageCompute, Offset = 0, Size = (uint)kernel.PushConstantBytes };
            var info = new VkPipelineLayoutCreateInfo
            {
                SType = StructurePipelineLayoutCreateInfo,
                SetLayoutCount = 1,
                SetLayouts = &setLayout,
                PushConstantRangeCount = kernel.PushConstantBytes > 0 ? 1u : 0u,
                PushConstantRanges = &range,
            };
            Check(vkCreatePipelineLayout(_device, &info, null, out layout), nameof(vkCreatePipelineLayout));
            _pipelineLayouts[key] = layout;
        }

        ulong module;
        fixed (uint* code = kernel.Spirv)
        {
            var info = new VkShaderModuleCreateInfo { SType = StructureShaderModuleCreateInfo, CodeSize = (nuint)kernel.Spirv.Length * sizeof(uint), Code = code };
            int result = vkCreateShaderModule(_device, &info, null, out module);
            if (result != Success)
            {
                throw new VulkanException($"{Name} rejected Vulkan kernel '{kernel.Name}' (vkCreateShaderModule: {VulkanDriver.Describe(result)}).");
            }
        }

        try
        {
            byte* entry = stackalloc byte[] { (byte)'m', (byte)'a', (byte)'i', (byte)'n', 0 };
            var info = new VkComputePipelineCreateInfo
            {
                SType = StructureComputePipelineCreateInfo,
                Stage = new VkPipelineShaderStageCreateInfo
                {
                    SType = StructurePipelineShaderStageCreateInfo,
                    Stage = ShaderStageCompute,
                    Module = module,
                    Name = entry,
                },
                Layout = layout,
                BasePipelineIndex = -1,
            };
            ulong handle;
            int result = vkCreateComputePipelines(_device, _pipelineCache, 1, &info, null, &handle);
            if (result != Success)
            {
                throw new VulkanException($"{Name} could not build Vulkan kernel '{kernel.Name}' (vkCreateComputePipelines: {VulkanDriver.Describe(result)}).");
            }

            pipeline = new Pipeline(handle, layout);
            _pipelines[kernel] = pipeline;
            return pipeline;
        }
        finally
        {
            vkDestroyShaderModule(_device, module, null);
        }
    }
}
