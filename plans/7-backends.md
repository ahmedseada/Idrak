# Backends plan: a minimum backend, devices by provider, Vulkan for Intel GPUs

**Status:** in progress on the `architecture` branch. Covers the "Shared work before any new backend" steps 2 and 4
of [README.md](README.md), public KV cache formats, and the first new backend (Vulkan, for Intel GPUs first).

## Impact

| Change | Who gains | Cost on CUDA and CPU |
|---|---|---|
| **Minimum backend**: 12 core operations (memory, copies, synchronization, name, capabilities); every other operation defaults to the host fallback (`HostCall`: copy to the CPU, run the CPU kernel, copy back what changed) | every new backend runs every model on day one, then gets fast kernel by kernel | none: both override every operation; CUDA's few gaps now fall back instead of failing |
| **Device providers** (`DeviceProviders`, `Device.Get` / `Parse` / `Available`, `DeviceType.Vulkan`) | new device kinds without touching `Device` or the layers | none (`Device.Default` still picks `cuda:0` first) |
| **Public KV cache formats** with a composed attention fallback (expand to float32, masked products) | new cache formats (FP8, 4-bit) and new backends without fused attention | none: built-in formats keep their kernels |
| **Vulkan backend** (SPIR-V generated in C#, like PTX) | Intel Iris Xe / Arc GPUs, then AMD (plan 3) | none |
| **Public backend API** (last phase) | backends shipped as separate packages | an API to keep stable |

Checked at every step: the full test list on the CPU, on the minimal test backend (`IDRAK_DEVICES=minimal`), on
lavapipe (Mesa's software Vulkan driver) here, and on CUDA plus `--bench-gemv` on the RTX 5070 Ti.

## Phases

1. ✅ **Minimum backend and device providers.** The whole test list passes on a backend with only memory and copies.
2. **Public KV cache formats** (`KeyValueLayout`): built-ins keep their paths; an outside format implements write and
   expand, and attends through the composed fallback.
3. **Vulkan runtime**: `libvulkan.so.1` / `vulkan-1.dll` through `LibraryImport`, one compute queue, a buffer pool,
   uploads and downloads (mapped memory where the device memory is host-visible: integrated GPUs copy once), dispatches
   batched into command buffers with a barrier between them. With no kernels, every operation runs through the host
   fallback, so the test list passes from the first day.
4. **SPIR-V generator and kernels**: a C# SPIR-V writer and a small kernel builder; kernels for the operations that
   matter most first (element-wise, reductions, softmax, norms, rotary positions, gathers, matrix products, packed
   int8/int4/bfloat16 products for decoding, decoding attention), each replacing its host fallback.
5. **Intel tuning** (plan 4): subgroup size per kernel, cooperative matrices (XMX) where the driver exposes them.
6. **Public backend API**: `Backend`, `Storage`, `DeviceProvider` and `HostCall` public, for backends in their own
   packages.

## The Vulkan contract (runtime ↔ generated kernels)

- A kernel is SPIR-V 1.3 words (Vulkan 1.1), entry point `main`, `GLCompute`, its local size in the module.
- Descriptor set 0: binding *i* is the *i*-th storage passed to the dispatch, a storage buffer of 32-bit words
  (`float[]`, std430, stride 4; integer data read with `OpBitcast`).
- Scalars in one push-constant block (at most 128 bytes), 4-byte members in the order the kernel declares them.
- Dispatch: `VulkanBackend.Dispatch(kernel, groupsX, groupsY, groupsZ, storages, pushConstants)`; dispatches run in
  order (a compute-to-compute barrier between them).
- Storage holds up to `maxStorageBufferRange` bytes (2 GiB or more on desktop drivers); larger tensors fall back.
