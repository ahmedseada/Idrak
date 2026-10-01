# Intel plan

**Status:** not supported; Intel GPUs and NPUs run everything on the CPU. Two different targets:

- **GPUs:** Iris Xe and Arc integrated graphics (most Intel laptops), Arc discrete cards (A- and B-series).
  General-purpose compute, so they can run every model and training, like an NVIDIA GPU.
- **NPUs:** Intel AI Boost (Core Ultra "Meteor Lake" and newer). Inference only, and only through a vendor compiler.

## Part 1: Intel GPUs

### The choice

| Option | What the user installs | Kernels | Fit with the no-dependency rule | Notes |
|---|---|---|---|---|
| **A. Vulkan compute** (recommended) | the graphics driver only | **SPIR-V generated in C#** | full | the same backend as the AMD plan; matrix units (XMX) through `VK_KHR_cooperative_matrix` where the driver exposes it |
| B. Level Zero | the graphics driver (it ships the Level Zero loader and GPU runtime) | SPIR-V (OpenCL flavor) generated in C# | full | closer to the CUDA driver API (explicit memory, command lists, unified shared memory); Intel only |
| C. OpenCL | the graphics driver | SPIR-V or OpenCL C | full | widest support, older API, fewest modern features |
| D. OpenVINO / DirectML add-on | an add-on package | none | as an optional package only | inference only |

**Recommendation: A first, B later if needed.** The Vulkan backend from the AMD plan runs on Intel GPUs with tuning
only. Level Zero is worth adding later for features Vulkan lacks on Intel, such as unified shared memory with
zero-copy on integrated GPUs, which matters because an integrated GPU shares the CPU's RAM.

🤔 To verify early: which Intel driver versions expose cooperative matrix (XMX) in Vulkan on Arc and on Xe-LPG/Xe2
integrated graphics, and with which input formats (FP16, bfloat16, int8).

### Phases (after the AMD plan's phases 1–2, which build the shared Vulkan backend)

1. **Run the full test list on Intel GPUs.** Fix driver-specific issues; set the subgroup size (Intel runs 8, 16 or
   32 lanes) per kernel. **Done when** every test passes on an Iris Xe laptop and an Arc card.
2. **Integrated-GPU memory.** Host-visible device memory on integrated GPUs, so uploads become plain copies or no copy;
   the memory limit counts the shared RAM. **Done when** an upload of a model's weights costs no second copy in RAM.
3. **Language models, tuned.** Few-row products and decoding attention tuned for Intel's execution units; XMX
   matrix products for prompts where available. **Done when** Qwen3-0.6B chats on Arc and on Iris Xe, with
   `--bench-vulkan` numbers recorded in the README.
4. **Training on Arc.** LoRA fine-tuning on discrete Arc (integrated GPUs are too slow and too memory-limited to
   target for training).

### Hardware to test on

| Device | Why |
|---|---|
| Core Ultra laptop (Arc integrated, Xe-LPG / Xe2) | the most common Intel GPU today; also has an NPU (Part 2) |
| 11th–13th gen Core laptop (Iris Xe) | very common; no XMX |
| Arc A770 / B580 | discrete; XMX matrix units; training target |

## Part 2: Intel NPU (AI Boost)

### Why it is different

The NPU runs no general kernels. It runs a **whole network compiled by Intel's compiler** (in OpenVINO, and a
compiler in the NPU driver used through OpenVINO's NPU plugin), with fixed shapes and INT8/FP16 math. So:

- the core library cannot generate kernels for it, as it does for GPUs;
- it is useful for **inference** (classifiers, embedders, small language models at fixed lengths), not training;
- the README already plans for this: an optional add-on package, with the core staying dependency-free.

### The choice

| Option | Dependency | Notes |
|---|---|---|
| **A. `Idrak.OpenVino` add-on** (recommended) | OpenVINO runtime (NuGet or system install) | export a model to OpenVINO's format (via the existing ONNX export), compile it for `NPU`, run it behind Idrak's `IPredictor` / `IChatModel` / `IEmbedder` interfaces |
| B. DirectML / Windows ML add-on | Windows only | also reaches AMD and Qualcomm NPUs on Windows 11; less control over Intel-specific features |
| C. Level Zero graph extension directly | the NPU driver | 🤔 the driver's graph interface exists for OpenVINO; it is not a documented public API, so too fragile to build on |

### Phases

1. **Spike:** run an exported ONNX classifier (the HousePrices or Sentiment sample) on the NPU through OpenVINO;
   measure latency and watts against the CPU. Stop here if it is not clearly better.
2. **`Idrak.OpenVino` package:** `OpenVinoModule` (like `Idrak.Onnx.Runtime`'s `OnnxModule`): load, compile for
   `NPU` / `GPU` / `CPU`, predict; plug into `Predictor` and the inference engine.
3. **Embeddings on the NPU:** an `IEmbedder` that runs a text encoder on the NPU, the first case that pays off
   (constant shapes, many calls, battery-friendly RAG on laptops).
4. **Small language models:** only if OpenVINO's NPU support for decoder models with KV caches holds up; fixed
   maximum lengths.

**Done when** a model exported from Idrak runs on the NPU behind the same interfaces as on the CPU, with outputs within
INT8/FP16 tolerance and a measured latency and power gain.

## Risks

- **GPU:** Intel's Vulkan compute driver maturity varies by generation; Iris Xe is far slower than Arc, so speed
  targets differ per device.
- **NPU:** depends on OpenVINO's supported operators and versions; models that use Idrak-only layers need an export
  path first (the ONNX export does not cover decoder blocks yet).
- **Two Intel targets compete for time:** the GPU work reuses the AMD Vulkan backend and comes first; the NPU is an
  add-on and can wait.

## Effort (rough)

GPU, after the shared Vulkan backend exists: 2–4 weeks for tests and tuning, plus 1–2 weeks for integrated-GPU memory.
NPU: 1 week spike; 2–3 weeks for the add-on package and embeddings; language models open-ended.
