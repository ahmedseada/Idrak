# Idrak.Vision.Plugins.Qwen: the Qwen-VL families as a plug-in

Qwen2.5-VL (`Qwen2_5_VLForConditionalGeneration`) and Qwen2-VL (`Qwen2VLForConditionalGeneration`) as one registration
each of Idrak's vision-language contracts, outside the library ([plans/done/16-qwen-vl.md](../../../plans/done/16-qwen-vl.md)), in
the package Idrak.Vision.Plugins. `QwenVisionPlugin.Register()` (or `VisionPlugins.RegisterAll()`) adds, for both, the
vision part (`VisionFamilies`) and the text decoder (`PretrainedArchitectures`); without it a Qwen-VL checkpoint is not
registered.

Checked against transformers 5.19 on tiny random checkpoints of both families, in transformers 5's and the hub's
config.json layouts (`tests/Idrak.Tests/data/vlm-qwen`, `tools/vlm/make_tiny_qwen_vl.py`; the `qwen vl` test group):
pixels and grids for a shrunk, a tall and an enlarged page, the tower and the merged features, prompt logits in one pass
and cached, 20 greedy tokens, two pages in one prompt, text only, the chat API, a tuning example's loss, and
`FineTuner.Train` with the merger trained.

## Use

```powershell
# the command line registers the package at startup: nothing to load
idrak run MODEL --image scan.png "Read the page."
```

```csharp
// an app references Idrak.Vision.Plugins (the legal OCR sample registers it beside Gemma 3's)
using Idrak.Vision.Plugins.Qwen;

QwenVisionPlugin.Register();
```

A page's resolution is Qwen's pixel budget: the vision option `max_pixels` (and `min_pixels`), about 28 x 28 pixels per
image token; the model folder's `preprocessor_config.json` gives the default.

## Files

| File | Holds |
|---|---|
| `QwenVision.cs` | `QwenVisionPlugin`; `QwenVisionFamily` (reads `vision_config` in either layout and the tensors under `visual.` or `model.visual.`, each shape checked); `QwenVision` (the descriptor: prompt format with one `<\|image_pad\|>` per token, causal attention, the options `min_pixels` and `max_pixels`, the tuning parts: the merger as `projector`, and the tower); `QwenImageTokens`; `QwenVisionConfig` |
| `QwenImageProcessor.cs` | transformers' `smart_resize` (halves rounded to even, as Python), Pillow's bicubic resize and CLIP's normalization through the library's `ImagePreprocessor`, the patches in merge-block order; `QwenImageGrid` |
| `QwenVisionTower.cs` | The tower: the patch embedding as one product, 2D rotary positions (half the pairs by row, half by column), blocks on the library's span attention (each image, or each window, its own keys; no score matrix stored), Qwen2.5-VL's windows and full-attention blocks, RMSNorm and SwiGLU; Qwen2-VL's LayerNorm and quick-GELU |
| `QwenPatchMerger.cs` | `ln_q`, merge² patches joined, two layers with exact GELU to the text width |
| `QwenImageEncoder.cs` | `IVisionEncoder` and `IVisionEncoderStages`: one block per image with its merged grid; each token's axis positions (`AxisPositions`: time 0, row, column), which the library places by `ImagePositionRules.Grid` |
| `QwenTextDecoder.cs` | The text decoder: Qwen2's, its rotary embedding on three axes (`RopeSettings.Sections` from `mrope_section`), either config layout, `model.layers.*` or `model.language_model.layers.*` |

What the library gives it, for any family with several rotary axes: the image position rule contract and its `grid`
rule, per-token rotary rows for a prompt with images (every existing rotary kernel, no new one), the decoding context's
rotary offsets after the prompt (recorded decode steps stay valid), the same in training (`IVisionEncoder.AxisPositions`
through `TrainingImage`), and fixed rotary buffers for recorded training passes. Vulkan and HIP run all of it through
the same operations.

Left: the speed and memory of a real page on the owner's card (plan 16, step 7) and the CER comparison against
bakrianoo's model (step 9).
