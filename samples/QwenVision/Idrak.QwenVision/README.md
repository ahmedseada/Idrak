# Idrak.QwenVision: the Qwen-VL families as a plug-in

Qwen2.5-VL (`Qwen2_5_VLForConditionalGeneration`) and Qwen2-VL (`Qwen2VLForConditionalGeneration`) as one registration
each of Idrak's vision-language contracts, outside the library ([plans/16-qwen-vl.md](../../../plans/16-qwen-vl.md)).
The library holds the contracts, the generic decoder capabilities and the generic rules; this assembly brings what is
Qwen's: its processor, tower, merger, position rule and text decoder. Built like
[Idrak.Gemma3Vision](../../Gemma3Vision/Idrak.Gemma3Vision).

**State: a skeleton (2026-10-13).** Every class and both registrations are in place; every method that needs a step of
plan 16 throws `NotSupportedException("Qwen-VL: … is not built yet (plan 16, step N)")` until that step lands. Loading
a Qwen-VL checkpoint with the plug-in registered fails with that message; without the plug-in it fails with
"not registered", as every family does.

## Use

```powershell
# the command line
idrak -P .\samples\QwenVision\Idrak.QwenVision\bin\Release\net10.0\Idrak.QwenVision.dll run Qwen/Qwen2.5-VL-3B-Instruct --image scan.png "Read the page."
```

```csharp
// an app
QwenVisionPlugin.Register();   // both families: the vision part and the text decoder of each
```

## Files, and the step of plan 16 that builds each

| File | Holds | Step |
|---|---|---|
| `QwenVision.cs` | `QwenVisionPlugin` (`Register`, the `-P` entry), `QwenVisionFamily` (`IVisionFamily`, one per variant), `QwenVision` (`PretrainedVision`, `IVisionTuningPart`: the prompt format `<\|vision_start\|><\|image_pad\|>…<\|vision_end\|>`, attention `causal`, positions `grid`, the options `min_pixels` and `max_pixels`, the tuning parts), `QwenImageTokens`, `QwenVisionConfig` | 0 (the keys), 5 |
| `QwenImageProcessor.cs` | The smart resize within the pixel budget to multiples of patch × merge, the pixel patches `[t × h × w, channels × temporal × patch × patch]` and the grid (`QwenImageGrid`, whose merged grid is the token layout) | 5 |
| `QwenPositions.cs` | The `grid` position rule (three rotary axes: time, height, width) and the rope delta, through the library's position-rule contract | 1, 2 |
| `QwenVisionTower.cs` | The vision transformer: patch embedding, 2D rotary positions, window and full attention blocks on the library's span attention, several images of different sizes in one pass | 5, 6 |
| `QwenPatchMerger.cs` | The norm, the merge² concatenation and the MLP to the text width (trains as the family's `projector`) | 5, 6 |
| `QwenImageEncoder.cs` | `IVisionEncoder` and `IVisionEncoderStages`: images to features, one block per image with its own merged grid | 5 |
| `QwenTextDecoder.cs` | The text decoder's registration under both architecture names: Qwen2 style with rotary sections, the old (`model.layers.*`) and new (`model.language_model.layers.*`) tensor names | 2, 5 |

What the library adds for it (not in this plug-in): the image position rule contract and its `sequential` default
(step 1), the multi-axis rotary positions and their CUDA kernels (step 2), the same in training (step 3), one
registration per family with Gemma 3's text decoder moved into its plug-in (step 4). Vulkan and HIP take each
contract's default (CUDA first).

Tests come with the fixtures of step 0 (`tools/vlm/make_tiny_qwen_vl.py`, `tests/Idrak.Tests/data/vlm`): pixels and
grid, expanded ids, positions and rope deltas, each tower stage, the merger, logits in one pass and cached, 20 greedy
tokens, a two-image prompt, a tuning fixture.
