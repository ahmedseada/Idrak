# Idrak.Vision.Plugins

The vision model families: vision-language towers and projectors, image models and line recognizers. The library
(`Idrak`, `Idrak.Vision`) keeps only generic defaults and fallbacks; the families live here, one namespace each
(`Idrak.Vision.Plugins.<Family>`), written on the library's public API only. It builds on Idrak.Vision (and through it
Idrak.Imaging and Idrak).

Nothing is registered: finding a checkpoint's vision family, text decoder or image model family by its architecture
name is not on the new design yet, so loading a model stops there.

## The families

| Namespace | Plug-in | What it holds | Read more |
|---|---|---|---|
| `Idrak.Vision.Plugins.Qwen` | `QwenVisionPlugin` ("qwen-vl") | Qwen2.5-VL and Qwen2-VL: their vision parts and text decoders | [Qwen/README.md](Qwen/README.md) |
| `Idrak.Vision.Plugins.Llava` | `LlavaPlugin` ("llava") | LLaVA: a CLIP tower and MLP projector, its text decoder | `Llava/Llava.cs` |
| `Idrak.Vision.Plugins.IdrakVision` | `IdrakVisionPlugin` ("idrak-vision") | IdrakVision, a family made to be created and pretrained | [IdrakVision/README.md](IdrakVision/README.md) |
| `Idrak.Vision.Plugins.Crnn` | `CrnnPlugin` ("crnn") | The CRNN line recognizer (convolutions, a bidirectional LSTM, CTC) as an image model family of the line-recognition task | [Crnn/README.md](Crnn/README.md) |

## Families

### Gemma 3 vision (`Idrak.Vision.Plugins.Gemma`)

The vision part of `Gemma3ForConditionalGeneration` (`Gemma3VisionFamily`), its text decoder being `Idrak.Text.Plugins`'
`GemmaPlugin`'s. The vision part: SigLIP's tower (the library's `VisionTransformerEncoder`) and Gemma 3's
projector (average pooling of the patch grid, the soft-embedding norm, the projection), in each of the checkpoints'
three namings (read through the library's `RenamedTensorStore`); the image preprocessing of the folder's
`preprocessor_config.json`; the prompt format (`<start_of_image>`, the soft tokens, `<end_of_image>`); image tokens
attending by the library's `image-blocks` rule; pan and scan (`Gemma3PanAndScan`, its options as vision options); and
fine-tuning of the projector and the tower (`IVisionTuningPart`), exported under the checkpoint's names.

```csharp
using Idrak.Models;
using Idrak.Vision.Plugins.Gemma;
using var model = PretrainedModel.Load(folder);    // model.Vision is a Gemma3Vision
```

License: Apache 2.0.
