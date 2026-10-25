# Idrak.Vision.Plugins

The vision model families: vision-language towers and projectors, image models and line recognizers. The library
(`Idrak`, `Idrak.Vision`) keeps only generic defaults and fallbacks; the families live here, one namespace each
(`Idrak.Vision.Plugins.<Family>`), written on the library's public API only. It builds on Idrak.Vision (and through it
Idrak.Imaging and Idrak).

```csharp
using Idrak.Vision.Plugins;

VisionPlugins.RegisterAll();      // every family of the package; calling it again changes nothing
```

- `VisionPlugins.RegisterAll()` registers every family, `VisionPlugins.UnregisterAll()` takes them back (the
  registries are left as they were found), `VisionPlugins.All` lists them.
- One family alone: its namespace's static `<Family>Plugin` class, with `Register()` and `Unregister()`.
- Each family is an `IIdrakPlugin` (`Idrak.Abstraction.Plugins`), recorded by name in `IdrakPlugins`; a plug-in
  registered later under the same name takes its place.
- The `idrak` tool registers this package at startup, before any `-P` plug-in (which can still take a family's place);
  `VisionPlugins.Package` is the package (an `IIdrakPluginPackage`: its name and `All`); the assembly declares it with
  `[assembly: IdrakPluginPackage]`, which `-P` and `IdrakPlugins.Load` read when a host loads `Idrak.Vision.Plugins.dll`
  itself.

## The families

| Namespace | Plug-in | Registers | Read more |
|---|---|---|---|
| `Idrak.Vision.Plugins.Qwen` | `QwenVisionPlugin` ("qwen-vl") | Qwen2.5-VL and Qwen2-VL: their vision parts (`VisionFamilies`) and text decoders (`PretrainedArchitectures`) | [Qwen/README.md](Qwen/README.md) |
| `Idrak.Vision.Plugins.Llava` | `LlavaPlugin` ("llava") | LLaVA: a CLIP tower and MLP projector (`VisionFamilies`), its text decoder (`PretrainedArchitectures`) | `Llava/Llava.cs` |
| `Idrak.Vision.Plugins.IdrakVision` | `IdrakVisionPlugin` ("idrak-vision") | IdrakVision, a family made to be created and pretrained (`VisionFamilies`, `PretrainedArchitectures`) | [IdrakVision/README.md](IdrakVision/README.md) |
| `Idrak.Vision.Plugins.Crnn` | `CrnnPlugin` ("crnn") | The CRNN line recognizer (convolutions, a bidirectional LSTM, CTC) as an image model family of the line-recognition task (`ImageModelFamilies`) | [Crnn/README.md](Crnn/README.md) |

A model whose family is not registered is refused with "not registered", naming this package.

## Families

### Gemma 3 vision (`Idrak.Vision.Plugins.Gemma`)

`Gemma3VisionPlugin.Register()` registers the vision part of `Gemma3ForConditionalGeneration` in `VisionFamilies`
(`Gemma3VisionFamily`) and, when it is missing, its text decoder through `Idrak.Text.Plugins`' `GemmaPlugin` (taken
back with it only then). The vision part: SigLIP's tower (the library's `VisionTransformerEncoder`) and Gemma 3's
projector (average pooling of the patch grid, the soft-embedding norm, the projection), in each of the checkpoints'
three namings (read through the library's `RenamedTensorStore`); the image preprocessing of the folder's
`preprocessor_config.json`; the prompt format (`<start_of_image>`, the soft tokens, `<end_of_image>`); image tokens
attending by the library's `image-blocks` rule; pan and scan (`Gemma3PanAndScan`, its options as vision options); and
fine-tuning of the projector and the tower (`IVisionTuningPart`), exported under the checkpoint's names.

```csharp
using Idrak.Models;
using Idrak.Vision.Plugins.Gemma;

Gemma3VisionPlugin.Register();                     // the vision family and its text decoder
using var model = PretrainedModel.Load(folder);    // model.Vision is a Gemma3Vision
```

License: Apache 2.0.
