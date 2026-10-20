# Idrak.Vision.Plugins.Crnn: the CRNN line recognizer

A line recognizer made of convolutions (pooling to height 1), a bidirectional LSTM over the columns and a score per
character and CTC's blank, as an image model family of the library's line-recognition task
(`ImageTask.LineRecognition`, plan 26 G14). `CrnnPlugin.Register()` (or `VisionPlugins.RegisterAll()`) registers
`CrnnFamily` in `ImageModelFamilies` under `CrnnForLineRecognition`.

| Type | Holds |
|---|---|
| `CrnnSettings` | The alphabet, line height, direction, reading order (`runs` by default), channels, hidden size, layers; `Labels` and `Lines` (the library's `LineRecognitionSettings`: stride 4, blank 0) |
| `CrnnRecognizer` | Makes the network (`Describe`, `Create`), loads a model folder through its family (`Load`), saves one (`Save`: config.json and model.safetensors), encodes transcriptions and reads lines through the library's `ModelLineReader` and `CtcDecoders`; an optional text normalizer for an app's form of text |
| `CrnnFamily` | Reads a model folder for `ImageModels.Load`: the settings from config.json, the network made from them, each tensor (`layers.{layer}.parameters.{i}`, `layers.{layer}.buffers.{i}`) read in with its shape checked |
| `MeasuredLineBatches` | How many lines go through the network at once, measured on the device (no card or memory size assumed) |

A folder saved before the move (`recognizer.ikm` and `recognizer.json`) still loads with `CrnnRecognizer.Load`.

```csharp
CrnnPlugin.Register();
using var recognizer = CrnnRecognizer.Load("model", Device.Default);
var lines = recognizer.Read(images, CtcDecoders.Greedy, new CtcDecodeOptions(), new MeasuredLineBatches(recognizer.Device));
```

The Arabic OCR app (`samples/Idrak.Samples.ArabicOcr`) trains and uses it; `idrak predict MODEL LINES` reads lines with it.
