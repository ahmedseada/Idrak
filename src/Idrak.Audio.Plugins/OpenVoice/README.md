# OpenVoice: a voice converter as a plug-in (plan 21, S9 route B)

`Idrak.Audio.Plugins.OpenVoice` (plan 26 moved it here from `samples/OpenVoice`); `OpenVoicePlugin.Register()`
registers it alone, `AudioPlugins.RegisterAll()` with the package's other families, and the `idrak` tool registers it at
startup.

OpenVoice's tone colour converter (MIT licence, `myshell-ai/OpenVoiceV2`'s `converter/`) as an `IVoiceConverter`
registered in `VoiceConverters` under `openvoice`. A few clips of a real person give a speaker embedding; any speech
(MMS-TTS Arabic or English, any synthesizer, a recording) is then changed to that person's timbre. Conversion works on
sound, not text, so it does not depend on the language. The library registers no converter family (CLAUDE.md: a family
that is not registered fails with "not registered").

```csharp
OpenVoicePlugin.Register();                                    // the idrak tool registers both at startup
VitsPlugin.Register();
using var converter = VoiceConverters.Load("models/openvoice-v2", new VoiceConverterLoadOptions { Device = device });
var store = VoiceStores.Get();                                 // IDRAK_VOICES, else voices/ under IDRAK_CACHE or ~/.cache/idrak
store.Add("ahmed", ["rec/1.wav", "rec/2.mp3"], converter, new VoiceAddOptions
{
    Consent = new VoiceConsent("Ahmed Seada", "I agree that this voice is used for speech made with Idrak.", DateTimeOffset.UtcNow),
    Languages = ["ar", "en"],
});
using var voiced = new VoicedSynthesizer(SpeechSynthesisModels.Load("models/mms-tts-ara", new() { Device = device }), store, converter);
var (audio, stats) = voiced.Speak("مرحبا، أنا إدراك", new SpeechSynthesisOptions { Voice = "ahmed" });
```

## What runs where

| Part | Layers | Notes |
|---|---|---|
| Spectrogram (`OpenVoiceConverter.Spectrogram`) | reflect padding of (n_fft − hop) / 2, `Stft.Power` (not centred, periodic Hann), √(power + 1e-6) | OpenVoice's `spectrogram_torch`; `win_length` must equal `filter_length` (it does in OpenVoice's checkpoints) |
| Reference encoder (`OpenVoiceReferenceEncoder`) | `LayerNorm` over the bins, six weight-normed `Conv2d` 3 × 3 stride 2 + ReLU, `Permute` and flatten, the core `GRU` (PyTorch's gates through `LoadGateWeights`), `Linear` | the embedding of several clips is the mean of theirs; a clip longer than 1.5 × `EmbedSeconds` (10 s, a setting) is embedded in windows and averaged, as OpenVoice's se_extractor splits its references, so memory stays bounded |
| Posterior encoder | the library's `PosteriorEncoder` (Idrak.Audio.Layers: a pointwise `Conv1d`, a 16-layer `WaveNet`, a projection) | noise of `VoiceConversionOptions.Tau` (0.3) around the means, seeded |
| Flow | the library's `ResidualCouplingBlock`: `Forward` with the source's embedding, `Reverse` with the target's | 4 couplings of 4 WaveNet layers, OpenVoice's model code |
| Decoder | the library's `HifiGanGenerator` (its slopes `leaky_relu_slope` 0.1 and `final_leaky_relu_slope` 0.01, settings) | resblock "1" only ("2" is refused) |

`OpenVoiceTensors` (the library's `RenamedTensorStore` with OpenVoice's name map) reads OpenVoice's tensor names under the transformers VITS names the library's `FromStored` factories take (`enc_q.pre` as `posterior_encoder.conv_pre`,
coupling `i` as `flow.flows.{2i}`, `dec.ups` as `decoder.upsampler`) and folds weight norm (`weight_g`, `weight_v`) where a
plain `.weight` is read. A v2 checkpoint (`zero_g` true) gives the posterior encoder and the decoder a zero embedding;
v1 gives them the source's and the target's. OpenVoice's audio watermark is not added.

`VoiceConversionOptions`: `Tau` (0 is deterministic), `Seed`, `Source` (the speech's embedding when known; otherwise
the speech is embedded first), `MaxSeconds` (30: longer speech is converted in pieces cut at the quietest hop before
each limit, so device memory stays bounded).

## Checked against the fixture

`tests/Idrak.Tests/data/audio/tiny-openvoice` (OpenVoice's own `SynthesizerTrn`, inter 8, hidden 16, embedding 16, hop
16 at 16 kHz, random weights): the spectrogram within 4.4e-7 of its peak, the embeddings within 1e-4, the conversion
with tau 0 within 3.0e-5 (v2) and 9.2e-6 (v1) of its peak; checked at 1e-4 (`IDRAK_FILTER="audio voices"`). The tests
also check the flow's inverse, the pieces, the folder store, the voice resolution and the kit's `VoiceConverterSuite`
and `VoiceStoreSuite`. The owner runs the real converter on the GPUs.

## Getting the converter (once)

OpenVoice ships the converter as a PyTorch checkpoint (`converter/checkpoint.pth`) in a subfolder of its repository.
`idrak pull` takes a subfolder, and Idrak's `PyTorchCheckpointReader` (the "pytorch" checkpoint format) reads the
checkpoint without running anything in it, so no Python is needed:

```powershell
cd D:\Projects\Idrak
dotnet run --project src\Idrak.Cli -c Release -- pull myshell-ai/OpenVoiceV2/converter
$converter = "$env:USERPROFILE\.cache\idrak\downloads\huggingface\models\myshell-ai\OpenVoiceV2"
$converter = (Get-ChildItem $converter -Directory | Sort-Object LastWriteTime | Select-Object -Last 1).FullName + "\converter"
```

`VoiceConverters.Load($converter)` (and `idrak voice add --converter $converter`) then reads it as it is. To keep a
safetensors copy instead, `tools/audio/convert_openvoice.py` still writes one (it needs torch).

## Installing a voice and speaking with it

`idrak voice` adds, lists, shows, removes, exports and imports voices, and `idrak speak --voice` speaks with one
(`-v` prints the cold and warm timings and the peak device memory). The tool registers the plug-in at startup.
`samples/OpenVoice/voice.cs` is the prototype these commands match, an app on this package.

```powershell
cd D:\Projects\Idrak
dotnet run --project src\Idrak.Cli -c Release -- voice add ahmed --from D:\Recordings\ahmed --converter $converter --speaker "Ahmed Seada" --consent "I, Ahmed Seada, agree that this voice is used for speech made with Idrak." -l ar -l en -d cuda
dotnet run --project src\Idrak.Cli -c Release -- voice show ahmed
dotnet run --project src\Idrak.Cli -c Release -- speak D:\Models\mms-tts-ara --input arabic.txt --voice ahmed -d cuda -o ahmed-ar.wav -v
dotnet run --project src\Idrak.Cli -c Release -- voice export ahmed -o ahmed-voice.zip
```

Record 1 to 3 minutes of one person's normal speech in a quiet room (WAV, FLAC or MP3; a folder takes every file the
codecs read; `idrak data audio rec\ -o clips --mono --split silence --min-seconds 3` cleans a long recording first).
`voice add` and `voice show` print each clip's agreement (the cosine of its embedding with the voice's): a clip far below the
others is another speaker or noise, and is better left out.

## Consent

A voice is a person's likeness. `Add` requires a `VoiceConsent` (who is speaking, the statement they agreed to, when),
stores it in voice.json, and `Import` refuses a voice without one. The library records the consent; it cannot verify
it. Using someone's voice without their agreement is the user's responsibility, and never something Idrak ships by
default.
