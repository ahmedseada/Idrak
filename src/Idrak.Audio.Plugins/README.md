# Idrak.Audio.Plugins

The speech and voice model families: recognition, synthesis, voice conversion and phonemizers. The library (`Idrak`,
`Idrak.Audio`) keeps only generic defaults; the families live here, one namespace each
(`Idrak.Audio.Plugins.<Family>`), written on the library's public API only. It builds on Idrak.Audio (and through it
Idrak). Nothing is registered: picking a family by a checkpoint's architecture name, or a phonemizer, romanizer or
processor by its name, is not on the new design yet.

## The families

| Namespace | What it holds | Came from |
|---|---|---|
| `Idrak.Audio.Plugins.Whisper` | The speech family `WhisperForConditionalGeneration` | `samples/WhisperSpeech` |
| `Idrak.Audio.Plugins.EspeakNg` | The "espeak-ng" phonemizer, which runs the espeak-ng program the user installs (`IDRAK_ESPEAK`, `IDRAK_ESPEAK_VOICE`) | `src/Idrak.Audio` |
| `Idrak.Audio.Plugins.Vits` | The speech synthesis family `VitsModel` (what a checkpoint's text needs, phonemes or romanised text, read by `SpeechSynthesisModels.TextNeeds`) and the "uroman-arabic" romanizer | `samples/VitsSpeech`, uroman's romanizer from `src/Idrak.Audio` |
| `Idrak.Audio.Plugins.OpenVoice` | The "openvoice" voice converter (its tensor names through the library's `RenamedTensorStore`) | `samples/OpenVoice` |
| `Idrak.Audio.Plugins.IdrakAudio` | The speech and speech synthesis family `IdrakAudio`, and the audio-language architecture `IdrakAudioForConditionalGeneration` | `samples/IdrakAudio` |
| `Idrak.Audio.Plugins.RnNoise` | "rnnoise", RNNoise's learned denoiser (xiph's classic recurrent network, ported to C#) as a live audio processor | xiph's RNNoise v0.1.1 (BSD-3-Clause) |

Each family's folder has its README. The library itself has no speech family, no voice converter, no romanizer,
no phonemizer and no learned denoiser, and starts no outside program for text: the espeak-ng phonemizer is this
package's only one.

## The rnnoise processor

`rnnoise` cleans speech of steady and changing noise (fans, traffic, keyboards, a room's hum) with RNNoise's small
recurrent network: 42 features of each 10 ms frame (band energies, the pitch and its correlation), three GRUs giving
a gain for each of 22 bands and a voice probability, a pitch filter, then the gains over the spectrum. It is live (it
cleans a microphone as it comes) and runs on the CPU, far faster than real time. Naming it in a pipeline
(`idrak audio process --pipeline`, `idrak transcribe --audio-clean`) is not on the new design yet.

| Option | Default | What it does |
|---|---|---|
| `model=PATH` | `IDRAK_RNNOISE_MODEL`, else `rnnoise/rnnoise.rnnn` in the cache (`IDRAK_CACHE` or `~/.cache/idrak`) | the model file (RNNoise's own format); a name without a folder or extension (`model=sh`) is `rnnoise/sh.rnnn` in the cache |
| `mix` | 1 | how much of the denoised sound is heard (0 to 1; the rest is the original, delayed as the output is) |
| `attenuation` | none | the most a band is turned down, in dB (above 0 to 120): some noise stays and the speech keeps its body |
| `convert` | true | sound at another rate than 48 kHz is resampled to 48 kHz and back inside the stage; false refuses it |

RNNoise works at 48 kHz in frames of 480 samples; each channel is denoised alone. The output lags the input by 959
samples at 48 kHz (a frame to fill and RNNoise's own frame of delay, about 20 ms): silence first, the end given when the
stage is flushed; that is the stage's latency, which a recording's `Apply` (and `idrak audio process`) takes off, so
the output lines up with the input sample for sample. At another rate the delay is padded to a whole number of samples
there (960 at 48 kHz for 16 kHz: a shift of exactly 320) and the resamplers' hold-back is the stage's `Delay`, what a
live listener waits for. Blocks of any size give the whole recording's samples. At 48 kHz the blocks are
cleaned in place, nothing allocated per block; a long block's channels run on separate cores.

**The model.** RNNoise's last classic release (v0.1.1) carries its weights only as C arrays. The plug-in reads them from
RNNoise's model file format (what its `rnnoise_model_from_file` reads: `rnnoise-nu model file version 1`, then each
layer's sizes, activation and weights), so any model trained for the classic network in that format works too. To
write xiph's own weights as that file once (Python with numpy, and git, which fetches RNNoise's sources):

```powershell
cd D:\Projects\Idrak
python tools\audio\make_rnnoise_reference.py --model-only "$env:USERPROFILE\.cache\idrak\rnnoise\rnnoise.rnnn"
```

or copy `tests\Idrak.Tests\data\audio\rnnoise\rnnoise.rnnn` (the same 300 KB file, written by that script) there. A
missing model is refused when a stage opens, naming where it looked and how to get it. `RnNoise/README.md` has the
port's details and RNNoise's licence.
`samples/WhisperSpeech/transcribe.cs` and `samples/OpenVoice/voice.cs` are apps on this package.

Loading a model whose family is picked by its architecture name is not on the new design yet.

License: Apache 2.0.
