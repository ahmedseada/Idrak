# VitsSpeech: a VITS text-to-speech family as a plug-in (plan 21, S5)

The family of transformers' `VitsModel`, the architecture the MMS-TTS checkpoints ship (`facebook/mms-tts-<lang>`, one
per language, Arabic among them): one safetensors file holds the text encoder, the stochastic duration predictor, the
flow and the HiFi-GAN decoder, so one plug-in speaks with no second checkpoint. It registers the family in
`SpeechSynthesisFamilies` under `VitsModel`; the library registers no family (CLAUDE.md: a family that is not registered
fails with "not registered").

```csharp
VitsSpeechPlugin.Register();                                   // or -P Idrak.VitsSpeech.dll in the command line
using var model = SpeechSynthesisModels.Load("models/mms-tts-eng", new SpeechSynthesisLoadOptions { Device = device });
var (audio, stats) = model.Speak("hello world", new SpeechSynthesisOptions { Rate = 1f, NoiseScale = 0.667f, Seed = 1 });
AudioEncoders.Save("hello.wav", audio);
```

## What runs where

| Part | Layers | Notes |
|---|---|---|
| Tokenizer (`VitsTokenizer`) | `vocab.json`, `tokenizer_config.json` | lower-cased and filtered to the vocabulary (`normalize`), a blank between symbols (`add_blank`, the pad id); `phonemize` is refused (no phonemizer in the library); `is_uroman` refuses non-Latin text (below) |
| Text encoder (`VitsTextEncoder`) | `Embedding` × √hidden, `Linear` q/k/v/out, a relative-position bias within `window_size` (gathers over the scores with precomputed indices, the embeddings sliced per length), post-norm `LayerNorm`, a feed-forward of two `Conv2d` (1, k) convolutions with the configured activation; a projection to the prior's means and log-variances | one utterance per pass, so no padding mask |
| Durations (`VitsStochasticDurationPredictor`) | pointwise convolutions, the dilated depth-separable convolution (depthwise `Conv2d` with groups, `LayerNorm`, exact GELU), the flows in reverse: the spline flows but the first, then the element-wise affine; the rational-quadratic spline inverted on the host in double | `noise_scale_duration` 0 gives deterministic durations; `use_stochastic_duration_prediction: false` uses `VitsDeterministicDurationPredictor` |
| Length regulation | `ceil(exp(log duration) · length scale)` per token, the prior's rows gathered once per frame (`EmbeddingLookup` with precomputed indices) | `SpeechSynthesisOptions.Rate` = 1 / length scale; the total is at least one frame |
| Flow (`VitsFlow`) | residual coupling in reverse: channels flipped, WaveNet (`Conv2d` with dilation, weight norm folded at load, tanh × sigmoid gates) | `noise_scale` 0 gives the means |
| Decoder (`HifiGanDecoder`) | `Conv2d` (1, 7), per stage leaky ReLU (`Maximum(x, slope · x)`), `ConvTranspose2d` (1, k) stride (1, rate), the residual blocks averaged; the last leaky ReLU (0.01, PyTorch's default, a setting `final_leaky_relu_slope`), `Conv2d` (1, 7), tanh | also `VitsSynthesizer.Vocoder`, an `IVocoder` over the flow's latents |

Every size, rate and scale is read from `config.json` (`VitsConfig`); `SpeechSynthesisOptions` carries the rate, the two
noise scales (null: the checkpoint's), the seed, the voice (multi-speaker checkpoints: the speaker index), the output
rate (resampled) and the sentence split (`MaxCharacters`: the pieces of `SpeakStream`).

## Checked against the fixture

`tests/Idrak.Tests/data/audio/tiny-vits` (a seeded `VitsModel`, hidden 32, 2 layers, 2 heads, flow 32, upsampling
4 × 4) with `noise_scale` 0 and `noise_scale_duration` 0: the ids of "abc ab" give `reference/waveform.npy` (112
samples) within 1e-4 (`IDRAK_FILTER="audio tts"`). The owner runs a real MMS-TTS checkpoint on the GPUs.

## Romanisation (decision 3 of plan 21)

MMS-TTS checkpoints for languages written in non-Latin scripts (Arabic among them: `facebook/mms-tts-ara`) set
`is_uroman` in `tokenizer_config.json`: they were trained on text romanised by uroman, and expect the same at
inference. `VitsTokenizer` romanises such text first, with the romanizer `TextRomanizers` holds for its script
(`VitsTokenizer.Romanizer` names one instead):

- **"uroman-arabic"** (built in, `UromanArabicRomanizer`): uroman's romanisation of the Arabic script, every code point
  of the Arabic blocks and presentation forms plus the three sequences uroman romanises together without a language
  code (alef + fathatan "an", maddah + alef, a word-initial alef + Persian yeh). `tools/audio/make_uroman_table.py`
  generates the table (`UromanArabicTable.cs`) and the cases `tests/Idrak.Tests/data/audio/uroman/arabic.json` from
  uroman itself; the test checks the plug-in gives uroman's output on every case. So "مرحبا بكم في إدراك" reads
  "mrhba bkm fy idrak", as the model was trained.
- A script no registered romanizer covers is refused, naming `TextRomanizers.Register`: an app registers an
  `ITextRomanizer` for it (asked before the earlier ones), or gives Latin text.

## Reused by the OpenVoice converter (plan 21, S9)

`samples/OpenVoice` (a voice converter plug-in) reads OpenVoice's tensors under these layers' names and runs
`VitsPosteriorEncoder` (transformers' `VitsPosteriorEncoder`: a spectrogram to latents, `spectrogram_bins` and
`posterior_encoder_num_wavenet_layers` from config.json), `VitsFlow.Forward` (the coupling layers forward, the inverse of
`Reverse`) and `HifiGanDecoder`. The project lets that assembly (and the tests) see its internal layers.

## Not done

- `SpeechSynthesisFamilies` has no `Create` (a VITS model from a description): MMS-TTS checkpoints are loaded, not made.
- The HiFi-GAN decoder is not registered in `Vocoders` by name (it needs a loaded checkpoint); `VitsSynthesizer.Vocoder`
  exposes it to a model that produces the same latents.
