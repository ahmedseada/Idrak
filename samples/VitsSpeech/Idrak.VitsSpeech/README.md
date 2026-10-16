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

**Memory.** The decoder makes the sound in pieces of `VitsSynthesizer.DecoderChunkSeconds` (10 s; 0: one pass), each
with `ContextFrames` of latents on both sides (its receptive field, computed from config.json) that are cut away again,
so the samples are a whole pass's and the device holds one piece at a time; a piece the device has no memory for is
decoded again in halves. Within a piece, with gradients off, each stage and residual step keeps only its result. For
83 s of sound from an MMS-sized checkpoint on the CPU: one pass peaked 3915 MB above its start, pieces of 10 s 502 MB,
with the same samples bit for bit (and 93 s against 149 s). With gradients on, one pass, every intermediate kept.

Every size, rate and scale is read from `config.json` (`VitsConfig`); `SpeechSynthesisOptions` carries the rate, the two
noise scales (null: the checkpoint's), the seed, the voice (multi-speaker checkpoints: the speaker), the output
rate (resampled) and the sentence split (`MaxCharacters`: the pieces of `SpeakStream`).

## Recorded speakers (plan 21, S9 route A)

A multi-speaker checkpoint (`num_speakers` and `speaker_embedding_size` in config.json, VCTK's shape) speaks each of
its recorded speakers: the speaker's row of `embed_speaker` conditions the duration predictor, every WaveNet layer of
the flow and the decoder, as transformers' `VitsModel` does with a `speaker_id`. `Voices` lists them by the names of a
`speakers.json` beside config.json (a list in index order, `["p225", "p226"]`, or a map of names to indices; speakers
it leaves out keep their indices), else by index: as far as this plug-in knows, a hub checkpoint's config holds only
the count, so write the names yourself when you know them. `--voice` (`SpeechSynthesisOptions.Voice`) takes a name in
any case or an index; no voice speaks the first speaker. `idrak speak MODEL --list-voices` lists them;
`VoicedSynthesizer` looks for a checkpoint speaker before the installed voices. Only single-speaker checkpoints
fine-tune (route C, below).

### Checkpoints that read phonemes

VCTK's and LJSpeech's checkpoints set `phonemize` in tokenizer_config.json: they were trained on espeak's IPA, as
transformers' `VitsTokenizer` makes it through the `phonemizer` package. `VitsTokenizer` phonemizes such text first
through `TextPhonemizers`: built in, `EspeakNgPhonemizer` runs the **espeak-ng program, which you install** (the
plug-in carries none): `espeak-ng -q --ipa -v LANGUAGE` on each stretch of text between punctuation, the marks kept in
place (the package's `preserve_punctuation`), white space collapsed. The program is `IDRAK_ESPEAK` (its full path) or
`espeak-ng` on the PATH; the voice is tokenizer_config.json's `language` (else en-us, what transformers asks for), or
`IDRAK_ESPEAK_VOICE`. Register another phonemizer (`ITextPhonemizer`) with `TextPhonemizers.Register`, and pick one by
name with `VitsTokenizer.Phonemizer`. The espeak program's output is close to the package's, not checked equal to it.

## Checked against the fixture

`tests/Idrak.Tests/data/audio/tiny-vits` (a seeded `VitsModel`, hidden 32, 2 layers, 2 heads, flow 32, upsampling
4 × 4) with `noise_scale` 0 and `noise_scale_duration` 0: the ids of "abc ab" give `reference/waveform.npy` (112
samples) within 1e-4 (`IDRAK_FILTER="audio tts"`). The owner runs a real MMS-TTS checkpoint on the GPUs.
`tiny-vits-speakers` and `tiny-vits-speakers-det` (three speakers, an 8-wide embedding; the stochastic and the
deterministic duration predictor) give transformers' waveform of each speaker within 1e-3 of its largest sample
(`IDRAK_FILTER="audio voices: recorded speakers"`).

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

## Fine-tuning a voice (plan 21, S9 route C)

Loaded with `SpeechSynthesisLoadOptions.Training`, the synthesizer is an `IAdversarialSpeechSynthesizer` that
`AdversarialSynthesisTrainer` (Idrak.Audio) fine-tunes on one person's transcribed recordings, VITS's way:

| Part | What training does |
|---|---|
| Weights | every tensor a parameter under its stored name; weight norms kept as gains and directions, folded per pass; saved back to the checkpoint's own layout (`model.safetensors`, the other files copied) |
| Posterior | the linear spectrogram (VITS's `spectrogram_torch`) through the posterior encoder, latents drawn, the flow forward |
| Durations | the monotonic alignment search on the host (`MonotonicAlignment`, VITS's recursion in float), the stochastic duration predictor's negative log-likelihood through its posterior flows (the spline composed from tensor operations); the deterministic predictor's squared log error when the checkpoint uses one |
| Losses | mel L1 × 45, KL × 1 (over the batch's frames), duration × 1 (over its tokens), least-squares adversarial × 1, feature matching × 2: every weight a setting (`AdversarialSynthesisOptions`) |
| Decoder | a random segment of `SegmentFrames` latent frames (32 by default, 8192 samples at hop 256): memory follows the batch, not an utterance's length |
| Discriminators | `VitsDiscriminators`: VITS's multi-period discriminator (a scale discriminator of grouped convolutions, periods 2, 3, 5, 7, 11), `scales` above 1 adds HiFi-GAN's pooled scales; weight-normed; `discriminator.json` lists every channel, kernel, stride, group and slope; read from the base folder when it holds `discriminator.safetensors`, else drawn new (MMS-TTS checkpoints ship none) |

```powershell
idrak voice train ahmed --from D:\rec\transcripts.jsonl --base facebook/mms-tts-ara -P Idrak.VitsSpeech.dll -l ar `
  --consent "I, Ahmed, agree that this voice is trained for speech made with Idrak." -d cuda
idrak speak facebook/mms-tts-ara "مرحبا، أنا إدراك" --voice ahmed -P Idrak.VitsSpeech.dll -o ahmed.wav
```

`--from` takes transcripts (JSON lines of `{"audio", "text"}`) or a folder of `name.wav` + `name.txt`; 30 to 60 minutes
of one person in a quiet room. The run folder resumes with `idrak voice train --resume RUN` or `idrak resume RUN`.
Fine-tuning defaults (all options): AdamW at 2e-4 with betas 0.8 and 0.99, a constant rate, no clipping, 40 audio seconds
a step, 2000 steps, a checkpoint every 500. Only single-speaker checkpoints train (route C is one person's voice).

**Checked** against a torch mirror of VITS's training step (`tests/Idrak.Tests/data/audio/vits-train`, the
`"audio voice train"` tests): both losses and every gradient of step 1 within 1e-4, the alignment paths exactly, three SGD
steps; resumed bit for bit on the CPU.

**What to expect, plainly.** The fixture proves the step is VITS's, not that the voice sounds good: no voice made here has
been listened to. The discriminators start new, so for the first thousand steps or so the adversarial terms are noisy
and the sound can get worse before it gets better. From clean, accurately transcribed speech, a few thousand steps
should give a voice recognisably the person's with the base model's pronunciation; Arabic goes through uroman
romanisation (no diacritics) as MMS was trained, so its pronunciation errors stay. Fine-tuning runs without dropout (a
later setting). The convolution's backward sums in a fixed order on CUDA too, so runs resume bit for bit there.

## Not done

- `SpeechSynthesisFamilies` has no `Create` (a VITS model from a description): MMS-TTS checkpoints are loaded, not made
  (and fine-tuned, above).
- The HiFi-GAN decoder is not registered in `Vocoders` by name (it needs a loaded checkpoint); `VitsSynthesizer.Vocoder`
  exposes it to a model that produces the same latents.
