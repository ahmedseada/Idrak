# Plan 31: an audio processing pipeline (clearer sound in, as images have)

**Status:** planned 2026-10-24 against `main` 0e1fc0b8; phases below, built in order, at most two agents at a time. P0 built
(contracts, registry, pipeline, the four cleaners as processors bit for bit), P1 (filters) and P2 (dynamics; `peak` is
`peaknorm` and agc's silence level `quiet`, since after a comma a registered name starts a step, and eq's and the shelves' gain is `db`), P3 (noise and repair:
`denoise-live`, `spectral-gate`, `declick`, `declip`, `deess`, `dereverb`) and P4 (the command line, `SpeechTrainingOptions.Processing`).

## Why

Images have a pipeline: `ImageTransforms` (a registry of named transforms) and `ImageTransformPipeline` (text such as
`grayscale, max_width=1024 resample=lanczos`, each step with its value and options), used wherever images come in.
Audio has four cleaning stages (`AudioCleaners`: highpass, denoise, trim, normalize) sharing one settings record, used
by training only, and none can run on live sound. Plan 30's live runs on a phone headset showed what a clearer input
would buy: Whisper's mistakes were mostly on unclear consonants.

## The shape

- **`IAudioProcessor`** (`Idrak.Audio.Abstractions`), registered in **`AudioProcessors`** (a `SlotTable`): a name, a
  summary, its option keys and parameters (for listings and checks), and `Open(step, rate, channels)` giving a stage
  that processes blocks in order (state kept between them: a filter's memory, a compressor's envelope) and is flushed
  at the end. Whether it can run on live sound is declared: a stage that needs the whole recording (trim, integrated
  loudness, a noise floor read from the pauses) says so, and a live pipeline refuses it naming the live variant.
- **`AudioPipeline`**: the image pipeline's text form, `highpass=80, dehum=50, denoise=1.5 quantile=0.2, presence=3,
  compress ratio=3, loudness=-23`, each step a name, an optional value and `key=value` options, checked against the
  registry (an unknown name or option names what is registered). `Apply(AudioData)` for a recording; `Open(rate,
  channels)` for blocks; `Wrap(IAudioInput)` so live sound is cleaned before transcription.
- **Speed and memory**: blocks are processed in place in pooled buffers, channels in parallel, filters in
  `double` state with `float` samples, spectral stages on the library's STFT; no stage allocates per block. Every
  stage's result is checked against a reference (scipy, pyloudnorm, or the formula's own definition) within 1e-5
  (rule 71). Cold and warm timings per stage in the bench.
- **No family code**: learned denoisers are plug-ins registering an `IAudioProcessor`, never library defaults.
- **Compatibility**: `AudioCleaners` and `--clean` keep working; their four stages become processors and the cleaners
  run through them, with the same samples out (checked bit for bit against today's).

## Phases

| Phase | What | Depends on | Agent |
|---|---|---|---|
| P0 | Contracts, `AudioProcessors`, `AudioPipeline` (text, JSON, checks), block runner with flush and latency; the four cleaners as processors (`highpass`, `denoise`, `trim`, `normalize`) with today's output bit for bit; `AudioCleaners` on top | - | one |
| P1 | Filters (streaming): `gain`, `dc`, `highpass`/`lowpass` (order), `bandpass`, `notch`, `dehum` (50/60 Hz and harmonics), `eq` (peaking), `lowshelf`, `highshelf`, `presence` (the "sharpness" of speech), `preemphasis`, `mono`, `resample`; references from scipy | P0 | A |
| P2 | Dynamics: `gate`, `compress`, `limit`, `agc` (streaming), `loudness` (EBU R128 / ITU-R BS.1770 integrated, whole recording; a short-term live variant), `peak`; references from pyloudnorm and the formulas | P0 | B (with P1) |
| P3 | Noise and repair: `denoise` live variant (a running noise floor), `spectral-gate`, `declick`, `declip`, `deess`, `dereverb` (late-reverb spectral suppression) | P0, the STFT | A |
| P4 | Wiring: `idrak transcribe --audio-clean PIPELINE` (files and `--live`, the live input wrapped), `idrak audio process IN -o OUT --pipeline P` (hear the result), `idrak audio processors` (the list with options), `--clean` taking the pipeline text in `pretrain`, `voice train`, `data audio`; `speak --audio-clean` for its output; the CLI's environment table | P0 to P2 | B (with P3) |
| P5 | A learned denoiser as a plug-in in `Idrak.Audio.Plugins` (RNNoise: a small recurrent model, its released weights), registered as the processor `rnnoise`; bench rows for every stage | P0, P4 | one |

Two agents work at once only on phases whose files do not overlap (P1 with P2, P3 with P4), each building with the
shared build lock and leaving the commit to the lead, who checks, builds once, runs the targeted tests and pushes.

## Tests (CPU, `IDRAK_FILTER="audio processing"`)

- P0: the pipeline text round trip and its refusals; blocks of any size give the whole recording's output; the four
  cleaners bit for bit as before; a live pipeline refuses a whole-recording stage by name.
- P1 to P3: each stage against its reference within 1e-5 on the fixture's signals (tones, speech, noise, hum, clicks,
  clipping), and its block-by-block output equal to its whole-recording output.
- P4: the CLI options end to end on the fixture; `--live FILE --audio-clean …` against the file cleaned first.
- P5: RNNoise's output against the reference implementation's on the fixture.

## The owner's runs

After P4: `idrak audio process` on a phone recording with a few pipelines, listened to; `transcribe --live` with
and without `--audio-clean` on the same speech, compared word for word. Exact commands at each phase's end.
