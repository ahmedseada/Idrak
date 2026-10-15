# The reference for Idrak.Audio (plan 21, phase S0)

Written by `tools/audio/make_tiny_audio.py` (nothing downloaded); the versions are in `manifest.json`, with every
file's size and SHA-256. Two runs write the same bytes on the same machine and library versions; other machines or
versions may differ in the last bits of the torch outputs (the WAV files and their samples are exact everywhere).

| Folder | What |
|---|---|
| `wav/` | generated signals (1.5 s: a 440 Hz tone sweeping to 880 Hz with a seeded noise burst; a second channel of a 660 Hz tone with a later, quieter burst) as `pcm8`, `pcm16`, `pcm24`, `pcm32`, `float32`, `float64` (mono), `stereo16`, `extensible16` (WAVE_FORMAT_EXTENSIBLE, stereo, channel mask), `list-chunk16` (a LIST/INFO chunk of odd size before `data`), `rate8k`, `rate44k` (the signal generated at those rates), `oversized16` (a `data` chunk claiming twice the samples the file holds); each with its decoded samples as `.npy` (float32, `[channels, samples]`: PCM as integer / 2^(bits-1), float as stored) and `files.json` (rate, channels, samples); `bad-truncated.wav` (cut inside the fmt chunk) and `bad-no-fmt.wav` (no fmt chunk) for the refusals |
| `flac/` | the same signal written by libFLAC (through soundfile) as `pcm16` (the default level), `fixed16` (level 0: fixed predictors only), `lpc16` (level 8: LPC up to order 12), `stereo16` (the stereo pairs), `pcm8`, `pcm24`, `rate44k`, `mixed16` (stereo: silence, full-scale noise, then the signal: constant and verbatim subframes) and `wasted16` (every sample a multiple of 4: wasted bits); each with its integers as floats in `.npy` (integer / 2^(bits-1)) and `files.json` (rate, channels, samples, bits); `bad-truncated.flac` (cut inside a frame) and `bad-crc.flac` (a byte of the first frame flipped) |
| `resample/` | `rate44k.wav` and `rate8k.wav` resampled to 16 kHz by `torchaudio.functional.resample(lowpass_filter_width=6, rolloff=0.99, resampling_method="sinc_interp_hann")`, and `resample.json` with the parameters and lengths |
| `features/` | for `pcm16.wav` (16 kHz already): `stft-magnitude.npy` (`torch.stft`, n_fft 400, hop 160, periodic Hann, centred, reflect padding, the magnitude `[201, frames]`), `mel-filterbank.npy` (`torchaudio.functional.melscale_fbanks(201, 0, 8000, 80, 16000, norm="slaney", mel_scale="slaney")`, `[201, 80]`), `log-mel.npy` (transformers' `WhisperFeatureExtractor` on the 30 s padded segment: power, mel, log10 clamped at 1e-10, floored at max - 8, (x + 4) / 4; `[80, 3000]`), `features.json` with every parameter |
| `tiny-whisper/` | a `WhisperForConditionalGeneration` (d_model 32, 2 + 2 layers, 4 heads, feed-forward 64, 80 mels, 1500 source and 448 target positions, vocabulary 600) with weights from a seeded generator (normal, std 0.5: at 0.02 the tied head repeats its last input token and every decoding gives one token over and over; norms 1, biases 0), `save_pretrained` with `preprocessor_config.json`; `tokenizer.json` (a byte-level BPE trained here on the corpus, unused tokens filling to the vocabulary, then the specials `<\|endoftext\|>`, `<\|startoftranscript\|>`, `<\|en\|>`, `<\|ar\|>`, `<\|translate\|>`, `<\|transcribe\|>`, `<\|notimestamps\|>` and 31 timestamp tokens `<\|0.00\|>` to `<\|0.60\|>`; their ids in `special_tokens.json`), `generation_config.json` (the start, language, task and no-timestamps ids, the end token suppressed first, `max_initial_timestamp_index` 5); `reference/`: `encoder-output.npy` for `features/log-mel.npy`, `logits-step1.npy` for the prompt `<\|startoftranscript\|><\|ar\|><\|transcribe\|><\|notimestamps\|>`, `language.json` (the language tokens' logits after `<\|startoftranscript\|>` and the one detected), and 12 new tokens each from transformers' plain `generate` with Whisper's processors passed in (the suppressed first tokens; with timestamps, `WhisperTimeStampLogitsProcessor`): `greedy-ids.json`, `beam4-ids.json` (a beam of 4, length penalty 1, early stopping), `timestamps-ids.json` and `timestamps-beam4-ids.json` (from the prompt without `<\|notimestamps\|>`) |
| `ctc/` | a tiny recognizer (two Conv2d (1, 3) stride (1, 2) with GELU over `[N, 1, T, 80]`, a linear to width 32, two `TransformerEncoderLayer`s (4 heads, feed-forward 64, GELU, post-norm), a head to 40 classes, blank 0; seed 2029) as `model.safetensors` (torch's names) with `tensors.json`; `mels.npy` (4 utterances of lengths 23, 17, 31, 12, padded to 31), `loss.json` (the labels, `ctc_loss` mean with zero_infinity at step 1 and after each of 3 SGD steps at rate 0.1), `grads-step1.safetensors`, `weights-step3.safetensors`, `alignment.json` (torchaudio's `forced_align` of each utterance from the step-1 log-probabilities) |
| `tiny-vits/` | a `VitsModel` (hidden 32, 2 layers, 2 heads, feed-forward 64, flow size 32, 1 prior flow, 2 WaveNet layers, upsample rates [4, 4] with kernels [8, 8], one residual block of kernel 3 and dilations [1, 3], 65 spectrogram bins, hop 16, no speaker embedding; seed 2030), a 20-symbol `vocab.json`; `reference/waveform.npy` and `input.json` for the ids of "abc ab" with `noise_scale 0` and `noise_scale_duration 0` (deterministic) |
| `transcripts/` | `train.jsonl` of `{"audio", "text"}` rows over the fixture's files and `train-sharegpt.jsonl` (the same as ShareGPT with an `<audio>` marker and an `"audios"` list) |

## How the C# tests read it

- `"audio codecs:"` (S1): every `wav/` file decodes to its `.npy` (PCM exact, float within 1e-6); the header from the
  first bytes; `oversized16` decodes what the file holds; the two bad files are refused naming the file.
- `"audio features:"` (S2): resampled samples within 1e-5; STFT magnitudes and log-mel within 1e-4 of the largest value.
- `"audio asr:"` (S4): the encoder output and the step-1 logits within 1e-4; the greedy, beam and timestamp ids exact.
- `"audio train:"` (S6): the CTC loss and gradients within 1e-4, three SGD steps' losses within 1e-4, the alignments exact.
- `"audio tts:"` (S5): the waveform within 1e-4.

## Regenerate

```
python -m venv audio && audio/bin/pip install torch torchaudio transformers tokenizers safetensors numpy soundfile
audio/bin/python tools/audio/make_tiny_audio.py tests/Idrak.Tests/data/audio
```

`--sections wav` writes the WAV files alone with numpy only.
