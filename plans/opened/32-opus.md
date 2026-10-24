# Plan 32: Opus decoding as a plug-in (voice notes in)

**Status:** planned 2026-10-24 against `main` f95fe521; phases below, built in order, at most two agents at a time.
**Scope (the owner, 2026-10-24): speech only, no music.** The decoder implements SILK, Opus' speech codec; CELT-only and
hybrid frames (music and wide-band audio) are refused naming them. O0 measures which modes voice-note settings give.

## Why

Voice notes, the largest store of a person's own speech (their dialect, their phone, their rooms), are Opus in Ogg
files (`.opus`; messaging apps, browsers' recorders and voice chats write the same format). Idrak.Audio reads WAV,
FLAC and MP3 only, so those notes cannot reach transcription, the processing pipeline (plan 31), `data audio`, or the
speech trainers (Whisper fine-tuning on one's own voice, plan 29; `voice train`) without an outside program, which
rule 70 keeps out of the library. A decoder behind the existing contract (`IAudioCodec`, registered in `AudioCodecs`)
makes every path that opens a file read them, with nothing else changed.

## The shape

- **A plug-in in `Idrak.Audio.Plugins/Opus/`**, named for the format (it reads any Ogg Opus file, voice notes being
  the reason), registered by `OpusPlugin` through `AudioPlugins.All` like `rnnoise`: no library default, an
  unregistered `.opus` refused naming `AudioCodecs`' registered formats as today.
- **`OpusCodec : IAudioCodec`**: name `opus`, extensions `.opus`, `.ogg` and `.oga` only when the first stream is
  Opus (an Ogg Vorbis file is not this codec's; `ReadInfo` answers null for it). `ReadInfo` from the first pages
  (OpusHead: channels, pre-skip, input rate, gain, mapping; the length from the last page's granule position, so -1
  from the first 64 KiB and the exact length from the whole file). `Decode` gives the whole file.
- **`OpusOptions`** (nothing fixed): the output rate (48000, or 8000/12000/16000/24000 decoded at that rate directly
  as libopus does, or `input`: the rate OpusHead says the sound was made at, 16000 for most voice notes), whether
  OpusHead's output gain is applied (default yes), the channels (as stored, or mixed to mono), and the stream to read
  when a file holds several; carried to the CLI and an environment variable where the codecs already read options.
- **Exact against libopus**: SILK is integer arithmetic, so its samples must be equal, and every packet's final range
  equal to libopus' (`OPUS_GET_FINAL_RANGE`). A file holding CELT-only or hybrid frames is a `NotSupportedException`
  naming the first such packet and its mode (speech only, by the owner's scope). Damaged input (a bad page CRC, a packet
  cut short, a TOC the stream cannot have) is an `InvalidDataException` naming the page or packet; a valid variant
  not read (a mapping family past 1, say) a `NotSupportedException` naming it.
- **Speed and memory**: decoding one frame allocates nothing (the decoder's state and scratch made once per stream);
  the file's packets are read from its bytes in place; independent streams and long files decode in parallel by
  stream only where state allows (Opus frames depend on the ones before, so one stream is serial, rule 79 measured).
  The target: several hundred times real time for a mono voice note on one core, cold and warm reported apart.

## Structure

```
src/Idrak.Audio.Plugins/Opus/
  OpusPlugin.cs          registers OpusCodec in AudioCodecs (IIdrakPlugin, in AudioPlugins.All); the options' variables
  OpusCodec.cs           IAudioCodec: ReadInfo, Decode; OpusOptions
  Ogg/
    OggReader.cs         pages (capture pattern, version, flags, granule, serial, sequence, CRC-32), packets across pages
    OpusHeader.cs        OpusHead (version, channels, pre-skip, input rate, gain, mapping family 0/1 and its table), OpusTags
  OpusDecoder.cs         one stream: TOC, frame packing (codes 0-3, padding), SILK frames (NB/MB/WB, 10-60 ms, LBRR
                         skipped), bandwidth changes, stereo, gain, pre-skip and end trimming by granule; CELT and
                         hybrid frames refused naming them
  RangeDecoder.cs        the entropy decoder (RFC 6716 4.1): symbols by ICDF, uniform integers, raw bits from the end
  Silk/
    SilkDecoder.cs       a SILK frame: VAD and LBRR flags, gains, then the parts below, 10/20/40/60 ms
    SilkNlsf.cs          NLSF codebooks, stabilisation, NLSF to LPC (fixed point, as the reference)
    SilkPitch.cs         pitch lags and contours, LTP filters and scaling
    SilkExcitation.cs    rate level, pulses by shell coding, LSBs, signs, the excitation
    SilkSynthesis.cs     LTP and LPC synthesis per subframe
    SilkStereo.cs        mid/side predictors and unmixing
    SilkResampler.cs     the decoder's resampler from the SILK rate to the output rate (libopus's, for exactness)
    SilkTables.cs        the ICDFs, codebooks and constants of RFC 6716 / libopus
  README.md              the format, what is read, the options, the reference and its licence (BSD-3-Clause)
tools/audio/make_opus_reference.sh     libopus from xiph's source, a small C harness (our Ogg reader in C, libopus' decoder),
                                       the fixtures written by libopus' own encoder in every mode
tests/Idrak.Tests/AudioOpusTests.cs    group "audio opus:"
tests/Idrak.Tests/data/audio/opus/     small .opus files and libopus' decoded samples (under 3 MB)
```

## Phases

| Phase | What | Depends on | Agent |
|---|---|---|---|
| O0 | The reference: libopus built from xiph's source (`git clone github.com/xiph/opus`), the C harness (Ogg pages read in C, `opus_decode_float` and `opus_decode` per packet, pre-skip and granule trimming, each packet's final range), fixtures from libopus' encoder: SILK narrow/medium/wide band, 10 to 60 ms frames, packing codes 0-3 with padding, 40 to 120 ms packets, mono and stereo, DTX, output gain, voice-note-like files (16 kHz input, mono, 12-48 kbit/s, 20 and 60 ms frames, the modes libopus picks recorded); one small CELT and one hybrid file for the refusal; damaged copies | - | one |
| O1 | The container and the bitstream: `OggReader`, `OpusHeader`, `RangeDecoder` (checked against libopus' `ec_dec` on the fixtures' symbols), the TOC and frame packing, `OpusCodec.ReadInfo`, the plug-in's registration, the refusals | O0 | one |
| O2 | SILK: every part under `Silk/` (the decoder's two halves, parameters and synthesis, can go to two agents), SILK fixtures equal sample for sample at the SILK rate and through the resampler at 48 and 16 kHz, final ranges equal | O1 | A (B) |
| O3 | The Opus layer for speech: frame packing, bandwidth changes, stereo, gain, trimming, the CELT and hybrid refusal; every fixture through `AudioCodecs.Decode`; speed (cold and warm) and allocation per frame | O2 | one |
| O4 | Wiring and docs: `idrak help audio` and `data audio` list `.opus`; `transcribe`, `audio process` and `--clean` read voice notes as they are; a bench row (decode speed); the README; the owner's run below | O3 | one |

Later, not in this plan: CELT and hybrid frames (music, wide-band audio) if they are ever needed; AAC in MP4 (`.m4a`, the other format voice and audio messages arrive in) as its own plug-in;
an Opus encoder (`IAudioEncoder`) if a path needs to write Opus.

## Tests (CPU, `IDRAK_FILTER="audio opus"`)

- O1: pages, CRC and packets across page boundaries; OpusHead and OpusTags; `ReadInfo` (channels, rate, exact length
  from the whole file, -1 from the first 64 KiB of a long one); refusals naming the cause; an Ogg Vorbis file is not
  Opus; registered by the plug-in only.
- O2-O3: every SILK fixture against libopus' samples and final ranges, equal; the length and pre-skip exact; the
  output rate options; the CELT and hybrid files refused naming the packet; decoding twice gives the same bits;
  nothing allocated per frame.
- O4: the CLI on a fixture `.opus` (`audio process`, `transcribe` with the tiny Whisper fixture, `data audio --clean`).

## The owner's run

After O4: a folder of one's own voice notes decoded (`idrak audio process note.opus -o note.wav`, listened to), then
`idrak transcribe whisper-… note.opus -l ar`, then `idrak data audio` over the folder with `--clean` to make a
fine-tuning set (plan 29). Only one's own notes, or others' with their consent.
