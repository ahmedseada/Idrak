# Plan 32: Opus decoding as a plug-in (voice notes in)

**Status:** O0 (the reference) done 2026-10-24 (25d1fbdd); O1 (the codec) next.
**Scope (the owner, 2026-10-24): the `IAudioCodec` contract and nothing around it, speech only.** One plug-in codec
(`ReadInfo` and `Decode`); every path that opens a file goes through `AudioCodecs`, so it needs no wiring, and cleaning
is the processing pipeline's (plan 31). SILK only: CELT-only and hybrid frames (music, wide-band audio) are refused
naming them, and the redundant CELT frames some SILK packets carry at a switch are skipped.

## Why

Voice notes, the largest store of a person's own speech (their dialect, their phone, their rooms), are Opus in Ogg
files (`.opus`; messaging apps, browsers' recorders and voice chats write the same format). Idrak.Audio reads WAV,
FLAC and MP3 only, so those notes cannot reach transcription, the processing pipeline (plan 31), `data audio`, or the
speech trainers (Whisper fine-tuning on one's own voice, plan 29; `voice train`) without an outside program, which
rule 70 keeps out of the library. A decoder behind the existing contract (`IAudioCodec`, registered in `AudioCodecs`)
makes every path that opens a file read them, with nothing else changed.

## The shape

- **`OpusCodec : IAudioCodec`** in `Idrak.Audio.Plugins/Opus/`, registered by `OpusPlugin` through `AudioPlugins.All`
  like `rnnoise` (no library default). Name `opus`, extensions `.opus`, `.ogg`, `.oga`; `ReadInfo` answers null for an
  Ogg stream that is not Opus (Vorbis). `ReadInfo`: channels and rate from OpusHead, the length from the last page's
  granule (-1 from the first 64 KiB of a longer file, as the contract allows). `Decode`: the first Opus stream, at
  48 kHz (`OpusOptions.SampleRate`, as `Mp3Options` carries its settings), pre-skip and end trimming and OpusHead's
  gain as RFC 7845 says.
- **Exact against libopus**: SILK is integer arithmetic, so the samples and each packet's final range equal O0's.
  Damaged input is an `InvalidDataException` naming the page or packet; CELT, hybrid or a mapping family past 1 a
  `NotSupportedException` naming it. Nothing allocated per frame.

## Structure

```
src/Idrak.Audio.Plugins/Opus/
  OpusPlugin.cs          registers OpusCodec in AudioCodecs (in AudioPlugins.All)
  OpusCodec.cs           IAudioCodec: ReadInfo, Decode; OpusOptions
  OggReader.cs           pages (CRC-32), packets across pages, OpusHead and OpusTags
  OpusPackets.cs         TOC and frame packing (codes 0-3, padding); CELT and hybrid refused; trimming and gain
  RangeDecoder.cs        the entropy decoder (RFC 6716 4.1)
  Silk/                  the SILK decoder (libopus 1.6.1's): frame, NLSF and LPC, pitch and LTP, excitation,
                         synthesis, stereo, CNG for DTX, the resampler to the output rate, the tables
  README.md              what is read and refused, the licence (BSD-3-Clause)
tests/Idrak.Tests/AudioOpusTests.cs    group "audio opus:"
```

## Phases

| Phase | What | Agent |
|---|---|---|
| O0 | The reference and fixtures (done, below) | - |
| O1 | The codec: the container, packets and range decoder (agent A) and the SILK decoder (agent B) side by side, joined by the lead; every fixture through `AudioCodecs.Decode` equal to O0's, the refusals | A and B |

### O0, done: what the later phases must know

Rebuilt for the owner's scope change: the decoder reads **SILK only**; CELT-only and hybrid frames are refused.

- **Pinned**: libopus **v1.6.1** (commit `22244de5a79bd1d6d623c32e72bf1954b56235be`), the newest release;
  v1.5.2 (`ddbe4838…`) is built only to compare. cmake, static, `CMAKE_C_FLAGS=-O2 -ffp-contract=off` (no fused
  multiply-adds), `OPUS_DISABLE_INTRINSICS=ON` (no SIMD paths, no run-time CPU detection: plain C in its own order),
  DRED, OSCE, deep PLC, QEXT and custom modes off (their defaults), so the decoder is RFC 6716 with RFC 8251. The
  exact defines are in `opus.json` (`libopus.build`).
- **Regenerate**: `python3 tools/audio/make_opus_reference.py tests/Idrak.Tests/data/audio/opus` (git, cmake, cc,
  numpy; ffmpeg optional). A fresh clone and build wrote the same bytes as an earlier run. The signals use only
  IEEE-exact operations, so they are the same everywhere. The float encoder calls the C library's log/exp, so another
  machine or C library may encode a few packets differently. The committed files stay the reference either way,
  because every decoding comes from the files themselves.
- **SILK is exact everywhere**: every SILK-only decoding is the same in the float and fixed-point builds, in 1.5.2
  and in the system's SIMD libopus 1.4 (through ffmpeg and libsndfile), at every rate. Those files are stored as
  int16 (÷ 32768 is opus_decode_float's output exactly, and opus_decode gives the same values). **O2 must be equal,
  sample for sample, at 48/24/16/12/8 kHz.**
- **Voice notes: the mode survey** (`opus.json` → `mode_survey`; libopus' encoder unforced, mono, VBR, 3 s):
  - From **16 kHz input** (OpusHead input rate 16000), every packet is **SILK WB** at 12-48 kbit/s, VOIP and
    AUDIO, 20 and 60 ms, for speech in a steady background.
  - For clean speech the same holds, except the clip's last ~0.1 s (a near-silent tail at about −55 dBFS), which
    becomes **CELT WB**: from 32 kbit/s with VOIP, from 24 kbit/s with AUDIO. At 48 kbit/s a few loud packets just
    before that tail are CELT too.
  - From **48 kHz input**, there is **no SILK at all** at 24 and 32 kbit/s: hybrid FB (all packets with background
    noise; 77-86 % of them for clean speech), the rest CELT FB.
  - So a SILK-only decoder reads voice notes encoded at 16 kHz and at most 16-24 kbit/s. An app that encodes at
    48 kHz writes hybrid, and the decoder will refuse those files whole.
- **Redundant CELT frames inside SILK-only packets** (RFC 6716 4.5.1): a SILK-only frame with 17 or more bits left
  after its SILK data carries a redundant CELT frame.
  - libopus writes them at a SILK→CELT mode change (in the survey's voice notes, the last SILK packet before a CELT
    tail).
  - It also writes them at **SILK's own bandwidth switch**: `silk-bandwidth-switch-mono` (WB→MB after libopus' ~2.6 s
    transition, then MB→WB) has a SILK→CELT one (19-27 bytes) in the last frame before each switch and a CELT→SILK
    one in the first frame after.
  - Their decoded samples are float, not int16-exact. 1.5.2 differs there by 4.7e-10, fixed point by 4.5e-5.
  - `silk-bandwidth-switch-mono-stripped` is that file with them cut off. libopus then finds none, and its output
    is int16-exact. It differs from the original in 717 samples at 48 kHz, in the two crossfades, by up to 0.0045
    (≈ −47 dBFS).
  - So a SILK decoder that skips redundant frames (it must still find them: `silk_tell` + 17 ≤ 8 × length) can be
    exact against `-stripped`, and within 0.0045 of the original only there. Decoding them would need CELT, which
    is out of scope.
- **Per-frame checks for O1/O2**: for every packet, `opus.json` holds:
  - **final range** (OPUS_GET_FINAL_RANGE). It equals the encoder's (checked) and is the same at every rate and in
    the fixed-point build.
  - **frame_ranges**: each frame of a multi-frame packet decoded alone.
  - **silk_tell**: per frame, the bit position (`ec_tell`) where the SILK data ended, as libopus' decoder found it.
  - **redundancy**: per frame, whether a redundant CELT frame follows, its direction and its bytes.
  - These come from a traced copy of `opus_decoder.c` (two lines added; its samples are checked equal to the
    reference). A frame with a redundant CELT frame XORs that frame's range into the final range, so O2 compares
    `silk_tell` there.
  - Frames of 0 or 1 byte (the DTX file's 1-byte packets) are concealment: PLC/CNG, so O2 needs SILK's CNG. Their
    final range is 0.
  - `symbols.json` traces the first SILK symbols (VAD and LBRR flags) and `range-coder.json` seeded
    `ec_enc`/`ec_dec` sequences, each with `rng`, `val`, `tell` and `tell_frac` after every symbol, for O1's
    `RangeDecoder`.
- **Packet codes in SILK**: libopus' encoder writes code 0, code 2 (120 ms: 2 × 60 ms VBR), code 3 VBR (100 ms:
  5 × 20 ms) and code 3 with padding (CBR: 20 ms and 80 ms = 2 × 40 ms). It never writes code 1 for SILK, so
  `repack-code1-*` is libopus' repacketizer joining each frame with itself: a valid new stream, decoded on its own.
  `repack-code3-12x10ms-pad300-*` has padding past 254 bytes (a multi-byte padding length).
- **Output gain**: `gain = celt_exp2(6.48814081e-4f * header_gain)`, that is `(float)exp(0.6931471805599453094 *
  (double)(6.48814081e-4f * gain))`. Every float sample is multiplied by it, which makes the gain file float32.
  opusfile does the same through `OPUS_SET_GAIN`.
- **int16 output** (`opus_decode`, float build): `opus_pcm_soft_clip` runs over each call's whole buffer, then
  samples are rounded to nearest-even and saturated. SILK output never passes 1.0, so for SILK files this equals
  float × 32768.
- **Trimming**: pre-skip × rate / 48000 is dropped and (last granule − pre-skip) × rate / 48000 kept. Every fixture
  divides exactly (lengths are multiples of 12 at 48 kHz, and no frame size divides them, so the end trim is real).
- **Refusals** as the C harness words them (`reference_error`): `bad-crc` → page 3 CRC; `bad-truncated` → page 7
  truncated body; `bad-packet-code1-odd` and `bad-packet-160ms` → packet 5 invalid (`OPUS_INVALID_PACKET`);
  `unsupported-family255` → mapping family 255; `not-opus-vorbis.ogg` → not an OpusHead. `refuse-celt-fb-20ms-mono`
  and `refuse-hybrid-swb-20ms-mono` (6 packets each, CELT-only and hybrid-only: checked) are for the
  `NotSupportedException`; their final ranges and hashes are recorded but no samples. `chained` and `multiplexed`
  hold two SILK streams, each decoding as its source. `long-silk-wb-20ms-stereo-cbr` (69 KB) is the one past 64 KiB.

Later, only if needed: CELT and hybrid frames; AAC (`.m4a`) as its own codec.

## Tests (CPU, `IDRAK_FILTER="audio opus"`)

- `ReadInfo` (channels, rate, length; -1 from the first 64 KiB of the long file; null for the Vorbis file); every SILK
  fixture's samples and final ranges equal to O0's at the stored rates; the refusals naming the page, packet or mode;
  registered by the plug-in only; nothing allocated per frame.

## The owner's run

One of one's own voice notes opened as any file is: `idrak audio process note.opus -o note.wav`, or `idrak transcribe
… note.opus`. A note recorded as hybrid is refused saying so (O0's survey: 48 kHz input gives hybrid).
