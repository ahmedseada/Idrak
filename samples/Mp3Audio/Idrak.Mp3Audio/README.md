# Mp3Audio: MPEG audio (MP3 and MP2) as an audio codec and encoder plug-in

`Mp3Codec` is an `IAudioCodec` (the contract every sound format implements in `Idrak.Audio.Abstractions`), registered
in `AudioCodecs` under "mp3" for `.mp3` and `.mp2`. Once registered, everything that reads sound takes MP3:
`AudioCodecs.Decode` and `Load`, `ChatAudio.Decode`, the speech recognizer, the tuning data readers. `Mp3Encoder` is an
`IAudioEncoder` registered in `AudioEncoders` under "mp3" for `.mp3`, so `AudioEncoders.Save("x.mp3", audio)` and every
`-o x.mp3` that saves through it (`samples/OpenVoice/voice.cs speak ... -o sample.mp3`) write MP3, with no outside
program. Another format (Ogg Vorbis or Opus, AAC) is another `IAudioCodec` and `IAudioEncoder` registered the same way;
nothing else changes.

```csharp
Mp3AudioPlugin.Register();                                   // or -P Idrak.Mp3Audio.dll in the command line
var audio = AudioCodecs.Load("talk.mp3", channels: 1);
AudioEncoders.Save("reply.mp3", audio);                      // the encoder's options: Register(encodeOptions: ...) or IDRAK_MP3_*
```

`Mp3AudioPlugin.Unregister()` removes both.

## What it reads

- MPEG-1, MPEG-2 and MPEG-2.5 Layer III (`.mp3`) and Layer II (`.mp2`); Layer I is refused by name (no fixture yet).
- Mono, stereo, joint stereo (mid/side and intensity), constant and variable bit rates, free format, CRC-protected frames.
- An ID3v2 tag before the stream (of any size: past the first 64 KiB the registry asks again with the whole file), and
  ID3v1, TAG+ and APE tags after it.
- A Xing, Info or LAME/Lavc tag: the length, and with `Mp3Options.Gapless` (on) the encoder's delay and padding trimmed,
  so the sound starts and ends where the encoded one did (as ffmpeg and minimp3 give it).
- A lost sync (junk inside the stream) is passed over as minimp3 passes it: the frame before the junk, which no next
  header confirms, is dropped and the stream found again.


**Telling MP3 from other formats.** The decoder passes over junk before the first frame, as minimp3 does, so on its own it
could take a frame sync that quiet 16-bit samples hold by chance (`FF Fx` byte pairs) for an MP3 stream. `ReadInfo`
therefore never claims a RIFF (WAV), FLAC, Ogg or MP4 file, and after junk it wants `Mp3Options.SyncFramesAfterJunk`
(4) frame headers in a row, each where the one before ends. `AudioCodecs` also offers a file to the codecs that list
its extension first, so a `.wav` recording is the WAV codec's.

## How it decodes

The frame decoder is minimp3's (https://github.com/lieff/minimp3, CC0, public domain) ported to C#: the Huffman
decoding, requantisation, stereo processing, alias reduction, IMDCT and polyphase synthesis on its scalar paths, in its
float arithmetic and order. `tools/audio/make_mp3_tables.py` copies its tables into `Mp3Tables.cs` (none typed by
hand). The frames are indexed first, then decoded in parallel in chunks of `Mp3Options.FramesPerTask` frames: each
chunk starts from the frame before it and back until 511 bytes of main data (the bit reservoir's reach), so its state
is the one decoding in order makes and the samples are the same bit for bit (the tests check it). Samples are clamped
to [-1, 1] (an MP3 overshoots full scale where the sound is loud).

## Checked against

`tests/Idrak.Tests/data/audio/mp3` (`tools/audio/make_tiny_audio.py --sections mp3`): 15 files from LAME, shine and
ffmpeg's Layer II encoder across the sample rates of MPEG-1, 2 and 2.5, decoded by ffmpeg (the reference) and mpg123
(which must agree with it). The plug-in gives ffmpeg's samples within 1e-4 and its exact lengths (`IDRAK_FILTER="mp3"`).

## The encoder

`Mp3Encoder` writes MPEG-1 Layer III at 32, 44.1 and 48 kHz and MPEG-2 Layer III (LSF) at 16, 22.05 and 24 kHz, one
or two channels, at a constant bit rate (every bit rate the standard has: 32-320 kbit/s for MPEG-1, 8-160 for MPEG-2)
with the bit reservoir. It is written from ISO/IEC 11172-3 and 13818-3, not from LAME or shine (LGPL: none of their code
is used or ported):

- **Filterbank.** The standard's polyphase analysis (the 512-tap window, the 64-to-32 matrixing folded to 32 x 32),
  then per granule the long-block MDCT of every subband (18 new samples and the 18 before, sine window) and the
  aliasing butterflies. `tools/audio/make_mp3_encoder_tables.py` writes `Mp3EncoderTables.cs`: the analysis window from
  the synthesis window the decoder holds (minimp3's, the standard's D[i] = 32 C[i] and its symmetry), the cosines and
  the window by formula, and the Huffman codes read back from the decoder's tables by decoding every bit pattern (each
  table a complete prefix code of xlen x xlen entries). The tool checks the standard's analysis and synthesis
  reconstruct white noise at 84 dB, through the MDCT and butterflies too.
- **Masking model.** Per scalefactor band the energy, spread across the bands (Schroeder's spreading function, as in
  model 2) less an offset of 10 + 0.4 z dB (z the band's Bark), or the threshold of hearing (Terhardt's, full scale
  taken as 90 dB SPL) where that is higher; the perceptual entropy from it.
- **Bits.** A frame's granules get what their entropy asks for: the reservoir lends to a frame that asks for more than
  its size and saves from one that asks for less, never past 511 bytes (MPEG-1) or 255 (MPEG-2); split over the granules
  by what each asks for (4095 bits at most). Joint stereo codes a frame as mid/side when that has the lower entropy,
  both held to the lower of the two channels' thresholds (no intensity stereo).
- **Loops.** The rate loop finds the smallest global gain whose Huffman codes fit (halving, then bracketed from the
  last pass's gain); the distortion loop amplifies through the scalefactors the bands whose noise passes the threshold
  and runs the rate loop again, keeping the pass with the least noise above the thresholds (8 passes at most, stopping
  after 2 that did not help). Then the region split (region0 and region1 counts) and the tables that code the big
  values in the fewest bits, over every split, and the gain lowered while that fits.
- **Stream.** Plain frames: header, side information, main data starting as far back as the reservoir reaches. No CRC,
  no ID3, no Xing or LAME tag.

A decoder gives the sound `Mp3Encoder.Delay` (1057) samples late (481 of the two polyphase filterbanks, 576 of the
MDCT's overlap) and adds silence to the end of the last frame: the stream holds no gapless information. Encoding the
same sound gives the same bytes every time, in parallel or in order, on every machine: every vector loop runs across
output lanes in the scalar order, the tables are generated constants, the cube roots Newton's in doubles.

### Settings

`Mp3EncodeOptions` (`new Mp3Encoder(options)`, `Mp3AudioPlugin.Register(encodeOptions: options)`); when `Register` is
given none it reads the IDRAK_MP3_* variables (`Mp3EncodeOptions.FromEnvironment`), the defaults where unset, so a
command line or `voice.cs` takes them too. A value MP3 does not take is refused naming those it does.

| Option | Variable | Default |
|---|---|---|
| `Bitrate` (kbit/s, CBR) | `IDRAK_MP3_BITRATE` | 128 for two channels at 32-48 kHz, 64 for one; 96 for two at 16-24 kHz, 64 for one |
| `SampleRate` | `IDRAK_MP3_RATE` | the input's when MP3 has it, else the nearest (resampled with `Resampler.Resample`) |
| `ChannelMode` | `IDRAK_MP3_CHANNELS` (`joint`, `stereo`, `mono`) | `JointStereo` (mid/side where it pays); `Mono` mixes any number of channels down |
| `Quality` | `IDRAK_MP3_QUALITY` (`fast`, `standard`, `high`) | `Standard`; `Fast`: the rate loop only; `High`: 24 passes, wider table search |
| `LowpassHz` | `IDRAK_MP3_LOWPASS` (0 for none) | by the bits a sample and channel gets: none from 2.5, about 17 kHz at 128 kbit/s 44.1 kHz stereo |
| `FramesPerTask` | `IDRAK_MP3_FRAMES_PER_TASK` | 32; 0 encodes in order on one thread (the same bytes) |

`AudioEncodeOptions` (bits per sample, float) mean nothing for MP3 and are ignored.

### Measured

On a 4-core x64 container shared with other work (`"audio codecs: mp3 encode:"` prints them), Standard quality:

| Sound | Parallel (4 threads) | In order |
|---|---|---|
| 3 minutes of 44.1 kHz stereo at 128 kbit/s, a chirp with noise under it | 1.4 s (133x real time) | 3.4 s |
| 3 minutes of 44.1 kHz stereo, a chirp alone | 0.55 s | 1.4 s |
| 60 s of 16 kHz mono at 64 kbit/s (the speech models' rate) | 30-36 ms, the first encode in the process included | |

Writing three minutes to a stream allocates about 0.4 MB (the work buffers come from the pool); the hot loops are
compiled optimized at once, since an encode is often a process's only one. Quality, as the SNR of a 1 kHz tone
(1.5 kHz in the second channel) at amplitude 0.5 after the delay, decoded by the plug-in: 50-61 dB at the default bit
rates (57 dB at 128 kbit/s 44.1 kHz stereo, 53 dB at 64 kbit/s 16 kHz mono), 40-53 dB at 32 kbit/s MPEG-1, 34-39 dB at
8 kbit/s MPEG-2 mono, 54-82 dB at the highest bit rates; the WAV fixture's sweep with its noise burst 34 dB at 16 kHz
mono 64 kbit/s. The tests hold floors 10 to 19 dB under these (40 dB at every default). Only synthetic sound was
measured (tones, chirps, noise, the fixture's sweep): no speech or music recording, and SNR is not a listening test. ffmpeg decodes every rate, bit rate and
channel mode it writes (504 files tried) without an error, to the same lengths, and to the plug-in's samples within
1.5e-5; `tests/Idrak.Tests/data/audio/mp3enc` keeps ten of them with ffmpeg's lengths, SNRs and block RMS
(`tools/audio/make_mp3_encoder_fixture.cs`), and the tests check the encoder still writes them byte for byte.

### Not done

- Short and mixed blocks (window switching): long blocks only, so a sharp attack after quiet can smear before it
  (pre-echo); that is where short blocks would help.
- Intensity stereo, scfsi (scalefactor sharing between granules), preflag, scalefac_scale 1.
- VBR and ABR; the Xing/Info and LAME tags (so no gapless trim and no seek table); ID3 tags; CRC.
- MPEG-2.5 (8, 11.025, 12 kHz: written at 16 kHz), Layer I and II encoding.
- The masking model has no tonality estimate (one offset for tones and noise) and no temporal masking.
- 8 kbit/s MPEG-2 stereo leaves a 24 kHz frame 3 bytes for both channels' main data: it decodes, nearly silent.
