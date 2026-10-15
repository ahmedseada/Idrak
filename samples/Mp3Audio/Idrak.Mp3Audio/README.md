# Mp3Audio: MPEG audio (MP3 and MP2) as an audio codec plug-in

`Mp3Codec` is an `IAudioCodec` (the contract every sound format implements in `Idrak.Audio.Abstractions`), registered
in `AudioCodecs` under "mp3" for `.mp3` and `.mp2`. Once registered, everything that reads sound takes MP3:
`AudioCodecs.Decode` and `Load`, `ChatAudio.Decode`, the speech recognizer, the tuning data readers. Another format
(Ogg Vorbis or Opus, AAC) is another `IAudioCodec` registered the same way; nothing else changes.

```csharp
Mp3AudioPlugin.Register();                                   // or -P Idrak.Mp3Audio.dll in the command line
var audio = AudioCodecs.Load("talk.mp3", channels: 1);
```

## What it reads

- MPEG-1, MPEG-2 and MPEG-2.5 Layer III (`.mp3`) and Layer II (`.mp2`); Layer I is refused by name (no fixture yet).
- Mono, stereo, joint stereo (mid/side and intensity), constant and variable bit rates, free format, CRC-protected frames.
- An ID3v2 tag before the stream (of any size: past the first 64 KiB the registry asks again with the whole file), and
  ID3v1, TAG+ and APE tags after it.
- A Xing, Info or LAME/Lavc tag: the length, and with `Mp3Options.Gapless` (on) the encoder's delay and padding trimmed,
  so the sound starts and ends where the encoded one did (as ffmpeg and minimp3 give it).
- A lost sync (junk inside the stream) is passed over as minimp3 passes it: the frame before the junk, which no next
  header confirms, is dropped and the stream found again.

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
