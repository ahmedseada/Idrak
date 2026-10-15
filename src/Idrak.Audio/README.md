# Idrak.Audio

Sound for Idrak (plan 21): the general machinery every audio application needs and no model family, with the families
as plug-ins (`samples/`), as `Idrak.Vision` and `Idrak.Imaging` are built.

Built so far (phase S1):

- `AudioData`: planar float32 samples `[channels, samples]` in [-1, 1] at a sample rate; a 16-bit file's samples stay
  as bytes until the floats are read (`TryGetSamples16`), `ToMono`, `Channel`, `Pad`, `Slice`.
- `AudioCodecs` and `AudioEncoders`: registries of sound formats by name (`Register`, `Unregister`, `Names`, `Get`,
  `Default`, `Origin`, `SetPolicy`), `AudioCodecs.ReadInfo` / `Decode` / `Load`, `AudioEncoders.Save`.
- "wav": RIFF/WAVE, PCM 8 to 32 bits, float 32 and 64, WAVE_FORMAT_EXTENSIBLE, chunks before `data` skipped, a data
  chunk larger than the file cut to the file; RF64 and compressed variants refused by name. The encoder writes PCM 8 to
  32 bits and float 32, extensible past two channels or 16 bits.
- "flac" (decode only): 4 to 32 bits, every subframe kind (constant, verbatim, fixed, LPC), wasted bits, the stereo
  pairs; an ID3v2 tag before the stream skipped; frames found by their sync code and decoded in parallel, every CRC
  checked (a damaged frame refused by number). `FlacOptions`: `KeepCutEnd` (a file cut inside its last frame keeps its
  whole frames, as a WAV cut short does), `CheckCrc`, `FramesPerTask`. MP3, Ogg and AAC are plug-ins (`AudioCodecs.Register`).

Voices (phase S9, route B): `VoiceConverters` (a registry of converter families, none in the library; OpenVoice is
the plug-in `samples/OpenVoice`), `VoiceStores` with the built-in "folder" store (`FolderVoiceStore`: one folder per
voice under `IDRAK_VOICES`, else voices/ under `IDRAK_CACHE` or `~/.cache/idrak`, with voice.json holding the consent,
the clips' SHA-256s, seconds and agreements, and embedding.safetensors), and `VoicedSynthesizer`, which speaks
`SpeechSynthesisOptions.Voice` as a checkpoint's speaker first, else as an installed voice (converted, a checkpoint of
its own, or a named speaker).

Splitting (phase S7): `AudioSplitters`, a registry of `IAudioSplitter`s with "silence" (the pieces between pauses: each
frame's energy against the loudest frame's, in one pass over each channel; pauses shorter than `MinSilenceSeconds` stay
inside a piece, `PadSeconds` of silence kept on both sides, a piece over `MaxSeconds` cut again at its quietest frame)
and "seconds" (pieces of a fixed length) built in; `AudioSplitOptions` holds every setting. `SpeechSynthesisOptions` and
`VoiceConversionOptions` gained `DecoderSeconds` (the decoder's piece for one call; null: the model's own setting).

The command line (`idrak help audio`): `idrak transcribe`, `idrak speak`, `idrak voice add|list|show|remove|export|import`,
`idrak data audio`, `idrak data mel`, and `--audio` on `idrak run` and `idrak chat`. The other phases are in
plans/21-audio.md.

```csharp
var audio = AudioCodecs.Load("clip.wav", channels: 1);   // mono, as the file's rate
AudioEncoders.Save("copy.wav", audio, new AudioEncodeOptions { BitsPerSample = 24 });
```
