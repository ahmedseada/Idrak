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

Planned next (the plan's phases): resampling and log-mel features on every device (S2), the "audio" chat part (S3),
speech recognition and synthesis contracts with their families as plug-ins (S4, S5), training (S6) and the command
line (S7).

```csharp
var audio = AudioCodecs.Load("clip.wav", channels: 1);   // mono, as the file's rate
AudioEncoders.Save("copy.wav", audio, new AudioEncodeOptions { BitsPerSample = 24 });
```
