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
  whole frames, as a WAV cut short does), `CheckCrc`, `FramesPerTask`. MP3 and MP2 decoding (`Mp3Codec`, a port of minimp3) and MP3 encoding (`Mp3Encoder`, written from the ISO specifications; `Mp3EncodeOptions`, the IDRAK_MP3_* variables) are built in too. Ogg and AAC would be plug-ins (`AudioCodecs.Register`).

Live sound (plan 30): `AudioCaptures`, a registry of `IAudioCapture`s that record from the system's input devices,
"winmm" (Windows' waveIn) and "alsa" (Linux's libasound) built in, each loading the system's own audio library at run
time (`ForThisMachine` picks the first that can run, `Devices` lists the inputs); `PcmStreamInput` reads a stream as it
arrives (a WAV header, or raw `s16le`/`f32le`); and `LiveTranscriber`, which transcribes any `IAudioInput` with any
recognizer as the sound comes: the open segment read again every `StepSeconds`, its words made final by a commit policy
(`LiveCommitPolicies`, "agreement" built in), a segment ended by a pause, by `WindowSeconds` or by the end, quiet never
transcribed. `LiveTranscriptionOptions` holds every setting.

Voices (phase S9, route B): `VoiceConverters` (a registry of converter families, none in the library; OpenVoice is
the plug-in `Idrak.Audio.Plugins.OpenVoice`), `VoiceStores` with the built-in "folder" store (`FolderVoiceStore`: one folder per
voice under `IDRAK_VOICES`, else voices/ under `IDRAK_CACHE` or `~/.cache/idrak`, with voice.json holding the consent,
the clips' SHA-256s, seconds and agreements, and embedding.safetensors), and `VoicedSynthesizer`, which speaks
`SpeechSynthesisOptions.Voice` as a checkpoint's speaker first, else as an installed voice (converted, a checkpoint of
its own, or a named speaker).

Splitting (phase S7): `AudioSplitters`, a registry of `IAudioSplitter`s with "silence" (the pieces between pauses: each
frame's energy against the loudest frame's, in one pass over each channel; pauses shorter than `MinSilenceSeconds` stay
inside a piece, `PadSeconds` of silence kept on both sides, a piece over `MaxSeconds` cut again at its quietest frame)
and "seconds" (pieces of a fixed length) built in; `AudioSplitOptions` holds every setting. `SpeechSynthesisOptions` and
`VoiceConversionOptions` gained `DecoderSeconds` (the decoder's piece for one call; null: the model's own setting).

Speech text (plan 22): the front end a synthesizer's tokenizer runs before it looks symbols up, in `Idrak.Audio.Abstractions`:
`ITextRomanizer` and `TextRomanizers` (a script's text to the Latin letters a checkpoint was trained on; none built in:
Idrak.Audio.Plugins' VITS family registers "uroman-arabic"; `For` picks the one covering a text's script),
`ITextPhonemizer` and `TextPhonemizers` (text to the phonemes a checkpoint reads; none built in, as the library starts no
outside program: Idrak.Audio.Plugins' "espeak-ng" runs the espeak-ng program the user installs; `KeepingPunctuation`
phonemizes between the marks). `Subtitles` writes a transcription's segments as
SubRip or WebVTT.

Layers (`Idrak.Audio.Layers`, plan 22): the generative speech layers the VITS and OpenVoice plug-ins used to carry,
each built from a checkpoint by a `FromStored` factory over `StoredLayers` (transformers' tensor names, a prefix per
part) or from the library's layers directly: `WaveNet` (gated residual layers with global conditioning, the tanh ·
sigmoid gate one kernel), `ResidualCouplingLayer` and `ResidualCouplingBlock` (mean-only affine coupling flows, `Forward`
and `Reverse`), `PosteriorEncoder` (a spectrogram's latents' means and log-stddevs), `HifiGanResidualBlock` and
`HifiGanGenerator` (`Samples` decodes long latents in pieces with `ContextFrames` of context, so device memory follows the
piece), `HifiGanDiscriminators` with `HifiGanDiscriminatorConfig` (VITS's multi-period and HiFi-GAN's multi-scale
discriminators, created from a seed or read from `discriminator.safetensors` or a release's `D_*.pth`, saved back),
`HifiGanVocoder` (HiFi-GAN as an `IVocoder` read from a checkpoint folder under transformers' names, SpeechT5's and
FastSpeech 2's vocoders among them, the log-mel standardized with the checkpoint's mean and scale; an app registers one
by name with `HifiGanVocoder.RegisterFolder`, so any synthesizer that speaks through a vocoder can use it). With
them `AdversarialLosses` (the least-squares GAN losses and feature matching), `MonotonicAlignment` (the search and its
durations, on the host in float as VITS adds) and `DifferentiableLogMel` (the log-mel of sound as tensor operations, so a
mel loss reaches the generator). A padded batch goes through them with `PaddedSequences` masks; each block the trainer
may checkpoint runs through `Checkpointing.RunBlock`, so `ActivationMemory.CheckpointBlocks()` reaches it.

Cleaning: `AudioCleaners`, a registry of `IAudioCleaner` stages a recording goes through before it is used, with
"highpass" (a second-order Butterworth high-pass: DC and rumble out), "denoise" (spectral gating: each frequency's noise
floor read from its quietest frames and gated off through the STFT and back, at most `DenoiseFloorDecibels`), "trim"
(the silence at both ends cut, a pad kept) and "normalize" (the speech level brought to `TargetDecibels`, the peak under
`PeakCeiling`) built in; `AudioCleaningOptions` holds every setting, and `Settings` for a plug-in's stage (a model that
isolates the voice registers under the same contract). The speech trainers run `SpeechTrainingOptions.Cleaning` over
every utterance before its features (cached under the cleaning), and `idrak data audio --clean` writes cleaned files.

Processing (plan 31): `AudioProcessors`, a registry of `IAudioProcessor`s, and `AudioPipeline`, a chain of them written
as text the way images have `ImageTransformPipeline`: `highpass=80, denoise=1.5 quantile=0.2, trim, normalize=-20 peak=0.9`
(steps comma-separated, each a name, an optional value and space-separated `key=value` options; JSON as a string or an
array of strings and `{"name", "value", ...}` objects). A processor declares its inputs (`AudioProcessorParameter`: kind,
range, unit, default), from which its keys and checks follow, and whether it is **live**: a live one opens an
`IAudioStage` that takes blocks in order, keeping its state between them (implement `Open`); a whole-recording one
implements `Apply` and its stage gives the result when flushed. `Apply` runs a recording (consecutive live steps over one
copy, written in place), `Open` gives one stage for blocks, and `Wrap(IAudioInput)` cleans live sound as it comes,
refusing a step that needs the whole recording. Built in: "highpass" (live), "denoise", "trim" and "normalize" (whole
recording), which the cleaning stages above run through with the same samples as before; the filters "gain", "dc",
"lowpass", "bandpass", "notch", "dehum" (mains hum and its harmonics), "eq", "lowshelf", "highshelf", "presence" (the
clarity of speech: a wide boost of the consonants' band), "preemphasis", "mono" and "resample" (live, Butterworth orders
on highpass, lowpass and bandpass); the dynamics "gate",
"compress", "limit" (look-ahead, never over its ceiling), "agc" and "loudness-live" (live), "loudness" (BS.1770 / R128
integrated, to a target LUFS under a true-peak ceiling) and "peaknorm" (whole recording). After a comma a registered
name starts a new step, so an option is best written after a space (`normalize=-20 peak=0.9`). Anyone adds one with
`AudioProcessors.Register`:

```csharp
sealed class Gain : IAudioProcessor
{
    public string Name => "gain";
    public string Summary => "a gain in dB";
    public bool Live => true;
    public IReadOnlyList<AudioProcessorParameter> Parameters { get; } =
        [new("", "Gain", AudioProcessorInput.Number) { Min = -60, Max = 24, Unit = "dB" }];

    public IAudioStage Open(AudioProcessorStep step, int sampleRate, int channels) =>
        new Stage((float)Math.Pow(10, step.Number() / 20), sampleRate, channels);

    sealed class Stage(float gain, int sampleRate, int channels) : IAudioStage
    {
        public int SampleRate => sampleRate;
        public int Channels => channels;
        public AudioData Process(AudioData block) { foreach (ref float x in block.Samples.AsSpan()) x *= gain; return block; }
        public AudioData Flush() => new([], channels, sampleRate);
    }
}

AudioProcessors.Register(new Gain());
var clean = AudioPipeline.Parse("highpass=80, gain=6").Apply(recording);
```

`SpeechTrainingOptions.Processing` runs a pipeline over every utterance after the cleaning stages (its text in the
options' JSON as `"processing"`, in the checkpoints and in the feature cache key, so processed rows stay apart).

The command line (`idrak help audio`): `idrak transcribe` (`--audio-clean PIPELINE` on files and `--live`),
`idrak speak` (`--audio-clean`), `idrak voice add|list|show|remove|export|import`, `idrak data audio`, `idrak data mel`,
`idrak audio process IN -o OUT --pipeline P` and `idrak audio processors`, `--clean` taking a pipeline on `pretrain`,
`voice train` and `data audio`, and `--audio` on `idrak run` and `idrak chat`. The other phases are in
plans/done/21-audio.md.

```csharp
var audio = AudioCodecs.Load("clip.wav", channels: 1);   // mono, as the file's rate
AudioEncoders.Save("copy.wav", audio, new AudioEncodeOptions { BitsPerSample = 24 });
```
