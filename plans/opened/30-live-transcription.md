# Plan 30: live transcription (`idrak transcribe --live`)

**Status:** planned and built 2026-10-24 against `main` 2a31e06b, CPU-checked (the ALSA driver against ALSA's `null`
device); the Windows microphone and the owner's live runs are open (commands at the end).

## What

`idrak transcribe MODEL --live` writes what is said into the microphone as it is said; `--live -` does the same for
sound piped in on standard input (any recorder: ffmpeg, arecord, sox), and `--live FILE` streams a recording through
the same path (to try the settings on known speech). Any registered speech recognizer works; nothing is Whisper's.

## Design

1. **Capture is a contract** (`Idrak.Audio.Abstractions`): `IAudioCapture` (a backend: whether it can run here and
   why not, its input devices, `Open`) gives an `IAudioInput` (the sound as blocks, `IAsyncEnumerable<AudioData>`),
   registered in `AudioCaptures` (a `SlotTable`). Built in, each loading the operating system's own audio library at
   run time as the GPU devices load their drivers (no native files shipped, no outside program):
   - `winmm` (Windows): waveIn through winmm.dll, the default device or one by number, at the rate asked (Windows
     converts).
   - `alsa` (Linux): libasound.so.2's PCM capture, `default` or any ALSA name (`hw:1,0`, `pulse`), devices from its
     name hints.
   macOS has no built-in capture yet: `--live` there names the pipe instead (`ffmpeg -f avfoundation ... | idrak
   transcribe MODEL --live -`). A plug-in can register one (`AudioCaptures.Register`).
2. **A pipe is an input too**: `PcmStreamInput` reads a stream as it arrives: a WAV header when the stream starts with
   one (its size field ignored, as a pipe's is unknown), else raw PCM (`s16le` or `f32le`, the rate and channels given).
3. **Live transcription is the library's, over any recognizer** (`LiveTranscriber`, `Idrak.Audio`): the sound since the
   last finished segment is kept; every `StepSeconds` of new sound it is transcribed again whole, and the words the last
   `Agreement` transcriptions agree on from the start are committed (the commit policy is a contract,
   `ILiveCommitPolicy`, registered in `LiveCommitPolicies`; built in: `agreement`, Whisper-streaming's local agreement).
   A segment ends at `SilenceSeconds` of quiet after speech, at `WindowSeconds`, or at the end of the input; its
   whole text is committed then, and the next segment is given the last text as its prompt. Stretches with no speech
   are never transcribed (the quiet is measured per 20 ms frame: under the loudest frame heard by `SilenceDecibels`,
   and under `SilenceFloorDecibels` before anything loud), so silence does not turn into invented text. The input is
   read on its own task while the model works, so no sound is lost when a step takes longer than the step. `KeepUp`
   (on for live sound) reads everything waiting before a step, so a slow step is followed by one over all that arrived;
   off (a recording given faster than it plays, `--live FILE`, `--every-step`) it steps through the sound at its own pace.
4. **The command line**: `--live` with no AUDIO records from the microphone (`--capture NAME`, `--input DEVICE`;
   `--list-inputs` lists the backends and their devices), `-` reads the pipe (`--input-rate`, `--input-channels`,
   `--input-format`), a file is streamed. The live line shows the segment as heard so far (committed words, then the
   tentative ones); each finished segment is printed as a line with its times. Ctrl+C stops the input: the last segment
   is finished, then `-o` writes the whole transcript (text, JSON, SRT or WebVTT) and `-j` prints it. `--step`,
   `--window`, `--silence`, `--silence-db`, `--silence-floor-db`, `--agreement` set the library's options; every
   transcription option of the command applies to each step.

Not in this plan: word-level timestamps for the committed words, a macOS capture, speaker turns.

## Tests (CPU, `IDRAK_FILTER="audio live"`)

- `PcmStreamInput`: raw `s16le` and `f32le`, stereo, a WAV header with an unknown size, blocks split at odd bytes.
- `LiveTranscriber` on a scripted recognizer: words committed when two transcriptions agree, a segment ended by quiet
  and by the window, the last one at the end, quiet never transcribed, the prompt carried; on tiny-whisper over a
  file streamed in blocks.
- The registry: `winmm` and `alsa` registered, the platform's default; ALSA's `null` device recorded through the
  driver for a second (where libasound is installed).
- `idrak transcribe tiny-whisper --live FILE` prints its segments and writes `-o` SRT; `--list-inputs`.

## The owner's runs

```powershell
cd D:\Projects\Idrak
git pull origin main
dotnet build -c Release tests\Idrak.Tests
$env:IDRAK_DEVICES = "cuda"; $env:IDRAK_FILTER = "audio live"; dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress --failures-only --failures-file fails.txt
$idrak = "src\Idrak.Cli\bin\Release\net10.0\Idrak.Cli.dll"
dotnet $idrak transcribe --list-inputs
dotnet $idrak transcribe openai/whisper-small --live --language ar -o work\live.srt
```
