# Whisper: a Whisper-style speech recognition family as a plug-in (plan 21, phase S4)

`Idrak.Audio.Plugins.Whisper` (plan 26 moved it here from `samples/WhisperSpeech`); `WhisperPlugin.Register()` registers
it alone, `AudioPlugins.RegisterAll()` with the package's other families, and the `idrak` tool registers it at startup.

Reads a checkpoint saved by transformers (`WhisperForConditionalGeneration`) into the library's speech contracts
(`ISpeechRecognizer`, `SpeechFamilies`, `SpeechModels.Load` in `Idrak.Audio`). Nothing of the family is in the library:
without this plug-in a Whisper checkpoint is refused with "Speech family 'WhisperForConditionalGeneration' is not
registered".

- **Features**: the library's "log-mel" extractor with the checkpoint's `preprocessor_config.json` (rate, n_fft, hop,
  mels, the 30 s segment); sound at another rate is resampled, several channels mixed to one.
- **Encoder** (`WhisperEncoder`): two Conv1d (as the library's Conv2d of (1, 3)) with the exact GELU, the stored
  positions, pre-norm encoder layers through `TransformerEncoderLayer` (attention over the 1500 frames through the
  library's attention paths, never a stored [1500, 1500] score matrix), a final LayerNorm.
- **Decoder** (`WhisperDecoder`): the library's `DecoderBlock` with a `CrossAttention` over the encoder's output
  (projected once per segment into every layer's keys and values and held in the `DecodingContext`), learned positions,
  the head tied to the embeddings; the key/value cache as any decoder's (float32, int8 or bfloat16).
- **Decoding**: greedy or beam search (`Idrak.Generation.BeamSearch`, transformers' semantics: 2·beams candidates a
  step, early stopping, the length penalty), from `<|startoftranscript|><|lang|><|task|>[<|notimestamps|>]`; the
  checkpoint's `suppress_tokens` and `begin_suppress_tokens`; with timestamps, the family's timestamp rules (a timestamp
  opens and closes each phrase, never back in time) and segments read from the timestamp tokens.
- **Language**: given (`TranscriptionOptions.Language`) or detected (one decoder step over the language tokens).

## Fine-tuning (plan 29)

`WhisperRecognizer` is also an `ISequenceSpeechModel`, so `SpeechTrainer` fine-tunes it with the "seq2seq" objective
(and `idrak pretrain WHISPER --data transcripts.jsonl --audio clips --language ar` does from the command line), as
transformers trains `WhisperForConditionalGeneration`:

- **Labels**: the prompt after `<|startoftranscript|>` (`<|lang|><|task|><|notimestamps|>`), then the text's tokens; the
  decoder reads the start token and the labels and is taught each label and then the end token. The language and task
  come from `SpeechModelOptions.Language` and `.Task`, else from `generation_config.json`'s `language` and `task`; a
  multilingual checkpoint with no language anywhere refuses to make labels. The same language is the transcription
  default when a request names none.
- **Sound**: the features the recognizer reads (each clip padded to the 30 s segment with Whisper's own silence), so
  training sees what transcription sees. A clip longer than the segment is left out (its sound would be cut, its text
  not), counted in the run's notes.
- **Weights**: training needs float32 projections: load with `SpeechModelOptions.Weights = EncoderWeights.Float32` (the
  CLI does); a model of packed float16 or bfloat16 weights refuses to give its network. `SpeechForwardOptions.Checkpointing`
  checkpoints the encoder's layers.
- **Saving**: `Save` (each checkpoint) writes a folder transformers and `SpeechModels.Load` both read: `config.json`,
  `generation_config.json` with the language and task, `model.safetensors` in float32 under transformers' names (the
  packed query, key and value split again, one tensor in memory at a time), and every other file of the source folder.

```csharp
WhisperPlugin.Register();
using var model = (WhisperRecognizer)SpeechModels.Load("models/whisper-small",
    new SpeechModelOptions { Device = device, Weights = EncoderWeights.Float32, Language = "ar" });
var result = SpeechTrainer.Train(model, new SpeechSource(SpeechSources.Transcripts, ["train.jsonl"]),
    new SpeechSource(SpeechSources.Transcripts, ["dev.jsonl"]), new SpeechTrainingOptions { Steps = 500, LearningRate = 1e-5f }, "runs/whisper-ar");
```

Not yet: timestamps in the labels, a language per utterance, adapters (LoRA) on the projections, a frozen encoder.

## The files a checkpoint needs

| File | What |
|---|---|
| `config.json` | `architectures: ["WhisperForConditionalGeneration"]`, `d_model`, the layer, head and feed-forward counts, `num_mel_bins`, `max_source_positions`, `max_target_positions`, `vocab_size`, `activation_function` |
| `model.safetensors` (or an index) | transformers' names: `model.encoder.*`, `model.decoder.*`, `proj_out.weight` when not tied |
| `tokenizer.json` | the byte-level BPE with the special tokens as added tokens |
| `preprocessor_config.json` | `sampling_rate`, `n_fft`, `hop_length`, `feature_size`, `chunk_length` |
| `generation_config.json` | `decoder_start_token_id`, `eos_token_id`, `lang_to_id`, `task_to_id`, `no_timestamps_token_id`, `suppress_tokens`, `begin_suppress_tokens` (missing: read from the tokenizer's added tokens and config.json) |

## Options (`TranscriptionOptions`)

`Language`, `Task`, `Timestamps`, `TimestampRules`, `Beams`, `LengthPenalty`, `MaxTokens`, `MinTokens`, `ChunkSeconds`
and `OverlapSeconds` (long sound in chunks), `CarryText` and `Prompt` (through `<|startofprev|>` when the checkpoint
has it), `SuppressTokens`. Not in this phase: sampling with a temperature and the temperature fallback ladder
(`Temperature` above 0 is refused with a message), batches of several segments through the encoder at once.

## Use

```csharp
WhisperPlugin.Register();                                     // or AudioPlugins.RegisterAll()
using var recognizer = SpeechModels.Load("models/whisper-tiny", new SpeechModelOptions { Device = device });
var result = recognizer.Transcribe(AudioCodecs.Decode("talk.wav"), new TranscriptionOptions { Language = "ar", Beams = 4, Timestamps = true });
foreach (var segment in result.Segments)
    Console.WriteLine($"[{segment.Start:F2} - {segment.End:F2}] {segment.Text}");
```

## Tests

`tests/Idrak.Tests/AudioAsrTests.cs` ("audio asr:") checks the plug-in on the fixture `tests/Idrak.Tests/data/audio/tiny-whisper`
(a seeded tiny model whose references came from transformers): the encoder output and the first step's logits within
1e-4, the greedy, beam-of-4 and timestamp ids (greedy and beam-of-4) exactly, Transcribe's chunks and streaming, a batch
through the encoder, and the testing kit's `SpeechRecognizerSuite`. On the CPU:

```
IDRAK_DEVICES=cpu IDRAK_FILTER="audio asr" dotnet run -c Release --no-build --project tests/Idrak.Tests
IDRAK_DEVICES=cpu IDRAK_FILTER="audio whisper train" dotnet run -c Release --no-build --project tests/Idrak.Tests
```

**Memory.** With gradients off, the encoder runs each layer in a scope of its own and keeps only its output, so the
device holds one layer's attention scores (heads × 1500 × 1500 for a 30 s segment) at a time. For a whisper-small-shaped
model transcribing 4.5 s on the CPU, the call peaked 6620 MB above the weights when every layer's intermediates stayed to
the end, and 863 MB now (the owner's CUDA run with whisper-small had reported 7.37 GB).

## Trying it on a real checkpoint

`idrak transcribe` transcribes WAV, FLAC or MP3 files through the plug-in (the tool registers it), as text, JSON, SRT
or WebVTT; `-v` prints the cold and warm stage timings and the peak device memory. `samples/WhisperSpeech/transcribe.cs`
is the prototype it matches, an app on this package.

```powershell
cd D:\Projects\Idrak
dotnet run --project src\Idrak.Cli -c Release -- pull openai/whisper-small
$whisper = (Get-ChildItem "$env:USERPROFILE\.cache\idrak\downloads\huggingface\models\openai\whisper-small" -Directory | Sort-Object LastWriteTime | Select-Object -Last 1).FullName
dotnet run --project src\Idrak.Cli -c Release -- transcribe $whisper sample-en.wav -l en --timestamps -d cuda -v
dotnet run --project src\Idrak.Cli -c Release -- transcribe $whisper sample-en.wav -o sample-en.srt -d cuda
```
