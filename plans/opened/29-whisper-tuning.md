# Plan 29: fine-tuning Whisper on your own speech

**Status:** planned and built 2026-10-24 against `main` 150530e0, CPU-checked; the owner's runs on the card are open
(commands at the end).

## Why

`idrak voice train` makes a voice that speaks like the owner; nothing makes a recognizer that hears the owner better.
Plan 21's S8 trained CTC, seq2seq and audio-language models from scratch on four minutes of speech: they memorized the
clips (held-out CER 0.84). The working way with little data is to fine-tune a pretrained recognizer, and the library
already holds every piece but one: the `seq2seq` objective (`Seq2SeqObjective`), the trainer (`SpeechTrainer`:
batches, accumulation, checkpoints, resume, held-out scoring) and `idrak pretrain MODEL_FOLDER --data ... --audio ...`,
which trains any speech folder whose family gives an `ITrainableSpeechModel`. Whisper's recognizer gives only
`ISpeechRecognizer`, so the trainer refuses it.

## Design

1. **Whisper implements `ISequenceSpeechModel`** (in `Idrak.Audio.Plugins`, `Whisper/`; no family code in the
   library). The trainer's contract maps onto transformers' Whisper training exactly:
   - The decoder starts from `<|startoftranscript|>`; the labels are the prompt's other tokens
     (`<|lang|><|transcribe|><|notimestamps|>`) then the text's tokens, and the end token is taught last. This is what
     transformers' `WhisperTokenizer` gives as labels with the language and task set (the text encoded as given).
   - `Encode` runs the encoder on the features the recognizer reads: the checkpoint's extractor, which pads or cuts a
     clip to its 30 s segment with Whisper's own silence values, so training sees what transcription sees. Every
     utterance is 3,000 frames and 1,500 encoder steps; nothing is masked (Whisper attends to its whole segment).
   - `Hidden` and `Head` are the decoder's body and its tied head through `EncoderMemory`, as the IdrakAudio
     sequence model does; `Generate` decodes with the recognizer's own beam search and suppressed tokens.
   - `SpeechForwardOptions.Checkpointing` checkpoints the encoder's layers.
   - The encoder's sinusoidal positions stay fixed (no gradient computed): OpenAI's Whisper holds them as a buffer and
     transformers 4 froze them. transformers 5.19 computes a gradient for them; it is the one tensor the test does not
     compare.
   - `Save` writes a folder transformers and `SpeechModels.Load` both read: config.json, model.safetensors (float32)
     under transformers' names (the packed query, key and value split back; the key projection has no bias in
     Whisper, and the zeros the packing adds receive no gradient, as softmax ignores a constant on every key), the
     tokenizer, preprocessor and every other non-weight file copied, and generation_config.json with the training
     language and task.
2. **The language and the task are settings, not constants.** `SpeechModelOptions.Language` and `.Task` (new, for any
   family whose prompt names them): the labels' language token and the transcription default when a request gives
   none. Without them the checkpoint's generation_config.json `language` and `task` are read (transformers' behaviour);
   a multilingual checkpoint with no language anywhere refuses to make labels, naming the three places to give one.
   The CLI: `idrak pretrain MODEL --language ar [--task transcribe]`.
3. **Training needs float32 weights.** A checkpoint kept as stored (float16 or bfloat16) has packed projections that
   do not train; the model refuses to hand out its network for training then, naming `SpeechModelOptions.Weights`.
   The CLI's speech training loads with `Weights = Float32`.
4. **A clip longer than the features' segment is left out** (`SpeechTrainer`, any model whose extractor pads to a
   segment): the extractor would cut its sound while its transcript stays whole. The run's notes count them. Models
   whose features are not padded (IdrakAudio, VITS) are not affected.

Not in this plan: timestamps in the labels, per-utterance languages in the transcripts, LoRA on Whisper's
projections (the speech trainer trains every parameter), the encoder frozen as an option.

## Tests (CPU, `IDRAK_FILTER="audio whisper train"`)

- The loss and every parameter's gradient of a two-utterance batch match transformers' `WhisperForConditionalGeneration`
  on tiny-whisper within 1e-4 (`tools/audio/make_tiny_audio.py` section `whisper-train`), the encoder's fixed positions
  aside.
- Batches count each clip as the whole segment it runs as (a 2 s clip is 30 s of the model's work), so an automatic
  batch holds what the device measured; models whose features are not padded batch as before.
- Labels: the prompt's tokens then the text's; `Text` drops them; a multilingual checkpoint without a language refuses.
- Save and load back: every stored tensor under its name, bit for bit; config, tokenizer and preprocessor copied;
  generation_config.json carries the language; the logits after loading are the same.
- A short `SpeechTrainer` run lowers the loss; its checkpoint loads as a Whisper recognizer and transcribes.
- A clip longer than the segment is left out with a note.

## The owner's runs

The GPU check of the new group, then a fine-tune of whisper-small on the S8 recordings (the same clips `voice train`
used), its held-out CER printed at each evaluation, and one held-out clip read by the base model and the tuned one.

```powershell
cd D:\Projects\Idrak
git pull origin main
dotnet build -c Release tests\Idrak.Tests
$env:IDRAK_DEVICES = "cuda"; $env:IDRAK_FILTER = "audio whisper train"; dotnet run -c Release --no-build --project tests\Idrak.Tests
$idrak = "src\Idrak.Cli\bin\Release\net10.0\Idrak.Cli.dll"
dotnet $idrak pretrain openai/whisper-small --data D:\Recordings\ahmed-phone\ar-s1 --eval D:\Recordings\ahmed-phone\ar-s1-val --language ar --metric cer --steps 500 --lr 1e-5 --eval-every 100 -o work\whisper-ahmed
$clip = (Get-ChildItem D:\Recordings\ahmed-phone\ar-s1-val\*.wav)[0].FullName
dotnet $idrak transcribe openai/whisper-small $clip --language ar
dotnet $idrak transcribe work\whisper-ahmed\checkpoint-500 $clip
```
