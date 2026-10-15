# IdrakAudio: speech models made from scratch, as a plug-in (plan 21, S6)

Idrak's own audio family (decision 1 of plan 21), registered in `SpeechFamilies` by `IdrakAudioPlugin.Register()` (or
`-P Idrak.IdrakAudio.dll` in the command line). Built so far: the **"ctc"** task, a speech recognizer trained from
scratch with `SpeechTrainer` (src/Idrak.Audio). The "seq2seq", "audio-language" and "tts" tasks of the plan come later; a
spec asking for one is refused by name.

```csharp
IdrakAudioPlugin.Register();
var created = SpeechModels.Create("models/my-asr", spec, tokenizer);          // a folder SpeechModels.Load reads
using var model = (ICtcSpeechModel)SpeechModels.Load("models/my-asr", new SpeechModelOptions { Device = device });
var train = SpeechSources.Create(new SpeechSourceOptions { Kind = "transcripts", Path = "data/train.jsonl" });
var result = SpeechTrainer.Train(model, train, evaluation: null, new SpeechTrainingOptions { Steps = 2000, Augmentations = ["speed", "spec-mask"] }, "runs/asr");
```

## The "ctc" model

Log-mel features (`AudioFeatureExtractors`, the model's `features` options) → a subsampler of `subsample_layers`
convolutions of stride 2 over time (kernel `subsample_kernel`) → `sinusoidal` positions (or `none`) → `layers`
transformer encoder blocks (`dim`, `heads`, `ffn_dim`, `activation`, `norm_epsilon`, `dropout`), optionally with a
Conformer-style depthwise convolution module (`convolution_module`, `convolution_kernel`) → a linear head to
`classes` (the tokenizer's vocabulary and the blank, `blank_id`). Every shape is a config.json key; `init_scale` scales
the initial weights and `max_seconds` bounds one utterance. Read back, it is an `ISpeechRecognizer` decoding through
`CtcDecoders` ("greedy" or "beam", a setting), batched, chunked and streamed alike.

## Training (SpeechTrainer)

`SpeechTrainingOptions`, all settings: steps, `BatchSeconds` (audio seconds per micro-batch, length-bucketed; halved and the
step repeated when the device runs out of memory), gradient accumulation, learning rate with warmup and the schedule
(`FineTuningSchedules`), the optimizer (`FineTuningOptimizers`, host optimizer optional), weight decay, gradient
clipping, bfloat16 activations, checkpointing, `SaveEvery` / `KeepCheckpoints`, `EvaluateEvery` with a `TuningMetrics`
metric (CER by default, WER too) and the decoder, augmentations (`AudioAugmentations`: speed, noise, spec-mask, seeded
per utterance and pass) and the seed. Checkpoints are `checkpoint-STEP/` folders (the model, the optimizer state,
train-state.json) that `SpeechTrainer.Resume` continues bit for bit on the CPU and `SpeechModels.Load` reads as models.
Features are extracted ahead of each step and cached (`FeatureCaches`), so a second pass extracts nothing; utterances
with fewer frames than labels are counted and left out.

## Checked (`IDRAK_FILTER="audio train"`)

On `tests/Idrak.Tests/data/audio/ctc/` (models from PyTorch with torch's CTC loss and torchaudio's forced alignment): the
step-1 loss within 1e-4 and every gradient within 1e-4 of each tensor's largest, three SGD steps, Viterbi alignment
exactly; a padded batch equal to each utterance alone; create and load; resume bit for bit; the micro-batch halved on
a forced out-of-memory; and a 200-step CPU run that halves the loss with a CER under 0.5.

## Not done yet

The "seq2seq", "audio-language" and "tts" tasks, `PretrainedModel.Create` with `"audio"`, distillation, and the
command line for training (`idrak create` / `pretrain` with the ctc task: S7 after this).
