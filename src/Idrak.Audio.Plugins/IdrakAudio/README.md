# IdrakAudio: speech models made from scratch, as a plug-in (plan 21, S6)

`Idrak.Audio.Plugins.IdrakAudio` (plan 26 moved it here from `samples/IdrakAudio`).

Idrak's own audio family (decision 1 of plan 21), registered in `SpeechFamilies` and `SpeechSynthesisFamilies` by
`IdrakAudioPlugin.Register()` (or `AudioPlugins.RegisterAll()`; the `idrak` tool registers it at startup). Built so far: the **"ctc"** and
**"seq2seq"** tasks, speech recognizers trained from scratch with `SpeechTrainer` (src/Idrak.Audio) and the matching
objective (`SpeechTrainingObjectives`: "ctc", "seq2seq"), the **"audio-language"** model (below), and the **"tts"**
task, a synthesizer trained from scratch with `SpeechSynthesisTrainer` and the "durations" objective
(`SpeechSynthesisObjectives`). A spec asking for a task not built is refused by name.

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

## The "seq2seq" model

The same encoder without the head (its normed states are the memory), then a text decoder made of the library's layers:
the token embedding, `max_target_positions` learned positions, `decoder_layers` pre-norm `DecoderBlock`s
(`decoder_dim`, `decoder_heads`, `decoder_ffn_dim`; causal self-attention, `CrossAttention` over each utterance's own
encoder steps, the feed-forward), a final layer norm and the head (`tie_embeddings`: the embedding's table, or its own
weight). `classes` are the tokenizer's tokens and one more, the default `start_token` and `end_token` (both settings).
Training reads the start token and the labels and teaches each next label and then the end token: each utterance's mean
cross-entropy through the fused `TokenCrossEntropy`, averaged over the batch (an utterance's loss is the same in any
batch). Reading uses `BeamSearch` on the key/value cache, greedy or with `TranscriptionOptions.Beams`; evaluation in
training uses the trainer's decoder setting ("greedy", or "beam" with `BeamWidth` beams). Activation checkpointing covers
the encoder's blocks (a text's tokens are far fewer than its sound's steps).

## The "audio-language" model (IdrakAudioLanguageFamily)

A chat model that hears: a Llama-style text decoder (architecture `IdrakAudioForConditionalGeneration`, its spec as
`text_config`, its tensors under `language_model.`) with an audio part, made with `PretrainedModel.Create` and
`CreateOptions.Parts["audio"]` (the library's "audio" model part over this family): the IdrakAudio encoder over log-mel
(`audio_config`, tensors under `audio_tower.`) and a projector (`audio_projector`: every `stack` encoder steps joined,
"mlp" or "linear", tensors under `audio_projector.`). A sound is one block of soft tokens: the chat template's audio
marker (`audio_token`, `<audio>` by default) expanded to `begin_audio`, the soft tokens and `end_audio` (each optional,
each a single token of the tokenizer); `audio_attention` says how they attend ("causal"). The `Pretrainer` trains it on
records with sounds (a "records" source with `Audios`), its stages naming `audio-encoder`, `audio-projector` or the
decoder's parts: an "align" stage (the audio part alone) then "joint" (`all`); the trained part is kept between stages
and in every checkpoint. `ChatGenerator` answers about sounds with the model's part (`model.OpenPart("audio")`).

## The "tts" model

A text-to-speech model made with `SpeechSynthesisModels.Create(folder, spec, tokenizer)` (the spec's `"task": "tts"`)
and read back with `SpeechSynthesisModels.Load` as an `ISpeechSynthesizer` (`IdrakAudioTtsModel`, an
`ITrainableSpeechSynthesizer`); `SpeechModels` refuses it, naming the synthesis loader. The text's labels (the
tokenizer's ids, `classes` its tokens; a character tokenizer is the natural one) → the token embedding and `sinusoidal`
positions → `layers` pre-norm blocks (`dim`, `heads`, `ffn_dim`, the same block as the recognizers', the optional
convolution module too) over each text's own labels → a layer norm: the text's states. A duration predictor reads them:
two (1, `duration_kernel`) convolutions of `duration_filters` channels over the labels, each with a ReLU and a layer
norm (the padding zeroed before each), and a linear to log(frames + `duration_offset`). Length regulation
(`LengthRegulation` in Idrak.Audio: one gather of rows by indices made on the host) repeats each label's state for its
frames (the alignment's in training, the predicted ones over `SpeechSynthesisOptions.Rate` in speech); a projection to
`decoder_dim` when it differs from `dim`; positions; `decoder_layers` pre-norm blocks (`decoder_dim`, `decoder_heads`,
`decoder_ffn_dim`) over each utterance's own frames; a layer norm and a linear to `mels`: a log-mel spectrogram in the
layout of the model's `features`.

```csharp
IdrakAudioPlugin.Register();
using var ctc = (ICtcSpeechModel)SpeechModels.Load("models/my-asr");                       // the owner's trained CTC model
SpeechAlignments.AlignToFile(ctc, train, "data/alignments.jsonl");                           // what `idrak data audio --align MODEL` runs
using var tts = (IdrakAudioTtsModel)SpeechSynthesisModels.Create("models/my-tts", spec, tokenizer);   // the CTC model's tokenizer
var result = SpeechSynthesisTrainer.Train(tts, train, null, new SpeechTrainingOptions { Steps = 20000, Durations = "data/alignments.jsonl" }, "runs/tts");
using var voice = SpeechSynthesisModels.Load(result.Checkpoint!);
var (audio, stats) = voice.Speak("مرحبا");                                                 // griffin-lim by default
```

**Training.** The durations come from the owner's CTC model: `SpeechAlignments.Align` takes each utterance's best path
(`CtcAlignment.Viterbi`) and the alignments file holds one JSON line per utterance (`audio`, `text`, `labels`, `starts`,
`durations`, `steps`, `step_seconds`, `log_probability`). For training, label i covers the steps from its start to the
next label's start (the silence after a label is its own, the leading silence the first label's), scaled to the
utterance's feature frames, so the durations sum to the frames (`SpeechAlignment.FrameDurations`). The synthesizer must
read the text into as many labels as the CTC model did (the same tokenizer); an utterance whose alignment is missing or
of another length is counted and left out. The loss ("durations"): each utterance's mean absolute log-mel error over
its frames and mels, plus `SpeechTrainingOptions.DurationLossWeight` (1) times its mean squared error of the log
durations over its labels, each averaged over the batch. `SpeechSynthesisTrainer` is `SpeechTrainer`'s loop (the same
options, batches, feature cache, out-of-memory halving, checkpoints resumed bit for bit), with `Durations` naming the
alignments file; feature augmentations ("spec-mask") are refused, since the features are the targets. The evaluation
score is the mean loss of the evaluation utterances that have alignments.

**Speaking.** `Speak` and `SpeakStream` split the text by `SpeechSynthesisOptions.MaxCharacters` (`SpeechSynthesisText`),
leave out the characters the tokenizer has no token for (`DropUnknownCharacters`, true; false refuses them), predict
each piece's durations, make its spectrogram in one device pass and turn it into sound through a `Vocoders`
vocoder: the call's `Vocoder`, else one the app set on the model (`IdrakAudioTtsModel.Vocoder`, a `GriffinLimVocoder` of
more iterations say), else the config's `vocoder`, else the library's **"griffin-lim"**. A vocoder whose mel bands or
rate differ from the model's features is refused naming both; a neural vocoder such as the VITS plug-in's HiFi-GAN can be
used when it is registered in `Vocoders` for this mel layout (the VITS decoder reads its own flow's latents, not a log-mel,
so it does not fit a model trained here unless one is trained for it).

**What it sounds like, plainly.** Without a neural vocoder, this model sounds like Griffin-Lim: the phases are guessed
by iteration, so speech comes out intelligible at best but buzzy and metallic, never natural. Training a HiFi-GAN for
this mel layout (adversarial training) is a later plan; until then the "tts" task is a working, trainable pipeline and a
baseline, not a voice to ship.

## Training (SpeechTrainer)

`SpeechTrainingOptions`, all settings: steps, `BatchSeconds` (audio seconds per micro-batch, length-bucketed; halved and the
step repeated when the device runs out of memory), gradient accumulation, learning rate with warmup and the schedule
(`FineTuningSchedules`), the optimizer (`FineTuningOptimizers`, host optimizer optional), weight decay, gradient
clipping, bfloat16 activations, checkpointing, `SaveEvery` / `KeepCheckpoints`, `EvaluateEvery` with a `TuningMetrics`
metric (CER by default, WER too) and the decoder, augmentations (`AudioAugmentations`: speed, noise, spec-mask, seeded
per utterance and pass) and the seed. Checkpoints are `checkpoint-STEP/` folders (the model, the optimizer state,
train-state.json) that `SpeechTrainer.Resume` continues bit for bit on the CPU and `SpeechModels.Load` reads as models.
Features are extracted ahead of each step and cached (`FeatureCaches`), so a second pass extracts nothing; utterances
the objective cannot train on (ctc: fewer frames than labels; seq2seq: more labels than the decoder's positions) are
counted and left out.

## Checked (`IDRAK_FILTER="audio train"`, `"audio seq2seq"`, `"audio language"`, `"audio tts train"`)

On `tests/Idrak.Tests/data/audio/ctc/` (models from PyTorch with torch's CTC loss and torchaudio's forced alignment): the
step-1 loss within 1e-4 and every gradient within 1e-4 of each tensor's largest, three SGD steps, Viterbi alignment
exactly; a padded batch equal to each utterance alone; create and load; resume bit for bit; the micro-batch halved on
a forced out-of-memory; and a 200-step CPU run that halves the loss with a CER under 0.5. On
`tests/Idrak.Tests/data/audio/seq2seq/` (two decoders from PyTorch, tied and untied, the second narrower than the
encoder): the step-1 loss and every gradient as for ctc, the decoder's states in a padded batch equal to each utterance
alone, torch's greedy texts exactly, three SGD steps; create and load, transcription and the recognizer suite; resume
bit for bit; and a 200-step CPU run that halves the loss with a CER under 0.5. On `tests/Idrak.Tests/data/audio/audio-language/` (a Llama
decoder with this encoder and projector in PyTorch): the log-mels, soft tokens, ids and blocks; joint and align
pretraining's step-1 loss and gradients and three steps' losses and weights; create, stages resumed, an answer.

On `tests/Idrak.Tests/data/audio/tts/` (the synthesizer in PyTorch, `make_tiny_audio.py --sections tts`, with a
zero duration and a decoder wider than the text encoder): the step-1 log-mel, log durations, both loss parts and every
gradient as for ctc, each utterance alone equal to the padded batch, three SGD steps; length regulation exactly; create,
load, speak through griffin-lim at the regulated length, stream and vocoder choice; alignments from a CTC model equal to
`CtcAlignment.Viterbi`, the file read back exactly; resume bit for bit; and a 200-step CPU run on tone speech that halves
the loss and the log-mel error and predicts a letter's frames within a third.

## Not done yet

Distillation, and the command line for training (`idrak create` / `pretrain` with the audio tasks).
