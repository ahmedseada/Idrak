# idrak-tune

One command-line tool for every fine-tuning job: LoRA / QLoRA adapters for any pretrained model Idrak loads
(Hugging Face ids, folders, .gguf files, Ollama models), trained on any dataset `idrak-data` reads, then evaluated, chatted
with and exported. Nothing in it is specific to a model family or to an application: a project that needs a tuned model
runs idrak-tune with its data instead of writing its own tuner.

The same tool is `idrak tune` in the `idrak` command-line tool (src/Idrak.Cli), which runs this code; this package stays
as a forwarder with the same commands and options.

Install from nuget.org (needs the .NET SDK): `dotnet tool install --global Idrak.FineTuning.Cli`, then `idrak-tune --help`.
From this repository:

```
dotnet run -c Release --project src/Idrak.FineTuning.Cli -- train <model> <data…> --out <dir>
dotnet run -c Release --project src/Idrak.FineTuning.Cli -- evaluate <adapter dir> <data…> --eval-fraction 0.02
dotnet run -c Release --project src/Idrak.FineTuning.Cli -- chat <adapter dir> "a message"
dotnet run -c Release --project src/Idrak.FineTuning.Cli -- export <adapter dir> --out <merged dir>
```

- **Data.** Conversations train the assistant's turns (OpenAI / Hugging Face messages, ShareGPT, Alpaca, question /
  answer pairs are recognized); text rows train every token. Columns map into a conversation with templates:
  `"data.csv?user={question}&assistant={answer}"`, and `--system` adds an instruction to conversations without one.
  `--eval-fraction` holds out part of the data; `evaluate` with the same data and fraction scores that part.
- **Fixed answers.** When every answer is one of a known list (labels, yes / no, multiple choice), `evaluate --choices
  a,b,c` (or `--choices auto`: the distinct answers in the data) scores each as the model's answer after the prompt and
  takes the most likely, generating nothing (the library's `AnswerScorer`: each prompt runs once for all its answers).
  It prints accuracy and recall per answer, the base model and the adapter side by side; `--out F.jsonl` writes each
  conversation's answer and every probability, for an application's own reports. `--samples 0` scores all rows.
- **Preference training.** `--loss dpo`, `--loss orpo` or `--loss simpo` trains on preference rows instead of
  conversations: TRL's layouts with a prompt and a chosen and a rejected answer (message lists or strings, or chosen and
  rejected as whole conversations that share their prompt). Only the answers are trained. DPO's reference is the model
  with its adapter disabled, so no second copy is loaded. `--beta` sets DPO's beta (0.1), ORPO's lambda (0.1) or
  SimPO's beta (2), `--margin` SimPO's target margin (1).
- **Optimizer and schedule.** `--optimizer adamw|adam|adamw8bit|sgd` (default adamw; `--weight-decay`, `--momentum`
  for SGD), `--schedule cosine|linear|constant|wsd` (default cosine; `--warmup` fraction of the steps, `--min-lr`,
  `--decay` for the decay part of warm-up/stable/decay).
- **Adapter type.** `--adapter-type dora` trains DoRA adapters (a magnitude per output on top of LoRA), saved in the
  PEFT format with `use_dora`; `lora` is the default. (`--adapter DIR` names an existing adapter folder.)
- **Long conversations.** A conversation longer than `--max-length` keeps its whole answer: the user message before it
  is shortened (its start kept) rather than the answer cut.
- **Progress.** A bar with the steps done, still to do and in total, the epoch and loss, elapsed time and ETA; notable
  events (graph recording, FP8 checks, evaluation losses) print above it. Ctrl+C stops after the current step and
  saves the adapter so far.
- **Output.** The adapter in the PEFT format, and `idrak-tuning.json`: the base model as named on the command line,
  the system prompt and the maximum length. Any program can then load the folder (`TuningManifest.Read(folder)
  .LoadModel(folder, device)`), and every idrak-tune command accepts the adapter folder in place of the model.
- **Speed.** Sequence packing, CUDA-graph replay of the training step, fused LoRA products on tensor cores (a frozen
  bfloat16 or 4-bit base read as stored in both directions), the optimizer over every adapter matrix in three passes;
  `--fp8` for the frozen base's forward products (checked against bfloat16 first), `--int4` / `--int8` for QLoRA.
  `train --profile` times a few steps per kernel instead of training.
- **Memory.** Results no backward step reads are released during the forward pass. When a step runs out of device memory
  the tuner steps down, each step at a small cost, and says so: feed-forward activations recomputed in the backward pass
  (`--recompute`), then activations held as bfloat16 between the passes (`--bf16-activations`), then activation
  checkpointing (`--checkpointing`, a third more compute). The flags start a run at that step.

`idrak-tune --help` lists every option.
