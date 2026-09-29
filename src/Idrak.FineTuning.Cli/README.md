# idrak-tune

One command-line tool for every fine-tuning job: LoRA / QLoRA adapters for any pretrained model Idrak loads
(Hugging Face ids, folders, .gguf files, Ollama models), trained on any dataset `idrak-data` reads, then evaluated, chatted
with and exported. Nothing in it is specific to a model family or to an application: a project that needs a tuned model
runs idrak-tune with its data instead of writing its own tuner.

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
- **Long conversations.** A conversation longer than `--max-length` keeps its whole answer: the user message before it
  is shortened (its start kept) rather than the answer cut.
- **Progress.** A bar with the steps done, still to do and in total, the epoch and loss, elapsed time and ETA; notable
  events (CUDA graph recording, FP8 checks, evaluation losses) print above it. Ctrl+C stops after the current step and
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
