# Idrak.Samples.CodingAgent

A coding agent on Idrak's own engine (CPU or CUDA): a language model of any family Idrak loads, in its own chat
template, reads, searches, edits and writes files and runs allowed commands (`dotnet`, `npm`, ...) in a workspace. The
loop is Idrak.Text's `CodingAgent`; tasks, suites, workspaces and verification are the coding plug-in of
`Idrak.Reinforcement.Plugins` (`CodingTask`, `CodingSuite`, `CodingVerifier`); training is the library's
reinforcement-learning trainer (`Idrak.Reinforcement`). The sample has no task layer or trainer of its own.

| Command | What it does |
|---|---|
| `agent <model> <task...> --workspace <dir>` | Works on a typed task in `<dir>`, streaming its reasoning and tool calls (`--out F` appends the transcript) |
| `agent-run <model> <suite> --out <runs.jsonl>` | Evaluates a model on a task suite: each task in a fresh copy, verified by its own commands; one JSON line per run (`--attempts N`, `--filter S`, `--work DIR`, `--rounds N`, `--temperature T`, `--adapter DIR`) |
| `agent-check <suite>` | Checks every task without a model: verification fails on the starting files and passes with the task's `solution/` (`--filter S`, `--work DIR`) |
| `agent-train <model> <suite> --out <dir>` | Trains the model's LoRA adapters on the suite by reinforcement learning: `ReinforcementTrainer` in the `coding` environment, the hidden tests the reward; checkpoints in `<dir>/checkpoints`, the adapter in `<dir>` |

`<model>` is a model folder, a Hugging Face id (`Qwen/Qwen3-0.6B`), a `.gguf` file or `store:name`. Model options:
`--cuda`, `--cpu`, `--vulkan`, `--device NAME`, `--int8`, `--int4`, `--bf16`, `--kv8`, `--kv16`, `--context N`,
`--adapter DIR`, `--no-think`, `--offload`, `--gpu-memory GiB`, `--matmul fp32|bf16|fp8`.

## A task suite

A suite is a folder of task folders, each with:

- `task.json`: `{"prompt", "language", "base", "setup": [...], "verify": [...], "tags": [...]}`. `setup` runs once
  before the agent (a restore, an install), `verify` after it: every command must exit with 0 for the task to pass.
- `workspace/`: the starting files (laid over `base`, a shared project folder relative to the task).
- `verify/`: hidden tests, copied in (over any copy the agent made) after the agent finishes. Keep them in a
  sub-folder (`tests/`): files the agent changes there are reported as test edits, and refused when the environment's
  `refuse-test-edits` is on.
- `solution/`: a reference solution laid over the workspace, for `agent-check`.

`suite/` holds a small C# demo suite: `csharp/sum-values` (a sum that skips the last value), `csharp/reverse-words`
(a method to implement) and `csharp/palindrome` (a check that should ignore case and punctuation). Each is a console
project whose hidden `tests/Checks.cs` is the program's checks; `setup` is `dotnet restore`, `verify` builds and runs
it, so the suite runs on the `coding` environment with no language profile, and on a C#-specific environment too.
Check it first (dotnet installed; 17 s for the three on a 4-core machine):

```powershell
dotnet run -c Release --project samples\Idrak.Samples.CodingAgent -- agent-check samples\Idrak.Samples.CodingAgent\suite
```

## Training on the suite

`agent-train` runs the library's trainer with Idrak.Text's language-model policy over a fine-tuning session (LoRA
adapters); each iteration plays episodes (a task drawn per group of episodes, the agent's own sampled replies and tool
calls), rewards each by the hidden tests (`tests-passed`), and updates the adapters with the algorithm's loss:

```powershell
dotnet run -c Release --project samples\Idrak.Samples.CodingAgent -- agent-train Qwen/Qwen3-0.6B samples\Idrak.Samples.CodingAgent\suite --out runs\agent-rl --algorithm grpo --group 8 --iterations 20 --cuda
dotnet run -c Release --project samples\Idrak.Samples.CodingAgent -- agent-run Qwen/Qwen3-0.6B samples\Idrak.Samples.CodingAgent\suite --adapter runs\agent-rl --out runs\after.jsonl --cuda
```

Options: `--algorithm grpo|ppo|reinforce`, `--iterations N`, `--group N`, `--episodes N`, `--lr R`, `--rank N`,
`--max-new N` (the longest reply, 1024 by default), `--rounds N`, `--filter S`, `--work DIR`, `--temperature T`,
`--adapter DIR` (go on training an adapter), `--resume` (from the latest checkpoint). Every other run setting comes from
its `IDRAK_RL_*` variable (`idrak help env` lists them: the KL weight, the clip, the estimator, the episodes stepped at
once, the save interval, `IDRAK_RL_LOG` for a JSON line per iteration, ...), and the commands run in the sandbox
`IDRAK_SANDBOX` names (`process` by default; `container` with `IDRAK_SANDBOX_IMAGE` set to an image with the .NET SDK).

The same run through the idrak tool, with tune's adapter and device options and a teacher term if wanted:

```powershell
dotnet run -c Release --project src\Idrak.Cli -- reinforce Qwen/Qwen3-0.6B --env coding --suite samples\Idrak.Samples.CodingAgent\suite --algorithm grpo --group 8 --iterations 20 --out runs\rl-demo
```

`data/agent-demo-*.jsonl` are tool-calling transcripts in the agent's layout, to fine-tune a model for the agent with
`idrak tune` first.
