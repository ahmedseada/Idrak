# Handover: how work on Idrak goes on

For a Claude Code session (any account) that picks this project up. Read this first, then `CLAUDE.md` (the owner's
standing rules; they win over anything here), then the plan you are asked to work on.

## Where things stand (2026-10-19)

- Branch `abstraction`, last commit pushed: the Plan 24 T4 merge (48d058d). Nothing is in progress and no agent runs.
- **Plan 24 (tokenizers)** — [24-tokenizers.md](24-tokenizers.md): L0, L1, T0, T1, T2, T3 (encodings, detection,
  GGUF) and T4 are built, merged and pushed; T5 is on hold for the owner. What is left, in order, is its section
  **"What is left (handover, 2026-10-19)"**: library-owned Unicode normalization (a real bug in the CLI), the T3
  callers (`TextEncoder`, `idrak tokenize --offsets --pair --special`), T4 leftovers, the owner's Windows/CUDA runs.
- **Scope: finish Plan 24 only.** No other plan is to be started (the owner dropped Plan 25, Idrak.Extensions.AI).
- **The owner said: no new work until they say so.** Do not start anything from "What is left" on your own; report
  status and wait for the owner to name the next item.

## How to work with the owner

- **Say what you are about to do, in one line, before each step**, and why. Never go quiet. Status questions ("plan
  status", "agent status") get a short table: phase, state, what runs now, what is next.
- **Do exactly what was asked.** "Just answer" means no code. A plan request means a plan, not code. Do not start
  the next phase, a new agent or a side fix unless the owner asked or `CLAUDE.md` makes it automatic.
- **When an instruction is ambiguous, ask before acting.** Example from this project: "do start new agents after the
  current agents finish" meant *don't*; two agents were started and had to be stopped. When a wording could mean
  "start" or "stop", "now" or "later", ask in one line.
- **Hand the owner commands, not chores**: after each merged piece, the exact PowerShell list for what changed only,
  repo at `D:\Projects\Idrak`: one `dotnet build -c Release tests\Idrak.Tests`, then the CPU filters, then the CUDA
  filters with `--no-build`. The owner runs GPU tests, apps, training and the full suite; you never do.
- **Report results faithfully**: test counts, what failed and why, what you could not do. Write findings and
  decisions into the plan's "as built" notes and the Runs table, not only into chat.

## The way of working (from CLAUDE.md, in practice)

1. **Read `docs/optimization.md`** before touching library code. Speed and memory first; card-agnostic; nothing
   fixed that someone may need to adjust (a setting, with an `IDRAK_*` variable where the library reads one, carried
   into the CLI's `src/Idrak.Cli/Shared/EnvironmentVariables.cs` table, which a test checks).
2. **Abstraction**: new behaviour = a contract + a `SlotTable` registry; library defaults registered from static
   constructors. No model-family code in the library.
3. **The fix workflow, strictly**: fix every error first (grep every affected type before building, so one build is
   enough) → update the tests → build once, only what changed (`dotnet build tests/Idrak.Tests` builds the library
   and CLI it references) → run the targeted filters once with `--no-build` → merge and push. If something fails:
   fix all failures, one rebuild, rerun only the failing filters. The owner noticed and objected to repeated builds.
4. **Tests**: never the full suite. Always `IDRAK_DEVICES=cpu IDRAK_FILTER="a|b|c" dotnet run --project
   tests/Idrak.Tests --no-build` (a test runs when its name contains any of the `|` parts, ignoring case). The usual
   tokenizer set: `tokenizers:|tokenizer training|cli pretrain|pretrained:|gguf:|plugins:|abstraction
   inventory|public API|cli health: every environment variable|cli arabic:`.
5. **Generated files**: when the public API changes, `IDRAK_UPDATE_API=1` with filter `public API` rewrites
   `api/*.txt`; when contracts change, `IDRAK_UPDATE_INVENTORY=1` with filter `abstraction inventory: plans` rewrites
   `plans/10-abstraction-inventory.md`. In a merge conflict on these, take either side and regenerate.
6. **Branch**: only `abstraction`. Delete the harness's `ccr-*` branch locally right after checkout; never push it.
   No version bump, no CHANGELOG, no packaging. Commit messages end with the attribution lines the session's
   instructions give; no model names in commits, code or docs.
7. **Agents**: at most two at a time, each in its own detached git worktree (`git worktree add --detach
   /home/user/Idrak-xx origin/abstraction`), committing locally only; the main session merges, builds once, runs the
   filters and pushes. Tell each agent the files the other one owns. A free slot takes the next ready piece only when
   the owner has asked for that work to go on.
8. **The reference is not the ceiling**: Hugging Face `tokenizers`, sentencepiece, PyTorch, llama.cpp fix what the
   output must be, never how fast Idrak may be.

## A fresh cloud container

- .NET SDK 10: `apt-get install -y dotnet-sdk-10.0` (packages.microsoft.com is reachable).
- The tokenizer references: `python3 -m venv /home/user/t0-venv && /home/user/t0-venv/bin/pip install -r
  tools/tokenizers/requirements.txt` (PyPI is reachable; huggingface.co is not, so real models are the owner's
  `make_fixtures.py --real` run). Run reference scripts with `-I`.
- Reference sources when behaviour must be matched exactly: `pip download --no-binary :all: --no-deps
  tokenizers==0.23.2` gives the Rust source (`tokenizers/src/...`); crates come from
  `https://crates.io/api/v1/crates/NAME/VERSION/download` (for example `unicode-normalization-alignments`).

## Pitfalls already met

- **tokenizers 0.23 truncation** encodes each sequence only until it has `max_length` tokens, then splits a pair by
  those lengths (not by `truncate_encodings` alone). Plan 24's T3 note has this and the other alignment rules
  (positional normalization, `Replace` standing for the match's last character, Precompiled's cursor including its
  start-of-text shift, Unigram fusing unknown runs before byte fallback).
- **.NET normalization on Linux allocates per call (ICU), and does nothing under `InvariantGlobalization`**: the
  open bug above.
- C# 14: `array.Reverse()` binds to the in-place span `Reverse`; use `Enumerable.Reverse(array)`.
- Writing special characters (U+2028, a BOM, combining marks) into source through tools can put the literal
  character in the file; write `\uXXXX` escapes. An `old_string` with a precomposed "é" will not match a file holding
  "e" + U+0301.
- A background wait loop (`until grep ... ; do sleep`) on a process that dies waits forever; check the producer's
  exit and stop such loops. Never use `tail -f` for a one-time wait.
- Tests that pass only because the test process has something the CLI lacks (ICU) prove nothing about the CLI.
