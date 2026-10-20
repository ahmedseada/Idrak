# Handover: how work on Idrak goes on

For a Claude Code session (any account) that picks this project up. Read this first, then `CLAUDE.md` (the owner's
standing rules; they win over anything here), then the plan you are asked to work on.

## Where things stand (2026-10-19, evening)

- Branch `abstraction`, last commit pushed: the merge of the second speed round (7b8c85f), then this handover's
  update. Nothing is in progress, no agent runs, no worktree is left.
- **Plan 24 (tokenizers)** — [24-tokenizers.md](24-tokenizers.md): every phase but T5 is built, merged and pushed, and
  so are the old handover's three leftovers (library-owned Unicode normalization, the T3 callers, the T4 leftovers)
  and two speed rounds; see its "Finished after the handover". The merged tree passed 147 of 147 targeted CPU tests.
  What is left is its section **"What is left (2026-10-19, after 7b8c85f)"**: the owner's runs (the exact list is in
  "The owner's runs"), T5 measured and not taken (same ids, about half the speed), and measured candidates nobody
  may start without the owner.
- **Plan 26 (model families as plug-in packages)** — [26-plugin-packages.md](26-plugin-packages.md): written 2026-10-19
  from the owner's decisions (one plug-in package per parent, one namespace per family, every family out of the library
  including the text language models, `Idrak.Nlp` renamed `Idrak.Text`, BERT and T5 first in the list). Widened
  2026-10-20: every reusable piece outside the library is a candidate, contract first, no copies in samples or plug-ins
  (only the override proofs keep their own); phases P0-P10 in two lanes. Nothing is built; the plug-in tests live in
  `tests/Idrak.Tests` (decided).
- **Plan 26 finished (2026-10-20):** every phase P0-P10 built, merged and pushed; the last merge is P7's. The library
  holds no model family: every family is in `Idrak.Text.Plugins`, `Idrak.Vision.Plugins` or `Idrak.Audio.Plugins`, and
  the CLI registers all three at startup. The owner's runs are in each phase's note (P7 asks for the full suite). **Next:
  the plan's "Leftovers for the owner"**, discussed with the owner one by one; none starts before the owner says so.
- **Scope:** Plan 24's owner runs, and Plan 26 only when the owner names a phase (the owner dropped Plan 25,
  Idrak.Extensions.AI).
- **No new work until the owner says so.** Report status and wait for the owner to name the next item.

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
- **Do not chase what a bench finds** (the owner, 2026-10-19). CLAUDE.md says a slow or memory-heavy path found in
  the library is fixed without asking; read as "anything a new measurement shows", it never ends: each fix round
  measures again and finds the next spot. Two speed rounds (agents E and F) were started that way after the asked
  work, and the owner objected ("you will chase an endless job"). Finish what was asked; list what measurements show
  in the plan's "What is left" with numbers; ask in one line before starting any agent or fix the owner did not
  name, and never announce a count of remaining work and then exceed it.
- **After every merge with `origin/abstraction`, build once, run the targeted filters, push only when green, then log the
  plan's status** (its phase table and a short status to the owner) after every successful merge and push (the owner,
  2026-10-20).
- **After merging an agent's work, build and run before pushing** (the owner, 2026-10-19): merge into the local
  `abstraction`, build once, run the targeted CPU filters once, push only when green. Two pieces each tested alone
  can fail together (a merge of C and E went out untested before this was asked; the joint run later passed).
- **Status means a table, now**: phase, state, what runs, what is next. The owner asks "plan status" often and wants
  it short and immediate, not an explanation.

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

- .NET SDK 10: `apt-get update && apt-get install -y dotnet-sdk-10.0`, from Ubuntu's own archive (noble-updates; SDK
  10.0.112 at `/usr/lib/dotnet`), not from packages.microsoft.com.
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
- **.NET normalization on Linux allocates per call (ICU), and does nothing under `InvariantGlobalization`**: fixed
  by `Idrak.Abstraction.Text.UnicodeNormalization` (Plan 24, "Finished after the handover", 1); never call
  `string.Normalize` or its span forms in `src/` (a test scans for them).
- C# 14: `array.Reverse()` binds to the in-place span `Reverse`; use `Enumerable.Reverse(array)`.
- Writing special characters (U+2028, a BOM, combining marks) into source through tools can put the literal
  character in the file; write `\uXXXX` escapes. An `old_string` with a precomposed "é" will not match a file holding
  "e" + U+0301.
- A background wait loop (`until grep ... ; do sleep`) on a process that dies waits forever; check the producer's
  exit and stop such loops. Never use `tail -f` for a one-time wait.
- Tests that pass only because the test process has something the CLI lacks (ICU) prove nothing about the CLI. The
  library no longer calls the platform's normalization (a test scans `src/`); `tokenizers: unicode normalization:`
  runs the real idrak tool in its own process.
- **Builds are fast when the compiler server stays warm** (corrected 2026-10-20; an earlier note here said 3 minutes,
  measured right after `dotnet build-server shutdown`, which makes every build cold). The first build of a session (or
  of a new worktree) restores and compiles everything and is slow; after it, a rebuild of `tests/Idrak.Tests` after
  library changes takes about a minute on 4 cores. The pattern:
  ```
  dotnet build tests/Idrak.Tests -v q --no-restore     # after the first build of the session
  IDRAK_DEVICES=cpu IDRAK_FILTER="..." dotnet run --project tests/Idrak.Tests --no-build
  ```
  Never `dotnet build-server shutdown`, `--disable-build-servers`, `dotnet clean` or deleting `obj`; never `-c Release`
  here (Release is the owner's Windows runs); never `dotnet run` without `--no-build`. Batch every fix before a build,
  read the code a failing test touches before changing its expectation, and do not build in parallel with agents when
  it can be avoided.
- **A stuck compiler server** (`VBCSCompiler` at 140% CPU with no build running) skewed timings for an hour; check
  `ps` before a speed measurement; only a server burning CPU with no build running is shut down, and only before
  measuring, never as a habit before builds (it makes the next build cold). Measure speed in a scratch Release
  program with `DOTNET_TieredCompilation=0` and medians, not in the shared test run (±50% there).
- **One `Regex` shared by threads** builds a matcher per match on every other thread (allocations and lost
  scaling); tokenizer patterns are `TokenizerRegex` with a per-thread copy. Use it for any new pattern.
- **unicode.org is blocked here**; the UCD files come from raw.githubusercontent.com's
  `unicode-org/unicodetools/main/unicodetools/data/ucd/<version>/`. PyPI's `unicodedata2` exists, but the earlier
  Imaging note found the package route unreliable; the generators read the UCD files.
- `TokenizerEncodeOptions.Offsets = false` (ids only) is the fast batch path; the detailed encode builds offsets and
  token strings, slower and several times the memory. A caller that needs only ids uses it.
