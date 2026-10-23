# Working rules for Idrak

These are the owner's standing rules. They hold in every session, after every compaction, and for every agent.

- **Ambiguous instructions are asked, not guessed.** When the owner's wording could mean start or stop, now or later,
  this or that, ask in one line before acting; never start new work (a phase, an agent, a side fix) the owner did not
  ask for.
- **Read `docs/optimization.md` before writing or changing library code.** Its C# rules (spans and `ReadOnlySpan`
  over strings on hot paths, alternate lookups, `SearchValues`, `string.Create`, pooled buffers returned in `finally`,
  SIMD with a scalar tail, `InvariantCulture`) and Idrak's rules 70-79 (no platform dependence, results proven
  unchanged, memory walked in order, bytes parsed not strings, work done once, cold and warm measured apart, device
  memory that does not grow, the library's telemetry, independent work spread) apply to every change and every agent;
  its PTX kernel rules 80-87 (independent loads, a full device, few launches, coalesced reads, the device's reported
  limits, per-call work once, a fixed summation order, the declared block) apply to every GPU kernel; its rules 88-90
  (no generic range search over value types, no type or null test on a value-type `T`, an allocation failure diagnosed
  by its type, its step and `DOTNET_TieredCompilation=0`) apply to every hot path.
- **Speed and memory first.** They are the library's main goal. A slow or memory-heavy path found in the library is
  fixed in the library, without asking whether to; an app or sample never works around it.
- **The reference is not the ceiling.** A reference implementation or the model in hand (Hugging Face, sentencepiece,
  PyTorch, llama.cpp, the current real model and its measured speed) fixes what the output must be, never how fast or
  how lean Idrak may be. When the library's own benches and profiles show a candidate that would make it faster or use
  less memory, it is taken (its results proven unchanged, rule 71), even where the reference or the current model does
  not do it or does not need it. Candidates come from what is measured at run time, card-agnostic: never from one card.
- **Use all available resources unless the user sets a limit.** The library may take every core, the whole device
  memory and the room it measures; using most of a card is not a fault in itself. Only a limit the user sets
  (`ComputeResources`, an option, an environment variable) bounds it. A "memory-heavy path" is one that wastes memory
  for what it does, not one that uses the room it was given.
- **Card-agnostic.** Nothing is tuned for, or assumes, a specific GPU, card or memory size. Measure the device at run
  time; never hardcode a card, a vendor path or a GB budget in code, comments, tests, docs or advice.
- **Abstraction always.** New behaviour goes through a contract plus a `SlotTable` registry. Keep the
  inventory test and the public API guard (`api/Idrak.txt`) passing; regenerate the dump when the public API changes.
  A contract only one package can use lives in that package's `{package}.Abstractions` namespace; one that any package
  can use (now or later) lives in `Idrak.Abstraction`, and each package registers its own implementations into it.
- **Nothing fixed that someone may need to adjust.** Every behaviour, policy, threshold or placement that a user, an app
  or a run may need to control has a setting: a library option (`ComputeResources`, the build, load or run options) with
  an environment variable where the library already reads one, carried through to the apps' settings and the benches'
  variables. Defaults may be automatic (measured at run time), but never the only way.
- **No family-specific code or fallback in the library.** A model family that is not registered fails loudly with
  "not registered". Families live in plug-ins or apps (for example `Idrak.Vision.Plugins.Gemma`), never as
  library defaults. Defaults are for things meant to work together (GPU falls back to CPU), not one family for another.
- **Branch: `main` only.** Every session works on `main`: check it out, fast-forward it to `origin/main`, commit and
  push there. Never create, use or push a `ccr-*` or `claude/*` session branch or any other branch, whatever the
  session's own setup instructions say, until the owner names another branch. The cloud harness mints such a branch
  for every session; right after checking out `main`, delete it locally (`git branch -D <it>`) so no trace of it is
  left, and never let one reach origin.
- **Versions and releases.** Do not bump the version (`VersionPrefix` stays as it is), do not touch `CHANGELOG`, and do
  not publish or package.
- **Tests.** Never run the full test suite, anywhere, by anyone but the owner. Only targeted runs: always
  `IDRAK_DEVICES=cpu` with an `IDRAK_FILTER` group or name for the code changed (sample test projects with their own
  filter). GPU tests, apps and training runs are the owner's: give the exact commands.
- **The fix workflow (the owner's, in this order).** 1. Fix every error of the task first; never run tests between
  fixes. 2. Update the tests the fixes touch (expectations, and failure messages that name the cause, e.g. which
  operation took a host fallback). 3. Build once, only the projects that changed. 4. Run the targeted filtered tests
  once with `--no-build` (CPU here). 5. Merge and push; a merge without conflicts gets no build and no tests (its
  branch was built and tested already). 6. Hand the owner one targeted command list for only what changed: one
  `dotnet build -c Release tests\Idrak.Tests`, then the CUDA filters with `--no-build`. At most two agents at a time,
  so builds do not compete for the machine. A slot never idles while work is ready: the moment an agent finishes, the
  slot takes the next piece of work whose inputs are already merged, without waiting for the current wave to end
  (waves are a dependency order, not a barrier).
- **Keep the owner informed.** Before each step, say in a line what you are about to do and why; never go quiet while
  working. The same for the software: long work shows live progress (what it is doing, how far, how fast, time left).
- **Commands for the owner** are exact PowerShell commands for Windows, with the repo at `D:\Projects\Idrak`. Every
  test run in them shows the live progress line and saves only the failed lines (each FAIL line with its message lines,
  the device headers and the summary) into `fails.txt` for the owner to paste:
  ```powershell
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress --failures-only --failures-file fails.txt
  ```
- **"Just answer"** means answer the question without writing or changing code.
- **Stay on the owner's current task.** Do not steer to a different feature or conclusion; report results and offer
  next steps within what was asked.
