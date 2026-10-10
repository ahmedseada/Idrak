# Working rules for Idrak

These are the owner's standing rules. They hold in every session, after every compaction, and for every agent.

- **Read `docs/optimization.md` before writing or changing library code.** Its C# rules (spans and `ReadOnlySpan`
  over strings on hot paths, alternate lookups, `SearchValues`, `string.Create`, pooled buffers returned in `finally`,
  SIMD with a scalar tail, `InvariantCulture`) and Idrak's rules 70-79 (no platform dependence, results proven
  unchanged, memory walked in order, bytes parsed not strings, work done once, cold and warm measured apart, device
  memory that does not grow, the library's telemetry, independent work spread) apply to every change and every agent.
- **Speed and memory first.** They are the library's main goal. A slow or memory-heavy path found in the library is
  fixed in the library, without asking whether to; an app or sample never works around it.
- **Card-agnostic.** Nothing is tuned for, or assumes, a specific GPU, card or memory size. Measure the device at run
  time; never hardcode a card, a vendor path or a GB budget in code, comments, tests, docs or advice.
- **Abstraction always.** New behaviour goes through a contract plus a `SlotTable` registry (decision 10). Keep the
  inventory test and the public API guard (`api/Idrak.txt`) passing; regenerate the dump when the public API changes.
- **No family-specific code or fallback in the library.** A model family that is not registered fails loudly with
  "not registered". Families live in plug-ins or apps (for example `samples/Gemma3Vision/Idrak.Gemma3Vision`), never as
  library defaults. Defaults are for things meant to work together (GPU falls back to CPU), not one family for another.
- **Versions and releases.** Do not bump the version (`VersionPrefix` stays as it is), do not touch `CHANGELOG`, and do
  not publish or package.
- **Tests.** Never run the full test suite, anywhere, by anyone but the owner. Only targeted runs: always
  `IDRAK_DEVICES=cpu` with an `IDRAK_FILTER` group or name for the code changed (sample test projects with their own
  filter). GPU tests, apps and training runs are the owner's: give the exact commands.
- **Builds.** Build only after a code change, and only the projects that changed; then run every filtered CPU test
  with `--no-build` on that one build. A merge without conflicts gets no build and no tests (its branch was built and
  tested already). At most two agents at a time, so builds do not compete for the machine.
- **Keep the owner informed.** Before each step, say in a line what you are about to do and why; never go quiet while
  working. The same for the software: long work shows live progress (what it is doing, how far, how fast, time left).
- **Commands for the owner** are exact PowerShell commands for Windows, with the repo at `D:\Projects\Idrak`.
- **"Just answer"** means answer the question without writing or changing code.
- **Stay on the owner's current task.** Do not steer to a different feature or conclusion; report results and offer
  next steps within what was asked.
