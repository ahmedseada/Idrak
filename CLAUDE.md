# Working rules for Idrak

These are the owner's standing rules. They hold in every session, after every compaction, and for every agent.

- **Card-agnostic.** Nothing is tuned for, or assumes, a specific GPU, card or memory size. Measure the device at run
  time; never hardcode a card, a vendor path or a GB budget in code, comments, tests, docs or advice.
- **Abstraction always.** New behaviour goes through a contract plus a `SlotTable` registry (decision 10). Keep the
  inventory test and the public API guard (`api/Idrak.txt`) passing; regenerate the dump when the public API changes.
- **No family-specific code or fallback in the library.** A model family that is not registered fails loudly with
  "not registered". Families live in plug-ins or apps (for example `samples/Gemma3Vision/Idrak.Gemma3Vision`), never as
  library defaults. Defaults are for things meant to work together (GPU falls back to CPU), not one family for another.
- **Versions and releases.** Do not bump the version (`VersionPrefix` stays as it is), do not touch `CHANGELOG`, and do
  not publish or package.
- **Tests.** The owner runs the tests, the training and the apps. Do not run them; build to check the code compiles,
  then give the exact commands. (Agents working on library code may run targeted `IDRAK_FILTER` tests only when the
  owner asks for it; never the full suite.)
- **Keep the owner informed.** Before each step, say in a line what you are about to do and why; never go quiet while
  working. The same for the software: long work shows live progress (what it is doing, how far, how fast, time left).
- **Commands for the owner** are exact PowerShell commands for Windows, with the repo at `D:\Projects\Idrak`.
- **"Just answer"** means answer the question without writing or changing code.
- **Stay on the owner's current task.** Do not steer to a different feature or conclusion; report results and offer
  next steps within what was asked.
