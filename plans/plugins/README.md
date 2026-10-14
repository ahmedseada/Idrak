# Plug-in plans

Optional packages outside the library: each one brings a dependency the library must not have (decision 10, CLAUDE.md:
no dependencies beyond Idrak, rule 70: no platform dependence), registers into the library's contracts, and is loaded
by an app (a reference and one `Register()` call) or by the command line (`idrak -P PLUGIN.dll ...`). The library works
the same without them. They live under `samples/`, beside the Gemma 3 and Qwen-VL plug-ins.

| Plan | Plug-in | Dependency | Status |
|---|---|---|---|
| [1](1-harfbuzz.md) | `Idrak.HarfBuzz`: HarfBuzz as a second text shaper, for checking fonts against it in process and for apps that want it to shape | HarfBuzzSharp (native HarfBuzz per platform) | planned 2026-10-14, not started |
