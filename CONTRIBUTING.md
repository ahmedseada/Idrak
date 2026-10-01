# Contributing to Idrak

Thank you for helping. Bug reports, benchmark runs on new hardware, and pull requests are all welcome.

## Before a pull request

- **Build:** `dotnet build -c Release` must show 0 warnings and 0 errors (warnings are errors).
- **Tests:** `dotnet run -c Release --project tests/Idrak.Tests` runs every test on every device present (CPU, and CUDA
  GPUs); all must pass. `IDRAK_FILTER=<text>` runs the tests whose names contain the text.
- **Speed:** changes to kernels or hot paths come with before and after numbers from the matching benchmark
  (`-- --bench-cpu`, `--bench-gemv`, `--bench-gemm`, `--bench-text`, `--bench-offload`).
- **No native dependencies** in the core packages (see `plans/README.md`), and no tuning to one specific card: choices
  that depend on the hardware read the device's own limits or are measured on it.
- **Style:** match the surrounding code. New C# files start with the license header (the `.editorconfig` offers it):

  ```csharp
  // Copyright (c) 2026 Ahmed Seada
  // Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.
  ```

## License and CLA

Idrak is licensed under the [Apache License 2.0](LICENSE). Contributions are accepted under the
[Contributor License Agreement](CLA.md): you keep the copyright in your work and give the project the rights it needs to
ship it, including under other license terms in the future. Accept it by checking the box in the pull request
template; a pull request is merged only after that.
