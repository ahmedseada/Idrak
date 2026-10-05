# Working on Idrak

Before writing or changing code in the library, read [docs/optimization.md](docs/optimization.md): the performance
rules (C# on .NET 10, each checked on this repository's SDK) and Idrak's own rules learned from measured work, among
them that nothing platform-specific goes into the library, that an optimization must prove its results unchanged, and
that every speed or memory change comes with before and after numbers.

Then follow [CONTRIBUTING.md](CONTRIBUTING.md): a Release build with 0 warnings, every test passing on every device
present (`dotnet run -c Release --project tests/Idrak.Tests`), and the matching benchmark for kernels and hot paths.
