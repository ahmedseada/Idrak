# Idrak.Onnx.Runtime

Runs ONNX models (exported by Idrak or any other tool) with ONNX Runtime as Idrak modules, so they work with predictors, the inference engine and the ASP.NET Core endpoints.

## Install

```bash
dotnet add package Idrak.Onnx.Runtime
```

It has ONNX Runtime as an `IOnnxRuntimeLoader` ("onnxruntime", Idrak.Abstraction's contract), which `idrak onnx check`
compares Idrak's import with; finding it by its name is not on the new design yet.

Part of [Idrak](https://www.nuget.org/packages/Idrak), a self-contained deep-learning library for .NET.

## Documentation

Guides, samples and the full API overview: https://github.com/ahmedseada/Idrak

License: Apache 2.0.
