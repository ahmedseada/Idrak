# Idrak.Onnx.Runtime

Runs ONNX models (exported by Idrak or any other tool) with ONNX Runtime as Idrak modules, so they work with predictors, the inference engine and the ASP.NET Core endpoints.

## Install

```bash
dotnet add package Idrak.Onnx.Runtime
```

Referenced (or loaded by a host, as the idrak tool's `-P` does), it registers ONNX Runtime as "onnxruntime" in
`OnnxRuntimeLoaders` (Idrak.Abstraction), where `idrak onnx check` finds it to compare Idrak's import with.

Part of [Idrak](https://www.nuget.org/packages/Idrak), a self-contained deep-learning library for .NET.

## Documentation

Guides, samples and the full API overview: https://github.com/ahmedseada/Idrak

License: Apache 2.0.
