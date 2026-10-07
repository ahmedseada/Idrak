# Idrak.AspNetCore

ASP.NET Core integration for Idrak: registers an InferenceEngine in dependency injection and maps prediction, generation, chat (the local chat API and the OpenAI-style /v1 API) and status endpoints.

The endpoints are a thin layer over the engine and the contracts of `Idrak.Abstraction`: `MapPredictor`
(`IPredictor<TIn, TOut>`), `MapGenerate` (`ITextModel`), `MapChatApi` and `MapCompletionsApi` (`IChatModel`, and
`IEmbedder` for `/v1/embeddings`), `MapIdrakStatus`. With server-side tool execution, `MapChatApi` runs the tools
through `ChatTools.WithTools`: those given to the options, or the chat model's own (`IToolChatModel.Tools`).

```csharp
app.MapChatApi("/api", "my-gpt", o => o.Tools(ToolExecution.Client));                 // the client runs the tools
app.MapChatApi("/agent", "my-gpt", o => o.Tools(ToolExecution.Server, maxRounds: 5, tools: registry));
```

## Install

```bash
dotnet add package Idrak.AspNetCore
```

Part of [Idrak](https://www.nuget.org/packages/Idrak), a self-contained deep-learning library for .NET.

## Documentation

Guides, samples and the full API overview: https://github.com/ahmedseada/Idrak

License: Apache 2.0.
