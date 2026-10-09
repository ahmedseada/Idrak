# Idrak.AspNetCore

Serve Idrak models over HTTP from an ASP.NET Core app: predictions, text generation, chat, and an OpenAI-compatible
`/v1` API (chat completions, embeddings) that OpenAI clients can call. It is a thin layer over the inference engine:
`AddIdrak()` registers the engine in dependency injection, and the `Map...` methods add the endpoints.

The endpoints are a thin layer over the engine and the contracts of `Idrak.Abstraction`: `MapPredictor`
(`IPredictor<TIn, TOut>`), `MapGenerate` (`ITextModel`), `MapChatApi` and `MapCompletionsApi` (`IChatModel`, and
`IEmbedder` for `/v1/embeddings`), `MapIdrakStatus`. With server-side tool execution, `MapChatApi` runs the tools
through `ChatTools.WithTools`: those given to the options, or the chat model's own (`IToolChatModel.Tools`).

```csharp
app.MapChatApi("/api", "my-gpt", o => o.Tools(ToolExecution.Client));                 // the client runs the tools
app.MapChatApi("/agent", "my-gpt", o => o.Tools(ToolExecution.Server, maxRounds: 5, tools: registry));
```

A chat model that reads images (a vision-language model added with `GenerativeModelBuilder.Images`) takes them on
`/v1/chat/completions` as `image_url` data URLs, on `/v1/chat/upload` as a `multipart/form-data` upload (`image`
files, `prompt`, `stream`, ...) and on `MapChatApi` as `images` or image parts; `"grayscale": true` reads them grey.
`ImageInputOptions` limits the images per request and allows http(s) image URLs (off by default):

```csharp
app.MapCompletionsApi("/v1", o => o.Images(i => { i.MaxImages = 4; i.AllowUrls = false; }));
```

## Install

```bash
dotnet add package Idrak.AspNetCore
```

Part of [Idrak](https://www.nuget.org/packages/Idrak), a self-contained deep-learning library for .NET.

## Documentation

Guides, samples and the full API overview: https://github.com/ahmedseada/Idrak

License: Apache 2.0.
