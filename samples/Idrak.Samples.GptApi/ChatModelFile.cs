// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Generation;
using Idrak.Samples.Gpt;

namespace Idrak.Samples.GptApi;

/// <summary>Loads the model served by the Ollama-compatible endpoints: a chat-trained GPT if one exists, otherwise the text GPT.</summary>
public static class ChatModelFile
{
    public static TextGenerator Load(IConfiguration configuration)
    {
        string textModel = Path.GetFullPath(configuration["Gpt:ModelPath"] ?? Path.Combine(AppContext.BaseDirectory, "models", "transformer.weights"));
        string? chatModel = configuration["Gpt:ChatModelPath"] is { Length: > 0 } c ? Path.GetFullPath(c) : null;
        string path = chatModel is not null && File.Exists(chatModel) ? chatModel : textModel;
        if (!File.Exists(path) || !File.Exists(GptConfig.ConfigPath(path)))
        {
            throw new FileNotFoundException($"No model at {path} yet (train one with the Transformer sample, or wait for the background training).");
        }

        var device = (configuration["Gpt:Device"] ?? "auto") switch
        {
            "cpu" => Device.Cpu,
            "cuda" when Device.IsCudaAvailable => Device.Cuda(),
            _ => Device.Default,
        };
        var gpt = CharGpt.Load(path, device);
        return new TextGenerator(gpt.Model, new CharTokenizer(gpt.Config.Vocabulary), gpt.Config.Context);
    }
}
