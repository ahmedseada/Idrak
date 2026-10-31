// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;

[assembly: IdrakPluginPackage(typeof(Idrak.Text.Plugins.TextPluginsPackage))]

namespace Idrak.Text.Plugins;

/// <summary>
/// The text model families (decoders, encoders, encoder-decoders, their GGUF mappings, tool-call and fill-in-the-middle formats and RoPE
/// scalings) and the coding languages (the coding agent's programs, prompt lines and folders) of Idrak.Text.Plugins. Each family is a plug-in in a namespace of its own
/// (<c>Idrak.Text.Plugins.Family</c>), whose static <c>FamilyPlugin</c> class registers it alone (<c>Register()</c>,
/// <c>Unregister()</c>; see <see cref="IIdrakPlugin"/>). <see cref="RegisterAll"/> registers every one of them through
/// <see cref="IdrakPlugins"/>; no model is loaded until one asks for its family, so registering costs only the names.
/// </summary>
public static class TextPlugins
{
    /// <summary>
    /// The package, as <c>-P</c> finds it too (<c>[assembly: IdrakPluginPackage]</c>): its name ("Idrak.Text.Plugins", as the
    /// library's "not registered" messages and the registries' origins name it) and <see cref="All"/>.
    /// </summary>
    public static IIdrakPluginPackage Package { get; } = new TextPluginsPackage();

    /// <summary>
    /// Every family plug-in of the package, in the order <see cref="RegisterAll"/> registers them: BERT
    /// (<see cref="Bert.BertPlugin"/>), T5 (<see cref="T5.T5Plugin"/>), Gemma (<see cref="Gemma.GemmaPlugin"/>), Llama
    /// (<see cref="Llama.LlamaPlugin"/>), Qwen (<see cref="Qwen.QwenPlugin"/>), Mistral (<see cref="Mistral.MistralPlugin"/>),
    /// harmony (<see cref="Harmony.HarmonyPlugin"/>), DeepSeek (<see cref="DeepSeek.DeepSeekPlugin"/>), then the coding
    /// languages: C# (<see cref="CSharp.CSharpLanguagePlugin"/>), TypeScript and Angular
    /// (<see cref="TypeScript.TypeScriptLanguagePlugin"/>), Python (<see cref="Python.PythonLanguagePlugin"/>), whose order is
    /// the order the coding agent lists them in. The tool-call formats are asked the last registered first, so detection
    /// asks "deepseek", "harmony", "mistral", "qwen3-coder", then the library's.
    /// </summary>
    public static IReadOnlyList<IIdrakPlugin> All { get; } =
    [
        Bert.BertPlugin.Plugin, T5.T5Plugin.Plugin, Gemma.GemmaPlugin.Plugin, Llama.LlamaPlugin.Plugin, Qwen.QwenPlugin.Plugin, Mistral.MistralPlugin.Plugin,
        Harmony.HarmonyPlugin.Plugin, DeepSeek.DeepSeekPlugin.Plugin, CSharp.CSharpLanguagePlugin.Plugin, TypeScript.TypeScriptLanguagePlugin.Plugin,
        Python.PythonLanguagePlugin.Plugin,
    ];

    /// <summary>Registers every family of the package (<see cref="IdrakPlugins.RegisterAll(IIdrakPluginPackage)"/>); once: calling it again changes nothing.</summary>
    public static void RegisterAll() => IdrakPlugins.RegisterAll(Package);

    /// <summary>Unregisters every family of the package, the last first, leaving the registries as <see cref="RegisterAll"/> found them.</summary>
    public static void UnregisterAll() => IdrakPlugins.UnregisterAll(Package);
}

/// <summary>
/// The package <c>-P</c> and <see cref="IdrakPlugins.Load"/> find through this assembly's
/// <see cref="IdrakPluginPackageAttribute"/>: <see cref="TextPlugins"/>' name and plug-ins.
/// </summary>
internal sealed class TextPluginsPackage : IIdrakPluginPackage
{
    /// <inheritdoc/>
    public string Name => "Idrak.Text.Plugins";

    /// <inheritdoc/>
    public IReadOnlyList<IIdrakPlugin> All => TextPlugins.All;
}
