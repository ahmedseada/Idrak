// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;

namespace Idrak.Vision.Plugins;

/// <summary>
/// The vision model families (vision-language towers and projectors, image models, line recognizers) of
/// Idrak.Vision.Plugins. Each family is a plug-in in a namespace of its own (<c>Idrak.Vision.Plugins.Family</c>), whose
/// static <c>FamilyPlugin</c> class registers it alone (<c>Register()</c>, <c>Unregister()</c>; see <see
/// cref="IIdrakPlugin"/>). <see cref="RegisterAll"/> registers every one of them through <see cref="IdrakPlugins"/>; no
/// model is loaded until one asks for its family, so registering costs only the names.
/// </summary>
public static class VisionPlugins
{
    /// <summary>The package's name, as the library's "not registered" messages name it: "Idrak.Vision.Plugins".</summary>
    public const string Package = "Idrak.Vision.Plugins";

    /// <summary>
    /// Every family plug-in of the package, in the order <see cref="RegisterAll"/> registers them: Qwen-VL
    /// (<see cref="IdrakVision.IdrakVisionPlugin"/>), the CRNN line recognizer (<see cref="Crnn.CrnnPlugin"/>), Gemma 3 vision
    /// (<see cref="Gemma.Gemma3VisionPlugin"/>).
    /// </summary>
    public static IReadOnlyList<IIdrakPlugin> All { get; } =
        [Qwen.QwenVisionPlugin.Plugin, Llava.LlavaPlugin.Plugin, IdrakVision.IdrakVisionPlugin.Plugin, Crnn.CrnnPlugin.Plugin,
         Gemma.Gemma3VisionPlugin.Plugin];

    /// <summary>Registers every family of the package (<see cref="IdrakPlugins.RegisterAll"/>); once: calling it again changes nothing.</summary>
    public static void RegisterAll() => IdrakPlugins.RegisterAll(All);

    /// <summary>Unregisters every family of the package, the last first, leaving the registries as <see cref="RegisterAll"/> found them.</summary>
    public static void UnregisterAll() => IdrakPlugins.UnregisterAll(All);

    /// <summary>The entry the idrak tool's <c>-P</c> (<c>--plugin</c>) calls when it loads this assembly: <see cref="RegisterAll"/>.</summary>
    public static void RegisterIdrakPlugin() => RegisterAll();
}
