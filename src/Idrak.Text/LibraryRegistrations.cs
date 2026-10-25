// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.


using Idrak.Abstraction.Reinforcement;
using Idrak.Text.Generation;
using Idrak.Text.Reinforcement;

namespace Idrak.Text;

/// <summary>
/// What this assembly adds to the registries of Idrak.Abstraction, which cannot name it: the reader of a model's own Jinja
/// chat template ("jinja" in <see cref="ChatTemplates"/>), which loading a pretrained model asks, and the image reader's
/// advisor ("image reader" in <see cref="SettingAdvisors"/>), and the feature caches ("memory" and "disk" in
/// <see cref="FeatureCaches"/>), the chat-prompt environment ("prompts" in <see cref="EnvironmentKinds"/>) and the text
/// rewards ("metric", "regex", "json", "length", "judge", "sum" in <see cref="RewardFunctions"/>, Plan 27 R3) and the plain-process
/// command sandbox ("process" in <see cref="CommandSandboxes"/>, R4), which the coding tools run commands in. The engine's text and
/// chat model kinds need no registry: <c>GenerativeModels</c> adds them to an engine. <c>LibraryDefaults.Ensure</c> (in
/// Idrak.Abstraction) calls <see cref="RegisterFor"/> once per registry, before that registry is first used.
/// </summary>
internal static class LibraryRegistrations
{
    private static void RegisterFor(Type registry)
    {
        if (registry == typeof(ChatTemplates))
        {
            ChatTemplates.Register(new JinjaTemplateReader());
        }
        else if (registry == typeof(SettingAdvisors))
        {
            SettingAdvisors.Register(new Generation.ImageReaderAdvisor());
        }
        else if (registry == typeof(FeatureCaches))
        {
            FeatureCaches.Register(new FeatureCacheKind(FeatureCaches.Memory, "in this process's memory, least recently used entries evicted to stay within a byte budget",
                options => new MemoryFeatureCache(options)));
            FeatureCaches.Register(new FeatureCacheKind(FeatureCaches.Disk, "one safetensors file per entry in a folder, which serves later runs and other processes",
                options => new DiskFeatureCache(options)));
        }
        else if (registry == typeof(EnvironmentKinds))
        {
            EnvironmentKinds.Register(new PromptsEnvironmentKind());
        }
        else if (registry == typeof(RewardFunctions))
        {
            foreach (var reward in TextRewards.Library())
            {
                RewardFunctions.Register(reward);
            }
        }
        else if (registry == typeof(CommandSandboxes))
        {
            CommandSandboxes.Register(new ProcessCommandSandbox());
        }
    }
}

// One of this assembly's feature caches, registered in FeatureCaches under its name.
internal sealed class FeatureCacheKind(string name, string summary, Func<FeatureCacheOptions, IFeatureCache> create) : IFeatureCacheKind
{
    public string Name => name;

    public string Summary => summary;

    public IReadOnlySet<string> Keys => OptionKeys.None;

    public IFeatureCache Create(FeatureCacheOptions request) => create(request);
}
