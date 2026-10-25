// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

[assembly: LibraryRegistrations(typeof(Idrak.Audio.LibraryRegistrations))]

namespace Idrak.Audio;

/// <summary>
/// What this assembly adds to the registries of Idrak.Abstraction and Idrak.Text, which cannot name it (plan 21: the
/// "audio" chat part of S3; the "audio" model part, the sources and data formats of S6). The registries of this assembly (the codecs, the
/// encoders) hold their own built-ins, as Idrak.Vision's do. <c>LibraryDefaults.Ensure</c> (in Idrak.Abstraction)
/// calls <see cref="RegisterFor"/> once per registry, before that registry is first used.
/// </summary>
internal sealed class LibraryRegistrations : ILibraryRegistrations
{
    private static readonly Dictionary<Type, Action> ByRegistry = new()
    {
        [typeof(ChatParts)] = () => ChatParts.Register(new AudioPartKind()),       // the "audio" message part (plan 21, S3)
        [typeof(ModelParts)] = () => ModelParts.Register(new AudioModelPart()),    // the audio part stored beside a decoder (plan 21, S6)
    };

    public void RegisterFor(Type registry)
    {
        if (ByRegistry.TryGetValue(registry, out var register))
        {
            register();
        }
    }
}
