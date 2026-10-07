// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.


namespace Idrak.Nlp;

/// <summary>
/// What this assembly adds to the registries of Idrak.Abstraction, which cannot name it: the reader of a model's own Jinja
/// chat template ("jinja" in <see cref="ChatTemplates"/>), which loading a pretrained model asks. The engine's text and
/// chat model kinds need no registry: <c>GenerativeModels</c> adds them to an engine. <c>LibraryDefaults.Ensure</c> (in
/// Idrak.Abstraction) calls <see cref="RegisterFor"/> once per registry, before that registry is first used.
/// </summary>
internal static class LibraryRegistrations
{
    private static void RegisterFor(Type registry)
    {
        if (registry == typeof(ChatTemplates))
        {
            ChatTemplates.Register("jinja", (folder, tokenizer) => JinjaChatTemplate.Load(folder, tokenizer));
        }
    }
}
