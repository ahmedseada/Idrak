// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Generation;
using Idrak.Models;

namespace Idrak.Nlp;

/// <summary>What Idrak.Nlp does with a <see cref="PretrainedModel"/> (Idrak): generate text with it and chat with it.</summary>
public static class PretrainedModelExtensions
{
    extension(PretrainedModel model)
    {
        /// <summary>A text generator for the model (int8 KV cache with <paramref name="cacheFormat"/>).</summary>
        public TextGenerator CreateGenerator(KeyValueFormat cacheFormat = KeyValueFormat.Float32, int? contextLength = null)
        {
            model.Network.Eval();
            return new TextGenerator(model.Network, model.Tokenizer ?? throw new InvalidOperationException("The model has no tokenizer."),
                Math.Min(contextLength ?? model.MaxPositions, model.MaxPositions)) { CacheFormat = cacheFormat };
        }

        /// <summary>A text generator for the model whose KV cache is stored by <paramref name="cacheLayout"/> (see <see cref="KeyValueLayouts"/>).</summary>
        public TextGenerator CreateGenerator(KeyValueLayout cacheLayout, int? contextLength = null)
        {
            ArgumentNullException.ThrowIfNull(cacheLayout);
            model.Network.Eval();
            return new TextGenerator(model.Network, model.Tokenizer ?? throw new InvalidOperationException("The model has no tokenizer."),
                Math.Min(contextLength ?? model.MaxPositions, model.MaxPositions)) { CacheLayout = cacheLayout };
        }

        /// <summary>A chat model using the model's own chat template (and its tool-call format).</summary>
        public ChatGenerator CreateChat(KeyValueFormat cacheFormat = KeyValueFormat.Float32, int? contextLength = null) =>
            new(model.CreateGenerator(cacheFormat, contextLength), model.ChatTemplate ?? throw new InvalidOperationException("The model has no chat template."));

        /// <summary>A chat model using the model's own chat template, its KV cache stored by <paramref name="cacheLayout"/>.</summary>
        public ChatGenerator CreateChat(KeyValueLayout cacheLayout, int? contextLength = null) =>
            new(model.CreateGenerator(cacheLayout, contextLength), model.ChatTemplate ?? throw new InvalidOperationException("The model has no chat template."));

        /// <summary>
        /// The model's chat template as a <see cref="JinjaChatTemplate"/> (the kind Idrak.Nlp reads), for what only a Jinja
        /// template does (its source, rendering without the generation prompt, fine-tuning transcripts); null when the model
        /// has no chat template, or one another reader made.
        /// </summary>
        public JinjaChatTemplate? JinjaTemplate => model.ChatTemplate as JinjaChatTemplate;
    }
}
