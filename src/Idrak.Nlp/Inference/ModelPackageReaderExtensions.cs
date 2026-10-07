// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Generation;
using Idrak.Layers;

namespace Idrak.Inference;

/// <summary>What Idrak.Nlp reads from a <see cref="ModelPackageReader"/> (Idrak): a language model as a text generator.</summary>
public static class ModelPackageReaderExtensions
{
    extension(ModelPackageReader package)
    {
        /// <summary>
        /// A <see cref="Generation.TextGenerator"/> for a language model stored with its architecture: the network named
        /// <paramref name="model"/> is rebuilt with its weights, the tokenizer is <paramref name="tokenizer"/>, and the
        /// context length is the architecture's token length (<c>Network.Tokens(length)</c>).
        /// </summary>
        public TextGenerator TextGenerator(string tokenizer, string model = ModelPackage.DefaultModelName, Device? device = null)
        {
            if (DecoderSpec.IsDescription(package.Architecture(model)))
            {
                var spec = DecoderSpec.FromJson(package.Architecture(model));
                var decoder = (Sequential)package.BuildModel(model, device);
                decoder.Eval();
                return new TextGenerator(decoder, package.Tokenizer(tokenizer), spec.MaxPositions);
            }

            var network = package.Network(model);
            if (network.InputKind != InputKind.Tokens)
            {
                throw new InvalidOperationException($"The network '{model}' does not read tokens (it starts with {network.InputKind}).");
            }

            var tokens = package.Tokenizer(tokenizer);
            var built = package.BuildNetwork(model, device);
            built.Eval();
            return new TextGenerator(built, tokens, network.InputShape[0]);
        }
    }
}
