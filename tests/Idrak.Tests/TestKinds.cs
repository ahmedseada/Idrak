// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Text;

// The tests' own kinds, each a name and a function: what a test registers to replace or add one of a registry's kinds
// (an app writes a class implementing the kind's interface).
internal static partial class Tests
{
    internal static class TestKinds
    {
        public static ITokenSamplerKind Sampler(string name, Func<SamplerRequest, ITokenSampler> create) => new SamplerKind(name, create);

        public static IRopeScaling RopeScaling(string name, Func<RopeScalingInput, RopeScalingResult> create) => new RopeScalingKind(name, create);

        public static ISampleSourceKind SampleSource(string name, Func<string, IReadOnlyDictionary<string, string>, ISampleSource> open) =>
            new SampleSourceKind(name, request => open(request.Path, request.Options));

        public static IProgressReporterKind ProgressReporter(string name, Func<ProgressRequest, IProgressReporter> create) => new ProgressKind(name, create);

        public static ITokenizerComponentKind<ITokenizerNormalizer> Normalizer(string type, Func<JsonObject, ITokenizerNormalizer> create) =>
            new ComponentKind<ITokenizerNormalizer>(type, create);

        public static ITokenizerComponentKind<IPreTokenizer> PreTokenizer(string type, Func<JsonObject, IPreTokenizer> create) =>
            new ComponentKind<IPreTokenizer>(type, create);

        public static ITokenizerComponentKind<ITokenizerDecoder> Decoder(string type, Func<JsonObject, ITokenizerDecoder> create) =>
            new ComponentKind<ITokenizerDecoder>(type, create);

        public static ITokenizerComponentKind<ITokenizerPostProcessor> PostProcessor(string type, Func<JsonObject, ITokenizerPostProcessor> create) =>
            new ComponentKind<ITokenizerPostProcessor>(type, create);

        public static ITokenizerModelKind TokenizerModel(string type, Func<TokenizerDefinition, TokenizerOptions, IPretrainedTokenizer> create) =>
            new TokenizerModelKind(type, request => create(request.Definition, request.Options));

        public static IGgufTokenizer GgufTokenizer(string model, Func<GgufTokenizerRequest, TokenizerDefinition> create) => new GgufTokenizerKind(model, create);

        public static IChatTemplateReader ChatTemplateReader(string name, Func<ChatTemplateRequest, IChatTemplate?> read) => new ChatReaderKind(name, read);

        public static ICtcDecoder CtcDecoder(string name, Func<ReadOnlyMemory<float>, int, int, CtcDecodeOptions, IReadOnlyList<CtcHypothesis>> decode) =>
            new CtcKind(name, decode);

        /// <summary>A family of its own: its spec and tensor names.</summary>
        public static IPretrainedArchitecture Architecture(Func<JsonObject, List<string>, DecoderSpec> spec, Func<string, string?> tensorName) =>
            new FunctionArchitecture(spec, tensorName);

        /// <summary>A family of its own that builds its network.</summary>
        public static INetworkBuildingArchitecture BuildingArchitecture(Func<JsonObject, List<string>, DecoderSpec> spec, Func<string, string?> tensorName,
            Func<PretrainedBuildContext, IEnumerable<Module>> build) => new BuildingFunctionArchitecture(spec, tensorName, build);

        /// <summary>A family of its own that writes its config.</summary>
        public static IConfigWritingArchitecture ConfigArchitecture(Func<JsonObject, List<string>, DecoderSpec> spec, Func<string, string?> tensorName,
            Func<DecoderSpec, JsonObject> config) => new ConfigFunctionArchitecture(spec, tensorName, config);

        private class FunctionKind<TRequest, T>(string name, Func<TRequest, T> create) : IKind<TRequest, T>
        {
            public string Name => name;

            public string Summary => "a test's own";

            public IReadOnlySet<string> Keys => OptionKeys.None;

            public T Create(TRequest request) => create(request);
        }

        private sealed class SamplerKind(string name, Func<SamplerRequest, ITokenSampler> create)
            : FunctionKind<SamplerRequest, ITokenSampler>(name, create), ITokenSamplerKind;

        private sealed class RopeScalingKind(string name, Func<RopeScalingInput, RopeScalingResult> create)
            : FunctionKind<RopeScalingInput, RopeScalingResult>(name, create), IRopeScaling;

        private sealed class SampleSourceKind(string name, Func<SampleSourceRequest, ISampleSource> create)
            : FunctionKind<SampleSourceRequest, ISampleSource>(name, create), ISampleSourceKind;

        private sealed class ProgressKind(string name, Func<ProgressRequest, IProgressReporter> create)
            : FunctionKind<ProgressRequest, IProgressReporter>(name, create), IProgressReporterKind;

        private sealed class ComponentKind<T>(string name, Func<JsonObject, T> create) : FunctionKind<JsonObject, T>(name, create), ITokenizerComponentKind<T>
            where T : class;

        private sealed class TokenizerModelKind(string name, Func<TokenizerModelRequest, IPretrainedTokenizer> create)
            : FunctionKind<TokenizerModelRequest, IPretrainedTokenizer>(name, create), ITokenizerModelKind;

        private sealed class GgufTokenizerKind(string name, Func<GgufTokenizerRequest, TokenizerDefinition> create)
            : FunctionKind<GgufTokenizerRequest, TokenizerDefinition>(name, create), IGgufTokenizer;

        private sealed class ChatReaderKind(string name, Func<ChatTemplateRequest, IChatTemplate?> create)
            : FunctionKind<ChatTemplateRequest, IChatTemplate?>(name, create), IChatTemplateReader;

        private sealed class CtcKind(string name, Func<ReadOnlyMemory<float>, int, int, CtcDecodeOptions, IReadOnlyList<CtcHypothesis>> decode) : ICtcDecoder
        {
            public string Name => name;

            public string Summary => "a test's own";

            public IReadOnlyList<CtcHypothesis> Decode(ReadOnlyMemory<float> logProbs, int steps, int classes, CtcDecodeOptions options) =>
                decode(logProbs, steps, classes, options);
        }

        private class FunctionArchitecture(Func<JsonObject, List<string>, DecoderSpec> spec, Func<string, string?> tensorName) : PretrainedArchitecture
        {
            public override DecoderSpec Spec(JsonObject config, List<string> notes) => spec(config, notes);

            public override string? TensorName(string name) => tensorName(name);
        }

        private sealed class BuildingFunctionArchitecture(Func<JsonObject, List<string>, DecoderSpec> spec, Func<string, string?> tensorName,
            Func<PretrainedBuildContext, IEnumerable<Module>> build) : FunctionArchitecture(spec, tensorName), INetworkBuildingArchitecture
        {
            public IEnumerable<Module> Build(PretrainedBuildContext context) => build(context);
        }

        private sealed class ConfigFunctionArchitecture(Func<JsonObject, List<string>, DecoderSpec> spec, Func<string, string?> tensorName,
            Func<DecoderSpec, JsonObject> config) : FunctionArchitecture(spec, tensorName), IConfigWritingArchitecture
        {
            public JsonObject Config(DecoderSpec spec) => config(spec);
        }
    }
}
