// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Abstraction.Plugins;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Text;
using Idrak.Text.Plugins;
using Idrak.Text.Plugins.DeepSeek;
using Idrak.Text.Plugins.Harmony;
using Idrak.Text.Plugins.Llama;
using Idrak.Text.Plugins.Mistral;
using Idrak.Text.Plugins.Qwen;
using Formats = Idrak.Abstraction.Generation.ToolCallFormats;

// Plan 26, P7: Llama, Mistral and Mixtral, Qwen2 and Qwen3 with their mixtures of experts, and the DeepSeek and harmony
// tool-call formats are Idrak.Text.Plugins' (architectures, GGUF mappings, GGUF pre-tokenizers, tool-call formats, RoPE
// llama3); the library keeps none of them. The families' outputs are checked by the groups that always checked them
// (pretrained:, gguf:, decoder:, rope scaling:, chat templates:, mixture of experts:, ...), which the runner runs with the
// package registered, as an app does.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] TextFamilyGroup =
    [
        ("plugins: llama: the library's family tables are empty (no architecture, GGUF mapping or RoPE scaling of its own, GPT-2's pre-tokenizer rule only); without the family a Llama folder and a llama GGUF file are refused as not registered, naming Idrak.Text.Plugins; Register (twice) adds LlamaForCausalLM, llama, six pre-tokenizer names, the llama3 scaling, Llama 3's GGUF fallback chat template and the pre-tokenizer fallback when none is set, once, from the plug-in, and Unregister takes them back; the idrak tool's own process chats with a Llama folder and runs a qwen3 GGUF file with nothing added (no -P)", LlamaPluginRegistration),
        ("plugins: qwen: without the family a Qwen3 folder and a qwen3 GGUF file are refused naming Idrak.Text.Plugins; Register adds Qwen2, Qwen3 and their mixtures of experts, the four GGUF mappings, four pre-tokenizer names and the qwen3-coder tool-call format, once, and the folder loads; Unregister takes them back", QwenPluginRegistration),
        ("plugins: mistral: Register adds MistralForCausalLM, MixtralForCausalLM, the tekken pre-tokenizer and the mistral tool-call format, with Llama's GGUF mapping when it is missing (taken back only then); unregistered, the architecture is refused naming Idrak.Text.Plugins", MistralPluginRegistration),
        ("plugins: deepseek: Register adds the deepseek tool-call format and the deepseek-r1-qwen pre-tokenizer (Qwen2's rule), once; unregistered, a DeepSeek template detects the library's json format and the format is refused naming Idrak.Text.Plugins", DeepSeekPluginRegistration),
        ("plugins: harmony: Register adds the harmony tool-call format, once, and a harmony template detects it; unregistered, the template detects the library's json format and the format is refused naming Idrak.Text.Plugins", HarmonyPluginRegistration),
    ];

    // P7's plug-ins, in TextPlugins.All's order (re-registered in it, so the tool-call formats keep the order they are asked in).
    private static readonly IIdrakPlugin[] TextFamilyPlugins = [LlamaPlugin.Plugin, QwenPlugin.Plugin, MistralPlugin.Plugin, HarmonyPlugin.Plugin, DeepSeekPlugin.Plugin];

    // Runs `body` with P7's families taken away, then registers again, in TextPlugins.All's order, those that were.
    private static void WithoutTextFamilies(Action body)
    {
        var was = TextFamilyPlugins.Where(p => p.IsRegistered).ToList();
        try
        {
            foreach (var plugin in Enumerable.Reverse(TextFamilyPlugins))
            {
                IdrakPlugins.Unregister(plugin);
            }

            Check(TextFamilyPlugins.All(p => !p.IsRegistered && IdrakPlugins.Find(p.Name) is null), "P7's families taken away");
            body();
        }
        finally
        {
            foreach (var plugin in Enumerable.Reverse(TextFamilyPlugins))
            {
                IdrakPlugins.Unregister(plugin);
            }

            foreach (var plugin in TextFamilyPlugins.Where(was.Contains))
            {
                IdrakPlugins.Register(plugin);
            }
        }
    }

    // The plug-in's place: named so, in its family's namespace of Idrak.Text.Plugins, listed in TextPlugins.All.
    private static void CheckTextPlugin(IIdrakPlugin plugin, string name, string ns) =>
        Check(plugin.Name == name && plugin.GetType().Namespace == ns && TextPlugins.All.Contains(plugin) && plugin.GetType().Assembly.GetName().Name == TextPlugins.Package,
            $"{name}: the plug-in \"{name}\" of TextPlugins.All, in {ns} ({plugin.Name}, {plugin.GetType().Namespace})");

    private static void LlamaPluginRegistration(Device device)
    {
        _ = device;
        CheckTextPlugin(LlamaPlugin.Plugin, "llama", "Idrak.Text.Plugins.Llama");
        Check(TextPlugins.All.SkipWhile(p => p != LlamaPlugin.Plugin).Take(5).SequenceEqual(TextFamilyPlugins),
            $"TextPlugins.All: Llama, Qwen, Mistral, harmony, DeepSeek in that order ({string.Join(", ", TextPlugins.All.Select(p => p.Name))})");

        // The library's tables hold no family: every architecture and GGUF mapping is a plug-in's or an app's.
        string library = Overrides.Library;
        Check(PretrainedArchitectures.Names.All(n => PretrainedArchitectures.Origin(n) != library && PretrainedArchitectures.Default(n) is null)
              && GgufArchitectures.Names.All(n => GgufArchitectures.Origin(n) != library && GgufArchitectures.Default(n) is null),
            $"no architecture is the library's: {string.Join(", ", PretrainedArchitectures.Names.Select(n => $"{n}={PretrainedArchitectures.Origin(n)}"))}");
        Check(GgufPreTokenizers.Names.Where(n => GgufPreTokenizers.Origin(n) == library).Order(StringComparer.Ordinal).SequenceEqual(["default", "gpt-2", "gpt2"])
              && RopeScalings.Default(LlamaRopeScaling.Name) is null && RopeScalings.Names.Where(n => RopeScalings.Origin(n) == library).Order(StringComparer.Ordinal).SequenceEqual(["dynamic", "linear", "yarn"]),
            "the library's pre-tokenizers are GPT-2's rule only, its RoPE scalings linear, yarn and dynamic");

        string folder = TempFolder();
        try
        {
            WithoutTextFamilies(() =>
            {
                Check(!PretrainedArchitectures.Names.Contains(LlamaPlugin.LlamaForCausalLM) && GgufArchitectures.Find(LlamaPlugin.GgufName) is null
                      && !RopeScalings.Contains(LlamaRopeScaling.Name) && !LlamaPlugin.PreTokenizers.Any(n => GgufPreTokenizers.TryGet(n, out _)) && GgufPreTokenizers.Fallback is null
                      && GgufChatTemplates.Find(LlamaPlugin.ChatTemplateName) is null,
                    "without the family nothing of it is registered, and no pre-tokenizer fallback is set");

                // A library user without the package: the folder and the GGUF file are refused, naming the package.
                string loaded = Failure<NotSupportedException>(() => PretrainedModel.Load(TestData("gguf/tiny-llama-hf"), new PretrainedOptions { Device = Device.Cpu }).Dispose()).Message;
                Check(loaded.Contains("No architecture 'LlamaForCausalLM' is registered", StringComparison.Ordinal) && loaded.Contains("Idrak.Text.Plugins", StringComparison.Ordinal)
                      && loaded.Contains("RegisterAll()", StringComparison.Ordinal), $"a Llama folder without the package: {loaded}");
                string prepared = Failure<NotSupportedException>(() => GgufModel.Prepare(TestData("gguf/tiny-llama.gguf"), Path.Combine(folder, "refused"))).Message;
                Check(prepared.Contains("GGUF architecture 'llama' is not registered", StringComparison.Ordinal) && prepared.Contains("Idrak.Text.Plugins", StringComparison.Ordinal),
                    $"a llama GGUF file without the package: {prepared}");
                string scaling = Failure<NotSupportedException>(() => RopeScalings.Get(LlamaRopeScaling.Name)).Message;
                Check(scaling.Contains("'llama3'", StringComparison.Ordinal) && scaling.Contains("Idrak.Text.Plugins", StringComparison.Ordinal), $"llama3 without the package: {scaling}");

                LlamaPlugin.Register();
                LlamaPlugin.Register();
                string assembly = TextPlugins.Package;
                Check(LlamaPlugin.IsRegistered && ReferenceEquals(IdrakPlugins.Find("llama"), LlamaPlugin.Plugin)
                      && PretrainedArchitectures.Origin(LlamaPlugin.LlamaForCausalLM) == assembly && PretrainedArchitectures.Names.Count(n => n == LlamaPlugin.LlamaForCausalLM) == 1
                      && GgufArchitectures.Origin(LlamaPlugin.GgufName) == assembly && RopeScalings.Origin(LlamaRopeScaling.Name) == assembly
                      && LlamaPlugin.PreTokenizers.All(n => GgufPreTokenizers.Origin(n) == assembly && GgufPreTokenizers.TryGet(n, out string? p) && p == LlamaArchitectures.SplitPattern)
                      && GgufPreTokenizers.Fallback == "llama3" && GgufChatTemplates.Origin(LlamaPlugin.ChatTemplateName) == assembly
                      && GgufChatTemplates.Names.Count(n => n == LlamaPlugin.ChatTemplateName) == 1,
                    "Register (twice) adds the architecture, the GGUF mapping, the six pre-tokenizer names, the scaling, the chat template and the fallback once, from the plug-in");
                using (var model = PretrainedModel.Load(TestData("gguf/tiny-llama-hf"), new PretrainedOptions { Device = Device.Cpu }))
                {
                    Check(ReferenceEquals(model.Architecture, PretrainedArchitectures.Get(LlamaPlugin.LlamaForCausalLM)), "the folder loads with the family");
                }

                LlamaPlugin.Unregister();
                Check(!LlamaPlugin.IsRegistered && !PretrainedArchitectures.Names.Contains(LlamaPlugin.LlamaForCausalLM) && GgufArchitectures.Find(LlamaPlugin.GgufName) is null
                      && !RopeScalings.Contains(LlamaRopeScaling.Name) && GgufPreTokenizers.Fallback is null && GgufChatTemplates.Find(LlamaPlugin.ChatTemplateName) is null,
                    "Unregister takes every part back");

                // A fallback someone else set is left alone.
                GgufPreTokenizers.Fallback = "gpt2";
                try
                {
                    LlamaPlugin.Register();
                    Check(GgufPreTokenizers.Fallback == "gpt2", "an app's fallback stays when the family registers");
                    LlamaPlugin.Unregister();
                    Check(GgufPreTokenizers.Fallback == "gpt2", "and when it is unregistered");
                }
                finally
                {
                    GgufPreTokenizers.Fallback = null;
                }
            });
        }
        finally
        {
            Directory.Delete(folder, true);
        }

        // The tool registers the package at startup: in its own process (nothing of this one's), chat and run need no -P.
        string cache = TempFolder();
        try
        {
            var (exit, output, error) = IdrakProcess("hello\n/exit\n", cache, "chat", TestData("gguf/tiny-llama-hf"), "--max-tokens", "2", "--seed", "1");
            Check(exit == 0 && error.Length == 0 && output.Contains(" tokens", StringComparison.Ordinal) && !output.Contains("not registered", StringComparison.Ordinal), $"idrak chat on a Llama folder: exit {exit}, {output}{error}");
            (exit, output, error) = IdrakProcess(null, cache, "run", TestData("gguf/tiny-qwen3-q8.gguf"), "hi", "--max-tokens", "2", "-q");
            Check(exit == 0 && error.Length == 0, $"idrak run on a qwen3 GGUF file: exit {exit}, {output}{error}");
        }
        finally
        {
            Directory.Delete(cache, true);
        }
    }

    // The idrak tool built beside the tests, in its own process on the CPU (input on its standard input, IDRAK_CACHE in cache).
    private static (int Exit, string Output, string Error) IdrakProcess(string? input, string cache, params string[] args)
    {
        string dll = Path.Combine(AppContext.BaseDirectory, "Idrak.Cli.dll");
        var info = new System.Diagnostics.ProcessStartInfo("dotnet")
        {
            RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false,
            StandardOutputEncoding = System.Text.Encoding.UTF8,
        };
        foreach (string arg in (string[])[dll, .. args, "-d", "cpu", "-C", Path.Combine(cache, "no-config.json")])
        {
            info.ArgumentList.Add(arg);
        }

        info.Environment["DOTNET_CLI_TELEMETRY_OPTOUT"] = "1";
        info.Environment["DOTNET_NOLOGO"] = "1";
        info.Environment["IDRAK_CACHE"] = cache;
        info.Environment.Remove("IDRAK_PLUGIN_PACKAGES");
        using var process = System.Diagnostics.Process.Start(info)!;
        process.StandardInput.Write(input ?? "");
        process.StandardInput.Close();
        var error = process.StandardError.ReadToEndAsync();
        string output = process.StandardOutput.ReadToEnd();
        process.WaitForExit();
        return (process.ExitCode, output, error.Result);
    }

    private static void QwenPluginRegistration(Device device)
    {
        _ = device;
        CheckTextPlugin(QwenPlugin.Plugin, "qwen", "Idrak.Text.Plugins.Qwen");
        string folder = TempFolder();
        try
        {
            WithoutTextFamilies(() =>
            {
                string[] gguf = ["qwen2", "qwen3", "qwen2moe", "qwen3moe"];
                Check(!QwenPlugin.Architectures.Any(PretrainedArchitectures.Names.Contains) && gguf.All(n => GgufArchitectures.Find(n) is null)
                      && !QwenPlugin.PreTokenizers.Any(n => GgufPreTokenizers.TryGet(n, out _)) && !Formats.Names.Contains(QwenPlugin.ToolCallFormat),
                    "without the family nothing of it is registered");
                string loaded = Failure<NotSupportedException>(() => PretrainedModel.Load(TestData("gguf/tiny-qwen3-q8-hf"), new PretrainedOptions { Device = Device.Cpu }).Dispose()).Message;
                Check(loaded.Contains("No architecture 'Qwen3ForCausalLM' is registered", StringComparison.Ordinal) && loaded.Contains("Idrak.Text.Plugins", StringComparison.Ordinal),
                    $"a Qwen3 folder without the package: {loaded}");
                string prepared = Failure<NotSupportedException>(() => GgufModel.Prepare(TestData("gguf/tiny-qwen3-q8.gguf"), Path.Combine(folder, "refused"))).Message;
                Check(prepared.Contains("GGUF architecture 'qwen3' is not registered", StringComparison.Ordinal) && prepared.Contains("Idrak.Text.Plugins", StringComparison.Ordinal),
                    $"a qwen3 GGUF file without the package: {prepared}");

                QwenPlugin.Register();
                QwenPlugin.Register();
                string assembly = TextPlugins.Package;
                Check(QwenPlugin.IsRegistered && QwenPlugin.Architectures.All(n => PretrainedArchitectures.Origin(n) == assembly && PretrainedArchitectures.Names.Count(m => m == n) == 1)
                      && gguf.All(n => GgufArchitectures.Origin(n) == assembly)
                      && QwenPlugin.PreTokenizers.All(n => GgufPreTokenizers.TryGet(n, out string? p) && p == QwenArchitectures.SplitPattern)
                      && Formats.Origin(QwenPlugin.ToolCallFormat) == assembly && Formats.Names.Count(n => n == QwenPlugin.ToolCallFormat) == 1
                      && GgufArchitectures.Get("qwen3moe") is { HuggingFace: QwenPlugin.Qwen3MoeForCausalLM, NormalizeTopK: true },
                    "Register (twice) adds the four architectures, the four GGUF mappings, the pre-tokenizer names and the tool-call format once");
                using (var model = PretrainedModel.Load(TestData("gguf/tiny-qwen3-q8-hf"), new PretrainedOptions { Device = Device.Cpu }))
                {
                    Check(ReferenceEquals(model.Architecture, PretrainedArchitectures.Get(QwenPlugin.Qwen3ForCausalLM)) && model.Spec.QkNorm, "the folder loads with the family (q/k norm)");
                }

                QwenPlugin.Unregister();
                Check(!QwenPlugin.IsRegistered && !QwenPlugin.Architectures.Any(PretrainedArchitectures.Names.Contains) && gguf.All(n => GgufArchitectures.Find(n) is null)
                      && !Formats.Names.Contains(QwenPlugin.ToolCallFormat), "Unregister takes every part back");
            });
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }

    private static void MistralPluginRegistration(Device device)
    {
        _ = device;
        CheckTextPlugin(MistralPlugin.Plugin, "mistral", "Idrak.Text.Plugins.Mistral");
        WithoutTextFamilies(() =>
        {
            string message = Failure<NotSupportedException>(() => PretrainedArchitectures.Get(MistralPlugin.MixtralForCausalLM)).Message;
            Check(message.Contains("No architecture 'MixtralForCausalLM' is registered", StringComparison.Ordinal) && message.Contains("Idrak.Text.Plugins", StringComparison.Ordinal),
                $"Mixtral without the package: {message}");

            // Alone, it brings Llama's GGUF mapping (Mistral's files are llama.cpp's "llama"), and takes it back.
            MistralPlugin.Register();
            MistralPlugin.Register();
            Check(MistralPlugin.IsRegistered && LlamaPlugin.IsRegistered && GgufArchitectures.Find(LlamaPlugin.GgufName) is { WithExperts: MistralPlugin.MixtralForCausalLM }
                  && MistralPlugin.Architectures.All(n => PretrainedArchitectures.Origin(n) == TextPlugins.Package && PretrainedArchitectures.Names.Count(m => m == n) == 1)
                  && GgufPreTokenizers.TryGet(MistralPlugin.PreTokenizer, out string? tekken) && tekken == MistralArchitectures.TekkenPattern
                  && Formats.Names.Count(n => n == MistralPlugin.ToolCallFormat) == 1 && PretrainedArchitectures.Get(MistralPlugin.MistralForCausalLM) is IConfigWritingArchitecture,
                "Register (twice) adds the two architectures, tekken and the tool-call format once, and Llama's mapping it needs");
            using (var model = PretrainedModel.Load(TestData("gguf/tiny-llama-hf"), new PretrainedOptions { Device = Device.Cpu, Architecture = MistralPlugin.MistralForCausalLM }))
            {
                Check(ReferenceEquals(model.Architecture, PretrainedArchitectures.Get(MistralPlugin.MistralForCausalLM)), "a folder in the common layout loads as Mistral");
            }

            MistralPlugin.Unregister();
            Check(!MistralPlugin.IsRegistered && !LlamaPlugin.IsRegistered && !MistralPlugin.Architectures.Any(PretrainedArchitectures.Names.Contains)
                  && !Formats.Names.Contains(MistralPlugin.ToolCallFormat) && !GgufPreTokenizers.TryGet(MistralPlugin.PreTokenizer, out _),
                "Unregister takes its parts back, and Llama's, having added it");
            LlamaPlugin.Register();
            MistralPlugin.Register();
            MistralPlugin.Unregister();
            Check(LlamaPlugin.IsRegistered && !MistralPlugin.IsRegistered, "a Llama family registered before stays");
        });
    }

    private static void DeepSeekPluginRegistration(Device device)
    {
        _ = device;
        CheckTextPlugin(DeepSeekPlugin.Plugin, "deepseek", "Idrak.Text.Plugins.DeepSeek");
        WithoutTextFamilies(() =>
        {
            Check(new JinjaChatTemplate(DeepSeekV31Template, ["<｜end▁of▁sentence｜>"], "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>").ToolCallFormatName == Formats.Json,
                "without the format a DeepSeek template detects the library's json");
            string message = Failure<NotSupportedException>(() => Formats.Create(DeepSeekPlugin.ToolCallFormat, new ToolCallContext(new ChatMLTemplate(), [], null))).Message;
            Check(message.Contains("'deepseek'", StringComparison.Ordinal) && message.Contains("Idrak.Text.Plugins", StringComparison.Ordinal), $"deepseek without the package: {message}");

            DeepSeekPlugin.Register();
            DeepSeekPlugin.Register();
            Check(DeepSeekPlugin.IsRegistered && Formats.Names.Count(n => n == DeepSeekPlugin.ToolCallFormat) == 1
                  && Formats.Origin(DeepSeekPlugin.ToolCallFormat) == TextPlugins.Package
                  && GgufPreTokenizers.TryGet(DeepSeekPlugin.PreTokenizer, out string? pattern) && pattern == QwenArchitectures.SplitPattern
                  && new JinjaChatTemplate(DeepSeekV31Template, ["<｜end▁of▁sentence｜>"], "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>").ToolCallFormatName == DeepSeekPlugin.ToolCallFormat,
                "Register (twice) adds the format and the pre-tokenizer once; the template detects it");
            DeepSeekPlugin.Unregister();
            Check(!DeepSeekPlugin.IsRegistered && !Formats.Names.Contains(DeepSeekPlugin.ToolCallFormat) && !GgufPreTokenizers.TryGet(DeepSeekPlugin.PreTokenizer, out _),
                "Unregister takes them back");
        });
    }

    private static void HarmonyPluginRegistration(Device device)
    {
        _ = device;
        CheckTextPlugin(HarmonyPlugin.Plugin, "harmony", "Idrak.Text.Plugins.Harmony");
        WithoutTextFamilies(() =>
        {
            Check(new JinjaChatTemplate(HarmonyTemplate, ["<|return|>", "<|call|>"]).ToolCallFormatName == Formats.Json,
                "without the format a harmony template detects the library's json");
            string message = Failure<NotSupportedException>(() => Formats.Create(HarmonyPlugin.ToolCallFormat, new ToolCallContext(new ChatMLTemplate(), [], null))).Message;
            Check(message.Contains("'harmony'", StringComparison.Ordinal) && message.Contains("Idrak.Text.Plugins", StringComparison.Ordinal), $"harmony without the package: {message}");

            HarmonyPlugin.Register();
            HarmonyPlugin.Register();
            Check(HarmonyPlugin.IsRegistered && Formats.Names.Count(n => n == HarmonyPlugin.ToolCallFormat) == 1
                  && Formats.Origin(HarmonyPlugin.ToolCallFormat) == TextPlugins.Package
                  && new JinjaChatTemplate(HarmonyTemplate, ["<|return|>", "<|call|>"]).ToolCallFormatName == HarmonyPlugin.ToolCallFormat,
                "Register (twice) adds the format once; the template detects it");
            HarmonyPlugin.Unregister();
            Check(!HarmonyPlugin.IsRegistered && !Formats.Names.Contains(HarmonyPlugin.ToolCallFormat), "Unregister takes it back");
        });
    }
}
