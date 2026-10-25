// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Plugins;
using Idrak.Abstraction.Testing;
using Idrak.Inference;
using Idrak.Models.Abstractions;
using Idrak.Vision;
using Idrak.Vision.Abstractions;
using Idrak.Vision.Plugins;
using Idrak.Vision.Plugins.Crnn;
using Idrak.Vision.Plugins.Gemma;
using Idrak.Vision.Plugins.IdrakVision;
using Idrak.Vision.Plugins.Llava;
using Idrak.Vision.Plugins.Qwen;

// Plan 26, phase P5: the vision families in Idrak.Vision.Plugins (Qwen-VL, LLaVA, IdrakVision, the CRNN line recognizer).
// Each family's behaviour keeps its own groups (qwen vl:, vision contracts, vlm scratch:, ...); these check the plug-ins
// and the CRNN family, which became an image model family of the line-recognition task here.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] VisionPluginGroup =
    [
        ("plugins: vision: QwenVisionPlugin (\"qwen-vl\"), LlavaPlugin (\"llava\"), IdrakVisionPlugin (\"idrak-vision\") and CrnnPlugin (\"crnn\") are VisionPlugins.All, in that order before Gemma3VisionPlugin (plugins: gemma), each in its family's namespace; Register twice registers their architectures once, from Idrak.Vision.Plugins; Unregister leaves VisionFamilies, PretrainedArchitectures and ImageModelFamilies as found", VisionPluginRegistration),
        ("plugins: crnn: a recognizer saved by CrnnRecognizer loads through ImageModels as CrnnForLineRecognition and reads lines as the recognizer does (texts, labels, scores), passes the image model family suite (a line at a time: its backward LSTM reads a batch's padding); a folder of the earlier package layout loads with the same readings; a missing or misshapen tensor and bad settings are refused naming them; unregistered, the folder is refused naming Idrak.Vision.Plugins", CrnnFamilyRoundTrip),
    ];

    private static readonly (IIdrakPlugin Plugin, string Name, string Namespace)[] VisionFamilyPlugins =
    [
        (QwenVisionPlugin.Plugin, "qwen-vl", "Idrak.Vision.Plugins.Qwen"),
        (LlavaPlugin.Plugin, "llava", "Idrak.Vision.Plugins.Llava"),
        (IdrakVisionPlugin.Plugin, "idrak-vision", "Idrak.Vision.Plugins.IdrakVision"),
        (CrnnPlugin.Plugin, "crnn", "Idrak.Vision.Plugins.Crnn"),
    ];

    private static void VisionPluginRegistration(Device device)
    {
        _ = device;
        Check(VisionPlugins.All.SequenceEqual([.. VisionFamilyPlugins.Select(p => p.Plugin), Gemma3VisionPlugin.Plugin]),
            $"VisionPlugins.All lists the four, in order, then Gemma 3 vision ({string.Join(", ", VisionPlugins.All.Select(p => p.Name))})");
        foreach (var (plugin, name, ns) in VisionFamilyPlugins)
        {
            Check(plugin.Name == name && plugin.GetType().Namespace == ns, $"{name}: named so, in {ns} ({plugin.Name}, {plugin.GetType().Namespace})");
        }

        string[] vision = [QwenVisionFamily.Qwen25VlArchitecture, QwenVisionFamily.Qwen2VlArchitecture, LlavaPlugin.Architecture, IdrakVisionFamily.Architecture];
        string[] image = [CrnnPlugin.Architecture];
        var was = VisionFamilyPlugins.Select(p => p.Plugin.IsRegistered).ToArray();
        static List<string> Sorted(IEnumerable<string> names) => [.. names.Order(StringComparer.Ordinal)];
        try
        {
            foreach (var (plugin, _, _) in Enumerable.Reverse(VisionFamilyPlugins))
            {
                plugin.Unregister();
            }

            Check(vision.All(a => VisionFamilies.Find(a) is null && !PretrainedArchitectures.Names.Contains(a)) && ImageModelFamilies.Find(CrnnPlugin.Architecture) is null,
                "unregistered: none of their architectures is registered");
            var (visionFound, textFound, imageFound) = (Sorted(VisionFamilies.Names), Sorted(PretrainedArchitectures.Names), Sorted(ImageModelFamilies.Names));
            for (int twice = 0; twice < 2; twice++)
            {
                QwenVisionPlugin.Register();
                LlavaPlugin.Register();
                IdrakVisionPlugin.Register();
                CrnnPlugin.Register();
            }

            Check(VisionFamilyPlugins.All(p => p.Plugin.IsRegistered && ReferenceEquals(IdrakPlugins.Find(p.Name), p.Plugin)), "registered and recorded by name");
            Check(Sorted(VisionFamilies.Names).SequenceEqual(Sorted(visionFound.Concat(vision))) && Sorted(PretrainedArchitectures.Names).SequenceEqual(Sorted(textFound.Concat(vision)))
                && Sorted(ImageModelFamilies.Names).SequenceEqual(Sorted(imageFound.Concat(image))), "Register twice registers each architecture once");
            Check(vision.All(a => VisionFamilies.Origin(a) == VisionPlugins.Package.Name && PretrainedArchitectures.Origin(a) == VisionPlugins.Package.Name)
                && ImageModelFamilies.Origin(CrnnPlugin.Architecture) == VisionPlugins.Package.Name, "every architecture is Idrak.Vision.Plugins'");
            foreach (var (plugin, _, _) in VisionFamilyPlugins)
            {
                plugin.Unregister();
            }

            Check(Sorted(VisionFamilies.Names).SequenceEqual(visionFound) && Sorted(PretrainedArchitectures.Names).SequenceEqual(textFound) && Sorted(ImageModelFamilies.Names).SequenceEqual(imageFound),
                "Unregister leaves the registries as found");
        }
        finally
        {
            for (int i = 0; i < was.Length; i++)
            {
                if (was[i])
                {
                    VisionFamilyPlugins[i].Plugin.Register();
                }
                else
                {
                    VisionFamilyPlugins[i].Plugin.Unregister();
                }
            }
        }
    }

    private static void CrnnFamilyRoundTrip(Device device)
    {
        bool wasRegistered = CrnnPlugin.IsRegistered;
        string folder = TempFolder("crnn-family");
        var settings = new CrnnSettings { Alphabet = ["a", "b", "c"], Height = 8, RightToLeft = false, Channels = [4, 4, 4], Hidden = 4, Layers = 1 };
        ImageData[] lines = [LineImage("abc"), LineImage("cab a"), LineImage("b", 16), LineImage("ca")];
        var options = new CtcDecodeOptions();
        try
        {
            CrnnPlugin.Register();
            string saved = Path.Combine(folder, "saved");
            IReadOnlyList<RecognizedLine> expected;
            using (var made = CrnnRecognizer.Create(settings, device, seed: 5))
            {
                expected = made.Read(lines, CtcDecoders.Greedy, options, new MeasuredLineBatches(device, ceiling: 1));
                made.Save(saved, new JsonObject { ["note"] = "the training's record" });

                // The earlier layout: the network's description and weights in a package, the settings beside it.
                string legacy = Path.Combine(folder, "legacy");
                Directory.CreateDirectory(legacy);
                ModelPackage.Create(Path.Combine(legacy, CrnnRecognizer.LegacyPackageFile))
                    .Architecture(ModelPackage.DefaultModelName, CrnnRecognizer.Describe(settings, device, 5).ToJson()).Weights(made.Network).Save();
                File.WriteAllText(Path.Combine(legacy, CrnnRecognizer.LegacySettingsFile), settings.ToJson().ToJsonString());
                using var old = CrnnRecognizer.Load(legacy, device);
                CheckSameReadings(expected, old.Read(lines, CtcDecoders.Greedy, options, new MeasuredLineBatches(device, ceiling: 1)), "the earlier layout");
            }

            var config = JsonNode.Parse(File.ReadAllText(Path.Combine(saved, CrnnRecognizer.ConfigFile)))!.AsObject();
            Check((string?)config["architectures"]![0] == CrnnPlugin.Architecture && (string?)config["reading_order"] == ReadingOrders.Runs && (string?)config["training"]!["note"] is not null
                  && CrnnRecognizer.IsRecognizerFolder(saved), $"config.json: {config}");
            using (var loaded = CrnnRecognizer.Load(saved, device))
            {
                CheckSameReadings(expected, loaded.Read(lines, CtcDecoders.Greedy, options, new MeasuredLineBatches(device, ceiling: 1)), "loaded through the family");
                Check(loaded.Encode("ab c").Labels.SequenceEqual([1, 2, 3]) && loaded.Text([3, 2, 1]) == "cba", "labels after the blank, text left to right");
            }

            using (var model = ImageModels.Load(saved, new ImageModelOptions { Device = device }))
            {
                Check(model.Task == ImageTask.LineRecognition && model.Channels == 1 && model.Labels!.SequenceEqual(["", "a", "b", "c"])
                      && model.Lines is { Height: 8, Stride: CrnnSettings.Stride, Blank: 0, RightToLeft: false, ReadingOrder: ReadingOrders.Runs }, "the image model");
                var reader = model.LineReader();
                for (int i = 0; i < lines.Length; i++)
                {
                    CheckSameReadings([expected[i]], reader.Read([lines[i]]), $"line {i + 1} through model.LineReader()");
                }
            }

            // The suite reads a line at a time: the backward LSTM reads a batch's right padding before a line, so a CRNN batches
            // lines of similar widths (MeasuredLineBatches), never the suite's mix of sizes.
            Conformance.CheckImageModelFamily(p => CrnnModelUnderTest(p, device), [new ImageFamilySample(saved, [.. lines], Tolerance: 0f)],
                new ContractCheckOptions { RandomCases = 4 }).ThrowIfFailed();

            // Refusals: a tensor the settings' network does not hold, one it holds in another shape, settings that make no network.
            string Edited(string name, Action<JsonObject> edit)
            {
                string copy = Path.Combine(folder, name);
                Directory.CreateDirectory(copy);
                File.Copy(Path.Combine(saved, CrnnRecognizer.WeightsFile), Path.Combine(copy, CrnnRecognizer.WeightsFile));
                var json = JsonNode.Parse(File.ReadAllText(Path.Combine(saved, CrnnRecognizer.ConfigFile)))!.AsObject();
                edit(json);
                File.WriteAllText(Path.Combine(copy, CrnnRecognizer.ConfigFile), json.ToJsonString());
                return copy;
            }

            ThrowsNaming<InvalidDataException>(() => CrnnRecognizer.Load(Edited("deeper", j => j["layers"] = 2), device).Dispose(), "a layer more", "has no layers.");
            ThrowsNaming<InvalidDataException>(() => CrnnRecognizer.Load(Edited("wider", j => j["hidden"] = 6), device).Dispose(), "another hidden size", "layers.", "the settings' network holds");
            ThrowsNaming<InvalidDataException>(() => CrnnRecognizer.Load(Edited("height", j => j["height"] = 10), device).Dispose(), "a height of 10", "multiple of 4");
            ThrowsNaming<FileNotFoundException>(() => CrnnRecognizer.Load(Path.Combine(folder, "nothing"), device).Dispose(), "an empty folder", "is not a line recognizer folder");

            CrnnPlugin.Unregister();
            ThrowsNaming<NotSupportedException>(() => ImageModels.Load(saved).Dispose(), "the family unregistered", CrnnPlugin.Architecture, "not registered", VisionPlugins.Package.Name);
        }
        finally
        {
            if (wasRegistered)
            {
                CrnnPlugin.Register();
            }
            else
            {
                CrnnPlugin.Unregister();
            }

            Directory.Delete(folder, recursive: true);
        }
    }

    // A CRNN model for the image family suite: loaded through ImageModels, each line read alone through model.LineReader().
    private static ImageModelUnderTest CrnnModelUnderTest(string path, Device device)
    {
        var model = ImageModels.Load(path, new ImageModelOptions { Device = device });
        var reader = model.LineReader();
        return new ImageModelUnderTest("line-recognition", model.Labels,
            images => [.. images.Select(image => reader.Read([image])[0].Labels.Select(v => (float)v).ToArray())], model);
    }

    // The same readings: texts and labels equal, the scores within float rounding.
    private static void CheckSameReadings(IReadOnlyList<RecognizedLine> expected, IReadOnlyList<RecognizedLine> actual, string what)
    {
        Check(expected.Count == actual.Count, $"{what}: {actual.Count} readings, {expected.Count} expected");
        for (int i = 0; i < expected.Count; i++)
        {
            Check(expected[i].Text == actual[i].Text && expected[i].Labels.SequenceEqual(actual[i].Labels) && Math.Abs(expected[i].LogProbability - actual[i].LogProbability) < 1e-5
                  && Math.Abs(expected[i].Confidence - actual[i].Confidence) < 1e-5,
                $"{what}, line {i + 1}: '{actual[i].Text}' {actual[i].LogProbability}, expected '{expected[i].Text}' {expected[i].LogProbability}");
        }
    }
}
