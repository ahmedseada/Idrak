// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Formats;
using Idrak.Audio.Abstractions;
using Idrak.Cli.Shared;
using Idrak.Data;
using Idrak.Models;
using Idrak.Audio.Plugins.OpenVoice;

// PyTorch checkpoints read without running them (PyTorchCheckpointReader, the "pytorch" checkpoint format) and pulls of
// a model kept in a repository's subfolder. The fixture comes from tools/pytorch/checkpoint_reference.py
// (pytorch-checkpoints/) and tools/audio/make_tiny_audio.py --sections voices (tiny-openvoice/checkpoint.pth).
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] PyTorchCheckpointGroup =
    [
        ("pytorch checkpoint: state.pth and its protocol 4 copy give every tensor of reference.safetensors exactly (float32, float16, bfloat16, float64, int64, int8, bool, a 0-d scalar, a parameter, a transposed view, two views of one storage) with their shapes and formats; wrapped.pth reads its \"model\" entry, and the whole with root \"\"; nested names join by '.'; the legacy format is refused naming it; a pickle that would call os.system is refused naming it and never run", PyTorchCheckpointReads),
        ("pytorch checkpoint: the \"pytorch\" format opens a folder with one checkpoint and no safetensors, or the file, and leaves a safetensors folder to safetensors; several checkpoints are refused naming them; OpenVoice's converter loads from tiny-openvoice's checkpoint.pth and converts to the same bits as from its safetensors", PyTorchCheckpointFormatAndOpenVoice),
        ("pytorch checkpoint: a pull of owner/name/folder takes that folder's config and weights (safetensors first, else one PyTorch checkpoint), refuses sharded .bin weights and several checkpoints naming them, and names the subfolder that holds a config.json when the top level has none; idrak pull reads owner/name/folder as a repository and a folder", PyTorchCheckpointPullChoice),
    ];

    private static string PyTorchData(string name) => TestData(Path.Combine("pytorch-checkpoints", name));

    private static void PyTorchCheckpointReads(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                              // host reading: once
        }

        using ITensorStore reference = SafeTensorsReader.Open(PyTorchData("reference.safetensors"));
        foreach (string file in new[] { "state.pth", "state-p4.pth" })
        {
            using var reader = PyTorchCheckpointReader.Open(PyTorchData(file));
            Check(reader.Names.Order(StringComparer.Ordinal).SequenceEqual(reference.Names.Order(StringComparer.Ordinal)), $"{file}: names [{string.Join(", ", reader.Names)}]");
            foreach (string name in reference.Names)
            {
                Check(reader.ShapeOf(name).SequenceEqual(reference.ShapeOf(name)), $"{file}: {name} is [{string.Join(", ", reader.ShapeOf(name))}], [{string.Join(", ", reference.ShapeOf(name))}] expected");
                Check(reader.Read(name).AsSpan().SequenceEqual(reference.Read(name)), $"{file}: {name}'s values differ from reference.safetensors");
            }

            Check(reader.FormatOf("half") == WeightFormat.Float16 && reader.FormatOf("bf16") == WeightFormat.BFloat16 && reader.FormatOf("dense.weight") == WeightFormat.Float32
                && reader.FormatOf("steps") is null && reader.Root is null, $"{file}: the formats and root");
        }

        using (var wrapped = PyTorchCheckpointReader.Open(PyTorchData("wrapped.pth")))
        {
            Check(wrapped.Root == "model" && wrapped.Contains("dense.weight") && !wrapped.Names.Any(n => n.StartsWith("optimizer", StringComparison.Ordinal)),
                $"wrapped.pth: root {wrapped.Root}, names [{string.Join(", ", wrapped.Names.Take(4))}…]");
            Check(wrapped.Read("transposed").AsSpan().SequenceEqual(reference.Read("transposed")), "wrapped.pth: the model's tensors");
        }

        using (var whole = PyTorchCheckpointReader.Open(PyTorchData("wrapped.pth"), root: ""))
        {
            Check(whole.Root is null && whole.Contains("model.dense.weight"), $"wrapped.pth with root \"\": [{string.Join(", ", whole.Names.Take(3))}…]");
        }

        using (var nested = PyTorchCheckpointReader.Open(PyTorchData("nested.pth")))
        {
            Check(nested.Names.Order(StringComparer.Ordinal).SequenceEqual(["encoder.w", "layers.0", "layers.1"]), $"nested.pth: [{string.Join(", ", nested.Names)}]");
            Check(nested.Read("layers.1").AsSpan().SequenceEqual(reference.Read("double")), "nested.pth: a list's tensor");
        }

        ThrowsNaming<InvalidDataException>(() => PyTorchCheckpointReader.Open(PyTorchData("wrapped.pth"), root: "missing").Dispose(), "a root that is not there", "'missing'", "model");
        ThrowsNaming<InvalidDataException>(() => PyTorchCheckpointReader.Open(PyTorchData("legacy.pth")).Dispose(), "the legacy format", "legacy format");
        ThrowsNaming<InvalidDataException>(() => PyTorchCheckpointReader.Open(PyTorchData("evil.pth")).Dispose(), "a pickle calling os.system", "system", "allowlist", "not run");
        Check(PyTorchCheckpointReader.IsCheckpoint(PyTorchData("state.pth")) && !PyTorchCheckpointReader.IsCheckpoint(PyTorchData("legacy.pth"))
            && !PyTorchCheckpointReader.IsCheckpoint(PyTorchData("reference.safetensors")), "IsCheckpoint tells a zip checkpoint from the legacy format and safetensors");
    }

    private static void PyTorchCheckpointFormatAndOpenVoice(Device device)
    {
        string root = TempFolder("pytorch-format");
        try
        {
            if (device.Type == DeviceType.Cpu)
            {
                string one = Path.Combine(root, "one");
                Directory.CreateDirectory(one);
                File.Copy(PyTorchData("state.pth"), Path.Combine(one, "weights.pth"));
                var format = CheckpointFormats.For(one);
                Check(format.Name == "pytorch" && CheckpointFormats.For(Path.Combine(one, "weights.pth")).Name == "pytorch"
                    && CheckpointFormats.For(TinyOpenVoice).Name == "safetensors", $"the formats: {format.Name} for a folder with one checkpoint; safetensors where there is a .safetensors");
                using (var store = format.Open(format.Prepare(Path.Combine(one, "weights.pth"))))
                {
                    Check(store.Contains("dense.weight") && format.Notes(one).Single().Contains("nothing in it is run", StringComparison.Ordinal), "the format opens the file's folder, with a note");
                }

                File.Copy(PyTorchData("nested.pth"), Path.Combine(one, "other.pt"));
                Check(CheckpointFormats.For(one).Name == "safetensors", "a folder with several checkpoints is not taken by the pytorch format");
                ThrowsNaming<InvalidDataException>(() => new PyTorchCheckpointFormat().Open(one).Dispose(), "several checkpoints", "other.pt", "weights.pth");
            }

            OpenVoicePlugin.Register();
            string converter = Path.Combine(root, "openvoice-pth");
            Directory.CreateDirectory(converter);
            File.Copy(Path.Combine(TinyOpenVoice, "config.json"), Path.Combine(converter, "config.json"));
            File.Copy(Path.Combine(TinyOpenVoice, "checkpoint.pth"), Path.Combine(converter, "checkpoint.pth"));
            var (source, clips) = OpenVoiceInputs();
            using var fromSafetensors = LoadOpenVoice(TinyOpenVoice, device);
            using var fromPickle = LoadOpenVoice(converter, device);
            var target = fromSafetensors.Embed(clips);
            Check(fromPickle.Embed(clips).AsSpan().SequenceEqual(target), "the embedding from checkpoint.pth equals the safetensors one");
            var exact = new VoiceConversionOptions { Tau = 0 };
            Check(fromPickle.Convert(source, target, exact).Samples.AsSpan().SequenceEqual(fromSafetensors.Convert(source, target, exact).Samples),
                "the conversion from checkpoint.pth equals the safetensors one bit for bit");
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void PyTorchCheckpointPullChoice(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;
        }

        RepoFile F(string path) => new(path, 100);
        List<RepoFile> repo =
        [
            F("README.md"), F("converter/config.json"), F("converter/checkpoint.pth"), F("base_speakers/ses/en-us.pth"),
            F("other/config.json"), F("other/model.safetensors"), F("other/model.pth"), F("other/tokenizer.json"),
        ];
        var converter = HuggingFaceModels.ChooseFiles("myshell-ai/OpenVoiceV2", repo, "converter").Select(f => f.Path).ToList();
        Check(converter.SequenceEqual(["converter/config.json", "converter/checkpoint.pth"]), $"converter/: [{string.Join(", ", converter)}]");
        var other = HuggingFaceModels.ChooseFiles("o/r", repo, "other/").Select(f => f.Path).Order(StringComparer.Ordinal).ToList();
        Check(other.SequenceEqual(["other/config.json", "other/model.safetensors", "other/tokenizer.json"]), $"safetensors before a checkpoint: [{string.Join(", ", other)}]");
        ThrowsNaming<FileNotFoundException>(() => HuggingFaceModels.ChooseFiles("myshell-ai/OpenVoiceV2", repo), "no config.json at the top", "no config.json", "pull myshell-ai/OpenVoiceV2/");
        ThrowsNaming<FileNotFoundException>(() => HuggingFaceModels.ChooseFiles("o/r", [F("config.json"), F("pytorch_model-00001-of-00002.bin"), F("pytorch_model.bin.index.json")]),
            "sharded .bin weights", "sharded PyTorch weights");
        ThrowsNaming<FileNotFoundException>(() => HuggingFaceModels.ChooseFiles("o/r", [F("config.json"), F("a.pth"), F("b.pt")]), "several checkpoints", "a.pth", "b.pt");
        var bin = HuggingFaceModels.ChooseFiles("o/r", [F("config.json"), F("pytorch_model.bin"), F("training_args.bin"), F("vocab.json")]).Select(f => f.Path).ToList();
        Check(bin.Contains("pytorch_model.bin") && !bin.Contains("training_args.bin") && bin.Contains("vocab.json"), $"pytorch_model.bin among several .bin: [{string.Join(", ", bin)}]");

        Check(PullTarget.SplitFolder("myshell-ai/OpenVoiceV2/converter") == ("myshell-ai/OpenVoiceV2", "converter")
            && PullTarget.SplitFolder("o/r/a/b") == ("o/r", "a/b") && PullTarget.SplitFolder("o/r") is null && PullTarget.SplitFolder("o/r/x.gguf") is null
            && PullTarget.SplitFolder("https://example.com/a/b/c") is null && PullTarget.SplitFolder("o/r:Q4") is null, "SplitFolder reads owner/name/folder only");
    }
}
