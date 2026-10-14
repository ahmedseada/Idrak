// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.IdrakVision;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Nlp;
using Idrak.Nlp.Abstractions;

// Plan 19, phase B5: distillation from a vision-language teacher: the teacher reads the student's images itself.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] DistillVisionGroup =
    [
        ("distill vision: a vision-language teacher scores a student's image sequences (other image token counts) as it scores its own encoding of the same records", DistillVisionTeacher),
        ("distill vision: pretraining with a teacher: alpha 0 is the plain loss, alpha 0.5 learns from the teacher; stored top-k logits with images read back; a text teacher is refused", DistillVisionTraining),
        ("distill vision: idrak distill from a vision-language teacher into a vision-language student on image records, with the family's plug-in", DistillVisionCli),
    ];

    // A student of the same family with another tower: 16 image tokens an image (no pooling) where the teacher has 4.
    private static PretrainedModel ScratchStudent(string folder, Device device)
    {
        var config = JsonNode.Parse(File.ReadAllText(ScratchData("tiny-idrak-vision/config.json")))!.AsObject();
        var spec = PretrainedArchitectures.Get(IdrakVisionFamily.Architecture).Spec(config, []);
        var vision = ScratchDescription();
        vision["vision_config"]!["pool_size"] = 1;
        vision["vision_config"]!["num_hidden_layers"] = 1;
        return PretrainedModel.Create(folder, spec, IdrakVisionFamily.Architecture, new ModelFiles { Tokenizer = BpeTokenizer.Load(PretrainData("tokenizer")) },
            new CreateOptions { Device = device, Seed = 9, Vision = vision });
    }

    private static List<TrainingSequence> Encoded(PretrainedModel model, TuningVision vision)
    {
        var encoder = new ChatTranscriptEncoder(model.JinjaTemplate!, model.Tokenizer!) { Vision = vision };
        using var records = TuningDataFormats.Get("messages").ReadPending(ScratchData("records.jsonl"));
        return [.. Enumerable.Range(0, records.Count).Select(i => encoder.Encode(records.Read(i), 128)!)];
    }

    private static void DistillVisionTeacher(Device device)
    {
        IdrakVisionPlugin.Register();
        string folder = TempFolder("distill-vision-teacher");
        try
        {
            using var teacherModel = LoadScratch(device);
            using var student = ScratchStudent(Path.Combine(folder, "student"), device);
            using var studentVision = TuningVision.Create(student);
            var studentSequences = Encoded(student, studentVision);
            Check(studentSequences.All(s => s.Images.All(i => i.Blocks.All(b => b.Tokens == 16))), "the student's images are 16 tokens");

            // The teacher's own reading of the same records: logits at its trained positions, softmax at temperature 1.
            using var teacherVision = TuningVision.Create(teacherModel);
            var own = Encoded(teacherModel, teacherVision);
            var expected = new List<float>();
            int vocabulary = teacherModel.Spec.Vocabulary;
            foreach (var sequence in own)
            {
                using var scope = new TensorScope();
                var features = teacherVision.Features(sequence.Images);
                var images = sequence.Images.SelectMany((im, k) => im.Blocks.Select(b => new PromptImage(b.Position, features[k]))).ToList();
                var ids = Tensor.From(sequence.Tokens[..^1].Select(t => (float)t).ToArray(), [1, sequence.Tokens.Length - 1], device);
                var logits = teacherModel.Network.Forward(ids, images, teacherVision.Attention).ToArray();
                for (int t = 0; t + 1 < sequence.Tokens.Length; t++)
                {
                    if (sequence.Trained[t + 1])
                    {
                        var row = logits.AsSpan(t * vocabulary, vocabulary);
                        float max = float.MinValue, sum = 0;
                        foreach (float v in row) { max = Math.Max(max, v); }
                        foreach (float v in row) { sum += MathF.Exp(v - max); }
                        foreach (float v in row) { expected.Add(MathF.Exp(v - max) / sum); }
                    }
                }
            }

            using var teacher = DistillationTeachers.FromModel(teacherModel, 0, 4096, studentVision.Images);
            teacher.Check(student.Tokenizer!);
            using var distributions = teacher.Distributions(studentSequences);
            int trained = studentSequences.Sum(s => s.TrainedTokens);
            Check(trained * vocabulary == expected.Count, $"{trained} trained positions of the student, {expected.Count / vocabulary} of the teacher's own");
            using var probabilities = distributions.Probabilities(0, trained, 1f, vocabulary, device);
            AssertClose([.. expected], probabilities.ToArray(), 1e-5f, "the teacher's distributions on the student's sequences against its own reading");
        }
        finally
        {
            IdrakVisionPlugin.Unregister();
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void DistillVisionTraining(Device device)
    {
        IdrakVisionPlugin.Register();
        string folder = TempFolder("distill-vision-training");
        try
        {
            using var teacherModel = LoadScratch(device);
            var options = VisionStage("joint", ["all"], 2) with { Optimizer = "adamw", Seed = 1 };
            IReadOnlyList<float> plain;
            using (var student = ScratchStudent(Path.Combine(folder, "plain"), device))
            {
                plain = Pretrainer.Train(student, options, Path.Combine(folder, "plain-run")).Losses;
            }

            using (var teacher = DistillationTeachers.FromModel(teacherModel, 0, 4096, TuningImages.None))
            using (var student = ScratchStudent(Path.Combine(folder, "zero"), device))
            {
                var zero = Pretrainer.Train(student, options with { Teacher = teacher, DistillationAlpha = 0f }, Path.Combine(folder, "zero-run")).Losses;
                AssertClose([.. plain], [.. zero], 1e-4f, "alpha 0 with a teacher is the plain loss");
            }

            using (var teacher = DistillationTeachers.FromModel(teacherModel, 0, 4096, TuningImages.None))
            using (var student = ScratchStudent(Path.Combine(folder, "half"), device))
            {
                var half = Pretrainer.Train(student, options with { Teacher = teacher, DistillationAlpha = 0.5f, DistillationTemperature = 2f }, Path.Combine(folder, "half-run")).Losses;
                Check(half.All(float.IsFinite) && Math.Abs(half[0] - plain[0]) > 1e-3f, $"alpha 0.5 learns from the teacher: {string.Join(", ", half)} against {string.Join(", ", plain)}");
            }

            // Stored top-k logits of sequences with images, read back as the on-the-fly teacher's top k.
            using (var student = ScratchStudent(Path.Combine(folder, "stored"), device))
            using (var vision = TuningVision.Create(student))
            {
                var sequences = Encoded(student, vision);
                string path = Path.Combine(folder, "teacher.topk");
                int written = TeacherLogitsWriter.Write(path, teacherModel, sequences, 8, null, 4096, vision.Images);
                Check(written == sequences.Count, $"{written} records written");
                using var stored = DistillationTeachers.FromFile(path);
                using var live = DistillationTeachers.FromModel(teacherModel, 8, 4096, vision.Images);
                int trained = sequences.Sum(s => s.TrainedTokens);
                using var a = stored.Distributions(sequences);
                using var b = live.Distributions(sequences);
                AssertClose(b.Probabilities(0, trained, 1f, teacherModel.Spec.Vocabulary, device).ToArray(),
                    a.Probabilities(0, trained, 1f, teacherModel.Spec.Vocabulary, device).ToArray(), 2e-3f, "stored top-k with images against the live teacher's");
            }

            // A text teacher cannot read the images.
            using (var text = PretrainedModel.Load(TinyLlama, new PretrainedOptions { Device = device }))
            using (var teacher = DistillationTeachers.FromModel(text, 0, 4096, TuningImages.None))
            using (var student = ScratchStudent(Path.Combine(folder, "text"), device))
            {
                Throws<NotSupportedException>(() => Pretrainer.Train(student, options with { Teacher = teacher }, Path.Combine(folder, "text-run")), "a text teacher for image records");
            }
        }
        finally
        {
            IdrakVisionPlugin.Unregister();
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void DistillVisionCli(Device device)
    {
        string plugin = typeof(IdrakVisionPlugin).Assembly.Location;
        string folder = TempFolder("distill-vision-cli");
        IdrakVisionPlugin.Register();
        try
        {
            string student = Path.Combine(folder, "student");
            using (ScratchStudent(student, device))
            {
            }

            var (code, output, error) = TrainCli("distill", "--teacher", ScratchData("tiny-idrak-vision"), "--student", student, "--data", ScratchData("records.jsonl"),
                "--epochs", "1", "--max-length", "128", "--batch-tokens", "256", "-o", Path.Combine(folder, "adapters"), "-P", plugin, "-d", device.ToString());
            Check(code == 0, $"idrak distill with images: exit {code}\n{output}\n{error}");
            Check(output.Contains("distilling", StringComparison.Ordinal), output);
            (code, output, _) = TrainCli("pretrain", student, "--data", ScratchData("records.jsonl"), "--trained", "all", "--sequences", "3", "--max-length", "128",
                "--batch-tokens", "384", "--steps", "1", "--teacher", ScratchData("tiny-idrak-vision"), "--alpha", "0.7", "-o", Path.Combine(folder, "run"), "-P", plugin, "-d", device.ToString());
            Check(code == 0 && output.Contains("teacher", StringComparison.Ordinal), $"pretrain with a teacher: {output}");
        }
        finally
        {
            IdrakVisionPlugin.Unregister();
            Directory.Delete(folder, recursive: true);
        }
    }
}
