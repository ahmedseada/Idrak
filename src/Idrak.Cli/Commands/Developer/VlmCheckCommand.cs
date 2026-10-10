// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Cli.Commands.Run;
using Idrak.Cli.Shared;
using Idrak.Generation;
using Idrak.Data.Abstractions;
using Idrak.Generation.Abstractions;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;

namespace Idrak.Cli.Commands.Developer;

/// <summary>
/// <c>idrak vlm check MODEL --reference DIR</c>: compares a vision-language model (any registered vision family whose
/// encoder shows its stages, <see cref="IVisionEncoderStages"/>; Gemma 3's with its plug-in) on the device with what
/// transformers saved for one image and prompt (tools/vlm/compare_real.py): the pixels, the vision encoder's output and
/// the projected image features, the logits of every prompt position and of transformers' greedy answer fed back
/// (teacher forcing), with Idrak's features and with transformers' own (to tell the image side from the decoder), and
/// Idrak's own greedy tokens. At each disagreement both top-5 lists and margins say whether it is a near-tie.
/// </summary>
/// <remarks>
/// A developer command beside <c>onnx check</c> (Idrak against another runtime) rather than a <c>check</c> option: it
/// reads a folder of arrays, needs an image, and reports per position; nothing in it depends on the device.
/// </remarks>
internal sealed class VlmCheckCommand : Command
{
    private const int Top = 5;

    public override string Name => "vlm check";

    public override string Summary => "Compares a vision-language model with a transformers reference: pixels, features, logits, tokens";

    public override string Usage =>
        "MODEL --reference DIR [options]\n\n" +
        "DIR is the folder tools/vlm/compare_real.py writes (manifest.json and .npy arrays: the prompt's ids, the pixels,\n" +
        "the vision output, the projected features, transformers' greedy tokens with the top 5 logits of every step, and\n" +
        "the top 5 of one teacher-forced pass). The model loads on the device with the chosen weights; the image named in\n" +
        "the manifest (or --image) is read as idrak run reads it, grey when the reference was, through the image transforms\n" +
        "the reference was made with (compare_real.py --image-transform, kept as \"image_transforms\"). Reported:\n" +
        "  - pixels: the largest difference from the reference's;\n" +
        "  - the encoder's output and the projected features from the reference's pixels: largest, relative and cosine;\n" +
        "  - the prompt's ids as Idrak's chat template and tokenizer make them;\n" +
        "  - teacher forcing: transformers' tokens fed back through the KV cache, with Idrak's features and with the\n" +
        "    reference's; top-1 agreement per position, and at each disagreement both top-5 lists and both margins (top 1\n" +
        "    minus top 2): below --tie on either side it is a near-tie (precision), above it a real difference;\n" +
        "  - Idrak's own greedy tokens and where they first leave transformers';\n" +
        "  - a verdict. Exit code 1 when a disagreement is not a near-tie (or the prompt's ids differ).\n\n" +
        "Options:\n" +
        "      --reference DIR   the folder compare_real.py wrote (required)\n" +
        "      --image FILE      the image (default: the manifest's \"image\", relative to DIR when not absolute)\n" +
        "      --steps N         compare the first N answer tokens (default: all the reference has)\n" +
        "      --tie X           a margin below X is a near-tie (default 0.5)\n" +
        "      --show N          disagreements printed in full (default 10)\n" +
        "      --vision-option K=V  a vision family option over the reference's (compare_real.py records the processor\n" +
        "                        options it used, such as --pan-and-scan, as \"vision_options\"; repeatable)\n" +
        "      --image-transform P  the image transforms instead of the reference's (the same syntax as run's)\n" +
        "  -w, --weights FORMAT  int8, int4, bf16 or a registered packed format (default: as stored); the encoder is float32\n" +
        "  -k, --kv FORMAT       the KV cache format (default float32)\n" +
        "      --context N       the context window in tokens (default 4096, at most the model's)\n" +
        "      --adapter DIR     merge a LoRA adapter into the weights as they are read\n\n" +
        "Examples:\n" +
        "  python tools/vlm/compare_real.py ./gemma-ocr scan.jpg --grayscale --prompt-file prompt.txt --out ref\n" +
        "  python tools/vlm/compare_real.py ./gemma-ocr scan.jpg --image-transform grayscale,max_width=1024,contrast=1.5 --out rc\n" +
        "  idrak vlm check ./gemma-ocr --reference ref -d cuda:0 -w bf16\n" +
        "  idrak vlm check ./gemma-ocr --reference ref -d cpu -j -O report.json";

    public override IReadOnlyCollection<string> ValueOptions => [.. ModelChoices.ValueOptions, "--reference", "--image", "--steps", "--tie", "--show", ModelChoices.VisionOption, ModelChoices.ImageTransformOption];

    public override IReadOnlyDictionary<string, string> ShortForms => ModelChoices.ShortForms;

    /// <summary>One answer step: the token transformers chose, both sides' top 5 and the differences on the reference's top 5.</summary>
    private sealed record Step(int Index, int Expected, int[] RefIds, float[] RefLogits, int[] Ids, float[] Logits, float TopFiveDifference)
    {
        public bool Agrees => Ids[0] == Expected;

        public float RefMargin => RefLogits[0] - RefLogits[1];

        public float Margin => Logits[0] - Logits[1];
    }

    private sealed record Pass(string Name, List<Step> Steps, int PromptAgree, int PromptRows, int AfterImageAgree, int AfterImageRows, float[] FullRowDifferences);

    public override int Run(CommandContext context)
    {
        string name = context.Argument(0, "MODEL (a Hugging Face id, a folder or an alias)");
        string folder = context.Option("--reference") ?? throw new UsageException("Missing --reference DIR (the folder tools/vlm/compare_real.py writes).");
        string manifestPath = Path.Combine(folder, "manifest.json");
        if (!File.Exists(manifestPath))
        {
            throw new UsageException($"No {manifestPath}: make the folder with tools/vlm/compare_real.py.");
        }

        var manifest = JsonNode.Parse(File.ReadAllText(manifestPath)) as JsonObject ?? throw new InvalidDataException($"{manifestPath} is not a JSON object.");
        if ((string?)manifest["kind"] != "idrak-vlm-reference")
        {
            throw new UsageException($"{manifestPath} is not a reference written by tools/vlm/compare_real.py (its \"kind\" is not idrak-vlm-reference).");
        }

        float tie = context.Option("--tie") is not { } t ? 0.5f
            : float.TryParse(t, NumberStyles.Float, CultureInfo.InvariantCulture, out float parsedTie) && parsedTie >= 0 ? parsedTie
            : throw new UsageException($"--tie needs a number such as 0.5, not '{t}'.");
        int show = context.IntOption("--show", 10);
        bool grayscale = (bool?)manifest["grayscale"] ?? false;
        // The family's options the reference was made with (compare_real.py's processor keyword arguments), as the
        // encoder is given them; a run's own --vision-option goes over them.
        var visionOptions = (manifest["vision_options"] is JsonObject saved ? VisionOptions.FromJson(saved) : VisionOptions.Empty).With(ModelChoices.VisionOptionsOf(context));

        // The image transforms the reference ran in Pillow before its processor (compare_real.py --image-transform), or the run's own.
        ImageTransformPipeline transforms;
        try
        {
            transforms = ModelChoices.ImageTransformsOf(context) ?? ImageTransformPipeline.FromJson(manifest["image_transforms"]);
        }
        catch (ArgumentException ex)
        {
            throw new UsageException($"image transforms of {manifestPath}: {ex.Message}");
        }

        // The reference's arrays.
        string Array(string file) => Path.Combine(folder, file);
        long[] promptIds = Npy.ReadInt64(Array("input_ids.npy")).Values;
        int[] ids = [.. promptIds.Select(i => (int)i)];
        int[] generated = [.. Npy.ReadInt64(Array("generated_ids.npy")).Values.Select(i => (int)i)];
        var (genTopIds, _) = Npy.ReadInt64(Array("gen_top_ids.npy"));
        var (genTopLogits, _) = Npy.ReadFloat32(Array("gen_top_logits.npy"));
        var (tfTopIds, _) = Npy.ReadInt64(Array("tf_top_ids.npy"));
        var (tfTopLogits, _) = Npy.ReadFloat32(Array("tf_top_logits.npy"));
        var (refPixels, pixelShape) = Npy.ReadFloat32(Array("pixel_values.npy"));
        var (refHidden, hiddenShape) = Npy.ReadFloat32(Array("vision_last_hidden_state.npy"));
        var (refFeatures, featureShape) = Npy.ReadFloat32(Array("image_features.npy"));
        (float[] Values, int[] Shape) fullRows = File.Exists(Array("gen_full_logits.npy")) ? Npy.ReadFloat32(Array("gen_full_logits.npy")) : ([], [0, 0]);
        int steps = Math.Min(generated.Length, context.IntOption("--steps", generated.Length));
        if (steps < 1)
        {
            throw new UsageException("--steps needs a positive number (and the reference at least one generated token).");
        }

        var choice = ModelChoices.Choose(context, name) with { Grayscale = grayscale, VisionOptions = visionOptions.Count > 0 ? visionOptions : null, ImageTransforms = transforms };
        var device = context.Device;
        context.Write($"Reference  {folder}: transformers {manifest["versions"]?["transformers"]}, torch {manifest["versions"]?["torch"]}, "
                      + $"{manifest["dtype"]} (vision {manifest["vision_dtype"]}), logits {manifest["logits_dtype"]}, {manifest["image_processor_class"]}, "
                      + $"grayscale {grayscale}, image transforms {(choice.Transforms.IsEmpty ? "none" : choice.Transforms)}, vision options {visionOptions}, {ids.Length} prompt tokens, {generated.Length} generated");
        context.Write($"Idrak      {choice.Model} on {device}, weights {choice.Weights ?? "as stored"}, KV cache {choice.Kv ?? "float32"}, encoder float32, "
                      + $"matrix products {MixedPrecision.Default}");

        using var noGrad = Autograd.NoGrad();
        using var model = ModelChoices.Load(context, choice);
        var vision = model.Vision ?? throw new UsageException($"{choice.Model} has no vision part: give a vision-language model.");
        var tokenizer = model.Tokenizer ?? throw new InvalidOperationException($"{choice.Model} has no tokenizer (tokenizer.json).");
        int vocabulary = tokenizer.VocabularySize;
        var watch = Stopwatch.StartNew();
        visionOptions.ThrowIfUnknown(vision.Family, vision.VisionOptionKeys);
        using var encoder = vision.CreateEncoder(new VisionEncoderOptions { Device = model.Device, VisionOptions = visionOptions });   // grey through the transforms
        double buildMs = watch.Elapsed.TotalMilliseconds;
        var stages = encoder as IVisionEncoderStages
            ?? throw new UsageException($"The vision family {vision.Family}'s encoder does not show its stages (IVisionEncoderStages): vlm check compares its pixels and its tower's output.");
        var json = new JsonObject
        {
            ["model"] = choice.Model,
            ["device"] = device.ToString(),
            ["weights"] = choice.Weights,
            ["kv"] = choice.Kv ?? "float32",
            ["reference"] = Path.GetFullPath(folder),
            ["reference_dtype"] = manifest["dtype"]?.DeepClone(),
            ["reference_vision_dtype"] = manifest["vision_dtype"]?.DeepClone(),
            ["grayscale"] = grayscale,
            ["image_transforms"] = choice.Transforms.ToString(),
            ["vision_options"] = visionOptions.ToJson(),
            ["steps"] = steps,
            ["tie"] = tie,
        };

        // 1. Pixels: the image read as idrak run reads it.
        context.Write("\nPixels");
        string? imagePath = context.Option("--image")
            ?? ((string?)manifest["image"] is { } named ? Path.IsPathRooted(named) ? named : Path.Combine(folder, named) : null);   // relative: to the folder
        float[]? ownPixels = null;
        float pixelDifference = 0;
        if (imagePath is not null && File.Exists(imagePath))
        {
            var chatImage = ChatImage.FromFile(imagePath);
            var decoded = choice.Transforms.Apply((bool?)manifest["exif_applied"] == false ? ImageCodecs.Decode(chatImage.Data.Span) : ImageInputs.Decode(chatImage));
            using (var own = stages.PixelValues(decoded, visionOptions))
            {
                ownPixels = own.ToArray();
            }

            if (ownPixels.Length != refPixels.Length)
            {
                throw new InvalidOperationException($"Idrak's preprocessing gives {ownPixels.Length} pixel values, the reference [{string.Join(", ", pixelShape)}].");
            }

            var p = Compare(refPixels, ownPixels);
            pixelDifference = p.MaxAbs;
            int differing = refPixels.Where((v, i) => Math.Abs(v - ownPixels[i]) > 1e-5f).Count();
            context.Write(string.Create(CultureInfo.InvariantCulture, $"  {Mark(p.MaxAbs <= 1e-5f)} {imagePath} ({decoded.Width} x {decoded.Height}) -> [{string.Join(", ", pixelShape)}]: max |Δ| {p.MaxAbs:G3}, ")
                          + $"{differing} of {refPixels.Length} values differ by more than 1e-5");
            json["pixels"] = new JsonObject { ["image"] = imagePath, ["max_abs"] = p.MaxAbs, ["differing"] = differing };
        }
        else
        {
            context.Write($"  --   the image ({imagePath ?? "none"}) is not here: the reference's pixels are used (give --image FILE to compare them)");
            json["pixels"] = null;
        }

        // 2. The encoder and the projector from the reference's pixels.
        context.Write("\nVision encoder and projector (from the reference's pixels)");
        watch.Restart();
        float[] hidden, features;
        using (var pixels = Tensor.From(refPixels, pixelShape, model.Device))
        using (var h = stages.Tower(pixels))
        using (var f = stages.Features(pixels))
        {
            hidden = h.ToArray();
            features = f.ToArray();
        }

        double encodeMs = watch.Elapsed.TotalMilliseconds;
        context.Detail(string.Create(CultureInfo.InvariantCulture, $"vision encoder built in {buildMs:F0} ms"));
        // The reference may keep only the first rows of each block's output (--vision-rows): compare those, block by block.
        int hiddenBlocks = hiddenShape.Length == 3 ? hiddenShape[0] : 1, refBlock = refHidden.Length / hiddenBlocks, ownBlock = hidden.Length / hiddenBlocks;
        float[] ownHidden = refBlock == ownBlock ? hidden
            : [.. Enumerable.Range(0, hiddenBlocks).SelectMany(b => hidden.AsSpan(b * ownBlock, Math.Min(refBlock, ownBlock)).ToArray())];
        var hiddenCompare = Compare(refHidden.AsSpan(0, Math.Min(refHidden.Length, ownHidden.Length)), ownHidden.AsSpan(0, Math.Min(refHidden.Length, ownHidden.Length)));
        var featureCompare = Compare(refFeatures, features);
        context.Write(string.Create(CultureInfo.InvariantCulture, $"  {Mark(hiddenCompare.Cosine >= 0.999)} encoder output [{string.Join(", ", hiddenShape)}]: {Describe(hiddenCompare)} ({encodeMs:F0} ms with the projector)"));
        context.Write($"  {Mark(featureCompare.Cosine >= 0.999)} image features [{string.Join(", ", featureShape)}]: {Describe(featureCompare)}");
        json["encoder"] = CompareJson(hiddenCompare);
        json["features"] = CompareJson(featureCompare);

        // The features the decoder reads on Idrak's path: from Idrak's own pixels (as idrak run), the same when they agree.
        float[] idrakFeatures = features;
        if (ownPixels is not null && pixelDifference > 0)
        {
            using var pixels = Tensor.From(ownPixels, pixelShape, model.Device);
            using var f = stages.Features(pixels);
            idrakFeatures = f.ToArray();
            var own = Compare(refFeatures, idrakFeatures);
            context.Write($"  {Mark(own.Cosine >= 0.999)} image features from Idrak's own pixels: {Describe(own)}");
            json["features_own_pixels"] = CompareJson(own);
        }

        // 3. The prompt's ids as Idrak makes them.
        context.Write("\nPrompt");
        bool promptSame = true;
        if (model.ChatTemplate is not null)
        {
            // The image's tokens as many as the reference's features hold (the image is not decoded or encoded here).
            var chat = ModelChoices.CreateChat(model, choice);
            // One block per row of the reference's features [blocks, tokens, width] (an image the family shows as several views).
            var counted = new LayoutOnly([.. Enumerable.Repeat(new ImageTokenLayout(featureShape.Length == 3 ? featureShape[1] : featureShape[0]), featureShape.Length == 3 ? featureShape[0] : 1)],
                vision.Width, model.Device);
            chat = new ChatGenerator(chat.Generator, chat.Template)
            {
                Images = new ChatImages(counted, vision.PromptFormat, vision.Attention) { Decode = _ => new ImageData([0f], 1, 1, 1) },
            };
            var messages = new List<ChatMessage>();
            if ((string?)manifest["system"] is { Length: > 0 } system)
            {
                messages.Add(new ChatMessage("system", system));
            }

            messages.Add(new ChatMessage("user", [ChatImage.FromBytes([0], "image/png"), new ChatText((string?)manifest["prompt"] ?? "")]));
            var own = chat.Generator.Tokenizer.Encode(chat.RenderPrompt(new ChatRequest(messages)));
            promptSame = own.SequenceEqual(ids);
            int at = Enumerable.Range(0, Math.Min(own.Count, ids.Length)).FirstOrDefault(i => own[i] != ids[i], Math.Min(own.Count, ids.Length));
            context.Write($"  {Mark(promptSame)} {own.Count} ids from Idrak's chat template and tokenizer, {ids.Length} in the reference"
                          + (promptSame ? "" : $"; first difference at {at}: reference [{Tokens(tokenizer, ids.Skip(at).Take(4))}], Idrak [{Tokens(tokenizer, own.Skip(at).Take(4))}]"));
            json["prompt"] = new JsonObject { ["same"] = promptSame, ["idrak_tokens"] = own.Count, ["reference_tokens"] = ids.Length, ["first_difference"] = promptSame ? null : at };
        }
        else
        {
            context.Write("  --   no chat template: the reference's ids are used unchecked");
        }

        int imageToken = vision.PromptFormat.ImageToken;
        int afterImage = Math.Max(0, System.Array.LastIndexOf(ids, imageToken) + 1);
        var layout = ModelChoices.CacheLayout(choice);

        // 4. Teacher forcing, 5. greedy.
        Pass Force(string label, float[] featureValues, bool promptRows)
        {
            using var featureBlocks = new Blocks(featureValues, featureShape, vision.Width, model.Device);
            using var decoding = new DecodingContext(model.Device, 1, ids.Length + steps + 1, layout);
            var images = ImagePrefill.Locate(ids, imageToken, featureBlocks.Tensors);
            using var input = Tensor.From([.. ids.Select(i => (float)i)], [1, ids.Length], model.Device);
            using var prefill = model.Network.ForwardCached(input, images, vision.Attention, decoding);
            int promptAgree = 0, promptRowsCount = 0, afterAgree = 0, afterRows = 0;
            float[] last;
            if (promptRows)
            {
                float[] all = prefill.ToArray();
                for (int p = 0; p < ids.Length; p++)
                {
                    int top = ArgMax(all.AsSpan(p * vocabulary, vocabulary));
                    bool same = top == (int)tfTopIds[p * Top];
                    promptRowsCount++;
                    promptAgree += same ? 1 : 0;
                    if (p >= afterImage)
                    {
                        afterRows++;
                        afterAgree += same ? 1 : 0;
                    }
                }

                last = all[^vocabulary..];
            }
            else
            {
                using var row = prefill.Narrow(1, ids.Length - 1, 1);
                last = row.ToArray();
            }

            var list = new List<Step>();
            var rowDifferences = new List<float>();
            for (int i = 0; i < steps; i++)
            {
                list.Add(MakeStep(i, generated[i], genTopIds, genTopLogits, last));
                if (i < fullRows.Shape[0])
                {
                    rowDifferences.Add(Compare(fullRows.Values.AsSpan(i * vocabulary, vocabulary), last).MaxAbs);
                }

                if (i + 1 < steps)
                {
                    using var next = Tensor.From([(float)generated[i]], [1, 1], model.Device);
                    using var logits = model.Network.ForwardCached(next, decoding);
                    last = logits.ToArray();
                }
            }

            return new Pass(label, list, promptAgree, promptRowsCount, afterAgree, afterRows, [.. rowDifferences]);
        }

        watch.Restart();
        var idrakPass = Force("Idrak's features", idrakFeatures, promptRows: true);
        double forcedMs = watch.Elapsed.TotalMilliseconds;
        var referencePass = Force("the reference's features", refFeatures, promptRows: false);
        context.Write(string.Create(CultureInfo.InvariantCulture, $"\nTeacher forcing: transformers' {steps} tokens fed back ({forcedMs:F0} ms)"));
        foreach (var pass in new[] { idrakPass, referencePass })
        {
            Report(context, tokenizer, pass, tie, show);
        }

        context.Write($"  prompt rows (Idrak's features): top-1 agrees at {idrakPass.PromptAgree} of {idrakPass.PromptRows} positions "
                      + $"({idrakPass.AfterImageAgree} of {idrakPass.AfterImageRows} after the image)");
        if (idrakPass.FullRowDifferences.Length > 0)
        {
            context.Write($"  whole logit rows of the first {idrakPass.FullRowDifferences.Length} steps: max |Δ| {string.Join(", ", idrakPass.FullRowDifferences.Select(d => d.ToString("G3", CultureInfo.InvariantCulture)))}");
        }

        // Idrak's own greedy tokens (as idrak run --temperature 0 picks them), from Idrak's features.
        var greedy = new List<int>();
        using (var featureBlocks = new Blocks(idrakFeatures, featureShape, vision.Width, model.Device))
        using (var decoding = new DecodingContext(model.Device, 1, ids.Length + steps + 1, layout))
        using (var input = Tensor.From([.. ids.Select(i => (float)i)], [1, ids.Length], model.Device))
        using (var prefill = model.Network.ForwardCached(input, ImagePrefill.Locate(ids, imageToken, featureBlocks.Tensors), vision.Attention, decoding))
        {
            float[] last;
            using (var row = prefill.Narrow(1, ids.Length - 1, 1))
            {
                last = row.ToArray();
            }

            for (int i = 0; i < steps; i++)
            {
                greedy.Add(ArgMax(last));
                if (i + 1 < steps)
                {
                    using var next = Tensor.From([(float)greedy[^1]], [1, 1], model.Device);
                    using var logits = model.Network.ForwardCached(next, decoding);
                    last = logits.ToArray();
                }
            }
        }

        int diverge = Enumerable.Range(0, steps).FirstOrDefault(i => greedy[i] != generated[i], steps);
        context.Write($"\nGreedy ({steps} tokens, Idrak's features)");
        if (diverge == steps)
        {
            context.Write($"  ok   the same {steps} tokens as transformers");
        }
        else
        {
            context.Write($"  DIFF first divergence at step {diverge} (position {ids.Length + diverge}); the first {diverge} tokens agree");
            context.Write($"       common:      {Shorten(tokenizer.Decode(generated.Take(diverge)), fromEnd: true)}");
            context.Write($"       transformers {Shorten(tokenizer.Decode(generated.Skip(diverge).Take(steps - diverge)))}");
            context.Write($"       Idrak        {Shorten(tokenizer.Decode(greedy.Skip(diverge)))}");
        }

        // The verdict.
        string verdict = Verdict(idrakPass, referencePass, diverge, steps, tie, pixelDifference, featureCompare, promptSame);
        context.Write($"\nVerdict: {verdict}");
        bool realDifference = idrakPass.Steps.Any(s => !s.Agrees && !NearTie(s, tie)) || !promptSame;
        json["teacher_forced"] = PassJson(tokenizer, idrakPass, tie);
        json["teacher_forced_reference_features"] = PassJson(tokenizer, referencePass, tie);
        json["greedy"] = new JsonObject
        {
            ["first_divergence"] = diverge == steps ? null : diverge,
            ["ids"] = new JsonArray([.. greedy.Select(i => (JsonNode)i)]),
            ["text"] = tokenizer.Decode(greedy),
            ["reference_text"] = tokenizer.Decode(generated.Take(steps)),
        };
        json["verdict"] = verdict;
        json["ok"] = !realDifference;
        context.WriteJson(json);
        return realDifference ? ExitCodes.Failed : ExitCodes.Ok;
    }

    private static Step MakeStep(int index, int expected, long[] refIds, float[] refLogits, float[] logits)
    {
        int[] reference = [.. refIds.Skip(index * Top).Take(Top).Select(i => (int)i)];
        float[] referenceLogits = [.. refLogits.Skip(index * Top).Take(Top)];
        int[] top = TopK(logits, Top);
        float worst = 0;
        for (int k = 0; k < Top; k++)
        {
            worst = Math.Max(worst, Math.Abs(logits[reference[k]] - referenceLogits[k]));
        }

        return new Step(index, expected, reference, referenceLogits, top, [.. top.Select(i => logits[i])], worst);
    }

    // A disagreement where either side's top two are within the tie margin and each side's choice is in the other's top 5.
    private static bool NearTie(Step s, float tie) =>
        (s.RefMargin < tie || s.Margin < tie) && s.Ids.Contains(s.Expected) && s.RefIds.Contains(s.Ids[0]);

    private static void Report(CommandContext context, ITokenizer tokenizer, Pass pass, float tie, int show)
    {
        var wrong = pass.Steps.Where(s => !s.Agrees).ToList();
        int ties = wrong.Count(s => NearTie(s, tie));
        float worst = pass.Steps.Max(s => s.TopFiveDifference);
        context.Write($"  {Mark(wrong.Count == 0)} {pass.Name}: top-1 agrees at {pass.Steps.Count - wrong.Count} of {pass.Steps.Count} steps"
                      + (wrong.Count == 0 ? "" : string.Create(CultureInfo.InvariantCulture, $"; {wrong.Count} disagreements ({ties} near-ties, {wrong.Count - ties} beyond --tie {tie}), the first at step {wrong[0].Index}"))
                      + string.Create(CultureInfo.InvariantCulture, $"; max |Δ| over the reference's top-5 logits {worst:G3}"));
        foreach (var s in wrong.Take(show))
        {
            context.Write(string.Create(CultureInfo.InvariantCulture, $"       step {s.Index}: reference {s.Expected} {Quote(tokenizer, s.Expected)} (margin {s.RefMargin:F3}), Idrak {s.Ids[0]} {Quote(tokenizer, s.Ids[0])} ")
                          + string.Create(CultureInfo.InvariantCulture, $"(margin {s.Margin:F3}): {(NearTie(s, tie) ? "near-tie" : "REAL DIFFERENCE")}"));
            context.Write($"         reference top-5: {TopList(tokenizer, s.RefIds, s.RefLogits)}");
            context.Write($"         Idrak     top-5: {TopList(tokenizer, s.Ids, s.Logits)}");
        }

        if (wrong.Count > show)
        {
            context.Write($"       ... {wrong.Count - show} more (--show N prints more; -j lists every one)");
        }
    }

    private static string Verdict(Pass idrak, Pass reference, int diverge, int steps, float tie, float pixelDifference, Comparison features, bool promptSame)
    {
        var parts = new List<string>();
        if (!promptSame)
        {
            parts.Add("the prompt's ids differ (chat template or tokenizer): fix that first.");
        }

        if (pixelDifference > 1e-5f)
        {
            parts.Add(string.Create(CultureInfo.InvariantCulture, $"the pixels differ (max |Δ| {pixelDifference:G3}): the image is read or preprocessed differently."));
        }

        var wrong = idrak.Steps.Where(s => !s.Agrees).ToList();
        var referenceWrong = reference.Steps.Where(s => !s.Agrees).ToList();
        if (wrong.Count == 0)
        {
            parts.Add($"Idrak picks transformers' token at all {steps} steps"
                      + (diverge == steps ? " and its own greedy answer is the same." : $"; its greedy answer still leaves at step {diverge} (cached decoding differs?)."));
        }
        else
        {
            var first = wrong[0];
            string firstKind = NearTie(first, tie)
                ? string.Create(CultureInfo.InvariantCulture, $"a near-tie (margins {first.RefMargin:F3} in transformers, {first.Margin:F3} in Idrak, below {tie}): precision, not a bug")
                : string.Create(CultureInfo.InvariantCulture, $"NOT a near-tie (margins {first.RefMargin:F3} and {first.Margin:F3}): a real difference");
            parts.Add($"the first disagreement, at step {first.Index}, is {firstKind}.");
            bool allTies = wrong.All(s => NearTie(s, tie));
            bool referenceClean = referenceWrong.All(s => NearTie(s, tie));
            if (!allTies && referenceClean)
            {
                parts.Add("With transformers' own image features the decoder agrees (or only near-ties): the difference comes from the image side "
                          + string.Create(CultureInfo.InvariantCulture, $"(features cosine {features.Cosine:F6}, relative {features.Relative:G3}): the encoder's precision or a vision bug."));
            }
            else if (!referenceClean)
            {
                parts.Add("With transformers' own image features the decoder still differs beyond a near-tie: the decoder (or its weight format) differs; "
                          + "compare with float32 weights, or send this report.");
            }
            else
            {
                parts.Add($"Every disagreement ({wrong.Count}) is a near-tie: the two runs differ only in rounding.");
            }
        }

        return string.Join(" ", parts);
    }

    private static JsonObject PassJson(ITokenizer tokenizer, Pass pass, float tie) => new()
    {
        ["features"] = pass.Name,
        ["agree"] = pass.Steps.Count(s => s.Agrees),
        ["steps"] = pass.Steps.Count,
        ["max_top5_logit_difference"] = pass.Steps.Max(s => s.TopFiveDifference),
        ["prompt_rows_agree"] = pass.PromptRows == 0 ? null : pass.PromptAgree,
        ["prompt_rows"] = pass.PromptRows == 0 ? null : pass.PromptRows,
        ["full_row_max_abs"] = new JsonArray([.. pass.FullRowDifferences.Select(d => (JsonNode)d)]),
        ["disagreements"] = new JsonArray([.. pass.Steps.Where(s => !s.Agrees).Select(s => (JsonNode)new JsonObject
        {
            ["step"] = s.Index,
            ["near_tie"] = NearTie(s, tie),
            ["reference"] = TopJson(tokenizer, s.RefIds, s.RefLogits),
            ["idrak"] = TopJson(tokenizer, s.Ids, s.Logits),
            ["reference_margin"] = s.RefMargin,
            ["idrak_margin"] = s.Margin,
        })]),
    };

    private static JsonArray TopJson(ITokenizer tokenizer, int[] ids, float[] logits) =>
        new([.. ids.Select((id, k) => (JsonNode)new JsonObject { ["id"] = id, ["token"] = tokenizer.TokenOf(id), ["logit"] = logits[k] })]);

    private static string TopList(ITokenizer tokenizer, int[] ids, float[] logits) =>
        string.Join(" | ", ids.Select((id, k) => $"{id} {Quote(tokenizer, id)} {logits[k].ToString("F3", CultureInfo.InvariantCulture)}"));

    private static string Quote(ITokenizer tokenizer, int id) => $"'{Escape(tokenizer.TokenOf(id))}'";

    private static string Tokens(ITokenizer tokenizer, IEnumerable<int> ids) => string.Join(", ", ids.Select(id => $"{id} {Quote(tokenizer, id)}"));

    private static string Escape(string text) => text.Replace("\\", "\\\\").Replace("\r", "\\r").Replace("\n", "\\n").Replace("\t", "\\t");

    private static string Shorten(string text, bool fromEnd = false)
    {
        string flat = Escape(text);
        return flat.Length <= 160 ? flat : fromEnd ? "…" + flat[^160..] : flat[..160] + "…";
    }

    private static string Mark(bool ok) => ok ? "ok  " : "DIFF";

    private readonly record struct Comparison(float MaxAbs, float Relative, double Cosine, float MeanAbs);

    private static Comparison Compare(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
    {
        int n = Math.Min(expected.Length, actual.Length);
        float maxAbs = 0, scale = 0;
        double dot = 0, ee = 0, aa = 0, sum = 0;
        for (int i = 0; i < n; i++)
        {
            float d = Math.Abs(expected[i] - actual[i]);
            maxAbs = Math.Max(maxAbs, d);
            scale = Math.Max(scale, Math.Abs(expected[i]));
            dot += (double)expected[i] * actual[i];
            ee += (double)expected[i] * expected[i];
            aa += (double)actual[i] * actual[i];
            sum += d;
        }

        double cosine = ee == 0 && aa == 0 ? 1 : ee == 0 || aa == 0 ? 0 : dot / Math.Sqrt(ee * aa);
        return new Comparison(maxAbs, scale == 0 ? maxAbs : maxAbs / scale, cosine, n == 0 ? 0 : (float)(sum / n));
    }

    private static string Describe(Comparison c) =>
        string.Create(CultureInfo.InvariantCulture, $"max |Δ| {c.MaxAbs:G3}, relative {c.Relative:G3}, mean |Δ| {c.MeanAbs:G3}, cosine {c.Cosine:F6}");

    private static JsonObject CompareJson(Comparison c) => new() { ["max_abs"] = c.MaxAbs, ["relative"] = c.Relative, ["mean_abs"] = c.MeanAbs, ["cosine"] = c.Cosine };

    private static int ArgMax(ReadOnlySpan<float> values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > values[best])
            {
                best = i;
            }
        }

        return best;
    }

    // The k largest values' indices, largest first (ties: the lower index first, as torch.topk and argmax).
    private static int[] TopK(float[] values, int k)
    {
        var best = new List<int>(k + 1);
        for (int i = 0; i < values.Length; i++)
        {
            if (best.Count == k && values[i] <= values[best[^1]])
            {
                continue;
            }

            int at = best.Count;
            while (at > 0 && values[i] > values[best[at - 1]])
            {
                at--;
            }

            best.Insert(at, i);
            if (best.Count > k)
            {
                best.RemoveAt(k);
            }
        }

        return [.. best];
    }

    // An encoder that only gives a layout (the reference's token count), for rendering the prompt without an image.
    // Features [blocks, tokens, width] (or [tokens, width]: one block) as one [1, tokens, width] tensor per block.
    private sealed class Blocks : IDisposable
    {
        public Blocks(float[] values, int[] shape, int width, Device device)
        {
            int count = shape.Length == 3 ? shape[0] : 1, each = values.Length / count;
            Tensors = [.. Enumerable.Range(0, count).Select(b => Tensor.From(values.AsSpan(b * each, each).ToArray(), [1, each / width, width], device))];
        }

        public IReadOnlyList<Tensor> Tensors { get; }

        public void Dispose()
        {
            foreach (var tensor in Tensors)
            {
                tensor.Dispose();
            }
        }
    }

    private sealed class LayoutOnly(IReadOnlyList<ImageTokenLayout> blocks, int width, Device device) : IVisionEncoder
    {
        public int Width => width;

        public Device Device => device;

        public IReadOnlyList<ImageTokenLayout> Blocks(ImageData image, VisionOptions? options = null) => blocks;

        public IReadOnlyList<ImageFeatures> Encode(IReadOnlyList<ImageData> images, VisionOptions? options = null) => throw new InvalidOperationException("not encoded here");

        public void Dispose()
        {
        }
    }
}
