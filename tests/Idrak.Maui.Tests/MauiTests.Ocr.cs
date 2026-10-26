// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Diagnostics;
using Idrak.Imaging;
using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;
using Idrak.Maui.Views;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Text;
using Idrak.Text.Generation;
using Idrak.Vision.Plugins.Gemma;
using Microsoft.Extensions.DependencyInjection;

using Device = Idrak.Abstraction.Device;

namespace Idrak.Maui.Tests;

// The OCR lab (plan 34, M6) on the CPU: a scripted vision reader streaming from a fake vision family's model, pages made
// here with the library's encoders; the repository's tiny Gemma 3 measured, refused under a memory limit and read end
// to end through the Gemma 3 vision plug-in (registered as the app registers it).
internal static partial class MauiTests
{
    private const string FakeVisionArchitecture = "FakeVisionForMauiTests";

    // A vision family that is only a name: the lab lists its models; the tests keep their reader in a fake service.
    private sealed class FakeVisionFamily : IVisionFamily
    {
        public string Name => FakeVisionArchitecture;

        public IPretrainedVision? Read(VisionCheckpoint checkpoint) => throw new NotSupportedException("the tests read with a scripted reader");
    }

    // A reader answering with `pieces` pieces of `piece`, each counting its token, then a final piece with the statistics;
    // moves the test's clock by `step` a piece; Gate holds it after its first piece until the read is stopped.
    private sealed class ScriptedReader(int pieces, string piece, GenerationStats stats, ManualTime? time = null, TimeSpan step = default) : IVisionReader
    {
        public List<(ImageReadRequest Request, ImageData Prepared)> Kept { get; } = [];

        public List<ImageReadRequest> Reads { get; } = [];

        public TaskCompletionSource FirstPiece { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);

        public bool Gate { get; init; }

        public void Keep(ImageReadRequest request, ImageData prepared) => Kept.Add((request, prepared));

        public IEnumerable<ChatChunk> Stream(ImageReadRequest request, CancellationToken cancellationToken = default)
        {
            Reads.Add(request);
            for (int i = 0; i < pieces; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                time?.Advance(step);
                yield return new ChatChunk(new ChatDelta(piece, "", [])) { Generated = i + 1 };
                FirstPiece.TrySetResult();
                if (Gate)
                {
                    cancellationToken.WaitHandle.WaitOne(TimeSpan.FromSeconds(30));
                }
            }

            yield return new ChatChunk(new ChatDelta("", "", []), true, "stop", new ChatMessage("assistant", string.Concat(Enumerable.Repeat(piece, pieces))), stats)
            {
                Generated = pieces,
            };
        }
    }

    // A reader service handing out one reader, counting the leases and saying whether each loaded it.
    private sealed class OneReader(IVisionReader reader) : IVisionReaderService
    {
        public List<VisionLoadRequest> Requests { get; } = [];

        public bool Unloaded { get; set; } = true;

        public ChatMemoryEstimate Estimate(VisionLoadRequest request) => new(request.Model, "cpu", 1000, 100, 1 << 30, "the memory cpu reports free now");

        public Task<VisionReaderLease> AcquireAsync(VisionLoadRequest request, CancellationToken cancellationToken = default)
        {
            lock (Requests)
            {
                Requests.Add(request);
            }

            bool load = Unloaded;
            Unloaded = false;
            var info = new VisionReaderInfo(request.Model, FakeVisionArchitecture, "cpu", request.Weights ?? "float32", request.CacheFormat, request.Context,
                TimeSpan.FromSeconds(1.5), Estimate(request));
            return Task.FromResult(new VisionReaderLease(reader, info, load ? TimeSpan.FromSeconds(1.5) : TimeSpan.Zero));
        }
    }

    // A model folder whose config.json names `architecture`.
    private static string ModelFolder(string name, string architecture)
    {
        string folder = Temp(name);
        File.WriteAllText(Path.Combine(folder, "config.json"), $$"""{"architectures": ["{{architecture}}"]}""");
        return folder;
    }

    private static string TinyGemma3() => Path.Combine(RepositoryRoot(), "tests", "Idrak.Tests", "data", "vlm", "tiny-gemma3");

    // The library tests' page: 12 x 20 in colour, a gradient.
    private static ChatImage GradientPage()
    {
        var pixels = new float[3 * 12 * 20];
        for (int i = 0; i < pixels.Length; i++)
        {
            pixels[i] = (i % 97) / 96f;
        }

        return Png(new ImageData(pixels, 3, 12, 20));
    }

    private static OcrViewModel OcrLab(IVisionReaderService readers, IModelStore store, OcrLabOptions? defaults = null, IdrakMauiOptions? options = null,
        Microsoft.Maui.Dispatching.IDispatcher? dispatcher = null, TimeProvider? time = null, MemorySettingsStore? settings = null, FakeShare? share = null,
        FakeClipboard? clipboard = null, FakePermissions? permissions = null, FakePicker? picker = null) =>
        new(store, readers, permissions ?? new FakePermissions(true), share ?? new FakeShare(), options ?? new IdrakMauiOptions { Device = "cpu" }, settings ?? new MemorySettingsStore(),
            defaults, dispatcher, clipboard ?? new FakeClipboard(), picker ?? new FakePicker(), time);

    private static void OcrSettingsMapped()
    {
        // Nothing set: every value is the library's own.
        var page = GradientPage();
        var empty = new OcrLabOptions();
        var request = empty.ToRequest(page);
        var library = new ImageReadRequest(page, "x");
        Check(OcrSettings.LibraryPrompt == new SyntheticPageOptions { Style = null! }.Prompt && request.Prompt == OcrSettings.LibraryPrompt && request.System is null
              && request.MaxTokens == library.MaxTokens && request.RepeatStop == library.RepeatStop && request.Transforms is { IsEmpty: true } && request.Image == page,
            $"request: '{request.Prompt}', {request.MaxTokens}, {request.RepeatStop}");
        var load = empty.ToLoad("lab/vlm", "/models/vlm", "cpu");
        Check(load is { Weights: null, Device: "cpu", Model: "lab/vlm" } && load.Context == new GenerationOptions().NumCtx && load.CacheFormat == new ImageReaderOptions().CacheFormat,
            $"load {load}");
        Check(OcrSettings.Definitions.Single(s => s.Key == "max_tokens").Placeholder == "2048" && OcrSettings.Definitions.Single(s => s.Key == "prompt").Placeholder == OcrSettings.LibraryPrompt,
            "the library's defaults shown");

        // Each setting to its option.
        var texts = new Dictionary<string, string>
        {
            ["prompt"] = "Extract details to JSON.", ["system"] = "You read Arabic.", ["pipeline"] = "grayscale, max_width=1024, contrast=1.5", ["max_tokens"] = "512",
            ["repeat_stop"] = "4", ["weights"] = "INT8", ["cache_format"] = "Int8", ["context"] = "4096", ["preview_px"] = "600",
        };
        Check(texts.Keys.Order().SequenceEqual(OcrSettings.Definitions.Select(s => s.Key).Order()), "a value for every setting");
        var set = new OcrLabOptions { Model = "lab/vlm@abc" };
        foreach (var setting in OcrSettings.Definitions)
        {
            setting.Set(set, texts[setting.Key]);
        }

        var mapped = set.ToRequest(page);
        Check(mapped is { Prompt: "Extract details to JSON.", System: "You read Arabic.", MaxTokens: 512, RepeatStop: 4 } && mapped.Transforms!.ToString() == "grayscale,max_width=1024,contrast=1.5",
            $"mapped {mapped.Prompt}, {mapped.Transforms}");
        Check(set.ToLoad("m", "p", "cpu") is { Weights: "int8", CacheFormat: KeyValueFormat.Int8, Context: 4096 } && set.PreviewPixels == 600, $"load {set.ToLoad("m", "p", "cpu")}");

        // Kept and read back; the app's defaults under the saved values; a bad saved value skipped by name.
        var store = new MemorySettingsStore();
        OcrSettings.Save(store, set);
        Check(store.Keys.All(k => k.StartsWith(OcrSettings.Prefix, StringComparison.Ordinal)), string.Join(", ", store.Keys));
        var read = OcrSettings.Load(store, out var skipped);
        Check(skipped.Count == 0 && read.Model == "lab/vlm@abc" && OcrSettings.Definitions.All(s => s.Get(read) == s.Get(set)), string.Join("; ", skipped));
        store.Set(OcrSettings.Prefix + "weights", null);
        store.Set(OcrSettings.Prefix + "context", "long");
        var over = OcrSettings.Load(store, out skipped, new OcrLabOptions { Weights = "int4" });
        Check(over.Weights == "int4" && over.MaxTokens == 512 && skipped.Count == 1 && skipped[0].StartsWith("Context", StringComparison.Ordinal), string.Join("; ", skipped));

        foreach (var (key, value, says) in new[]
                 {
                     ("pipeline", "grayscale,emboss", "Unknown image transform 'emboss'"), ("weights", "int9", "int8"), ("cache_format", "int3", "float32"),
                     ("max_tokens", "0", "1 or more"), ("context", "8", "16 or more"), ("repeat_stop", "-1", "0 or more"),
                 })
        {
            try
            {
                OcrSettings.Definitions.Single(s => s.Key == key).Set(new OcrLabOptions(), value);
                Check(false, $"{key}={value} was taken");
            }
            catch (FormatException e)
            {
                Check(e.Message.Contains(says, StringComparison.Ordinal), $"{key}={value}: {e.Message}");
            }
        }
    }

    private static void OcrStreams()
    {
        string cache = Temp("ocr-cache");
        VisionFamilies.Register(new FakeVisionFamily());
        var probe = new CountingTransform("maui-ocr-probe");
        ImageTransforms.Register(probe);
        ComputeResources.CacheFolder = cache;
        try
        {
            WithCountingCodecs((png, _) =>
            {
                string vision = ModelFolder("ocr-vision", FakeVisionArchitecture), text = ModelFolder("ocr-text", "Qwen3ForCausalLM");
                var store = new ListedStore(Stored(vision, "lab/vision"), Stored(text, "lab/text"), Stored(vision, "lab/unfinished") with { Complete = false });
                var reader = new ScriptedReader(1000, "ab", Stats(1000, 20));
                var readers = new OneReader(reader);
                var dispatcher = new QueuedDispatcher();
                var settings = new MemorySettingsStore();
                var share = new FakeShare();
                var clipboard = new FakeClipboard();
                var lab = OcrLab(readers, store, new OcrLabOptions { PreviewPixels = 10 }, dispatcher: dispatcher, time: new ManualTime(), settings: settings, share: share,
                    clipboard: clipboard);

                // The readers: the store's complete models a registered vision family reads; the first is picked and loaded.
                lab.RefreshAsync().GetAwaiter().GetResult();
                Check(lab.VisionModels.Count == 1 && lab.VisionModels[0] is { Title: "lab/vision", Architecture: FakeVisionArchitecture } && lab.ModelsText.StartsWith("1 vision", StringComparison.Ordinal),
                    $"{string.Join(", ", lab.VisionModels)}; {lab.ModelsText}");
                Check(lab.LoadModelAsync().GetAwaiter().GetResult() && lab.ModelStatus.Contains("lab/vision (FakeVisionForMauiTests) on cpu", StringComparison.Ordinal)
                      && lab.ModelStatus.Contains("loaded in 1.5 s", StringComparison.Ordinal) && lab.ModelStatus.Contains("Measured before loading", StringComparison.Ordinal)
                      && settings.Get(OcrSettings.Prefix + "model") == lab.SelectedModel!.Key, lab.ModelStatus);

                // The photo decoded once and prepared once; the page shown as the model sees it.
                lab.Pipeline = "grayscale,maui-ocr-probe";
                int before = png.Decodes;
                Check(lab.OpenAsync(Png(TestPicture(40, 30)), "page.png").GetAwaiter().GetResult(), lab.Error);
                Check(png.Decodes == before + 1 && probe.Runs == 1 && lab.Prepared is { Channels: 1, Width: 40 } && lab.HasPage && lab.PageText.Contains("40 x 30, grey", StringComparison.Ordinal),
                    $"{png.Decodes - before} decodes, {probe.Runs} runs; {lab.PageText}");
                Check(settings.Get(OcrSettings.Prefix + "pipeline") == "grayscale,maui-ocr-probe", "the pipeline kept");

                // Read: the page handed over as prepared, the request the settings' (the library's instruction), the
                // text in two showings for a thousand pieces at one time, the rate, cold.
                Check(lab.ReadAsync().GetAwaiter().GetResult(), $"not read: {lab.Error}");
                Check(lab.UiUpdates == 2 && dispatcher.Dispatched <= 3, $"{lab.UiUpdates} showings and {dispatcher.Dispatched} dispatches for 1,000 pieces");
                dispatcher.Run();
                var (asked, handed) = reader.Kept.Single();
                Check(ReferenceEquals(handed, lab.Prepared) && asked.Prompt == OcrSettings.LibraryPrompt && asked.Transforms!.ToString() == "grayscale,maui-ocr-probe"
                      && reader.Reads.Single() == asked, $"the reader was given '{asked.Prompt}', {asked.Transforms}");
                Check(png.Decodes == before + 1 && probe.Runs == 1, $"reading decoded {png.Decodes - before - 1} more times, transformed {probe.Runs - 1} more");
                Check(lab.Text == string.Concat(Enumerable.Repeat("ab", 1000)) && lab.LiveText.StartsWith("1,000 tokens · 50.0 tokens/s · prompt 5 tokens", StringComparison.Ordinal)
                      && lab.LiveText.EndsWith("cold", StringComparison.Ordinal) && lab.HasCold && !lab.HasWarm, $"'{lab.LiveText}', {lab.Text.Length} characters");

                // A second read is warm; the page is handed over again, nothing decoded or transformed again.
                Check(lab.ReadAsync().GetAwaiter().GetResult(), lab.Error);
                dispatcher.Run();
                Check(lab.LiveText.EndsWith("warm", StringComparison.Ordinal) && lab.WarmText.StartsWith("Warm: 50.0 tokens/s last, 50.0 mean over 1 read", StringComparison.Ordinal)
                      && reader.Kept.Count == 2 && png.Decodes == before + 1 && probe.Runs == 1, $"'{lab.WarmText}'");

                // Copied and shared.
                Check(lab.CopyAsync().GetAwaiter().GetResult() && clipboard.Text == lab.Text, "copied");
                Check(lab.ShareAsync().GetAwaiter().GetResult() && share.Shared is { } shared && shared.StartsWith(Path.Combine(cache, "ocr"), StringComparison.Ordinal)
                      && File.ReadAllText(shared).TrimEnd() == lab.Text, $"shared {share.Shared}");

                // Another pipeline prepares again from the decoded photo (not decoded again); the same one does nothing.
                lab.Pipeline = "maui-ocr-probe";
                Check(lab.PrepareAsync().GetAwaiter().GetResult() && probe.Runs == 2 && lab.Prepared is { Channels: 3 } && png.Decodes == before + 1, $"{probe.Runs} runs");
                Check(lab.PrepareAsync().GetAwaiter().GetResult() && probe.Runs == 2, "prepared twice for one pipeline");
                lab.Pipeline = "grayscale,emboss";
                Check(!lab.PrepareAsync().GetAwaiter().GetResult() && lab.Error.Contains("Unknown image transform 'emboss'", StringComparison.Ordinal), lab.Error);
            });
        }
        finally
        {
            ImageTransforms.Unregister(probe.Name);
            VisionFamilies.Unregister(FakeVisionArchitecture);
            ComputeResources.CacheFolder = null;
        }
    }

    private static void OcrBatched()
    {
        // 1,000 pieces a millisecond apart (a second in all) with a 250 ms interval: a showing at the first piece, about one a
        // quarter second after, and the last; the text is every piece whatever the showings.
        VisionFamilies.Register(new FakeVisionFamily());
        try
        {
            var time = new ManualTime();
            var reader = new ScriptedReader(1000, "x", Stats(1000, 1), time, TimeSpan.FromMilliseconds(1));
            var dispatcher = new InlineDispatcher();
            var lab = OcrLab(new OneReader(reader), new ListedStore(Stored(ModelFolder("ocr-batched", FakeVisionArchitecture), "lab/vision")),
                options: new IdrakMauiOptions { Device = "cpu", ProgressInterval = TimeSpan.FromMilliseconds(250) }, dispatcher: dispatcher, time: time);
            lab.RefreshAsync().GetAwaiter().GetResult();
            Check(lab.OpenAsync(Png(TestPicture(16, 16)), "page.png").GetAwaiter().GetResult() && lab.ReadAsync().GetAwaiter().GetResult(), lab.Error);
            Check(lab.Text.Length == 1000 && lab.UiUpdates is >= 4 and <= 7 && dispatcher.Dispatched <= lab.UiUpdates + 2,
                $"{lab.UiUpdates} showings, {dispatcher.Dispatched} dispatches for 1,000 pieces");

            // Stop mid-read keeps what came.
            var gated = new ScriptedReader(10, "partial", Stats(10, 1)) { Gate = true };
            var stopping = OcrLab(new OneReader(gated), new ListedStore(Stored(ModelFolder("ocr-stop", FakeVisionArchitecture), "lab/vision")), dispatcher: new InlineDispatcher());
            stopping.RefreshAsync().GetAwaiter().GetResult();
            stopping.OpenAsync(Png(TestPicture(16, 16)), "page.png").GetAwaiter().GetResult();
            var reading = stopping.ReadAsync();
            Check(gated.FirstPiece.Task.Wait(TimeSpan.FromSeconds(30)), "the first piece never came");
            Check(stopping.IsReading && !stopping.IsIdle, "reading: Stop only");
            stopping.Stop();
            Check(reading.Wait(TimeSpan.FromSeconds(30)) && reading.Result, $"stopping did not end the read: {stopping.Error}");
            Check(!stopping.IsReading && stopping.Text == "partial" && stopping.LiveText.EndsWith("stopped", StringComparison.Ordinal) && stopping.Status.StartsWith("Stopped", StringComparison.Ordinal),
                $"'{stopping.Text}', '{stopping.LiveText}', '{stopping.Status}'");
        }
        finally
        {
            VisionFamilies.Unregister(FakeVisionArchitecture);
        }
    }

    private static void OcrRefused()
    {
        Gemma3VisionPlugin.Register();                                          // the app's family, as MauiProgram registers it
        string folder = TinyGemma3();
        long? limit = ComputeResources.CpuMemoryLimit;
        using var models = new ModelService(new ManualMemoryPressure(), new IdrakMauiOptions());
        var service = new VisionReaderService(models);
        try
        {
            // Measured: the checkpoint's every value in float32 (the decoder's and the vision part's), the cache by the context.
            var config = (JsonObject)JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "config.json")))!;
            var spec = PretrainedArchitectures.Get(Gemma3VisionFamily.Architecture).Spec(config, []);
            long total = 0;
            using (var tensors = CheckpointFormats.For(folder).Open(folder))
            {
                foreach (string name in tensors.Names)
                {
                    total += tensors.ShapeOf(name).Aggregate(1L, (a, b) => a * b);
                }
            }

            var request = new VisionLoadRequest("lab/tiny-gemma3", folder, "cpu", null, KeyValueFormat.Float32, 128);
            var estimate = service.Estimate(request);
            Check(total > spec.ParameterCount && estimate.WeightBytes == total * sizeof(float)
                  && estimate.CacheBytes == ModelMemory.KeyValueCacheBytes(spec, 128, KeyValueLayouts.For(KeyValueFormat.Float32)) && estimate.Fits,
                $"estimate {estimate}: {estimate.WeightBytes} bytes for {total} values ({spec.ParameterCount} the decoder's)");
            var packed = service.Estimate(request with { Weights = "int8", CacheFormat = KeyValueFormat.Int8 });
            Check(packed.WeightBytes < estimate.WeightBytes && packed.CacheBytes < estimate.CacheBytes, $"int8: {packed}");

            // A limit below the need refuses before loading, naming it; the page shows the refusal.
            ComputeResources.CpuMemoryLimit = 4096;
            var refused = Throws<ModelDoesNotFitException>(() => service.AcquireAsync(request), "a reader above the limit");
            Check(refused.Message.Contains("does not fit", StringComparison.Ordinal) && refused.Message.Contains("CPU memory limit of 4 KB", StringComparison.Ordinal)
                  && models.Loaded.Count == 0, refused.Message);
            var page = OcrLab(service, new ListedStore(Stored(folder, "lab/tiny-gemma3")));
            page.RefreshAsync().GetAwaiter().GetResult();
            Check(page.VisionModels.Single().Architecture == Gemma3VisionFamily.Architecture, page.ModelsText);
            Check(!page.LoadModelAsync().GetAwaiter().GetResult() && page.Error.Contains("does not fit", StringComparison.Ordinal)
                  && page.Error.Contains("CPU memory limit", StringComparison.Ordinal) && page.Model is null, page.Error);
            ComputeResources.CpuMemoryLimit = limit;

            // Without its vision family: not listed (the families registered named), and refused by the service naming the registry.
            Gemma3VisionPlugin.Unregister();
            page.RefreshAsync().GetAwaiter().GetResult();
            Check(page.HasNoVisionModels && (VisionFamilies.Names.Count == 0 ? page.ModelsText.Contains("No vision family is registered", StringComparison.Ordinal)
                      : page.ModelsText.Contains("among the 1 other model", StringComparison.Ordinal)), page.ModelsText);
            bool named = false;
            try
            {
                service.Estimate(request);
            }
            catch (NotSupportedException e)
            {
                named = e.Message.Contains("Gemma3ForConditionalGeneration", StringComparison.Ordinal) && e.Message.Contains("not registered", StringComparison.Ordinal);
            }

            Check(named, "an unregistered vision family was not refused by name");

            // A text model is not a reader.
            var texts = OcrLab(service, new ListedStore(Stored(ModelFolder("ocr-text-only", "Qwen3ForCausalLM"), "lab/text")));
            texts.RefreshAsync().GetAwaiter().GetResult();
            Check(texts.HasNoVisionModels, "a text model listed as a reader");
        }
        finally
        {
            ComputeResources.CpuMemoryLimit = limit;
            Gemma3VisionPlugin.Register();
        }
    }

    private static void OcrEndToEnd()
    {
        Gemma3VisionPlugin.Register();
        string folder = TinyGemma3();
        const string Pipeline = "grayscale,max_width=8";
        var page = GradientPage();

        // The library's own read of the page with the same transforms and instruction (what the CLI's reader runs).
        string expected;
        int promptTokens;
        using (var model = PretrainedModel.Load(folder, new PretrainedOptions { Device = Device.Cpu }))
        using (var reader = new ImageReader(model, new ImageReaderOptions { ContextLength = 256 }))
        {
            var reading = reader.Read(new ImageReadRequest(page, OcrSettings.LibraryPrompt) { MaxTokens = 6, Transforms = ImageTransformPipeline.Parse(Pipeline) });
            (expected, promptTokens) = (reading.Text, reading.Stats!.PromptTokens);
        }

        WithCountingCodecs((png, _) =>
        {
            var pressure = new ManualMemoryPressure();
            var options = new IdrakMauiOptions { Device = "cpu", ModelKeepAlive = null };
            using var models = new ModelService(pressure, options);
            var lab = OcrLab(new VisionReaderService(models), new ListedStore(Stored(folder, "lab/tiny-gemma3")),
                new OcrLabOptions { MaxTokens = 6, Context = 256, Pipeline = Pipeline }, options);
            lab.RefreshAsync().GetAwaiter().GetResult();
            Check(lab.LoadModelAsync().GetAwaiter().GetResult() && lab.Model is { Device: "cpu", Weights: "float32", Context: 256 } && models.Loaded.Single().Host == VisionReaderService.HostName,
                $"{lab.Error}; {lab.ModelStatus}");
            int before = png.Decodes;
            Check(lab.OpenAsync(page, "gradient.png").GetAwaiter().GetResult() && lab.Prepared is { Channels: 1, Width: 8 }, lab.Error);

            Check(lab.ReadAsync().GetAwaiter().GetResult(), $"not read: {lab.Error}");
            Check(lab.Text == expected && lab.LiveText.Contains($"prompt {promptTokens} tokens", StringComparison.Ordinal) && lab.LiveText.Contains(" · cold", StringComparison.Ordinal)
                  && png.Decodes == before + 1,
                $"the lab read '{lab.Text}' ({lab.LiveText}), the library '{expected}' after {promptTokens} prompt tokens; {png.Decodes - before} decodes");
            Check(lab.ReadAsync().GetAwaiter().GetResult() && lab.Text == expected && lab.LiveText.Contains(" · warm", StringComparison.Ordinal) && png.Decodes == before + 1,
                $"warm: '{lab.Text}' ({lab.LiveText})");

            // Memory pressure takes the idle reader: the next read loads it again and is cold again, with the same answer.
            pressure.Raise(MemoryPressureLevel.Critical, "onLowMemory");
            Check(models.Loaded.Count == 0, "the idle reader stayed loaded under pressure");
            Check(lab.ReadAsync().GetAwaiter().GetResult() && lab.Text == expected && lab.LiveText.Contains(" · cold", StringComparison.Ordinal) && models.Loaded.Count == 1,
                $"after the pressure: '{lab.Text}' ({lab.LiveText}) {lab.Error}");
            Console.WriteLine($"    the tiny Gemma 3 through the lab: '{expected}'");
        });
    }

    private static void OcrPageOpens()
    {
        using var services = new ServiceCollection().BuildServiceProvider();
        var page = MobileFeatures.Open(MobileFeatures.Ocr, services);
        Check(page is OcrPage { BindingContext: OcrViewModel { OpenModels: not null } } && page.Title == "OCR", $"page {page.GetType().Name}, title {page.Title}");
        Check(MobileFeatures.Origin(MobileFeatures.Ocr) == Overrides.Library
              && MobileFeatures.Get(MobileFeatures.Ocr) is OcrFeature { Order: 60, Needs: PhoneNeeds.Camera | PhoneNeeds.Model } feature && ((IMobileFeature)feature).State == FeatureState.Ready
              && MobileFeatures.Place(feature) == 60 && feature.ModelFamilies.Count == 0, "the library's OCR feature");

        // No reader in the store: said, with what to do; the camera refused: said, nothing taken.
        var refused = new FakePermissions(false);
        var picker = new FakePicker();
        var lab = OcrLab(new OneReader(new ScriptedReader(1, "x", Stats(1, 1))), new ListedStore(), permissions: refused, picker: picker);
        lab.RefreshAsync().GetAwaiter().GetResult();
        Check(lab.HasNoVisionModels && !lab.ReadAsync().GetAwaiter().GetResult() && lab.Error.Contains("pull one in Models", StringComparison.Ordinal), lab.Error);
        Check(!lab.TakePhotoAsync(camera: true).GetAwaiter().GetResult() && lab.IsPermissionRefused && picker.Captures == 0, lab.Error);
        Check(!lab.OpenAsync(ChatImage.FromBytes([0, 0, 0, 24, (byte)'f', (byte)'t', (byte)'y', (byte)'p', (byte)'h', (byte)'e', (byte)'i', (byte)'c', 0, 0]), "IMG.HEIC").GetAwaiter().GetResult()
              && lab.Error.Contains("HEIC", StringComparison.Ordinal) && lab.HasNoPage, lab.Error);
    }
}
