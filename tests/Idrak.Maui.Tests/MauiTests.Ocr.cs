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
using Microsoft.Extensions.DependencyInjection;

using Device = Idrak.Abstraction.Device;

namespace Idrak.Maui.Tests;

// The OCR lab (plan 34, M6) on the CPU: its page over a scripted vision reader.
internal static partial class MauiTests
{
    private const string FakeVisionArchitecture = "FakeVisionForMauiTests";

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
            var info = new VisionReaderInfo(request.Model, FakeVisionArchitecture, "cpu", request.CacheFormat, request.Context, TimeSpan.FromSeconds(1.5), Estimate(request));
            return Task.FromResult(new VisionReaderLease(reader, info, load ? TimeSpan.FromSeconds(1.5) : TimeSpan.Zero));
        }
    }

    private static OcrViewModel OcrLab(IVisionReaderService readers, IModelStore store, OcrLabOptions? defaults = null, IdrakMauiOptions? options = null,
        Microsoft.Maui.Dispatching.IDispatcher? dispatcher = null, TimeProvider? time = null, MemorySettingsStore? settings = null, FakeShare? share = null,
        FakeClipboard? clipboard = null, FakePermissions? permissions = null, FakePicker? picker = null) =>
        new(store, readers, permissions ?? new FakePermissions(true), share ?? new FakeShare(), options ?? new IdrakMauiOptions { Device = "cpu" }, settings ?? new MemorySettingsStore(),
            defaults, dispatcher, clipboard ?? new FakeClipboard(), picker ?? new FakePicker(), time);

    private static void OcrPageOpens()
    {
        using var services = new ServiceCollection().BuildServiceProvider();
        var feature = new OcrFeature();
        var page = feature.CreatePage(services, feature.CreateViewModel(services));
        Check(page is OcrPage { BindingContext: OcrViewModel { OpenModels: not null } } && page.Title == "Read text", $"page {page.GetType().Name}, title {page.Title}");
        Check(feature is { Name: MobileFeatures.Ocr, Order: 60, Needs: PhoneNeeds.Camera | PhoneNeeds.Model } && ((IMobileFeature)feature).State == FeatureState.Ready
              && MobileFeatures.Place(feature) == 60 && feature.ModelFamilies.Count == 0, "the library's OCR feature");

        // The camera refused: said, nothing taken.
        var refused = new FakePermissions(false);
        var picker = new FakePicker();
        var lab = OcrLab(new OneReader(new ScriptedReader(1, "x", Stats(1, 1))), new ListedStore(), permissions: refused, picker: picker);
        Check(!lab.TakePhotoAsync(camera: true).GetAwaiter().GetResult() && lab.IsPermissionRefused && picker.Captures == 0, lab.Error);
    }
}
