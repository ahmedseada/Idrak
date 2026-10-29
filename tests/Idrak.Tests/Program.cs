// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Self-contained test runner (no test framework dependency). Runs on the CPU and every CUDA GPU;
//   IDRAK_DEVICES=cpu,cuda:0,vulkan:1 …   only these devices (… -- --list-devices shows them all, as a table)
//   dotnet run --project tests/Idrak.Tests                  run every test on every available device
//   dotnet run --project tests/Idrak.Tests -- --dump-ptx f  write the generated CUDA kernels to f
//   … -- --dump-spirv folder                                                     write the generated Vulkan kernels (.spv) and their contracts to folder
//   … -- --bench-cpu                                                             time the CPU decoding kernels (bfloat16 products, cached attention, sampling)
//   … -- --bench-gemm / --bench-gemv / --bench-fp8                               time large products / decoding-sized packed products
//   … -- --bench-vulkan [sections]                                               time every Vulkan device (or those IDRAK_DEVICES names): dispatches, copies, products, attention, decoders
//   … -- --bench-hip [launch|copies|fill]                                        time every HIP device (or those IDRAK_DEVICES names): GCN launch cost (code object load cold and warm, tiny and large fill_f32), copies, fill bandwidth; see BenchHip.cs
//   … -- --bench-text [text] [tokenizers] [trainers]                             time the text paths (default all): text: tokenizer, chat template, parsing, chunking, BM25 on the README; tokenizers: each fixture tokenizer's load, encodes and batches on the fixture corpus (IDRAK_BENCH_TEXT_CHARACTERS, _SECONDS, _LOADS, _WORKERS, _TOKENIZERS, _REFERENCE); trainers: bpe, unigram, wordpiece (IDRAK_BENCH_TEXT_TRAIN_FILES, _TRAIN_MB, _TRAIN_VOCAB, _TRAINERS, _WORKERS); see BenchText.cs
//   … -- --bench-window                                                          time windowed and soft-capped attention, kernels against the composed path (IDRAK_DEVICES, default every GPU)
//   … -- --bench-spans [spans|composed|fastest] [train]                          time a 4,096-token bidirectional attention pass: tiled, composed, the measured choice (default all); train: forward and backward (IDRAK_DEVICES, default the CPU)
//   … -- --bench-conv                                                            time convolutions (forward, both gradients) per path: host fallback, composed, implicit tiles, depthwise, the measured choice (IDRAK_DEVICES, default every GPU)
//   … -- --bench-vision [bf16]                                                   time Gemma 3 4B's image encoder and projector (random weights, one 896 x 896 image; IDRAK_DEVICES, default the CPU); bf16: packed bfloat16 weights as the checkpoint stores them, float32 and bfloat16 products, batches of 1 and 5
//   … -- --bench-image                                                         time the CPU side of reading a page (decoders, every image transform, the preprocessor, the preview PNG): cold, warm, bytes, output hashes
//   … -- --bench-dispatch                                                        time a million tiny operations through the dispatcher and straight to the CPU's kernel
//   … -- --bench-offload                                                         time training steps with weights or optimizer state in system memory
//   IDRAK_FILTER=retrieval dotnet run --project tests/Idrak.Tests   only tests whose name contains the text ("a|b": any of several)
//   … -- --list                                    the names of the tests IDRAK_FILTER keeps, one a line, in the order they run (nothing run)
//   IDRAK_TIMEOUT=60 …    a test still running after this many seconds (default 300) is reported as HANG and the run stops
//   … -- --progress (IDRAK_TEST_PROGRESS=1)        the library's progress line on the error output: a bar, the share done, tests a second, the
//                                                   time left, passed and failed so far and the test running (in place on a terminal, a line a
//                                                   minute when redirected; IDRAK_PROGRESS picks the reporter)
//   … -- --failures-only (IDRAK_TEST_FAILURES_ONLY=1)  only the device headers, the FAIL, HANG and SKIP lines and the summary on the standard output
//                                                   (the tests' own lines dropped)
//   a test whose data is not here (Tests.Skip: the owner's references, written by a tool's run) is a SKIP line naming the run
//   that writes it, counted apart from passed and failed ("…, 2 skipped") and not written to the failures file
//   … -- --failures-file test-output/fails.txt (IDRAK_TEST_FAILURES_FILE)  the runner also writes those lines to the file as they
//                                                   happen (its folder made; test-output/ is where runs write, CLAUDE.md):
//                                                   … -- --progress --failures-only --failures-file test-output/fails.txt
//   IDRAK_TEST_REPOSITORY=folder (default: the checkout above this program)  where the tests read the repository's files (api/,
//                                                   plans/, src/); IDRAK_TEST_DATA=folder (default: its tests/Idrak.Tests/data) the test data.
//                                                   Where there is no repository, a test that needs it is a NOT RUN line ("not run here: needs the
//                                                   repository (what)"), counted apart ("…, 3 not run here") and written to the failures file
//   The groups any process can run (an app's too) are tests/Idrak.Tests.Library's, run by its TestRunner; this program adds
//   the groups that need the desktop (Idrak.Cli, ASP.NET Core, the native ONNX Runtime, a child process or an outside program).

using Idrak.Gpu.Cuda;
using Idrak.Tests.Runner;

Idrak.Abstraction.Devices.DeviceProviders.Register(new Tests.MinimalProvider());   // only by name: IDRAK_DEVICES=cpu,minimal

// The text model families (Llama, Qwen, Mistral, Gemma, BERT, T5, …) are Idrak.Text.Plugins' (plan 26, P7); the
// library registers none. The runner registers them at startup, as an app (and the idrak tool) does, so every test and
// child process (the cluster ranks) finds them; a test of a family's registration takes it away and puts it back.
TestSuite.Prepare();

// The idrak tool's servers (serve, ui, the cluster dashboard: ASP.NET Core), registered as its Program does, since the tests
// run its commands in this process (CommandLine.Run).
Idrak.Cli.Commands.Serve.CommandServers.Register(new Idrak.Cli.AspNetCoreServers());

// The tests never read this machine's calibrated link profiles (plan 23, C8: idrak cluster calibrate keeps them in the
// tuning cache, where every topology would pick them up): a file nobody writes, unless a run names one. Child processes inherit it.
if (Environment.GetEnvironmentVariable("IDRAK_CLUSTER_PROFILES_FILE") is null)
{
    Environment.SetEnvironmentVariable("IDRAK_CLUSTER_PROFILES_FILE", Path.Combine(Path.GetTempPath(), "idrak-tests-no-calibration", "profiles.txt"));
}

if (args is ["--cluster-child", var clusterScenario, .. var clusterArguments])
{
    return DesktopTests.ClusterChild(clusterScenario, clusterArguments);                   // a rank process of the cluster tests (plan 23, C5)
}

if (args is ["--list-devices"])
{
    return Tests.ListDevices();
}

if (args is ["--bench-gemm"])
{
    return Tests.BenchGemm();
}

if (args is ["--bench-gemv"])
{
    return Tests.BenchGemv();
}

if (args is ["--bench-cpu"])
{
    return Tests.BenchCpu();
}

if (args is ["--bench-vulkan", .. var benchSections])
{
    return Tests.BenchVulkan(benchSections);
}

if (args is ["--bench-hip", .. var hipSections])
{
    return DesktopTests.BenchHip(hipSections);
}

if (args is ["--bench-text", .. var textSections])
{
    return DesktopTests.BenchText(textSections);
}

if (args is ["--bench-window"])
{
    return Tests.BenchWindow();
}

if (args is ["--bench-spans", .. var spanPath])
{
    bool spanTraining = spanPath.Contains("train");
    string[] spanPaths = [.. spanPath.Where(a => a != "train")];
    return Tests.BenchSpans(spanPaths is [var chosenPath] ? chosenPath : "all", spanTraining);
}

if (args is ["--bench-conv"])
{
    return Tests.BenchConv();
}

if (args is ["--bench-vision", .. var visionArguments] && visionArguments.Length <= 1)
{
    return Tests.BenchVision(visionArguments.FirstOrDefault());
}

if (args is ["--bench-image"])
{
    return Tests.BenchImage();
}

if (args is ["--bench-read"])
{
    return Tests.BenchRead();
}

if (args is ["--bench-alloc"])
{
    return Tests.BenchAlloc();
}

if (args is ["--bench-offload"])
{
    return Tests.BenchOffload();
}

if (args is ["--bench-dispatch"])
{
    return Tests.BenchDispatch();
}

if (args is ["--bench-fp8"])
{
    return Tests.BenchFp8Quantizers();
}

if (args is ["--dump-ptx", var ptxPath])
{
    File.WriteAllText(ptxPath, PtxKernels.Source);
    File.WriteAllText(Path.ChangeExtension(ptxPath, ".tensorcore.ptx"), PtxKernels.TensorCoreSource);
    for (int d = 8; d <= PtxKernels.SpanMaxDim; d += d < PtxKernels.FlashMaxDim ? 8 : 16)   // the span modules, one per padded head size
    {
        File.WriteAllText(Path.ChangeExtension(ptxPath, $".spans_f32_d{d}.ptx"), PtxKernels.SpanFloatModule(d).Source);
        if (d % 16 == 0 && d <= PtxKernels.FlashMaxDim)
        {
            File.WriteAllText(Path.ChangeExtension(ptxPath, $".spans_tc_d{d}.ptx"), PtxKernels.SpanTensorModule(d).Source);
        }

        if (d <= PtxKernels.FlashMaxDim)
        {
            File.WriteAllText(Path.ChangeExtension(ptxPath, $".spans_tcp_d{d}.ptx"), PtxKernels.SpanPipelinedModule(d).Source);
        }
    }

    File.WriteAllText(Path.ChangeExtension(ptxPath, ".spans_bwd_wide.ptx"), PtxKernels.SpanBackwardWideModule().Source);
    Console.WriteLine($"Wrote {PtxKernels.Source.Length} characters of PTX to {ptxPath}");
    return 0;
}

if (args is ["--dump-spirv", var spirvFolder])
{
    return Tests.DumpSpirv(spirvFolder);
}

// IDRAK_FILTER: one text, or several separated by "|" (a test runs when its name contains any of them, ignoring case), so
// every group a change touches runs in one invocation.
var filters = TestRunner.ParseFilters(Environment.GetEnvironmentVariable("IDRAK_FILTER"));

// --list: the names of the tests the filter keeps, one a line, in the order they run (nothing is run).
if (args is ["--list"])
{
    foreach (string name in TestRunner.List(DesktopTests.All, filters))
    {
        Console.WriteLine(name);
    }

    return 0;
}

// What the run is on, so a pasted result says where it passed (each GPU's line adds its compute capability and driver).
string runHeader = $"Idrak tests: {System.Runtime.InteropServices.RuntimeInformation.OSDescription} "
                   + $"({System.Runtime.InteropServices.RuntimeInformation.OSArchitecture}), {System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription}";

// Every device: the CPU and each GPU found (IDRAK_DEVICES=cpu,cuda:1,vulkan:0 … to choose, including devices not
// listed by default, such as a software Vulkan driver), since the library runs on any of them.
List<Device>? devices = Environment.GetEnvironmentVariable("IDRAK_DEVICES") is { Length: > 0 } chosen
    ? [.. chosen.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(Device.Parse).Distinct()]
    : null;

// --progress and --failures-only (or their variables): the runner's lines go to the standard output as it was; the tests'
// own lines go through a writer that takes the progress line off the screen first, or nowhere when only failures show.
bool progress = args.Contains("--progress") || Tests.RunnerFlag("IDRAK_TEST_PROGRESS");
bool failuresOnly = args.Contains("--failures-only") || Tests.RunnerFlag("IDRAK_TEST_FAILURES_ONLY");

// --verbose and --profile, as the CLI's: the library's work as it happens (kernel choices measured, host fallbacks, graph
// recordings that failed, prompt passes, decoding) and one profile report per test (IDRAK_PROFILE_PROBES and the other
// IDRAK_PROFILE_* settings apply), both written as the runner's own lines after a "test:" line naming the test, so they
// reach the failures file beside its result. Flags only: IDRAK_VERBOSE and IDRAK_PROFILE would also turn on the CLI
// commands the tests run, whose output they check.
bool verbose = args.Contains("--verbose");
var profileOptions = args.Contains("--profile") ? ProfileOptions.FromEnvironment() : null;

// --failures-file PATH (IDRAK_TEST_FAILURES_FILE): the header, the device headers, the failures and the summary written by
// the runner itself as they happen, so the console keeps the progress line and the lines in order (a pipe to another
// program prints them late, onto the progress line).
string? failuresPath = Array.IndexOf(args, "--failures-file") is var at and >= 0
    ? (at + 1 < args.Length ? args[at + 1] : throw new InvalidOperationException("--failures-file needs a path."))
    : Environment.GetEnvironmentVariable("IDRAK_TEST_FAILURES_FILE") is { Length: > 0 } named ? named : null;

// The progress line on the error output (its writer is the lock both share). Only a line drawn in place is taken off the
// screen for the tests' own lines; a redirected one ("plain") is a line of its own at its interval, so other output goes
// around it as it is.
var screen = Console.Error;
var bar = progress ? ProgressReporters.Start(Tests.ProgressLabel(0, 0)) : null;                     // the runner sets its total
var inPlace = bar is not null && ProgressReporters.Choose() == ProgressReporters.ConsoleReporter ? bar : null;
var summary = TestRunner.Run(DesktopTests.All, new TestRunOptions
{
    Filters = filters,
    Devices = devices,
    TimeoutSeconds = int.TryParse(Environment.GetEnvironmentVariable("IDRAK_TIMEOUT"), out int t) ? t : 300,
    RepositoryRoot = Environment.GetEnvironmentVariable("IDRAK_TEST_REPOSITORY") is { Length: > 0 } repository ? repository : null,
    DataFolder = Environment.GetEnvironmentVariable("IDRAK_TEST_DATA") is { Length: > 0 } data ? data : null,
    Output = Console.Out,
    OutputLock = screen,
    FailuresOnly = failuresOnly,
    FailuresFile = failuresPath,
    Progress = bar,
    TestOutput = inPlace is null ? null : new Tests.ProgressAwareWriter(Console.Out, screen, inPlace),
    Header = runHeader,
    FullErrors = Environment.GetEnvironmentVariable("IDRAK_TRACE") == "1",
    Verbose = verbose ? lines => new Idrak.Cli.Shared.VerboseLines(lines) : null,
    Profile = profileOptions,
    Hung = _ => Environment.Exit(3),
});
return summary.ExitCode;

internal abstract partial class DesktopTests : Tests
{
    public static new (string Name, Action<Device> Run)[] All => [.. InventoryGroup, .. Basic, .. Advanced, .. TensorOpsGroup, .. Decoding, .. Generation, .. Simplified, .. Callbacks, .. AspNetCore, .. Retrieval, .. Onnx, .. Quantization, .. Decoder, .. Pretrained, .. ChatTemplates, .. FineTuning, .. TrainingRegistryGroup, .. MixedPrecisionGroup, .. CodingToolsGroup, .. DatasetsGroup, .. GgufGroup, .. NamingGroup, .. StreamingText, .. OffloadingGroup, .. TuningGroup, .. CpuTuningGroup, .. AppHostingGroup, .. AbstractionGroup, .. OperationGroup, .. DeviceContractGroup, .. TrainingContractGroup, .. DomainContractGroup, .. LayerContractGroup, .. ConformanceKitGroup, .. OpPluginGroup, .. PackedPluginGroup, .. ModelPluginGroup, .. PluginPackageGroup, .. DatasetPluginGroup, .. MinimalBackendGroup, .. KeyValuePluginGroup, .. OutsidePluginGroup, .. VulkanGroup, .. VulkanRuntimeGroup, .. SpirvGroup, .. VulkanCnnGroup, .. VulkanPromptGroup, .. VulkanMatrixGroup, .. VulkanTrainingGroup, .. VulkanFusedGroup, .. VulkanLimitsGroup, .. VulkanReductionGroup, .. VulkanGraphGroup, .. HipGroup, .. HipBenchGroup, .. GcnGroup, .. GcnInterpreterGroup, .. GcnPointwiseGroup, .. GcnRowsGroup, .. GcnProductsGroup, .. GcnAttentionGroup, .. GcnDecodingGroup, .. GcnPluginGroup, .. ModelFamilyGroup, .. MixtureOfExpertsGroup, .. VulkanMixedMatrixGroup, .. CliRunGroup, .. CliDeveloperGroup, .. CliServeGroup, .. CliModelsGroup, .. CliDesignGroup, .. CliMeasureGroup, .. CliRetrievalGroup, .. CliTrainDataGroup, .. CliHealthGroup, .. CliPolishGroup, .. DataLoaderGroup, .. ImageDecodingGroup, .. WindowKernelGroup, .. SpanAttentionGroup, .. DistillationGroup, .. CliArabicGroup, .. VisionGroup, .. ToolContractGroup, .. ChatPartGroup, .. VisionEncoderGroup, .. CliImageGroup, .. ImagePrefillGroup, .. VisionTuningReferenceGroup, .. VisionContractGroup, .. QwenVisionGroup, .. PanAndScanGroup, .. ImageTransformGroup, .. FontGroup, .. ShapingGroup, .. PageGroup, .. ScanGroup, .. SyntheticDataGroup, .. RandomTextGroup, .. TokenizerTrainingGroup, .. TokenizerGroup, .. TokenizerReaderGroup, .. TokenizerPipelineGroup, .. TokenizerWordPieceGroup, .. TokenizerUnigramGroup, .. TokenizerGgufGroup, .. TokenizerEncodingGroup, .. UnicodeNormalizationGroup, .. HiddenBoxingGroup, .. PretrainGroup, .. CliPretrainGroup, .. VlmScratchGroup, .. PretrainVisionGroup, .. DistillVisionGroup, .. VisionTuningContractGroup, .. VisionTuningPhase6Group, .. VisionSequenceGroup, .. RecurrentGroup, .. VisionDetectionGroup, .. VisionAugmentationGroup, .. VisionLayersGroup, .. OverrideLoopGroup, .. KindTableGroup, .. ImageFamilyGroup, .. VisionKernelsGroup, .. CliVisionGroup, .. StructuredAnswerGroup, .. ConversationTuningGroup, .. ProfilerGroup, .. AudioCodecGroup, .. AudioFlacGroup, .. AudioMp3Group, .. AudioMp3EncodeGroup, .. AudioFeatureGroup, .. AudioTtsGroup, .. AudioChatGroup, .. AudioAsrGroup, .. AudioVoiceGroup, .. AudioSpeakersGroup, .. PyTorchCheckpointGroup, .. AudioTrainGroup, .. AudioCleaningGroup, .. AudioSeq2SeqGroup, .. AudioLanguageGroup, .. AudioLanguageFamilyGroup, .. AudioTtsTrainGroup, .. AudioCliGroup, .. AudioCliTrainGroup, .. AudioVoiceTrainGroup, .. AudioWhisperTrainGroup, .. AudioLiveGroup, .. AudioPlaybackGroup, .. AudioProcessingGroup, .. AudioFilterGroup, .. AudioDynamicsGroup, .. AudioRepairGroup, .. AudioProcessingCliGroup, .. AudioRnNoiseGroup, .. LayerLibraryGroup, .. ClusterGroup, .. ClusterSimGroup, .. ClusterTrainingGroup, .. ClusterTopologyGroup, .. ClusterOverlapGroup, .. ClusterTransportGroup, .. ClusterShardingGroup, .. ClusterPipelineGroup, .. ClusterCliGroup, .. ClusterDashboardGroup, .. ClusterMemoryGroup, .. ClusterStopGroup, .. ClusterControlGroup, .. ClusterAdvisorGroup, .. ClusterTuningGroup, .. EngineGroup, .. GenericComponentGroup, .. BertGroup, .. GenericVisionComponentGroup, .. T5Group, .. VisionPluginGroup, .. GemmaGroup, .. TextFamilyGroup, .. ReinforcementContractGroup, .. ReinforcementTextGroup, .. ReinforcementClusterGroup, .. DistillationClusterGroup, .. ReinforcementGroup, .. CodingPluginGroup, .. CodingLanguageGroup, .. CSharpCodingGroup, .. CliReinforceGroup, .. ClassicControlGroup, .. OffPolicyGroup, .. PolicyNetworkGroup, .. CliReinforceNumericGroup, .. ApiSurfaceGroup, .. TestRunnerGroup, .. DesktopRunnerGroup];
}
