// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Diagnostics;

namespace Idrak.Maui.Abstractions;

/// <summary>A model loaded in one of the <see cref="IModelService"/>'s hosts.</summary>
/// <param name="Host">The host's name ("chat").</param>
/// <param name="Model">The model as the host reports it.</param>
public sealed record HostedModel(string Host, LoadedModel Model);

/// <summary>What one memory-pressure signal released (<see cref="IModelService.Released"/>).</summary>
/// <param name="Signal">The signal.</param>
/// <param name="Unloaded">The models unloaded, as "host/name".</param>
/// <param name="CachedMemoryReleased">Whether the devices' cached blocks were returned to the system.</param>
public sealed record MemoryRelease(MemoryPressureSignal Signal, IReadOnlyList<string> Unloaded, bool CachedMemoryReleased);

/// <summary>
/// Models kept loaded between uses, through the library's <see cref="ModelHost{TModel}"/>: one host a name ("chat",
/// "speech"), each loading a model once and unloading it after its keep-alive (<see cref="IdrakMauiOptions.ModelKeepAlive"/>)
/// or when <see cref="IdrakMauiOptions.MaxLoadedModels"/> needs the room. On the phone's memory-pressure signal it returns
/// the devices' cached memory (<see cref="ComputeResources.ReleaseCachedMemory"/>) and, from
/// <see cref="IdrakMauiOptions.UnloadModelsAt"/> up, unloads every model not in use.
/// </summary>
public interface IModelService : IDisposable
{
    /// <summary>The host named <paramref name="name"/>, made on first use with <paramref name="load"/> (later calls get the same host).</summary>
    /// <exception cref="InvalidOperationException">A host of that name holds another type of model.</exception>
    ModelHost<TModel> Host<TModel>(string name, Func<string, TModel> load) where TModel : class, IDisposable;

    /// <summary>Every model loaded now, by host.</summary>
    IReadOnlyList<HostedModel> Loaded { get; }

    /// <summary>
    /// Gives memory back as a signal of <paramref name="signal"/>'s level asks: the cached device memory always, the models
    /// not in use from <see cref="IdrakMauiOptions.UnloadModelsAt"/> up (and a full garbage collection at
    /// <see cref="MemoryPressureLevel.Critical"/>). Called on each signal of the source; callable by hand.
    /// </summary>
    MemoryRelease Release(MemoryPressureSignal signal);

    /// <summary>Raised after each release, on the thread that released.</summary>
    event EventHandler<MemoryRelease>? Released;
}

/// <summary>A model in the model store.</summary>
/// <param name="Id">Its Hugging Face id ("Qwen/Qwen3-0.6B").</param>
/// <param name="Revision">The commit it was pulled at (its first 12 characters, as the folder names it).</param>
/// <param name="Folder">Its folder.</param>
/// <param name="Bytes">The bytes of its files.</param>
/// <param name="Files">How many files it has.</param>
/// <param name="Complete">False while a download is unfinished (a partial file is there; pulling again resumes it).</param>
/// <param name="Modified">When a file of it last changed.</param>
public sealed record StoredModel(string Id, string Revision, string Folder, long Bytes, int Files, bool Complete, DateTimeOffset Modified);

/// <summary>
/// The models pulled into the app's storage, over the library's downloads (<c>Idrak.Data</c>: <c>Downloader</c>,
/// <c>HuggingFaceModels</c>), under <see cref="ComputeResources.CacheFolder"/>'s <c>downloads/huggingface/models</c>, where
/// loading a model by its id looks.
/// </summary>
public interface IModelStore
{
    /// <summary>The folder pulled models are kept in.</summary>
    string Folder { get; }

    /// <summary>
    /// Pulls <paramref name="model"/> (once; an unfinished pull resumes): a Hugging Face id ("Qwen/Qwen3-0.6B") takes the
    /// files the library reads, an id followed by a file's path ending in .gguf ("unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf")
    /// that one file. Shows its progress on <paramref name="progress"/>, or on a reporter of its own
    /// (<see cref="ProgressReporters.Start"/>, the app's "maui" one).
    /// </summary>
    /// <exception cref="ArgumentException">The name is neither.</exception>
    Task<StoredModel> PullAsync(string model, IProgressReporter? progress = null, CancellationToken cancellationToken = default);

    /// <summary>The models pulled, each revision apart, by id.</summary>
    IReadOnlyList<StoredModel> List();

    /// <summary>Removes <paramref name="id"/> ("owner/name": every revision; "owner/name@revision": that one); false when it is not there.</summary>
    /// <exception cref="ArgumentException">Not a model id.</exception>
    bool Remove(string id);
}

/// <summary>A device as the device information reads it.</summary>
/// <param name="Info">What the device reports (<see cref="DeviceListing.Describe"/>), its facts included.</param>
/// <param name="Memory">Its memory in use, cached and its limit, or null when its backend could not start.</param>
/// <param name="MeasuredChoices">How many kernel choices its backend measured and keeps (0 where it measures none).</param>
public sealed record DeviceDetails(DeviceInfo Info, MemoryUsage? Memory, int MeasuredChoices);

/// <summary>The process and the machine it runs on.</summary>
/// <param name="OperatingSystem">The operating system's description.</param>
/// <param name="Architecture">The process's architecture ("Arm64").</param>
/// <param name="Runtime">The .NET runtime's description.</param>
/// <param name="Processors">Logical processors.</param>
/// <param name="AvailableMemory">The memory the process may use, as the runtime measures it.</param>
/// <param name="WorkingSet">The process's working set now.</param>
/// <param name="ManagedHeap">The managed heap's size now.</param>
/// <param name="CacheFolder">The library's cache folder (<see cref="ComputeResources.CacheFolder"/>).</param>
/// <param name="CpuThreads">The CPU threads the library may use (<see cref="ComputeResources.MaxCpuThreads"/>).</param>
public sealed record ProcessFacts(string OperatingSystem, string Architecture, string Runtime, int Processors, long AvailableMemory, long WorkingSet,
    long ManagedHeap, string CacheFolder, int CpuThreads);

/// <summary>One reading of the devices.</summary>
/// <param name="Devices">The devices read.</param>
/// <param name="Backends">Each device kind beyond the CPU, with how many it found and why none.</param>
/// <param name="Process">The process.</param>
/// <param name="DefaultDevice">The device models are loaded on when none is named (<see cref="Device.Default"/>).</param>
/// <param name="Elapsed">How long the reading took (starting each backend the first time).</param>
public sealed record DeviceReport(IReadOnlyList<DeviceDetails> Devices, IReadOnlyList<BackendInfo> Backends, ProcessFacts Process, string DefaultDevice, TimeSpan Elapsed);

/// <summary>
/// The devices as the library sees them inside the app's process: the CPU with its features, each GPU of every backend
/// (on a phone, the Vulkan driver loaded in-process) with its measured kernel width, subgroup sizes and memory budget, and
/// the memory each uses.
/// </summary>
public interface IDeviceInformation
{
    /// <summary>
    /// Reads <paramref name="devices"/> ("cpu", "vulkan:0"), or every device of every backend when null, starting each
    /// backend that is not yet (one that cannot start is reported with its error).
    /// </summary>
    DeviceReport Read(IReadOnlyList<string>? devices = null);
}

/// <summary>Whether the app may use what a feature needs.</summary>
/// <param name="Need">What was asked.</param>
/// <param name="Granted">Whether it may.</param>
/// <param name="Message">What to tell the user: nothing when granted, how to allow it when refused.</param>
public sealed record PermissionOutcome(PhoneNeeds Need, bool Granted, string Message);

/// <summary>The phone's permissions for the microphone and the camera (MAUI Permissions in an app).</summary>
public interface IPermissionService
{
    /// <summary>Whether every permission <paramref name="needs"/> takes is granted, without asking.</summary>
    Task<PermissionOutcome> CheckAsync(PhoneNeeds needs);

    /// <summary>Asks for every permission <paramref name="needs"/> takes that is not granted yet.</summary>
    Task<PermissionOutcome> RequestAsync(PhoneNeeds needs);

    /// <summary>Opens the app's page in the system settings, where a refused permission is allowed.</summary>
    void OpenSettings();
}

/// <summary>Sharing a file out of the app (a failures file, a log, a recording): MAUI Share in an app.</summary>
public interface IShareService
{
    /// <summary>Offers <paramref name="path"/> to the phone's share sheet.</summary>
    /// <exception cref="FileNotFoundException">There is no such file.</exception>
    Task ShareFileAsync(string path, string? title = null, string? contentType = null);
}
