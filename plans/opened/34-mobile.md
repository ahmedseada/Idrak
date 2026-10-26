# Plan 34: Idrak.Maui and Idrak.Mobile (the library on a phone, as an app; Android first)

**Status:** planned 2026-10-26 against `main` 5c7d54a; M0, M1 and M3 built (459e41a, b7b6739, 44303bd); M2 and the Models lab under way; the Self-test lab after M7. Phases below, built in order, at most two agents
at a time. Android is the target; iOS is later and outside this plan.

## Why

Today the library is tested on a phone through Termux: Ubuntu in proot, the .NET SDK inside it, a Vulkan driver built
from Mesa's sources, `VK_ICD_FILENAMES` and a heap limit (installation/android-termux.md). That proves the library's C#
runs on ARM64 Linux; it does not prove it runs as an Android app, and it never touches the phone's own Vulkan driver.

The owner's request (October 2026): a .NET MAUI app (XAML pages, not MAUI Blazor) that

- tests the library on the phone without Termux: the phone's own driver loaded in the app's process, the test groups
  run from the app, the failures shared out;
- showcases what the library can do, one page (a "lab") per capability: chat, speech, audio processing, OCR, images,
  retrieval, devices. An audio lab is one example of the pattern, not the purpose;
- is two parts: **Idrak.Maui**, a MAUI library any MAUI app adds to get the features (a multi-feature plug-in), and
  **Idrak.Mobile**, the app itself, which makes the registrations, hosts the models and is the worked example.

The owner's answers: the app in `apps/`; the plan on `main`; Android finished first, iOS later; the test groups split
into a library the app references; at most two agents; no CI work (the owner tests on the phone); the bench last;
nothing that needs a shell.

## What the library has and lacks (surveyed 2026-10-26)

- **Capture is a contract**: `IAudioCapture` / `IAudioInput` in `AudioCaptures` (a `SlotTable`), built in `winmm` and
  `alsa`, each loading the system's library at run time. Nothing for Android.
- **Playback has no contract at all.** Synthesis returns `AudioData`; nothing in the library plays it.
- **Registrations survive trimming.** Library defaults load lazily by assembly name with `[DynamicDependency]`; every
  library package but Idrak.Cli, Idrak.AspNetCore and Idrak.Onnx.Runtime sets `IsAotCompatible`. Plug-in packages need
  their explicit call (`TextPlugins.RegisterAll()` and so on).
- **Vulkan names `libvulkan.so` on Android** (`VulkanDriver.CandidateNames`) but the in-app path was never run. Memory is
  the heap size less a reserve; `VK_EXT_memory_budget` is not read, so on a phone almost all of RAM looks free. A lost
  device throws with no recovery.
- **The cache root moves only by `IDRAK_CACHE`.** It falls back to `UserProfile/.cache/idrak`, which on Android may be
  empty and give a relative path. Loading a `.gguf` always prepares into that cache.
- **Console**: the progress line's and `ConsoleStatus`'s width reads catch only `IOException`.
- **Child processes**: the espeak-ng phonemizer (needed by some VITS voices), git in the coding workspace, the command
  sandbox and the localhost cluster. None is on an inference path.
- **The tests are one executable** (`tests/Idrak.Tests`, groups concatenated in `Tests.All`, its own runner); an app
  cannot call them.
- **The CPU has NEON paths** (`TiledKernel.Neon`, DotProd and RDM detected).

## The shape

### Idrak.Maui (`src/Idrak.Maui`, a MAUI class library)

- **`UseIdrak(options)`** on `MauiAppBuilder`: registers the services, the platform parts into the library's
  registries, and the features chosen.
- **`IdrakMauiOptions`**: device (`auto`, `cpu`, `vulkan:N`), cache folder (default: the app's own storage), every
  `ComputeResources` limit, the Hugging Face token, the features to enable, the progress interval. Each maps to the
  library's option and its existing variable; defaults are measured, never the only way.
- **`IMobileFeature`** (`Idrak.Maui.Abstractions`), registered in **`MobileFeatures`** (a `SlotTable`): name, title,
  summary, what it needs from the phone (microphone, camera, network, a model) and from the library (packages, model
  families), and the factories of its page and view model. The library's labs register as defaults; a host app adds its
  own or replaces one by name.
- **Services**: models loaded once through the library's `ModelHost` and released on the phone's memory-pressure signal
  (with `ComputeResources.ReleaseCachedMemory`); a model store over the library's downloads (pull with progress, list,
  sizes, remove); device information over the library's device list, memory usage and CPU features; permissions
  (microphone, camera); sharing a file out (the failures file, a log, a recording).
- **`maui` progress reporter** in `ProgressReporters`: a bindable view of what is running, how far, how fast and the
  time left, as the CLI's line shows it.
- **Reusable views**, one per feature, without platform code; platform code only under `Platforms/Android` (and later
  `Platforms/iOS`) behind contracts.

### Idrak.Mobile (`apps/Idrak.Mobile`, the app)

- The example on its own: it references the library's projects only, never a sample. Code a sample already has (a
  reader, a chat loop, argument parsing) may be copied in and adapted; nothing is referenced from `samples/`.
- `apps/Directory.Build.props` and `.targets` of its own: not packable, the library's projects from source through
  `build/IdrakFromSource.targets`.
- Navigation by a flyout and navigation pages (no MAUI Shell); the flyout lists the registered features.
- **A modern look in both themes.** Light and dark follow the phone's setting by default, with a choice in the
  settings page (system, light, dark) kept between runs. Colours, type sizes and spacing are resources in one
  dictionary per theme (`AppThemeBinding` and dynamic resources), never set on a page; cards, rounded corners,
  the system's font, room for the phone's text size. Every page is checked in both themes.
- **The library's icon is the app's icon** (`assets/icon.svg`): the launcher's adaptive icon takes the icon's dark
  ground as its background colour and the graph mark (the icon without its rounded square, which the launcher's own
  mask replaces) as its foreground, read from the repository's SVG at build time so the two never drift apart; the
  splash screen shows the same mark on the same ground; the theme's accent colours come from the icon's own.
- **A settings page** over every option of `IdrakMauiOptions`: device, theme, the cache and models folder, memory
  and thread limits (empty means measured), the Hugging Face token (kept in the phone's secure storage), the
  progress interval, which labs show. Settings are kept between runs and applied at start-up, before the first
  device or download; a change that needs a restart says so.
- Registers the families it shows (each plug-in by its own call; EspeakNg is not registered), the settings page over
  `IdrakMauiOptions`, and the labs.

### Labs, in build order

| Lab | Packages | What it shows |
|---|---|---|
| Devices | Idrak, Idrak.Gpu | The devices found, the Vulkan driver loaded in-process, the widths and subgroup sizes measured, CPU features, memory as measured |
| Self-test | the test library | A test group by filter on `cpu` or `vulkan:N`, with live progress; the failures file shared out |
| Models | Idrak.Data | Pull into app storage with progress, list, remove |
| Chat | Idrak.Text, Idrak.Text.Plugins | A small model streams its answer with tokens per second |
| Transcribe | Idrak.Audio, Idrak.Audio.Plugins | Live transcription from the microphone (`LiveTranscriber`), with an optional cleaning pipeline |
| Speak | Idrak.Audio, Idrak.Audio.Plugins | Synthesis played on the speaker; voices that need espeak-ng are listed with that reason |
| Audio process | Idrak.Audio | Record, apply a named pipeline, play before and after, its latency and delay |
| OCR | Idrak.Imaging, Idrak.Vision, Idrak.Vision.Plugins | A photo or the camera, decoded by the library's own codecs, read by a small reader; a vision-language reader only where measured memory allows |
| Image | Idrak.Imaging | Named transforms and scan profiles on a photo, cold and warm times |
| Search | Idrak.Text | Embeddings and retrieval over the user's own notes |
| Bench | the bench rows | The library's bench rows on the phone, results shared out (last) |

## Phases

| Phase | What | Depends on | Agent |
|---|---|---|---|
| M0 | **Library: running inside an app.** One cache-root setting in Idrak.Abstraction (`IDRAK_CACHE` its variable), read by downloads, `.gguf` preparing, every tuning and feature cache, the voice store and RnNoise; never a relative path. Console width reads catch every failure. Vulkan reads `VK_EXT_memory_budget` when the driver has it; a lost device names the operation and takes the existing host retry where it is on | none | A |
| M1 | **Library: sound out, and Android sound.** `IAudioPlayback` / `IAudioOutput` in `Idrak.Audio.Abstractions`, in **`AudioPlaybacks`** (a `SlotTable`), mirroring capture (available here and why not, devices, open, blocks written in order, drain). Built in: `winmm` (waveOut), `alsa` (PCM playback), and `aaudio` capture and playback for Android, from `libaaudio.so` loaded at run time (no Java binding, nothing shipped). `ForThisMachine` picks `aaudio` on Android; a refused microphone permission is named | none | B |
| M2 | **The test groups as a library.** The groups and the runner's logic (filter, devices, progress, failures file, timeout) move to `tests/Idrak.Tests.Library`; `tests/Idrak.Tests` keeps its program and its commands unchanged. Tests find repository files through one root; where it is missing (on a phone), a test that needs the repository reports "not run here: needs the repository" instead of failing, and the summary counts them apart | M0 | A |
| M3 | **Idrak.Maui core, and a first app to open.** The project (`net10.0;net10.0-android`), `UseIdrak`, the options, `IMobileFeature` and `MobileFeatures`, the services, the `maui` reporter, the theme; the app `apps/Idrak.Mobile` (`apps/Idrak.Mobile.slnx`) with its navigation, both themes, the icon, the settings page and the Devices lab, so the owner can open it in Visual Studio early | M0, M1 | B |
| M4 | **The Models lab** (pull with progress, sizes, free space, remove). The Self-test lab moves after M7 (the owner, 2026-10-26: the features first; M2 is kept for it) | M3 | B |
| M5 | Chat lab; Transcribe lab | M4 (Models) | A; B |
| M6 | Speak and Audio process labs; OCR and Image labs | M5 | A; B |
| M7 | Search lab. **Held: the owner decides after testing the existing features** | M6 | A |
| M7b | **The Self-test lab** (**held: the owner decides after testing the existing features**): the app's own feature over the test library (M2), a group on `cpu` or `vulkan:N` with live progress, the failures file shared out | M2, M7 | A |
| F | **Fixes from the owner's first runs** (a phone and a tablet): the memory measured from the system where the runtime's own figure is its heap budget (Mono reported 143-172 MB); portable 128-bit vector paths where NEON is not reported but vectors are accelerated (Mono); CoreCLR as an app build option, measured against Mono; a Vulkan kernel the driver refuses retried without the subgroup-size request, then its smaller variants, then that operation on the host, never the whole device; the device list measured cold and warm; the CPU named by its core groups | M6 | A |
| R | **The redesign the owner approved** (2026-10-27): a landing with a carousel on the library and the app, bottom navigation with Home, Chat, Voice and Vision on phones and a side rail with multi-pane pages on tablets (chosen by the window's measured width), an All labs sheet or page grouped by Language, Voice, Vision and the device, every lab laid out for its own feature, light and dark, the source commit shown instead of the version | M6 | B |
| M8 | Bench lab: the bench rows callable from the app as the tests are (M2's pattern). **Held: the owner decides when** | M7 | A |

**Done when:**

- **M0, M1, M2:** the CPU groups they touch pass here (`IDRAK_DEVICES=cpu`, filtered), the public API dump and the
  inventory are regenerated and pass, and the owner's test commands give the same counts as before.
- **M3, M4:** the app installs on the owner's phone; the Devices lab shows `vulkan:0` from the phone's own driver and
  the CPU; the Self-test lab runs a group on both and shares its failures file; the counts are compared with the
  Termux runs.
- **M5:** Qwen3-0.6B int8 chats on `cpu` and `vulkan:0` with the same greedy tokens as on the desktop; speech into the
  microphone is written live.
- **M6, M7:** each lab's output matches the CLI's on the same input (rule 71).
- **M8:** the bench rows run on the phone and their results are shared out.

## Choices measured, not fixed

- **Runtime.** .NET for Android runs Mono by default; CoreCLR is an opt-in. The CPU backend's speed differs between
  them; both are measured with the Self-test timings and the chat's tokens per second, and the choice stays a build
  property of the app.
- **Heap.** The Termux heap limit came from a glibc process in proot. Inside an app it is measured first; if a runtime
  needs one, it becomes an app setting sized from the phone's measured memory, never a fixed size.
- **Models' place.** The app's internal storage by default; the app-specific external folder as a setting, for large
  models.

## Testing

- **Here (this session's container):** the .NET 10 SDK is installed for the session; M0, M1 and M2 are built and their
  CPU groups run, filtered. Whether the `maui-android` workload and the Android SDK install here is tried at M3: if they
  do, the APK is built here before each push; running it stays on the owner's phone.
- **The owner:** the phone runs and the full test lists. After each phase, one command list in the order the working
  rules set.

One-time setup on the owner's Windows machine (the Android SDK into a folder of the owner's choosing):

```powershell
winget install Microsoft.OpenJDK.17
dotnet workload install maui-android
cd D:\Projects\Idrak; dotnet build apps\Idrak.Mobile -f net10.0-android -t:InstallAndroidDependencies "-p:AndroidSdkDirectory=D:\Android\Sdk" "-p:AcceptAndroidSDKLicenses=True"
```

Then, with the phone connected by USB with debugging on, per phase:

```powershell
cd D:\Projects\Idrak; git checkout main; git pull origin main
dotnet build -c Release tests\Idrak.Tests
dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress --failures-only --failures-file test-output\fails.txt
dotnet build -c Release apps\Idrak.Mobile -f net10.0-android -t:Install "-p:AndroidSdkDirectory=D:\Android\Sdk"
```

The test line carries the `IDRAK_FILTER` of what changed; the app is opened on the phone, and its shared files saved
under `test-output\phone\`.

## Out of this plan

- iOS, held for the owner to decide (later: AudioToolbox capture and playback through the same contracts, a full AOT build, the CPU until the Apple
  plan brings Metal).
- CI, versions, packaging and publishing.
- Anything that needs a shell or a child process: Termux, the CLI, espeak-ng, the coding workspace and its sandbox, the
  localhost cluster.
- Serving the HTTP API from the phone.

## Risks

- **Phones' Vulkan drivers differ** from the Mesa driver of the Termux runs. Every limit is measured at run time
  already; the first in-app GPU run is itself a result. A lab that hits a lost device says so and offers the CPU.
- **Android ends an app under memory pressure without an exception.** The model host sizes from measured limits, a
  model that does not fit is refused with the limit named, and the pressure signal releases cached memory first.
- **Trimming in Release builds.** The library's packages are trim-checked; code marked as needing unreferenced code
  (tools from delegates, reflection JSON in model packages) stays out of the labs or uses its `JsonTypeInfo` overload.
- **Microphone and camera permissions** are asked by the app at first use; a refusal is shown with how to allow it.

## Effort (rough)

M0-M2: 1-2 weeks; M3-M4: 2 weeks; M5-M7: 3-4 weeks; M8: 3 days.

## Rules that apply

docs/optimization.md throughout (rule 70 most of all: Android's sound through the library itself, images through the
library's own codecs); abstraction always (`AudioPlaybacks`, `MobileFeatures`); nothing fixed that someone may need to
adjust; card-agnostic (no phone, GPU or memory size named in code or advice); the fix workflow and the targeted tests
of CLAUDE.md.
