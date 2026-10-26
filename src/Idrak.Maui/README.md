# Idrak.Maui

Idrak inside a .NET MAUI app (plain MAUI with XAML and MVVM; Android first).

```csharp
var builder = MauiApp.CreateBuilder();
builder.UseMauiApp<App>().UseIdrak(options =>
{
    options.Theme = IdrakThemeChoice.System;     // the app's defaults; what the user saves in the settings page wins
});
```

- **`UseIdrak`** applies the options before any device or download: the cache folder (the app's own storage by
  default, `ComputeResources.CacheFolder`), the device (`auto`, `cpu`, `vulkan:0`), every `ComputeResources` limit
  (empty means measured), the Hugging Face token and the progress interval. Settings are kept between runs (MAUI
  Preferences; the token in SecureStorage) and a change that needs a restart says so.
- **Features** (`IMobileFeature` in `MobileFeatures`): a page and its view model, with what it needs from the phone. The
  library registers its own (Devices, Models, Chat); an app adds its own or replaces one by name.
- **Models** page: pull a Hugging Face id (and a revision) into the cache folder with its progress (bytes, rate, time
  left), cancellable and resumed by pulling again; the token from the settings when set; the space checked before the
  first byte (`StorageHeadroomMiB` keeps some free); the models stored with sizes and folders, removing one after a
  confirmation; the storage's free space. `SuggestedModels` (set by the app) shows as chips to tap.
- **Chat** page: a finished model of the store, loaded once through the model service on the settings' device in the
  settings' weights and key/value cache formats, after checking it fits the memory the device has (refused with the
  limit named otherwise); the answer streams into a bubble with its token count and tokens per second, the bubble
  updated at most once a progress interval; Stop, a new chat, copy, thinking where the template has a switch; the first
  answer after a load (cold) apart from the warm ones. The system prompt, temperature, top-p, top-k, max new tokens,
  context and seed are settings over the library's `GenerationOptions`. Idrak.Maui registers no model family: the app
  registers the ones it shows (`QwenPlugin.Register()` and so on), and a model of another family fails as not registered.
- **Services**: models kept loaded through `ModelHost` and released on the phone's memory pressure, the model store
  (pull with progress, list, sizes, remove), device information (the Vulkan driver in the app's process, its measured
  width and subgroup sizes, the memory budget, CPU features), permissions and sharing a file out.
- **Planned features**: a feature not built yet is registered with its real name, title and place in the menu
  (`Order`) as `FeatureState.Planned`, with a line on what is coming (`Coming`); the library's placeholder page shows
  that and what it will need, and the menu marks it "planned". The library's: Chat, Transcribe, Speak, Audio process,
  OCR, Image and Search (`PlannedFeature`); its phase registers the built feature under the same name, so the menu does
  not change. A feature that sets no order or state is ready and takes the library's place of its name, else comes last.
- **`maui`** progress reporter: the library's progress (downloads, loads, runs) as a bindable model, updated on the UI
  thread at most once a progress interval.
- **Themes**: `IdrakStyles` merged by the app, with a light and a dark colour dictionary from the library's icon, swapped
  by `IdrakThemes.Apply`; every key may be overridden by the app.
