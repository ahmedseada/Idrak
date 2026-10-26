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
  library registers its own (Devices, Models); an app adds its own or replaces one by name.
- **Models** page: pull a Hugging Face id (and a revision) into the cache folder with its progress (bytes, rate, time
  left), cancellable and resumed by pulling again; the token from the settings when set; the space checked before the
  first byte (`StorageHeadroomMiB` keeps some free); the models stored with sizes and folders, removing one after a
  confirmation; the storage's free space. `SuggestedModels` (set by the app) shows as chips to tap.
- **Services**: models kept loaded through `ModelHost` and released on the phone's memory pressure, the model store
  (pull with progress, list, sizes, remove), device information (the Vulkan driver in the app's process, its measured
  width and subgroup sizes, the memory budget, CPU features), permissions and sharing a file out.
- **`maui`** progress reporter: the library's progress (downloads, loads, runs) as a bindable model, updated on the UI
  thread at most once a progress interval.
- **Themes**: `IdrakStyles` merged by the app, with a light and a dark colour dictionary from the library's icon, swapped
  by `IdrakThemes.Apply`; every key may be overridden by the app.
