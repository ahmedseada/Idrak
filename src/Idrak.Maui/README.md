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
  library registers its own (Devices, Models, Chat, Transcribe, Speak, Audio process); an app adds its own or replaces one
  by name.
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
- **Transcribe** page: live transcription from the microphone through the library's `LiveTranscriber` (the
  microphone permission asked first; the capture `AudioCaptures.ForThisMachine()` picks, aaudio on Android), with an
  optional live cleaning pipeline (`AudioPipeline` text, live steps only); or an audio file picked and decoded by the
  library's own codecs, read through the same live path or whole. A speech model is picked from the store's complete
  models whose architecture a registered speech family reads, loaded once through the model service (host `speech`)
  on the settings' device. The committed words and the tentative ones (lighter) as they arrive, the level, the time
  and how far behind the model runs, updated at most once a progress interval; stop commits the last segment; copy
  and share the text. Every `LiveTranscriptionOptions` value is a setting on the page (empty: the library's
  default), kept in the settings store under `idrak.transcribe.`. The library registers no family: the app does
  (Idrak.Mobile: `WhisperPlugin.Register()`).
- **Speak** page: text read aloud on the speaker by a synthesis model of the store whose architecture a registered
  synthesis family reads, loaded once through the model service (host `synthesis`) and wrapped in a `VoicedSynthesizer`,
  so the model's speakers and the voice store's installed voices speak. The text is spoken piece by piece
  (`SpeakStream`, sentences up to `SpeakSettings.DefaultMaxCharacters`, a setting) and played through `ISoundPlayer` as
  the first piece is made, the next made while one plays; Stop cuts it. The time to the first sound and the real-time
  factor (the synthesis' own time over the sound's), the first speech after a load (cold) apart from the later ones
  (warm); the speech played again, saved as WAV or MP3 (the library's encoders) and shared. A model or a voice that needs
  what is not registered here is listed with its reason, not hidden: a checkpoint that reads phonemes when no phonemizer
  is registered (the library's `SpeechSynthesisModels.TextNeeds` and `SpeechTextNeeds.Missing`; espeak-ng is an outside
  program an app does not run), an embedding voice whose converter is not registered. Every `SpeechSynthesisOptions` value,
  the speaker and the saved format are settings (empty: the library's default), kept under `idrak.speak.`. The library
  registers no family: Idrak.Mobile calls `VitsPlugin.RegisterFamily()` (VITS without espeak-ng) and
  `IdrakAudioPlugin.Register()`.
- **Audio process** page: a recording from the microphone (the permission asked first; kept in pooled buffers up to the
  longest recording, a setting) or a file decoded by the library's codecs, run through a pipeline written as the
  library's text (`AudioPipeline`, checked as it is written), the registered processors listed as chips (name, summary,
  parameters, live or whole recording) that append a step; applied whole or in blocks as live sound goes, timed cold
  (the pipeline's first apply) and warm; its latency and delay; the level (RMS and peak) before and after; the original
  and the processed played on the speaker (A/B, one at a time); the result saved and shared. Monitoring, off by default
  and with a feedback warning, plays the microphone through the pipeline's live steps on a short low-latency buffer,
  its delay shown. Settings kept under `idrak.audio_process.`.
- **Sound out**: `ISoundPlayer` (`SoundPlayer`, registered by `UseIdrak`) plays a recording or blocks as they come
  through the library's `AudioPlaybacks` (aaudio on Android), held to the sound's pace by the output's back-pressure,
  drained at the end; one sound at a time.
- **Services**: models kept loaded through `ModelHost` and released on the phone's memory pressure, the model store
  (pull with progress, list, sizes, remove), device information (the Vulkan driver in the app's process, its measured
  width and subgroup sizes, the memory budget, CPU features), permissions and sharing a file out.
- **Planned features**: a feature not built yet is registered with its real name, title and place in the menu
  (`Order`) as `FeatureState.Planned`, with a line on what is coming (`Coming`); the library's placeholder page shows
  that and what it will need, and the menu marks it "planned". The library's: OCR, Image and Search (`PlannedFeature`); its phase registers the built feature under the same name, so the menu does
  not change. A feature that sets no order or state is ready and takes the library's place of its name, else comes last.
- **`maui`** progress reporter: the library's progress (downloads, loads, runs) as a bindable model, updated on the UI
  thread at most once a progress interval.
- **Themes**: `IdrakStyles` merged by the app, with a light and a dark colour dictionary from the library's icon, swapped
  by `IdrakThemes.Apply`; every key may be overridden by the app.
