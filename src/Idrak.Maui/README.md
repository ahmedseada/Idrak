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
  (empty means measured), the CPU's vector paths (`ComputeResources.CpuVectors`: automatic, native, portable or
  scalar), the Hugging Face token and the progress interval. Settings are kept between runs (MAUI
  Preferences; the token in SecureStorage) and a change that needs a restart says so.
- **Features** (`IMobileFeature` in `MobileFeatures`): a page and its view model, with what it needs from the phone, its
  group among the labs (Language, Voice, Vision, the device's), a tagline and an icon, and its content as a view a section
  hosts (`CreateView`; the page wraps the same view). The library registers its own (Devices, Models, Chat, Transcribe,
  Speak, Studio, Read text, Image); an app adds its own or replaces one by name.
- **Navigation** (no Shell, no Blazor; the approved design in `apps/Idrak.Mobile/Design`): `AppHostPage` shows one
  section of `AppSections` at a time (Home, Chat, Voice with Transcribe, Speak and Studio, Vision with Read text and
  Image; All labs and Settings at the rail's foot), with a floating bottom bar on a narrow window and a side rail on a
  wide one, chosen by the window's measured width against `WideLayoutWidth` (840 by default, a setting) and switched
  live on rotation or split screen (`AppLayout`). All labs is a sheet over Home on a narrow window and a page on a wide
  one (grouped, searchable, filtered by state and needs). Views open places through `IAppNavigator`; an app adds a
  section or replaces one by name. `AppHostPage.CreateWindow(services)` is the app's window.
- **Home**: the source commit the app was built from ("main · 628b6bb", read from git at build time into assembly
  metadata, `SourceInfo`), the device's status, a carousel of the app's slides (`IdrakMauiOptions.Home`, moving every
  `CarouselInterval` while shown), the device measured (`DeviceStatus`), the labs to try with live previews; two
  columns with the models on a wide window.
- **Models** page: pull a Hugging Face id (and a revision) into the cache folder with its progress (bytes, rate, time
  left), cancellable and resumed by pulling again; the token from the settings when set; the space checked before the
  first byte (`StorageHeadroomMiB` keeps some free); the models stored with sizes and folders, removing one after a
  confirmation; the storage's free space. `SuggestedModels` (set by the app) shows as chips to tap.
- **Chat** (the answers in Markdown: paragraphs, lists, bold, inline and fenced code with Copy, `MarkdownView`, a block
  redrawn only when it changes; "Thought for N s"; tokens per second, tokens, first token, copy and ask again; the chats
  kept between runs through `IConversationStore`, files in the app's storage by default; on a wide window the
  conversations with search and an inspector with the live rate's line, the memory split and quick settings) page: a finished model of the store, loaded once through the model service on the settings' device in the
  settings' weights and key/value cache formats, after checking it fits the memory the device has (refused with the
  limit named otherwise); the answer streams into a bubble with its token count and tokens per second, the bubble
  updated at most once a progress interval; Stop, a new chat, copy, thinking where the template has a switch; the first
  answer after a load (cold) apart from the warm ones. The composer's microphone dictates into the message through
  `IDictation` (the library's is the Transcribe lab's: its speech model, language, cleaning and live settings), live,
  until tapped again; with no speech model ready it says so and offers Models. The system prompt, temperature, top-p, top-k, max new tokens,
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
- **Studio** (the audio process) page, its pipeline also as a flow of steps: a recording from the microphone (the permission asked first; kept in pooled buffers up to the
  longest recording, a setting) or a file decoded by the library's codecs, run through a pipeline written as the
  library's text (`AudioPipeline`, checked as it is written), the registered processors listed as chips (name, summary,
  parameters, live or whole recording) that append a step; applied whole or in blocks as live sound goes, timed cold
  (the pipeline's first apply) and warm; its latency and delay; the level (RMS and peak) before and after; the original
  and the processed played on the speaker (A/B, one at a time); the result saved and shared. Monitoring, off by default
  and with a feedback warning, plays the microphone through the pipeline's live steps on a short low-latency buffer,
  its delay shown. Settings kept under `idrak.audio_process.`.
- **Generate** page (Vision, after Read text and Image): text typed or pasted (Arabic and English; the library shapes it)
  drawn as pages by the library's `SyntheticPages` (what `idrak data synth` does) in the phone's fonts (`$ANDROID_ROOT/fonts`,
  listed, never all read) or fonts added from files (checked, copied into the app's storage), picked by the text's
  characters when none is chosen; the scan effects (`ScanProfiles`) as chips that stack into one look ("aged+handled+photo"),
  a random one, every look one a page, or clean; the layouts as chips; a seed with a shuffle. The pages are made on a
  worker thread, each shown as it is made, Stop keeps those made; a large preview (pinch to zoom, before and after: the
  same drawing without its scan) and the pages as a grid; a page saved to the phone's Pictures (`IPictureSaver`, Android
  10 and later), shared, or the set written as a data set as the CLI writes one. Every option the CLI exposes is a setting
  (`GenerateSettings`, under `idrak.generate.`).
- **Terminal** page (this device's group; from All labs and Home): the app's command runner (`ICommandRunner`, which the
  app registers; Idrak.Mobile's `IdrakCommandRunner` runs the idrak commands of Idrak.Cli.Commands in the app's own process,
  allowing no child process, network listener or console input, so those commands are refused with their reason). One
  command at a time, off the UI thread, in the app's storage (so `idrak models list` lists what Models pulled). What a
  command writes is taken from its thread without waiting and drawn once a display frame into a virtualised list of lines,
  one batch a frame; a carriage return rewrites the line in place (the console's progress lines). Suggestions are the
  runner's commands and options (one the app does not allow greyed with why); Stop, the history (kept), copy, clear,
  `help`; on a wide window the commands pane beside the output. The kept lines and history sizes are `TerminalLabOptions`.
- **Answering at once** (the owner's lag notes): every section's view is made once and kept in the host (the one shown
  visible, the others hidden), and with `PrebuildSections` (on by default) the tabs, their labs and the All labs sheet are
  made ahead, one a turn of the UI thread after Home first shows; a tap switches first and tells the view it is shown after
  (`SectionFade`, 0 by default, fades alongside, never before). A page shown again keeps its lists when nothing changed; the
  kept chats, the models and the devices are read on worker threads. Live drawing (the Transcribe waveform, Studio's and the
  chat microphone's meters, Speak's playback meter) reads `LiveLevels`, made on the capture or playback thread one
  `LiveLevelWindow` of sound at a time (played sound shown when it plays), once a display frame from `FrameClocks`
  ("display": Android's Choreographer; "timer"), at most `LiveFrameRate`, with no allocation a frame. The settings page
  has a search over every setting (title, help, value, choices) and an App part with a restart (`IAppRestarter`; Android's
  starts the launcher activity again and ends the process), offered when a saved change waits for the next start.
- **Sound out**: `ISoundPlayer` (`SoundPlayer`, registered by `UseIdrak`) plays a recording or blocks as they come
  through the library's `AudioPlaybacks` (aaudio on Android), held to the sound's pace by the output's back-pressure,
  drained at the end; one sound at a time.
- **Read text** (OCR) page, the photo first and large: a photo taken with the camera (the permission asked first) or picked from the gallery through MAUI's
  media picker, its bytes as they came, decoded once by the library's own codecs and turned upright by its EXIF
  orientation (HEIC is refused, naming the formats the library reads); an optional preprocessing pipeline
  (`ImageTransformPipeline` text, such as `grayscale,max_width=1024,contrast=1.5`) run on it once, and the page shown as
  the model will see it (shrunk for the screen, PNG made once). A vision-language model is picked from the store's
  complete models whose architecture a registered vision family reads, loaded once through `IVisionReaderService`
  (host `vision`) after the same measured check as Chat (the decoder's weights by format, the vision part's, the
  attention cache for the context; refused with the limit named), and reads through the library's `ImageReader`: the
  page is handed over as prepared (`ImageReader.Keep`), so a read decodes and transforms nothing; the answer streams
  with its tokens per second, shown at most once a progress interval; Stop keeps what came; cold and warm apart; copy and
  share. The instruction (the library's reading pages' own by default), the system text, the pipeline, the answer's
  length, the repetition stop, the formats and the context are settings under `idrak.ocr.` (an app gives its own
  defaults by registering an `OcrLabOptions`). The library registers no family: the app does (Idrak.Mobile: Gemma 3
  vision, Qwen-VL, LLaVA).
- **Image** page: a photo as above; the registered image transforms as chips (name, summary, inputs) that add a step to
  the pipeline's box, the scan profiles to draw steps from (a seed setting repeats a draw); the pipeline run on the
  decoded photo step by step with each step's time, the first run of a transform in the process (cold) apart from the
  warm runs; the original and the result shown; the result saved (in the cache folder's `images`) and shared as PNG or
  JPEG (its quality a setting), or another registered encoder. Settings under `idrak.image.`.
- **Services**: models kept loaded through `ModelHost` and released on the phone's memory pressure, the model store
  (pull with progress, list, sizes, remove), device information (the Vulkan driver in the app's process, its measured
  width and subgroup sizes, the memory budget, CPU features), permissions and sharing a file out.
- **Planned features**: a feature not built yet is registered with its real name, title and place in the menu
  (`Order`) as `FeatureState.Planned`, with a line on what is coming (`Coming`); the library's placeholder page shows
  that and what it will need, and the menu marks it "planned". The library's still planned one is Search (`PlannedFeature`); its phase registers the built feature under the same name, so the menu does
  not change. A feature that sets no order or state is ready and takes the library's place of its name, else comes last.
- **`maui`** progress reporter: the library's progress (downloads, loads, runs) as a bindable model, updated on the UI
  thread at most once a progress interval.
- **Themes**: `IdrakStyles` merged by the app (with Idrak's icons, `IdrakIcons`: the design's 24-unit stroke icons as
  path geometries, drawn by `IconView`), with a light and a dark colour dictionary (the design's palette; text at 4.5:1 or
  more on its grounds) and gradients as brushes, swapped by `IdrakThemes.Apply`; Inter and JetBrains Mono by alias (the
  app bundles them). No page sets a colour or a size but through a key; every key may be overridden by the app. Motion
  is light and only on screen: a cross-fade between sections, the record button's pulse, the streaming caret, the
  carousel.
