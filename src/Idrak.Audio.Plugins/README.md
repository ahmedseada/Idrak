# Idrak.Audio.Plugins

The speech and voice model families: recognition, synthesis, voice conversion and phonemizers. The library (`Idrak`,
`Idrak.Audio`) keeps only generic defaults and fallbacks; the families live here, one namespace each
(`Idrak.Audio.Plugins.<Family>`), written on the library's public API only. It builds on Idrak.Audio (and through it
Idrak).

```csharp
using Idrak.Audio.Plugins;

AudioPlugins.RegisterAll();      // every family of the package; calling it again changes nothing
```

- `AudioPlugins.RegisterAll()` registers every family, `AudioPlugins.UnregisterAll()` takes them back (the registries
  are left as they were found), `AudioPlugins.All` lists them.
- One family alone: its namespace's static `<Family>Plugin` class, with `Register()` and `Unregister()`.
- Each family is an `IIdrakPlugin` (`Idrak.Abstraction.Plugins`), recorded by name in `IdrakPlugins`; a plug-in
  registered later under the same name takes its place.
- The `idrak` tool registers this package at startup, before any `-P` plug-in (which can still take a family's place);
  `RegisterIdrakPlugin()` is the `-P` entry, for a host that loads `Idrak.Audio.Plugins.dll` itself.

## The families

| Namespace | Plug-in (`IdrakPlugins` name) | Registers | Came from |
|---|---|---|---|
| `Idrak.Audio.Plugins.Whisper` | `WhisperPlugin` ("whisper") | `SpeechFamilies`: `WhisperForConditionalGeneration` | `samples/WhisperSpeech` |
| `Idrak.Audio.Plugins.EspeakNg` | `EspeakNgPlugin` ("espeak-ng") | `TextPhonemizers`: "espeak-ng", which runs the espeak-ng program the user installs (`IDRAK_ESPEAK`, `IDRAK_ESPEAK_VOICE`) | `src/Idrak.Audio` |
| `Idrak.Audio.Plugins.Vits` | `VitsPlugin` ("vits") | `SpeechSynthesisFamilies`: `VitsModel`; `TextRomanizers`: "uroman-arabic"; the espeak-ng plug-in when missing | `samples/VitsSpeech`, uroman's romanizer from `src/Idrak.Audio` |
| `Idrak.Audio.Plugins.OpenVoice` | `OpenVoicePlugin` ("openvoice") | `VoiceConverters`: "openvoice" (its tensor names through the library's `RenamedTensorStore`) | `samples/OpenVoice` |
| `Idrak.Audio.Plugins.IdrakAudio` | `IdrakAudioPlugin` ("idrak-audio") | `SpeechFamilies` and `SpeechSynthesisFamilies`: `IdrakAudio`; `PretrainedArchitectures` and `AudioFamilies`: `IdrakAudioForConditionalGeneration` | `samples/IdrakAudio` |

Each family's folder has its README. The library itself registers no speech family, no voice converter, no romanizer
and no phonemizer, and starts no outside program for text: the espeak-ng phonemizer is this package's only one.
`samples/WhisperSpeech/transcribe.cs` and `samples/OpenVoice/voice.cs` are apps on this package.

A model whose family is not registered is refused with "not registered", naming this package.

License: Apache 2.0.
