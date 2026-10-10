# Plan 15: the Arabic legal OCR app (two Gemma 3 readers, fine-tuning in the app)

The owner's request (October 2026): a new app that reads Arabic legal scans with two models, switched on the reading
page, and fine-tunes our own model from the app with live progress.

- **bakrianoo's model**: `bakrianoo/arabic-legal-documents-ocr-1.0`, Gemma-3-4B-IT fine-tuned on Arabic legal documents
  (a full checkpoint; the Gemma 3 vision plug-in reads it as it is).
- **Ours**: `google/gemma-3-4b-it` plus our LoRA adapter, tuned in the app on the owner's arabic-legal-ocr data
  (`train.json`, `val.json`, `downloaded_images.zip`), CER scored on the evaluation set while it trains.

Decisions (the owner's answers): a local web app (ASP.NET Core, a page in `wwwroot`, like `Idrak.Samples.GptApi`); the
two models with a switch on the inference page; reading and fine-tuning in the app. **The main goal is bakrianoo's model
run with its own preprocessing; our model is an extra after it.**

bakrianoo's preprocessing (its model card, "Mandatory Image Preprocessing"): grayscale, shrink to at most 1024 pixels
wide keeping the aspect ratio (LANCZOS), `ImageEnhance.Contrast(1.5)`; the vLLM route sends it as a JPEG at quality 95
(base64), the Transformers route passes the image itself. The library's Pillow-exact transforms give it as the pipeline
`grayscale,max_width=1024,contrast=1.5` (`jpeg=95` added for the vLLM route). Prompt `Extract details to JSON.`, bfloat16
weights, up to 2048 new tokens, the answer parsed with `json_repair` (the app repairs the JSON the same way: unclosed
strings and brackets, trailing or missing commas, single quotes, unquoted keys, Python literals, code fences).

## Steps

| # | Step | State |
|---|---|---|
| 0 | **bakrianoo's model with its own preprocessing** in the app: download, load (bfloat16), read with the card's pipeline and prompt, the answer streamed and its JSON repaired, the preprocessed image shown, CER on the evaluation pages | **Done** (CPU tests with the tiny Gemma 3; the page driven in headless Chromium); to run on the owner's card |
| 0b | **App-agnostic parts in the library** (the owner: "any app agnostic logic or contract or plugin move to lib"): `JsonRepairs` (contract and registry; the library's lenient repair, as json_repair), `AnswerTexts` (contract and registry: raw, values; the shared normalization), `ModelImages` (moved from the CLI) and `ImageReader` (a vision-language model reading a page, greedy, streamed; ArabicOcr's reader and the app use it), `ModelHost.MaxLoaded` (one large model on a device). The app keeps its settings, endpoints and page | **Done** |
| 1 | **Library: conversation tuning** (extra, after 0) (`Idrak.Nlp.ConversationTuning`). The CLI's image-conversation tuning glue (`TuneTool`: read the data files through `TuningDataFormats` with their images, hold out an evaluation share, encode with `ChatTranscriptEncoder`, the vision side through `TuningVision`, the answer scorer, train, keep trained vision parts, save the adapter with its `TuningImages` and `TuningManifest`, also when stopped) moves into the library as `Prepare` (returns the prepared run: sequences, vision, scorer, tuning options) and `Train` (progress, cancellation, saving). `idrak tune` and the app both call it; the CLI keeps its extras (answer balancing, profiling, the memory report) between the two | |
| 2 | **The app** (`samples/Idrak.Samples.LegalOcr`). One model on the GPU at a time (loading one unloads the other; tuning takes the GPU, reading waits). Settings in `appsettings.json`: the models folder, the two Hugging Face ids, the data files, the precision. Endpoints: status, models (with download), read (a scan in, the answer streamed by server-sent events), the evaluation pages (a page of `val.json` with its expected text and CER), tuning (start, stop, progress streamed, the adapters made). Page: a Read tab (model switch, drop a scan or pick an evaluation page, the answer as it is generated, time, tokens a second, CER against the expected text) and a Fine-tune tab (settings, start and stop, live step, loss, learning rate, CER, time left, the log) | |
| 3 | **Tests** (`samples/Idrak.Samples.LegalOcr.Tests`, CPU, a tiny Gemma 3 vision checkpoint made by the test): the model host loads, switches and unloads; a read streams an answer; a tuning job reports progress, stops on request, writes an adapter that the switch then lists and reads with | |
| 4 | **Owner's commands**: download both models, run the app, tune, compare | |

Rules that apply: speed and memory first (one model resident; bfloat16 by default, int8 an option; the device's
memory measured, never a card assumed), the library's telemetry and progress, contract plus registry for anything new
in the library, the API dump and inventory kept current, no family code in the library (the Gemma 3 plug-in is the
app's, as in ArabicOcr).
