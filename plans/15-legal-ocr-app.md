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
| 1 | **Library: conversation tuning** (`Idrak.Nlp.ConversationTuning`). `Prepare` reads the data files through `TuningDataFormats` (images from a folder or a zip), takes an evaluation file or a seeded held-out share, limits the conversations, builds the image side (`TuningVision` with a `FeatureCaches` cache), tokenizes with `ChatTranscriptEncoder` (stage progress, a log line per stage) and makes the answer scorer; `Train` runs `FineTuner` with per-step progress and cancellation, and saves the adapter, the trained vision parts, `tuning_images.json` and `idrak-tuning.json`, also when stopped (`ConversationTuningResult.Stopped`). `TuningAnswerScorer.Progress` reports each answer scored inside training. `idrak tune` on conversation files now calls it (keeping its balancing, profiling, distill hook and memory report between the two) | **Done** (`conversation tuning` group; `cli vision` unchanged) |
| 2 | **The app**: the switch lists bakrianoo's model, the base Gemma 3 4B and every adapter in `AdaptersFolder` (read as its base with the adapter merged, its own preprocessing and prompt). Read tab: also "Evaluate the model" (the first N evaluation pages in a row, CER per page, mean and median, time left). Fine-tune tab: the run's settings (base: a model of the switch or an adapter to continue; weights bf16/int8/int4; preprocessing; epochs, learning rate, rank, alpha, longest sequence, tokens per batch, accumulation, first N pages, CER pages and interval, answer tokens, evaluation and save intervals, the base's CER first, the projector), start, stop-and-save, live progress by server-sent events (stage, download, step, epoch, loss, evaluation loss and CER before and now, learning rate, tokens a second, elapsed, time left, device memory), a loss and CER chart, the log, the adapters with their results ("Read with it", "Continue tuning"). Tuning takes the device: the reader unloads and refuses loads and reads until it ends | **Done** (driven in headless Chromium on the tiny Gemma 3: evaluate, train, the adapter read with) |
| 3 | **Tests** (`samples/Idrak.Samples.LegalOcr.Tests`, CPU, the tiny Gemma 3): a run reports its steps, scores CER before and while training, writes the adapter, its preparation, manifest and record, and the switch reads with it; continuing an adapter; reading refused while tuning, a second run and bad settings refused, stop saves the adapter so far; the evaluation and tuning endpoints over HTTP | **Done** (7 of 7) |
| 2b | **Read tab: preprocessing for poor scans** (the owner: low-quality characters; invert helped a lot). The library's transforms grew to 19, all Pillow 12.3 byte for byte (fixtures from `make_image_transforms.py`): `scale`, `pad`, `equalize`, `gamma`, `blur` (Pillow's three box blurs), `unsharp`, `median`, `min_filter`, `max_filter`, `binarize` (fixed or Otsu) beside the earlier ones; each describes its inputs (`IImageTransform.Label`, `Parameters`: kind, range, choices). The app's suggested values are its own (`LegalOcr:PreprocessingDefaults`). The page: every step its own row with its own inputs (a slider with a number, or a list), tick to run, move up or down, remove, "Add a step"; start from the model's steps, the model card's, the vLLM route's, none, or your own (kept across models and reloads); a live preview of the page as the model will see it (`/api/preview`, no model needed); one full-width Read button at the top of the settings, pinned while scrolling, Stop while reading | **Done** (driven in headless Chromium) |
| 4 | **Owner's commands**: run the app on the card, evaluate bakrianoo's model, tune our adapter, compare | Given |

Rules that apply: speed and memory first (one model resident; bfloat16 by default, int8 an option; the device's
memory measured, never a card assumed), the library's telemetry and progress, contract plus registry for anything new
in the library, the API dump and inventory kept current, no family code in the library (the Gemma 3 plug-in is the
app's, as in ArabicOcr).
