# Idrak.Samples.ArabicOcr

Text from document images, Arabic first (any script works the same way). An application built on Idrak's public API
(plan 13, "The OCR sample"): the library has nothing OCR-specific; this app brings the pieces together.

## Two readers

| Reader | What it is | Use it for |
|---|---|---|
| `--reader vlm` | A Gemma 3 vision-language checkpoint (for example `bakrianoo/arabic-legal-documents-ocr-1.0`), through the Gemma 3 plug-in in `samples/Gemma3Vision`, which this app registers itself (no `-P`). Optionally with a LoRA adapter from `idrak tune` (plan 12), whose saved image preparation (`tuning_images.json`) is applied. Greedy decoding; the text streams as it is generated. | Reading whole pages now, with no training of your own; structured answers (the legal-documents fine-tune answers in JSON); drafting the transcriptions you train the line reader on. Needs a GPU for speed (4B parameters). |
| `--reader lines` | The app's own line recognizer: the page cut into lines (horizontal projection on the binarized page, small skew undone), each line read by a small CNN whose pooling ends at height 1, a bidirectional LSTM and CTC, decoded by the library's `CtcDecoders` ("greedy" or "beam"). Trained here, on your lines. | Fast, small, offline reading of one kind of document once you have a few thousand transcribed lines; runs well on the CPU. |

## Build and run

From PowerShell (the repository at `D:\Projects\Idrak`):

```powershell
cd D:\Projects\Idrak
dotnet build -c Release samples\Idrak.Samples.ArabicOcr
$ocr = "D:\Projects\Idrak\samples\Idrak.Samples.ArabicOcr\bin\Release\net10.0\Idrak.Samples.ArabicOcr.exe"
& $ocr --help
```

Every command takes `--device cpu`, `--device cuda`, `--device vulkan` (or `cuda:1`, `vulkan:0`, `auto`), `-j` for JSON
and `--help`. Nothing is sized for a particular card: the line reader's batches are measured on the device (the first
batch is one line; its peak memory against half the free memory the device reports sets the rest; a batch the device has
no room for is halved), and training splits a step into micro-batches when the device runs out of memory.

## The owner's path: the model card's own data

The data of `bakrianoo/arabic-legal-documents-ocr-1.0` (its `data` folder on Hugging Face) is LlamaFactory ShareGPT: one
record per scanned page, an `<image>` placeholder and an `images` list, the assistant's answer being the page's content as
JSON text. The folder holds:

| File | What |
|---|---|
| `downloaded_images.zip` (160 MB) | the page scans. Hugging Face flags it "pickle"; the app never unpacks it: the library's zip reader opens only the image entries the data names, read in place, and no other entry (a `.pkl` included) is ever opened. |
| `train.json` (18.1 MB) | training pages (ShareGPT, a JSON array) |
| `val.json` (306 kB) | held-out pages |
| `ocr-images-sft.jsonl` (19.9 MB) | the same data as JSON Lines; read the same way (the format, "messages" or "sharegpt", is detected) |

Download the model and its data folder (the browser's "download" on each file of the model page works; or, with
Python's `huggingface_hub` installed, `hf download`):

```powershell
hf download bakrianoo/arabic-legal-documents-ocr-1.0 --local-dir D:\Models\arabic-legal-documents-ocr-1.0
hf download bakrianoo/arabic-legal-documents-ocr-1.0 --include "data/*" --local-dir D:\Data\arabic-legal-ocr
$model = "D:\Models\arabic-legal-documents-ocr-1.0"
$data = "D:\Data\arabic-legal-ocr\data"
$zip = "$data\downloaded_images.zip"
```

**1. How well does the vision-language model read the held-out pages?** Page by page against `val.json`'s answers;
`--truth-text values` compares the text values of the JSON answers (in order) instead of the raw JSON:

```powershell
& $ocr eval "$data\val.json" --images $zip --reader vlm --model $model --grayscale --truth-text values --device cuda -o D:\Data\arabic-legal-ocr\vlm-val.txt
```

(The data's own prompt goes to the model with each page; `--prompt` replaces it. `--limit 20` scores the first 20 pages.)

**2. Fine-tune it on `train.json` (optional).** That is the library's `idrak tune` (plan 12; its phase 5 has the commands
for this model). This prints them for your folders:

```powershell
& $ocr tune-vlm --model $model --data "$data\train.json" --eval "$data\val.json" --images $zip --grayscale -o D:\Models\arabic-legal-lora
```

Then add `--adapter D:\Models\arabic-legal-lora` to every `--reader vlm` / `--prefill vlm` command below.

**3. Cut the training pages into lines, each with a draft snapped to the page's known text.** Each line is read by the
vision-language model, and that reading is matched to the best span of the page's answer (edit distance, in order from
the top of the page); the span is written as the line's draft. Start with a few hundred pages:

```powershell
$lines = "D:\Data\arabic-legal-ocr\lines"
& $ocr cut "$data\train.json" --images $zip -o $lines --prefill vlm --model $model --grayscale --limit 300 --device cuda
```

A data file given to `cut` is its own truth. Scans of your own find their page in a data file by their image's bytes:
`& $ocr cut D:\Scans -o $lines --prefill vlm --model $model --truth "$data\train.json" --images $zip`.

**4. Review.** In `$lines`, each `NAME-line-NN.png` has one of:

| File | Meaning | Used by `train` |
|---|---|---|
| `NAME-line-NN.txt` (filled) | corrected, accepted text | always |
| `NAME-line-NN.txt` (empty) | not transcribed yet | no |
| `NAME-line-NN.aligned.txt` | draft snapped to the page's known text: mostly right, check it | with `--include-drafts` |
| `NAME-line-NN.draft.txt` | the vision-language reader's reading as it is (no known text, or no span close enough) | with `--include-drafts` (after an aligned one) |

Accept the aligned drafts in one go, then correct what is wrong:

```powershell
& $ocr accept $lines
Get-ChildItem $lines -Filter *.draft.txt | Select-Object -First 20 | ForEach-Object { notepad $_.FullName }
Rename-Item "$lines\train-0001-line-03.draft.txt" "train-0001-line-03.txt"
```

(Correct a draft, save it, rename it to `.txt`. `& $ocr accept $lines --drafts` accepts every remaining draft as it is.)
`cut` never overwrites a filled `.txt`, keeps existing drafts, and gives the same line images when run again; a
NAME is the scan's file name, or `train-0001` for record 1 of `train.json`.

**5. Held-out lines** from the validation pages, the same way:

```powershell
$heldOut = "D:\Data\arabic-legal-ocr\lines-val"
& $ocr cut "$data\val.json" --images $zip -o $heldOut --prefill vlm --model $model --grayscale --device cuda
& $ocr accept $heldOut
```

**6. Train the line recognizer** on the accepted lines, scored on the held-out ones every epoch (the best epoch is kept):

```powershell
& $ocr train --data $lines --eval $heldOut -o D:\Models\arabic-lines --device cuda
& $ocr train --data $lines --eval $heldOut -o D:\Models\arabic-lines --device cpu
```

**7. Compare both readers** on the held-out lines and on the validation pages:

```powershell
& $ocr eval $heldOut --reader lines --model D:\Models\arabic-lines --single-line
& $ocr eval $heldOut --reader vlm --model $model --grayscale --truth-text values --device cuda
& $ocr eval "$data\val.json" --images $zip --reader lines --model D:\Models\arabic-lines --truth-text values
& $ocr eval "$data\val.json" --images $zip --reader vlm --model $model --grayscale --truth-text values --device cuda
```

A page's JSON answer lists its values in the JSON's order, which need not be the page's top-to-bottom order, so the line
reader's page score can be worse than its line score for that reason alone; the held-out lines compare the readers on
equal terms.

## Scans only, no text

The same steps with your own scans in `D:\Scans`:

```powershell
& $ocr read D:\Scans --reader vlm --model $model --grayscale -o D:\Scans\vlm-text --device cuda
& $ocr cut D:\Scans -o D:\Scans\lines --prefill vlm --model $model --grayscale --device cuda
```

then correct each `.draft.txt` and rename it to `.txt` (step 4), keep a tenth of the pages' lines apart as held-out lines
(move their `.png` and `.txt` to another folder), train (step 6) and compare (step 7). Without `--prefill`, `cut` writes an
empty `.txt` beside each line for you to type into.

## Reading text

```powershell
& $ocr read D:\Scans\page-001.png --reader lines --model D:\Models\arabic-lines
& $ocr read D:\Scans --reader lines --model D:\Models\arabic-lines --decoder beam --beam-width 10 -j -o D:\Scans\json
& $ocr read D:\Scans\page-001.png --reader vlm --model $model --adapter D:\Models\arabic-legal-lora --pan-and-scan --device cuda
```

Text is written as UTF-8 without a byte order mark, lines ending as the console's. With `-j`, the line reader gives each
line's box (in the deskewed page), text and score.

## Right to left

The recognizer reads a line image's columns one after another. For a right-to-left script (`--direction rtl`, the
default) each line image is flipped horizontally before the network sees it, so its columns run in reading order, and the
labels are the transcription in logical Unicode order (as typed and stored: NFC, white space as single spaces). Inside a
right-to-left line, a run of digits or Latin letters is drawn left to right, so the flipped image shows it reversed; such
runs are reversed in the labels for training and back after decoding, and the output is logical order throughout. Use
`--direction ltr` for left-to-right scripts (no flip).

## Training details

- Data: line images (PNG, JPEG, BMP, PGM) with UTF-8 `.txt` beside them; drafts only with `--include-drafts`.
- Alphabet: the training transcriptions' characters (Unicode scalars, NFC), saved with the model; a held-out or `eval`
  text with characters the model has never seen is reported (and counted as errors), not a crash.
- Network: three or more convolution blocks (`--channels 32,64,128`), 2x2 pooling twice then 2x1 down to height 1, a
  bidirectional LSTM (`--hidden 128`, `--layers 2`), a linear layer to the alphabet and CTC's blank, built with the
  library's `NetworkBuilder`. Lines are scaled to `--height` (48) and keep their width; a batch is padded to its widest
  line and each line's length goes to `Losses.Ctc`.
- Augmentations through the library's `Augmentations` registry, mild by default (`--augment`; `none` for none):
  `shift(pixels=2), affine(degrees=1, translate=0.01, scale=0.95:1.05, shear=3), color-jitter(brightness=0.2, contrast=0.3), noise(std=0.03)`
  (`noise` is the app's own registration: the registry has none).
- The model folder: `recognizer.ikm` (the network's description and the best epoch's weights) and `recognizer.json`
  (alphabet, height, direction, sizes, preprocessing, and the run: every epoch's loss, CER and WER).

## Tests

```powershell
$env:IDRAK_DEVICES="cpu"; dotnet run -c Release --project samples\Idrak.Samples.ArabicOcr.Tests
$env:IDRAK_DEVICES="cuda"; dotnet run -c Release --project samples\Idrak.Samples.ArabicOcr.Tests
$env:IDRAK_DEVICES="vulkan"; dotnet run -c Release --project samples\Idrak.Samples.ArabicOcr.Tests
```

Generated data only: eight Arabic letters drawn as strokes with their dots (no fonts), pages of such lines, a ShareGPT file
with its images in a zip, and the repository's tiny Gemma 3 (random weights: the vision-language reader's plumbing, not its
text). `$env:IDRAK_FILTER="ocr train"` runs the tests whose name contains it.
