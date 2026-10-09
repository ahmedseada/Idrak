# Copyright (c) 2026 Ahmed Seada
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.
"""Reads a scan with a Gemma 3 vision-language model in transformers and saves the answer (plan 11, image transforms).

The Python twin of `idrak run MODEL --image SCAN --image-transform ...`: the same image transforms, done in Pillow
(tools/vlm/image_transforms.py, which Idrak's transforms match byte for byte), then the model's own processor, then
generate(). For bakrianoo/arabic-legal-documents-ocr-1.0 its card's whole pipeline is

  python tools/vlm/ocr_transformers.py MODEL scan.jpg --image-transform grayscale,max_width=1024,contrast=1.5 --out answer.txt

(the card's prompt "Extract details to JSON." is the default; the card's OpenAI/vLLM path also re-encodes the grey image
as JPEG quality 95 before base64: add ",jpeg=95"). Writes the answer to --out as UTF-8 (no BOM) and, beside it, a .json
of figures: the image's sizes before and after the transforms, the prompt's tokens, the image blocks, the timings, the
tokens per second, the dtypes, the device, the versions and whether the answer parses as JSON (and, when the
json_repair package is installed, the repaired document, as the card parses the model's output with json_repair.loads).

Decoding: greedy by default (do_sample=False, repetition penalty 1: what `idrak run --temperature 0` does, for
comparing); --sample uses the model's generation_config.json instead (the card's own example calls generate() without
do_sample, and the fine-tune's config samples: do_sample, top_k 64, top_p 0.95), with --seed for a repeatable run.

Precision: --dtype bfloat16 (default) for the whole model; --vision-float32 keeps the vision tower and the projector in
float32 (what Idrak does: its encoder is float32); --dtype float32 for everything (17 GB; --device cpu or a large card).

Needs: torch, transformers (4.50 or later), pillow, numpy; json_repair optional.

Examples (PowerShell or bash; a non-ASCII prompt is safer in a UTF-8 file given with --prompt-file):
  python tools/vlm/ocr_transformers.py MODEL scan.jpg --image-transform grayscale,max_width=1024,contrast=1.5 --out a.txt
  python tools/vlm/ocr_transformers.py MODEL scan.jpg --image-transform grayscale,max_width=1024,contrast=1.5,jpeg=95 --sample --seed 1 --out b.txt
  python tools/vlm/ocr_transformers.py MODEL scan.jpg --grayscale --pan-and-scan --vision-float32 --prompt-file prompt.txt --out c.txt
"""
import argparse
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import image_transforms  # noqa: E402  (Pillow only)

CARD_PROMPT = "Extract details to JSON."


def log(message):
    print(message, flush=True)


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n", 1)[1])
    p.add_argument("model", help="a model folder, or a Hugging Face id (downloaded by transformers)")
    p.add_argument("image", help="the scan (PNG, JPEG, ...)")
    p.add_argument("--prompt", help=f'the prompt\'s text (default: the card\'s "{CARD_PROMPT}")')
    p.add_argument("--prompt-file", help="read the prompt from this UTF-8 file instead")
    p.add_argument("--system", help="a system message (none by default, as the card)")
    p.add_argument("--image-transform", help="image transforms in Pillow before the processor, in Idrak's syntax "
                                             "(such as grayscale,max_width=1024,contrast=1.5)")
    p.add_argument("--grayscale", action="store_true", help="the grayscale transform first (as idrak --grayscale)")
    p.add_argument("--no-exif", action="store_true", help="do not apply the EXIF orientation (idrak always does)")
    p.add_argument("--pan-and-scan", action="store_true", help="do_pan_and_scan=True: the whole image, then its crops")
    p.add_argument("--pan-and-scan-min-crop-size", type=int, help="pan_and_scan_min_crop_size (default 256)")
    p.add_argument("--pan-and-scan-max-crops", type=int, help="pan_and_scan_max_num_crops (default 4)")
    p.add_argument("--pan-and-scan-min-ratio", type=float, help="pan_and_scan_min_ratio_to_activate (default 1.2)")
    p.add_argument("--dtype", choices=["bfloat16", "float32", "float16"], default="bfloat16", help="the model's dtype")
    p.add_argument("--vision-float32", action="store_true", help="the vision tower and projector in float32 (as Idrak)")
    p.add_argument("--device", default="auto", help="auto (CUDA when available, else CPU), cpu, cuda or cuda:N")
    p.add_argument("--max-tokens", type=int, default=2048, help="the most tokens to generate (default 2048, as the card)")
    p.add_argument("--sample", action="store_true", help="sample with the model's generation_config.json (the card's example)")
    p.add_argument("--seed", type=int, help="the random seed (with --sample: a repeatable answer)")
    p.add_argument("--fast-processor", action="store_true", help="torchvision's image processor (default: Pillow's, as Idrak)")
    p.add_argument("--out", required=True, help="the answer's file (UTF-8); the figures go to the same name with .json")
    args = p.parse_args()
    if args.prompt is not None and args.prompt_file is not None:
        p.error("give the prompt with --prompt or --prompt-file, not both")
    try:
        image_transforms.parse(args.image_transform)
    except ValueError as e:
        p.error(f"--image-transform: {e}")
    return args


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    args = parse()
    import numpy as np
    import torch
    import transformers
    import PIL
    from PIL import Image, ImageOps
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    prompt = (open(args.prompt_file, encoding="utf-8-sig").read().strip() if args.prompt_file
              else args.prompt if args.prompt is not None else CARD_PROMPT)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    dtypes = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}
    dtype = dtypes[args.dtype]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        log("CUDA is not available to this torch build: running on the CPU")
        device = "cpu"

    # The image: upright by its EXIF orientation, then the transforms (grayscale first with --grayscale).
    with Image.open(args.image) as opened:
        image = opened if args.no_exif else ImageOps.exif_transpose(opened)
        image = image.convert("L") if image.mode in ("L", "LA", "1") else image.convert("RGB")
        image.load()
    original = list(image.size)
    pipeline = image_transforms.describe(args.image_transform)
    if args.grayscale and not any(n == "grayscale" for n, _, _ in image_transforms.parse(pipeline)):
        pipeline = "grayscale" + ("," + pipeline if pipeline else "")
    started = time.time()
    image = image_transforms.apply(image, pipeline)
    transform_ms = (time.time() - started) * 1000
    transformed = {"width": image.size[0], "height": image.size[1], "mode": image.mode}
    image = image.convert("RGB")                                         # what the processor's convert_rgb does
    log(f"image {args.image}: {original[0]} x {original[1]}"
        + (f" -> {transformed['width']} x {transformed['height']} {transformed['mode']} ({pipeline}, {transform_ms:.0f} ms)" if pipeline else ""))

    # The model.
    version = tuple(int(x) for x in transformers.__version__.split(".")[:2] if x.isdigit())
    log(f"loading {args.model} ({args.dtype}{', vision float32' if args.vision_float32 else ''}) on {device} "
        f"with transformers {transformers.__version__}, torch {torch.__version__}")
    started = time.time()
    model = Gemma3ForConditionalGeneration.from_pretrained(args.model, **{"dtype" if version >= (4, 56) else "torch_dtype": dtype})
    model = model.to(device).eval()
    inner = model.model if hasattr(model, "model") and hasattr(model.model, "vision_tower") else model
    vision_dtype = torch.float32 if args.vision_float32 else dtype
    if vision_dtype != dtype:
        inner.vision_tower.to(vision_dtype)
        inner.multi_modal_projector.to(vision_dtype)
    load_seconds = time.time() - started
    if version >= (5, 0):
        processor = AutoProcessor.from_pretrained(args.model, backend="torchvision" if args.fast_processor else "pil")
    else:
        processor = AutoProcessor.from_pretrained(args.model, use_fast=args.fast_processor)
    if getattr(processor, "chat_template", None) is None and getattr(processor.tokenizer, "chat_template", None):
        processor.chat_template = processor.tokenizer.chat_template

    messages = []
    if args.system:
        messages.append({"role": "system", "content": [{"type": "text", "text": args.system}]})
    messages.append({"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]})
    vision_options = {}
    if args.pan_and_scan:
        vision_options["do_pan_and_scan"] = True
        for name, value in [("pan_and_scan_min_crop_size", args.pan_and_scan_min_crop_size),
                            ("pan_and_scan_max_num_crops", args.pan_and_scan_max_crops),
                            ("pan_and_scan_min_ratio_to_activate", args.pan_and_scan_min_ratio)]:
            if value is not None:
                vision_options[name] = value
    if vision_options:
        text_messages = [{"role": m["role"], "content": [{k: v for k, v in c.items() if k != "image"} for c in m["content"]]}
                         for m in messages]
        rendered = processor.apply_chat_template(text_messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=rendered, images=[[image]], return_tensors="pt", add_special_tokens=False, **vision_options)
    else:
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt")
    prompt_tokens = int(inputs["input_ids"].shape[1])
    blocks = int(inputs["pixel_values"].shape[0])
    model_inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}
    model_inputs["pixel_values"] = model_inputs["pixel_values"].to(vision_dtype)

    decoding = {"max_new_tokens": args.max_tokens}
    if not args.sample:
        decoding.update(do_sample=False, num_beams=1, repetition_penalty=1.0, top_k=None, top_p=None)
    log(f"prompt: {prompt_tokens} tokens, {blocks} image block{'s' if blocks != 1 else ''}; "
        + ("sampling with the model's generation_config" if args.sample else "greedy"))
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.time()
    with torch.inference_mode():
        out = model.generate(**model_inputs, **decoding)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    seconds = time.time() - started
    generated = out[0, prompt_tokens:]
    answer = processor.decode(generated, skip_special_tokens=True)
    n = int(generated.shape[0])
    log(f"generated {n} tokens in {seconds:.1f} s ({n / seconds if seconds else 0:.1f} tokens/s)")

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(answer)
    figures = {
        "model": args.model, "image": os.path.abspath(args.image), "image_size": original,
        "image_transforms": pipeline or None, "transformed": transformed, "transform_ms": round(transform_ms, 1),
        "system": args.system, "prompt": prompt, "prompt_tokens": prompt_tokens, "image_blocks": blocks,
        "vision_options": vision_options, "decoding": "sample (generation_config)" if args.sample else "greedy",
        "seed": args.seed, "generation_config": model.generation_config.to_dict() if model.generation_config else None,
        "generated_tokens": n, "seconds": round(seconds, 3), "tokens_per_second": round(n / seconds, 2) if seconds else None,
        "load_seconds": round(load_seconds, 1), "dtype": args.dtype, "vision_dtype": str(vision_dtype).replace("torch.", ""),
        "device": device, "device_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "image_processor_class": type(processor.image_processor).__name__,
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
                     "pillow": PIL.__version__, "numpy": np.__version__, "platform": platform.platform()},
    }
    stripped = answer.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        json.loads(stripped)
        figures["json"] = "valid"
    except ValueError:
        figures["json"] = "invalid"
        try:
            import json_repair
            figures["json"] = "repaired (json_repair)"
            figures["repaired"] = json_repair.loads(stripped)
        except ImportError:
            pass
    figures_path = os.path.splitext(args.out)[0] + ".json"
    if os.path.abspath(figures_path) == os.path.abspath(args.out):
        figures_path = args.out + ".figures.json"
    with open(figures_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(figures, f, indent=1, ensure_ascii=False, default=str)
        f.write("\n")
    log(f"wrote {os.path.abspath(args.out)} and {os.path.abspath(figures_path)} (JSON: {figures['json']})")


if __name__ == "__main__":
    main()
