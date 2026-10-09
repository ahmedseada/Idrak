# Copyright (c) 2026 Ahmed Seada
# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.
"""transformers' side of a real-model comparison for a Gemma 3 vision-language model (plan 11, phase 7).

Runs a `Gemma3ForConditionalGeneration` (a local folder or a Hugging Face id) on one image and one prompt and saves
what `idrak vlm check MODEL --reference DIR` compares against:

  input_ids.npy                 int64 [1, L]: the whole prompt as the processor builds it (image tokens included)
  pixel_values.npy              float32 [B, 3, S, S]: the processor's pixels, B = 1 (with --pan-and-scan: 1 + its crops,
                                the whole image first)
  vision_last_hidden_state.npy  float32 [B, rows, width]: the vision tower's output (after post_layernorm); all 4,096
                                rows by default, the first --vision-rows only when given (to keep the folder small)
  image_features.npy            float32 [B, tokens, text width]: the projector's output (the soft tokens' embeddings)
  generated_ids.npy             int64 [N]: N greedy tokens from generate() (no sampling, repetition penalty 1)
  gen_top_ids.npy, gen_top_logits.npy
                                int64 / float32 [N, 5]: the five best ids and their logits at each generate() step
                                (the logits greedy decoding chose from; the margin is column 0 minus column 1)
  tf_top_ids.npy, tf_top_logits.npy
                                int64 / float32 [L + N, 5]: the same from one teacher-forced pass over prompt and answer
                                (row p predicts token p + 1; the image block bidirectional, as generate's prefill)
  gen_full_logits.npy           float32 [K, vocabulary]: the whole logit rows of the first K generate() steps (--full-rows)
  manifest.json                 versions, dtypes, device, grayscale flag, processor class, prompt, rendered text, the
                                answer's text, the model's generation_config, and checks (teacher-forced top-1 against
                                generate's tokens)

Needs: torch, transformers (4.50 or later), pillow, numpy; accelerate for --offload. Windows, Linux and macOS; CUDA when
available, else the CPU.

Choosing the precision (the logits' last bits decide near-ties, so say which one you ran):
  --dtype bfloat16                         the whole model in bfloat16 (as most people run it; 8.6 GB on the GPU)
  --dtype bfloat16 --vision-dtype float32  the decoder in bfloat16, the vision tower and projector in float32 (what
                                           `idrak run -w bf16` does: Idrak's encoder is float32); fits a 12-16 GB card
                                           (the weights are loaded in --dtype, then the vision side is cast: exact for
                                           a checkpoint stored in bfloat16, as the real model is)
  --dtype float32                          everything in float32 (17 GB of weights): on the CPU (--device cpu, about
                                           20 GB of RAM), or on a smaller GPU with --offload (device_map="auto": the
                                           layers that do not fit stay in CPU memory; slow but exact)

Pan and scan (Gemma 3's crops of a tall or wide page; transformers' do_pan_and_scan, off by default): --pan-and-scan
turns it on with Gemma3Processor's defaults (crops of 256 pixels at least, 4 at most, from an aspect ratio of 1.2);
--pan-and-scan-min-crop-size, --pan-and-scan-max-crops and --pan-and-scan-min-ratio change them. The manifest records
them as "vision_options" (transformers' names), which `idrak vlm check` gives the encoder.

Image transforms (a fine-tune's own preparation of a scan, done in Pillow before the processor): --image-transform
takes Idrak's syntax ("grayscale,max_width=1024,contrast=1.5"; see tools/vlm/image_transforms.py), applied after the
EXIF orientation (and after --grayscale's grey). The manifest records it as "image_transforms", which `idrak vlm check`
runs on the image the same way (the library's transforms give Pillow's bytes exactly), so pixels compare like for like.

Examples (PowerShell or bash; a prompt with non-ASCII text is safer in a UTF-8 file given with --prompt-file):
  python tools/vlm/compare_real.py MODEL scan.jpg --grayscale --prompt-file prompt.txt --out ref-bf16
  python tools/vlm/compare_real.py MODEL scan.jpg --grayscale --prompt-file prompt.txt --vision-dtype float32 --out ref-v32
  python tools/vlm/compare_real.py MODEL scan.jpg --grayscale --prompt-file prompt.txt --dtype float32 --device cpu --out ref-f32
  python tools/vlm/compare_real.py MODEL scan.jpg --grayscale --prompt-file prompt.txt --pan-and-scan --out ref-pas
  python tools/vlm/compare_real.py MODEL scan.jpg --prompt "Extract details to JSON." --image-transform grayscale,max_width=1024,contrast=1.5 --out ref-card
  idrak vlm check MODEL --reference ref-bf16 -d cuda:0 -w bf16
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import image_transforms  # noqa: E402  (Pillow only)

TOP = 5


def log(message):
    print(message, flush=True)


def dump_json(path, value):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=1, ensure_ascii=False, default=str)
        f.write("\n")


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n", 1)[1])
    p.add_argument("model", help="a model folder, or a Hugging Face id (downloaded by transformers)")
    p.add_argument("image", help="the image file")
    p.add_argument("--prompt", help="the prompt's text (after the image in the user message)")
    p.add_argument("--prompt-file", help="read the prompt from this UTF-8 file instead (safer for Arabic on Windows)")
    p.add_argument("--system", help="a system message (none by default, as idrak run without -s)")
    p.add_argument("--out", required=True, help="the folder to write (created)")
    p.add_argument("--grayscale", action="store_true",
                   help='turn the image grey first: ImageOps.exif_transpose, convert("L"), convert("RGB") (as idrak --grayscale)')
    p.add_argument("--no-exif", action="store_true", help="do not apply the EXIF orientation (idrak always does)")
    p.add_argument("--image-transform", help="image transforms in Pillow before the processor, in Idrak's syntax "
                                             "(such as grayscale,max_width=1024,contrast=1.5)")
    p.add_argument("--dtype", choices=["bfloat16", "float32", "float16"], default="bfloat16", help="the model's dtype")
    p.add_argument("--vision-dtype", choices=["bfloat16", "float32", "float16"],
                   help="the vision tower's and projector's dtype (default: --dtype)")
    p.add_argument("--device", default="auto", help="auto (CUDA when available, else CPU), cpu, cuda or cuda:N")
    p.add_argument("--offload", action="store_true",
                   help='device_map="auto": layers that do not fit on the GPU stay in CPU memory (needs accelerate)')
    p.add_argument("--steps", type=int, default=64, help="greedy tokens to generate (default 64)")
    p.add_argument("--full-rows", type=int, default=8, help="whole logit rows saved for the first K steps (default 8)")
    p.add_argument("--vision-rows", type=int, default=0, help="save only the first N rows of the vision output (0: all)")
    p.add_argument("--attn", help="attn_implementation (eager, sdpa); default: transformers' choice")
    p.add_argument("--fast-processor", action="store_true", help="use_fast=True (torchvision); default use_fast=False (Pillow)")
    p.add_argument("--pan-and-scan", action="store_true", help="do_pan_and_scan=True: the whole image, then its crops")
    p.add_argument("--pan-and-scan-min-crop-size", type=int, help="pan_and_scan_min_crop_size (default 256)")
    p.add_argument("--pan-and-scan-max-crops", type=int, help="pan_and_scan_max_num_crops (default 4)")
    p.add_argument("--pan-and-scan-min-ratio", type=float, help="pan_and_scan_min_ratio_to_activate (default 1.2)")
    args = p.parse_args()
    if args.image_transform:
        try:
            image_transforms.parse(args.image_transform)
        except ValueError as e:
            p.error(f"--image-transform: {e}")
    if (args.prompt is None) == (args.prompt_file is None):
        p.error("give the prompt with --prompt TEXT or --prompt-file FILE (one of them)")
    return args


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")       # a legacy Windows code page never stops the run
    args = parse()
    import torch
    import transformers
    import PIL
    from PIL import Image, ImageOps
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    prompt = args.prompt if args.prompt is not None else open(args.prompt_file, encoding="utf-8-sig").read().strip()
    os.makedirs(args.out, exist_ok=True)
    dtypes = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}
    dtype = dtypes[args.dtype]
    vision_dtype = dtypes[args.vision_dtype or args.dtype]
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        log("CUDA is not available to this torch build: running on the CPU")
        device = "cpu"

    # The model. transformers 4.56 and later take dtype=, earlier ones torch_dtype= (an unknown keyword is not refused).
    version = tuple(int(x) for x in transformers.__version__.split(".")[:2] if x.isdigit())
    kwargs = {"dtype" if version >= (4, 56) else "torch_dtype": dtype}
    if args.attn:
        kwargs["attn_implementation"] = args.attn
    if args.offload:
        kwargs["device_map"] = "auto"
    log(f"loading {args.model} ({args.dtype}{', vision ' + args.vision_dtype if args.vision_dtype else ''}) "
        f"on {'device_map=auto' if args.offload else device} with transformers {transformers.__version__}, torch {torch.__version__}")
    started = time.time()
    model = Gemma3ForConditionalGeneration.from_pretrained(args.model, **kwargs)
    if not args.offload:
        model = model.to(device)
    model.eval()
    inner = model.model if hasattr(model, "model") and hasattr(model.model, "vision_tower") else model
    if vision_dtype != dtype:
        inner.vision_tower.to(vision_dtype)
        inner.multi_modal_projector.to(vision_dtype)
    loaded_dtype = next(inner.language_model.parameters()).dtype
    assert loaded_dtype == dtype, f"the decoder loaded as {loaded_dtype}, not {dtype}"
    first_device = model.device if not args.offload else next(model.parameters()).device
    log(f"loaded in {time.time() - started:.1f} s")

    # The image, as idrak reads it: upright by its EXIF orientation, grey when asked, then the image transforms.
    with Image.open(args.image) as opened:
        exif_orientation = opened.getexif().get(0x0112, 1)
        image = opened if args.no_exif else ImageOps.exif_transpose(opened)
        image = image.convert("L") if image.mode in ("L", "LA", "1") else image.convert("RGB")
        image.load()
    original_size = list(image.size)
    pipeline = image_transforms.describe(args.image_transform)
    steps = pipeline
    if args.grayscale and not any(name == "grayscale" for name, _, _ in image_transforms.parse(pipeline)):
        steps = "grayscale" + ("," + pipeline if pipeline else "")
    image = image_transforms.apply(image, steps).convert("RGB")
    if steps:
        log(f"image transforms {steps}: {original_size[0]} x {original_size[1]} -> {image.size[0]} x {image.size[1]}")

    # Pillow's image processor unless --fast-processor (transformers 5 names it backend=, 4.x use_fast=).
    if version >= (5, 0):
        processor = AutoProcessor.from_pretrained(args.model, backend="torchvision" if args.fast_processor else "pil")
    else:
        try:
            processor = AutoProcessor.from_pretrained(args.model, use_fast=args.fast_processor)
        except ImportError as e:
            # transformers 4.x hands use_fast=False to the tokenizer too, whose slow form needs sentencepiece (and a
            # tokenizer.model): the Pillow image processor with the fast tokenizer instead (the same ids).
            log(f"AutoProcessor(use_fast=False) failed ({str(e).strip().splitlines()[0]}); "
                "using the slow image processor with the fast tokenizer")
            from transformers import AutoImageProcessor, AutoTokenizer, Gemma3Processor
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            seq = 256
            config_file = os.path.join(args.model, "processor_config.json")
            if os.path.isfile(config_file):
                with open(config_file, encoding="utf-8") as f:
                    seq = json.load(f).get("image_seq_length", seq)
            processor = Gemma3Processor(image_processor=AutoImageProcessor.from_pretrained(args.model, use_fast=False),
                                        tokenizer=tokenizer, chat_template=tokenizer.chat_template, image_seq_length=seq)
    if getattr(processor, "chat_template", None) is None and getattr(processor.tokenizer, "chat_template", None):
        processor.chat_template = processor.tokenizer.chat_template      # kept in tokenizer_config.json only
    messages = []
    if args.system:
        messages.append({"role": "system", "content": [{"type": "text", "text": args.system}]})
    messages.append({"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]})
    text_messages = [{"role": m["role"], "content": [{k: v for k, v in c.items() if k != "image"} for c in m["content"]]}
                     for m in messages]
    rendered = processor.apply_chat_template(text_messages, tokenize=False, add_generation_prompt=True)
    # Pan and scan: the processor's keyword arguments (transformers' names), given to the call as transformers takes
    # them (Gemma3Processor passes its own defaults otherwise, whatever preprocessor_config.json says).
    vision_options = {}
    if args.pan_and_scan:
        vision_options["do_pan_and_scan"] = True
        for name, value in [("pan_and_scan_min_crop_size", args.pan_and_scan_min_crop_size),
                            ("pan_and_scan_max_num_crops", args.pan_and_scan_max_crops),
                            ("pan_and_scan_min_ratio_to_activate", args.pan_and_scan_min_ratio)]:
            if value is not None:
                vision_options[name] = value
    if vision_options:
        inputs = processor(text=rendered, images=[[image]], return_tensors="pt", add_special_tokens=False, **vision_options)
    else:
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt")
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    token_type_ids = inputs.get("token_type_ids")
    image_token = model.config.image_token_index if hasattr(model.config, "image_token_index") else model.config.image_token_id
    if token_type_ids is None:
        token_type_ids = (input_ids == image_token).long()
    L = input_ids.shape[1]
    log(f"prompt: {L} tokens, pixels {list(pixel_values.shape)} ({type(processor.image_processor).__name__})")
    np.save(os.path.join(args.out, "input_ids.npy"), input_ids.numpy().astype(np.int64))
    np.save(os.path.join(args.out, "pixel_values.npy"), pixel_values.numpy().astype(np.float32))

    pv = pixel_values.to(first_device, vision_dtype)
    with torch.no_grad():
        # The vision tower and the projector, called as get_image_features calls them.
        vision = inner.vision_tower(pixel_values=pv).last_hidden_state
        features = inner.multi_modal_projector(vision)
        rows = vision.shape[1] if args.vision_rows <= 0 else min(args.vision_rows, vision.shape[1])
        np.save(os.path.join(args.out, "vision_last_hidden_state.npy"), vision[:, :rows].float().cpu().numpy())
        np.save(os.path.join(args.out, "image_features.npy"), features.float().cpu().numpy())
        log(f"vision: {list(vision.shape)} -> features {list(features.shape)}")

        # Greedy generation: no sampling and no repetition penalty, whatever generation_config.json says (idrak's
        # --temperature 0 is the same), with the raw logits of every step.
        model_inputs = {k: v.to(first_device) for k, v in inputs.items() if hasattr(v, "to")}
        model_inputs["pixel_values"] = pv
        started = time.time()
        gen = model.generate(**model_inputs, max_new_tokens=args.steps, do_sample=False, num_beams=1,
                             repetition_penalty=1.0, output_logits=True, return_dict_in_generate=True)
        seconds = time.time() - started
        generated = gen.sequences[0, L:].cpu()
        N = generated.shape[0]
        step_logits = torch.stack([s[0].float() for s in gen.logits[:N]]).cpu()                     # [N, vocabulary]
        top = step_logits.topk(TOP, dim=-1)
        np.save(os.path.join(args.out, "generated_ids.npy"), generated.numpy().astype(np.int64))
        np.save(os.path.join(args.out, "gen_top_ids.npy"), top.indices.numpy().astype(np.int64))
        np.save(os.path.join(args.out, "gen_top_logits.npy"), top.values.numpy().astype(np.float32))
        K = min(args.full_rows, N)
        np.save(os.path.join(args.out, "gen_full_logits.npy"), step_logits[:K].numpy().astype(np.float32))
        answer = processor.tokenizer.decode(generated.tolist(), skip_special_tokens=False)
        log(f"generated {N} tokens in {seconds:.1f} s")

        # One teacher-forced pass over prompt and answer (the image block bidirectional through token_type_ids).
        full_ids = torch.cat([input_ids, generated.unsqueeze(0)], dim=1).to(first_device)
        full_types = torch.cat([token_type_ids, torch.zeros(1, N, dtype=token_type_ids.dtype)], dim=1).to(first_device)
        out = model(input_ids=full_ids, pixel_values=pv, token_type_ids=full_types,
                    attention_mask=torch.ones_like(full_ids))
        logits = out.logits[0]
        tf_ids, tf_values = [], []
        for start in range(0, logits.shape[0], 64):                       # in chunks: float32 rows of 262,144 values
            t = logits[start:start + 64].float().topk(TOP, dim=-1)
            tf_ids.append(t.indices.cpu())
            tf_values.append(t.values.cpu())
        tf_ids = torch.cat(tf_ids)
        tf_values = torch.cat(tf_values)
        np.save(os.path.join(args.out, "tf_top_ids.npy"), tf_ids.numpy().astype(np.int64))
        np.save(os.path.join(args.out, "tf_top_logits.npy"), tf_values.numpy().astype(np.float32))

    tf_answer = tf_ids[L - 1:L - 1 + N, 0]
    disagree = [i for i in range(N) if int(tf_answer[i]) != int(generated[i])]
    margins = (top.values[:, 0] - top.values[:, 1]).tolist()
    near = sorted(range(N), key=lambda i: margins[i])[:5]
    generation_config = model.generation_config.to_dict() if model.generation_config is not None else {}
    gpu = torch.cuda.get_device_name(first_device) if str(first_device).startswith("cuda") else None
    manifest = {
        "kind": "idrak-vlm-reference",
        "version": 1,
        "model": args.model,
        "architectures": getattr(model.config, "architectures", None),
        "image": os.path.abspath(args.image),
        "image_size": list(image.size),
        "exif_orientation": int(exif_orientation),
        "exif_applied": not args.no_exif,
        "grayscale": bool(args.grayscale),
        "image_transforms": pipeline or None,
        "original_image_size": original_size,
        "vision_options": vision_options,
        "image_blocks": int(pixel_values.shape[0]),
        "system": args.system,
        "prompt": prompt,
        "rendered": rendered,
        "prompt_tokens": int(L),
        "steps": int(N),
        "requested_steps": args.steps,
        "stopped_early": int(N) < args.steps,
        "generated_text": answer,
        "dtype": args.dtype,
        "vision_dtype": args.vision_dtype or args.dtype,
        "logits_dtype": str(out.logits.dtype).replace("torch.", ""),
        "device": str(first_device),
        "device_name": gpu,
        "offload": bool(args.offload),
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "processor_class": type(processor).__name__,
        "image_processor_class": type(processor.image_processor).__name__,
        "use_fast": bool(args.fast_processor),
        "image_token_id": int(image_token),
        "generation": {"do_sample": False, "num_beams": 1, "repetition_penalty": 1.0, "max_new_tokens": args.steps,
                       "seconds": round(seconds, 3)},
        "model_generation_config": generation_config,
        "checks": {
            "teacher_forced_top1_equals_generated": len(disagree) == 0,
            "teacher_forced_disagreements": disagree[:20],
            "smallest_generate_margins": [{"step": i, "margin": round(margins[i], 4)} for i in near],
        },
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "pillow": PIL.__version__, "numpy": np.__version__,
                     "cuda": torch.version.cuda, "platform": platform.platform()},
        "files": {name: list(np.load(os.path.join(args.out, name), mmap_mode="r").shape)
                  for name in sorted(os.listdir(args.out)) if name.endswith(".npy")},
    }
    rp = generation_config.get("repetition_penalty")
    if rp not in (None, 1, 1.0):
        log(f"note: the model's generation_config sets repetition_penalty {rp}; this run used 1.0 (as idrak --temperature 0)")
    dump_json(os.path.join(args.out, "manifest.json"), manifest)
    log(f"teacher-forced top-1 equals generate's tokens: {len(disagree) == 0}"
        + ("" if not disagree else f" (differs at steps {disagree[:10]})"))
    log(f"smallest margins between generate's top two logits: "
        + ", ".join(f"step {i}: {margins[i]:.3f}" for i in near))
    log(f"wrote {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
