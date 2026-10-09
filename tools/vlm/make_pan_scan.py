"""Gemma 3's pan and scan on the tiny model: transformers' crops, prompts, logits and greedy tokens (plan 11, "Pan and scan").

    vlm/bin/python tools/vlm/make_pan_scan.py tests/Idrak.Tests/data/vlm-pan-scan [--data tests/Idrak.Tests/data/vlm]
    vlm-4.57.6/bin/python tools/vlm/make_pan_scan.py --check tests/Idrak.Tests/data/vlm-pan-scan   # optional, 4.5x

Reads the tiny model of tools/vlm/make_tiny.py (its tiny-gemma3-v5 folder: weights, tokenizer, chat template, processor
config) and nothing from huggingface.co. Draws a tall, a wide and a near-square test image (make_tiny.py's drawing at
other sizes) and runs transformers' Gemma3Processor with do_pan_and_scan=True and a minimum crop size that suits the
tiny model's 56-pixel images (32; the other settings keep Gemma3Processor's defaults: at most 4 crops, ratio 1.2). It
writes each image's pixel values (the whole image, then its crops), their projected features, and for a prompt with
each image the processor's ids, the logits (token_type_ids given, so each crop's block is bidirectional), and 20 greedy
tokens with each step's logits. The near-square image is checked to give exactly the ids, pixels and logits it gives
with pan and scan off. With --check, an older transformers (4.5x) recomputes the crops, ids and logits from the files
and reports whether they agree (it writes nothing).

Reruns write the same bytes (fixed seeds, one thread, deterministic algorithms, sorted JSON) on the same machine and
library versions; manifest.json records the versions and each file's SHA-256.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_tiny  # noqa: E402  (the tiny model's chat template, image drawing and JSON helpers)

# The pan and scan settings of the reference: Gemma3Processor's defaults, but crops of 32 pixels at least (its default
# of 256 suits 896-pixel encoders; the tiny model's is 56).
PAN_AND_SCAN = {"do_pan_and_scan": True, "pan_and_scan_min_crop_size": 32, "pan_and_scan_max_num_crops": 4,
                "pan_and_scan_min_ratio_to_activate": 1.2}
# (name, width, height, what): tall gives 3 crops (the last 2 rows shorter), wide asks for 5 and gets the maximum of
# 4, near-square (ratio 1.125) gets none.
IMAGES = [("tall", 60, 170, "portrait, ratio 2.83: 3 crops of 60 x 57 (the last 60 x 56)"),
          ("wide", 300, 60, "landscape, ratio 5: 5 crops wanted, 4 at most: 75 x 60 each"),
          ("square", 90, 80, "landscape, ratio 1.125 (below 1.2): no crops")]
SYSTEM = "Read the scan."
QUESTION = "What is in this image?"
GENERATED = 20


def chat(name):
    """make_tiny.py's image prompt: a system line, the image, the question."""
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}]


def draw(width, height, seed):
    """make_tiny.py's colourful test image, at another size (its strokes drawn from another seed)."""
    make_tiny.IMAGE_SIZE = (width, height)
    state = np.random.default_rng
    make_tiny.np.random.default_rng = lambda _seed: state(seed)
    try:
        return make_tiny.draw_image()
    finally:
        make_tiny.np.random.default_rng = state


def processor_for(folder):
    from transformers import AutoTokenizer, Gemma3Processor
    try:
        from transformers.models.gemma3.image_processing_pil_gemma3 import Gemma3ImageProcessorPil as ImageProcessor
    except ImportError:                                  # transformers 4.5x: the slow processor is the PIL one
        from transformers import Gemma3ImageProcessor as ImageProcessor
    tokenizer = AutoTokenizer.from_pretrained(folder)
    image_processor = ImageProcessor.from_pretrained(folder)
    return Gemma3Processor(image_processor=image_processor, tokenizer=tokenizer, chat_template=make_tiny.CHAT_TEMPLATE,
                           image_seq_length=make_tiny.MM_TOKENS)


def inputs_for(processor, image, pan_and_scan):
    rendered = processor.apply_chat_template(chat(None), tokenize=False, add_generation_prompt=True)
    kwargs = dict(PAN_AND_SCAN) if pan_and_scan else {}
    out = processor(text=rendered, images=[[image]], return_tensors="pt", add_special_tokens=False, do_convert_rgb=True,
                    **kwargs)
    return rendered, out


def crop_boxes(image_processor, image):
    """The crops transformers cuts, as [top, left, height, width] of the original (from its own pan_and_scan)."""
    arr = np.asarray(image).transpose(2, 0, 1)          # channels first, as the PIL processor of 5.x holds it
    h, w = arr.shape[1:]
    tagged = np.arange(h * w, dtype=np.int64).reshape(1, h, w)
    kwargs = {k: v for k, v in PAN_AND_SCAN.items() if k != "do_pan_and_scan"}
    try:
        crops = image_processor.pan_and_scan(image=tagged, **kwargs)
    except TypeError:                                    # 4.5x: (image, ...) with a data format
        from transformers.image_utils import ChannelDimension
        crops = image_processor.pan_and_scan(tagged, input_data_format=ChannelDimension.FIRST, **kwargs)
    boxes = []
    for c in crops:
        first = int(c[0, 0, 0])
        boxes.append([first // w, first % w, int(c.shape[1]), int(c.shape[2])])
    return boxes


def main(out, data):
    import torch
    import transformers
    import tokenizers
    import PIL
    from PIL import Image
    from transformers import Gemma3ForConditionalGeneration

    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    model_dir = os.path.join(data, "tiny-gemma3-v5")
    shutil.rmtree(os.path.join(out, "reference"), ignore_errors=True)   # its own files only (compare/ is compare_real.py's)
    os.makedirs(os.path.join(out, "reference"))
    ref = os.path.join(out, "reference")
    processor = processor_for(model_dir)
    model = Gemma3ForConditionalGeneration.from_pretrained(model_dir, attn_implementation="eager", dtype=torch.float32).eval()
    ids = {t: processor.tokenizer.convert_tokens_to_ids(t) for t in ["<start_of_image>", "<end_of_image>", "<image_soft_token>", "<pad>"]}

    facts = {"pan_and_scan": PAN_AND_SCAN, "system": SYSTEM, "question": QUESTION, "images": {}}
    image_files = {}
    for seed, (name, width, height, what) in enumerate(IMAGES):
        img = draw(width, height, 10 + seed)
        file = f"image-{name}.png"
        img.save(os.path.join(out, file), optimize=False, compress_level=9)
        image_files[file] = f"PNG, RGB 8-bit, {width} x {height}: {what}"
        with Image.open(os.path.join(out, file)) as im:
            image = im.convert("RGB")
            image.load()

        rendered, inputs = inputs_for(processor, image, pan_and_scan=True)
        _, plain = inputs_for(processor, image, pan_and_scan=False)
        pixel_values = inputs["pixel_values"]
        prompt_ids = inputs["input_ids"]
        tti = inputs["token_type_ids"]
        boxes = crop_boxes(processor.image_processor, image)
        assert pixel_values.shape[0] == 1 + len(boxes), (pixel_values.shape, boxes)
        # Each crop is the original's rectangle, then the processor's own resize, rescale and normalize.
        for k, (top, left, h, w) in enumerate(boxes):
            alone = processor.image_processor(images=image.crop((left, top, left + w, top + h)), return_tensors="pt",
                                              do_convert_rgb=True, do_pan_and_scan=False)["pixel_values"]
            assert torch.equal(alone[0], pixel_values[1 + k]), f"{name}: crop {k} is not the processor of its rectangle"
        assert torch.equal(pixel_values[:1], plain["pixel_values"]), f"{name}: the whole image differs from pan and scan off"

        with torch.no_grad():
            features = model.model.get_image_features(pixel_values, return_dict=True).pooler_output
            logits = model(input_ids=prompt_ids, pixel_values=pixel_values, token_type_ids=tti,
                           attention_mask=inputs["attention_mask"]).logits
            gen = model.generate(**inputs, max_new_tokens=GENERATED, do_sample=False, output_logits=True,
                                 return_dict_in_generate=True, eos_token_id=None, pad_token_id=ids["<pad>"])
        new_tokens = gen.sequences[0, prompt_ids.shape[1]:].tolist()
        step_logits = torch.stack(gen.logits, dim=1)

        make_tiny.save_npy(ref, f"pixel_values-{name}.npy", pixel_values.numpy())
        make_tiny.save_npy(ref, f"image_features-{name}.npy", features.numpy())
        make_tiny.save_npy(ref, f"prompt-{name}-input_ids.npy", prompt_ids.numpy().astype(np.int64))
        make_tiny.save_npy(ref, f"prompt-{name}-logits.npy", logits.numpy())
        make_tiny.save_npy(ref, f"prompt-{name}-generate-logits.npy", step_logits.numpy())

        row = prompt_ids[0].tolist()
        soft = [i for i, t in enumerate(row) if t == ids["<image_soft_token>"]]
        runs = [i for i in soft if i == 0 or row[i - 1] != ids["<image_soft_token>"]]
        entry = {
            "file": file, "width": width, "height": height, "num_crops": len(boxes), "crops_top_left_height_width": boxes,
            "pixel_values_shape": list(pixel_values.shape), "rendered_by_chat_template": rendered,
            "expanded_text": processor.decode(row, skip_special_tokens=False), "input_ids": row,
            "tokens": processor.tokenizer.convert_ids_to_tokens(row), "token_type_ids": tti[0].tolist(),
            "image_block_starts": runs, "image_tokens": len(soft),
            "new_tokens": new_tokens, "text": processor.tokenizer.decode(new_tokens),
            "last_position_argmax": int(logits[0, -1].argmax())}
        if not boxes:
            # Not cropped: exactly what pan and scan off gives (ids, pixels, logits).
            with torch.no_grad():
                plain_logits = model(input_ids=plain["input_ids"], pixel_values=plain["pixel_values"],
                                     token_type_ids=plain["token_type_ids"], attention_mask=plain["attention_mask"]).logits
            entry["same_as_pan_and_scan_off"] = bool(torch.equal(plain["input_ids"], prompt_ids)
                                                     and torch.equal(plain["pixel_values"], pixel_values)
                                                     and torch.equal(plain_logits, logits))
            assert entry["same_as_pan_and_scan_off"], name
        facts["images"][name] = entry

    versions = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
                "tokenizers": tokenizers.__version__, "pillow": PIL.__version__, "numpy": np.__version__}
    files = {}
    for root, dirs, names in os.walk(out):
        dirs[:] = sorted(d for d in dirs if d != "compare")
        for n in sorted(names):
            p = os.path.join(root, n)
            rel = os.path.relpath(p, out).replace(os.sep, "/")
            if rel in ("manifest.json", "README.md"):
                continue
            files[rel] = {"bytes": os.path.getsize(p), "sha256": make_tiny.sha256(p)}
    arrays = {}
    for n in sorted(os.listdir(ref)):
        arr = np.load(os.path.join(ref, n))
        arrays["reference/" + n] = {"dtype": str(arr.dtype), "shape": list(arr.shape)}
    model_files = {n: make_tiny.sha256(os.path.join(model_dir, n)) for n in sorted(os.listdir(model_dir))}
    make_tiny.dump_json(os.path.join(out, "manifest.json"), {
        "versions": versions, "model": "../vlm/tiny-gemma3-v5", "model_sha256": model_files,
        "image_processor_class": type(processor.image_processor).__name__, "image_files": image_files,
        "arrays": arrays, "facts": facts, "files": files})
    print(json.dumps({"total_bytes": sum(f["bytes"] for f in files.values()),
                      "crops": {k: v["num_crops"] for k, v in facts["images"].items()},
                      "prompt_tokens": {k: len(v["input_ids"]) for k, v in facts["images"].items()}}))


def check(out):
    """An older transformers against the files: the crops, the ids and the logits (nothing is written)."""
    import torch
    import transformers
    from PIL import Image
    from transformers import Gemma3ForConditionalGeneration
    with open(os.path.join(out, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    model_dir = os.path.normpath(os.path.join(out, manifest["model"]))
    if int(transformers.__version__.split(".")[0]) < 5:   # 4.x reads the 4.x layout (the same weights, make_tiny.py)
        model_dir = os.path.join(os.path.dirname(model_dir), "tiny-gemma3-pre452")
    processor = processor_for(model_dir)
    model = Gemma3ForConditionalGeneration.from_pretrained(model_dir, attn_implementation="eager", torch_dtype=torch.float32).eval()
    report = {"transformers": transformers.__version__}
    for name, entry in manifest["facts"]["images"].items():
        with Image.open(os.path.join(out, entry["file"])) as im:
            image = im.convert("RGB")
            image.load()
        _, inputs = inputs_for(processor, image, pan_and_scan=True)
        pixels = np.load(os.path.join(out, "reference", f"pixel_values-{name}.npy"))
        logits = np.load(os.path.join(out, "reference", f"prompt-{name}-logits.npy"))
        with torch.no_grad():
            own = model(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                        token_type_ids=inputs["token_type_ids"], attention_mask=inputs["attention_mask"]).logits.numpy()
        report[name] = {"same_ids": inputs["input_ids"][0].tolist() == entry["input_ids"],
                        "same_crops": crop_boxes(processor.image_processor, image) == entry["crops_top_left_height_width"],
                        "max_abs_pixels": float(np.abs(inputs["pixel_values"].numpy() - pixels).max()),
                        "max_abs_logits": float(np.abs(own - logits).max())}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out")
    parser.add_argument("--data", default=None, help="make_tiny.py's output folder (default: next to OUT, vlm)")
    parser.add_argument("--check", action="store_true", help="compare this transformers with the files instead of writing")
    args = parser.parse_args()
    if args.check:
        check(args.out)
    else:
        main(os.path.abspath(args.out), os.path.abspath(args.data or os.path.join(os.path.dirname(os.path.abspath(args.out)), "vlm")))
