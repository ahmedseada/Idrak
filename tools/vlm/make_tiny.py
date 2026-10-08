"""A tiny random Gemma 3 vision-language model and transformers' outputs for it (plan 11, phase 0).

    python -m venv vlm && vlm/bin/pip install torch transformers tokenizers safetensors pillow numpy jinja2
    for v in 4.51.3 4.52.4 4.57.6; do      # older transformers, sharing vlm's torch through a .pth file
        python -m venv vlm-$v && vlm-$v/bin/pip install "transformers==$v" safetensors pillow numpy jinja2
        echo "$PWD/vlm/lib/python3.13/site-packages" > vlm-$v/lib/python3.13/site-packages/zz_torch.pth
    done
    vlm/bin/python tools/vlm/make_tiny.py tests/Idrak.Tests/data/vlm \
        --legacy-python vlm-4.51.3/bin/python --legacy-python vlm-4.52.4/bin/python --legacy-python vlm-4.57.6/bin/python

Nothing is downloaded: the model is built from configs, the tokenizer with the `tokenizers` library, the test image is
drawn here. The main run (transformers 5) writes the reference outputs and its own layout (tiny-gemma3-v5); each
--legacy-python loads the same weights in its transformers, checks its logits against the reference and writes the
layout its save_pretrained produces. A layout no given version writes is written by renaming the tensors here (the
manifest says who wrote each folder). See README.md in the output folder for the files.

Reruns write the same bytes (fixed seeds, one thread, deterministic algorithms, sorted JSON) on the same machine and
library versions; manifest.json records the versions and each file's SHA-256.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

# ------------------------------------------------------------------ sizes (shared with the legacy run)

VISION = dict(hidden_size=8, intermediate_size=16, num_hidden_layers=2, num_attention_heads=2, num_channels=3,
              image_size=56, patch_size=14, layer_norm_eps=1e-6, hidden_act="gelu_pytorch_tanh", attention_dropout=0.0,
              vision_use_head=False, model_type="siglip_vision_model")
MM_TOKENS = 4                      # 56 / 14 = 4 x 4 patches, average-pooled 2 x 2 (kernel 4 / sqrt(4)) to 2 x 2 tokens
SLIDING_WINDOW = 8                 # longer than one image (4 soft tokens), shorter than the prompts
IMAGE_SIZE = (100, 75)             # the test image (width, height): not square, not a multiple of 8 or 16, shrunk to 56

# Gemma's specials, in Gemma's order where it has one (pad 0, eos 1, bos 2, unk 3); <image_soft_token> last, as in
# the real vocabulary (262,144 there).
SPECIALS = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>", "<start_of_turn>", "<end_of_turn>", "<start_of_image>",
            "<end_of_image>"]
IMAGE_TOKEN = "<image_soft_token>"

# Gemma 3's chat template (google/gemma-3-4b-it, tokenizer_config.json), image parts included.
CHAT_TEMPLATE = (
    "{{ bos_token }}\n"
    "{%- if messages[0]['role'] == 'system' -%}\n"
    "    {%- if messages[0]['content'] is string -%}\n"
    "        {%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}\n"
    "    {%- else -%}\n"
    "        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}\n"
    "    {%- endif -%}\n"
    "    {%- set loop_messages = messages[1:] -%}\n"
    "{%- else -%}\n"
    "    {%- set first_user_prefix = \"\" -%}\n"
    "    {%- set loop_messages = messages -%}\n"
    "{%- endif -%}\n"
    "{%- for message in loop_messages -%}\n"
    "    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}\n"
    "        {{ raise_exception(\"Conversation roles must alternate user/assistant/user/assistant/...\") }}\n"
    "    {%- endif -%}\n"
    "    {%- if (message['role'] == 'assistant') -%}\n"
    "        {%- set role = \"model\" -%}\n"
    "    {%- else -%}\n"
    "        {%- set role = message['role'] -%}\n"
    "    {%- endif -%}\n"
    "    {{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else \"\") }}\n"
    "    {%- if message['content'] is string -%}\n"
    "        {{ message['content'] | trim }}\n"
    "    {%- elif message['content'] is iterable -%}\n"
    "        {%- for item in message['content'] -%}\n"
    "            {%- if item['type'] == 'image' -%}\n"
    "                {{ '<start_of_image>' }}\n"
    "            {%- elif item['type'] == 'text' -%}\n"
    "                {{ item['text'] | trim }}\n"
    "            {%- endif -%}\n"
    "        {%- endfor -%}\n"
    "    {%- else -%}\n"
    "        {{ raise_exception(\"Invalid content type\") }}\n"
    "    {%- endif -%}\n"
    "    {{ '<end_of_turn>\n' }}\n"
    "{%- endfor -%}\n"
    "{%- if add_generation_prompt -%}\n"
    "    {{'<start_of_turn>model\n'}}\n"
    "{%- endif -%}\n")

IMAGE_CHAT = [{"role": "system", "content": [{"type": "text", "text": "Read the scan."}]},
              {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is in this image?"}]}]
TEXT_CHAT = [{"role": "user", "content": "Write one line about the moon and the sun in the night."}]
GENERATED = 20

CORPUS = ("the image is a scan of a page with lines of text and a box in the corner . what is in this image ? "
          "write one line about the moon and the sun in the night . read the scan . user model "
          "The Image What Write Read A page shows dark strokes on a light ground ; a red mark and a blue mark .")


# ------------------------------------------------------------------ helpers

def dump_json(path, value):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def save_npy(folder, name, array):
    np.save(os.path.join(folder, name), np.ascontiguousarray(array))
    return name


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def safetensors_index(path):
    """Tensor names, dtypes and shapes from a safetensors header (sorted by name), and its metadata."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", {})
    return {k: {"dtype": v["dtype"], "shape": v["shape"]} for k, v in sorted(header.items())}, meta


def text_config_dict(vocab_size, ids):
    """The text config in the format the real google/gemma-3-4b-it config.json uses (rope_theta,
    rope_local_base_freq, rope_scaling, sliding_window_pattern), which every transformers version reads."""
    return dict(model_type="gemma3_text", vocab_size=vocab_size, hidden_size=24, intermediate_size=48,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                query_pre_attn_scalar=12, hidden_activation="gelu_pytorch_tanh", rms_norm_eps=1e-6,
                max_position_embeddings=256, sliding_window=SLIDING_WINDOW, sliding_window_pattern=2,
                layer_types=["sliding_attention", "full_attention"], rope_theta=1_000_000.0,
                rope_local_base_freq=10_000.0, rope_scaling={"rope_type": "linear", "factor": 8.0},
                attention_bias=False, attention_dropout=0.0, final_logit_softcapping=None, attn_logit_softcapping=None,
                tie_word_embeddings=True, pad_token_id=ids["<pad>"], bos_token_id=ids["<bos>"],
                eos_token_id=ids["<eos>"])


def full_config_dict(vocab_size, ids):
    return dict(architectures=["Gemma3ForConditionalGeneration"], model_type="gemma3",
                text_config=text_config_dict(vocab_size, ids), vision_config=dict(VISION), mm_tokens_per_image=MM_TOKENS,
                boi_token_index=ids["<start_of_image>"], eoi_token_index=ids["<end_of_image>"],
                image_token_index=ids[IMAGE_TOKEN], initializer_range=0.02, tie_word_embeddings=True,
                eos_token_id=[ids["<eos>"], ids["<end_of_turn>"]], torch_dtype="float32")


# The three tensor layouts, by the folder each is written to.
LAYOUTS = {
    "tiny-gemma3-pre452": "transformers before 4.52: language_model.model.*, vision_tower.vision_model.*, "
                          "multi_modal_projector.*",
    "tiny-gemma3": "transformers 4.52 to 4.57: model.language_model.*, model.vision_tower.vision_model.*, "
                   "model.multi_modal_projector.*",
    "tiny-gemma3-v5": "transformers 5: language_model.model.*, vision_tower.* (no vision_model), "
                      "multi_modal_projector.*"}


def canonical(name):
    """Any layout's tensor (or module) name as transformers 5's module name: model.language_model.*,
    model.vision_tower.* (SiglipVisionModel without the vision_model level), model.multi_modal_projector.*, lm_head.*"""
    for a, b in [("language_model.model.", "model.language_model."), ("language_model.lm_head.", "lm_head."),
                 ("model.vision_tower.vision_model.", "model.vision_tower."), ("vision_tower.vision_model.", "model.vision_tower."),
                 ("vision_tower.", "model.vision_tower."), ("multi_modal_projector.", "model.multi_modal_projector.")]:
        if name.startswith(a):
            return b + name[len(a):]
    return name


def in_layout(name, layout):
    """A canonical name in one of LAYOUTS."""
    rules = {"tiny-gemma3-pre452": [("model.language_model.", "language_model.model."), ("lm_head.", "language_model.lm_head."),
                                    ("model.vision_tower.", "vision_tower.vision_model."),
                                    ("model.multi_modal_projector.", "multi_modal_projector.")],
             "tiny-gemma3": [("model.vision_tower.", "model.vision_tower.vision_model.")],
             "tiny-gemma3-v5": [("model.language_model.", "language_model.model."), ("lm_head.", "language_model.lm_head."),
                                ("model.vision_tower.", "vision_tower."),
                                ("model.multi_modal_projector.", "multi_modal_projector.")]}[layout]
    for a, b in rules:
        if name.startswith(a):
            return b + name[len(a):]
    return name


def layout_of(names):
    for layout in LAYOUTS:
        if all(in_layout(canonical(n), layout) == n for n in names):
            return layout
    raise ValueError(f"unknown layout: {sorted(names)[:5]}")


SHARED_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "preprocessor_config.json",
                "processor_config.json")


# ------------------------------------------------------------------ tokenizer

def train_merges(corpus, count):
    """Byte-pair merges over the words of `corpus` ('▁' marks a space, as in Gemma's SentencePiece vocabulary); ties
    are broken by the pair's text, so the result is fixed."""
    words = {}
    for w in corpus.split(" "):
        key = tuple("▁" + w)
        words[key] = words.get(key, 0) + 1
    merges = []
    for _ in range(count):
        pairs = {}
        for word, n in words.items():
            for a, b in zip(word, word[1:]):
                pairs[(a, b)] = pairs.get((a, b), 0) + n
        if not pairs:
            break
        best = sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        merges.append(best)
        new = {}
        for word, n in words.items():
            out, i = [], 0
            while i < len(word):
                if i + 1 < len(word) and (word[i], word[i + 1]) == best:
                    out.append(word[i] + word[i + 1])
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            new[tuple(out)] = new.get(tuple(out), 0) + n
        words = new
    return merges


def build_tokenizer(folder):
    """Gemma-style tokenizer.json (BPE over '▁'-joined text with byte fallback) and tokenizer_config.json."""
    from tokenizers import Tokenizer, decoders, normalizers, processors
    from tokenizers.models import BPE

    vocab = {}
    for s in SPECIALS:
        vocab[s] = len(vocab)
    for b in range(256):
        vocab[f"<0x{b:02X}>"] = len(vocab)
    # Single characters: the ones the corpus and prompts use (anything else falls back to bytes).
    used = set(CORPUS + json.dumps(IMAGE_CHAT + TEXT_CHAT, ensure_ascii=False)) - {" ", "\\"}
    for c in ["▁", "\n"] + sorted(used):
        vocab.setdefault(c, len(vocab))
    merges = train_merges(CORPUS, 60)
    for a, b in merges:
        vocab.setdefault(a + b, len(vocab))
    vocab["\n\n"] = len(vocab)  # Gemma has it as one token; the processor puts "\n\n" around each image
    vocab[IMAGE_TOKEN] = len(vocab)
    ids = {s: vocab[s] for s in SPECIALS + [IMAGE_TOKEN]}

    tok = Tokenizer(BPE(vocab=vocab, merges=merges + [("\n", "\n")], unk_token="<unk>", byte_fallback=True,
                        fuse_unk=True))
    tok.normalizer = normalizers.Replace(" ", "▁")
    tok.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse()])
    tok.post_processor = processors.TemplateProcessing(single="<bos> $A", pair="<bos> $A <bos> $B",
                                                       special_tokens=[("<bos>", ids["<bos>"])])
    tok.add_special_tokens(SPECIALS + [IMAGE_TOKEN])
    text = tok.to_str(pretty=False)
    with open(os.path.join(folder, "tokenizer.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(json.loads(text), indent=1, sort_keys=True, ensure_ascii=False) + "\n")

    added = {str(i): {"content": s, "lstrip": False, "normalized": False, "rstrip": False, "single_word": False,
                      "special": True} for s, i in ids.items()}
    config = {"add_bos_token": True, "add_eos_token": False, "added_tokens_decoder": added, "bos_token": "<bos>",
              "eos_token": "<eos>", "pad_token": "<pad>", "unk_token": "<unk>", "boi_token": "<start_of_image>",
              "eoi_token": "<end_of_image>", "image_token": IMAGE_TOKEN,
              "extra_special_tokens": {"boi_token": "<start_of_image>", "eoi_token": "<end_of_image>",
                                       "image_token": IMAGE_TOKEN},
              "chat_template": CHAT_TEMPLATE, "clean_up_tokenization_spaces": False, "model_max_length": 256,
              "padding_side": "left", "processor_class": "Gemma3Processor", "spaces_between_special_tokens": False,
              "tokenizer_class": "GemmaTokenizer"}
    dump_json(os.path.join(folder, "tokenizer_config.json"), config)
    dump_json(os.path.join(folder, "special_tokens_map.json"),
              {"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>", "unk_token": "<unk>",
               "boi_token": "<start_of_image>", "eoi_token": "<end_of_image>", "image_token": IMAGE_TOKEN})
    return len(vocab), ids


def write_processor_files(folder):
    dump_json(os.path.join(folder, "preprocessor_config.json"), {
        "do_convert_rgb": None, "do_normalize": True, "do_pan_and_scan": None, "do_rescale": True, "do_resize": True,
        "image_mean": [0.5, 0.5, 0.5], "image_processor_type": "Gemma3ImageProcessor", "image_seq_length": MM_TOKENS,
        "image_std": [0.5, 0.5, 0.5], "pan_and_scan_max_num_crops": None, "pan_and_scan_min_crop_size": None,
        "pan_and_scan_min_ratio_to_activate": None, "processor_class": "Gemma3Processor", "resample": 2,
        "rescale_factor": 1 / 255, "size": {"height": VISION["image_size"], "width": VISION["image_size"]}})
    dump_json(os.path.join(folder, "processor_config.json"),
              {"image_seq_length": MM_TOKENS, "processor_class": "Gemma3Processor"})


# ------------------------------------------------------------------ the test image

def draw_image():
    """A colorful image: a saturated hue sweep across, brightness and saturation falling down, a few strokes (black,
    white, colored), a circle and a box, so resizing blends colors and a swapped channel order shows."""
    from PIL import Image, ImageDraw
    w, h = IMAGE_SIZE
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    hue = (x / w + 0.15 * np.sin(y / 6.0)) % 1.0
    sat = 1.0 - 0.6 * (y / h)
    val = 1.0 - 0.35 * (y / h) * (x / w)
    i = np.floor(hue * 6.0)
    f = hue * 6.0 - i
    p, q, t = val * (1 - sat), val * (1 - f * sat), val * (1 - (1 - f) * sat)
    i = i.astype(int) % 6
    r = np.choose(i, [val, q, p, p, t, val])
    g = np.choose(i, [t, val, val, q, p, p])
    b = np.choose(i, [p, p, t, val, val, q])
    rgb = np.stack([r, g, b], axis=-1) * 255.0
    img = Image.fromarray(np.clip(np.round(rgb), 0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(3)
    colors = [(0, 0, 0), (255, 255, 255), (20, 20, 120), (250, 240, 30)]
    for k in range(6):
        x0, y0 = int(rng.integers(4, w - 30)), int(rng.integers(4, h - 10))
        d.line([(x0, y0), (x0 + int(rng.integers(12, 30)), y0 + int(rng.integers(-8, 9)))], fill=colors[k % 4], width=2)
    d.rectangle([w - 26, 6, w - 6, 26], outline=(0, 0, 0), width=2)
    d.ellipse([w - 24, h - 28, w - 4, h - 8], fill=(255, 0, 255), outline=(255, 255, 255), width=2)
    d.rectangle([8, h - 18, 22, h - 6], fill=(255, 0, 0))
    d.rectangle([28, h - 18, 42, h - 6], fill=(0, 0, 255))
    return img


def rgba_image(img):
    """The image with an alpha channel: opaque on the left fading to clear on the right, and a fully transparent box
    whose hidden color is pure green (dropping alpha shows green; compositing onto white would show white)."""
    from PIL import Image, ImageDraw
    w, h = img.size
    alpha = np.clip(np.round(255.0 * (1.0 - np.mgrid[0:h, 0:w][1] / (w - 1))), 0, 255).astype(np.uint8)
    rgba = np.concatenate([np.asarray(img), alpha[..., None]], axis=-1)
    rgba[20:40, 30:60] = (0, 255, 0, 0)
    return Image.fromarray(rgba, "RGBA")


def write_images(folder):
    from PIL import Image
    img = draw_image()
    files = {}
    img.save(os.path.join(folder, "image.png"), optimize=False, compress_level=9)
    files["image.png"] = "PNG, RGB 8-bit, no metadata"
    rgba_image(img).save(os.path.join(folder, "image-rgba.png"), optimize=False, compress_level=9)
    files["image-rgba.png"] = "PNG, RGBA 8-bit: alpha falls from 255 (left) to 0 (right); a fully clear box hides pure green"
    img.quantize(colors=16, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(
        os.path.join(folder, "image-palette.png"), optimize=False, compress_level=9)
    files["image-palette.png"] = "PNG, palette (mode P), 16 colors by median cut, no dithering, no transparency"
    variants = [("image.jpg", dict(quality=85, subsampling=2), "baseline JPEG, YCbCr 4:2:0, quality 85"),
                ("image-progressive.jpg", dict(quality=85, subsampling=2, progressive=True),
                 "progressive JPEG, YCbCr 4:2:0, quality 85"),
                ("image-444.jpg", dict(quality=90, subsampling=0), "baseline JPEG, YCbCr 4:4:4, quality 90"),
                ("image-restart.jpg", dict(quality=85, subsampling=2, restart_marker_blocks=5),
                 "baseline JPEG, 4:2:0, quality 85, a restart marker every 5 MCUs"),
                ("image-gray.jpg", dict(quality=85), "baseline JPEG, one grayscale component, quality 85")]
    for name, opts, what in variants:
        src = img.convert("L") if "gray" in name else img
        src.save(os.path.join(folder, name), "JPEG", optimize=False, **opts)
        files[name] = what
    return img, files


# ------------------------------------------------------------------ weights

def randomize(model, torch):
    """Every parameter from one seeded generator, in name order, at scales that keep activations near 1, so the
    reference exercises every weight (transformers' init leaves the norms at 1 and the projector at 0)."""
    g = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
            n = torch.randn(p.shape, generator=g, dtype=torch.float32)
            if name.endswith("bias"):
                v = 0.1 * n
            elif "vision_tower" in name and ("layer_norm" in name or "layernorm" in name):
                v = 1.0 + 0.1 * n                       # LayerNorm: y * w + b
            elif name.endswith("norm.weight"):
                v = 1.0 * n                             # Gemma's RMSNorm: y * (1 + w); 1.0 keeps greedy output varied
            elif "embed_tokens" in name:
                v = 0.2 * n                             # times sqrt(32) in the model
            elif "position_embedding" in name:
                v = 0.5 * n
            elif "patch_embedding" in name:
                v = n / np.sqrt(np.prod(p.shape[1:]))
            elif "mm_input_projection_weight" in name:
                v = n / np.sqrt(p.shape[0])             # [vision, text], used as x @ W
            else:
                v = n / np.sqrt(p.shape[-1])            # Linear [out, in]
            p.copy_(v)


# ------------------------------------------------------------------ the main (current transformers) run

def finish_folder(folder):
    """After save_pretrained: no generation_config.json (not part of the reference), config.json with sorted keys."""
    gen_cfg = os.path.join(folder, "generation_config.json")
    if os.path.exists(gen_cfg):
        os.remove(gen_cfg)
    with open(os.path.join(folder, "config.json"), encoding="utf-8") as f:
        dump_json(os.path.join(folder, "config.json"), json.load(f))
    os.chmod(os.path.join(folder, "model.safetensors"), 0o644)


def main_run(out, legacy_pythons):
    import torch
    import transformers
    import tokenizers
    import safetensors
    import PIL
    from PIL import Image, features
    from transformers import (AutoTokenizer, Gemma3Config, Gemma3ForConditionalGeneration, Gemma3Processor)

    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    import shutil
    os.makedirs(out, exist_ok=True)
    for layout in LAYOUTS:
        shutil.rmtree(os.path.join(out, layout), ignore_errors=True)
    shutil.rmtree(os.path.join(out, "reference"), ignore_errors=True)
    work = os.path.join(out, "_main")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)

    vocab_size, ids = build_tokenizer(work)
    write_processor_files(work)
    img, image_files = write_images(out)

    cfg = full_config_dict(vocab_size, ids)
    config = Gemma3Config(**{k: v for k, v in cfg.items() if k not in ("architectures", "model_type")})
    config.architectures = ["Gemma3ForConditionalGeneration"]
    model = Gemma3ForConditionalGeneration._from_config(config, attn_implementation="eager", torch_dtype=torch.float32)
    randomize(model, torch)
    model.eval()
    model.generation_config.eos_token_id = None
    model.generation_config.pad_token_id = ids["<pad>"]
    model.save_pretrained(work, safe_serialization=True)
    finish_folder(work)
    main_layout = layout_of(safetensors_index(os.path.join(work, "model.safetensors"))[0])
    new_dir = os.path.join(out, main_layout)
    os.rename(work, new_dir)

    tokenizer = AutoTokenizer.from_pretrained(new_dir)
    from transformers.models.gemma3.image_processing_pil_gemma3 import Gemma3ImageProcessorPil
    image_processor = Gemma3ImageProcessorPil.from_pretrained(new_dir)
    processor = Gemma3Processor(image_processor=image_processor, tokenizer=tokenizer, chat_template=CHAT_TEMPLATE,
                                image_seq_length=MM_TOKENS)

    ref = os.path.join(out, "reference")
    os.makedirs(ref, exist_ok=True)
    facts = {}

    # The tokenizer: transformers' wrapper must agree with the raw tokenizers library.
    from tokenizers import Tokenizer
    raw = Tokenizer.from_file(os.path.join(new_dir, "tokenizer.json"))
    sample = "What is in this image?\n\nthe moon"
    assert raw.encode(sample).ids == tokenizer(sample)["input_ids"], "tokenizer wrapper disagrees with tokenizers"
    facts["tokenizer_sample"] = {"text": sample, "ids": raw.encode(sample).ids}

    # Pixels: decoded files, then the processor.
    decoded = {}
    for name in sorted(image_files):
        if not name.endswith(".jpg"):
            continue
        with Image.open(os.path.join(out, name)) as im:
            arr = np.asarray(im.convert("RGB") if im.mode not in ("RGB", "L") else im)
        decoded[name] = arr
    unique = {}
    jpeg_decodes = {}
    for name, arr in sorted(decoded.items(), key=lambda kv: (kv[0] != "image.jpg", kv[0])):
        key = hashlib.sha256(arr.tobytes() + str(arr.shape).encode()).hexdigest()
        if key not in unique:
            unique[key] = save_npy(ref, "decoded-" + name.replace(".jpg", ".npy"), arr)
        jpeg_decodes[name] = {"pil_decoded": unique[key], "shape": list(arr.shape), "mode": "L" if arr.ndim == 2 else "RGB"}
    facts["jpeg_decodes"] = jpeg_decodes

    def pixels(image):
        # do_convert_rgb as Gemma3Processor sets it (preprocessor_config.json leaves it null, like the real model's).
        return processor.image_processor(images=image, return_tensors="pt", do_convert_rgb=True)["pixel_values"]

    with Image.open(os.path.join(out, "image.png")) as im:
        png = im.convert("RGB")
        png.load()
    assert np.array_equal(np.asarray(png), np.asarray(img)), "PNG round trip"
    with Image.open(os.path.join(out, "image.jpg")) as im:
        jpg = im.convert("RGB")
        jpg.load()
    pv_png = pixels(png)
    pv_jpg = pixels(jpg)
    pv_gray = pixels(png.convert("L"))
    save_npy(ref, "pixel_values-png.npy", pv_png.numpy())
    save_npy(ref, "pixel_values-jpg.npy", pv_jpg.numpy())
    save_npy(ref, "pixel_values-png-gray.npy", pv_gray.numpy())
    # The RGBA and palette files go to the processor as PIL opens them (mode RGBA, mode P): its convert_to_rgb does it.
    with Image.open(os.path.join(out, "image-rgba.png")) as im:
        im.load()
        rgba = im.copy()
    with Image.open(os.path.join(out, "image-palette.png")) as im:
        im.load()
        pal = im.copy()
    pv_rgba = pixels(rgba)
    pv_pal = pixels(pal)
    save_npy(ref, "pixel_values-rgba.npy", pv_rgba.numpy())
    save_npy(ref, "pixel_values-palette.npy", pv_pal.numpy())

    def expected(rgb_image):
        resized = rgb_image.resize((VISION["image_size"],) * 2, Image.BILINEAR)
        e = ((np.asarray(resized).astype(np.float64) * (1 / 255)).astype(np.float32) - np.float32(0.5)) / np.float32(0.5)
        return e.transpose(2, 0, 1)

    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    white.alpha_composite(rgba)
    l_fixed = np.asarray(png).astype(np.int64) @ np.array([19595, 38470, 7471])
    facts["pixels_check"] = {
        "image_processor_class": type(processor.image_processor).__name__,
        "png_equals_PIL_bilinear_uint8_then_rescale_then_normalize": bool(np.array_equal(pv_png.numpy()[0],
                                                                                         expected(png))),
        "gray_is_PIL_convert_L_then_RGB": bool(np.array_equal(pv_gray.numpy()[0], expected(png.convert("L").convert("RGB")))),
        "gray_channels_equal": bool(torch.equal(pv_gray[0, 0], pv_gray[0, 1]) and torch.equal(pv_gray[0, 1], pv_gray[0, 2])),
        "PIL_L_is_(19595R+38470G+7471B+32768)>>16": bool(np.array_equal(
            np.asarray(png.convert("L")).astype(np.int64), (l_fixed + 32768) >> 16)),
        "rgba_drops_alpha_keeps_hidden_rgb": bool(np.array_equal(pv_rgba.numpy()[0], expected(rgba.convert("RGB")))),
        "rgba_max_abs_diff_vs_compositing_on_white": float(np.abs(pv_rgba.numpy()[0] - expected(white.convert("RGB"))).max()),
        "palette_is_palette_lookup": bool(np.array_equal(pv_pal.numpy()[0], expected(pal.convert("RGB"))))}

    # The vision tower and the projector.
    with torch.no_grad():
        vision = model.model.vision_tower(pixel_values=pv_png)
        embeds = model.model.vision_tower.embeddings(pv_png)
        projected = model.model.multi_modal_projector(vision.last_hidden_state)
        feats = model.model.get_image_features(pv_png, return_dict=True).pooler_output
    assert torch.equal(projected, feats)
    save_npy(ref, "vision_embeddings.npy", embeds.numpy())
    save_npy(ref, "vision_last_hidden_state.npy", vision.last_hidden_state.numpy())
    save_npy(ref, "image_features.npy", projected.numpy())

    # The prompt with the image.
    rendered = processor.apply_chat_template(IMAGE_CHAT, tokenize=False, add_generation_prompt=True)
    with_image = [{"role": m["role"], "content": [dict(c, image=png) if c["type"] == "image" else c
                                                   for c in m["content"]]} for m in IMAGE_CHAT]
    inputs = processor.apply_chat_template(with_image, tokenize=True, add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt")
    via_call = processor(text=rendered, images=[[png]], return_tensors="pt", add_special_tokens=False)
    ids_image = inputs["input_ids"][0].tolist()
    assert ids_image == via_call["input_ids"][0].tolist(), "apply_chat_template and processor() disagree"
    assert torch.equal(inputs["pixel_values"], pv_png)
    tti = inputs["token_type_ids"]
    save_npy(ref, "prompt-image-input_ids.npy", inputs["input_ids"].numpy().astype(np.int64))

    with torch.no_grad():
        out_img = model(input_ids=inputs["input_ids"], pixel_values=pv_png, token_type_ids=tti,
                        attention_mask=inputs["attention_mask"])
        out_causal = model(input_ids=inputs["input_ids"], pixel_values=pv_png,
                           attention_mask=inputs["attention_mask"])
    save_npy(ref, "prompt-image-logits.npy", out_img.logits.numpy())

    # The masks transformers builds for this prompt (eager: additive 0 / -inf), as [first, last] key per query row.
    with torch.no_grad():
        embeds_in = model.get_input_embeddings()(inputs["input_ids"])
        masks = Gemma3ForConditionalGeneration.create_masks_for_generate(
            config=model.config, inputs_embeds=embeds_in, attention_mask=inputs["attention_mask"],
            past_key_values=None, position_ids=torch.arange(len(ids_image)).unsqueeze(0), token_type_ids=tti)
    mask_rows = {}
    for kind, m in sorted(masks.items()):
        allowed = (m[0, 0] == 0).numpy() if m.dtype != torch.bool else m[0, 0].numpy()
        rows = []
        for q in range(allowed.shape[0]):
            keys = np.nonzero(allowed[q])[0]
            assert keys.size and keys[-1] - keys[0] + 1 == keys.size, f"{kind} row {q} is not one range"
            rows.append([int(keys[0]), int(keys[-1])])
        mask_rows[kind] = rows

    soft = [i for i, t in enumerate(ids_image) if t == ids[IMAGE_TOKEN]]
    facts["image_prompt"] = {
        "messages": IMAGE_CHAT, "rendered_by_chat_template": rendered,
        "expanded_text": processor.decode(ids_image, skip_special_tokens=False),
        "input_ids": ids_image, "tokens": tokenizer.convert_ids_to_tokens(ids_image),
        "token_type_ids": tti[0].tolist(), "soft_token_positions": soft,
        "boi_position": ids_image.index(ids["<start_of_image>"]), "eoi_position": ids_image.index(ids["<end_of_image>"]),
        "mask_rows_first_last_key": mask_rows,
        "max_abs_logit_change_without_token_type_ids": float((out_img.logits - out_causal.logits).abs().max()),
        "max_abs_logit_change_without_token_type_ids_before_first_soft_token": float(
            (out_img.logits - out_causal.logits)[0, :soft[0]].abs().max()),
        "last_position_argmax": int(out_img.logits[0, -1].argmax())}

    # Greedy generation (20 tokens; no end-of-sequence stop) and its per-step logits.
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=GENERATED, do_sample=False, output_scores=True,
                             output_logits=True, return_dict_in_generate=True, eos_token_id=None,
                             pad_token_id=ids["<pad>"])
    new_tokens = gen.sequences[0, len(ids_image):].tolist()
    step_logits = torch.stack(gen.logits, dim=1)  # [1, steps, vocab]
    save_npy(ref, "prompt-image-generate-logits.npy", step_logits.numpy())
    # The same tokens recomputed without a cache (one full pass, image block bidirectional, new tokens causal).
    with torch.no_grad():
        full_ids = gen.sequences
        full_tti = torch.cat([tti, torch.zeros(1, GENERATED, dtype=tti.dtype)], dim=1)
        full = model(input_ids=full_ids, pixel_values=pv_png, token_type_ids=full_tti)
    recomputed = full.logits[0, len(ids_image) - 1:-1].argmax(-1).tolist()
    facts["generate"] = {"new_tokens": new_tokens, "text": tokenizer.decode(new_tokens),
                         "matches_one_pass_without_cache": recomputed == new_tokens,
                         "max_abs_diff_step_logits_vs_one_pass": float(
                             (full.logits[0, len(ids_image) - 1:-1] - step_logits[0]).abs().max())}

    # A text-only prompt (longer than the sliding window).
    text_ids = processor.apply_chat_template(TEXT_CHAT, tokenize=True, add_generation_prompt=True, return_dict=True,
                                             return_tensors="pt")
    with torch.no_grad():
        out_text = model(input_ids=text_ids["input_ids"])
    save_npy(ref, "prompt-text-input_ids.npy", text_ids["input_ids"].numpy().astype(np.int64))
    save_npy(ref, "prompt-text-logits.npy", out_text.logits.numpy())
    facts["text_prompt"] = {"messages": TEXT_CHAT,
                            "rendered_by_chat_template": processor.apply_chat_template(TEXT_CHAT, tokenize=False,
                                                                                       add_generation_prompt=True),
                            "input_ids": text_ids["input_ids"][0].tolist(),
                            "last_position_argmax": int(out_text.logits[0, -1].argmax())}

    # The model internals the later phases rely on, read from the running code.
    lm = model.model.language_model
    facts["model_facts"] = {
        "embed_scale": float(lm.embed_tokens.embed_scale),
        "attention_scaling": float(lm.layers[0].self_attn.scaling),
        "layer_types": list(model.config.text_config.layer_types),
        "sliding_window": model.config.text_config.sliding_window,
        "rope_parameters": model.config.text_config.rope_parameters,
        "projector": {"patches_per_image": model.model.multi_modal_projector.patches_per_image,
                      "tokens_per_side": model.model.multi_modal_projector.tokens_per_side,
                      "kernel_size": model.model.multi_modal_projector.kernel_size,
                      "norm_eps": model.model.multi_modal_projector.mm_soft_emb_norm.eps},
        "lm_head_tied": bool(model.lm_head.weight.data_ptr() == lm.embed_tokens.weight.data_ptr()),
        "inv_freq_full_attention": lm.rotary_emb.full_attention_inv_freq.tolist(),
        "inv_freq_sliding_attention": lm.rotary_emb.sliding_attention_inv_freq.tolist(),
    }

    # The other layouts: written by older transformers (--legacy-python), or renamed here.
    reports = {main_layout: {"written_by": f"transformers {transformers.__version__} save_pretrained"}}
    for py in legacy_pythons or []:
        args = [py, "-I", os.path.abspath(__file__), "--legacy", os.path.abspath(out), "--source", main_layout]
        result = subprocess.run(args, check=True, capture_output=True, text=True).stdout
        report = json.loads(result.strip().splitlines()[-1])
        layout = report.pop("layout")
        if layout in reports:
            reports[layout].setdefault("also_written_by", []).append(report)
        else:
            report.pop("model_safetensors_same_bytes_as_first_writer")
            reports[layout] = report
    from safetensors.torch import load_file, save_file
    state = {canonical(k): v for k, v in load_file(os.path.join(new_dir, "model.safetensors")).items()}
    for layout in LAYOUTS:
        folder = os.path.join(out, layout)
        if layout in reports:
            continue
        os.makedirs(folder, exist_ok=True)
        save_file({in_layout(k, layout): v for k, v in sorted(state.items())}, os.path.join(folder, "model.safetensors"),
                  metadata={"format": "pt"})
        dump_json(os.path.join(folder, "config.json"), cfg)
        reports[layout] = {"written_by": "renamed by make_tiny.py (no --legacy-python of that version given); "
                                         "config.json in the real gemma-3-4b-it's format"}
    for layout in LAYOUTS:
        folder = os.path.join(out, layout)
        for name in SHARED_FILES:
            if folder != new_dir:
                shutil.copyfile(os.path.join(new_dir, name), os.path.join(folder, name))
        # Every folder loads in this transformers and gives the reference logits.
        loaded = Gemma3ForConditionalGeneration.from_pretrained(folder, attn_implementation="eager",
                                                                dtype=torch.float32).eval()
        with torch.no_grad():
            lg = loaded(input_ids=inputs["input_ids"], pixel_values=pv_png, token_type_ids=tti).logits
        tensors, meta = safetensors_index(os.path.join(folder, "model.safetensors"))
        reports[layout].update({"description": LAYOUTS[layout], "metadata": meta, "tensors": tensors,
                                "has_lm_head": any("lm_head" in t for t in tensors),
                                f"loads_in_transformers_{transformers.__version__}_same_logits":
                                    bool(torch.equal(lg, out_img.logits))})

    versions = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
                "tokenizers": tokenizers.__version__, "safetensors": safetensors.__version__, "pillow": PIL.__version__,
                "libjpeg": features.version("jpg"), "zlib": features.version("zlib"), "numpy": np.__version__}
    files = {}
    for root, dirs, names in os.walk(out):
        dirs.sort()
        for n in sorted(names):
            p = os.path.join(root, n)
            rel = os.path.relpath(p, out).replace(os.sep, "/")
            if rel in ("manifest.json", "README.md"):
                continue
            files[rel] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
    npy = {}
    for n in sorted(os.listdir(ref)):
        arr = np.load(os.path.join(ref, n))
        npy["reference/" + n] = {"dtype": str(arr.dtype), "shape": list(arr.shape)}
    manifest = {
        "versions": versions,
        "image_files": image_files,
        "layouts": reports,
        "token_ids": ids, "vocab_size": vocab_size,
        "arrays": npy,
        "facts": facts,
        "files": files}
    dump_json(os.path.join(out, "manifest.json"), manifest)
    print(json.dumps({"total_bytes": sum(f["bytes"] for f in files.values())}))


# ------------------------------------------------------------------ the legacy (transformers 4.x) run

def legacy_run(out, source):
    import shutil
    import torch
    import transformers
    from safetensors.numpy import load_file
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration

    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    src = os.path.join(out, source)
    with open(os.path.join(src, "tokenizer.json"), encoding="utf-8") as f:
        vocab = json.load(f)["model"]["vocab"]
    ids = {s: vocab[s] for s in SPECIALS + [IMAGE_TOKEN]}
    cfg = full_config_dict(len(vocab), ids)
    config = Gemma3Config(**{k: v for k, v in cfg.items() if k not in ("architectures", "model_type")})
    config.architectures = ["Gemma3ForConditionalGeneration"]
    model = Gemma3ForConditionalGeneration._from_config(config, attn_implementation="eager", torch_dtype=torch.float32)
    weights = {canonical(k): torch.from_numpy(v) for k, v in load_file(os.path.join(src, "model.safetensors")).items()}
    weights.setdefault("lm_head.weight", weights["model.language_model.embed_tokens.weight"])
    state = {k: weights[canonical(k)] for k in model.state_dict()}
    model.load_state_dict(state, strict=True)
    model.tie_weights()
    model.eval()
    work = os.path.join(out, "_legacy")
    shutil.rmtree(work, ignore_errors=True)
    model.save_pretrained(work, safe_serialization=True)
    finish_folder(work)
    layout = layout_of(safetensors_index(os.path.join(work, "model.safetensors"))[0])
    target = os.path.join(out, layout)
    same = None
    if os.path.exists(target):  # an earlier --legacy-python wrote this layout: keep it, compare
        same = sha256(os.path.join(target, "model.safetensors")) == sha256(os.path.join(work, "model.safetensors"))
        shutil.rmtree(work)
    else:
        os.rename(work, target)

    ref = os.path.join(out, "reference")
    ids_image = torch.from_numpy(np.load(os.path.join(ref, "prompt-image-input_ids.npy")))
    pv = torch.from_numpy(np.load(os.path.join(ref, "pixel_values-png.npy")))
    tti = (ids_image == ids[IMAGE_TOKEN]).long()
    ids_text = torch.from_numpy(np.load(os.path.join(ref, "prompt-text-input_ids.npy")))
    with torch.no_grad():
        img = model(input_ids=ids_image, pixel_values=pv, token_type_ids=tti).logits.numpy()
        txt = model(input_ids=ids_text).logits.numpy()
        tower = model.vision_tower if hasattr(model, "vision_tower") else model.model.vision_tower
        vis = tower(pixel_values=pv).last_hidden_state.numpy()
        gen = model.generate(input_ids=ids_image, pixel_values=pv, token_type_ids=tti,
                             attention_mask=torch.ones_like(ids_image), max_new_tokens=GENERATED, do_sample=False,
                             eos_token_id=None, pad_token_id=ids["<pad>"])
    ref_img = np.load(os.path.join(ref, "prompt-image-logits.npy"))
    ref_txt = np.load(os.path.join(ref, "prompt-text-logits.npy"))
    ref_vis = np.load(os.path.join(ref, "vision_last_hidden_state.npy"))
    soft = np.nonzero(tti[0].numpy())[0]
    report = {"layout": layout, "model_safetensors_same_bytes_as_first_writer": same,
              "written_by": f"transformers {transformers.__version__} save_pretrained (torch {torch.__version__})",
              "check_max_abs_diff_vision_last_hidden_state": float(np.abs(vis - ref_vis).max()),
              "check_max_abs_diff_image_prompt_logits": float(np.abs(img - ref_img).max()),
              "check_max_abs_diff_image_prompt_logits_before_image": float(np.abs(img - ref_img)[0, :soft[0]].max()),
              "check_max_abs_diff_image_prompt_logits_last_position": float(np.abs(img[0, -1] - ref_img[0, -1]).max()),
              "check_max_abs_diff_text_prompt_logits": float(np.abs(txt - ref_txt).max()),
              "check_greedy_tokens": gen[0, ids_image.shape[1]:].tolist()}
    # Rounded so the manifest stays byte-identical across reruns when the differences are float noise.
    report = {k: (float(f"{v:.1e}") if isinstance(v, float) else v) for k, v in report.items()}
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", help="output folder (tests/Idrak.Tests/data/vlm)")
    ap.add_argument("--legacy-python", action="append",
                    help="a python with an older transformers (4.51 for the pre-4.52 layout, 4.57 for 4.52-4.57's); "
                         "repeatable")
    ap.add_argument("--legacy", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--source", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.legacy:
        legacy_run(a.out, a.source)
    else:
        main_run(a.out, a.legacy_python)
