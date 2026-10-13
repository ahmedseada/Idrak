"""Tiny random Qwen2.5-VL and Qwen2-VL checkpoints and transformers' outputs for them (plan 16, step 0).

    python -m venv vlm && vlm/bin/pip install torch torchvision transformers tokenizers safetensors pillow numpy jinja2
    vlm/bin/python tools/vlm/make_tiny_qwen_vl.py tests/Idrak.Tests/data/vlm-qwen

The reference the Qwen-VL plug-in (samples/QwenVision/Idrak.QwenVision) and the library's multi-axis rotary positions
are checked against. Nothing is downloaded: configs, weights and a byte-level BPE tokenizer (Qwen's pre-tokenizer
regex, its special tokens) are made here; the pages are phase 0's test image (tests/Idrak.Tests/data/vlm/image.png,
100 x 75) and two cut from it, written as PNG files:

    page.png        the test image (shrunk by the pixel budget: 32 x 48, an 8 x 12 patch grid, 4 x 6 tokens)
    page-tall.png   its left 40 x 75 (shrunk to 56 x 32 by the budget, 14 x 8 patches, 7 x 4 tokens)
    page-small.png  its top-left 20 x 15 (enlarged by min_pixels to 32 x 40, 8 x 10 patches, 4 x 5 tokens)

The vision window (window_size 24: 3 x 3 merged tokens) cuts every grid into full and partial windows.

Models (both float32, eager attention):

    tiny-qwen2.5-vl       Qwen2_5_VLForConditionalGeneration as transformers 5 saves it (text_config and
                          rope_parameters in config.json; the tensors under the hub's names, model.* and visual.*):
                          windowed blocks 0 and 2, full-attention blocks 1 and 3, RMSNorm, SwiGLU; tied embeddings
    tiny-qwen2.5-vl-flat  the same weights with the hub checkpoints' config.json (transformers 4.x): the text keys at
                          the top, rope_scaling {"type": "mrope"}, vision_config's in_chans; loaded back by
                          transformers to the same logits
    tiny-qwen2-vl         Qwen2VLForConditionalGeneration (transformers 5's save): full attention, LayerNorm,
                          quick-GELU; untied
    tiny-qwen2-vl-flat    the same with the hub's config.json

For each model, in reference/ (the model's tag in the name): each page's pixel values and grid; the patch embedding,
the tower's last hidden state (in patch order) and the merged features; the image prompt's ids, position ids
[3, tokens] and rope delta, its logits and 20 greedy tokens with each step's logits; the same for a prompt with two
pages; a text-only prompt's logits and greedy tokens; and a tuning example's loss (an answer after the image prompt,
the answer's tokens scored). manifest.json records the versions, the facts and each file's SHA-256; reruns write the
same bytes on the same machine and library versions.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_tiny  # noqa: E402  (phase 0's helpers: JSON and .npy writing)

GENERATED = 20
PATCH, MERGE, TEMPORAL = 4, 2, 2
MIN_PIXELS, MAX_PIXELS = 1024, 2048
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>",
            "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>", "<|vision_end|>",
            "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>"]
QWEN_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|"
                r"\s+(?!\S)|\s+")

# Qwen2-VL's and Qwen2.5-VL's chat template (chat_template.json of the hub's Instruct models).
CHAT_TEMPLATE = (
    "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}"
    "{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}"
    "<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n"
    "{% else %}{% for content in message['content'] %}"
    "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
    "{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}"
    "{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>"
    "{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")

IMAGE_CHAT = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Read the scan."}]}]
TWO_IMAGE_CHAT = [{"role": "system", "content": [{"type": "text", "text": "Read the scans."}]},
                  {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "and"}, {"type": "image"},
                                               {"type": "text", "text": "What is in these images?"}]}]
TEXT_CHAT = [{"role": "user", "content": "Write one line about the moon and the sun in the night."}]
ANSWER = "A page with lines of text and a box in the corner."
CORPUS = ("the image is a scan of a page with lines of text and a box in the corner . what is in this image ? "
          "read the scan . write one line about the moon and the sun in the night . you are a helpful assistant . "
          "system user assistant picture and these scans images")

VISION_25 = dict(depth=4, hidden_size=32, intermediate_size=48, num_heads=2, in_channels=3, patch_size=PATCH,
                 spatial_merge_size=MERGE, temporal_patch_size=TEMPORAL, window_size=24, out_hidden_size=64,
                 fullatt_block_indexes=[1, 3], hidden_act="silu", tokens_per_second=2)
VISION_2 = dict(depth=2, embed_dim=32, mlp_ratio=2, num_heads=2, in_channels=3, patch_size=PATCH,
                spatial_merge_size=MERGE, temporal_patch_size=TEMPORAL, hidden_size=64, hidden_act="quick_gelu")
MROPE = [2, 3, 3]                       # head size 16: 8 frequency pairs, 2 on time, 3 on height, 3 on width
MODELS = {"tiny-qwen2.5-vl": dict(family="qwen2.5", tied=True), "tiny-qwen2-vl": dict(family="qwen2", tied=False)}


def byte_chars():
    """GPT-2's byte-to-unicode table, as the ByteLevel pre-tokenizer maps bytes."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def train_merges(words, count):
    """Byte-pair merges over byte-level words; ties broken by the pair's text, so the result is fixed."""
    counts = {}
    for w in words:
        counts[tuple(w)] = counts.get(tuple(w), 0) + 1
    merges = []
    for _ in range(count):
        pairs = {}
        for word, n in counts.items():
            for a, b in zip(word, word[1:]):
                pairs[(a, b)] = pairs.get((a, b), 0) + n
        if not pairs:
            break
        best = sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        merges.append(best)
        new = {}
        for word, n in counts.items():
            out, i = [], 0
            while i < len(word):
                if i + 1 < len(word) and (word[i], word[i + 1]) == best:
                    out.append(word[i] + word[i + 1])
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            new[tuple(out)] = new.get(tuple(out), 0) + n
        counts = new
    return merges


def build_tokenizer(folder):
    """A Qwen2-style tokenizer.json (Qwen's Split regex, ByteLevel, BPE, no BOS) and tokenizer_config.json."""
    from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers
    from tokenizers.models import BPE

    table = byte_chars()
    vocab = {table[b]: i for i, b in enumerate(sorted(table))}
    words = ["".join(table[b] for b in (" " + w).encode("utf-8")) for w in CORPUS.split(" ")]
    merges = train_merges(words, 80)
    for a, b in merges:
        vocab.setdefault(a + b, len(vocab))
    ids = {}
    for s in SPECIALS:
        ids[s] = vocab[s] = len(vocab)
    tok = Tokenizer(BPE(vocab=vocab, merges=merges, ignore_merges=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN_PATTERN), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(SPECIALS)
    with open(os.path.join(folder, "tokenizer.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(json.loads(tok.to_str(pretty=False)), indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    added = {str(i): {"content": s, "lstrip": False, "normalized": False, "rstrip": False, "single_word": False,
                      "special": True} for s, i in ids.items()}
    make_tiny.dump_json(os.path.join(folder, "tokenizer_config.json"), {
        "add_bos_token": False, "add_prefix_space": False, "added_tokens_decoder": added, "bos_token": None,
        "chat_template": CHAT_TEMPLATE, "clean_up_tokenization_spaces": False, "eos_token": "<|im_end|>",
        "errors": "replace", "model_max_length": 512, "pad_token": "<|endoftext|>", "split_special_tokens": False,
        "tokenizer_class": "Qwen2Tokenizer", "unk_token": None})
    return len(vocab), ids


def text_config(vocab_size, ids, tied):
    return dict(vocab_size=vocab_size, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, hidden_act="silu", rms_norm_eps=1e-6,
                max_position_embeddings=512, rope_parameters={"rope_type": "default", "rope_theta": 1000000.0,
                                                              "mrope_section": MROPE},
                tie_word_embeddings=tied, bos_token_id=ids["<|endoftext|>"], eos_token_id=ids["<|im_end|>"],
                pad_token_id=ids["<|endoftext|>"], use_sliding_window=False)


def preprocessor_config():
    return {"do_convert_rgb": True, "do_normalize": True, "do_rescale": True, "do_resize": True, "image_mean": MEAN,
            "image_processor_type": "Qwen2VLImageProcessor", "image_std": STD, "max_pixels": MAX_PIXELS,
            "merge_size": MERGE, "min_pixels": MIN_PIXELS, "patch_size": PATCH, "resample": 3,
            "rescale_factor": 1 / 255, "temporal_patch_size": TEMPORAL}


def randomize(model, torch):
    """Every parameter from one seeded generator, in name order, at scales that keep activations near 1."""
    g = torch.Generator().manual_seed(4321)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
            n = torch.randn(p.shape, generator=g, dtype=torch.float32)
            if name.endswith("bias"):
                v = 0.1 * n
            elif "norm" in name or "ln_q" in name:
                v = 1.0 + 0.2 * n
            elif "embed_tokens" in name:
                v = 0.5 * n
            elif "patch_embed" in name:
                v = n / np.sqrt(np.prod(p.shape[1:]))
            else:
                v = 1.5 * n / np.sqrt(p.shape[-1])      # Linear [out, in]
            p.copy_(v)


def flat_layout(src, dst, family):
    """The hub's layout (transformers 4.x): text keys at the top, rope_scaling mrope, in_chans, model.* and visual.*."""
    from safetensors.numpy import load_file, save_file

    os.makedirs(dst)
    for name in os.listdir(src):
        if name not in ("config.json", "model.safetensors"):
            shutil.copy(os.path.join(src, name), os.path.join(dst, name))
    with open(os.path.join(src, "config.json"), encoding="utf-8") as f:
        nested = json.load(f)
    text = dict(nested["text_config"])
    rope = text.pop("rope_parameters")
    flat = {k: v for k, v in nested.items() if k not in ("text_config", "vision_config")}
    for k in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
              "num_key_value_heads", "hidden_act", "rms_norm_eps", "max_position_embeddings", "bos_token_id",
              "eos_token_id", "use_sliding_window", "sliding_window", "max_window_layers", "attention_dropout",
              "initializer_range", "use_cache"):
        if k in text:
            flat[k] = text[k]
    flat["rope_theta"] = rope["rope_theta"]
    flat["rope_scaling"] = {"type": "mrope", "mrope_section": rope["mrope_section"]}
    flat["model_type"] = nested["model_type"]
    flat["tie_word_embeddings"] = nested.get("tie_word_embeddings", False)
    vision = {k: v for k, v in nested["vision_config"].items() if k not in ("dtype", "rope_parameters")}
    vision["in_chans"] = vision.pop("in_channels")
    flat["vision_config"] = vision
    make_tiny.dump_json(os.path.join(dst, "config.json"), flat)
    tensors = load_file(os.path.join(src, "model.safetensors"))
    renamed = {}
    for k, v in tensors.items():
        if k.startswith("model.language_model."):
            k = "model." + k[len("model.language_model."):]
        elif k.startswith("model.visual."):
            k = "visual." + k[len("model.visual."):]
        renamed[k] = v
    save_file(renamed, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})


def main(out):
    import torch
    import transformers
    import tokenizers
    import safetensors
    import PIL
    from PIL import Image
    from transformers import AutoTokenizer
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil, smart_resize
    from transformers.vision_utils import get_vision_window_index

    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    ref = os.path.join(out, "reference")
    os.makedirs(ref)

    image_path = os.path.join(HERE, "..", "..", "tests", "Idrak.Tests", "data", "vlm", "image.png")
    with Image.open(image_path) as im:
        base = im.convert("RGB")
        base.load()
    pages = {"page": base, "page-tall": base.crop((0, 0, 40, 75)), "page-small": base.crop((0, 0, 20, 15))}
    for name, page in pages.items():
        page.save(os.path.join(out, name + ".png"))
    with Image.open(os.path.join(out, "page.png")) as im:
        assert np.array_equal(np.asarray(im.convert("RGB")), np.asarray(base))

    facts, layouts = {}, {}
    for model_name, spec in MODELS.items():
        folder = os.path.join(out, model_name)
        os.makedirs(folder)
        vocab_size, ids = build_tokenizer(folder)
        make_tiny.dump_json(os.path.join(folder, "preprocessor_config.json"), preprocessor_config())
        text = text_config(vocab_size, ids, spec["tied"])
        if spec["family"] == "qwen2.5":
            from transformers import Qwen2_5_VLConfig as Config, Qwen2_5_VLForConditionalGeneration as Model
            from transformers import Qwen2_5_VLProcessor as Processor
            vision = VISION_25
        else:
            from transformers import Qwen2VLConfig as Config, Qwen2VLForConditionalGeneration as Model
            from transformers import Qwen2VLProcessor as Processor
            vision = VISION_2
        config = Config(text_config=text, vision_config=dict(vision), image_token_id=ids["<|image_pad|>"],
                        video_token_id=ids["<|video_pad|>"], vision_start_token_id=ids["<|vision_start|>"],
                        vision_end_token_id=ids["<|vision_end|>"], tie_word_embeddings=spec["tied"])
        config.architectures = [Model.__name__]
        model = Model._from_config(config, attn_implementation="eager", dtype=torch.float32)
        randomize(model, torch)
        if spec["tied"]:
            model.tie_weights()
        model.eval()
        model.generation_config.eos_token_id = None
        model.generation_config.pad_token_id = ids["<|endoftext|>"]
        model.save_pretrained(folder, safe_serialization=True)
        make_tiny.finish_folder(folder)

        tokenizer = AutoTokenizer.from_pretrained(folder)
        image_processor = Qwen2VLImageProcessorPil(min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, patch_size=PATCH,
                                                   merge_size=MERGE, temporal_patch_size=TEMPORAL,
                                                   image_mean=MEAN, image_std=STD)
        from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor
        video_processor = Qwen2VLVideoProcessor(patch_size=PATCH, merge_size=MERGE, temporal_patch_size=TEMPORAL)
        processor = Processor(image_processor=image_processor, tokenizer=tokenizer, video_processor=video_processor,
                              chat_template=CHAT_TEMPLATE)
        tag = model_name[len("tiny-"):]
        model_facts = {"pages": {}}

        def save(name, array, dtype=None):
            a = array.detach().numpy() if hasattr(array, "detach") else np.asarray(array)
            make_tiny.save_npy(ref, f"{tag}-{name}.npy", a.astype(dtype) if dtype else a)

        with torch.no_grad():
            for page_name, page in pages.items():
                px = processor.image_processor(images=page, return_tensors="pt")
                pv, grid = px["pixel_values"], px["image_grid_thw"]
                save(f"{page_name}-pixel_values", pv)
                visual = model.model.visual
                embedded = visual.patch_embed(pv)
                result = visual(pv, grid_thw=grid)
                hidden = result.last_hidden_state
                if spec["family"] == "qwen2.5":
                    window_index, cu = get_vision_window_index(grid, MERGE, vision["window_size"], PATCH)
                    unit = MERGE * MERGE
                    hidden = hidden.reshape(-1, unit, hidden.shape[-1])[torch.argsort(window_index)].reshape(hidden.shape)
                    windows = cu.tolist()
                else:
                    windows = None
                save(f"{page_name}-patch_embed", embedded)
                save(f"{page_name}-tower", hidden)
                save(f"{page_name}-features", result.pooler_output)
                h, w = page.height, page.width
                model_facts["pages"][page_name] = {
                    "size": [w, h], "resized": list(smart_resize(h, w, PATCH * MERGE, MIN_PIXELS, MAX_PIXELS)),
                    "grid_thw": grid[0].tolist(), "tokens": int(grid[0].prod()) // (MERGE * MERGE),
                    "cu_window_seqlens": windows}

            def run(chat, images, name):
                rendered = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=rendered, images=images or None, return_tensors="pt", add_special_tokens=False)
                input_ids = inputs["input_ids"]
                if images:
                    positions, deltas = model.model.get_rope_index(input_ids, inputs["mm_token_type_ids"],
                                                                   inputs["image_grid_thw"])
                    save(f"{name}-position_ids", positions[:, 0], np.int64)
                    delta = int(deltas[0, 0])
                else:
                    delta = 0
                logits = model(**inputs).logits
                gen = model.generate(**inputs, max_new_tokens=GENERATED, do_sample=False, output_logits=True,
                                     return_dict_in_generate=True, eos_token_id=None,
                                     pad_token_id=ids["<|endoftext|>"])
                new_tokens = gen.sequences[0, input_ids.shape[1]:].tolist()
                save(f"{name}-input_ids", input_ids, np.int64)
                save(f"{name}-logits", logits)
                save(f"{name}-generate-logits", torch.stack(gen.logits, dim=1))
                model_facts[name] = {"rendered": rendered, "input_ids": input_ids[0].tolist(), "rope_delta": delta,
                                     "new_tokens": new_tokens, "text": tokenizer.decode(new_tokens)}
                return inputs

            image_inputs = run(IMAGE_CHAT, [pages["page"]], "prompt-image")
            run(TWO_IMAGE_CHAT, [pages["page"], pages["page-tall"]], "prompt-two-images")
            run(TEXT_CHAT, [], "prompt-text")

            # Tuning: the image prompt and an answer; only the answer's tokens (and <|im_end|>) are scored.
            answer = tokenizer(ANSWER + "<|im_end|>", add_special_tokens=False)["input_ids"]
            ids_full = torch.cat([image_inputs["input_ids"], torch.tensor([answer])], dim=1)
            labels = torch.full_like(ids_full, -100)
            labels[0, -len(answer):] = torch.tensor(answer)
            tune_inputs = dict(image_inputs)
            tune_inputs["input_ids"] = ids_full
            tune_inputs["attention_mask"] = torch.ones_like(ids_full)
            tune_inputs["mm_token_type_ids"] = torch.cat(
                [image_inputs["mm_token_type_ids"], torch.zeros((1, len(answer)), dtype=image_inputs["mm_token_type_ids"].dtype)], dim=1)
            tuned = model(**tune_inputs, labels=labels)
            save("tuning-input_ids", ids_full, np.int64)
            save("tuning-logits", tuned.logits)
            model_facts["tuning"] = {"answer": ANSWER, "answer_ids": answer, "loss": float(tuned.loss)}

        layouts[model_name] = {"tensors": make_tiny.safetensors_index(os.path.join(folder, "model.safetensors"))[0]}
        flat = folder + "-flat"
        flat_layout(folder, flat, spec["family"])
        reloaded = Model.from_pretrained(flat, attn_implementation="eager", dtype=torch.float32).eval()
        with torch.no_grad():
            again = reloaded(**image_inputs).logits
        model_facts["flat_layout_same_logits"] = bool(torch.equal(again, model(**image_inputs).logits))
        layouts[model_name + "-flat"] = {"tensors": make_tiny.safetensors_index(os.path.join(flat, "model.safetensors"))[0]}
        facts[model_name] = model_facts

    versions = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
                "tokenizers": tokenizers.__version__, "safetensors": safetensors.__version__, "pillow": PIL.__version__,
                "numpy": np.__version__}
    files = {}
    for root, dirs, names in os.walk(out):
        dirs.sort()
        for n in sorted(names):
            p = os.path.join(root, n)
            rel = os.path.relpath(p, out).replace(os.sep, "/")
            if rel != "manifest.json":
                with open(p, "rb") as f:
                    files[rel] = {"bytes": os.path.getsize(p), "sha256": hashlib.sha256(f.read()).hexdigest()}
    make_tiny.dump_json(os.path.join(out, "manifest.json"), {
        "versions": versions, "token_ids": ids, "facts": facts, "layouts": layouts, "files": files,
        "processor": {"patch_size": PATCH, "merge_size": MERGE, "temporal_patch_size": TEMPORAL,
                      "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS}})
    print(json.dumps({"total_bytes": sum(f["bytes"] for f in files.values())}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", help="output folder (tests/Idrak.Tests/data/vlm-qwen)")
    main(ap.parse_args().out)
