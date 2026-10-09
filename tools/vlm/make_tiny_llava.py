"""A tiny random LLaVA (LlavaForConditionalGeneration) and transformers' outputs for it (plan 11, "Vision contracts").

    python -m venv vlm && vlm/bin/pip install torch transformers tokenizers safetensors pillow numpy jinja2
    vlm/bin/python tools/vlm/make_tiny_llava.py tests/Idrak.Tests/data/vlm-llava

The second vision-language family of Idrak's tests, written to prove the vision contracts general: a CLIP vision tower
(a class token, a bias-free patch convolution, pre_layrnorm, quick-GELU layers), the hidden states of
vision_feature_layer with vision_feature_select_strategy, a two-layer MLP projector (exact GELU), a Llama text decoder,
and plain causal attention over the image tokens. Nothing is downloaded: the configs, weights and tokenizer are made
here, and the test image is phase 0's (tests/Idrak.Tests/data/vlm/image.png, 100 x 75), resized by its shortest edge
(bicubic) and center-cropped as LlavaImageProcessor does. Two models share the text decoder's sizes:

    tiny-llava        vision_feature_layer -2, strategy "default" (the class token dropped): 16 image tokens
    tiny-llava-full   vision_feature_layer [-3, -1], strategy "full": 17 image tokens, the projector over 2 x width

For each: the pixel values, the selected vision hidden states, the projected features, the prompt's ids and logits, and
20 greedy tokens with each step's logits. Reruns write the same bytes on the same machine and library versions;
manifest.json records the versions and each file's SHA-256.
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
import make_tiny  # noqa: E402  (phase 0's helpers: the tokenizer's merges, JSON and .npy writing)

SPECIALS = ["<unk>", "<s>", "</s>", "<pad>"]
IMAGE_TOKEN = "<image>"
IMAGE_SIDE, PATCH = 28, 7
VISION = dict(model_type="clip_vision_model", hidden_size=16, intermediate_size=32, num_hidden_layers=3,
              num_attention_heads=2, num_channels=3, image_size=IMAGE_SIDE, patch_size=PATCH, layer_norm_eps=1e-5,
              hidden_act="quick_gelu", projection_dim=8, attention_dropout=0.0)
VARIANTS = {"tiny-llava": dict(vision_feature_layer=-2, vision_feature_select_strategy="default"),
            "tiny-llava-full": dict(vision_feature_layer=[-3, -1], vision_feature_select_strategy="full")}
GENERATED = 20

# A LLaVA-style chat template ("<s>USER: <image>\n... ASSISTANT:"), written to take a message's content as a string or as
# parts (llava-hf's own template takes parts only, and leaves <s> to the tokenizer's post-processor).
CHAT_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "{% if message['role'] == 'user' %}{{ 'USER: ' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' }}{% endif %}"
    "{% if message['content'] is string %}{{ message['content'] + ' ' }}"
    "{% else %}{% for item in message['content'] %}"
    "{% if item['type'] == 'image' %}{{ '<image>\n' }}{% elif item['type'] == 'text' %}{{ item['text'] + ' ' }}{% endif %}"
    "{% endfor %}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}")

IMAGE_CHAT = [{"role": "system", "content": [{"type": "text", "text": "Read the scan."}]},
              {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is in this image?"}]}]
CORPUS = ("the image is a scan of a page with lines of text and a box in the corner . what is in this image ? "
          "read the scan . USER ASSISTANT A page shows dark strokes on a light ground ; a red mark and a blue mark .")


def build_tokenizer(folder):
    """A Llama-style tokenizer.json (BPE over '▁'-joined text, byte fallback, <s> added) and tokenizer_config.json."""
    from tokenizers import Tokenizer, decoders, normalizers, processors
    from tokenizers.models import BPE

    vocab = {}
    for s in SPECIALS:
        vocab[s] = len(vocab)
    for b in range(256):
        vocab[f"<0x{b:02X}>"] = len(vocab)
    used = set(CORPUS + json.dumps(IMAGE_CHAT, ensure_ascii=False)) - {" ", "\\"}
    for c in ["▁", "\n"] + sorted(used):
        vocab.setdefault(c, len(vocab))
    merges = make_tiny.train_merges(CORPUS, 60)
    for a, b in merges:
        vocab.setdefault(a + b, len(vocab))
    vocab[IMAGE_TOKEN] = len(vocab)
    ids = {s: vocab[s] for s in SPECIALS + [IMAGE_TOKEN]}

    tok = Tokenizer(BPE(vocab=vocab, merges=merges, unk_token="<unk>", byte_fallback=True, fuse_unk=True))
    tok.normalizer = normalizers.Replace(" ", "▁")
    tok.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse()])
    tok.post_processor = processors.TemplateProcessing(single="<s> $A", pair="<s> $A <s> $B",
                                                       special_tokens=[("<s>", ids["<s>"])])
    tok.add_special_tokens(SPECIALS + [IMAGE_TOKEN])
    with open(os.path.join(folder, "tokenizer.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(json.loads(tok.to_str(pretty=False)), indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    added = {str(i): {"content": s, "lstrip": False, "normalized": False, "rstrip": False, "single_word": False,
                      "special": True} for s, i in ids.items()}
    make_tiny.dump_json(os.path.join(folder, "tokenizer_config.json"), {
        "add_bos_token": True, "add_eos_token": False, "added_tokens_decoder": added, "bos_token": "<s>",
        "eos_token": "</s>", "pad_token": "<pad>", "unk_token": "<unk>", "image_token": IMAGE_TOKEN,
        "extra_special_tokens": {"image_token": IMAGE_TOKEN}, "chat_template": CHAT_TEMPLATE,
        "clean_up_tokenization_spaces": False, "model_max_length": 256, "padding_side": "left",
        "processor_class": "LlavaProcessor", "tokenizer_class": "LlamaTokenizer"})
    make_tiny.dump_json(os.path.join(folder, "special_tokens_map.json"),
                        {"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>", "unk_token": "<unk>",
                         "image_token": IMAGE_TOKEN})
    return len(vocab), ids


def text_config(vocab_size, ids):
    return dict(model_type="llama", vocab_size=vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8, hidden_act="silu", rms_norm_eps=1e-5,
                max_position_embeddings=256, rope_theta=10000.0, attention_bias=False, mlp_bias=False,
                tie_word_embeddings=False, bos_token_id=ids["<s>"], eos_token_id=ids["</s>"], pad_token_id=ids["<pad>"])


def processor_files(folder, tokens):
    make_tiny.dump_json(os.path.join(folder, "preprocessor_config.json"), {
        "crop_size": {"height": IMAGE_SIDE, "width": IMAGE_SIDE}, "do_center_crop": True, "do_convert_rgb": True,
        "do_normalize": True, "do_rescale": True, "do_resize": True, "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_processor_type": "LlavaImageProcessor", "image_std": [0.26862954, 0.26130258, 0.27577711],
        "processor_class": "LlavaProcessor", "resample": 3, "rescale_factor": 1 / 255,
        "size": {"shortest_edge": IMAGE_SIDE}})


def randomize(model, torch):
    """Every parameter from one seeded generator, in name order, at scales that keep activations near 1."""
    g = torch.Generator().manual_seed(4321)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
            n = torch.randn(p.shape, generator=g, dtype=torch.float32)
            if name.endswith("bias"):
                v = 0.1 * n
            elif "layer_norm" in name or "layrnorm" in name or "layernorm" in name:
                v = 1.0 + 0.1 * n
            elif name.endswith("norm.weight"):
                v = 1.0 + 0.3 * n                       # Llama's RMSNorm: y * w
            elif "embed_tokens" in name or "class_embedding" in name or "position_embedding" in name:
                v = 0.5 * n
            elif "patch_embedding" in name:
                v = n / np.sqrt(np.prod(p.shape[1:]))
            else:
                v = 1.5 * n / np.sqrt(p.shape[-1])      # Linear [out, in]; a little over 1 keeps greedy output varied
            p.copy_(v)


def main(out):
    import torch
    import transformers
    import tokenizers
    import safetensors
    import PIL
    from PIL import Image
    from transformers import (AutoTokenizer, CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration,
                              LlavaProcessor)
    from transformers.models.llava.image_processing_pil_llava import LlavaImageProcessorPil

    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    os.makedirs(out, exist_ok=True)
    for name in list(VARIANTS) + ["reference"]:
        shutil.rmtree(os.path.join(out, name), ignore_errors=True)
    ref = os.path.join(out, "reference")
    os.makedirs(ref)
    image_path = os.path.join(HERE, "..", "..", "tests", "Idrak.Tests", "data", "vlm", "image.png")
    with Image.open(image_path) as im:
        image = im.convert("RGB")
        image.load()

    facts, layouts = {}, {}
    for variant, choice in VARIANTS.items():
        folder = os.path.join(out, variant)
        os.makedirs(folder)
        vocab_size, ids = build_tokenizer(folder)
        processor_files(folder, None)
        config = LlavaConfig(vision_config=CLIPVisionConfig(**{k: v for k, v in VISION.items() if k != "model_type"}),
                             text_config=LlamaConfig(**{k: v for k, v in text_config(vocab_size, ids).items() if k != "model_type"}),
                             image_token_index=ids[IMAGE_TOKEN], projector_hidden_act="gelu",
                             multimodal_projector_bias=True, tie_word_embeddings=False, **choice)
        config.architectures = ["LlavaForConditionalGeneration"]
        model = LlavaForConditionalGeneration._from_config(config, attn_implementation="eager", dtype=torch.float32)
        randomize(model, torch)
        model.eval()
        model.generation_config.eos_token_id = None
        model.generation_config.pad_token_id = ids["<pad>"]
        model.save_pretrained(folder, safe_serialization=True)
        make_tiny.finish_folder(folder)

        tokenizer = AutoTokenizer.from_pretrained(folder)
        image_processor = LlavaImageProcessorPil.from_pretrained(folder)
        processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer, patch_size=PATCH,
                                   vision_feature_select_strategy=choice["vision_feature_select_strategy"],
                                   chat_template=CHAT_TEMPLATE, num_additional_image_tokens=1)
        processor.save_pretrained(folder)                  # transformers 5: the template goes to chat_template.jinja
        for config_file in ("processor_config.json", "preprocessor_config.json", "tokenizer_config.json"):
            p = os.path.join(folder, config_file)
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    make_tiny.dump_json(p, json.load(f))

        pixel_values = processor.image_processor(images=image, return_tensors="pt")["pixel_values"]
        rendered = processor.apply_chat_template(IMAGE_CHAT, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=rendered, images=[image], return_tensors="pt", add_special_tokens=False)
        assert torch.equal(inputs["pixel_values"], pixel_values)
        prompt_ids = inputs["input_ids"][0].tolist()
        tag = "" if variant == "tiny-llava" else "-full"
        with torch.no_grad():
            tower = model.model.vision_tower(pixel_values=pixel_values, output_hidden_states=True)
            embeds = model.model.vision_tower.embeddings(pixel_values)
            pre = model.model.vision_tower.pre_layrnorm(embeds)
            features = model.model.get_image_features(pixel_values=pixel_values, return_dict=True).pooler_output[0]
            layer = choice["vision_feature_layer"]
            chosen = [tower.hidden_states[i] for i in ([layer] if isinstance(layer, int) else layer)]
            if choice["vision_feature_select_strategy"] == "default":
                chosen = [h[:, 1:] for h in chosen]
            selected = torch.cat(chosen, dim=-1)
            out_img = model(**inputs)
            gen = model.generate(**inputs, max_new_tokens=GENERATED, do_sample=False, output_logits=True,
                                 return_dict_in_generate=True, eos_token_id=None, pad_token_id=ids["<pad>"])
        new_tokens = gen.sequences[0, len(prompt_ids):].tolist()
        make_tiny.save_npy(ref, f"pixel_values{tag}.npy", pixel_values.numpy())
        make_tiny.save_npy(ref, f"vision_selected{tag}.npy", selected.numpy())
        make_tiny.save_npy(ref, f"image_features{tag}.npy", features.unsqueeze(0).numpy())
        make_tiny.save_npy(ref, f"prompt-image-input_ids{tag}.npy", inputs["input_ids"].numpy().astype(np.int64))
        make_tiny.save_npy(ref, f"prompt-image-logits{tag}.npy", out_img.logits.numpy())
        make_tiny.save_npy(ref, f"prompt-image-generate-logits{tag}.npy", torch.stack(gen.logits, dim=1).numpy())
        image_positions = [i for i, t in enumerate(prompt_ids) if t == ids[IMAGE_TOKEN]]
        facts[variant] = {
            "vision_feature_layer": layer, "vision_feature_select_strategy": choice["vision_feature_select_strategy"],
            "rendered_by_chat_template": rendered, "expanded_text": processor.decode(prompt_ids),
            "input_ids": prompt_ids, "image_tokens": len(image_positions), "first_image_token": image_positions[0],
            "features_shape": list(features.shape), "new_tokens": new_tokens, "text": tokenizer.decode(new_tokens),
            "hidden_states_0_is_pre_layrnorm": bool(torch.equal(tower.hidden_states[0], pre)),
            "hidden_states_count": len(tower.hidden_states)}
        layouts[variant] = {"tensors": make_tiny.safetensors_index(os.path.join(folder, "model.safetensors"))[0]}

    # The image's pixels after the resize by shortest edge (37 x 28 for 100 x 75) and the center crop, as checks.
    processor_pixels = np.load(os.path.join(ref, "pixel_values.npy"))[0]
    resized = image.resize((int(IMAGE_SIDE * image.width / image.height), IMAGE_SIDE), Image.BICUBIC)
    left = (resized.width - IMAGE_SIDE) // 2
    crop = np.asarray(resized)[:, left:left + IMAGE_SIDE]
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    expected = (((crop.astype(np.float64) * (1 / 255)).astype(np.float32) - mean) / std).transpose(2, 0, 1)
    facts["pixels_check"] = {"resized": [resized.width, resized.height], "crop_left": left,
                             "equals_PIL_bicubic_resize_then_center_crop": bool(np.allclose(processor_pixels, expected, atol=1e-6))}

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
        "versions": versions, "image": "../vlm/image.png", "token_ids": ids, "facts": facts, "layouts": layouts,
        "files": files})
    print(json.dumps({"total_bytes": sum(f["bytes"] for f in files.values())}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", help="output folder (tests/Idrak.Tests/data/vlm-llava)")
    main(ap.parse_args().out)
