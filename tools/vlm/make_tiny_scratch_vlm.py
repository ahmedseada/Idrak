"""The reference for vision-language models made from scratch: a tiny IdrakVision model (plan 19, phase B0).

    vlm/bin/python tools/vlm/make_tiny_scratch_vlm.py tests/Idrak.Tests/data/vlm-scratch

IdrakVision is Idrak's own family (src/Idrak.Vision.Plugins/IdrakVision): a SigLIP-style tower (a patch convolution with a bias, learned
positions, pre-LayerNorm encoder layers with biased attention and a GELU-tanh MLP, a final LayerNorm), the patches
average-pooled pool_size x pool_size, a projector of linear layers with GELU-tanh between them, and a Llama decoder.
Each image becomes <start_of_image>, mm_tokens_per_image soft tokens (their embeddings the projected features,
unscaled) and <end_of_image>; the soft tokens of one image attend to each other in both directions, the rest is causal.

Nothing is downloaded: the decoder's sizes and tokenizer are the from-scratch text reference's (../pretrain), the images
the vision reference's (../vlm), every weight drawn here from one seeded generator. The script writes:

- tiny-idrak-vision/: the model folder (config.json, model.safetensors, tokenizer, ChatML template with image parts,
  preprocessor_config.json written by transformers' SiglipImageProcessor);
- records.jsonl and images/: three conversations with images (messages format), and tuning.json: each record's
  rendered and expanded text, ids, trained mask and image blocks;
- reference/: per image its pixel values, tower output and features; the first record's logits;
- joint/: one step with every parameter trained (SGD 0.05, the mean token cross entropy over the three records'
  trained tokens): every gradient, three steps' losses, the weights after them; align/: the same with the decoder frozen
  (tower and projector trained).

Reruns write the same bytes (fixed seeds, one thread, deterministic algorithms, sorted JSON) on the same machine and
library versions; manifest.json records the versions and each file's SHA-256. README.md (by hand) is kept.
"""
import argparse
import json
import math
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_tiny import dump_json, save_npy, sha256, safetensors_index  # noqa: E402
from make_tiny_pretrain import CHAT_TEMPLATE  # noqa: E402
from make_tiny_pretrain import TEXT as PRETRAIN_TEXT  # noqa: E402

ARCHITECTURE = "IdrakVisionForConditionalGeneration"
TEXT = dict(PRETRAIN_TEXT, max_position_embeddings=128)     # the records with their images run past the text reference's 64
VISION = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=3, num_attention_heads=2, num_channels=3, image_size=28,
              patch_size=7, layer_norm_eps=1e-6, hidden_act="gelu_pytorch_tanh", pool_size=2)
PROJECTOR = dict(hidden_act="gelu_pytorch_tanh", depth=2, bias=True)
TOKENS_PER_IMAGE = (VISION["image_size"] // VISION["patch_size"] // VISION["pool_size"]) ** 2
RATE = 0.05
STEPS = 3
SEED = 2027

RECORDS = [
    {"messages": [{"role": "user", "content": [{"type": "image", "path": "images/image.png"}, {"type": "text", "text": "Read the scan."}]},
                  {"role": "assistant", "content": "the court decision 12/05/1446"}]},
    {"messages": [{"role": "system", "content": "You read pages."},
                  {"role": "user", "content": [{"type": "text", "text": "What is in this image?"}, {"type": "image", "path": "images/image-palette.png"}]},
                  {"role": "assistant", "content": "a page with lines of text"}]},
    {"messages": [{"role": "user", "content": [{"type": "image", "path": "images/image.png"}, {"type": "image", "path": "images/image-palette.png"},
                                               {"type": "text", "text": "Compare."}]},
                  {"role": "assistant", "content": "two pages, one in few colors"}]},
]


def build(torch, vocab, text_config):
    nn, F = torch.nn, torch.nn.functional
    from transformers import LlamaForCausalLM
    v = VISION
    D, H = v["hidden_size"], v["num_attention_heads"]

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(D, D) for _ in range(4))

        def forward(self, x):
            b, n, _ = x.shape
            split = lambda t: t.view(b, n, H, D // H).transpose(1, 2)
            q, k, val = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
            w = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(D // H), dim=-1)
            return self.out_proj((w @ val).transpose(1, 2).reshape(b, n, D))

    class Mlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1, self.fc2 = nn.Linear(D, v["intermediate_size"]), nn.Linear(v["intermediate_size"], D)

        def forward(self, x):
            return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_norm1, self.self_attn = nn.LayerNorm(D, eps=v["layer_norm_eps"]), Attention()
            self.layer_norm2, self.mlp = nn.LayerNorm(D, eps=v["layer_norm_eps"]), Mlp()

        def forward(self, x):
            x = x + self.self_attn(self.layer_norm1(x))
            return x + self.mlp(self.layer_norm2(x))

    class Embeddings(nn.Module):
        def __init__(self):
            super().__init__()
            p = v["patch_size"]
            self.patch_embedding = nn.Conv2d(v["num_channels"], D, p, stride=p)
            self.position_embedding = nn.Embedding((v["image_size"] // p) ** 2, D)

        def forward(self, pixels):
            return self.patch_embedding(pixels).flatten(2).transpose(1, 2) + self.position_embedding.weight

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(Layer() for _ in range(v["num_hidden_layers"]))

    class Tower(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings, self.encoder = Embeddings(), Encoder()
            self.post_layernorm = nn.LayerNorm(D, eps=v["layer_norm_eps"])

        def forward(self, pixels):
            x = self.embeddings(pixels)
            for layer in self.encoder.layers:
                x = layer(x)
            return self.post_layernorm(x)

    class Projector(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.linear_1, self.linear_2 = nn.Linear(D, width), nn.Linear(width, width)

        def forward(self, x):
            b, n, d = x.shape
            side, k = int(math.isqrt(n)), v["pool_size"]
            x = x.view(b, side // k, k, side // k, k, d).mean(dim=(2, 4)).reshape(b, -1, d)     # row-major pooled grid
            return self.linear_2(F.gelu(self.linear_1(x), approximate="tanh"))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower = Tower()
            self.multi_modal_projector = Projector(text_config.hidden_size)
            self.language_model = LlamaForCausalLM(text_config)

    return Model()


def randomize(model, torch):
    g = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
            n = torch.randn(p.shape, generator=g, dtype=torch.float32)
            if name.endswith("bias"):
                v = 0.1 * n
            elif "layer_norm" in name or "post_layernorm" in name:
                v = 1.0 + 0.1 * n
            elif name.endswith("norm.weight"):
                v = 1.0 + 0.1 * n
            elif "embed_tokens" in name or "position_embedding" in name:
                v = 0.5 * n
            elif "patch_embedding" in name:
                v = n / np.sqrt(np.prod(p.shape[1:]))
            else:
                v = n / np.sqrt(p.shape[-1])
            p.copy_(v)


def render(messages, add_generation_prompt):
    import jinja2
    env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False)
    return env.from_string(CHAT_TEMPLATE).render(messages=messages, add_generation_prompt=add_generation_prompt)


def expand(text):
    block = "<start_of_image>" + "<image>" * TOKENS_PER_IMAGE + "<end_of_image>"
    return text.replace("<image>", block)


def encode_record(record, tok, ids):
    messages = record["messages"]
    full = expand(render(messages, False))
    prefix = expand(render(messages[:-1], True))
    answer = messages[-1]["content"] + "<|im_end|>"
    assert full.startswith(prefix + answer), (full, prefix, answer)
    enc = tok.encode(full, add_special_tokens=False)
    trained = [prefix_len <= s and e <= prefix_len + len(answer) for (s, e) in enc.offsets for prefix_len in [len(prefix)]]
    image_positions = [i for i, t in enumerate(enc.ids) if t == ids["<image>"]]
    blocks = [image_positions[i:i + TOKENS_PER_IMAGE][0] for i in range(0, len(image_positions), TOKENS_PER_IMAGE)]
    images = [part["path"] for m in messages if isinstance(m["content"], list) for part in m["content"] if part["type"] == "image"]
    return {"rendered": render(messages, False), "expanded": full, "input_ids": enc.ids, "trained": trained,
            "image_blocks": [[b, TOKENS_PER_IMAGE] for b in blocks], "images": images}


def forward(torch, model, encoded, pixels_of):
    """The decoder's logits for one record: image tokens' embeddings replaced by their features, image blocks bidirectional."""
    ids = torch.tensor([encoded["input_ids"]])
    lm = model.language_model
    embeds = lm.model.embed_tokens(ids)
    features = model.multi_modal_projector(model.vision_tower(torch.cat([pixels_of[p] for p in encoded["images"]])))
    pieces, last = [], 0
    for k, (start, count) in enumerate(encoded["image_blocks"]):
        pieces += [embeds[:, last:start], features[k:k + 1]]
        last = start + count
    pieces.append(embeds[:, last:])
    embeds = torch.cat(pieces, dim=1)
    n = ids.shape[1]
    allowed = torch.tril(torch.ones(n, n, dtype=torch.bool))
    for start, count in encoded["image_blocks"]:
        allowed[start:start + count, start:start + count] = True
    mask = torch.where(allowed, 0.0, float("-inf"))[None, None]
    return lm(inputs_embeds=embeds, attention_mask=mask).logits


def loss_of(torch, model, encoded_records, pixels_of):
    total, count = 0.0, 0
    for e in encoded_records:
        logits = forward(torch, model, e, pixels_of)[0]
        targets = torch.tensor(e["input_ids"][1:])
        rows = torch.tensor([t for t in range(len(e["input_ids"]) - 1) if e["trained"][t + 1]])
        total = total + torch.nn.functional.cross_entropy(logits[rows], targets[rows], reduction="sum")
        count += len(rows)
    return total / count, count


def named(model):
    """Checkpoint names: the decoder under language_model.model., the head tied (not stored)."""
    out = {}
    for name, p in model.named_parameters():
        if name.startswith("language_model.lm_head"):
            continue
        out[name] = p
    return out


def train(torch, model_dir, encoded, pixels_of, out, trained_parts, build_model):
    from safetensors.torch import save_file, load_file
    model = build_model()
    state = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict({**state, "language_model.lm_head.weight": state["language_model.model.embed_tokens.weight"]}, strict=True)
    model.language_model.tie_weights()
    params = {n: p for n, p in named(model).items()}
    for n, p in params.items():
        p.requires_grad_(any(n.startswith(t) for t in trained_parts))
    opt = torch.optim.SGD([p for p in params.values() if p.requires_grad], lr=RATE, momentum=0.0)
    losses = []
    os.makedirs(out, exist_ok=True)
    for step in range(1, STEPS + 1):
        opt.zero_grad(set_to_none=True)
        loss, count = loss_of(torch, model, encoded, pixels_of)
        loss.backward()
        losses.append(float(loss.detach()))
        if step == 1:
            grads = {n: p.grad.detach().clone().contiguous() for n, p in params.items() if p.requires_grad}
            save_file(dict(sorted(grads.items())), os.path.join(out, "grads-step1.safetensors"), metadata={"format": "pt"})
        opt.step()
    weights = {n: p.detach().clone().contiguous() for n, p in params.items()}
    save_file(dict(sorted(weights.items())), os.path.join(out, f"weights-step{STEPS}.safetensors"), metadata={"format": "pt"})
    return {"losses": losses, "trained_tokens": count, "trained": trained_parts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    args = parser.parse_args()
    import torch
    import transformers
    import tokenizers
    import safetensors
    from PIL import Image
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from transformers import LlamaConfig, SiglipImageProcessor
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    here = os.path.dirname(os.path.abspath(__file__))
    data = os.path.normpath(os.path.join(here, "..", "..", "tests", "Idrak.Tests", "data"))
    out = os.path.abspath(args.out)
    readme = os.path.join(out, "README.md")
    kept = open(readme, encoding="utf-8").read() if os.path.exists(readme) else None
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    if kept is not None:
        with open(readme, "w", encoding="utf-8", newline="\n") as f:
            f.write(kept)

    tokenizer_path = os.path.join(data, "pretrain", "tokenizer", "tokenizer.json")
    tok = Tokenizer.from_file(tokenizer_path)
    specials = ["<pad>", "<s>", "</s>", "<|im_start|>", "<|im_end|>", "<start_of_image>", "<image>", "<end_of_image>"]
    ids = {s: tok.token_to_id(s) for s in specials}
    text_config = LlamaConfig(vocab_size=tok.get_vocab_size(), pad_token_id=ids["<pad>"], bos_token_id=ids["<s>"],
                              eos_token_id=[ids["<|im_end|>"], ids["</s>"]], attn_implementation="eager", **TEXT)

    model_dir = os.path.join(out, "tiny-idrak-vision")
    os.makedirs(model_dir)
    build_model = lambda: build(torch, tok.get_vocab_size(), text_config).float()
    model = build_model()
    randomize(model, torch)
    model.language_model.tie_weights()
    save_file({n: p.detach().contiguous() for n, p in sorted(named(model).items())}, os.path.join(model_dir, "model.safetensors"), metadata={"format": "pt"})
    text = {k: v for k, v in text_config.to_dict().items() if k in (
        "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
        "max_position_embeddings", "rms_norm_eps", "rope_theta", "hidden_act", "attention_bias", "mlp_bias", "tie_word_embeddings",
        "bos_token_id", "eos_token_id", "pad_token_id")}
    text["model_type"] = "llama"
    text["rope_theta"] = TEXT["rope_theta"]
    dump_json(os.path.join(model_dir, "config.json"), {
        "architectures": [ARCHITECTURE], "model_type": "idrak_vision", "text_config": text, "vision_config": dict(VISION),
        "projector_config": dict(PROJECTOR), "image_token_index": ids["<image>"], "boi_token_index": ids["<start_of_image>"],
        "eoi_token_index": ids["<end_of_image>"], "mm_tokens_per_image": TOKENS_PER_IMAGE, "image_attention": "image-blocks",
        "tie_word_embeddings": True, "torch_dtype": "float32"})
    shutil.copy(tokenizer_path, os.path.join(model_dir, "tokenizer.json"))
    dump_json(os.path.join(model_dir, "tokenizer_config.json"), {
        "add_bos_token": False, "bos_token": "<s>", "eos_token": "<|im_end|>", "pad_token": "<pad>", "chat_template": CHAT_TEMPLATE,
        "model_max_length": TEXT["max_position_embeddings"], "tokenizer_class": "PreTrainedTokenizerFast"})
    dump_json(os.path.join(model_dir, "special_tokens_map.json"), {"bos_token": "<s>", "eos_token": "<|im_end|>", "pad_token": "<pad>"})
    dump_json(os.path.join(model_dir, "generation_config.json"),
              {"bos_token_id": ids["<s>"], "eos_token_id": [ids["<|im_end|>"], ids["</s>"]], "pad_token_id": ids["<pad>"]})
    processor = SiglipImageProcessor(size={"height": VISION["image_size"], "width": VISION["image_size"]}, resample=Image.BILINEAR,
                                     do_rescale=True, rescale_factor=1 / 255, do_normalize=True, image_mean=[0.5] * 3, image_std=[0.5] * 3)
    processor.save_pretrained(model_dir)
    with open(os.path.join(model_dir, "preprocessor_config.json"), encoding="utf-8") as f:
        dump_json(os.path.join(model_dir, "preprocessor_config.json"), json.load(f))

    os.makedirs(os.path.join(out, "images"))
    for name in ("image.png", "image-palette.png"):
        shutil.copy(os.path.join(data, "vlm", name), os.path.join(out, "images", name))
    with open(os.path.join(out, "records.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for record in RECORDS:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    pixels_of = {}
    ref = os.path.join(out, "reference")
    os.makedirs(ref)
    with torch.no_grad():
        for name in ("images/image.png", "images/image-palette.png"):
            pv = processor(images=Image.open(os.path.join(out, name)).convert("RGB"), return_tensors="pt")["pixel_values"].float()
            pixels_of[name] = pv
            stem = os.path.splitext(os.path.basename(name))[0]
            tower = model.vision_tower(pv)
            save_npy(ref, f"{stem}-pixel_values.npy", pv.numpy())
            save_npy(ref, f"{stem}-tower.npy", tower.numpy())
            save_npy(ref, f"{stem}-features.npy", model.multi_modal_projector(tower).numpy())

    encoded = [encode_record(r, tok, ids) for r in RECORDS]
    with torch.no_grad():
        save_npy(ref, "record0-logits.npy", forward(torch, model, encoded[0], pixels_of)[0].numpy())
    dump_json(os.path.join(out, "tuning.json"), {"records": encoded, "tokens_per_image": TOKENS_PER_IMAGE, "image_token": ids["<image>"]})

    joint = train(torch, model_dir, encoded, pixels_of, os.path.join(out, "joint"),
                  ["vision_tower.", "multi_modal_projector.", "language_model."], build_model)
    align = train(torch, model_dir, encoded, pixels_of, os.path.join(out, "align"), ["vision_tower.", "multi_modal_projector."], build_model)
    dump_json(os.path.join(out, "steps.json"), {
        "rate": RATE, "steps": STEPS, "optimizer": "sgd, no momentum, decay or clipping",
        "loss": "mean cross entropy over the three records' trained tokens (the answer and <|im_end|>)",
        "joint": joint, "align": align})

    files, tensors = {}, {}
    for root, _, names in sorted(os.walk(out)):
        for name in sorted(names):
            path = os.path.join(root, name)
            rel = os.path.relpath(path, out).replace(os.sep, "/")
            if rel in ("manifest.json", "README.md"):
                continue
            files[rel] = {"bytes": os.path.getsize(path), "sha256": sha256(path)}
            if name.endswith(".safetensors"):
                tensors[rel] = safetensors_index(path)[0]
    dump_json(os.path.join(out, "manifest.json"), {
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
                     "tokenizers": tokenizers.__version__, "safetensors": safetensors.__version__, "numpy": np.__version__},
        "summary": {"joint_losses": joint["losses"], "align_losses": align["losses"]}, "tensors": tensors, "files": files})
    print(f"wrote {out}: joint {joint['losses']}, align {align['losses']}")


if __name__ == "__main__":
    main()
