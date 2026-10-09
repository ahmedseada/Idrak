"""A tiny vision fine-tune in transformers: the reference for tuning vision-language models (plan 12, phase 0).

    python -m venv vlm && vlm/bin/pip install torch transformers tokenizers safetensors pillow numpy jinja2
    pip download llamafactory==0.9.5 --no-deps -d lf      # read only (never installed): its gemma3 template is checked
    vlm/bin/python tools/vlm/make_tiny_tuning.py tests/Idrak.Tests/data/vlm-tuning --llamafactory-wheel lf/llamafactory-0.9.5-py3-none-any.whl

Starts from plan 11's tiny models (tests/Idrak.Tests/data/vlm/tiny-gemma3 and vlm-llava/tiny-llava; nothing is
downloaded) and fine-tunes each on three image-answer records with a hand-written LoRA (no peft) on q, k, v and o of the
text decoder, the vision tower frozen: one run with the projector frozen ("lora") and one with it trained too
("projector"). One step is all records in one batch, the loss the mean token cross-entropy over the trained tokens of the
batch; plain SGD with a fixed rate, no clipping, no schedule, three steps. Written: the records (messages, the rendered
and expanded text, token ids, the trained mask, image positions, LlamaFactory's ShareGPT form and its labels), the
adapters' initial values (a PEFT folder Idrak's LoadAdapter reads), the frozen image features, the loss of every step,
the gradients of step 1 (adapters, and the projector in the projector run) and the adapters (and projector) after step 3.

The masking (plan 12's decision): every prompt token untrained (the system and user text, the image tokens, the image's
begin and end tokens and the "\n\n" around them, the turn headers); the answer and the token that ends the assistant's
turn trained (Gemma 3: <end_of_turn>); the "\n" the template writes after <end_of_turn> untrained. That is the rule of
Idrak's ChatTranscriptEncoder (the assistant's span runs from the template's generation prompt to the end-of-turn text,
trimmed), reproduced here from the template alone. LlamaFactory's gemma3 template (0.9.5, read from its wheel; see
llamafactory_gemma3) gives the same token ids for each record and trains one token more: the "\n" after <end_of_turn>
(its assistant slot is "{{content}}<end_of_turn>\n"). Both masks are written.

Reruns write the same bytes (fixed seeds, one thread, deterministic algorithms, sorted JSON) on the same machine and
library versions; manifest.json records the versions and each file's SHA-256.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_tiny  # noqa: E402  (JSON and .npy writing, the Gemma 3 chat template)
import make_tiny_llava  # noqa: E402  (the LLaVA chat template)

DATA = os.path.normpath(os.path.join(HERE, "..", "..", "tests", "Idrak.Tests", "data"))
RANK, ALPHA = 2, 4.0                       # scale alpha / rank = 2
LEARNING_RATE = 0.2
STEPS = 3
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
INIT_SEED = 2026

# The records: (system or None, the user's parts, the answer). Images by file name in tests/Idrak.Tests/data/vlm (PNGs,
# so every decoder gives the same pixels). The words are the tiny tokenizers' corpus, so most are whole tokens.
RECORDS = [
    ("Read the scan.", [("image", "image.png"), ("text", "What is in this image?")],
     "A page shows dark strokes on a light ground ; a red mark and a blue mark ."),
    (None, [("image", "image-palette.png"), ("text", "Read the scan.")],
     "a red mark and a blue mark ."),
    (None, [("image", "image.png"), ("text", "the image"), ("image", "image-palette.png"), ("text", "What is in this image?")],
     "the image is a scan of a page with lines of text ."),
]

USER_PROBE, ASSISTANT_PROBE = "⁠ns-user-probe⁠", "⁠ns-assistant-probe⁠"   # ChatTranscriptEncoder's

MODELS = {
    "gemma3": dict(folder="vlm/tiny-gemma3", template=make_tiny.CHAT_TEMPLATE, marker="<start_of_image>",
                   projector=["model.multi_modal_projector.mm_input_projection_weight",
                              "model.multi_modal_projector.mm_soft_emb_norm.weight"]),
    "llava": dict(folder="vlm-llava/tiny-llava", template=make_tiny_llava.CHAT_TEMPLATE, marker="<image>",
                  projector=["multi_modal_projector.linear_1.bias", "multi_modal_projector.linear_1.weight",
                             "multi_modal_projector.linear_2.bias", "multi_modal_projector.linear_2.weight"]),
}


def messages_of(record):
    """The record as chat messages: the system's and the assistant's content as strings, the user's as parts."""
    system, parts, answer = record
    user = [{"type": "image"} if kind == "image" else {"type": "text", "text": value} for kind, value in parts]
    return ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user}, {"role": "assistant", "content": answer}]


def images_of(record):
    return [value for kind, value in record[1] if kind == "image"]


# ------------------------------------------------------------------ Idrak's masking, from the template alone

class TemplateSpans:
    """ChatTranscriptEncoder's rule (src/Idrak.Nlp/FineTuning.cs): the assistant's header is what the generation prompt
    adds, its end the text after the answer in a probe conversation (trimmed; the EOS token when nothing is left); each
    assistant span runs from after the header to after the end (or to the text's end when the end is not found)."""

    def __init__(self, tokenizer, template, eos_token):
        def render(messages, gen):
            return tokenizer.apply_chat_template(messages, chat_template=template, tokenize=False,
                                                 add_generation_prompt=gen)
        self.render = lambda messages: render(messages, False)
        user = {"role": "user", "content": USER_PROBE}
        prompt, bare = render([user], True), render([user], False)
        pair = render([user, {"role": "assistant", "content": ASSISTANT_PROBE}], False)
        answer = pair.index(ASSISTANT_PROBE)
        header = prompt[len(bare):] if prompt.startswith(bare) else ""
        if not header.strip():
            header = pair[pair.index(USER_PROBE) + len(USER_PROBE):answer].strip()
        end = pair[answer + len(ASSISTANT_PROBE):].rstrip()
        self.header, self.end = header, (end if end else eos_token)

    def spans(self, text):
        out, at = [], text.find(self.header)
        while at >= 0:
            start = at + len(self.header)
            end = text.find(self.end, start) if self.end else -1
            end = len(text) if end < 0 else end + len(self.end)
            out.append((start, end))
            at = text.find(self.header, end)
        return out


def encode_segments(tokenizer, segments):
    """Token ids and trained flags of (text, trained) segments, each tokenized on its own (as the encoder does)."""
    ids, trained = [], []
    for text, train in segments:
        if text:
            part = tokenizer(text, add_special_tokens=False)["input_ids"]
            ids += part
            trained += [train] * len(part)
    return ids, trained


# ------------------------------------------------------------------ LlamaFactory's gemma3 template (0.9.5), reproduced

LF_FILES = ("llamafactory/data/template.py", "llamafactory/data/mm_plugin.py",
            "llamafactory/data/processor/supervised.py", "llamafactory/extras/constants.py")


def llamafactory_source(wheel):
    """The wheel's version, SHA-256 and the lines this reproduction follows (asserted, so a different wheel fails)."""
    with open(wheel, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    with zipfile.ZipFile(wheel) as z:
        text = {name: z.read(name).decode("utf-8") for name in LF_FILES}
    template = text["llamafactory/data/template.py"]
    gemma3 = template[template.index('    name="gemma3",'):]
    gemma3 = gemma3[:gemma3.index("\n)\n")]
    for line in ['format_user=StringFormatter(slots=["<start_of_turn>user\\n{{content}}<end_of_turn>\\n<start_of_turn>model\\n"])',
                 'format_assistant=StringFormatter(slots=["{{content}}<end_of_turn>\\n"])',
                 'format_system=StringFormatter(slots=["{{content}}\\n\\n"])',
                 'format_prefix=EmptyFormatter(slots=[{"bos_token"}])',
                 'mm_plugin=get_mm_plugin("gemma3", image_token="<image_soft_token>")',
                 "template_class=Llama2Template"]:
        assert line in gemma3, f"llamafactory's gemma3 template changed: {line}"
    assert "elements += self.format_user.apply(content=system_text + message[\"content\"])" in template
    assert "token_ids += tokenizer.encode(elem, add_special_tokens=False)" in template
    plugin = text["llamafactory/data/mm_plugin.py"]
    assert 'image_str = full_image_sequence if self.expand_mm_tokens else boi_token' in plugin
    assert 'IMAGE_PLACEHOLDER = os.getenv("IMAGE_PLACEHOLDER", "<image>")' in text["llamafactory/extras/constants.py"]
    supervised = text["llamafactory/data/processor/supervised.py"]
    assert "source_label = [IGNORE_INDEX] * source_len" in supervised and "target_label = target_ids" in supervised
    version = os.path.basename(wheel).split("-")[1]
    return {"version": version, "wheel": os.path.basename(wheel), "wheel_sha256": digest,
            "gemma3_template": gemma3.strip() + "\n)"}


def sharegpt_of(record):
    """The record in LlamaFactory's ShareGPT format: an <image> placeholder per image in the human turn, an images list."""
    system, parts, answer = record
    value = "".join("<image>" if kind == "image" else text for kind, text in parts)
    out = {"conversations": [{"from": "human", "value": value}, {"from": "gpt", "value": answer}],
           "images": images_of(record)}
    if system:
        out["system"] = system
    return out


def llamafactory_encode(tokenizer, processor, sharegpt):
    """LlamaFactory 0.9.5's supervised encoding of one single-turn ShareGPT record with the gemma3 template
    (Gemma3Plugin.process_messages, Llama2Template._encode, SupervisedDatasetProcessor._encode_data_example; no
    cutoff, train_on_prompt and mask_history off, efficient_eos off): input ids and labels (-100 untrained)."""
    human, gpt = sharegpt["conversations"]
    assert human["value"].count("<image>") == len(sharegpt["images"])
    user = human["value"].replace("<image>", "{{image}}").replace("{{image}}", processor.full_image_sequence)
    system_text = sharegpt["system"] + "\n\n" if sharegpt.get("system") else ""
    prompt = [tokenizer.bos_token_id] + tokenizer.encode(
        "<start_of_turn>user\n" + system_text + user + "<end_of_turn>\n<start_of_turn>model\n", add_special_tokens=False)
    response = tokenizer.encode(gpt["value"] + "<end_of_turn>\n", add_special_tokens=False)
    return prompt + response, [-100] * len(prompt) + response


# ------------------------------------------------------------------ LoRA by hand

def lora_linear(torch, base, a, b, scale):
    class LoraLinear(torch.nn.Module):
        """base(x) + scale · (x Aᵀ) Bᵀ, A [rank, in] and B [out, rank] as peft stores them."""

        def __init__(self):
            super().__init__()
            self.base = base
            self.lora_A = torch.nn.Parameter(a.clone())
            self.lora_B = torch.nn.Parameter(b.clone())

        def forward(self, x):
            return self.base(x) + scale * ((x @ self.lora_A.t()) @ self.lora_B.t())

    return LoraLinear()


def save_adapter(folder, adapters, base_name, torch):
    from safetensors.torch import save_file
    os.makedirs(folder, exist_ok=True)
    save_file({k: v.detach().contiguous() for k, v in sorted(adapters.items())},
              os.path.join(folder, "adapter_model.safetensors"), metadata={"format": "pt"})
    make_tiny.dump_json(os.path.join(folder, "adapter_config.json"), {
        "base_model_name_or_path": base_name, "bias": "none", "fan_in_fan_out": False, "inference_mode": False,
        "lora_alpha": ALPHA, "lora_dropout": 0.0, "peft_type": "LORA", "r": RANK, "target_modules": sorted(TARGETS),
        "task_type": "CAUSAL_LM", "use_dora": False, "use_rslora": False})


# ------------------------------------------------------------------ one model

def run_model(name, spec, out, lf, torch):
    import transformers
    from PIL import Image
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    folder = os.path.join(DATA, spec["folder"])
    with open(os.path.join(folder, "config.json"), encoding="utf-8") as f:
        architecture = json.load(f)["architectures"][0]
    model_class = getattr(transformers, architecture)
    tokenizer = AutoTokenizer.from_pretrained(folder)
    if name == "gemma3":
        from transformers import Gemma3Processor
        from transformers.models.gemma3.image_processing_pil_gemma3 import Gemma3ImageProcessorPil
        processor = Gemma3Processor(image_processor=Gemma3ImageProcessorPil.from_pretrained(folder), tokenizer=tokenizer,
                                    chat_template=spec["template"], image_seq_length=make_tiny.MM_TOKENS)
        pixel_kw = dict(do_convert_rgb=True)
    else:
        from transformers import LlavaProcessor
        from transformers.models.llava.image_processing_pil_llava import LlavaImageProcessorPil
        processor = LlavaProcessor(image_processor=LlavaImageProcessorPil.from_pretrained(folder), tokenizer=tokenizer,
                                   patch_size=make_tiny_llava.PATCH, vision_feature_select_strategy="default",
                                   chat_template=spec["template"], num_additional_image_tokens=1)
        pixel_kw = {}
    image_token = tokenizer.convert_tokens_to_ids("<image_soft_token>" if name == "gemma3" else "<image>")

    def load_image(file):
        with Image.open(os.path.join(DATA, "vlm", file)) as im:
            im.load()
            return im.copy()

    spans_rule = TemplateSpans(tokenizer, spec["template"], tokenizer.eos_token)
    records, inputs_all = [], []
    for index, record in enumerate(RECORDS):
        messages = messages_of(record)
        files = images_of(record)
        images = [load_image(f) for f in files]
        text = spans_rule.render(messages)
        spans = spans_rule.spans(text)
        assert len(spans) == 1, f"{name} record {index}: {spans}"
        (start, end), = spans
        # The markers are all in the prompt (before the answer): expanded there as the processor expands them.
        prompt = text[:start]
        if name == "gemma3":
            expanded_prompt = prompt.replace(spec["marker"], processor.full_image_sequence)
        else:
            one = processor(text=spec["marker"], images=[images[:1]], add_special_tokens=False)["input_ids"][0]
            expanded_prompt = prompt.replace(spec["marker"], spec["marker"] * list(one).count(image_token))
        segments = [(expanded_prompt, False), (text[start:end], True), (text[end:], False)]
        ids, trained = encode_segments(tokenizer, segments)
        expanded = "".join(s for s, _ in segments)

        # transformers' processor on the whole rendered conversation: the same ids (the stand-in for Idrak's rendering).
        rendered_by_processor = processor.apply_chat_template(messages, tokenize=False)
        assert rendered_by_processor == text, f"{name} record {index}: processor and tokenizer render differently"
        batch = processor(text=text, images=[images], return_tensors="pt", add_special_tokens=False, **pixel_kw)
        assert batch["input_ids"][0].tolist() == ids, (
            f"{name} record {index}: segment-wise ids differ from the processor's\n{ids}\n{batch['input_ids'][0].tolist()}")
        assert tokenizer.decode(ids, skip_special_tokens=False) == expanded or name == "llava"

        positions, i = [], 0
        while i < len(ids):
            if ids[i] == image_token:
                j = i
                while j < len(ids) and ids[j] == image_token:
                    j += 1
                positions.append({"position": i, "tokens": j - i})
                i = j
            else:
                i += 1
        assert len(positions) == len(files), f"{name} record {index}: {positions}"

        entry = {"messages": messages, "images": files, "rendered": text, "assistant_span": [start, end],
                 "expanded": expanded, "input_ids": ids, "trained": trained,
                 "trained_tokens": sum(trained[1:]), "image_blocks": positions,
                 "tokens": tokenizer.convert_ids_to_tokens(ids)}
        if name == "gemma3":
            entry["token_type_ids"] = batch["token_type_ids"][0].tolist()
            share = sharegpt_of(record)
            lf_ids, lf_labels = llamafactory_encode(tokenizer, processor, share)
            assert lf_ids == ids, f"record {index}: LlamaFactory's ids differ\n{lf_ids}\n{ids}"
            lf_trained = [label != -100 for label in lf_labels]
            assert all(lf_labels[k] in (-100, ids[k]) for k in range(len(ids)))
            entry["sharegpt"] = share
            entry["llamafactory"] = {"input_ids": lf_ids, "labels": lf_labels, "trained": lf_trained,
                                     "same_ids": True,
                                     "differs_from_trained_at": [k for k in range(len(ids)) if lf_trained[k] != trained[k]]}
        records.append(entry)
        inputs_all.append(batch)

    # The model, a copy per run (the projector run changes the projector).
    def load_model():
        m = model_class.from_pretrained(folder, attn_implementation="eager", dtype=torch.float32)
        m.eval()                                          # no dropout anywhere (all 0); the masks do not depend on it
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    model = load_model()
    language = model.model.language_model
    layers = list(language.layers)
    # Checkpoint names of the adapted projections (what Idrak's LoadAdapter looks up: "base_model.model." + name).
    checkpoint = make_tiny.safetensors_index(os.path.join(folder, "model.safetensors"))[0]
    prefix = next(k for k in checkpoint if k.endswith("layers.0.self_attn.q_proj.weight"))[:-len("0.self_attn.q_proj.weight")]

    g = torch.Generator().manual_seed(INIT_SEED)
    init = {}
    for li, layer in enumerate(layers):
        for t in TARGETS:
            base = getattr(layer.self_attn, t)
            key = f"base_model.model.{prefix}{li}.self_attn.{t}"
            a = (torch.rand((RANK, base.in_features), generator=g, dtype=torch.float32) * 2 - 1) / np.sqrt(base.in_features)
            b = (torch.rand((base.out_features, RANK), generator=g, dtype=torch.float32) * 2 - 1) * 0.2
            init[key + ".lora_A.weight"] = a
            init[key + ".lora_B.weight"] = b
    base_name = os.path.basename(spec["folder"])          # as Idrak's SaveAdapter names the base (the folder's name)
    save_adapter(os.path.join(out, name, "adapter-init"), init, base_name, torch)

    # The frozen image features of each image (with the checkpoint's projector).
    feature_dir = os.path.join(out, name, "features")
    os.makedirs(feature_dir, exist_ok=True)
    with torch.no_grad():
        for file in sorted({f for r in RECORDS for f in images_of(r)}):
            pv = processor.image_processor(images=load_image(file), return_tensors="pt", **pixel_kw)["pixel_values"]
            feats = model.model.get_image_features(pixel_values=pv, return_dict=True).pooler_output
            feats = feats[0].unsqueeze(0) if isinstance(feats, (list, tuple)) else feats    # LLaVA: one per image
            make_tiny.save_npy(feature_dir, file.rsplit(".", 1)[0] + ".npy", feats.numpy())

    runs = {}
    for run in ("lora", "projector"):
        model = load_model()
        layers = list(model.model.language_model.layers)
        adapters = {}
        for li, layer in enumerate(layers):
            for t in TARGETS:
                key = f"base_model.model.{prefix}{li}.self_attn.{t}"
                wrapped = lora_linear(torch, getattr(layer.self_attn, t), init[key + ".lora_A.weight"],
                                      init[key + ".lora_B.weight"], ALPHA / RANK)
                setattr(layer.self_attn, t, wrapped)
                adapters[key + ".lora_A.weight"] = wrapped.lora_A
                adapters[key + ".lora_B.weight"] = wrapped.lora_B
        projector = {}
        if run == "projector":
            named = dict(model.named_parameters())
            for k in spec["projector"]:
                module_name = k if k in named else "model." + k
                p = named[module_name]
                p.requires_grad_(True)
                projector[k] = p
        params = {**adapters, **projector}

        losses, step1 = [], None
        for step in range(STEPS):
            for p in params.values():
                p.grad = None
            total = sum(r["trained_tokens"] for r in records)
            loss_sum = torch.zeros((), dtype=torch.float32)
            per_record = []
            for r, batch in zip(records, inputs_all):
                ids = batch["input_ids"]
                kw = {"token_type_ids": batch["token_type_ids"]} if "token_type_ids" in batch else {}
                logits = model(input_ids=ids, pixel_values=batch["pixel_values"],
                               attention_mask=batch["attention_mask"], **kw).logits[0].float()
                ce = torch.nn.functional.cross_entropy(logits[:-1], ids[0, 1:], reduction="none")
                mask = torch.tensor(r["trained"][1:], dtype=torch.float32)
                loss_sum = loss_sum + (ce * mask).sum()
                per_record.append({"cross_entropy_sum": float((ce * mask).sum().detach()),
                                   "trained_cross_entropy": [float(v) for v, m in zip(ce.tolist(), r["trained"][1:]) if m]})
            loss = loss_sum / total
            loss.backward()
            losses.append(loss.item())
            if step == 0:
                step1 = {"per_record": per_record,
                         "grads": {k: p.grad.detach().clone() for k, p in params.items()},
                         "grad_norm": float(torch.sqrt(sum((p.grad.double() ** 2).sum() for p in params.values())))}
            with torch.no_grad():
                for p in params.values():
                    p -= LEARNING_RATE * p.grad
        run_dir = os.path.join(out, name, run)
        os.makedirs(run_dir, exist_ok=True)
        save_file({k: v.contiguous() for k, v in sorted(step1["grads"].items())},
                  os.path.join(run_dir, "grads-step1.safetensors"), metadata={"format": "pt"})
        save_adapter(os.path.join(run_dir, f"adapter-step{STEPS}"), adapters, base_name, torch)
        if projector:
            save_file({k: v.detach().contiguous() for k, v in sorted(projector.items())},
                      os.path.join(run_dir, f"projector-step{STEPS}.safetensors"), metadata={"format": "pt"})
        runs[run] = {"trained": sorted(params), "losses": losses, "step1_grad_norm": step1["grad_norm"],
                     "step1_per_record": step1["per_record"],
                     "files": {"grads_step1": f"{run}/grads-step1.safetensors",
                               "adapter_after": f"{run}/adapter-step{STEPS}",
                               **({"projector_after": f"{run}/projector-step{STEPS}.safetensors"} if projector else {})}}
        print(f"{name} {run}: losses {losses}", file=sys.stderr)

    settings = {"model": "../" + spec["folder"], "architecture": architecture, "checkpoint_prefix": prefix,
                "lora": {"rank": RANK, "alpha": ALPHA, "scale": ALPHA / RANK, "dropout": 0.0, "targets": TARGETS,
                         "init": f"torch.Generator().manual_seed({INIT_SEED}); per layer, per target in order q, k, v, o: "
                                 "A = U(-1, 1) / sqrt(in) [rank, in], then B = 0.2 U(-1, 1) [out, rank] "
                                 "(B is not zero, so step 1 has gradients for A too)"},
                "optimizer": {"kind": "sgd", "learning_rate": LEARNING_RATE, "momentum": 0.0, "weight_decay": 0.0,
                              "max_gradient_norm": 0.0, "schedule": "constant"},
                "steps": STEPS, "batch": "all records in one batch, every step, in order",
                "loss": "sum over records and positions t of trained[t + 1] * cross_entropy(logits[t], input_ids[t + 1]), "
                        "divided by the batch's trained tokens (sum of trained_tokens)",
                "tower": "frozen", "image_token": image_token,
                "assistant_header": spans_rule.header, "assistant_end": spans_rule.end,
                "projector_parameters": spec["projector"]}
    result = {"settings": settings, "records": records, "runs": runs}
    if name == "gemma3":
        result["llamafactory_gemma3"] = lf
    make_tiny.dump_json(os.path.join(out, name, "tuning.json"), result)
    return {"losses": {k: v["losses"] for k, v in runs.items()}}


def main(out, wheel):
    import torch
    import transformers
    import tokenizers
    import safetensors
    import PIL

    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    lf = llamafactory_source(wheel)
    os.makedirs(out, exist_ok=True)
    for name in MODELS:
        shutil.rmtree(os.path.join(out, name), ignore_errors=True)
    summary = {name: run_model(name, spec, out, lf, torch) for name, spec in MODELS.items()}

    versions = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
                "tokenizers": tokenizers.__version__, "safetensors": safetensors.__version__, "pillow": PIL.__version__,
                "numpy": np.__version__, "llamafactory_read_from_wheel": lf["version"]}
    files = {}
    for root, dirs, names in os.walk(out):
        dirs.sort()
        for n in sorted(names):
            p = os.path.join(root, n)
            rel = os.path.relpath(p, out).replace(os.sep, "/")
            if rel in ("manifest.json", "README.md"):
                continue
            files[rel] = {"bytes": os.path.getsize(p), "sha256": make_tiny.sha256(p)}
    tensors = {rel: make_tiny.safetensors_index(os.path.join(out, rel))[0] for rel in files if rel.endswith(".safetensors")}
    make_tiny.dump_json(os.path.join(out, "manifest.json"), {
        "versions": versions, "images": "../vlm/", "summary": summary, "tensors": tensors, "files": files})
    print(json.dumps({"total_bytes": sum(f["bytes"] for f in files.values()), **summary}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", help="output folder (tests/Idrak.Tests/data/vlm-tuning)")
    ap.add_argument("--llamafactory-wheel", required=True,
                    help="llamafactory-0.9.5-py3-none-any.whl (pip download llamafactory==0.9.5 --no-deps); read, never installed")
    args = ap.parse_args()
    main(args.out, args.llamafactory_wheel)
