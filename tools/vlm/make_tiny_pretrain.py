"""The reference for training language models from scratch (plan 19, phase A0).

    python -m venv vlm && vlm/bin/pip install torch transformers tokenizers safetensors numpy jinja2
    vlm/bin/python tools/vlm/make_tiny_pretrain.py tests/Idrak.Tests/data/pretrain

Nothing is downloaded. The script writes, from fixed word lists and seeded generators:

- corpus.txt: Arabic and English documents (lines of words, numbers and punctuation), separated by lines of "---", as
  `idrak data text` writes them;
- tokenizer/: a byte-level BPE trained by Hugging Face tokenizers from corpus.txt (train.json holds its vocabulary and
  merges in training order, its settings, and the ids of sample texts);
- metaspace/: the same with the Metaspace pre-tokenizer;
- tokenizer-full/: byte level again, merging until no pair is left (minimum frequency 1), which meets many count ties;
- tiny-llama/: a LlamaForCausalLM with every weight drawn from one seeded generator, the tokenizer and a ChatML chat
  template that walks content parts (the template plan 19 gives created models);
- tokens.bin: every document of corpus.txt encoded and followed by </s>, as uint16 (what `idrak data tokens` writes);
- step/: training on the first four 32-token windows of tokens.bin with every parameter trained, the mean token cross
  entropy over every position; plain SGD at 0.2 (step 1's loss and gradients, three steps' losses and the weights after
  them) and AdamW (three steps' losses, the weights and the optimizer's moments after them); adamw-folded/ the same with
  the bias corrections folded into the step size, as Idrak's AdamW computes it.

Reruns write the same bytes (fixed seeds, one thread, deterministic algorithms, sorted JSON) on the same machine and
library versions; manifest.json records the versions and each file's SHA-256. README.md in the output folder (written by
hand) says how the C# tests read the files.
"""
import argparse
import json
import os
import random
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_tiny import dump_json, sha256, safetensors_index  # noqa: E402

SPECIALS = ["<pad>", "<s>", "</s>", "<|im_start|>", "<|im_end|>", "<start_of_image>", "<image>", "<end_of_image>"]
VOCAB = 512
MIN_FREQUENCY = 2
TEXT = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=64, rms_norm_eps=1e-6, rope_theta=10000.0, hidden_act="silu",
            attention_bias=False, mlp_bias=False, tie_word_embeddings=True)
WINDOW = 32
WINDOWS = 4
STEPS = 3
SGD_RATE = 0.2
ADAMW = dict(lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
INIT_SEED = 2026

# The template plan 19 gives created models (src/Idrak.Nlp/Resources/chatml-parts.jinja): ChatML turns, content as a
# string or a list of parts, an image part written as the family's marker (<image> here) and an audio part as <audio>.
CHAT_TEMPLATE = (
    "{%- for message in messages -%}\n"
    "{{- '<|im_start|>' + message['role'] + '\\n' -}}\n"
    "{%- if message['content'] is string -%}\n"
    "{{- message['content'] -}}\n"
    "{%- else -%}\n"
    "{%- for item in message['content'] -%}\n"
    "{%- if item['type'] == 'image' -%}\n"
    "{{- '<image>' -}}\n"
    "{%- elif item['type'] == 'audio' -%}\n"
    "{{- '<audio>' -}}\n"
    "{%- elif item['type'] == 'text' -%}\n"
    "{{- item['text'] -}}\n"
    "{%- endif -%}\n"
    "{%- endfor -%}\n"
    "{%- endif -%}\n"
    "{{- '<|im_end|>\\n' -}}\n"
    "{%- endfor -%}\n"
    "{%- if add_generation_prompt -%}\n"
    "{{- '<|im_start|>assistant\\n' -}}\n"
    "{%- endif -%}\n")

ARABIC = ("وزارة العدل المحكمة القاضي الدعوى الحكم الطلب الموضوع التاريخ السيد المحترم رقم القرار الجلسة الشاهد "
          "العقد البيع الشراء الأرض المنزل المبلغ الريال الشهر السنة اليوم المدينة الرياض جدة مكة الشرقية الكتاب "
          "الخطاب المرفق النسخة الأصل التوقيع الختم الإدارة العامة الشؤون القانونية النظام المادة الفقرة البند "
          "الموافقة الرفض التنفيذ الاستئناف الطعن المدعي المدعى عليه الوكيل الشركة المؤسسة الفرع").split()
ENGLISH = ("the court judge case ruling request subject date mister honorable number decision session witness contract "
           "sale purchase land house amount riyal month year day city office letter attachment copy original signature "
           "stamp general administration legal affairs system article paragraph clause approval refusal execution "
           "appeal claim plaintiff defendant agent company establishment branch page line scan read write").split()
PUNCTUATION = [".", ",", ":", ";", "؟", "،", "(", ")", "-", "/"]


def corpus(seed=2026, documents=96):
    """Documents of 6 to 10 lines; a line is 6 to 14 words of one language, with numbers, dates and punctuation."""
    r = random.Random(seed)
    docs = []
    for d in range(documents):
        words = ARABIC if d % 2 == 0 else ENGLISH
        lines = []
        for _ in range(r.randint(6, 10)):
            parts = []
            for _ in range(r.randint(6, 14)):
                x = r.random()
                if x < 0.08:
                    parts.append(str(r.randint(1, 9999)))
                elif x < 0.12:
                    parts.append(f"{r.randint(1, 28):02d}/{r.randint(1, 12):02d}/{r.randint(1400, 1447)}")
                else:
                    parts.append(r.choice(words))
                if r.random() < 0.1:
                    parts[-1] += r.choice(PUNCTUATION)
            if r.random() < 0.15:
                parts[0] = parts[0].capitalize()
            lines.append(" ".join(parts))
        docs.append("\n".join(lines))
    return docs


def train_tokenizer(corpus_path, out, metaspace, vocab=VOCAB, min_frequency=MIN_FREQUENCY):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    if metaspace:
        tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=True)
        tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always", split=True)
        alphabet = []
    else:
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)
        tok.decoder = decoders.ByteLevel()
        alphabet = pre_tokenizers.ByteLevel.alphabet()
    trainer = trainers.BpeTrainer(vocab_size=vocab, min_frequency=min_frequency, special_tokens=SPECIALS,
                                  initial_alphabet=alphabet, show_progress=False)
    tok.train([corpus_path], trainer)
    os.makedirs(out, exist_ok=True)
    tok.save(os.path.join(out, "tokenizer.json"), pretty=True)
    with open(os.path.join(out, "tokenizer.json"), encoding="utf-8") as f:
        model = json.load(f)["model"]
    return tok, model


def write_tokenizer_reference(tok, model, out, docs, metaspace, vocab=VOCAB, min_frequency=MIN_FREQUENCY):
    samples = [line for doc in docs[:4] for line in doc.split("\n")][:16]
    samples += ["", " ", "  two  spaces", "<|im_start|>user\nread the scan<|im_end|>\n", "رقم 1446/05/12 (copy)"]
    encodings = [{"text": t, "ids": tok.encode(t, add_special_tokens=False).ids} for t in samples]
    dump_json(os.path.join(out, "train.json"), {
        "trainer": {"vocab_size": vocab, "min_frequency": min_frequency, "special_tokens": SPECIALS,
                    "pre_tokenizer": "metaspace" if metaspace else "bytelevel", "add_prefix_space": False,
                    "files": ["../corpus.txt"]},
        "vocab": sorted(model["vocab"].items(), key=lambda kv: kv[1]),
        "merges": model["merges"],
        "encodings": encodings,
    })


def randomize(model, torch):
    """Every parameter from one seeded generator in name order, at scales that keep activations near 1."""
    g = torch.Generator().manual_seed(INIT_SEED)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
            n = torch.randn(p.shape, generator=g, dtype=torch.float32)
            if name.endswith("norm.weight"):
                v = 1.0 + 0.1 * n
            elif "embed_tokens" in name:
                v = 0.5 * n
            else:
                v = n / np.sqrt(p.shape[-1])            # Linear [out, in]
            p.copy_(v)


def write_model(out, tokenizer_dir, ids):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(vocab_size=VOCAB, pad_token_id=ids["<pad>"], bos_token_id=ids["<s>"],
                         eos_token_id=[ids["<|im_end|>"], ids["</s>"]], **TEXT)
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).float().eval()
    randomize(model, torch)
    model.save_pretrained(out, safe_serialization=True)
    gen = os.path.join(out, "generation_config.json")
    if os.path.exists(gen):
        os.remove(gen)
    with open(os.path.join(out, "config.json"), encoding="utf-8") as f:
        dump_json(os.path.join(out, "config.json"), json.load(f))
    os.chmod(os.path.join(out, "model.safetensors"), 0o644)
    shutil.copy(os.path.join(tokenizer_dir, "tokenizer.json"), os.path.join(out, "tokenizer.json"))
    dump_json(os.path.join(out, "tokenizer_config.json"), {
        "add_bos_token": False, "bos_token": "<s>", "eos_token": "<|im_end|>", "pad_token": "<pad>",
        "chat_template": CHAT_TEMPLATE, "model_max_length": TEXT["max_position_embeddings"],
        "tokenizer_class": "PreTrainedTokenizerFast"})
    dump_json(os.path.join(out, "special_tokens_map.json"),
              {"bos_token": "<s>", "eos_token": "<|im_end|>", "pad_token": "<pad>"})
    dump_json(os.path.join(out, "generation_config.json"),
              {"bos_token_id": ids["<s>"], "eos_token_id": [ids["<|im_end|>"], ids["</s>"]], "pad_token_id": ids["<pad>"]})
    return model


class FoldedAdamW:
    """AdamW as Idrak computes it: decoupled decay, then p -= lr * sqrt(1 - b2^t) / (1 - b1^t) * m / (sqrt(v) + eps), the
    bias corrections folded into the step size (so epsilon is eps / sqrt(1 - b2^t) of torch's). Only the epsilon's place
    differs from torch.optim.AdamW; it shows where a second moment is near epsilon squared."""

    def __init__(self, params, lr, betas, eps, weight_decay):
        import torch
        self.params, self.lr, (self.b1, self.b2), self.eps, self.wd = list(params), lr, betas, eps, weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0
        self.state = {}

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None

    def step(self):
        import torch
        self.t += 1
        size = self.lr * np.sqrt(1 - self.b2 ** self.t) / (1 - self.b1 ** self.t)
        with torch.no_grad():
            for p, m, v in zip(self.params, self.m, self.v):
                if p.grad is None:
                    continue
                p.mul_(1 - self.lr * self.wd)
                m.mul_(self.b1).add_(p.grad, alpha=1 - self.b1)
                v.mul_(self.b2).addcmul_(p.grad, p.grad, value=1 - self.b2)
                p.addcdiv_(m, v.sqrt().add_(self.eps), value=-float(size))


def train_steps(model_dir, tokens, out, optimizer_kind):
    """STEPS steps on the first WINDOWS windows, every parameter trained; the loss before each update."""
    import torch
    from safetensors.torch import save_file
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).train()
    windows = np.stack([tokens[i * WINDOW: i * WINDOW + WINDOW + 1] for i in range(WINDOWS)]).astype(np.int64)
    x = torch.from_numpy(windows[:, :-1])
    y = torch.from_numpy(windows[:, 1:])
    params = [p for _, p in model.named_parameters()]
    if optimizer_kind == "sgd":
        opt = torch.optim.SGD(params, lr=SGD_RATE, momentum=0.0)
    elif optimizer_kind == "adamw":
        opt = torch.optim.AdamW(params, **ADAMW)
    else:
        opt = FoldedAdamW(params, **ADAMW)
    losses = []
    os.makedirs(out, exist_ok=True)
    for step in range(1, STEPS + 1):
        opt.zero_grad(set_to_none=True)
        logits = model(input_ids=x).logits
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss.backward()
        losses.append(float(loss))
        if step == 1 and optimizer_kind == "sgd":
            grads = {n: p.grad.detach().clone().contiguous() for n, p in model.named_parameters()}
            save_file(dict(sorted(grads.items())), os.path.join(out, "grads-step1.safetensors"), metadata={"format": "pt"})
        opt.step()
    weights = {n: p.detach().clone().contiguous() for n, p in model.named_parameters()}
    save_file(dict(sorted(weights.items())), os.path.join(out, f"weights-step{STEPS}.safetensors"), metadata={"format": "pt"})
    result = {"losses": losses}
    if optimizer_kind == "adamw":
        state = {}
        names = {id(p): n for n, p in model.named_parameters()}
        for p, s in opt.state.items():
            state[names[id(p)] + ".exp_avg"] = s["exp_avg"].contiguous()
            state[names[id(p)] + ".exp_avg_sq"] = s["exp_avg_sq"].contiguous()
        save_file(dict(sorted(state.items())), os.path.join(out, f"state-step{STEPS}.safetensors"), metadata={"format": "pt"})
        result["steps_in_state"] = STEPS
    return result, windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    args = parser.parse_args()
    import torch
    import transformers
    import tokenizers
    import safetensors
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    out = os.path.abspath(args.out)
    readme = os.path.join(out, "README.md")
    kept = open(readme, encoding="utf-8").read() if os.path.exists(readme) else None   # written by hand, kept
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    if kept is not None:
        with open(readme, "w", encoding="utf-8", newline="\n") as f:
            f.write(kept)
    docs = corpus()
    corpus_path = os.path.join(out, "corpus.txt")
    with open(corpus_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n---\n".join(docs) + "\n")

    tok, model_json = train_tokenizer(corpus_path, os.path.join(out, "tokenizer"), metaspace=False)
    write_tokenizer_reference(tok, model_json, os.path.join(out, "tokenizer"), docs, metaspace=False)
    mtok, mmodel = train_tokenizer(corpus_path, os.path.join(out, "metaspace"), metaspace=True)
    write_tokenizer_reference(mtok, mmodel, os.path.join(out, "metaspace"), docs, metaspace=True)
    # Merging until no pair is left (minimum frequency 1, a vocabulary larger than the corpus reaches): many ties.
    ftok, fmodel = train_tokenizer(corpus_path, os.path.join(out, "tokenizer-full"), metaspace=False, vocab=4096, min_frequency=1)
    write_tokenizer_reference(ftok, fmodel, os.path.join(out, "tokenizer-full"), docs, metaspace=False, vocab=4096, min_frequency=1)

    ids = {s: tok.token_to_id(s) for s in SPECIALS}
    tokens = []
    for doc in docs:
        tokens += tok.encode(doc, add_special_tokens=False).ids + [ids["</s>"]]
    tokens = np.asarray(tokens, dtype=np.uint16)
    tokens.tofile(os.path.join(out, "tokens.bin"))

    model_dir = os.path.join(out, "tiny-llama")
    write_model(model_dir, os.path.join(out, "tokenizer"), ids)
    sgd, windows = train_steps(model_dir, tokens, os.path.join(out, "step"), "sgd")
    adamw, _ = train_steps(model_dir, tokens, os.path.join(out, "step", "adamw"), "adamw")
    folded, _ = train_steps(model_dir, tokens, os.path.join(out, "step", "adamw-folded"), "folded")
    dump_json(os.path.join(out, "step", "step.json"), {
        "window": WINDOW, "windows": WINDOWS, "window_ids": windows.tolist(), "steps": STEPS,
        "loss": "mean cross entropy over every position of every window (targets: the next token)",
        "sgd": {"learning_rate": SGD_RATE, "momentum": 0.0, "weight_decay": 0.0, "max_gradient_norm": 0.0,
                "losses": sgd["losses"], "files": {"grads_step1": "grads-step1.safetensors",
                                                    "weights_after": f"weights-step{STEPS}.safetensors"}},
        "adamw": {**{k: (list(v) if isinstance(v, tuple) else v) for k, v in ADAMW.items()}, "max_gradient_norm": 0.0,
                  "losses": adamw["losses"], "files": {"weights_after": f"adamw/weights-step{STEPS}.safetensors",
                                                       "state_after": f"adamw/state-step{STEPS}.safetensors"}},
        "adamw_folded": {"what": "AdamW with the bias corrections folded into the step size, as Idrak computes it (epsilon's place only differs)",
                         "losses": folded["losses"], "files": {"weights_after": f"adamw-folded/weights-step{STEPS}.safetensors"}},
    })

    files = {}
    tensors = {}
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
        "special_tokens": ids, "tokens": int(tokens.size), "documents": len(docs),
        "summary": {"sgd_losses": sgd["losses"], "adamw_losses": adamw["losses"]},
        "tensors": tensors, "files": files,
    })
    print(f"wrote {out}: {len(docs)} documents, {tokens.size} tokens, SGD losses {sgd['losses']}, AdamW {adamw['losses']}")


if __name__ == "__main__":
    main()
