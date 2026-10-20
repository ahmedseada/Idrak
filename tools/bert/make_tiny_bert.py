"""A tiny random BERT and transformers' outputs for it (plan 26, P9: the BERT plug-in, Idrak.Text.Plugins.Bert).

    python -m venv ref && ref/bin/pip install -r tools/bert/requirements.txt
    ref/bin/python -I tools/bert/make_tiny_bert.py                  # the tiny model and its references (no download)
    ref/bin/python -I tools/bert/make_tiny_bert.py --real           # the real models' references (downloads, the owner's)

The tiny model uses the committed WordPiece tokenizer (tests/Idrak.Tests/data/tokenizers/wordpiece-uncased): its
vocabulary, 2 post-norm layers of width 32 with 4 heads, a feed-forward of 64 with exact GELU, 64 learned positions,
2 token types, LayerNorm epsilon 1e-12. Every parameter is drawn from a fixed torch generator (LayerNorm gains near 1,
biases and the classifier non-zero, so every path counts). One checkpoint holds all three architectures' tensors under
transformers' names: "bert." (embeddings, encoder, pooler), "cls.predictions." (the masked-LM transform and its bias;
the decoder is tied to the word embeddings, as transformers ties it) and "classifier." (3 labels); config.json names
BertForMaskedLM, and the tests read it as BertModel and BertForSequenceClassification too.

reference.safetensors (float32) holds, for three single texts (single0..2) and a padded batch of three text pairs with
token types (batch): the ids, type ids and lengths the tokenizer gives, transformers' last hidden states
(BertModel), pooled output (the pooler: dense + tanh of the first token), masked-LM logits at two positions per
sequence (the [MASK] or the second position, and the last real one) from BertForMaskedLM, classification logits from
BertForSequenceClassification, and sentence vectors: the mean of the real tokens' states and the first token's state,
each scaled to length 1. reference.json has the texts, positions and sizes. Reruns write the same bytes on the same
machine and library versions; manifest.json records the versions and each file's SHA-256.

--real downloads google-bert/bert-base-uncased and aubmindlab/bert-base-arabertv02 into <cache>/models/<org>--<name>/
(the cache is IDRAK_CACHE, else ~/.cache/idrak; --cache overrides it) and writes only references: real/<name>.json
(the files' sizes and SHA-256, the versions, the texts and ids) and real/<name>.safetensors (hidden states, the pooled
output when the checkpoint has a pooler, the 32 largest masked-LM logits at each [MASK] with their ids). The model
files themselves are not committed.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOKENIZER = os.path.join(ROOT, "tests", "Idrak.Tests", "data", "tokenizers", "wordpiece-uncased")
OUT = os.path.join(ROOT, "tests", "Idrak.Tests", "data", "plugins", "bert")
SEED = 26009
SIZES = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64, max_position_embeddings=64,
             type_vocab_size=2, layer_norm_eps=1e-12, hidden_act="gelu", num_labels=3)
SINGLES = ["العلم نور والجهل ظلام.",
           "كتب الطالب [MASK] في الدفتر.",
           "سعر الكتاب 25.50 ريال، ورقم الطلب ORD-2024."]
PAIRS = [("العلم نور", "والجهل ظلام."),
         ("ذهب الولد إلى [MASK] صباحا", "وعاد مساء."),
         ("مرحبا", "بكم في المكتبة العامة")]
REAL_MODELS = {"google-bert/bert-base-uncased": ["the capital of france is [MASK].", "a man went to the [MASK] to buy milk ."],
               "aubmindlab/bert-base-arabertv02": ["عاصمة فرنسا هي [MASK] .", "ذهب الرجل إلى [MASK] لشراء الحليب ."]}
REAL_FILES = ["config.json", "vocab.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
TOP = 32


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def versions():
    import safetensors
    import tokenizers
    import transformers
    return {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__, "safetensors": safetensors.__version__}


def write_json(path, value):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write("\n")


def randomize(model, generator):
    """Every parameter from the generator, in name order: LayerNorm gains 1 + 0.1 N, biases 0.1 N, embeddings 0.5 N, and
    matrices N / sqrt(fan in), so activations stay of order one and every term shows in the outputs."""
    with torch.no_grad():
        for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
            noise = torch.randn(p.shape, generator=generator, dtype=torch.float32)
            if name.endswith("LayerNorm.weight"):
                p.copy_(1 + 0.1 * noise)
            elif name.endswith("bias") or p.dim() == 1:
                p.copy_(0.1 * noise)
            elif "embeddings" in name:
                p.copy_(0.5 * noise)
            else:
                p.copy_(noise / (p.shape[1] ** 0.5))


def encode(tokenizer, text, pair=None):
    e = tokenizer.encode(text, pair) if pair is not None else tokenizer.encode(text)
    return e.ids, e.type_ids


def padded(rows, pad):
    width = max(len(ids) for ids, _ in rows)
    ids = torch.tensor([r[0] + [pad] * (width - len(r[0])) for r in rows], dtype=torch.long)
    types = torch.tensor([r[1] + [0] * (width - len(r[1])) for r in rows], dtype=torch.long)
    lengths = torch.tensor([len(r[0]) for r in rows], dtype=torch.long)
    mask = (torch.arange(width)[None, :] < lengths[:, None]).long()
    return ids, types, lengths, mask


def positions(ids, lengths, mask_id):
    """Two positions per sequence: its [MASK] (else position 1), and its last real token."""
    rows = []
    for row, length in zip(ids.tolist(), lengths.tolist()):
        first = row.index(mask_id) if mask_id in row[:length] else 1
        rows.append([first, length - 1])
    return torch.tensor(rows, dtype=torch.long)


def unit(x):
    return x / x.norm(dim=-1, keepdim=True)


def make_tiny(out):
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from transformers import BertConfig, BertForMaskedLM, BertForSequenceClassification, BertModel

    torch.manual_seed(SEED)
    tokenizer = Tokenizer.from_file(os.path.join(TOKENIZER, "tokenizer.json"))
    vocab = tokenizer.get_vocab_size()
    pad_id, mask_id = tokenizer.token_to_id("[PAD]"), tokenizer.token_to_id("[MASK]")
    config = BertConfig(vocab_size=vocab, pad_token_id=pad_id, architectures=["BertForMaskedLM"], **SIZES)
    config._attn_implementation = "eager"
    generator = torch.Generator().manual_seed(SEED)

    base = BertModel(config, add_pooling_layer=True).eval()
    randomize(base, generator)
    mlm = BertForMaskedLM(config).eval()
    classifier = BertForSequenceClassification(config).eval()
    mlm.bert.load_state_dict({k: v for k, v in base.state_dict().items() if not k.startswith("pooler.")}, strict=True)
    classifier.bert.load_state_dict(base.state_dict(), strict=True)
    with torch.no_grad():
        for name, p in sorted(mlm.cls.named_parameters(), key=lambda kv: kv[0]):
            if name == "predictions.decoder.weight":
                continue                                                       # tied to the word embeddings
            noise = torch.randn(p.shape, generator=generator, dtype=torch.float32)
            p.copy_(1 + 0.1 * noise if name.endswith("LayerNorm.weight") else 0.1 * noise if p.dim() == 1 else noise / (p.shape[1] ** 0.5))
        for name, p in sorted(classifier.classifier.named_parameters(), key=lambda kv: kv[0]):
            noise = torch.randn(p.shape, generator=generator, dtype=torch.float32)
            p.copy_(0.1 * noise if p.dim() == 1 else noise / (p.shape[1] ** 0.5))
    decoder = mlm.cls.predictions.decoder
    assert torch.equal(decoder.weight, mlm.bert.embeddings.word_embeddings.weight), "the decoder is tied to the word embeddings"
    assert torch.equal(decoder.bias, mlm.cls.predictions.bias), "the decoder's bias is the predictions' bias"

    model_dir = os.path.join(out, "tiny-bert")
    os.makedirs(model_dir)
    tensors = {f"bert.{k}": v for k, v in base.state_dict().items()}
    tensors.update({f"cls.{k}": v for k, v in mlm.cls.state_dict().items() if k not in ("predictions.decoder.weight", "predictions.decoder.bias")})
    tensors.update({f"classifier.{k}": v for k, v in classifier.classifier.state_dict().items()})
    save_file({k: v.detach().contiguous().to(torch.float32) for k, v in sorted(tensors.items())}, os.path.join(model_dir, "model.safetensors"),
              metadata={"format": "pt"})
    config.save_pretrained(model_dir)

    refs, cases = {}, {}

    def record(name, rows):
        ids, types, lengths, mask = padded(rows, pad_id)
        with torch.no_grad():
            out_base = base(input_ids=ids, attention_mask=mask, token_type_ids=types)
            logits = mlm(input_ids=ids, attention_mask=mask, token_type_ids=types).logits
            classes = classifier(input_ids=ids, attention_mask=mask, token_type_ids=types).logits
        hidden = out_base.last_hidden_state
        where = positions(ids, lengths, mask_id)
        rows_mlm = torch.stack([logits[b, where[b]] for b in range(ids.shape[0])])     # [n, 2, vocab]
        weights = mask.unsqueeze(-1).float()
        mean = (hidden * weights).sum(1) / weights.sum(1)
        refs.update({f"{name}.ids": ids.float(), f"{name}.types": types.float(), f"{name}.lengths": lengths.float(),
                     f"{name}.hidden": hidden, f"{name}.pooled": out_base.pooler_output, f"{name}.mlm_positions": where.float(),
                     f"{name}.mlm": rows_mlm, f"{name}.classes": classes, f"{name}.mean": unit(mean), f"{name}.first": unit(hidden[:, 0])})
        cases[name] = {"ids": ids.tolist(), "types": types.tolist(), "lengths": lengths.tolist(), "mlm_positions": where.tolist(),
                       "argmax_at_positions": rows_mlm.argmax(-1).tolist(), "classes_argmax": classes.argmax(-1).tolist()}

    for i, text in enumerate(SINGLES):
        record(f"single{i}", [encode(tokenizer, text)])
        cases[f"single{i}"]["text"] = text
    record("batch", [encode(tokenizer, a, b) for a, b in PAIRS])
    cases["batch"]["pairs"] = [list(p) for p in PAIRS]
    save_file({k: v.detach().contiguous().to(torch.float32) for k, v in sorted(refs.items())}, os.path.join(out, "reference.safetensors"),
              metadata={"format": "pt"})
    write_json(os.path.join(out, "reference.json"), {
        "seed": SEED, "sizes": dict(SIZES, vocab_size=vocab), "pad_id": pad_id, "mask_id": mask_id,
        "tokenizer": "tests/Idrak.Tests/data/tokenizers/wordpiece-uncased", "attention": "eager", "cases": cases,
        "tensors": {k: list(v.shape) for k, v in sorted(refs.items())}})


def manifest(out, extra=None):
    files = {}
    for folder, _, names in os.walk(out):
        for name in sorted(names):
            p = os.path.join(folder, name)
            rel = os.path.relpath(p, out).replace(os.sep, "/")
            if rel in ("manifest.json", "README.md") or rel.startswith("real/"):
                continue
            files[rel] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
    write_json(os.path.join(out, "manifest.json"), {"files": dict(sorted(files.items())), "versions": versions(), **(extra or {})})


def clean(out):
    """Removes everything in the fixture folder except the hand-written README.md and --real's real/."""
    os.makedirs(out, exist_ok=True)
    for entry in os.listdir(out):
        if entry in ("README.md", "real"):
            continue
        path = os.path.join(out, entry)
        shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)


def make_real(out, cache):
    from huggingface_hub import list_repo_files, snapshot_download
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, BertForMaskedLM, BertModel

    real = os.path.join(out, "real")
    os.makedirs(real, exist_ok=True)
    for repo, texts in REAL_MODELS.items():
        local = os.path.join(cache, "models", repo.replace("/", "--"))
        listed = set(list_repo_files(repo))
        weights = ["model.safetensors"] if "model.safetensors" in listed else ["pytorch_model.bin"]
        print(f"{repo}: downloading {', '.join(REAL_FILES + weights)} into {local}", file=sys.stderr)
        snapshot_download(repo_id=repo, local_dir=local, allow_patterns=REAL_FILES + weights)
        present = sorted(n for n in REAL_FILES + weights if os.path.isfile(os.path.join(local, n)))
        tokenizer = AutoTokenizer.from_pretrained(local)
        base, info = BertModel.from_pretrained(local, attn_implementation="eager", output_loading_info=True)
        mlm = BertForMaskedLM.from_pretrained(local, attn_implementation="eager").eval()
        base.eval()
        has_pooler = not any(k.startswith("pooler.") for k in info["missing_keys"])
        refs, cases = {}, []
        for i, text in enumerate(texts):
            encoded = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                out_base = base(**encoded)
                logits = mlm(**encoded).logits[0]
            ids = encoded["input_ids"][0]
            masks = (ids == tokenizer.mask_token_id).nonzero().flatten()
            refs[f"text{i}.hidden"] = out_base.last_hidden_state[0]
            if has_pooler:
                refs[f"text{i}.pooled"] = out_base.pooler_output[0]
            top = logits[masks].topk(TOP, dim=-1)
            refs[f"text{i}.mlm_top_values"] = top.values
            refs[f"text{i}.mlm_top_ids"] = top.indices.float()
            cases.append({"text": text, "ids": ids.tolist(), "mask_positions": masks.tolist(),
                          "top_tokens": [tokenizer.convert_ids_to_tokens(row) for row in top.indices.tolist()]})
        name = repo.split("/")[-1]
        save_file({k: v.detach().contiguous().to(torch.float32) for k, v in sorted(refs.items())}, os.path.join(real, f"{name}.safetensors"),
                  metadata={"format": "pt"})
        write_json(os.path.join(real, f"{name}.json"), {
            "repo": repo, "folder": f"models/{repo.replace('/', '--')}", "architecture": "BertForMaskedLM", "has_pooler": has_pooler,
            "top": TOP, "attention": "eager", "cases": cases, "versions": versions(),
            "files": {n: {"bytes": os.path.getsize(os.path.join(local, n)), "sha256": sha256(os.path.join(local, n))} for n in present}})
        print(f"{repo}: references written to {real}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=OUT, help="the fixture folder (default: tests/Idrak.Tests/data/plugins/bert)")
    parser.add_argument("--real", action="store_true", help="download the real models into the cache, write real/<name>.json and .safetensors")
    parser.add_argument("--cache", default=None, help="the cache root for --real (default: IDRAK_CACHE, else ~/.cache/idrak)")
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)                                                   # one summation order, run to run
    if args.real:
        cache = args.cache or os.environ.get("IDRAK_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "idrak")
        make_real(args.out, cache)
        return
    clean(args.out)
    make_tiny(args.out)
    manifest(args.out, {"seed": SEED})


if __name__ == "__main__":
    main()
