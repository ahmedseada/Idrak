"""Two tiny random T5 models and transformers' outputs for them (plan 26, P10: the T5 plug-in, Idrak.Text.Plugins.T5).

    python -m venv ref && ref/bin/pip install -r tools/t5/requirements.txt
    ref/bin/python -I tools/t5/make_tiny_t5.py                  # the tiny models and their references (no download)
    ref/bin/python -I tools/t5/make_tiny_t5.py --real           # the real models' references (downloads, the owner's)

Both tiny models use the committed Unigram tokenizer (tests/Idrak.Tests/data/tokenizers/unigram, 1,002 ids with its two
sentinel tokens): width 32, 4 heads of 6 (an inner width of 24, not the model's), a feed-forward of 48, 3 encoder and 2
decoder layers, 32 relative-position buckets with a max distance of 20 (so the log-spaced buckets are reached by short
texts), RMS norm epsilon 1e-6. Every parameter is drawn from a fixed torch generator (norm gains near 1, the embedding
0.2 N, matrices N / sqrt(fan in), the bias tables 0.5 N).

- tiny-t5 is T5 v1.0: ReLU feed-forward, the head tied to the shared embedding (and the decoder's output scaled by
  d_model^-0.5 before it); its checkpoint holds shared.weight only (no lm_head.weight), as transformers saves a tied T5.
- tiny-t5-gated is T5 v1.1 / mT5: gated GELU (gelu_new) feed-forward, its own lm_head.weight, config.json's
  tie_word_embeddings false (no scaling). Its head's row of the end token is pointed at the decoder's state after four
  greedy tokens of the first text, so some generations end before their limit.

The config.json files are written as the original checkpoints have them (transformers 5 rewrites tie_word_embeddings on
save); the references come from each model as transformers loads it back from its folder (eager attention, float32).

reference.safetensors (float32) holds, for each model (tiny-t5., tiny-t5-gated.) and each case (three single texts,
single0..2, and a padded batch of three, batch): the source ids and lengths, the encoder's last hidden states, and the
decoder's logits for a fixed target prefix (the decoder start token, then four tokens; the batch's rows share it).
reference.json holds the texts, ids, the targets, and the generations of each single text: greedy (16 new tokens at
most), and beam search (3 beams, 3 returned, 10 new tokens at most, length_penalty 1, early_stopping true and false) with
transformers' sequence scores, each as ids after the start token and as text (skip_special_tokens). It also holds
transformers' relative-position buckets (_relative_position_bucket) of relative positions -700..700 for the tiny models'
setting (32, 20) and the real models' (32, 128), both ways. Reruns write the same bytes on the same machine and library
versions; manifest.json records the versions and each file's SHA-256.

--real downloads google-t5/t5-small and google/mt5-small into <cache>/models/<org>--<name>/ (the cache is IDRAK_CACHE,
else ~/.cache/idrak; --cache overrides it) and writes only references: real/<name>.json (the files' sizes and SHA-256, the
versions, the texts, ids and greedy generations) and real/<name>.safetensors (the encoder's states and the decoder's
logits for the greedy target). The model files themselves are not committed.
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
TOKENIZER = os.path.join(ROOT, "tests", "Idrak.Tests", "data", "tokenizers", "unigram")
OUT = os.path.join(ROOT, "tests", "Idrak.Tests", "data", "plugins", "t5")
SEED = 26010
SIZES = dict(d_model=32, d_kv=6, d_ff=48, num_layers=3, num_decoder_layers=2, num_heads=4, relative_attention_num_buckets=32,
             relative_attention_max_distance=20, layer_norm_epsilon=1e-6, dropout_rate=0.1, initializer_factor=1.0)
MODELS = {"tiny-t5": dict(feed_forward_proj="relu", tie_word_embeddings=True),
          "tiny-t5-gated": dict(feed_forward_proj="gated-gelu", tie_word_embeddings=False)}
SINGLES = ["العلم نور والجهل ظلام.",
           "translate English to Arabic: The book is on the table.",
           "مرحبا بكم في المكتبة العامة"]
BATCH = ["سعر الكتاب خمسون ريالا", "The quick brown fox jumps over the lazy dog.", "نور"]
PREFIX = 4                     # target tokens after the start token for the logits
GREEDY_TOKENS = 16
BEAMS = 3
BEAM_TOKENS = 10
BUCKET_RANGE = 700
END_STEP = 4                   # the greedy step of single0 the gated model's end token is pointed at
REAL_MODELS = {"google-t5/t5-small": ["translate English to German: The house is wonderful.",
                                      "summarize: The library opens at nine and closes at five every day except Friday."],
               "google/mt5-small": ["العلم نور والجهل ظلام.", "The quick brown fox jumps over the lazy dog."]}
REAL_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "spiece.model", "generation_config.json"]
REAL_TOKENS = 20


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
    """Every parameter from the generator, in name order: norm gains 1 + 0.1 N, the shared embedding 0.2 N (so the blocks, not
    the input token, steer the next token), the bias tables 0.5 N, matrices N / sqrt(fan in)."""
    with torch.no_grad():
        for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
            noise = torch.randn(p.shape, generator=generator, dtype=torch.float32)
            if p.dim() == 1:
                p.copy_(1 + 0.1 * noise)
            elif "relative_attention_bias" in name:
                p.copy_(0.5 * noise)
            elif name in ("shared.weight", "encoder.embed_tokens.weight", "decoder.embed_tokens.weight"):
                p.copy_(0.2 * noise)
            else:
                p.copy_(noise / (p.shape[1] ** 0.5))


def force_end(model, ids, eos):
    """Points the untied head's row of the end token at the decoder's state after END_STEP greedy tokens of `ids`, with a
    logit 0.5 above that step's best, so generations end (greedy and some beams) and both stopping rules are exercised."""
    source = torch.tensor([ids], dtype=torch.long)
    captured = {}
    with torch.no_grad():
        tokens = model.generate(source, do_sample=False, num_beams=1, max_new_tokens=END_STEP + 1)[0]
        handle = model.lm_head.register_forward_hook(lambda module, inputs, output: captured.__setitem__("h", inputs[0][0, -1].clone()))
        logits = model(input_ids=source, decoder_input_ids=tokens[None, :END_STEP + 1]).logits[0, -1]
        handle.remove()
        h = captured["h"]
        model.lm_head.weight[eos] = h / h.norm() * ((logits.max() + 0.5) / h.norm())


def padded(rows, pad):
    width = max(len(ids) for ids in rows)
    ids = torch.tensor([r + [pad] * (width - len(r)) for r in rows], dtype=torch.long)
    lengths = torch.tensor([len(r) for r in rows], dtype=torch.long)
    mask = (torch.arange(width)[None, :] < lengths[:, None]).long()
    return ids, lengths, mask


def buckets():
    from transformers.models.t5.modeling_t5 import T5Attention
    relative = torch.arange(-BUCKET_RANGE, BUCKET_RANGE + 1, dtype=torch.long)
    table = {}
    for count, distance in ((SIZES["relative_attention_num_buckets"], SIZES["relative_attention_max_distance"]), (32, 128)):
        for bidirectional in (True, False):
            key = f"{count}/{distance}/{'bidirectional' if bidirectional else 'one-way'}"
            table[key] = T5Attention._relative_position_bucket(relative[None, :], bidirectional=bidirectional, num_buckets=count,
                                                               max_distance=distance)[0].tolist()
    return {"from": -BUCKET_RANGE, "to": BUCKET_RANGE, "tables": table}


def until_end(tokens, end):
    """The tokens before the first end token (a finished beam is padded after it)."""
    return tokens[:tokens.index(end)] if end in tokens else tokens


def generations(model, tokenizer, ids):
    """Greedy and beam search of one source (ids [1, t]) as transformers' generate gives them."""
    source = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        produced = model.generate(source, do_sample=False, num_beams=1, max_new_tokens=GREEDY_TOKENS)[0].tolist()[1:]
    greedy = until_end(produced, model.config.eos_token_id)
    record = {"greedy": greedy, "greedy_ended": model.config.eos_token_id in produced, "greedy_text": tokenizer.decode(greedy, skip_special_tokens=True)}
    for early in (True, False):
        with torch.no_grad():
            out = model.generate(source, do_sample=False, num_beams=BEAMS, num_return_sequences=BEAMS, max_new_tokens=BEAM_TOKENS,
                                 length_penalty=1.0, early_stopping=early, output_scores=True, return_dict_in_generate=True)
        rows = [{"ids": until_end(sequence[1:], model.config.eos_token_id), "ended": model.config.eos_token_id in sequence[1:], "score": score}
                for sequence, score in zip(out.sequences.tolist(), out.sequences_scores.tolist())]
        record[f"beam_{'early' if early else 'late'}"] = rows
    return record


def make_tiny(out):
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from transformers import T5Config, T5ForConditionalGeneration

    torch.manual_seed(SEED)
    tokenizer = Tokenizer.from_file(os.path.join(TOKENIZER, "tokenizer.json"))
    vocab = tokenizer.get_vocab_size()
    pad_id, eos_id = tokenizer.token_to_id("<pad>"), tokenizer.token_to_id("</s>")
    refs, record = {}, {"seed": SEED, "sizes": dict(SIZES, vocab_size=vocab), "tokenizer": "tests/Idrak.Tests/data/tokenizers/unigram",
                        "attention": "eager", "models": {}, "buckets": buckets()}
    generator = torch.Generator().manual_seed(SEED)
    for name, kind in MODELS.items():
        config = T5Config(vocab_size=vocab, pad_token_id=pad_id, eos_token_id=eos_id, decoder_start_token_id=pad_id,
                          feed_forward_proj=kind["feed_forward_proj"], tie_word_embeddings=kind["tie_word_embeddings"], **SIZES)
        config._attn_implementation = "eager"
        model = T5ForConditionalGeneration(config).eval()
        if not kind["tie_word_embeddings"]:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(vocab, SIZES["d_model"]))     # its own head
        randomize(model, generator)
        if not kind["tie_word_embeddings"]:
            force_end(model, tokenizer.encode(SINGLES[0]).ids, eos_id)

        # The checkpoint as transformers saves one: the tied tensors once (shared.weight), lm_head.weight when untied.
        model_dir = os.path.join(out, name)
        os.makedirs(model_dir)
        state = model.state_dict()
        tensors = {k: v for k, v in state.items() if k not in ("encoder.embed_tokens.weight", "decoder.embed_tokens.weight")}
        if kind["tie_word_embeddings"]:
            tensors.pop("lm_head.weight", None)
        save_file({k: v.detach().contiguous().to(torch.float32) for k, v in sorted(tensors.items())}, os.path.join(model_dir, "model.safetensors"),
                  metadata={"format": "pt"})
        write_json(os.path.join(model_dir, "config.json"), dict(
            SIZES, architectures=["T5ForConditionalGeneration"], model_type="t5", vocab_size=vocab, pad_token_id=pad_id, eos_token_id=eos_id,
            decoder_start_token_id=pad_id, feed_forward_proj=kind["feed_forward_proj"], tie_word_embeddings=kind["tie_word_embeddings"],
            is_encoder_decoder=True, n_positions=512))

        loaded = T5ForConditionalGeneration.from_pretrained(model_dir, attn_implementation="eager").eval()
        assert loaded.config.scale_decoder_outputs == kind["tie_word_embeddings"], "the decoder's output scaling follows tie_word_embeddings"
        assert torch.equal(loaded.lm_head.weight, model.lm_head.weight), "the head reads back as saved"
        generator_target = torch.Generator().manual_seed(SEED + len(name))
        cases = {}

        def run(case, rows):
            ids, lengths, mask = padded(rows, pad_id)
            target = torch.cat([torch.full((1, 1), pad_id, dtype=torch.long),
                                torch.randint(3, vocab - 2, (1, PREFIX), generator=generator_target)], dim=1).repeat(ids.shape[0], 1)
            with torch.no_grad():
                encoded = loaded.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
                logits = loaded(input_ids=ids, attention_mask=mask, decoder_input_ids=target).logits
            refs.update({f"{name}.{case}.ids": ids.float(), f"{name}.{case}.lengths": lengths.float(), f"{name}.{case}.encoder": encoded,
                         f"{name}.{case}.target": target.float(), f"{name}.{case}.logits": logits})
            cases[case] = {"ids": ids.tolist(), "lengths": lengths.tolist(), "target": target.tolist()}

        for i, text in enumerate(SINGLES):
            ids = tokenizer.encode(text).ids
            run(f"single{i}", [ids])
            cases[f"single{i}"].update(text=text, **generations(loaded, tokenizer, ids))
        run("batch", [tokenizer.encode(text).ids for text in BATCH])
        cases["batch"]["texts"] = BATCH
        record["models"][name] = {"feed_forward_proj": kind["feed_forward_proj"], "tie_word_embeddings": kind["tie_word_embeddings"], "cases": cases}

    save_file({k: v.detach().contiguous().to(torch.float32) for k, v in sorted(refs.items())}, os.path.join(out, "reference.safetensors"),
              metadata={"format": "pt"})
    record["tensors"] = {k: list(v.shape) for k, v in sorted(refs.items())}
    write_json(os.path.join(out, "reference.json"), record)


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
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    real = os.path.join(out, "real")
    os.makedirs(real, exist_ok=True)
    for repo, texts in REAL_MODELS.items():
        local = os.path.join(cache, "models", repo.replace("/", "--"))
        listed = set(list_repo_files(repo))
        weights = ["model.safetensors"] if "model.safetensors" in listed else ["pytorch_model.bin"]
        wanted = [n for n in REAL_FILES if n in listed] + weights
        print(f"{repo}: downloading {', '.join(wanted)} into {local}", file=sys.stderr)
        snapshot_download(repo_id=repo, local_dir=local, allow_patterns=wanted)
        present = sorted(n for n in wanted if os.path.isfile(os.path.join(local, n)))
        tokenizer = AutoTokenizer.from_pretrained(local)
        model = T5ForConditionalGeneration.from_pretrained(local, attn_implementation="eager").eval()
        refs, cases = {}, []
        for i, text in enumerate(texts):
            ids = tokenizer(text, return_tensors="pt")["input_ids"]
            with torch.no_grad():
                greedy = model.generate(ids, do_sample=False, num_beams=1, max_new_tokens=REAL_TOKENS)[0]
                encoded = model.encoder(input_ids=ids).last_hidden_state[0]
                logits = model(input_ids=ids, decoder_input_ids=greedy[None, :]).logits[0]
            refs[f"text{i}.encoder"] = encoded
            refs[f"text{i}.logits"] = logits
            tokens = greedy.tolist()
            cases.append({"text": text, "ids": ids[0].tolist(), "target": tokens, "greedy": tokens[1:],
                          "greedy_text": tokenizer.decode(tokens[1:], skip_special_tokens=True)})
        name = repo.split("/")[-1]
        save_file({k: v.detach().contiguous().to(torch.float32) for k, v in sorted(refs.items())}, os.path.join(real, f"{name}.safetensors"),
                  metadata={"format": "pt"})
        write_json(os.path.join(real, f"{name}.json"), {
            "repo": repo, "folder": f"models/{repo.replace('/', '--')}", "architecture": model.config.architectures[0] if model.config.architectures else "T5ForConditionalGeneration",
            "max_new_tokens": REAL_TOKENS, "attention": "eager", "cases": cases, "versions": versions(),
            "files": {n: {"bytes": os.path.getsize(os.path.join(local, n)), "sha256": sha256(os.path.join(local, n))} for n in present}})
        print(f"{repo}: references written to {real}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=OUT, help="the fixture folder (default: tests/Idrak.Tests/data/plugins/t5)")
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
