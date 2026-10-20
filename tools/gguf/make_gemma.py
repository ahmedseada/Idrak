"""Tiny Gemma, Gemma 2 and Gemma 3 text models as GGUF files written by llama.cpp's own converter, beside the Hugging
Face folders they were converted from (plan 26, leftover round L1: GGUF loading for the Gemma plug-in).

    python -m venv ref && ref/bin/pip install -r tools/gguf/requirements.txt
    git clone --depth 1 --branch b11756 --filter=blob:none --sparse https://github.com/ggml-org/llama.cpp llama.cpp
    git -C llama.cpp sparse-checkout set conversion gguf-py
    ref/bin/python tools/gguf/make_gemma.py --llama-cpp llama.cpp        # writes tests/Idrak.Tests/data/gguf/tiny-gemma*

Nothing is downloaded but llama.cpp's converter (the tag above, LLAMA_CPP_TAG; convert_hf_to_gguf.py with its own
gguf-py, so no gguf package from PyPI is used). The three models share the tiny Gemma 3 of the vision-language tests
(tests/Idrak.Tests/data/vlm/tiny-gemma3): its tokenizer (tokenizer.json, tokenizer_config.json with its chat template,
special_tokens_map.json) and its text sizes (width 24, 4 heads of 8 with 2 key/value heads, a feed-forward of 48, 366
tokens). Gemma's converter reads a SentencePiece tokenizer.model, which the tiny Gemma 3 does not have: one is written
here from its tokenizer.json (the same pieces in id order; a merged piece scores minus its merge's rank, so
SentencePiece's highest-score-first merging takes tokenizer.json's merges in their order; checked below to give
tokenizer.json's ids on every test text).

- tiny-gemma: GemmaForCausalLM, 2 layers.
- tiny-gemma2: Gemma2ForCausalLM, 2 layers (a window of 8 in the first, the second global), soft-capping 2 (attention)
  and 3 (logits) so it acts on these small scores, query_pre_attn_scalar 8.
- tiny-gemma3: Gemma3ForCausalLM, 6 layers (llama.cpp's files do not store the window pattern and llama.cpp assumes
  five windowed layers in six: the sixth layer is the global one), a window of 8, RoPE base 1,000,000 with a linear
  scaling of 8 on the global layer and 10,000 unscaled on the windowed ones, query and key norms,
  query_pre_attn_scalar 8.

llama.cpp's files store no query_pre_attn_scalar either: llama.cpp takes the head width (8 here), so the models are made
with that. Every parameter is drawn from a fixed torch generator in name order (norm gains 0.1 N, as Gemma stores them:
a difference from 1; the embedding 0.5 N; matrices N / sqrt(fan in)), the matrices rounded to float16, since the GGUF
files are written as llama.cpp's f16 (matrices float16, norms float32): the Hugging Face folders (float32 safetensors)
then hold exactly the GGUF's values, and the norms are the only tensors the converter changes (it adds Gemma's 1:
checked here on the written files). Reruns write the same bytes on the same machine and library versions;
gemma-manifest.json records the versions, the llama.cpp tag and each file's SHA-256.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

LLAMA_CPP_TAG = "b11756"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TINY = os.path.join(ROOT, "tests", "Idrak.Tests", "data", "vlm", "tiny-gemma3")
OUT = os.path.join(ROOT, "tests", "Idrak.Tests", "data", "gguf")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
SEED = 26018
SIZES = dict(hidden_size=24, intermediate_size=48, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
             max_position_embeddings=256, rms_norm_eps=1e-6, pad_token_id=0, eos_token_id=1, bos_token_id=2,
             tie_word_embeddings=True, attention_bias=False, attention_dropout=0.0)
MODELS = {
    "tiny-gemma": ("GemmaConfig", "GemmaForCausalLM",
                   dict(num_hidden_layers=2, hidden_act="gelu_pytorch_tanh", hidden_activation="gelu_pytorch_tanh", rope_theta=10000.0)),
    "tiny-gemma2": ("Gemma2Config", "Gemma2ForCausalLM",
                    dict(num_hidden_layers=2, hidden_activation="gelu_pytorch_tanh", rope_theta=10000.0, query_pre_attn_scalar=8,
                         sliding_window=8, attn_logit_softcapping=2.0, final_logit_softcapping=3.0)),
    "tiny-gemma3": ("Gemma3TextConfig", "Gemma3ForCausalLM",
                    dict(num_hidden_layers=6, hidden_activation="gelu_pytorch_tanh", rope_theta=1_000_000.0,
                         rope_local_base_freq=10_000.0, rope_scaling={"rope_type": "linear", "factor": 8.0}, query_pre_attn_scalar=8,
                         sliding_window=8, sliding_window_pattern=6, final_logit_softcapping=None, attn_logit_softcapping=None)),
}
# Texts whose SentencePiece ids must equal tokenizer.json's (the tests encode these and chat prompts with both files).
TEXTS = ["the image is a scan of a page with lines of text", "Write one line about the moon and the sun in the night.",
         "what is in this image ?\n\nread the scan", "A red mark and a blue mark; 42 strokes, dark on light.",
         "  two  spaces and\ttabs ", "العلم نور"]


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_json(path, value):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.write("\n")


def sentencepiece_model(tokenizer_json):
    """tokenizer.json's BPE (no pre-tokenizer, ' ' replaced by '▁', byte fallback) as a SentencePiece BPE ModelProto."""
    from sentencepiece import sentencepiece_model_pb2 as pb
    with open(tokenizer_json, encoding="utf-8") as f:
        tok = json.load(f)
    vocab = tok["model"]["vocab"]
    merges = [tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m) for m in tok["model"]["merges"]]
    rank = {}
    for i, (a, b) in enumerate(merges):
        rank.setdefault(a + b, i)
    special = {t["content"]: t["id"] for t in tok["added_tokens"]}
    pieces = sorted(vocab.items(), key=lambda kv: kv[1])
    assert [i for _, i in pieces] == list(range(len(pieces))), "ids are not contiguous"
    model = pb.ModelProto()
    T = pb.ModelProto.SentencePiece
    for piece, i in pieces:
        p = model.pieces.add()
        p.piece = piece
        if piece == "<unk>":
            p.type, p.score = T.UNKNOWN, 0.0
        elif piece in ("<pad>", "<eos>", "<bos>"):
            p.type, p.score = T.CONTROL, 0.0
        elif piece in special:
            p.type, p.score = T.USER_DEFINED, 0.0
        elif len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
            p.type, p.score = T.BYTE, 0.0
        else:
            p.type, p.score = T.NORMAL, -float(rank[piece]) if piece in rank else 0.0
    trainer = model.trainer_spec
    trainer.model_type = pb.TrainerSpec.BPE
    trainer.vocab_size = len(pieces)
    trainer.byte_fallback = True
    trainer.unk_id, trainer.bos_id, trainer.eos_id, trainer.pad_id = vocab["<unk>"], vocab["<bos>"], vocab["<eos>"], vocab["<pad>"]
    trainer.unk_piece, trainer.bos_piece, trainer.eos_piece, trainer.pad_piece = "<unk>", "<bos>", "<eos>", "<pad>"
    normalizer = model.normalizer_spec
    normalizer.name = "identity"
    normalizer.add_dummy_prefix = False
    normalizer.remove_extra_whitespaces = False
    normalizer.escape_whitespaces = True
    return model.SerializeToString(), len(pieces)


def check_tokenizer(folder):
    """The written tokenizer.model encodes every test text as tokenizer.json does."""
    import sentencepiece
    from tokenizers import Tokenizer
    sp = sentencepiece.SentencePieceProcessor(model_file=os.path.join(folder, "tokenizer.model"))
    hf = Tokenizer.from_file(os.path.join(folder, "tokenizer.json"))
    for text in TEXTS:
        a, b = sp.encode(text), hf.encode(text, add_special_tokens=False).ids
        assert a == b, f"tokenizer.model and tokenizer.json disagree on {text!r}: {a} vs {b}"


def randomize(model, generator):
    with torch.no_grad():
        for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
            noise = torch.randn(p.shape, generator=generator, dtype=torch.float32)
            if p.dim() == 1:
                p.copy_(0.1 * noise)                                   # Gemma's gains: a difference from 1
            elif name.endswith("embed_tokens.weight"):
                p.copy_((0.5 * noise).half().float())
            else:
                p.copy_((noise / (p.shape[1] ** 0.5)).half().float())


def make(name, llama_cpp, vocab_size, spm):
    import transformers
    config_class, model_class, extra = MODELS[name]
    config = getattr(transformers, config_class)(vocab_size=vocab_size, **SIZES, **extra)
    torch.manual_seed(SEED)
    model = getattr(transformers, model_class)(config).eval()
    randomize(model, torch.Generator().manual_seed(SEED + len(name)))
    folder = os.path.join(OUT, f"{name}-hf")
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)
    model.save_pretrained(folder, safe_serialization=True)
    os.remove(os.path.join(folder, "generation_config.json")) if os.path.exists(os.path.join(folder, "generation_config.json")) else None
    for file in TOKENIZER_FILES:
        shutil.copyfile(os.path.join(TINY, file), os.path.join(folder, file))
    with open(os.path.join(folder, "tokenizer.model"), "wb") as f:
        f.write(spm)
    check_tokenizer(folder)

    gguf = os.path.join(OUT, f"{name}.gguf")
    subprocess.run([sys.executable, os.path.join(llama_cpp, "convert_hf_to_gguf.py"), folder, "--outfile", gguf, "--outtype", "f16"],
                   check=True, cwd=llama_cpp, stdout=subprocess.DEVNULL)
    check_gguf(llama_cpp, gguf, model)
    return folder, gguf


def check_gguf(llama_cpp, path, model):
    """The converter's tensors: every norm gain is the checkpoint's plus 1 (float32), every matrix the checkpoint's."""
    sys.path.insert(0, os.path.join(llama_cpp, "gguf-py"))
    import gguf
    reader = gguf.GGUFReader(path)
    state = {k: v.float().numpy() for k, v in model.state_dict().items()}
    names = {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight"}
    parts = {"attn_norm": "input_layernorm", "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
             "attn_output": "self_attn.o_proj", "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm", "ffn_gate": "mlp.gate_proj",
             "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj", "post_attention_norm": "post_attention_layernorm",
             "post_ffw_norm": "post_feedforward_layernorm",
             "ffn_norm": "pre_feedforward_layernorm" if "model.layers.0.pre_feedforward_layernorm.weight" in state else "post_attention_layernorm"}
    seen = 0
    for t in reader.tensors:
        if t.name in names:
            hf = names[t.name]
        else:
            _, layer, part, kind = t.name.split(".")
            hf = f"model.layers.{layer}.{parts[part]}.{kind}"
        stored = np.array(t.data, dtype=np.float32).reshape(state[hf].shape)
        expected = state[hf] + 1 if hf.endswith("norm.weight") else state[hf]
        assert np.array_equal(stored, expected.astype(np.float32)), f"{t.name}: not the checkpoint's {hf}"
        seen += 1
    assert seen == len(state) - ("lm_head.weight" in state), f"{path}: {seen} tensors for {len(state)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-cpp", required=True, help=f"a llama.cpp checkout at {LLAMA_CPP_TAG} (conversion and gguf-py)")
    args = parser.parse_args()
    llama_cpp = os.path.abspath(args.llama_cpp)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    spm, vocab_size = sentencepiece_model(os.path.join(TINY, "tokenizer.json"))
    files = {}
    for name in MODELS:
        folder, gguf = make(name, llama_cpp, vocab_size, spm)
        for path in [gguf] + [os.path.join(folder, f) for f in sorted(os.listdir(folder))]:
            files[os.path.relpath(path, OUT).replace(os.sep, "/")] = sha256(path)

    import sentencepiece
    import tokenizers
    import transformers
    head = subprocess.run(["git", "-C", llama_cpp, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    write_json(os.path.join(OUT, "gemma-manifest.json"), {
        "script": "tools/gguf/make_gemma.py", "llama.cpp": {"tag": LLAMA_CPP_TAG, "commit": head},
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
                     "tokenizers": tokenizers.__version__, "sentencepiece": sentencepiece.__version__, "numpy": np.__version__},
        "files": files})


if __name__ == "__main__":
    main()
