"""GGUF test fixtures for Idrak (model loading) (needs: pip install gguf numpy).

    python tools/gguf/make_fixtures.py tests/Idrak.Tests/data/gguf

quants.gguf holds, for every supported ggml type, a tensor of random blocks ("q.<TYPE>") and the float32 values the
gguf package's reference dequantizer gives for them ("expected.<TYPE>"). The model fixtures (tiny Llama and Qwen3
models, each as a GGUF file and as the Hugging Face folder it corresponds to) are written by the same script.
"""
import json
import os
import struct
import sys

import numpy as np
import gguf
from gguf import GGMLQuantizationType as T
from gguf import quants

TYPES = ["F16", "BF16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ4_NL", "IQ4_XS"]


def random_blocks(qtype, blocks, rng):
    """Random block bytes whose reference dequantization is finite (random half-float scales can be inf / nan)."""
    values, size = gguf.GGML_QUANT_SIZES[qtype]
    out = []
    while len(out) < blocks:
        raw = rng.integers(0, 256, size=size, dtype=np.uint8)
        deq = quants.dequantize(raw.reshape(1, size), qtype).astype(np.float32)
        if np.all(np.isfinite(deq)) and np.max(np.abs(deq)) < 1e6:
            out.append(raw)
    return np.stack(out).reshape(blocks * size)


def quants_file(folder):
    rng = np.random.default_rng(7)
    writer = gguf.GGUFWriter(os.path.join(folder, "quants.gguf"), "llama")
    for name in TYPES:
        qtype = T[name]
        values, size = gguf.GGML_QUANT_SIZES[qtype]
        blocks = max(1, 1024 // values)
        raw = random_blocks(qtype, blocks, rng)
        expected = quants.dequantize(raw.reshape(1, blocks * size), qtype).astype(np.float32).reshape(-1)
        writer.add_tensor(f"q.{name}", raw.reshape(1, blocks * size), raw_dtype=qtype)
        writer.add_tensor(f"expected.{name}", expected)
    writer.add_string("test.string", "héllo")
    writer.add_array("test.numbers", [1, 2, 3])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


# ------------------------------------------------------------------ tiny models

QWEN2_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
LLAMA3_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def byte_level_vocab():
    """GPT-2's byte-to-character table, a few merges, and special tokens."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    table = dict(zip(bs, (chr(c) for c in cs)))
    tokens = [table[b] for b in range(256)]
    merges = [("Ġ", "t"), ("h", "e"), ("Ġt", "he"), ("i", "n"), ("Ġ", "a"), ("e", "r"), ("o", "n"), ("Ġa", "n")]
    for a, b in merges:
        tokens.append(a + b)
    specials = ["<|begin|>", "<|end|>", "<|im_start|>", "<|im_end|>"]
    return tokens, merges, specials


CHAT_TEMPLATE = ("{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{%- endfor %}"
                 "{%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}")


def permute(w, n_head):
    """llama.cpp's convert_hf_to_gguf reordering of Llama's q / k rows."""
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def write_safetensors(path, tensors):
    header, offset, blobs = {}, 0, []
    for name, array in tensors.items():
        data = np.ascontiguousarray(array, dtype=np.float32).tobytes()
        header[name] = {"dtype": "F32", "shape": list(array.shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blobs.append(data)
    text = json.dumps(header).encode()
    text += b" " * ((8 - len(text) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)))
        f.write(text)
        for blob in blobs:
            f.write(blob)


def model(folder, name, arch, quant):
    rng = np.random.default_rng(11 if arch == "llama" else 12)
    tokens, merges, specials = byte_level_vocab()
    vocab = tokens + specials
    dim, layers, heads, kv_heads, head_dim, ff = 64, 2, 4, 2, 16, 128
    V = len(vocab)

    def w(*shape):
        return (rng.standard_normal(shape) * 0.08).astype(np.float32)

    hf = {"model.embed_tokens.weight": w(V, dim), "model.norm.weight": 1 + w(dim)}
    for i in range(layers):
        p = f"model.layers.{i}"
        hf[f"{p}.input_layernorm.weight"] = 1 + w(dim)
        hf[f"{p}.post_attention_layernorm.weight"] = 1 + w(dim)
        hf[f"{p}.self_attn.q_proj.weight"] = w(heads * head_dim, dim)
        hf[f"{p}.self_attn.k_proj.weight"] = w(kv_heads * head_dim, dim)
        hf[f"{p}.self_attn.v_proj.weight"] = w(kv_heads * head_dim, dim)
        hf[f"{p}.self_attn.o_proj.weight"] = w(dim, heads * head_dim)
        hf[f"{p}.mlp.gate_proj.weight"] = w(ff, dim)
        hf[f"{p}.mlp.up_proj.weight"] = w(ff, dim)
        hf[f"{p}.mlp.down_proj.weight"] = w(dim, ff)
        if arch == "qwen3":
            hf[f"{p}.self_attn.q_norm.weight"] = 1 + w(head_dim)
            hf[f"{p}.self_attn.k_norm.weight"] = 1 + w(head_dim)

    names = {"input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm", "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
             "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
             "mlp.down_proj": "ffn_down", "self_attn.q_norm": "attn_q_norm", "self_attn.k_norm": "attn_k_norm"}
    writer = gguf.GGUFWriter(os.path.join(folder, f"{name}.gguf"), arch)
    writer.add_name(name)
    writer.add_context_length(256)
    writer.add_embedding_length(dim)
    writer.add_block_count(layers)
    writer.add_feed_forward_length(ff)
    writer.add_head_count(heads)
    writer.add_head_count_kv(kv_heads)
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_rope_freq_base(10000.0)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("llama-bpe" if arch == "llama" else "qwen2")
    writer.add_token_list(vocab)
    writer.add_token_types([1] * len(tokens) + [3] * len(specials))
    writer.add_token_merges([f"{a} {b}" for a, b in merges])
    writer.add_bos_token_id(vocab.index("<|begin|>"))
    writer.add_eos_token_id(vocab.index("<|im_end|>"))
    writer.add_add_bos_token(False)
    writer.add_chat_template(CHAT_TEMPLATE)

    def add(gname, array):
        """Quantized when asked and the rows fit whole blocks; the Hugging Face copy gets the values the GGUF holds."""
        if quant is not None and array.ndim == 2 and array.shape[1] % 32 == 0:
            q = quants.quantize(array, quant)
            writer.add_tensor(gname, q, raw_dtype=quant)
            return quants.dequantize(q, quant).astype(np.float32).reshape(array.shape)
        writer.add_tensor(gname, array)
        return array

    back = {"model.embed_tokens.weight": add("token_embd.weight", hf["model.embed_tokens.weight"]),
            "model.norm.weight": add("output_norm.weight", hf["model.norm.weight"])}
    for i in range(layers):
        for key, gname in names.items():
            for suffix in ["weight"]:
                hname = f"model.layers.{i}.{key}.{suffix}"
                if hname not in hf:
                    continue
                array = hf[hname]
                if arch == "llama" and key == "self_attn.q_proj":
                    stored = add(f"blk.{i}.{gname}.{suffix}", permute(array, heads))
                    back[hname] = unpermute(stored, heads)
                elif arch == "llama" and key == "self_attn.k_proj":
                    stored = add(f"blk.{i}.{gname}.{suffix}", permute(array, kv_heads))
                    back[hname] = unpermute(stored, kv_heads)
                else:
                    back[hname] = add(f"blk.{i}.{gname}.{suffix}", array)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # The Hugging Face folder with the same (dequantized) weights, tokenizer and template.
    out = os.path.join(folder, f"{name}-hf")
    os.makedirs(out, exist_ok=True)
    write_safetensors(os.path.join(out, "model.safetensors"), back)
    config = {"architectures": ["LlamaForCausalLM" if arch == "llama" else "Qwen3ForCausalLM"], "vocab_size": V, "hidden_size": dim,
              "intermediate_size": ff, "num_hidden_layers": layers, "num_attention_heads": heads, "num_key_value_heads": kv_heads,
              "head_dim": head_dim, "max_position_embeddings": 256, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": True,
              "hidden_act": "silu"}
    json.dump(config, open(os.path.join(out, "config.json"), "w"))
    pattern = LLAMA3_PATTERN if arch == "llama" else QWEN2_PATTERN
    tokenizer = {
        "added_tokens": [{"id": len(tokens) + i, "content": s, "special": True} for i, s in enumerate(specials)],
        "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}]},
        "decoder": {"type": "ByteLevel"},
        "model": {"type": "BPE", "vocab": {t: i for i, t in enumerate(tokens)}, "merges": [f"{a} {b}" for a, b in merges]},
    }
    json.dump(tokenizer, open(os.path.join(out, "tokenizer.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump({"chat_template": CHAT_TEMPLATE, "bos_token": "<|begin|>", "eos_token": "<|im_end|>"},
              open(os.path.join(out, "tokenizer_config.json"), "w"))


def unpermute(w, n_head):
    return w.reshape(n_head, w.shape[0] // n_head // 2, 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def main():
    folder = sys.argv[1]
    os.makedirs(folder, exist_ok=True)
    quants_file(folder)
    model(folder, "tiny-llama", "llama", None)
    model(folder, "tiny-qwen3-q8", "qwen3", T.Q8_0)


if __name__ == "__main__":
    main()
