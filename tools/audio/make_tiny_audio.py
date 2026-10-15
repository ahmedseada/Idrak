"""The reference for Idrak.Audio (plan 21, phase S0): generated WAV files and their samples, torchaudio's resampling and
features, a tiny random Whisper-style model and transformers' outputs for it, a tiny CTC recognizer's loss, gradients
and alignments, a tiny VITS synthesizer's waveform, and transcript records.

    python -m venv audio && audio/bin/pip install torch torchaudio transformers tokenizers safetensors numpy
    audio/bin/python tools/audio/make_tiny_audio.py tests/Idrak.Tests/data/audio

Nothing is downloaded: the signals are generated here, the models are built from configs with seeded weights, the
tokenizer is trained here. Reruns write the same bytes (fixed seeds, one thread, deterministic algorithms, sorted
JSON) on the same machine and library versions; manifest.json records the versions and each file's SHA-256.
--sections wav runs the part that needs numpy only (the WAV files), for a machine without torch.
"""
import argparse
import json
import math
import os
import platform
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vlm"))
from make_tiny import dump_json, save_npy, sha256, safetensors_index  # noqa: E402

SECTIONS = ["wav", "resample", "features", "whisper", "ctc", "vits", "transcripts"]
RATE = 16000
SECONDS = 1.5

# Whisper's feature extraction (transformers' WhisperFeatureExtractor defaults).
N_FFT, HOP, MELS, FMIN, FMAX, CHUNK_SECONDS = 400, 160, 80, 0, 8000, 30

# The Whisper-style vocabulary: 600 tokens with these specials at the end (the timestamp tokens in 0.02 s steps).
SPECIALS = ["<|endoftext|>", "<|startoftranscript|>", "<|en|>", "<|ar|>", "<|translate|>", "<|transcribe|>",
            "<|notimestamps|>"]
TIMESTAMPS = 31
VOCAB = 600

CORPUS = [
    "حسب الذمة جدة. الإقامة عبر 1407 هـ. سعود بن خديجة",
    "كذلك قدمت النقض مع صاحب عنيزة إدارة العامة قبل الديوان اللازمة",
    "the quick brown fox jumps over the lazy dog",
    "a recording of one voice, read slowly, with numbers 12 345 and dates",
    "وفق ينبع قبل الرهن. كانت تكون القرار",
    "speech recognition from scratch needs many hours of transcribed sound",
    "لم حضر قرار عليه الموظف أرفق الشهادة الرسمية",
    "the mel filterbank has eighty bands between zero and eight thousand hertz",
]


# ------------------------------------------------------------------ signals and WAV files

def signal(rate, seed=2027):
    """[2, samples] in [-1, 1]: a 440 Hz tone sweeping to 880 Hz with a seeded noise burst (left), a 660 Hz tone
    with the burst later and quieter (right)."""
    n = int(rate * SECONDS)
    t = np.arange(n, dtype=np.float64) / rate
    sweep = np.sin(2 * math.pi * (440 * t + (880 - 440) / (2 * SECONDS) * t * t))
    tone = np.sin(2 * math.pi * 660 * t)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    burst = np.zeros(n)
    a, b = int(0.5 * rate), int(0.7 * rate)
    burst[a:b] = noise[a:b] * np.hanning(b - a)
    left = 0.6 * sweep + 0.3 * burst
    c, d = int(1.0 * rate), int(1.1 * rate)
    right = np.zeros(n)
    right[c:d] = noise[c:d] * np.hanning(d - c)
    right = 0.4 * tone + 0.15 * right
    return np.clip(np.stack([left, right]), -1, 1)


def quantize(samples, bits):
    """Integer PCM of `samples` (float, [channels, n]) at `bits` (8: unsigned), and the floats a decoder gives back."""
    full = 1 << (bits - 1)
    ints = np.clip(np.round(samples * full), -full, full - 1).astype(np.int64)
    decoded = (ints / full).astype(np.float32)
    if bits == 8:
        return (ints + 128).astype(np.uint8), decoded
    return ints, decoded


def pcm_bytes(ints, bits):
    """Interleaved little-endian bytes of integer samples [channels, n]."""
    inter = np.ascontiguousarray(ints.T)                       # [n, channels]
    if bits == 8:
        return inter.astype(np.uint8).tobytes()
    if bits == 16:
        return inter.astype("<i2").tobytes()
    if bits == 32:
        return inter.astype("<i4").tobytes()
    as32 = inter.astype("<i4").tobytes()                       # 24-bit: the low three bytes of each int32
    arr = np.frombuffer(as32, dtype=np.uint8).reshape(-1, 4)[:, :3]
    return np.ascontiguousarray(arr).tobytes()


def wav_bytes(samples, rate, kind, channels=None, list_chunk=False, truncate=False, no_fmt=False, oversized=False):
    """A RIFF/WAVE file. kind: pcm8, pcm16, pcm24, pcm32, float32, float64, extensible16 (WAVE_FORMAT_EXTENSIBLE
    with a channel mask). Returns (bytes, decoded float32 [channels, n])."""
    if channels is not None:
        samples = samples[:channels]
    c, n = samples.shape
    if kind.startswith("pcm"):
        bits = int(kind[3:])
        ints, decoded = quantize(samples, bits)
        data = pcm_bytes(ints, bits)
        tag, is_float = 1, False
    elif kind == "extensible16":
        bits = 16
        ints, decoded = quantize(samples, bits)
        data = pcm_bytes(ints, bits)
        tag, is_float = 0xFFFE, False
    elif kind == "float32":
        bits = 32
        decoded = samples.astype(np.float32)
        data = np.ascontiguousarray(decoded.T).astype("<f4").tobytes()
        tag, is_float = 3, True
    elif kind == "float64":
        bits = 64
        wide = samples.astype(np.float64)
        decoded = wide.astype(np.float32)
        data = np.ascontiguousarray(wide.T).astype("<f8").tobytes()
        tag, is_float = 3, True
    else:
        raise ValueError(kind)

    block = c * bits // 8
    fmt = struct.pack("<HHIIHH", tag, c, rate, rate * block, block, bits)
    if tag == 0xFFFE:
        sub = struct.pack("<H", 1 if not is_float else 3) + bytes.fromhex("000000001000800000aa00389b71")   # KSDATAFORMAT_SUBTYPE_PCM
        mask = 0x3 if c == 2 else (0x4 if c == 1 else (1 << c) - 1)
        fmt += struct.pack("<HHI", 22, bits, mask) + sub
    chunks = []
    if not no_fmt:
        chunks.append((b"fmt ", fmt))
    if list_chunk:
        info = b"INFO" + b"ISFT" + struct.pack("<I", 11) + b"make_tiny\0\0"
        chunks.append((b"LIST", info))                                               # 23 bytes: odd, padded below
    chunks.append((b"data", data))
    body = b"WAVE"
    for cid, payload in chunks:
        # oversized: the data chunk claims twice the samples the file holds (a phone that never fixed the header).
        body += cid + struct.pack("<I", len(payload) * 2 if (cid == b"data" and oversized) else len(payload)) + payload
        if len(payload) % 2:
            body += b"\0"
    file = b"RIFF" + struct.pack("<I", len(body)) + body
    return (file[:30] if truncate else file), decoded                             # truncated: cut inside the fmt chunk


def write_wavs(out):
    folder = os.path.join(out, "wav")
    os.makedirs(folder, exist_ok=True)
    base = signal(RATE)
    files = {}

    def put(name, data, decoded, rate):
        with open(os.path.join(folder, name + ".wav"), "wb") as f:
            f.write(data)
        if decoded is not None:
            save_npy(folder, name + ".npy", decoded)
            files[name] = {"rate": rate, "channels": int(decoded.shape[0]), "samples": int(decoded.shape[1])}

    for kind in ["pcm8", "pcm16", "pcm24", "pcm32", "float32", "float64"]:
        data, decoded = wav_bytes(base, RATE, kind, channels=1)
        put(kind, data, decoded, RATE)
    data, decoded = wav_bytes(base, RATE, "pcm16")
    put("stereo16", data, decoded, RATE)
    data, decoded = wav_bytes(base, RATE, "extensible16")
    put("extensible16", data, decoded, RATE)
    data, decoded = wav_bytes(base, RATE, "pcm16", channels=1, list_chunk=True)
    put("list-chunk16", data, decoded, RATE)
    for rate, name in [(8000, "rate8k"), (44100, "rate44k")]:
        data, decoded = wav_bytes(signal(rate), rate, "pcm16", channels=1)
        put(name, data, decoded, rate)
    data, decoded = wav_bytes(base, RATE, "pcm16", channels=1, oversized=True)
    put("oversized16", data, decoded, RATE)
    data, _ = wav_bytes(base, RATE, "pcm16", channels=1, truncate=True)
    put("bad-truncated", data, None, RATE)
    data, _ = wav_bytes(base, RATE, "pcm16", channels=1, no_fmt=True)
    put("bad-no-fmt", data, None, RATE)
    dump_json(os.path.join(folder, "files.json"), files)
    return files


# ------------------------------------------------------------------ torch sections

def torch_setup():
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch


def load_wav_npy(out, name):
    return np.load(os.path.join(out, "wav", name + ".npy"))


def write_resample(out):
    import torchaudio.functional as F
    torch = torch_setup()
    folder = os.path.join(out, "resample")
    os.makedirs(folder, exist_ok=True)
    params = dict(lowpass_filter_width=6, rolloff=0.99, resampling_method="sinc_interp_hann", target=RATE)
    for name, rate in [("rate44k", 44100), ("rate8k", 8000)]:
        wave = torch.from_numpy(load_wav_npy(out, name))
        res = F.resample(wave, rate, RATE, lowpass_filter_width=6, rolloff=0.99, resampling_method="sinc_interp_hann")
        save_npy(folder, name + "-16k.npy", res.numpy().astype(np.float32))
        params[name] = {"from": rate, "samples_in": int(wave.shape[1]), "samples_out": int(res.shape[1])}
    dump_json(os.path.join(folder, "resample.json"), params)


def log_mel(torch, F, audio):
    """transformers' WhisperFeatureExtractor._torch_extract_fbank_features for one 30 s segment, as float32 [80, 3000]."""
    window = torch.hann_window(N_FFT, periodic=True)
    stft = torch.stft(audio, N_FFT, HOP, window=window, center=True, pad_mode="reflect", return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    fbanks = F.melscale_fbanks(N_FFT // 2 + 1, FMIN, FMAX, MELS, RATE, norm="slaney", mel_scale="slaney")
    mel = fbanks.T @ magnitudes
    log_spec = torch.clamp(mel, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0, fbanks, stft.abs()


def write_features(out):
    import torchaudio.functional as F
    from transformers import WhisperFeatureExtractor
    torch = torch_setup()
    folder = os.path.join(out, "features")
    os.makedirs(folder, exist_ok=True)
    audio = torch.from_numpy(load_wav_npy(out, "pcm16")[0])
    _, fbanks, magnitude = log_mel(torch, F, audio)
    save_npy(folder, "stft-magnitude.npy", magnitude.numpy().astype(np.float32))
    save_npy(folder, "mel-filterbank.npy", fbanks.numpy().astype(np.float32))
    padded = torch.zeros(CHUNK_SECONDS * RATE)
    padded[: audio.shape[0]] = audio
    ours, _, _ = log_mel(torch, F, padded)
    extractor = WhisperFeatureExtractor(feature_size=MELS, sampling_rate=RATE, hop_length=HOP, chunk_length=CHUNK_SECONDS, n_fft=N_FFT)
    theirs = extractor(audio.numpy(), sampling_rate=RATE, return_tensors="np").input_features[0]
    assert np.abs(ours.numpy() - theirs).max() < 1e-4, "the log-mel here differs from WhisperFeatureExtractor's"
    save_npy(folder, "log-mel.npy", theirs.astype(np.float32))
    dump_json(os.path.join(folder, "features.json"), {
        "n_fft": N_FFT, "hop_length": HOP, "window": "hann periodic", "center": True, "pad_mode": "reflect", "mels": MELS,
        "fmin": FMIN, "fmax": FMAX, "sample_rate": RATE, "norm": "slaney", "mel_scale": "slaney", "padding_seconds": CHUNK_SECONDS,
        "power": 2, "log": "log10 clamped at 1e-10, floored at max - 8, then (x + 4) / 4",
        "stft_frames": int(magnitude.shape[1]), "log_mel_frames": int(theirs.shape[1]),
        "source": "wav/pcm16.wav (16 kHz already: no resampling)",
    })
    return extractor


def train_tokenizer(folder):
    """A byte-level BPE over CORPUS with the Whisper-style specials and timestamp tokens, 600 tokens in all."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    timestamps = [f"<|{0.02 * i:.2f}|>" for i in range(TIMESTAMPS)]
    specials = SPECIALS + timestamps
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB - len(specials), min_frequency=1, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=[])
    tok.train_from_iterator(CORPUS * 4, trainer)
    # The small corpus runs out of pairs before the target: unused tokens fill the vocabulary to its size, as a cut
    # vocabulary's gaps would, before the specials (which Whisper puts last).
    unused = [f"<|unused{i}|>" for i in range(VOCAB - len(specials) - tok.get_vocab_size())]
    if unused:
        tok.add_tokens(unused)
    tok.add_special_tokens(specials)
    assert tok.get_vocab_size() == VOCAB, tok.get_vocab_size()
    os.makedirs(folder, exist_ok=True)
    tok.save(os.path.join(folder, "tokenizer.json"))
    ids = {s: tok.token_to_id(s) for s in specials}
    return tok, ids


def randomize(model, torch, seed):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1 and ("norm" in name.lower() or name.endswith("bias")):
                p.fill_(1.0 if "weight" in name else 0.0)
            else:
                p.copy_(torch.randn(p.shape, generator=gen) * 0.02)


def write_whisper(out, extractor):
    from transformers import WhisperConfig, WhisperForConditionalGeneration
    from safetensors.torch import save_file
    torch = torch_setup()
    folder = os.path.join(out, "tiny-whisper")
    tok, ids = train_tokenizer(folder)
    config = WhisperConfig(vocab_size=VOCAB, d_model=32, encoder_layers=2, decoder_layers=2, encoder_attention_heads=4,
                           decoder_attention_heads=4, encoder_ffn_dim=64, decoder_ffn_dim=64, num_mel_bins=MELS,
                           max_source_positions=1500, max_target_positions=448,
                           bos_token_id=ids["<|endoftext|>"], eos_token_id=ids["<|endoftext|>"], pad_token_id=ids["<|endoftext|>"],
                           decoder_start_token_id=ids["<|startoftranscript|>"], dropout=0.0, attention_dropout=0.0, activation_dropout=0.0)
    model = WhisperForConditionalGeneration(config)
    randomize(model, torch, 2028)
    model.eval()
    model.save_pretrained(folder, safe_serialization=True)
    extractor.save_pretrained(folder)
    gen = {
        "decoder_start_token_id": ids["<|startoftranscript|>"], "eos_token_id": ids["<|endoftext|>"], "pad_token_id": ids["<|endoftext|>"],
        "bos_token_id": ids["<|endoftext|>"], "lang_to_id": {"<|en|>": ids["<|en|>"], "<|ar|>": ids["<|ar|>"]},
        "task_to_id": {"transcribe": ids["<|transcribe|>"], "translate": ids["<|translate|>"]},
        "no_timestamps_token_id": ids["<|notimestamps|>"], "max_length": 448, "is_multilingual": True,
        "suppress_tokens": [], "begin_suppress_tokens": [ids["<|endoftext|>"]],
    }
    dump_json(os.path.join(folder, "generation_config.json"), gen)
    dump_json(os.path.join(folder, "special_tokens.json"), ids)

    ref = os.path.join(folder, "reference")
    os.makedirs(ref, exist_ok=True)
    features = torch.from_numpy(np.load(os.path.join(out, "features", "log-mel.npy")))[None]
    prompt = [ids["<|startoftranscript|>"], ids["<|ar|>"], ids["<|transcribe|>"], ids["<|notimestamps|>"]]
    with torch.no_grad():
        encoded = model.model.encoder(features).last_hidden_state
        save_npy(ref, "encoder-output.npy", encoded[0].numpy().astype(np.float32))
        logits = model(input_features=features, decoder_input_ids=torch.tensor([prompt])).logits
        save_npy(ref, "logits-step1.npy", logits[0, -1].numpy().astype(np.float32))
        greedy = model.generate(features, decoder_input_ids=torch.tensor([prompt]), max_new_tokens=8, min_new_tokens=8, do_sample=False, num_beams=1)
        beam = model.generate(features, decoder_input_ids=torch.tensor([prompt]), max_new_tokens=8, min_new_tokens=8, do_sample=False,
                              num_beams=4, length_penalty=1.0, early_stopping=True)
        stamped = model.generate(features, decoder_input_ids=torch.tensor([prompt[:3]]), max_new_tokens=8, min_new_tokens=8, do_sample=False, num_beams=1)
    dump_json(os.path.join(ref, "greedy-ids.json"), {"prompt": prompt, "ids": greedy[0].tolist()})
    dump_json(os.path.join(ref, "beam4-ids.json"), {"prompt": prompt, "ids": beam[0].tolist(), "num_beams": 4, "length_penalty": 1.0, "early_stopping": True})
    dump_json(os.path.join(ref, "timestamps-ids.json"), {"prompt": prompt[:3], "ids": stamped[0].tolist()})
    index, _ = safetensors_index(os.path.join(folder, "model.safetensors"))
    dump_json(os.path.join(folder, "tensors.json"), index)


class CtcEncoder:
    """Two Conv2d (1, 3) stride (1, 2) with GELU over [N, 1, T, mels], a linear to width 32, two encoder layers, a head."""

    @staticmethod
    def build(torch, mels=MELS, width=32, classes=40):
        nn = torch.nn
        channels, bins = 4, (((mels - 3) // 2 + 1) - 3) // 2 + 1
        return nn.ModuleDict({
            "conv1": nn.Conv2d(1, channels, (1, 3), stride=(1, 2)),
            "conv2": nn.Conv2d(channels, channels, (1, 3), stride=(1, 2)),
            "proj": nn.Linear(channels * bins, width),
            "layer0": nn.TransformerEncoderLayer(width, 4, 64, dropout=0.0, activation="gelu", batch_first=True),
            "layer1": nn.TransformerEncoderLayer(width, 4, 64, dropout=0.0, activation="gelu", batch_first=True),
            "head": nn.Linear(width, classes),
        })

    @staticmethod
    def forward(torch, m, mel):
        x = torch.nn.functional.gelu(m["conv1"](mel[:, None]))
        x = torch.nn.functional.gelu(m["conv2"](x))                 # [N, C, T, bins]
        x = x.permute(0, 2, 1, 3).flatten(2)                        # [N, T, C * bins]
        x = m["proj"](x)
        x = m["layer1"](m["layer0"](x))
        return torch.log_softmax(m["head"](x), dim=-1)              # [N, T, classes]


def write_ctc(out):
    import torchaudio.functional as F
    from safetensors.torch import save_file
    torch = torch_setup()
    folder = os.path.join(out, "ctc")
    os.makedirs(folder, exist_ok=True)
    torch.manual_seed(2029)
    model = CtcEncoder.build(torch)
    randomize(model, torch, 2029)
    gen = torch.Generator().manual_seed(2029)
    lengths = [23, 17, 31, 12]
    labels = [[5, 7, 7, 12, 3], [1, 2], [30, 9, 9, 9, 14, 2, 8], [4, 4, 4]]
    mels = [torch.randn(t, MELS, generator=gen) for t in lengths]
    T = max(lengths)
    batch = torch.zeros(len(lengths), T, MELS)
    for i, m in enumerate(mels):
        batch[i, : m.shape[0]] = m
    targets = torch.tensor([l for ls in labels for l in ls])
    target_lengths = torch.tensor([len(l) for l in labels])
    input_lengths = torch.tensor(lengths)
    save_npy(folder, "mels.npy", batch.numpy().astype(np.float32))
    save_file({k: v.detach().clone() for k, v in model.state_dict().items()}, os.path.join(folder, "model.safetensors"))

    def step():
        logp = CtcEncoder.forward(torch, model, batch)                     # [N, T, C]
        return torch.nn.functional.ctc_loss(logp.transpose(0, 1), targets, input_lengths, target_lengths, blank=0, reduction="mean", zero_infinity=True), logp

    loss, logp = step()
    loss.backward()
    save_file({k: p.grad.detach().clone() for k, p in model.named_parameters()}, os.path.join(folder, "grads-step1.safetensors"))
    alignments = []
    with torch.no_grad():
        for i, ls in enumerate(labels):
            path, _ = F.forced_align(logp[i : i + 1, : lengths[i]], torch.tensor([ls]), blank=0)
            alignments.append(path[0].tolist())
    losses = [float(loss)]
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(3):
        opt.step()
        opt.zero_grad()
        loss, _ = step()
        losses.append(float(loss))
        loss.backward()
    save_file({k: v.detach().clone() for k, v in model.state_dict().items()}, os.path.join(folder, "weights-step3.safetensors"))
    dump_json(os.path.join(folder, "loss.json"), {"losses": losses, "sgd_lr": 0.1, "blank": 0, "classes": 40, "reduction": "mean", "zero_infinity": True,
                                                  "lengths": lengths, "labels": labels})
    dump_json(os.path.join(folder, "alignment.json"), {"alignments": alignments, "note": "torchaudio.functional.forced_align per utterance, frame by frame, blank 0"})
    index, _ = safetensors_index(os.path.join(folder, "model.safetensors"))
    dump_json(os.path.join(folder, "tensors.json"), index)


def write_vits(out):
    from transformers import VitsConfig, VitsModel
    torch = torch_setup()
    folder = os.path.join(out, "tiny-vits")
    os.makedirs(folder, exist_ok=True)
    symbols = list("_ abcdefghijklmnopqrstuvwx")[:20]
    vocab = {s: i for i, s in enumerate(symbols)}
    dump_json(os.path.join(folder, "vocab.json"), vocab)
    config = VitsConfig(vocab_size=len(symbols), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, ffn_dim=64,
                        flow_size=32, prior_encoder_num_flows=1, prior_encoder_num_wavenet_layers=2, posterior_encoder_num_wavenet_layers=2,
                        upsample_rates=[4, 4], upsample_kernel_sizes=[8, 8], upsample_initial_channel=32, resblock_kernel_sizes=[3],
                        resblock_dilation_sizes=[[1, 3]], spectrogram_bins=65, hop_length=16, speaker_embedding_size=0, num_speakers=1,
                        noise_scale=0.0, noise_scale_duration=0.0, wavenet_dilation_rate=1, wavenet_kernel_size=3,
                        duration_predictor_num_flows=2, duration_predictor_filter_channels=32, depth_separable_num_layers=1)
    model = VitsModel(config)
    randomize(model, torch, 2030)
    model.eval()
    model.save_pretrained(folder, safe_serialization=True)
    ids = torch.tensor([[vocab[c] for c in "abc ab"]])
    with torch.no_grad():
        output = model(ids)
    ref = os.path.join(folder, "reference")
    os.makedirs(ref, exist_ok=True)
    save_npy(ref, "waveform.npy", output.waveform[0].numpy().astype(np.float32))
    dump_json(os.path.join(ref, "input.json"), {"text": "abc ab", "ids": ids[0].tolist(), "noise_scale": 0.0, "noise_scale_duration": 0.0,
                                                "waveform_samples": int(output.waveform.shape[1]), "sequence_lengths": output.sequence_lengths.tolist()})
    index, _ = safetensors_index(os.path.join(folder, "model.safetensors"))
    dump_json(os.path.join(folder, "tensors.json"), index)


def write_transcripts(out):
    folder = os.path.join(out, "transcripts")
    os.makedirs(folder, exist_ok=True)
    files = ["wav/pcm16.wav", "wav/stereo16.wav", "wav/rate8k.wav", "wav/rate44k.wav", "wav/float32.wav", "wav/list-chunk16.wav"]
    with open(os.path.join(folder, "train.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for i, text in enumerate(CORPUS[:6]):
            f.write(json.dumps({"audio": files[i], "text": text}, ensure_ascii=False) + "\n")
    with open(os.path.join(folder, "train-sharegpt.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for i, text in enumerate(CORPUS[:6]):
            row = {"conversations": [{"from": "human", "value": "<audio>Transcribe."}, {"from": "gpt", "value": text}], "audios": [files[i]]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ main

def manifest(out, sections):
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ["torch", "torchaudio", "transformers", "tokenizers", "safetensors"]:
        try:
            versions[name] = __import__(name).__version__
        except ImportError:
            versions[name] = None
    files = {}
    for root, _, names in os.walk(out):
        for n in sorted(names):
            if n in ("manifest.json", "README.md"):
                continue
            p = os.path.join(root, n)
            files[os.path.relpath(p, out).replace(os.sep, "/")] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
    dump_json(os.path.join(out, "manifest.json"), {"versions": versions, "sections": sections, "files": files})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--sections", default=",".join(SECTIONS), help="comma-separated: " + ", ".join(SECTIONS))
    args = parser.parse_args()
    sections = [s.strip() for s in args.sections.split(",") if s.strip()]
    os.makedirs(args.out, exist_ok=True)
    extractor = None
    if "wav" in sections:
        write_wavs(args.out)
    if "resample" in sections:
        write_resample(args.out)
    if "features" in sections:
        extractor = write_features(args.out)
    if "whisper" in sections:
        if extractor is None:
            from transformers import WhisperFeatureExtractor
            extractor = WhisperFeatureExtractor(feature_size=MELS, sampling_rate=RATE, hop_length=HOP, chunk_length=CHUNK_SECONDS, n_fft=N_FFT)
        write_whisper(args.out, extractor)
    if "ctc" in sections:
        write_ctc(args.out)
    if "vits" in sections:
        write_vits(args.out)
    if "transcripts" in sections:
        write_transcripts(args.out)
    manifest(args.out, sections)
    print("wrote", args.out, "sections", ",".join(sections))


if __name__ == "__main__":
    main()
