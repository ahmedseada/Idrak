"""The reference for Idrak.Audio (plan 21, phase S0): generated WAV files and their samples, torchaudio's resampling and
features, a tiny random Whisper-style model and transformers' outputs for it, a tiny CTC recognizer's loss, gradients
and alignments, a tiny encoder-decoder recognizer's loss, gradients and greedy texts, a tiny IdrakAudio synthesizer's
loss and gradients, a tiny VITS synthesizer's waveform, and transcript records.

    python -m venv audio && audio/bin/pip install torch torchaudio transformers tokenizers safetensors numpy soundfile
    audio/bin/python tools/audio/make_tiny_audio.py tests/Idrak.Tests/data/audio

Nothing is downloaded: the signals are generated here, the models are built from configs with seeded weights, the
tokenizer is trained here. Reruns write the same bytes (fixed seeds, one thread, deterministic algorithms, sorted
JSON) on the same machine and library versions; manifest.json records the versions and each file's SHA-256.
--sections wav runs the part that needs numpy only (the WAV files), for a machine without torch; --sections flac needs
soundfile (libsndfile with libFLAC) and numpy; --sections mp3 needs ffmpeg (with libmp3lame and libshine) and soundfile
(libsndfile with mpg123, a second decoder the first is checked against).
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

SECTIONS = ["wav", "flac", "mp3", "resample", "features", "whisper", "whisper-train", "ctc", "seq2seq", "audio-language", "tts", "vits", "vits-speakers", "voices", "vits-train", "transcripts"]
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


# ------------------------------------------------------------------ FLAC files (soundfile: libsndfile with libFLAC)

def write_flacs(out):
    """FLAC files of the WAV section's signal, each with the integer samples it holds (as floats, the integer over
    2^(bits-1)) in a .npy: the default level, level 0 (fixed predictors only) and level 8 (LPC up to order 12, exhaustive),
    stereo (libFLAC's left/side, right/side or mid/side), 8, 16 and 24 bits, 44.1 kHz, a stereo file of silence then
    full-scale noise then the tone (constant and verbatim subframes), a file whose samples are all multiples of 4 (wasted
    bits), and two damaged copies: one cut inside a frame, one with a byte of the first frame's data flipped (its CRC)."""
    import soundfile as sf
    folder = os.path.join(out, "flac")
    os.makedirs(folder, exist_ok=True)
    base = signal(RATE)
    files = {}

    def put(name, ints, bits, rate, level=None):
        # ints: [channels, n] at `bits`; soundfile takes int16 for 8 and 16 bits (8: the value in the high byte) and int32
        # for 24 (the value in the high three bytes), as libsndfile scales them.
        subtype = {8: "PCM_S8", 16: "PCM_16", 24: "PCM_24"}[bits]
        frames = np.ascontiguousarray(ints.T)
        data = (frames << 8).astype(np.int16) if bits == 8 else frames.astype(np.int16) if bits == 16 else (frames << 8).astype(np.int32)
        path = os.path.join(folder, name + ".flac")
        kwargs = {} if level is None else {"compression_level": level}
        sf.write(path, data, rate, format="FLAC", subtype=subtype, **kwargs)
        decoded = (ints / float(1 << (bits - 1))).astype(np.float32)
        back, _ = sf.read(path, dtype="int32", always_2d=True)                    # libsndfile's own decoding agrees
        assert np.array_equal(back.T >> (32 - bits), ints), name
        save_npy(folder, name + ".npy", decoded)
        files[name] = {"rate": rate, "channels": int(ints.shape[0]), "samples": int(ints.shape[1]), "bits": bits}
        return path

    def ints(samples, bits):
        full = 1 << (bits - 1)
        return np.clip(np.round(samples * full), -full, full - 1).astype(np.int64)

    mono = base[:1]
    put("pcm16", ints(mono, 16), 16, RATE)
    put("fixed16", ints(mono, 16), 16, RATE, level=0.0)
    put("lpc16", ints(mono, 16), 16, RATE, level=1.0)
    put("stereo16", ints(base, 16), 16, RATE)
    put("pcm8", ints(mono, 8), 8, RATE)
    put("pcm24", ints(mono, 24), 24, RATE)
    put("rate44k", ints(signal(44100)[:1], 16), 16, 44100)
    n = base.shape[1]
    rng = np.random.default_rng(31)
    mixed = np.zeros((2, n))
    q = n // 4
    mixed[:, q:2 * q] = rng.uniform(-1, 1, (2, q))                                 # full-scale noise: verbatim
    mixed[:, 2 * q:] = base[:, 2 * q:]
    put("mixed16", ints(mixed, 16), 16, RATE)
    put("wasted16", ints(mono, 16) & ~3, 16, RATE)
    with open(os.path.join(folder, "pcm16.flac"), "rb") as f:
        whole = f.read()
    with open(os.path.join(folder, "bad-truncated.flac"), "wb") as f:
        f.write(whole[: len(whole) // 2])
    first = whole.index(b"\xff\xf8", 42)                                         # the first frame (after the metadata)
    flipped = bytearray(whole)
    flipped[first + 40] ^= 0x5A
    with open(os.path.join(folder, "bad-crc.flac"), "wb") as f:
        f.write(bytes(flipped))
    dump_json(os.path.join(folder, "files.json"), files)
    return files


# ------------------------------------------------------------------ MP3 and MP2 files (ffmpeg's encoders and decoder)

def write_mp3s(out):
    """MPEG audio files of the WAV section's signal through ffmpeg (LAME, shine, ffmpeg's Layer II encoder), each with
    ffmpeg's decoding (mp3float; gapless: a LAME tag's delay and padding trimmed) as float32 [channels, samples] in a
    .npy. mpg123 (through libsndfile) decodes each too: the two decoders must agree within 2e-4 and in length, so the
    reference is not one decoder's quirk."""
    import subprocess
    import tempfile
    import soundfile as sf
    folder = os.path.join(out, "mp3")
    os.makedirs(folder, exist_ok=True)
    files = {}
    work = tempfile.mkdtemp()

    def source(rate, channels):
        path = os.path.join(work, f"s{rate}-{channels}.wav")
        if not os.path.exists(path):
            x = signal(rate)[:channels]
            sf.write(path, np.ascontiguousarray(x.T), rate, subtype="FLOAT")
        return path

    def ffmpeg(*args):
        subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)

    def decode(path, channels):
        raw = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", path, "-f", "f32le", "-c:a", "pcm_f32le", "-"],
                             check=True, capture_output=True).stdout
        return np.frombuffer(raw, dtype="<f4").reshape(-1, channels).T.copy()

    def put(name, rate, channels, encoder_args, ext="mp3", extra=(), mpg123=True):
        path = os.path.join(folder, f"{name}.{ext}")
        ffmpeg("-i", source(rate, channels), *encoder_args, *extra, path)
        ref = decode(path, channels)
        if mpg123 and ext == "mp3":
            other, _ = sf.read(path, dtype="float32", always_2d=True)
            other = other.T
            assert other.shape == ref.shape, (name, other.shape, ref.shape)
            assert np.abs(other - ref).max() < 2e-4, (name, float(np.abs(other - ref).max()))
        save_npy(folder, f"{name}.npy", ref.astype(np.float32))
        files[name] = {"file": f"{name}.{ext}", "rate": rate, "channels": channels, "samples": int(ref.shape[1])}

    lame, shine = ["-c:a", "libmp3lame"], ["-c:a", "libshine"]
    put("lame-joint44k", 44100, 2, lame + ["-b:a", "128k"])
    put("lame-stereo48k", 48000, 2, lame + ["-b:a", "192k", "-joint_stereo", "0"])
    put("lame-vbr32k-mono", 32000, 1, lame + ["-q:a", "2"])
    put("lame-22k-mono", 22050, 1, lame + ["-b:a", "64k"])
    put("lame-vbr24k", 24000, 2, lame + ["-q:a", "4"])
    put("lame-16k-mono", 16000, 1, lame + ["-b:a", "32k"])
    put("lame-8k-mono", 8000, 1, lame + ["-b:a", "16k"])
    put("lame-11k", 11025, 2, lame + ["-b:a", "32k"])
    put("lame-12k-mono", 12000, 1, lame + ["-b:a", "24k"])
    put("lame-notag44k", 44100, 2, lame + ["-b:a", "128k", "-write_xing", "0"], mpg123=False)
    put("shine44k", 44100, 2, shine + ["-b:a", "128k"], mpg123=False)
    put("shine32k-mono", 32000, 1, shine + ["-b:a", "64k"], mpg123=False)
    put("tagged44k", 44100, 2, lame + ["-b:a", "128k", "-write_id3v1", "1", "-id3v2_version", "3", "-metadata", "title=Idrak fixture",
                                         "-metadata", "artist=make_tiny_audio.py"])
    put("mp2-44k", 44100, 2, ["-c:a", "mp2", "-b:a", "192k"], ext="mp2")
    put("mp2-24k-mono", 24000, 1, ["-c:a", "mp2", "-b:a", "64k"], ext="mp2")
    with open(os.path.join(folder, "lame-joint44k.mp3"), "rb") as f:
        whole = f.read()
    with open(os.path.join(folder, "bad-truncated.mp3"), "wb") as f:
        f.write(whole[: len(whole) // 2 + 37])                                # cut inside a frame
    dump_json(os.path.join(folder, "files.json"), files)
    return files


# ------------------------------------------------------------------ voices: a tiny OpenVoice tone colour converter

def write_voices(out):
    """A tiny random OpenVoice converter (openvoice-cli's SynthesizerTrn with n_speakers 0: the reference encoder, the
    posterior encoder, the flows and the HiFi-GAN decoder), saved as model.safetensors (the state dict, weight norm's
    weight_g and weight_v as stored, as tools/audio/convert_openvoice.py writes it) and as checkpoint.pth
    (torch.save({"model": state_dict}), as OpenVoice ships it), with OpenVoice's config.json; and its outputs: the linear spectrogram of wav/pcm16, the speaker embeddings of the
    source and of two target clips (and their mean, as extract_se takes it), and the conversion of the source to the
    target with tau 0 (no noise), zero_g on (OpenVoice v2) and off (v1). The reference encoder's six weight-normed
    convolutions get gains of ten times the drawn ones' size: at the drawn gains (about 0.16 a filter) the signal fades
    through them, the GRU sees almost nothing, and every full-length clip gets the same embedding (references that
    cannot tell a right encoder from a wrong one)."""
    torch = torch_setup()
    from openvoice_cli.models import SynthesizerTrn
    from openvoice_cli.mel_processing import spectrogram_torch
    from safetensors.torch import save_file
    folder = os.path.join(out, "tiny-openvoice")
    ref = os.path.join(folder, "reference")
    os.makedirs(ref, exist_ok=True)
    data = {"sampling_rate": 16000, "filter_length": 64, "hop_length": 16, "win_length": 64, "n_speakers": 0}
    model_cfg = {"inter_channels": 8, "hidden_channels": 16, "filter_channels": 32, "n_heads": 2, "n_layers": 2, "kernel_size": 3,
                 "p_dropout": 0.0, "resblock": "1", "resblock_kernel_sizes": [3], "resblock_dilation_sizes": [[1, 3, 5]],
                 "upsample_rates": [4, 4], "upsample_initial_channel": 32, "upsample_kernel_sizes": [8, 8], "gin_channels": 16, "zero_g": True}
    config = {"_version_": "v2", "data": data, "model": model_cfg}

    def build(zero_g):
        m = SynthesizerTrn(0, data["filter_length"] // 2 + 1, n_speakers=0, **{**model_cfg, "zero_g": zero_g})
        randomize(m, torch, 2031, std=0.2)
        with torch.no_grad():
            for name, p in m.ref_enc.convs.named_parameters():
                if name.endswith("weight_g"):
                    p.copy_(p.abs() * 10)
        return m.eval()

    model = build(True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state, os.path.join(folder, "model.safetensors"))
    torch.save({"model": state}, os.path.join(folder, "checkpoint.pth"))         # as OpenVoice ships it: read by PyTorchCheckpointReader
    dump_json(os.path.join(folder, "config.json"), config)

    def spec(samples):
        y = torch.from_numpy(samples.astype(np.float32))[None]
        return spectrogram_torch(y, data["filter_length"], data["sampling_rate"], data["hop_length"], data["win_length"], center=False)

    def embed(m, samples):
        with torch.no_grad():
            return m.ref_enc(spec(samples).transpose(1, 2)).unsqueeze(-1)                # [1, gin, 1]

    source = load_wav_npy(out, "pcm16")[0]
    other = load_wav_npy(out, "stereo16")[1]                                        # the second channel: another "speaker"
    clips = [other, other[len(other) // 2:]]
    save_npy(ref, "spectrogram.npy", spec(source)[0].numpy().astype(np.float32))     # [freq, frames]
    src = embed(model, source)
    embs = [embed(model, c) for c in clips]
    tgt = torch.stack(embs).mean(0)
    save_npy(ref, "embedding-source.npy", src[0, :, 0].numpy().astype(np.float32))
    save_npy(ref, "embedding-clips.npy", torch.cat([e[0, :, 0][None] for e in embs]).numpy().astype(np.float32))
    save_npy(ref, "embedding-target.npy", tgt[0, :, 0].numpy().astype(np.float32))
    y = spec(source)
    lengths = torch.LongTensor([y.size(-1)])
    with torch.no_grad():
        converted = model.voice_conversion(y, lengths, sid_src=src, sid_tgt=tgt, tau=0.0)[0][0, 0]
        v1 = build(False)
        v1.load_state_dict(state)
        converted_v1 = v1.voice_conversion(y, lengths, sid_src=src, sid_tgt=tgt, tau=0.0)[0][0, 0]
    save_npy(ref, "converted-v2.npy", converted.numpy().astype(np.float32))
    save_npy(ref, "converted-v1.npy", converted_v1.numpy().astype(np.float32))
    dump_json(os.path.join(ref, "input.json"), {"source": "wav/pcm16 (its samples)", "clips": ["wav/stereo16 channel 1", "wav/stereo16 channel 1, second half"],
                                               "tau": 0.0, "frames": int(y.size(-1)), "samples": int(converted.numel())})
    index, _ = safetensors_index(os.path.join(folder, "model.safetensors"))
    dump_json(os.path.join(folder, "tensors.json"), index)


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


def randomize(model, torch, seed, std=0.02):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1 and ("norm" in name.lower() or name.endswith("bias")):
                p.fill_(1.0 if "weight" in name else 0.0)
            else:
                p.copy_(torch.randn(p.shape, generator=gen) * std)


def write_whisper(out, extractor):
    """A tiny random Whisper and transformers' outputs for it. The weights are drawn at std 0.5: at the usual 0.02 the
    tied head makes the model repeat its last input token, and greedy, beam and timestamp decoding all give one token
    over and over (references that cannot tell a right decoder from a wrong one). The decodings are transformers' plain
    generate with Whisper's processors passed in (the suppressed first tokens; with timestamps, the timestamp rules), as
    the plug-in applies them."""
    from transformers import WhisperConfig, WhisperForConditionalGeneration, GenerationConfig, LogitsProcessorList
    from transformers.generation.logits_process import SuppressTokensAtBeginLogitsProcessor, WhisperTimeStampLogitsProcessor
    from transformers.generation.utils import GenerationMixin
    torch = torch_setup()
    folder = os.path.join(out, "tiny-whisper")
    tok, ids = train_tokenizer(folder)
    eos = ids["<|endoftext|>"]
    config = WhisperConfig(vocab_size=VOCAB, d_model=32, encoder_layers=2, decoder_layers=2, encoder_attention_heads=4,
                           decoder_attention_heads=4, encoder_ffn_dim=64, decoder_ffn_dim=64, num_mel_bins=MELS,
                           max_source_positions=1500, max_target_positions=448,
                           bos_token_id=eos, eos_token_id=eos, pad_token_id=eos,
                           decoder_start_token_id=ids["<|startoftranscript|>"], dropout=0.0, attention_dropout=0.0, activation_dropout=0.0,
                           begin_suppress_tokens=[eos], suppress_tokens=[])
    model = WhisperForConditionalGeneration(config)
    randomize(model, torch, 2028, std=0.5)
    model.eval()
    model.save_pretrained(folder, safe_serialization=True)
    extractor.save_pretrained(folder)
    gen = {
        "decoder_start_token_id": ids["<|startoftranscript|>"], "eos_token_id": eos, "pad_token_id": eos,
        "bos_token_id": eos, "lang_to_id": {"<|en|>": ids["<|en|>"], "<|ar|>": ids["<|ar|>"]},
        "task_to_id": {"transcribe": ids["<|transcribe|>"], "translate": ids["<|translate|>"]},
        "no_timestamps_token_id": ids["<|notimestamps|>"], "max_initial_timestamp_index": 5, "max_length": 448, "is_multilingual": True,
        "suppress_tokens": [], "begin_suppress_tokens": [eos],
    }
    dump_json(os.path.join(folder, "generation_config.json"), gen)
    dump_json(os.path.join(folder, "special_tokens.json"), ids)

    ref = os.path.join(folder, "reference")
    os.makedirs(ref, exist_ok=True)
    features = torch.from_numpy(np.load(os.path.join(out, "features", "log-mel.npy")))[None]
    prompt = [ids["<|startoftranscript|>"], ids["<|ar|>"], ids["<|transcribe|>"], ids["<|notimestamps|>"]]
    new_tokens = 12

    def generate(prompt, beams, timestamps):
        settings = GenerationConfig(decoder_start_token_id=ids["<|startoftranscript|>"], eos_token_id=eos, pad_token_id=eos, bos_token_id=eos,
                                    max_new_tokens=new_tokens, num_beams=beams, length_penalty=1.0, early_stopping=True, do_sample=False,
                                    no_timestamps_token_id=ids["<|notimestamps|>"], max_initial_timestamp_index=5)
        model.generation_config = settings
        processors = [SuppressTokensAtBeginLogitsProcessor([eos], len(prompt))]
        if timestamps:
            processors.append(WhisperTimeStampLogitsProcessor(settings, begin_index=len(prompt)))
        out = GenerationMixin.generate(model, input_features=features, decoder_input_ids=torch.tensor([prompt]),
                                       logits_processor=LogitsProcessorList(processors))
        return out[0, len(prompt):].tolist()

    with torch.no_grad():
        encoded = model.model.encoder(features).last_hidden_state
        save_npy(ref, "encoder-output.npy", encoded[0].numpy().astype(np.float32))
        logits = model(input_features=features, decoder_input_ids=torch.tensor([prompt])).logits
        save_npy(ref, "logits-step1.npy", logits[0, -1].numpy().astype(np.float32))
        start = model(input_features=features, decoder_input_ids=torch.tensor([prompt[:1]])).logits[0, -1]
        languages = {code: float(start[i]) for code, i in [("en", ids["<|en|>"]), ("ar", ids["<|ar|>"])]}
        greedy = generate(prompt, 1, False)
        beam = generate(prompt, 4, False)
        stamped = generate(prompt[:3], 1, True)
        stamped_beam = generate(prompt[:3], 4, True)
    assert greedy != beam, "beam search should differ from greedy on the fixture"
    dump_json(os.path.join(ref, "greedy-ids.json"), {"prompt": prompt, "ids": greedy, "max_new_tokens": new_tokens})
    dump_json(os.path.join(ref, "beam4-ids.json"), {"prompt": prompt, "ids": beam, "num_beams": 4, "length_penalty": 1.0, "early_stopping": True, "max_new_tokens": new_tokens})
    dump_json(os.path.join(ref, "timestamps-ids.json"), {"prompt": prompt[:3], "ids": stamped, "max_new_tokens": new_tokens})
    dump_json(os.path.join(ref, "timestamps-beam4-ids.json"), {"prompt": prompt[:3], "ids": stamped_beam, "num_beams": 4, "max_new_tokens": new_tokens})
    dump_json(os.path.join(ref, "language.json"), {"logits": languages, "detected": max(languages, key=languages.get)})
    index, _ = safetensors_index(os.path.join(folder, "model.safetensors"))
    dump_json(os.path.join(folder, "tensors.json"), index)


def write_whisper_train(out):
    """Plan 29: transformers' training pass of tiny-whisper (as write_whisper saved it) on a batch of two utterances, as
    Idrak's "seq2seq" objective computes it. The features: the fixture's 30 s log-mel and the same reversed in time
    ([2, 3000, 80], the trainer's layout). Each utterance's labels are the prompt after <|startoftranscript|>
    (<|ar|><|transcribe|><|notimestamps|>) then its text's tokens; the decoder reads the start token and the labels
    (padded with the end token), is taught each label and then the end token, and the loss is each utterance's mean
    cross-entropy averaged over the batch. Written: the labels, the loss and every trained parameter's gradient under
    transformers' names (tied, so the head's part is in embed_tokens; the encoder's fixed positions get none)."""
    from transformers import WhisperForConditionalGeneration
    from safetensors.torch import save_file
    torch = torch_setup()
    F = torch.nn.functional
    folder = os.path.join(out, "tiny-whisper")
    ids = json.load(open(os.path.join(folder, "special_tokens.json"), encoding="utf-8"))
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(folder, "tokenizer.json"))
    model = WhisperForConditionalGeneration.from_pretrained(folder, dtype=torch.float32)
    model.train()
    one = np.load(os.path.join(out, "features", "log-mel.npy")).astype(np.float32)          # [80, 3000]
    features = torch.from_numpy(np.stack([one, one[:, ::-1].copy()]))                           # [2, 80, 3000]
    texts = [CORPUS[0], CORPUS[2]]
    prefix = [ids["<|ar|>"], ids["<|transcribe|>"], ids["<|notimestamps|>"]]
    labels = [prefix + tok.encode(t).ids for t in texts]
    eos, sot = ids["<|endoftext|>"], ids["<|startoftranscript|>"]
    length = max(len(l) for l in labels) + 1
    inputs = torch.full((2, length), eos, dtype=torch.long)
    targets = torch.full((2, length), -100, dtype=torch.long)
    for b, row in enumerate(labels):
        inputs[b, 0] = sot
        inputs[b, 1:len(row) + 1] = torch.tensor(row)
        targets[b, :len(row)] = torch.tensor(row)
        targets[b, len(row)] = eos
    logits = model(input_features=features, decoder_input_ids=inputs).logits
    ce = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=-100, reduction="none")      # [2, length]
    counts = (targets != -100).sum(1)
    loss = (ce.sum(1) / counts).mean()
    loss.backward()
    grads = {name: p.grad.detach().contiguous() for name, p in model.named_parameters() if p.grad is not None}
    ref = os.path.join(folder, "reference")
    os.makedirs(ref, exist_ok=True)
    save_file(grads, os.path.join(ref, "train-grads.safetensors"))
    dump_json(os.path.join(ref, "train.json"), {"texts": texts, "labels": labels, "loss": float(loss), "reversed_second": True,
                                                "language": "ar", "task": "transcribe"})


# The IdrakAudio plug-in's "ctc" task in torch (plan 21, S6): the same layers, names and order as
# src/Idrak.Audio.Plugins/IdrakAudio builds, written here from torch's own modules so the references do not come from
# Idrak. Features [N, T, mels] padded with zeros; two Conv2d (1, k) stride (1, 2) over time with the mels as channels
# (each output past an utterance's frames set to zero, so a batch gives each utterance what it gives alone), sinusoidal
# positions, pre-norm blocks whose attention sees each utterance's own steps (key padding), optionally a Conformer-style
# convolution module between attention and feed-forward, a final LayerNorm and a linear head (blank 0).
CTC = dict(mels=MELS, dim=32, layers=2, heads=4, ffn_dim=64, subsample_layers=2, subsample_kernel=3, classes=40, blank_id=0,
           convolution_kernel=5, norm_epsilon=1e-5, activation="gelu_tanh")


def ctc_modules(torch, convolution):
    nn = torch.nn
    F = torch.nn.functional
    c = CTC
    act = lambda x: F.gelu(x, approximate="tanh")

    class ConvolutionModule(nn.Module):
        def __init__(self):
            super().__init__()
            d = c["dim"]
            self.pointwise1 = nn.Linear(d, 2 * d)
            self.depthwise = nn.Conv2d(d, d, (1, c["convolution_kernel"]), padding=(0, c["convolution_kernel"] // 2), groups=d)
            self.depthwise_norm = nn.LayerNorm(d, eps=c["norm_epsilon"])
            self.pointwise2 = nn.Linear(d, d)

        def forward(self, x, keep):                                         # keep [N, T, 1]: 1 on an utterance's steps
            d = c["dim"]
            h = self.pointwise1(x)
            h = (h[..., :d] * torch.sigmoid(h[..., d:])) * keep               # GLU, padding zeroed before the depthwise convolution
            h = self.depthwise(h.transpose(1, 2)[:, :, None, :])[:, :, 0, :].transpose(1, 2)
            return self.pointwise2(F.silu(self.depthwise_norm(h)))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            d, eps = c["dim"], c["norm_epsilon"]
            self.norm1 = nn.LayerNorm(d, eps=eps)
            self.self_attn = nn.MultiheadAttention(d, c["heads"], batch_first=True)
            if convolution:
                self.conv_norm = nn.LayerNorm(d, eps=eps)
                self.conv = ConvolutionModule()
            self.norm2 = nn.LayerNorm(d, eps=eps)
            self.linear1 = nn.Linear(d, c["ffn_dim"])
            self.linear2 = nn.Linear(c["ffn_dim"], d)

        def forward(self, x, pad, keep):
            h = self.norm1(x)
            x = x + self.self_attn(h, h, h, key_padding_mask=pad, need_weights=False)[0]
            if convolution:
                x = x + self.conv(self.conv_norm(x), keep)
            return x + self.linear2(act(self.linear1(self.norm2(x))))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            d, k = c["dim"], c["subsample_kernel"]
            self.subsample = nn.ModuleList([nn.Conv2d(c["mels"] if i == 0 else d, d, (1, k), stride=(1, 2), padding=(0, k // 2))
                                            for i in range(c["subsample_layers"])])
            self.layers = nn.ModuleList([Block() for _ in range(c["layers"])])
            self.norm = nn.LayerNorm(d, eps=c["norm_epsilon"])
            self.head = nn.Linear(d, c["classes"])

        def forward(self, mel, frames):                                     # mel [N, T, mels]; frames [N]
            x, steps = self.encode(mel, frames)
            return torch.log_softmax(self.head(x), dim=-1), steps

        def encode(self, mel, frames):                                      # the normed states [N, T', dim] and each utterance's steps
            k = c["subsample_kernel"]
            x = mel.transpose(1, 2)[:, :, None, :]                          # [N, mels, 1, T]
            steps = frames.clone()
            for conv in self.subsample:
                x = act(conv(x))
                steps = (steps + 2 * (k // 2) - k) // 2 + 1
                keep = (torch.arange(x.shape[-1])[None, :] < steps[:, None]).to(x.dtype)
                x = x * keep[:, None, None, :]
            x = x[:, :, 0, :].transpose(1, 2)                               # [N, T', dim]
            t, d = x.shape[1], x.shape[2]
            pos = torch.arange(t, dtype=torch.float64)[:, None]
            i = torch.arange(d)[None, :]
            angle = pos / torch.pow(torch.tensor(10000.0, dtype=torch.float64), (2 * (i // 2)).to(torch.float64) / d)
            x = x + torch.where(i % 2 == 0, torch.sin(angle), torch.cos(angle)).to(x.dtype)
            pad = torch.arange(t)[None, :] >= steps[:, None]                # True: a padded step
            keep = (~pad).to(x.dtype)[..., None]
            for layer in self.layers:
                x = layer(x, pad, keep)
            return self.norm(x), steps

    return Model()


def ctc_initialize(torch, model, seed):
    """Weights at 1/sqrt(fan-in), biases and norm offsets drawn too (at the usual zeros and ones a wrong mapping of them
    would not show), so every tensor's gradient carries information."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.copy_((1.0 if name.endswith("weight") else 0.0) + 0.1 * torch.randn(p.shape, generator=gen))
            elif p.dim() == 1:
                p.copy_(0.1 * torch.randn(p.shape, generator=gen))
            else:
                fan_in = p[0].numel()
                p.copy_(torch.randn(p.shape, generator=gen) / math.sqrt(fan_in))


def ctc_config(convolution):
    c = CTC
    return {
        "architectures": ["IdrakAudio"], "model_type": "idrak-audio", "task": "ctc",
        "mels": c["mels"], "dim": c["dim"], "layers": c["layers"], "heads": c["heads"], "ffn_dim": c["ffn_dim"],
        "subsample_layers": c["subsample_layers"], "subsample_kernel": c["subsample_kernel"], "positions": "sinusoidal",
        "convolution_module": convolution, "convolution_kernel": c["convolution_kernel"], "activation": c["activation"],
        "norm_epsilon": c["norm_epsilon"], "dropout": 0.0, "classes": c["classes"], "blank_id": c["blank_id"],
        "features": {"sample_rate": RATE, "window_size": N_FFT, "hop_size": HOP, "window": "hann", "center": True, "mels": MELS,
                     "min_frequency": 0, "max_frequency": FMAX, "mel_scale": "slaney", "mel_norm": "slaney", "log_floor": 1e-10,
                     "dynamic_range": 8.0, "normalize": [4.0, 0.25], "pad_seconds": None},
        "max_seconds": 30,
    }


def ctc_tokenizer(folder):
    """39 single-character tokens (Metaspace, no merges): token i is label i + 1, label 0 the blank."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    symbols = ["▁"] + list("abcdefghijklmnopqrstuvwxyz") + list("0123456789") + ["'", "-"]
    assert len(symbols) == CTC["classes"] - 1
    tok = Tokenizer(models.BPE(vocab={s: i for i, s in enumerate(symbols)}, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    tok.save(os.path.join(folder, "tokenizer.json"))


def write_ctc(out):
    import torchaudio.functional as F
    from safetensors.torch import save_file
    torch = torch_setup()
    folder = os.path.join(out, "ctc")
    if os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder)
    ref = os.path.join(folder, "reference")
    os.makedirs(ref)
    gen = torch.Generator().manual_seed(2029)
    frames = [96, 71, 123, 50]
    labels = [[5, 7, 7, 12, 3], [1, 2], [30, 9, 9, 9, 14, 2, 8], [4, 4, 4]]
    T = max(frames)
    batch = torch.zeros(len(frames), T, MELS)
    for i, t in enumerate(frames):
        batch[i, :t] = torch.randn(t, MELS, generator=gen)
    targets = torch.tensor([l for ls in labels for l in ls])
    target_lengths = torch.tensor([len(l) for l in labels])
    frames_t = torch.tensor(frames)
    save_npy(folder, "mels.npy", batch.numpy().astype(np.float32))

    def ctc(model):
        logp, steps = model(batch, frames_t)
        loss = torch.nn.functional.ctc_loss(logp.transpose(0, 1), targets, steps, target_lengths, blank=CTC["blank_id"], reduction="mean", zero_infinity=True)
        return loss, logp, steps

    results = {}
    for name, convolution, seed in [("model", False, 2029), ("model-conv", True, 2031)]:
        torch.manual_seed(seed)
        model = ctc_modules(torch, convolution)
        ctc_initialize(torch, model, seed)
        sub = os.path.join(folder, name)
        os.makedirs(sub)
        dump_json(os.path.join(sub, "config.json"), ctc_config(convolution))
        save_file({k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}, os.path.join(sub, "model.safetensors"))
        ctc_tokenizer(sub)
        index, _ = safetensors_index(os.path.join(sub, "model.safetensors"))
        dump_json(os.path.join(sub, "tensors.json"), index)
        loss, logp, steps = ctc(model)
        loss.backward()
        suffix = "" if name == "model" else "-conv"
        save_file({k: p.grad.detach().clone().contiguous() for k, p in model.named_parameters()}, os.path.join(ref, f"grads-step1{suffix}.safetensors"))
        results[name] = {"loss_step1": float(loss), "steps": steps.tolist()}
        if name == "model":
            save_npy(ref, "logp-step1.npy", logp.detach().numpy().astype(np.float32))
            alignments, scores = [], []
            with torch.no_grad():
                for i, ls in enumerate(labels):
                    path, score = F.forced_align(logp[i: i + 1, : int(steps[i])].detach(), torch.tensor([ls]), blank=CTC["blank_id"])
                    alignments.append(path[0].tolist())
                    scores.append(float(score[0].sum()))
            per = torch.nn.functional.ctc_loss(logp.transpose(0, 1), targets, steps, target_lengths, blank=0, reduction="none", zero_infinity=True)
            results[name]["per_utterance_step1"] = per.tolist()
            losses = [float(loss)]
            opt = torch.optim.SGD(model.parameters(), lr=0.1)
            for _ in range(3):
                opt.step()
                opt.zero_grad()
                loss, _, _ = ctc(model)
                losses.append(float(loss))
                loss.backward()
            results[name]["sgd_losses"] = losses
            save_file({k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}, os.path.join(ref, "weights-step3.safetensors"))
            dump_json(os.path.join(ref, "alignment.json"), {"alignments": alignments, "path_log_probs": scores, "blank": CTC["blank_id"],
                                                             "note": "torchaudio.functional.forced_align of each utterance's steps of logp-step1.npy (the model's step-1 log-probabilities)"})
    assert len(set(round(x, 3) for x in results["model"]["sgd_losses"])) == 4, "the SGD losses should differ step to step"
    dump_json(os.path.join(ref, "loss.json"), {"frames": frames, "labels": labels, "blank": CTC["blank_id"], "classes": CTC["classes"], "reduction": "mean",
                                               "zero_infinity": True, "sgd_lr": 0.1, "sgd_momentum": 0.0, "results": results})


# The IdrakAudio plug-in's "seq2seq" task in torch: the "ctc" encoder without its head (its normed states the memory) and
# a pre-norm text decoder: token embedding and learned positions, blocks of causal self-attention, cross-attention over
# each utterance's own encoder steps (the padding masked) and a feed-forward, a final LayerNorm and the head (the
# embedding's table, or its own weight). The decoder reads the start token and the labels and is taught each next label
# and then the end token; the loss is each utterance's mean cross-entropy, averaged over the batch.
SEQ = dict(decoder_layers=2, max_target_positions=16, start_token=39, end_token=39)


def seq2seq_modules(torch, decoder_dim, decoder_heads, tie):
    nn = torch.nn
    F = torch.nn.functional
    c = CTC
    act = lambda x: F.gelu(x, approximate="tanh")
    encoder = ctc_modules(torch, False)
    del encoder.head

    class Attention(nn.Module):
        def __init__(self, inputs):
            super().__init__()
            d = decoder_dim
            self.q_proj = nn.Linear(d, d)
            self.k_proj = nn.Linear(inputs, d)
            self.v_proj = nn.Linear(inputs, d)
            self.out_proj = nn.Linear(d, d)

        def forward(self, x, memory, mask=None, causal=False):
            n, length, d = x.shape
            h, dh = decoder_heads, decoder_dim // decoder_heads
            split = lambda t: t.view(n, t.shape[1], h, dh).transpose(1, 2)
            o = F.scaled_dot_product_attention(split(self.q_proj(x)), split(self.k_proj(memory)), split(self.v_proj(memory)), attn_mask=mask, is_causal=causal)
            return self.out_proj(o.transpose(1, 2).reshape(n, length, d))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            d, eps = decoder_dim, c["norm_epsilon"]
            self.norm1 = nn.LayerNorm(d, eps=eps)
            self.self_attn = Attention(d)
            self.norm2 = nn.LayerNorm(d, eps=eps)
            self.cross_attn = Attention(c["dim"])
            self.norm3 = nn.LayerNorm(d, eps=eps)
            self.linear1 = nn.Linear(d, c["ffn_dim"])
            self.linear2 = nn.Linear(c["ffn_dim"], d)

        def forward(self, x, memory, mask):
            h = self.norm1(x)
            x = x + self.self_attn(h, h, causal=True)
            x = x + self.cross_attn(self.norm2(x), memory, mask)
            return x + self.linear2(act(self.linear1(self.norm3(x))))

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            d = decoder_dim
            self.embed = nn.Embedding(c["classes"], d)
            self.positions = nn.Embedding(SEQ["max_target_positions"], d)
            self.layers = nn.ModuleList([Block() for _ in range(SEQ["decoder_layers"])])
            self.norm = nn.LayerNorm(d, eps=c["norm_epsilon"])
            if not tie:
                self.head = nn.Linear(d, c["classes"], bias=False)

        def forward(self, tokens, memory, steps):                           # tokens [N, L]; memory [N, T', dim]; steps [N]
            mask = (torch.arange(memory.shape[1])[None, :] < steps[:, None])[:, None, None, :]   # True: a step the row attends to
            x = self.embed(tokens) + self.positions.weight[: tokens.shape[1]]
            for layer in self.layers:
                x = layer(x, memory, mask)
            x = self.norm(x)
            return x @ self.embed.weight.T if tie else self.head(x)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            for name, module in encoder.named_children():
                self.add_module(name, module)
            self.decoder = Decoder()

        def forward(self, mel, frames, tokens):
            memory, steps = encoder.encode(mel, frames)
            return self.decoder(tokens, memory, steps), memory, steps

    return Model()


def seq2seq_config(decoder_dim, decoder_heads, tie):
    config = ctc_config(False)
    config["task"] = "seq2seq"
    del config["blank_id"]
    config.update({"decoder_layers": SEQ["decoder_layers"], "decoder_dim": decoder_dim, "decoder_heads": decoder_heads, "decoder_ffn_dim": CTC["ffn_dim"],
                   "max_target_positions": SEQ["max_target_positions"], "tie_embeddings": tie, "start_token": SEQ["start_token"], "end_token": SEQ["end_token"]})
    return config


def write_seq2seq(out):
    from safetensors.torch import save_file
    torch = torch_setup()
    F = torch.nn.functional
    folder = os.path.join(out, "seq2seq")
    if os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder)
    ref = os.path.join(folder, "reference")
    os.makedirs(ref)
    gen = torch.Generator().manual_seed(2029)
    frames = [96, 71, 123, 50]
    labels = [[5, 7, 7, 12, 3], [1, 2], [30, 9, 9, 9, 14, 2, 8], [4, 4, 4]]
    T = max(frames)
    batch = torch.zeros(len(frames), T, MELS)
    for i, t in enumerate(frames):
        batch[i, :t] = torch.randn(t, MELS, generator=gen)
    frames_t = torch.tensor(frames)
    start, end, length = SEQ["start_token"], SEQ["end_token"], max(len(l) for l in labels) + 1
    tokens = torch.full((len(labels), length), end)                          # the end token as padding after each row's own
    targets = torch.zeros(len(labels), length, dtype=torch.long)
    weights = torch.zeros(len(labels), length)
    for i, ls in enumerate(labels):
        tokens[i, 0] = start
        tokens[i, 1: len(ls) + 1] = torch.tensor(ls)
        targets[i, : len(ls)] = torch.tensor(ls)
        targets[i, len(ls)] = end
        weights[i, : len(ls) + 1] = 1
    save_npy(folder, "mels.npy", batch.numpy().astype(np.float32))

    def seq_loss(model):
        logits, memory, steps = model(batch, frames_t, tokens)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").view(targets.shape)
        return ((ce * weights).sum(1) / weights.sum(1)).mean(), memory, steps

    def greedy(model, memory, steps):
        """Each utterance alone (its own steps), from the start token, the likeliest token each step until the end token."""
        texts = []
        with torch.no_grad():
            for i in range(len(frames)):
                own = memory[i: i + 1, : int(steps[i])]
                ids = [start]
                while len(ids) < SEQ["max_target_positions"]:
                    nxt = int(model.decoder(torch.tensor([ids]), own, steps[i: i + 1].clamp(max=own.shape[1]))[0, -1].argmax())
                    if nxt == end:
                        break
                    ids.append(nxt)
                texts.append(ids[1:])
        return texts

    results = {}
    for name, decoder_dim, decoder_heads, tie, seed in [("model", CTC["dim"], CTC["heads"], True, 2033), ("model-untied", 24, 3, False, 2035)]:
        torch.manual_seed(seed)
        model = seq2seq_modules(torch, decoder_dim, decoder_heads, tie)
        ctc_initialize(torch, model, seed)
        sub = os.path.join(folder, name)
        os.makedirs(sub)
        dump_json(os.path.join(sub, "config.json"), seq2seq_config(decoder_dim, decoder_heads, tie))
        save_file({k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}, os.path.join(sub, "model.safetensors"))
        ctc_tokenizer(sub)
        index, _ = safetensors_index(os.path.join(sub, "model.safetensors"))
        dump_json(os.path.join(sub, "tensors.json"), index)
        loss, memory, steps = seq_loss(model)
        loss.backward()
        suffix = "" if name == "model" else "-untied"
        save_file({k: p.grad.detach().clone().contiguous() for k, p in model.named_parameters()}, os.path.join(ref, f"grads-step1{suffix}.safetensors"))
        results[name] = {"loss_step1": float(loss), "steps": steps.tolist(), "greedy_step1": greedy(model, memory.detach(), steps)}
        if name == "model":
            losses = [float(loss)]
            opt = torch.optim.SGD(model.parameters(), lr=0.1)
            for _ in range(3):
                opt.step()
                opt.zero_grad()
                loss, _, _ = seq_loss(model)
                losses.append(float(loss))
                loss.backward()
            results[name]["sgd_losses"] = losses
            save_file({k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}, os.path.join(ref, "weights-step3.safetensors"))
    assert len(set(round(x, 3) for x in results["model"]["sgd_losses"])) == 4, "the SGD losses should differ step to step"
    dump_json(os.path.join(ref, "loss.json"), {"frames": frames, "labels": labels, "start_token": start, "end_token": end, "classes": CTC["classes"],
                                               "reduction": "each utterance's mean over its labels and the end token, then the mean over the batch",
                                               "sgd_lr": 0.1, "sgd_momentum": 0.0, "results": results})


# The IdrakAudio plug-in's audio-language model in torch: a Llama text decoder (transformers' LlamaForCausalLM, its tensors
# under language_model.) with the "ctc" encoder without its head as the audio tower (audio_tower.) and a projector
# (audio_projector.: every STACK encoder steps joined, linear_1, GELU (tanh), linear_2); a sound's soft tokens replace the
# embeddings of the audio token's run between the begin and end tokens. Three conversations of a sound and its answer,
# rendered by the chat template and tokenized by the pretrain fixture's tokenizer; the loss is the mean cross-entropy
# over the three records' trained tokens (each answer and its <|im_end|>), as one pretraining step takes it.
AL_STACK = 4
AL_TEXT = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
               max_position_embeddings=128, rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=True, hidden_act="silu",
               attention_bias=False, mlp_bias=False)
AL_TEMPLATE = (
    "{%- for message in messages -%}\n"
    "{{- '<|im_start|>' + message['role'] + '\\n' -}}\n"
    "{%- if message['content'] is string -%}\n"
    "{{- message['content'] -}}\n"
    "{%- else -%}\n"
    "{%- for item in message['content'] -%}\n"
    "{%- if item['type'] == 'audio' -%}\n"
    "{{- '<image>' -}}\n"
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
AL_RECORDS = [
    {"messages": [{"role": "user", "content": [{"type": "audio", "path": "wav/pcm16.wav"}, {"type": "text", "text": "Transcribe."}]},
                  {"role": "assistant", "content": "the quick brown fox"}]},
    {"messages": [{"role": "user", "content": [{"type": "text", "text": "What is said? "}, {"type": "audio", "path": "wav/pcm8.wav"}]},
                  {"role": "assistant", "content": "hello there"}]},
    {"messages": [{"role": "system", "content": "You listen."},
                  {"role": "user", "content": [{"type": "audio", "path": "wav/list-chunk16.wav"}, {"type": "text", "text": "Again."}]},
                  {"role": "assistant", "content": "a short tone"}]},
]


def write_audio_language(out):
    import jinja2
    import torchaudio.functional as F
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from transformers import LlamaConfig, LlamaForCausalLM
    torch = torch_setup()
    nn = torch.nn
    folder = os.path.join(out, "audio-language")
    if os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder)
    os.makedirs(os.path.join(folder, "wav"))
    model_dir = os.path.join(folder, "model")
    os.makedirs(model_dir)
    data = os.path.normpath(os.path.join(out, ".."))
    tokenizer_path = os.path.join(data, "pretrain", "tokenizer", "tokenizer.json")
    tok = Tokenizer.from_file(tokenizer_path)
    ids = {t: tok.token_to_id(t) for t in ["<pad>", "<s>", "</s>", "<|im_start|>", "<|im_end|>", "<start_of_image>", "<image>", "<end_of_image>"]}
    sound, begin, end = ids["<image>"], ids["<start_of_image>"], ids["<end_of_image>"]

    # Each sound's features (the WAV fixture's decoded samples, 16 kHz mono) and soft tokens.
    mels, tokens_of = {}, {}
    for record in AL_RECORDS:
        for m in record["messages"]:
            for part in (m["content"] if isinstance(m["content"], list) else []):
                if part["type"] == "audio" and part["path"] not in mels:
                    name = os.path.splitext(os.path.basename(part["path"]))[0]
                    import shutil
                    shutil.copy(os.path.join(out, "wav", name + ".wav"), os.path.join(folder, part["path"]))
                    audio = torch.from_numpy(load_wav_npy(out, name)[0].astype(np.float32))
                    mel = log_mel(torch, F, audio)[0].T.contiguous()            # [frames, mels]
                    mels[part["path"]] = mel
                    save_npy(folder, f"features-{name}.npy", mel.numpy().astype(np.float32))

    torch.manual_seed(2041)
    text_config = LlamaConfig(vocab_size=tok.get_vocab_size(), pad_token_id=ids["<pad>"], bos_token_id=ids["<s>"],
                              eos_token_id=[ids["<|im_end|>"], ids["</s>"]], attn_implementation="eager", **AL_TEXT)

    class Projector(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_1 = nn.Linear(AL_STACK * CTC["dim"], AL_TEXT["hidden_size"])
            self.linear_2 = nn.Linear(AL_TEXT["hidden_size"], AL_TEXT["hidden_size"])

        def forward(self, states):                                          # [steps, dim] -> [tokens, text width]
            steps = states.shape[0]
            tokens = (steps + AL_STACK - 1) // AL_STACK
            padded = torch.cat([states, states.new_zeros(tokens * AL_STACK - steps, states.shape[1])])
            return self.linear_2(torch.nn.functional.gelu(self.linear_1(padded.reshape(tokens, AL_STACK * states.shape[1])), approximate="tanh"))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = LlamaForCausalLM(text_config)
            tower = ctc_modules(torch, False)
            del tower.head
            self.audio_tower = tower
            self.audio_projector = Projector()

        def features(self, path):
            mel = mels[path]
            states, steps = self.audio_tower.encode(mel[None], torch.tensor([mel.shape[0]]))
            return self.audio_projector(states[0, : int(steps[0])])

    def build():
        model = Model()
        ctc_initialize(torch, model, 2041)
        model.language_model.tie_weights()
        return model

    for path, mel in mels.items():
        with torch.no_grad():
            tokens_of[path] = int(build().features(path).shape[0])

    env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False)

    def render(messages, add_generation_prompt):
        return env.from_string(AL_TEMPLATE).render(messages=messages, add_generation_prompt=add_generation_prompt)

    def expand(text, paths):
        pieces = text.split("<image>")
        assert len(pieces) == len(paths) + 1
        result = pieces[0]
        for k, p in enumerate(paths):
            result += "<start_of_image>" + "<image>" * tokens_of[p] + "<end_of_image>" + pieces[k + 1]
        return result

    def encode(record):
        messages = record["messages"]
        paths = [part["path"] for m in messages if isinstance(m["content"], list) for part in m["content"] if part["type"] == "audio"]
        full = expand(render(messages, False), paths)
        prefix = expand(render(messages[:-1], True), paths)
        answer = messages[-1]["content"] + "<|im_end|>"
        assert full.startswith(prefix + answer), (full, prefix, answer)
        enc = tok.encode(full, add_special_tokens=False)
        trained = [len(prefix) <= s and e <= len(prefix) + len(answer) for (s, e) in enc.offsets]
        positions = [i for i, t in enumerate(enc.ids) if t == sound and (i == 0 or enc.ids[i - 1] != sound)]
        return {"input_ids": enc.ids, "trained": trained, "audios": paths, "blocks": [[q, tokens_of[p]] for q, p in zip(positions, paths)]}

    encoded = [encode(r) for r in AL_RECORDS]

    def loss_of(model):
        total, count = 0.0, 0
        for e in encoded:
            idt = torch.tensor([e["input_ids"]])
            embeds = model.language_model.model.embed_tokens(idt)
            pieces, last = [], 0
            for (start, n), path in zip(e["blocks"], e["audios"]):
                pieces += [embeds[:, last:start], model.features(path)[None]]
                last = start + n
            pieces.append(embeds[:, last:])
            logits = model.language_model(inputs_embeds=torch.cat(pieces, dim=1)).logits[0]
            targets = torch.tensor(e["input_ids"][1:])
            rows = torch.tensor([t for t in range(len(e["input_ids"]) - 1) if e["trained"][t + 1]])
            total = total + torch.nn.functional.cross_entropy(logits[rows], targets[rows], reduction="sum")
            count += len(rows)
        return total / count, count

    def named(model):
        return {n: p for n, p in model.named_parameters() if not n.startswith("language_model.lm_head")}

    model = build()
    save_file({n: p.detach().clone().contiguous() for n, p in sorted(named(model).items())}, os.path.join(model_dir, "model.safetensors"), metadata={"format": "pt"})
    audio_config = ctc_config(False)
    for key in ("architectures", "model_type", "classes", "blank_id"):
        audio_config.pop(key)
    audio_config["task"] = "audio-language"
    text = {k: v for k, v in text_config.to_dict().items() if k in (
        "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
        "max_position_embeddings", "rms_norm_eps", "rope_theta", "hidden_act", "attention_bias", "mlp_bias", "tie_word_embeddings",
        "bos_token_id", "eos_token_id", "pad_token_id")}
    text["model_type"] = "llama"
    text["rope_theta"] = AL_TEXT["rope_theta"]
    dump_json(os.path.join(model_dir, "config.json"), {
        "architectures": ["IdrakAudioForConditionalGeneration"], "model_type": "idrak_audio", "text_config": text, "audio_config": audio_config,
        "audio_projector": {"stack": AL_STACK, "type": "mlp", "text_dim": AL_TEXT["hidden_size"]}, "audio_token_index": sound,
        "boa_token_index": begin, "eoa_token_index": end, "audio_attention": "causal", "tie_word_embeddings": True, "torch_dtype": "float32"})
    import shutil
    shutil.copy(tokenizer_path, os.path.join(model_dir, "tokenizer.json"))
    dump_json(os.path.join(model_dir, "tokenizer_config.json"), {
        "add_bos_token": False, "bos_token": "<s>", "eos_token": "<|im_end|>", "pad_token": "<pad>", "chat_template": AL_TEMPLATE,
        "model_max_length": AL_TEXT["max_position_embeddings"], "tokenizer_class": "PreTrainedTokenizerFast"})
    dump_json(os.path.join(model_dir, "special_tokens_map.json"), {"bos_token": "<s>", "eos_token": "<|im_end|>", "pad_token": "<pad>"})
    dump_json(os.path.join(model_dir, "generation_config.json"), {"bos_token_id": ids["<s>"], "eos_token_id": [ids["<|im_end|>"], ids["</s>"]], "pad_token_id": ids["<pad>"]})
    with open(os.path.join(folder, "records.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for record in AL_RECORDS:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    from safetensors.torch import load_file
    steps = {}
    for kind, prefixes in [("joint", ["language_model.", "audio_tower.", "audio_projector."]), ("align", ["audio_tower.", "audio_projector."])]:
        trained_model = Model()
        state = load_file(os.path.join(model_dir, "model.safetensors"))
        trained_model.load_state_dict({**state, "language_model.lm_head.weight": state["language_model.model.embed_tokens.weight"]}, strict=True)
        trained_model.language_model.tie_weights()
        params = named(trained_model)
        for n, p in params.items():
            p.requires_grad_(any(n.startswith(t) for t in prefixes))
        opt = torch.optim.SGD([p for p in params.values() if p.requires_grad], lr=0.05, momentum=0.0)
        losses = []
        sub = os.path.join(folder, kind)
        os.makedirs(sub)
        for step in range(1, 4):
            opt.zero_grad(set_to_none=True)
            loss, count = loss_of(trained_model)
            loss.backward()
            losses.append(float(loss.detach()))
            if step == 1:
                grads = {n: p.grad.detach().clone().contiguous() for n, p in params.items() if p.requires_grad}
                save_file(dict(sorted(grads.items())), os.path.join(sub, "grads-step1.safetensors"), metadata={"format": "pt"})
            opt.step()
        save_file({n: p.detach().clone().contiguous() for n, p in sorted(params.items())}, os.path.join(sub, "weights-step3.safetensors"), metadata={"format": "pt"})
        assert len(set(round(x, 4) for x in losses)) == 3, losses
        steps[kind] = {"losses": losses, "trained_tokens": count, "trained": prefixes}
    dump_json(os.path.join(folder, "steps.json"), dict(steps, loss="mean cross entropy over the three records' trained tokens (each answer and <|im_end|>)", sgd_lr=0.05))
    dump_json(os.path.join(folder, "tuning.json"), {"records": encoded, "tokens_per_audio": tokens_of, "audio_token": sound, "begin": begin, "end": end, "stack": AL_STACK})


# The IdrakAudio plug-in's "tts" task in torch: characters through a token embedding and sinusoidal positions, pre-norm
# blocks over each text's own labels (key padding), a final LayerNorm (the text's hidden states); a duration predictor
# (the padding zeroed before each of two Conv2d (1, k) over the labels, each with a ReLU and a LayerNorm, then a linear
# to log(duration + offset)); the hidden states projected to the decoder's width when it differs, repeated for each
# label's frames (length regulation, the given durations in training), sinusoidal positions, pre-norm blocks over each
# utterance's own frames, a final LayerNorm and a linear to the mels. The loss: each utterance's mean |log-mel - target|
# over its frames and mels, plus the weight times its mean squared error of the log durations over its labels, each
# averaged over the batch.
TTS = dict(classes=12, dim=16, heads=2, ffn_dim=32, layers=2, decoder_layers=2, decoder_dim=24, decoder_heads=2, decoder_ffn_dim=40,
           duration_filters=20, duration_kernel=3, duration_offset=1.0, mels=20, norm_epsilon=1e-5, duration_weight=1.0)


def sinusoids(torch, t, d):
    pos = torch.arange(t, dtype=torch.float64)[:, None]
    i = torch.arange(d)[None, :]
    angle = pos / torch.pow(torch.tensor(10000.0, dtype=torch.float64), (2 * (i // 2)).to(torch.float64) / d)
    return torch.where(i % 2 == 0, torch.sin(angle), torch.cos(angle)).to(torch.float32)


def tts_modules(torch):
    nn = torch.nn
    F = torch.nn.functional
    c = TTS
    act = lambda x: F.gelu(x, approximate="tanh")

    class Block(nn.Module):
        def __init__(self, d, heads, ffn):
            super().__init__()
            eps = c["norm_epsilon"]
            self.norm1 = nn.LayerNorm(d, eps=eps)
            self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
            self.norm2 = nn.LayerNorm(d, eps=eps)
            self.linear1 = nn.Linear(d, ffn)
            self.linear2 = nn.Linear(ffn, d)

        def forward(self, x, pad):
            h = self.norm1(x)
            x = x + self.self_attn(h, h, h, key_padding_mask=pad, need_weights=False)[0]
            return x + self.linear2(act(self.linear1(self.norm2(x))))

    class Text(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(c["classes"], c["dim"])
            self.layers = nn.ModuleList([Block(c["dim"], c["heads"], c["ffn_dim"]) for _ in range(c["layers"])])
            self.norm = nn.LayerNorm(c["dim"], eps=c["norm_epsilon"])

        def forward(self, labels, lengths):                                  # labels [N, L] (0 as padding); lengths [N]
            x = self.embed(labels) + sinusoids(torch, labels.shape[1], c["dim"])
            pad = torch.arange(labels.shape[1])[None, :] >= lengths[:, None]
            for layer in self.layers:
                x = layer(x, pad)
            return self.norm(x)

    class Duration(nn.Module):
        def __init__(self):
            super().__init__()
            d, f, k = c["dim"], c["duration_filters"], c["duration_kernel"]
            self.conv1 = nn.Conv2d(d, f, (1, k), padding=(0, k // 2))
            self.norm1 = nn.LayerNorm(f, eps=c["norm_epsilon"])
            self.conv2 = nn.Conv2d(f, f, (1, k), padding=(0, k // 2))
            self.norm2 = nn.LayerNorm(f, eps=c["norm_epsilon"])
            self.linear = nn.Linear(f, 1)

        def forward(self, hidden, lengths):                                  # the log durations [N, L]
            keep = (torch.arange(hidden.shape[1])[None, :] < lengths[:, None]).to(hidden.dtype)[..., None]
            conv = lambda layer, x: torch.relu(layer((x * keep).transpose(1, 2)[:, :, None, :]))[:, :, 0, :].transpose(1, 2)
            h = self.norm1(conv(self.conv1, hidden))
            h = self.norm2(conv(self.conv2, h))
            return self.linear(h)[..., 0]

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            d, dd = c["dim"], c["decoder_dim"]
            if d != dd:
                self.proj = nn.Linear(d, dd)
            self.layers = nn.ModuleList([Block(dd, c["decoder_heads"], c["decoder_ffn_dim"]) for _ in range(c["decoder_layers"])])
            self.norm = nn.LayerNorm(dd, eps=c["norm_epsilon"])
            self.mel = nn.Linear(dd, c["mels"])

        def forward(self, hidden, lengths, durations):                      # durations: each utterance's frames per label
            h = self.proj(hidden) if hasattr(self, "proj") else hidden
            frames = [int(sum(d)) for d in durations]
            t = max(frames)
            rows = []
            for b, d in enumerate(durations):
                own = torch.repeat_interleave(h[b, : int(lengths[b])], torch.tensor(d), dim=0)
                rows.append(torch.cat([own, torch.zeros(t - own.shape[0], h.shape[2])]))
            x = torch.stack(rows) + sinusoids(torch, t, h.shape[2])
            pad = torch.arange(t)[None, :] >= torch.tensor(frames)[:, None]
            for layer in self.layers:
                x = layer(x, pad)
            return self.mel(self.norm(x))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.text = Text()
            self.duration = Duration()
            self.decoder = Decoder()

        def forward(self, labels, lengths, durations):
            hidden = self.text(labels, lengths)
            return self.decoder(hidden, lengths, durations), self.duration(hidden, lengths)

    return Model()


def tts_config():
    c = TTS
    return {
        "architectures": ["IdrakAudio"], "model_type": "idrak-audio", "task": "tts",
        "mels": c["mels"], "dim": c["dim"], "layers": c["layers"], "heads": c["heads"], "ffn_dim": c["ffn_dim"], "positions": "sinusoidal",
        "convolution_module": False, "convolution_kernel": 5, "activation": "gelu_tanh", "norm_epsilon": c["norm_epsilon"], "dropout": 0.0,
        "classes": c["classes"], "decoder_layers": c["decoder_layers"], "decoder_dim": c["decoder_dim"], "decoder_heads": c["decoder_heads"],
        "decoder_ffn_dim": c["decoder_ffn_dim"], "duration_filters": c["duration_filters"], "duration_kernel": c["duration_kernel"],
        "duration_offset": c["duration_offset"],
        "features": {"sample_rate": RATE, "window_size": N_FFT, "hop_size": HOP, "window": "hann", "center": True, "mels": c["mels"],
                     "min_frequency": 0, "max_frequency": FMAX, "mel_scale": "slaney", "mel_norm": "slaney", "log_floor": 1e-10,
                     "dynamic_range": 8.0, "normalize": [4.0, 0.25], "pad_seconds": None},
    }


def tts_tokenizer(folder):
    """12 single-character tokens (Metaspace, no merges): token i is label i."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    symbols = ["▁"] + list("abcdefghijk")
    assert len(symbols) == TTS["classes"]
    tok = Tokenizer(models.BPE(vocab={s: i for i, s in enumerate(symbols)}, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    tok.save(os.path.join(folder, "tokenizer.json"))


def write_tts(out):
    from safetensors.torch import save_file
    torch = torch_setup()
    folder = os.path.join(out, "tts")
    if os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder)
    ref = os.path.join(folder, "reference")
    os.makedirs(ref)
    c = TTS
    labels = [[3, 5, 7, 1, 9], [2, 11], [4, 4, 6, 8, 10, 0, 1], [7, 3, 2]]
    durations = [[2, 3, 0, 4, 1], [5, 2], [1, 1, 3, 2, 0, 2, 4], [3, 3, 2]]
    frames = [sum(d) for d in durations]
    lengths = torch.tensor([len(l) for l in labels])
    L, T, M = max(len(l) for l in labels), max(frames), c["mels"]
    ids = torch.zeros(len(labels), L, dtype=torch.long)
    log_target = torch.zeros(len(labels), L)
    for i, ls in enumerate(labels):
        ids[i, : len(ls)] = torch.tensor(ls)
        log_target[i, : len(ls)] = torch.log(torch.tensor(durations[i], dtype=torch.float32) + c["duration_offset"])
    gen = torch.Generator().manual_seed(2037)
    target = torch.zeros(len(labels), T, M)
    for i, t in enumerate(frames):
        target[i, :t] = torch.randn(t, M, generator=gen)
    save_npy(folder, "log-mels.npy", target.numpy().astype(np.float32))
    keep_frames = (torch.arange(T)[None, :] < torch.tensor(frames)[:, None]).to(torch.float32)
    keep_labels = (torch.arange(L)[None, :] < lengths[:, None]).to(torch.float32)

    def tts_loss(model):
        mel, logd = model(ids, lengths, durations)
        l1 = ((mel - target).abs() * keep_frames[..., None]).sum((1, 2)) / (torch.tensor(frames, dtype=torch.float32) * M)
        mse = ((logd - log_target) ** 2 * keep_labels).sum(1) / lengths.to(torch.float32)
        return l1.mean() + c["duration_weight"] * mse.mean(), l1.mean(), mse.mean(), mel, logd

    torch.manual_seed(2037)
    model = tts_modules(torch)
    ctc_initialize(torch, model, 2037)
    sub = os.path.join(folder, "model")
    os.makedirs(sub)
    dump_json(os.path.join(sub, "config.json"), tts_config())
    save_file({k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}, os.path.join(sub, "model.safetensors"))
    tts_tokenizer(sub)
    index, _ = safetensors_index(os.path.join(sub, "model.safetensors"))
    dump_json(os.path.join(sub, "tensors.json"), index)
    loss, l1, mse, mel, logd = tts_loss(model)
    loss.backward()
    save_file({k: p.grad.detach().clone().contiguous() for k, p in model.named_parameters()}, os.path.join(ref, "grads-step1.safetensors"))
    save_npy(ref, "log-mel-step1.npy", mel.detach().numpy().astype(np.float32))
    results = {"loss_step1": loss.item(), "mel_l1_step1": l1.item(), "duration_mse_step1": mse.item(), "log_durations_step1": logd.detach().tolist()}
    losses = [float(loss)]
    opt = torch.optim.SGD(model.parameters(), lr=0.03)
    for _ in range(3):
        opt.step()
        opt.zero_grad()
        loss, _, _, _, _ = tts_loss(model)
        losses.append(float(loss))
        loss.backward()
    results["sgd_losses"] = losses
    save_file({k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}, os.path.join(ref, "weights-step3.safetensors"))
    assert len(set(round(x, 3) for x in losses)) == 4, "the SGD losses should differ step to step"
    dump_json(os.path.join(ref, "loss.json"), {"labels": labels, "durations": durations, "frames": frames, "mels": M,
                                               "duration_offset": c["duration_offset"], "duration_weight": c["duration_weight"],
                                               "reduction": "each utterance's mean |log-mel error| over its frames and mels, plus the weight times its mean squared log-duration error over its labels, each averaged over the batch",
                                               "sgd_lr": 0.03, "sgd_momentum": 0.0, "results": results})


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


# ------------------------------------------------------------------ vits-speakers: recorded speakers (plan 21, S9 route A)

def write_vits_speakers(out):
    """Two tiny multi-speaker VitsModels (VCTK's shape: num_speakers and speaker_embedding_size, the speaker embedding
    conditioning the duration predictor, the flow and the decoder), one with the stochastic duration predictor and one
    with the deterministic one, and transformers' waveform of the same ids as each speaker (noise 0).
    speakers.json names the speakers in index order, as an app gives VCTK's (the hub's config holds only their count)."""
    from transformers import VitsConfig, VitsModel
    torch = torch_setup()
    symbols = list("_ abcdefghijklmnopqrstuvwx")[:20]
    vocab = {s: i for i, s in enumerate(symbols)}
    names = ["p225", "p226", "p227"]
    for folder_name, stochastic, seed in (("tiny-vits-speakers", True, 2041), ("tiny-vits-speakers-det", False, 2042)):
        folder = os.path.join(out, folder_name)
        os.makedirs(folder, exist_ok=True)
        dump_json(os.path.join(folder, "vocab.json"), vocab)
        dump_json(os.path.join(folder, "speakers.json"), names)
        config = VitsConfig(vocab_size=len(symbols), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, ffn_dim=64,
                            flow_size=32, prior_encoder_num_flows=1, prior_encoder_num_wavenet_layers=2, posterior_encoder_num_wavenet_layers=2,
                            upsample_rates=[4, 4], upsample_kernel_sizes=[8, 8], upsample_initial_channel=32, resblock_kernel_sizes=[3],
                            resblock_dilation_sizes=[[1, 3]], spectrogram_bins=65, hop_length=16, speaker_embedding_size=8, num_speakers=len(names),
                            noise_scale=0.0, noise_scale_duration=0.0, wavenet_dilation_rate=1, wavenet_kernel_size=3,
                            use_stochastic_duration_prediction=stochastic, duration_predictor_num_flows=2, duration_predictor_filter_channels=32,
                            depth_separable_num_layers=1)
        model = VitsModel(config)
        randomize(model, torch, seed)
        gen = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():                                                 # speakers far enough apart that their sounds differ
            for name, p in model.named_parameters():
                if name.startswith("embed_speaker") or (".cond" in name and p.dim() > 1):
                    p.copy_(torch.randn(p.shape, generator=gen) * (1.0 if name.startswith("embed_speaker") else 0.3))
        model.eval()
        model.save_pretrained(folder, safe_serialization=True)
        ids = torch.tensor([[vocab[c] for c in "abc ab"]])
        ref = os.path.join(folder, "reference")
        os.makedirs(ref, exist_ok=True)
        lengths = {}
        with torch.no_grad():
            for speaker in range(len(names)):
                output = model(ids, speaker_id=torch.tensor([speaker]))
                tag = str(speaker)
                save_npy(ref, f"waveform-{tag}.npy", output.waveform[0].numpy().astype(np.float32))
                lengths[tag] = int(output.waveform.shape[1])
        dump_json(os.path.join(ref, "input.json"), {"text": "abc ab", "ids": ids[0].tolist(), "noise_scale": 0.0, "noise_scale_duration": 0.0,
                                                    "speakers": names, "waveform_samples": lengths})
        index, _ = safetensors_index(os.path.join(folder, "model.safetensors"))
        dump_json(os.path.join(folder, "tensors.json"), index)


# ------------------------------------------------------------------ vits-train: fine-tuning a VITS checkpoint (plan 21, S9 route C)

# The settings of the tiny run: the decoder's segment in frames, the loss weights (VITS's c_mel 45, c_kl 1, the duration,
# adversarial and feature-matching terms; feature matching is VITS's feature_loss, twice the sum of the layers' mean
# errors), the mel bands of the mel loss, the SGD rates of the generator and the discriminators.
VITS_TRAIN = {"segment_frames": 8, "mel_weight": 45.0, "kl_weight": 1.0, "duration_weight": 1.0, "adversarial_weight": 1.0,
              "feature_matching_weight": 2.0, "mels": 16, "generator_lr": 0.002, "discriminator_lr": 0.002}

# The discriminators' description (the plug-in's discriminator.json): VITS's multi-period discriminator (a scale
# discriminator and one per period) at tiny widths, with a second scale (HiFi-GAN's average-pooled multi-scale input).
VITS_DISCRIMINATOR = {"periods": [2, 3], "period_channels": [4, 8, 8], "period_kernel": 5, "period_stride": 3, "post_kernel": 3,
                      "scales": 2, "scale_channels": [4, 8, 8], "scale_kernels": [5, 9, 3], "scale_strides": [1, 4, 1], "scale_groups": [1, 2, 1],
                      "pool_kernel": 4, "pool_stride": 2, "pool_padding": 2, "leaky_relu_slope": 0.1}


def vits_train_config():
    from transformers import VitsConfig
    return VitsConfig(vocab_size=20, hidden_size=16, num_hidden_layers=2, num_attention_heads=2, window_size=2, ffn_dim=32, ffn_kernel_size=3,
                      flow_size=8, prior_encoder_num_flows=2, prior_encoder_num_wavenet_layers=2, posterior_encoder_num_wavenet_layers=2,
                      upsample_rates=[4, 4], upsample_kernel_sizes=[8, 8], upsample_initial_channel=16, resblock_kernel_sizes=[3, 5],
                      resblock_dilation_sizes=[[1, 3], [1, 2]], spectrogram_bins=65, speaker_embedding_size=0, num_speakers=1,
                      wavenet_kernel_size=3, wavenet_dilation_rate=2, duration_predictor_num_flows=2, duration_predictor_flow_bins=6,
                      duration_predictor_tail_bound=5.0, depth_separable_num_layers=2, noise_scale=0.667, noise_scale_duration=0.8,
                      sampling_rate=16000, hidden_dropout=0.0, attention_dropout=0.0, activation_dropout=0.0, wavenet_dropout=0.0,
                      duration_predictor_dropout=0.0, layerdrop=0.0)


def vits_initialize(torch, module, seed):
    """Weights at 1/sqrt(fan-in) (a weight norm's directions too, its gains near 1), drawn biases, norms and flows, so
    every gradient is of a size a check against its tensor's largest can tell apart from a wrong one."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if name.endswith("original0"):
                p.copy_(0.8 + 0.4 * torch.rand(p.shape, generator=gen))
            elif "norm" in name and name.endswith("weight") and p.dim() == 1:
                p.copy_(1 + 0.1 * torch.randn(p.shape, generator=gen))
            elif p.dim() == 1 or name.endswith("translate") or name.endswith("log_scale"):
                p.copy_(0.1 * torch.randn(p.shape, generator=gen))
            else:
                p.copy_(torch.randn(p.shape, generator=gen) / math.sqrt(p.numel() / p.shape[0]))


def vits_discriminators(torch, d):
    """VITS's MultiPeriodDiscriminator (DiscriminatorS and DiscriminatorP, weight-normed), the scale discriminators
    reading the sound average-pooled once more each (HiFi-GAN's MultiScaleDiscriminator)."""
    import torch.nn as nn
    import torch.nn.functional as F
    wn = nn.utils.parametrizations.weight_norm
    slope = d["leaky_relu_slope"]

    class Scale(nn.Module):
        def __init__(self):
            super().__init__()
            convs, c_in = [], 1
            for c, k, s, g in zip(d["scale_channels"], d["scale_kernels"], d["scale_strides"], d["scale_groups"]):
                convs.append(wn(nn.Conv1d(c_in, c, k, s, groups=g, padding=(k - 1) // 2)))
                c_in = c
            self.convs = nn.ModuleList(convs)
            self.conv_post = wn(nn.Conv1d(c_in, 1, d["post_kernel"], 1, padding=(d["post_kernel"] - 1) // 2))

        def forward(self, x):
            fmap = []
            for layer in self.convs:
                x = F.leaky_relu(layer(x), slope)
                fmap.append(x)
            x = self.conv_post(x)
            fmap.append(x)
            return torch.flatten(x, 1, -1), fmap

    class Period(nn.Module):
        def __init__(self, period):
            super().__init__()
            self.period = period
            convs, c_in, k = [], 1, d["period_kernel"]
            for j, c in enumerate(d["period_channels"]):
                s = d["period_stride"] if j < len(d["period_channels"]) - 1 else 1
                convs.append(wn(nn.Conv2d(c_in, c, (k, 1), (s, 1), padding=((k - 1) // 2, 0))))
                c_in = c
            self.convs = nn.ModuleList(convs)
            self.conv_post = wn(nn.Conv2d(c_in, 1, (d["post_kernel"], 1), 1, padding=((d["post_kernel"] - 1) // 2, 0)))

        def forward(self, x):
            fmap = []
            b, c, t = x.shape
            if t % self.period != 0:
                n_pad = self.period - (t % self.period)
                x = F.pad(x, (0, n_pad), "reflect")
                t = t + n_pad
            x = x.view(b, c, t // self.period, self.period)
            for layer in self.convs:
                x = F.leaky_relu(layer(x), slope)
                fmap.append(x)
            x = self.conv_post(x)
            fmap.append(x)
            return torch.flatten(x, 1, -1), fmap

    class Discriminators(nn.Module):
        def __init__(self):
            super().__init__()
            self.discriminators = nn.ModuleList([Scale() for _ in range(d["scales"])] + [Period(p) for p in d["periods"]])
            self.pool = nn.AvgPool1d(d["pool_kernel"], d["pool_stride"], padding=d["pool_padding"])

        def forward(self, y):
            outputs, fmaps, x = [], [], y
            for i, disc in enumerate(self.discriminators):
                if i < d["scales"]:
                    if i > 0:
                        x = self.pool(x)
                    out, fmap = disc(x)
                else:
                    out, fmap = disc(y)
                outputs.append(out)
                fmaps.append(fmap)
            return outputs, fmaps

    return Discriminators()


def maximum_path(value):
    """VITS's monotonic alignment search (monotonic_align/core.pyx) on one utterance's [frames, tokens] scores, in
    float32 as the Cython code adds them: the path [frames, tokens] of ones."""
    value = np.array(value, dtype=np.float32)
    t_y, t_x = value.shape
    path = np.zeros((t_y, t_x), dtype=np.int32)
    neg = np.float32(-1e9)
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            v_cur = neg if x == y else value[y - 1, x]
            if x == 0:
                v_prev = np.float32(0.0) if y == 0 else neg
            else:
                v_prev = value[y - 1, x - 1]
            value[y, x] = np.float32(value[y, x] + max(v_prev, v_cur))
    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
            index -= 1
    return path


def vits_spectrogram(torch, samples, n_fft, hop):
    """VITS's spectrogram_torch: reflect-padded by (n_fft - hop) / 2, frames not centred, a periodic Hann window,
    sqrt(power + 1e-6): [bins, frames]."""
    import torch.nn.functional as F
    pad = (n_fft - hop) // 2
    y = F.pad(samples[None, None], (pad, pad), mode="reflect")[0]
    spec = torch.stft(y, n_fft, hop_length=hop, win_length=n_fft, window=torch.hann_window(n_fft), center=False, normalized=False,
                      onesided=True, return_complex=True)
    return torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-6)[0]


def vits_sdp_nll(torch, sdp, hidden, durations, noise):
    """VITS's stochastic duration predictor forward (training) on one utterance (no padding), with the posterior's noise
    given: nll + log q, on transformers' VitsStochasticDurationPredictor's weights. In VITS's order: no flip after the
    element-wise affine (its flows are [affine, conv flow, flip, conv flow, flip, ...]), the order whose inverse is
    the reverse transformers and VITS speak with; transformers' own forward flips after the affine too, which trains
    flows that its reverse does not undo (a model fine-tuned that way speaks with durations it was not trained on)."""
    import torch.nn.functional as F
    mask = torch.ones(1, 1, hidden.shape[2])
    inputs = sdp.conv_pre(torch.detach(hidden))
    inputs = sdp.conv_dds(inputs, mask)
    inputs = sdp.conv_proj(inputs) * mask
    h = sdp.post_conv_pre(durations)
    h = sdp.post_conv_dds(h, mask)
    h = sdp.post_conv_proj(h) * mask
    latents_posterior = noise[None]
    logdet_q = 0
    for i, flow in enumerate(sdp.post_flows):
        latents_posterior, logdet = flow(latents_posterior, mask, global_conditioning=inputs + h)
        if i > 0:
            latents_posterior = torch.flip(latents_posterior, [1])
        logdet_q = logdet_q + logdet
    first, second = torch.split(latents_posterior, [1, 1], dim=1)
    logdet_q = logdet_q + torch.sum(F.logsigmoid(first) + F.logsigmoid(-first), [1, 2])
    logq = torch.sum(-0.5 * (math.log(2 * math.pi) + noise[None] ** 2), [1, 2]) - logdet_q
    first = durations - torch.sigmoid(first)
    first = torch.log(torch.clamp_min(first, 1e-5))
    logdet = torch.sum(-first, [1, 2])
    latents = torch.cat([first, second], dim=1)
    for i, flow in enumerate(sdp.flows):
        latents, ld = flow(latents, mask, global_conditioning=inputs)
        if i > 0:
            latents = torch.flip(latents, [1])
        logdet = logdet + ld
    nll = torch.sum(0.5 * (math.log(2 * math.pi) + latents ** 2), [1, 2]) - logdet
    return (nll + logq)[0]


def write_vits_train(out):
    """A tiny random VitsModel (its decoder weight-normed, as a training checkpoint holds it) and tiny discriminators,
    and VITS's training step on two utterances (with the noise and the decoder's segments drawn here and saved): the
    generator's and the discriminators' losses and every gradient of step 1, the monotonic alignment search's scores and
    paths, three SGD steps' losses and the weights after them. The utterances go through one at a time (no padding,
    so no mask), their terms added up as VITS's batch losses are: the KL over the batch's frames, the duration over its
    tokens, the mel, adversarial and feature-matching terms averaged over the utterances (their segments are of one length)."""
    from transformers import VitsModel
    from safetensors.torch import save_file
    import torch.nn.functional as F
    import torchaudio
    torch = torch_setup()
    folder = os.path.join(out, "vits-train")
    if os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder)
    sub = os.path.join(folder, "model")
    ref = os.path.join(folder, "reference")
    os.makedirs(sub)
    os.makedirs(ref)
    s = VITS_TRAIN
    config = vits_train_config()
    model = VitsModel(config)
    model.decoder.apply_weight_norm()
    vits_initialize(torch, model, 2041)
    disc = vits_discriminators(torch, VITS_DISCRIMINATOR)
    vits_initialize(torch, disc, 2042)
    model.train()
    disc.train()
    symbols = list("_ abcdefghijklmnopqr")
    dump_json(os.path.join(sub, "vocab.json"), {c: i for i, c in enumerate(symbols)})
    config.architectures = ["VitsModel"]
    config.to_json_file(os.path.join(sub, "config.json"))
    save_file({k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}, os.path.join(sub, "model.safetensors"))
    save_file({k: v.detach().clone().contiguous() for k, v in disc.state_dict().items()}, os.path.join(sub, "discriminator.safetensors"))
    dump_json(os.path.join(sub, "discriminator.json"), VITS_DISCRIMINATOR)
    index, _ = safetensors_index(os.path.join(sub, "model.safetensors"))
    dump_json(os.path.join(sub, "tensors.json"), index)

    hop = int(np.prod(config.upsample_rates))
    n_fft = (config.spectrogram_bins - 1) * 2
    seg = s["segment_frames"]
    mel_basis = torchaudio.functional.melscale_fbanks(config.spectrogram_bins, 0.0, config.sampling_rate / 2, s["mels"], config.sampling_rate,
                                                      norm="slaney", mel_scale="slaney").T            # [mels, bins]
    gen = torch.Generator().manual_seed(2043)
    texts = ["abc ab", "hello cab"]
    frames = [20, 28]
    ids = [torch.tensor([symbols.index(c) for c in t]) for t in texts]
    audio, specs = [], []
    for u, f in enumerate(frames):
        n = torch.arange(f * hop, dtype=torch.float32) / config.sampling_rate
        x = sum(0.2 * torch.sin(2 * math.pi * (300 + 210 * k + 50 * u) * n + k) * (1 + 0.5 * torch.sin(2 * math.pi * (3 + k) * n)) for k in range(3))
        x = (x + 0.01 * torch.randn(x.shape, generator=gen)).to(torch.float32)
        audio.append(x)
        specs.append(vits_spectrogram(torch, x, n_fft, hop))
        save_npy(folder, f"audio-{u}.npy", x.numpy().astype(np.float32))
        save_npy(ref, f"spectrogram-{u}.npy", specs[u].numpy().astype(np.float32))            # [bins, frames]
    total_frames, total_tokens, count = sum(frames), sum(len(i) for i in ids), len(ids)

    def mel_of_spec(spec):
        return torch.log(torch.clamp(mel_basis @ spec, min=1e-5))

    def mel_of_samples(y):
        return mel_of_spec(vits_spectrogram(torch, y, n_fft, hop))

    def draw(step):
        g = torch.Generator().manual_seed(3000 + step)
        out_ = []
        for u in range(count):
            post = torch.randn(config.flow_size, frames[u], generator=g)
            dur = torch.randn(2, len(ids[u]), generator=g)
            start = int(torch.randint(0, frames[u] - seg + 1, (1,), generator=g))
            out_.append((post, dur, start))
        return out_

    def generator_pass(u, draws, record=None):
        post_noise, dur_noise, start = draws[u]
        enc = model.text_encoder(ids[u][None], padding_mask=torch.ones(1, len(ids[u]), 1))
        hidden = enc.last_hidden_state.transpose(1, 2)
        m_p, logs_p = enc.prior_means.transpose(1, 2), enc.prior_log_variances.transpose(1, 2)            # [1, F, T]
        ymask = torch.ones(1, 1, frames[u])
        pe = model.posterior_encoder
        h = pe.conv_pre(specs[u][None]) * ymask
        h = pe.wavenet(h, ymask, None)
        stats = pe.conv_proj(h) * ymask
        m_q, logs_q = torch.split(stats, config.flow_size, dim=1)
        z = m_q + post_noise[None] * torch.exp(logs_q)
        z_p = model.flow(z, ymask, None, reverse=False)
        with torch.no_grad():
            s_p_sq_r = torch.exp(-2 * logs_p)
            nc1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
            nc2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)
            nc3 = torch.matmul(z_p.transpose(1, 2), m_p * s_p_sq_r)
            nc4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True)
            neg_cent = (nc1 + nc2 + nc3 + nc4)[0]                                                        # [frames, T]
            path = maximum_path(neg_cent.numpy())
            attn = torch.from_numpy(path.astype(np.float32))
        if record is not None:
            record["neg_cent"].append(neg_cent.numpy().astype(np.float32))
            record["durations"].append([int(v) for v in path.sum(0)])
        w = attn.sum(0)[None, None]                                                                      # [1, 1, T]
        m_pf = torch.matmul(attn[None], m_p.transpose(1, 2)).transpose(1, 2)
        logs_pf = torch.matmul(attn[None], logs_p.transpose(1, 2)).transpose(1, 2)
        kl = torch.sum(logs_pf - logs_q - 0.5 + 0.5 * (z_p - m_pf) ** 2 * torch.exp(-2 * logs_pf))
        nll = vits_sdp_nll(torch, model.duration_predictor, hidden, w, dur_noise)
        y_hat = model.decoder(z[:, :, start:start + seg])[0, 0]                                           # [seg * hop]
        y = audio[u][start * hop:(start + seg) * hop]
        mel = torch.mean(torch.abs(mel_of_spec(specs[u])[:, start:start + seg] - mel_of_samples(y_hat)))
        return {"kl": kl, "nll": nll, "mel": mel, "y": y, "y_hat": y_hat}

    def discriminator_loss(passes):
        total = 0
        for p in passes:
            real, _ = disc(p["y"][None, None])
            fake, _ = disc(p["y_hat"].detach()[None, None])
            total = total + sum(torch.mean((1 - r) ** 2) + torch.mean(f ** 2) for r, f in zip(real, fake)) / count
        return total

    def generator_loss(passes):
        terms = {"mel": 0, "kl": 0, "duration": 0, "adversarial": 0, "feature_matching": 0}
        for p in passes:
            with torch.no_grad():
                _, fmap_r = disc(p["y"][None, None])
            fake, fmap_g = disc(p["y_hat"][None, None])
            terms["mel"] = terms["mel"] + s["mel_weight"] * p["mel"] / count
            terms["kl"] = terms["kl"] + s["kl_weight"] * p["kl"] / total_frames
            terms["duration"] = terms["duration"] + s["duration_weight"] * p["nll"] / total_tokens
            terms["adversarial"] = terms["adversarial"] + s["adversarial_weight"] * sum(torch.mean((1 - f) ** 2) for f in fake) / count
            fm = sum(torch.mean(torch.abs(r.detach() - g)) for dr, dg in zip(fmap_r, fmap_g) for r, g in zip(dr, dg))
            terms["feature_matching"] = terms["feature_matching"] + s["feature_matching_weight"] * fm / count
        return sum(terms.values()), terms

    opt_g = torch.optim.SGD(model.parameters(), lr=s["generator_lr"])
    opt_d = torch.optim.SGD(disc.parameters(), lr=s["discriminator_lr"])
    record = {"neg_cent": [], "durations": []}
    draws_out, starts, steps = {}, [], []
    for step in range(1, 4):
        draws = draw(step)
        for u, (post, dur, start) in enumerate(draws):
            draws_out[f"step{step}.posterior.{u}"] = post.contiguous()
            draws_out[f"step{step}.duration.{u}"] = dur.contiguous()
        starts.append([d[2] for d in draws])
        passes = [generator_pass(u, draws, record if step == 1 else None) for u in range(count)]
        loss_d = discriminator_loss(passes)
        opt_d.zero_grad()
        loss_d.backward()
        if step == 1:
            save_file({k: p.grad.detach().clone().contiguous() for k, p in disc.named_parameters()}, os.path.join(ref, "grads-discriminator-step1.safetensors"))
        opt_d.step()
        loss_g, terms = generator_loss(passes)
        opt_g.zero_grad()
        opt_d.zero_grad()
        loss_g.backward()
        if step == 1:
            grads = {k: p.grad.detach().clone().contiguous() for k, p in model.named_parameters() if p.grad is not None}
            missing = [k for k, p in model.named_parameters() if p.grad is None]
            save_file(grads, os.path.join(ref, "grads-generator-step1.safetensors"))
            if step == 1:
                for u in range(count):
                    save_npy(ref, f"neg-cent-{u}.npy", record["neg_cent"][u])
        opt_g.step()
        steps.append({"discriminator": float(loss_d), "generator": float(loss_g), "terms": {k: float(v) for k, v in terms.items()}})
    save_file({k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}, os.path.join(ref, "weights-step3.safetensors"))
    save_file({k: v.detach().clone().contiguous() for k, v in disc.state_dict().items()}, os.path.join(ref, "discriminator-step3.safetensors"))
    save_file(draws_out, os.path.join(ref, "draws.safetensors"))
    assert len(set(round(x["generator"], 4) for x in steps)) == 3, "the generator's losses should differ step to step"
    dump_json(os.path.join(ref, "steps.json"), {
        **s, "texts": texts, "ids": [i.tolist() for i in ids], "frames": frames, "hop": hop, "n_fft": n_fft, "starts": starts,
        "durations": record["durations"], "steps": steps, "untrained": missing,
        "order": "each step: the generator's pass of every utterance, the discriminators' loss (the sound made detached) and their SGD step, "
                 "then the generator's loss with the stepped discriminators and its SGD step",
    })


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
    for name in ["torch", "torchaudio", "transformers", "tokenizers", "safetensors", "soundfile"]:
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
    previous = os.path.join(out, "manifest.json")
    if os.path.exists(previous):                                              # a run of some sections keeps the others'
        with open(previous, encoding="utf-8") as f:
            sections = sorted(set(sections) | set(json.load(f).get("sections", [])), key=SECTIONS.index)
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
    if "flac" in sections:
        write_flacs(args.out)
    if "mp3" in sections:
        write_mp3s(args.out)
    if "resample" in sections:
        write_resample(args.out)
    if "features" in sections:
        extractor = write_features(args.out)
    if "whisper" in sections:
        if extractor is None:
            from transformers import WhisperFeatureExtractor
            extractor = WhisperFeatureExtractor(feature_size=MELS, sampling_rate=RATE, hop_length=HOP, chunk_length=CHUNK_SECONDS, n_fft=N_FFT)
        write_whisper(args.out, extractor)
    if "whisper-train" in sections:
        write_whisper_train(args.out)
    if "ctc" in sections:
        write_ctc(args.out)
    if "seq2seq" in sections:
        write_seq2seq(args.out)
    if "audio-language" in sections:
        write_audio_language(args.out)
    if "tts" in sections:
        write_tts(args.out)
    if "vits" in sections:
        write_vits(args.out)
    if "vits-speakers" in sections:
        write_vits_speakers(args.out)
    if "voices" in sections:
        write_voices(args.out)
    if "vits-train" in sections:
        write_vits_train(args.out)
    if "transcripts" in sections:
        write_transcripts(args.out)
    manifest(args.out, sections)
    print("wrote", args.out, "sections", ",".join(sections))


if __name__ == "__main__":
    main()
