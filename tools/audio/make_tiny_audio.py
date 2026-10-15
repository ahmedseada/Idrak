"""The reference for Idrak.Audio (plan 21, phase S0): generated WAV files and their samples, torchaudio's resampling and
features, a tiny random Whisper-style model and transformers' outputs for it, a tiny CTC recognizer's loss, gradients
and alignments, a tiny VITS synthesizer's waveform, and transcript records.

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

SECTIONS = ["wav", "flac", "mp3", "resample", "features", "whisper", "ctc", "vits", "voices", "transcripts"]
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
    weight_g and weight_v as stored, as tools/audio/convert_openvoice.py writes it), with OpenVoice's config.json; and its outputs: the linear spectrogram of wav/pcm16, the speaker embeddings of the
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


# The IdrakAudio plug-in's "ctc" task in torch (plan 21, S6): the same layers, names and order as
# samples/IdrakAudio/Idrak.IdrakAudio builds, written here from torch's own modules so the references do not come from
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
            return torch.log_softmax(self.head(self.norm(x)), dim=-1), steps

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
    if "ctc" in sections:
        write_ctc(args.out)
    if "vits" in sections:
        write_vits(args.out)
    if "voices" in sections:
        write_voices(args.out)
    if "transcripts" in sections:
        write_transcripts(args.out)
    manifest(args.out, sections)
    print("wrote", args.out, "sections", ",".join(sections))


if __name__ == "__main__":
    main()
