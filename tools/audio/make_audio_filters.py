"""The reference for Idrak.Audio's filters (plan 31, phase P1): short generated signals and what scipy makes of them
through each filter, read by the C# tests "audio processing: filters:".

    python -m venv audio && audio/bin/pip install numpy scipy
    audio/bin/python tools/audio/make_audio_filters.py tests/Idrak.Tests/data/audio/filters

Nothing is downloaded: the signals are generated here (seeded), 0.3 s at 16 kHz (and one at 8 kHz), float32 [channels,
samples]. Each case is a pipeline written as Idrak writes it (`dehum=50 harmonics=4, presence=4`), parsed here with the
same defaults, and computed in float64 from the float32 input, each step's output rounded to float32 as Idrak's are:

- the linear filters as second-order sections through scipy.signal.sosfilt, started settled on the first sample's level
  (zi = sosfilt_zi(sos) * x[0]: the state a constant input of that level leaves), as Idrak's filters start;
  highpass/lowpass: scipy.signal.butter(order, cutoff, fs=rate, output="sos"); bandpass: the high-pass at `low` then the
  low-pass at `high`, both of `order`; notch and dehum: scipy.signal.iirnotch (dehum: at the mains frequency and its
  multiples below half the rate); eq, presence, lowshelf, highshelf: the RBJ cookbook's coefficients computed here; dc:
  [1, -1] / [1, -pole]; preemphasis: [1, -a] / [1];
- gain: x * 10^(dB / 20); mono: the channels' mean;
- resample: torchaudio.functional.resample's sinc_interp_hann kernel (lowpass_filter_width, rolloff) written in numpy
  in float64 (no torch needed), the sound padded as torchaudio pads it, cut to ceil(length * new / orig).

Writes signal-<name>.npy for each signal and <case>.npy for each case's output (float32 [channels, samples]) and
cases.json (each case's pipeline, input, rate, channels and length; the signals; numpy's and scipy's versions). Reruns
write the same bytes.
"""
import json
import math
import os
import sys

import numpy as np
import scipy
from scipy import signal

RATE = 16000
SECONDS = 0.3


def signals():
    n = int(RATE * SECONDS)
    t = np.arange(n) / RATE
    rng = np.random.default_rng(31)
    tones = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 3000 * t) + 0.1 * np.sin(2 * np.pi * 6500 * t)

    # Speech-like: a voiced sound (a pitch gliding round 120 Hz, harmonics up to 7 kHz shaped by three formants) in bursts
    # of 80 ms with 40 ms pauses (raised-cosine edges), a little breath noise.
    f0 = 120 + 20 * np.sin(2 * np.pi * 3 * t)
    phase = 2 * np.pi * np.cumsum(f0) / RATE
    voiced = np.zeros(n)
    for k in range(1, 60):
        f = k * 120
        if f > 7000:
            break
        formants = sum(1 / (1 + ((f - c) / w) ** 2) for c, w in ((700, 150), (1200, 200), (2600, 300)))
        voiced += (0.2 + formants) / k * np.sin(k * phase)
    period, on, edge = int(0.12 * RATE), int(0.08 * RATE), int(0.01 * RATE)
    envelope = np.zeros(n)
    for start in range(0, n, period):
        for i in range(start, min(n, start + on)):
            j = i - start
            ramp = min(j, on - 1 - j)
            envelope[i] = 1.0 if ramp >= edge else 0.5 - 0.5 * math.cos(math.pi * ramp / edge)
    speech = voiced * envelope + 0.01 * rng.standard_normal(n)
    speech *= 0.5 / np.max(np.abs(speech))

    noise = rng.uniform(-0.5, 0.5, n)
    hum = sum(a * np.sin(2 * np.pi * f * t + 0.3 * f / 50) for f, a in ((50, 0.3), (100, 0.1), (150, 0.08), (200, 0.05), (250, 0.03)))
    hum = hum + 0.2 * np.sin(2 * np.pi * 1000 * t)
    dc = 0.25 + 0.3 * np.sin(2 * np.pi * 440 * t) + 0.02 * rng.uniform(-1, 1, n)
    stereo = np.stack([speech, 0.5 * noise + hum])

    t8 = np.arange(int(8000 * SECONDS)) / 8000
    tones8k = 0.3 * np.sin(2 * np.pi * 440 * t8) + 0.2 * np.sin(2 * np.pi * 2500 * t8)
    out = {"tones": (tones, RATE), "speech": (speech, RATE), "noise": (noise, RATE), "hum": (hum, RATE), "dc": (dc, RATE),
           "stereo": (stereo, RATE), "tones8k": (tones8k, 8000)}
    return {k: (np.atleast_2d(v).astype(np.float32), r) for k, (v, r) in out.items()}


# The cases: (name, pipeline, input signal).
CASES = [
    ("gain-cut", "gain=-6", "tones"),
    ("gain-boost", "gain=4.5", "speech"),
    ("dc", "dc", "dc"),
    ("dc-pole", "dc pole=0.98", "dc"),
    ("highpass", "highpass", "dc"),
    ("highpass-order4", "highpass=200 order=4", "speech"),
    ("highpass-order3", "highpass=150 order=3", "hum"),
    ("highpass-stereo", "highpass=100 order=4", "stereo"),
    ("lowpass", "lowpass=1000", "tones"),
    ("lowpass-order5", "lowpass=2500 order=5", "noise"),
    ("lowpass-order8", "lowpass=3000 order=8", "speech"),
    ("bandpass", "bandpass", "noise"),
    ("bandpass-order4", "bandpass low=500 high=2000 order=4", "speech"),
    ("notch", "notch=440", "tones"),
    ("notch-wide", "notch=1000 q=10", "hum"),
    ("dehum", "dehum", "hum"),
    ("dehum-60", "dehum=60 harmonics=6 q=20", "speech"),
    ("dehum-stereo", "dehum=50 harmonics=12 q=15", "stereo"),
    ("eq-boost", "eq=1000 gain=6 q=2", "noise"),
    ("eq-cut", "eq=3000 gain=-12 q=0.7", "speech"),
    ("lowshelf", "lowshelf=200 gain=6", "speech"),
    ("lowshelf-slope", "lowshelf=300 gain=-9 slope=0.5", "hum"),
    ("lowshelf-q", "lowshelf=150 gain=4 q=0.9", "dc"),
    ("highshelf", "highshelf=4000 gain=4", "noise"),
    ("highshelf-q", "highshelf=2500 gain=-6 q=1.2", "speech"),
    ("presence", "presence", "speech"),
    ("presence-strong", "presence=6 frequency=4000 q=1", "speech"),
    ("presence-soft", "presence=-4", "noise"),
    ("preemphasis", "preemphasis", "speech"),
    ("preemphasis-0.9", "preemphasis=0.9", "dc"),
    ("mono", "mono", "stereo"),
    ("resample-8k", "resample=8000", "speech"),
    ("resample-44k", "resample=44100", "tones"),
    ("resample-22k", "resample=22050 width=8 rolloff=0.95", "noise"),
    ("resample-up", "resample=16000", "tones8k"),
    ("resample-stereo", "resample=12000", "stereo"),
    ("chain", "dc, highpass=80 order=4, dehum=50 harmonics=4, presence=4, gain=-3", "stereo"),
    ("chain-mono", "mono, lowpass=3400 order=4, resample=8000, preemphasis", "stereo"),
]


def parse(text):
    """Idrak's pipeline text: steps by commas, each NAME[=VALUE] then KEY=VALUE options."""
    steps = []
    for item in text.split(","):
        words = item.split()
        name, _, value = words[0].partition("=")
        options = dict(w.split("=", 1) for w in words[1:])
        steps.append((name, value or None, options))
    return steps


def settled(sos, x):
    """sosfilt started settled on x[0] (Idrak's start)."""
    y, _ = signal.sosfilt(sos, x, zi=signal.sosfilt_zi(sos) * x[0])
    return y


def row(b, a):
    return np.array([b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]])


def peaking(f, db, q, rate):
    w0 = 2 * np.pi * f / rate
    alpha, a = np.sin(w0) / (2 * q), 10 ** (db / 40)
    return row([1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a], [1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a])


def shelf(high, f, db, options, rate):
    w0 = 2 * np.pi * f / rate
    a, cos = 10 ** (db / 40), np.cos(w0)
    if "q" in options:
        alpha = np.sin(w0) / (2 * float(options["q"]))
    else:
        slope = float(options.get("slope", 1))
        alpha = np.sin(w0) / 2 * np.sqrt((a + 1 / a) * (1 / slope - 1) + 2)
    r = 2 * np.sqrt(a) * alpha
    if high:
        b = [a * ((a + 1) + (a - 1) * cos + r), -2 * a * ((a - 1) + (a + 1) * cos), a * ((a + 1) + (a - 1) * cos - r)]
        d = [(a + 1) - (a - 1) * cos + r, 2 * ((a - 1) - (a + 1) * cos), (a + 1) - (a - 1) * cos - r]
    else:
        b = [a * ((a + 1) - (a - 1) * cos + r), 2 * a * ((a - 1) - (a + 1) * cos), a * ((a + 1) - (a - 1) * cos - r)]
        d = [(a + 1) + (a - 1) * cos + r, -2 * ((a - 1) + (a + 1) * cos), (a + 1) + (a - 1) * cos - r]
    return row(b, d)


def notch(f, q, rate):
    b, a = signal.iirnotch(f, q, fs=rate)
    return row(b, a)


def butter(order, f, kind, rate):
    return signal.butter(order, f, btype=kind, fs=rate, output="sos")


def ta_resample(x, orig, new, width, rolloff):
    """torchaudio.functional.resample (sinc_interp_hann) in numpy, float64."""
    g = math.gcd(orig, new)
    orig, new = orig // g, new // g
    base = min(orig, new) * rolloff
    half = int(math.ceil(width * orig / base))
    idx = np.arange(-half, half + orig, dtype=np.float64)[None, :] / orig
    t = (np.arange(0, -new, -1, dtype=np.float64)[:, None] / new + idx) * base
    t = np.clip(t, -width, width)
    window = np.cos(t * np.pi / width / 2) ** 2
    t = t * np.pi
    with np.errstate(invalid="ignore", divide="ignore"):
        sinc = np.where(t == 0, 1.0, np.sin(t) / t)
    kernel = sinc * window * base / orig                                   # [new, taps]
    length = x.shape[-1]
    padded = np.pad(x, (half, half + orig))
    taps = kernel.shape[1]
    frames = (padded.shape[0] - taps) // orig + 1
    windows = np.lib.stride_tricks.sliding_window_view(padded, taps)[::orig][:frames]   # [frames, taps]
    out = (windows @ kernel.T).reshape(-1)                                # frame by frame, the phases interleaved
    return out[: int(math.ceil(new * length / orig))]


def run_step(name, value, options, x, rate):
    """One step over x (float64 [channels, n]); returns (y, rate)."""
    num = lambda key, default: float(options.get(key, default))  # noqa: E731
    channel = lambda sos: np.stack([settled(sos, c) for c in x])  # noqa: E731
    if name == "gain":
        return x * 10 ** (float(value) / 20), rate
    if name == "dc":
        return channel(np.array([[1, -1, 0, 1, -num("pole", 0.995), 0]])), rate
    if name == "highpass":
        return channel(butter(int(num("order", 2)), float(value or 60), "highpass", rate)), rate
    if name == "lowpass":
        return channel(butter(int(num("order", 2)), float(value), "lowpass", rate)), rate
    if name == "bandpass":
        order = int(num("order", 2))
        return channel(np.vstack([butter(order, num("low", 300), "highpass", rate), butter(order, num("high", 3400), "lowpass", rate)])), rate
    if name == "notch":
        return channel(notch(float(value), num("q", 30), rate)[None]), rate
    if name == "dehum":
        mains, q = float(value or 50), num("q", 30)
        rows = [notch(k * mains, q, rate) for k in range(1, int(num("harmonics", 4)) + 1) if k * mains < rate / 2]
        return channel(np.array(rows)), rate
    if name == "eq":
        return channel(peaking(float(value), num("gain", 0), num("q", 1), rate)[None]), rate
    if name in ("lowshelf", "highshelf"):
        return channel(shelf(name == "highshelf", float(value), num("gain", 0), options, rate)[None]), rate
    if name == "presence":
        return channel(peaking(num("frequency", 3000), float(value or 3), num("q", 0.7), rate)[None]), rate
    if name == "preemphasis":
        return channel(np.array([[1, -float(value or 0.97), 0, 1, 0, 0]])), rate
    if name == "mono":
        return x.mean(axis=0, keepdims=True), rate
    if name == "resample":
        new = int(value)
        return np.stack([ta_resample(c, rate, new, int(num("width", 6)), num("rolloff", 0.99)) for c in x]), new
    raise ValueError(f"no reference for {name}")


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join("tests", "Idrak.Tests", "data", "audio", "filters")
    os.makedirs(folder, exist_ok=True)
    made = signals()
    inputs = {}
    for name, (x, rate) in made.items():
        np.save(os.path.join(folder, f"signal-{name}.npy"), np.ascontiguousarray(x))
        inputs[name] = {"rate": rate, "channels": x.shape[0], "length": x.shape[1]}
    cases = []
    for name, pipeline, source in CASES:
        x, rate = made[source]
        for step in parse(pipeline):
            y, rate = run_step(*step, x.astype(np.float64), rate)
            x = y.astype(np.float32)
        np.save(os.path.join(folder, f"{name}.npy"), np.ascontiguousarray(x))
        cases.append({"name": name, "pipeline": pipeline, "input": source, "rate": rate, "channels": x.shape[0], "length": x.shape[1]})
    with open(os.path.join(folder, "cases.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"cases": cases, "inputs": inputs, "versions": {"numpy": np.__version__, "scipy": scipy.__version__}}, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
