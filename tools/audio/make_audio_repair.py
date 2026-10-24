"""The reference for Idrak.Audio's noise and repair processors (plan 31, phase P3): denoise-live, spectral-gate,
declick, declip, deess and dereverb, each written here from its equations (the ones the C# comments in
src/Idrak.Audio/Processing/Repair.cs state), frame by frame or sample by sample, in float64 with the samples kept in
float32 where the C# keeps them so.

    python -m venv audio && audio/bin/pip install numpy scipy noisereduce
    audio/bin/python tools/audio/make_audio_repair.py tests/Idrak.Tests/data/audio/repair

Nothing is downloaded: the signals are generated here from fixed seeds. Writes, into the folder:
- the inputs, float32 [channels, samples] .npy: speechwhite16k and speechpink16k (mono, 16 kHz, 3 s: voiced bursts in
  white or pink noise, the pauses at noise only), clicks16k (mono, 16 kHz, 1 s: two tones with clicks of 1 to 6 samples
  added) and its clean version, clipped16k (mono, 16 kHz, 1 s: a tone pair clipped at 0.5) and its unclipped version,
  sibilant16k (mono, 16 kHz, 2 s: voiced bursts with loud noise bursts over 4 kHz), sibilant48k (stereo, 48 kHz, 0.5 s),
  reverb16k (mono, 16 kHz, 3 s: voiced bursts convolved with an exponentially decaying noise response, T60 0.6 s);
- for every case, its output as float32 .npy (a live stage's output is longer by its latency: silence first, as the
  library's live stages give it);
- repair.json: each case (its pipeline text, input and output files, the latency), each effect measured on the
  outputs (with the margins asserted here), noisereduce's stationary gate's effect on the same input (a cross-check of
  spectral-gate), the versions and every file's SHA-256.
Every effect is asserted here with its margin; the script fails if one is not met.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import sys

import numpy as np

# ------------------------------------------------------------------ shared definitions (as the C# stages)

SILENT_POWER = 1e-30   # a mean square under it is silence: an envelope snaps to its target (P2's convention)
GAIN_SNAP = 1e-12      # a gain this close to its target is the target (P2's convention)
QUIET_SIGMA = 1e-7     # the least residual scale a repair stage uses (-140 dBFS): digital silence is not all clicks


def coef(ms, rate):
    return 0.0 if ms <= 0 else math.exp(-1000.0 / (ms * rate))


def follow(value, target, a, snap):
    diff = (value - target) * a
    return target + diff if abs(diff) >= snap else target


def db_gain(db):
    return 10.0 ** (db / 20.0)


def window_size(seconds, rate):
    """The largest power of two of samples not over `seconds` (at least 16), as denoise takes its window."""
    size = 16
    while size * 2 <= seconds * rate:
        size *= 2
    return size


def hann(size):
    n = np.arange(size, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / size)


# ------------------------------------------------------------------ the streaming STFT (denoise-live, dereverb)

def stream_spectral(x, size, make_frame):
    """A live spectral stage over x ([channels, n] float32): per channel, frame k holds the samples (k+1)H - N to
    (k+1)H (H = N / 4, zeros before the start and after the end), windowed by a periodic Hann; its spectrum (numpy's
    rfft) goes through the stage's frame function, back (irfft), windowed again and overlap-added; each sample is the sum
    divided by the windows' squares over it. The output lags by N - 1 samples: N - 1 zeros, then the n samples."""
    channels, n = x.shape
    hop = size // 4
    w = hann(size)
    norm = np.array([sum(w[i + j * hop] ** 2 for j in range(4)) for i in range(hop)])
    last = (n - 1 + size - hop) // hop
    out = np.zeros((channels, n + size - 1), np.float32)
    for c in range(channels):
        xp = np.concatenate([np.zeros(size - hop), x[c].astype(np.float64), np.zeros(size + hop)])
        acc = np.zeros(len(xp) + size)
        frame = make_frame()
        for k in range(last + 1):
            spectrum = np.fft.rfft(xp[k * hop:k * hop + size] * w)
            inside = (k + 1) * hop - size >= 0 and (k + 1) * hop <= n
            spectrum = frame(k, spectrum, inside)
            acc[k * hop:k * hop + size] += np.fft.irfft(spectrum, size) * w
        y = acc[size - hop:size - hop + n] / np.resize(norm, n)
        out[c, size - 1:] = y.astype(np.float32)
    return out


def denoise_live(x, rate, strength=1.5, floor=-25, window=0.032, track=1.5, smooth=50, bias=1.6, frames=2):
    """A running noise floor per frequency: S, the magnitude smoothed over the frames (one pole, `smooth` ms at the frame
    rate; the first frame inside the recording starts it), its least value over the last D = round(track fs / H) frames
    inside the recording (a minimum statistic), times `bias` (1.6: the mean magnitude of white noise over that minimum
    at the default window and smoothing), is the noise. A frame's gain at a frequency is
    (|X| - strength · noise) / |X| within [floor, 1], averaged with the gains of the `frames` - 1 frames before it; frames
    before the first inside frame pass as they are, frames that reach past the end (the flush) use the last noise."""
    size = window_size(window, rate)
    hop = size // 4
    a = 0.0 if smooth <= 0 else math.exp(-hop / (smooth / 1000.0 * rate))
    depth = max(1, int(round(track * rate / hop)))
    floor_gain = db_gain(floor)

    def make():
        st = {"S": None, "hist": [], "noise": None, "gains": []}

        def frame(k, spectrum, inside):
            m = np.abs(spectrum)
            if inside:
                st["S"] = m.copy() if st["S"] is None else a * st["S"] + (1 - a) * m
                st["hist"].append(st["S"].copy())
                st["hist"] = st["hist"][-depth:]
                st["noise"] = bias * np.min(np.stack(st["hist"]), axis=0)
            if st["noise"] is None:
                g = np.ones_like(m)
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    g = np.where(m > 0, np.clip((m - strength * st["noise"]) / m, floor_gain, 1.0), floor_gain)
            st["gains"].append(g)
            st["gains"] = st["gains"][-frames:]
            smoothed = np.clip(np.mean(np.stack(st["gains"]), axis=0), floor_gain, 1.0)
            return spectrum * smoothed
        return frame

    return stream_spectral(x, size, make), size - 1


def dereverb(x, rate, strength=1.0, t60=0.5, delay=50, floor=-15, window=0.032, smooth=20):
    """Late reverberation taken off (Lebart, Boucher and Denbigh, 2001). P, the power spectrum smoothed over the frames
    (one pole, `smooth` ms; from 0 at the start); the late reverberation's power is P of D = max(1, round(delay fs / H))
    frames before, decayed by exp(-2 δ D H / fs) with δ = 3 ln 10 / T60; the gain is sqrt(1 - strength · late / P), no
    lower than `floor` dB (1 where P is 0)."""
    size = window_size(window, rate)
    hop = size // 4
    a = 0.0 if smooth <= 0 else math.exp(-hop / (smooth / 1000.0 * rate))
    d = max(1, int(round(delay / 1000.0 * rate / hop)))
    decay = math.exp(-2 * (3 * math.log(10) / t60) * d * hop / rate)
    floor_gain = db_gain(floor)

    def make():
        st = {"P": None, "hist": []}

        def frame(k, spectrum, inside):
            p = np.abs(spectrum) ** 2
            st["P"] = (1 - a) * p if st["P"] is None else a * st["P"] + (1 - a) * p
            st["hist"].append(st["P"].copy())
            late = decay * st["hist"][-1 - d] if len(st["hist"]) > d else np.zeros_like(p)
            st["hist"] = st["hist"][-(d + 1):]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(st["P"] > 0, late / st["P"], 0.0)
            g = np.maximum(np.sqrt(np.maximum(0.0, 1.0 - strength * ratio)), floor_gain)
            return spectrum * g
        return frame

    return stream_spectral(x, size, make), size - 1


# ------------------------------------------------------------------ the whole-recording STFT (spectral-gate)

def stft_torch(x, size):
    """torch.stft as the library's Stft.ComplexOnHost: centred (reflect padding of N / 2), periodic Hann, hop N / 4;
    the spectra rounded to float32 as the library stores them. [frames, bins] complex64."""
    hop = size // 4
    pad = size // 2
    xp = np.concatenate([x[1:pad + 1][::-1], x, x[-pad - 1:-1][::-1]]).astype(np.float64)
    w = hann(size).astype(np.float32).astype(np.float64)
    frames = (len(xp) - size) // hop + 1
    spec = np.stack([np.fft.rfft(xp[f * hop:f * hop + size] * w) for f in range(frames)])
    return spec.astype(np.complex64)


def istft_torch(spec, size, length):
    """torch.istft as the library's Stft.Inverse: each frame back, windowed and rounded to float32, overlap-added in
    float64 and divided by the windows' squares, the centre padding trimmed."""
    hop = size // 4
    w = hann(size).astype(np.float32).astype(np.float64)
    frames = spec.shape[0]
    full = size + hop * (frames - 1)
    total = np.zeros(full)
    env = np.zeros(full)
    for f in range(frames):
        y = (np.fft.irfft(spec[f].astype(np.complex128), size) * w).astype(np.float32)
        total[f * hop:f * hop + size] += y
        env[f * hop:f * hop + size] += w * w
    start = size // 2
    out = np.zeros(length, np.float32)
    kept = min(length, full - 2 * start)
    out[:kept] = (total[start:start + kept] / env[start:start + kept]).astype(np.float32)
    return out


def triangle(half):
    k = np.array([(half + 1 - abs(j)) / (half + 1) for j in range(-half, half + 1)])
    return k


def smooth_axis(m, half, axis):
    """The mask smoothed along one axis by a triangle of half-width `half` (noisereduce's shape), divided by the weights
    that fall inside (the edges keep their level)."""
    if half == 0:
        return m
    k = triangle(half)
    m = np.moveaxis(m, axis, -1)
    n = m.shape[-1]
    out = np.zeros_like(m)
    for i in range(n):
        lo, hi = max(0, i - half), min(n - 1, i + half)
        weights = k[lo - i + half:hi - i + half + 1]
        out[..., i] = (m[..., lo:hi + 1] * weights).sum(axis=-1) / weights.sum()
    return np.moveaxis(out, -1, axis)


def spectral_gate(x, rate, threshold=1.5, reduce=1.0, quantile=0.5, smooth=50, spread=40, window=0.032):
    """noisereduce's stationary gate: the noise profile, each frequency's mean and standard deviation in dB
    (10 log10 max(power, 1e-20)) over the quietest `quantile` share of the frames (by their energy; ties by order); a
    bin is signal when its power is over mean + threshold · deviation; the 0/1 mask smoothed by triangles of half-width
    floor(smooth / hop) frames and floor(spread / (fs / N)) bins (the weights inside the edges); the gain
    reduce · mask + 1 - reduce."""
    size = window_size(window, rate)
    hop = size // 4
    out = np.zeros_like(x)
    for c in range(x.shape[0]):
        spec = stft_torch(x[c], size)
        re = spec.real.astype(np.float64)
        im = spec.imag.astype(np.float64)
        p = re * re + im * im
        energy = p.sum(axis=1)
        order = np.argsort(energy, kind="stable")
        quiet = min(max(int(round(quantile * len(energy))), 1), len(energy))
        db = 10 * np.log10(np.maximum(p[order[:quiet]], 1e-20))
        thr = 10 ** ((db.mean(axis=0) + threshold * db.std(axis=0)) / 10)
        mask = (p > thr).astype(np.float64)
        mask = smooth_axis(mask, int(math.floor(smooth / 1000.0 * rate / hop)), 0)
        mask = smooth_axis(mask, int(math.floor(spread / (rate / size))), 1)
        g = (reduce * mask + (1 - reduce)).astype(np.float32)
        out[c] = istft_torch(spec * g, size, x.shape[1])
    return out, 0


# ------------------------------------------------------------------ the repair frames (declick, declip)

def levinson(r, order):
    """The predictor a[1..p] (x[n] ~ sum a_k x[n-k]) from the autocorrelation r[0..p] (Levinson-Durbin); zeros when r[0]
    is 0."""
    a = np.zeros(order + 1)
    if r[0] <= 0:
        return a[1:]
    err = r[0]
    for i in range(1, order + 1):
        acc = r[i]
        for j in range(1, i):
            acc -= a[j] * r[i - j]
        k = acc / err
        prev = a.copy()
        a[i] = k
        for j in range(1, i):
            a[j] = prev[j] - k * prev[i - j]
        err *= 1 - k * k
        if err <= 0:
            break
    return a[1:]


def autocorrelation(seg, order):
    return np.array([np.dot(seg[j:], seg[:len(seg) - j]) for j in range(order + 1)])


def lsar(x, s, e, a):
    """The samples s..e of x replaced by the least-squares AR interpolation (Janssen, Veldhuis and Vries, 1986): those
    that minimise the prediction error summed over n = s..e + p, given the p samples on each side."""
    p = len(a)
    b = np.concatenate([[1.0], -a])
    m = e - s + 1
    rb = np.array([np.dot(b[:p + 1 - j], b[j:]) for j in range(p + 1)])
    mat = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if abs(i - j) <= p:
                mat[i, j] = rb[abs(i - j)]
    known = np.zeros(m + p)
    for t in range(s, e + p + 1):
        acc = 0.0
        for k in range(p + 1):
            if not (s <= t - k <= e):
                acc += b[k] * float(x[t - k])
        known[t - s] = acc
    rhs = np.array([-sum(b[k] * known[i + k] for k in range(p + 1)) for i in range(m)])
    return np.linalg.solve(mat, rhs)


def residual(x, i, a):
    """The forward residual: x[i] - sum a_k x[i - k]."""
    acc = float(x[i])
    for k in range(1, len(a) + 1):
        acc -= a[k - 1] * float(x[i - k])
    return acc


def backward(x, i, a):
    """The backward residual: x[i] - sum a_k x[i + k] (the autocorrelation method's predictor read backwards)."""
    acc = float(x[i])
    for k in range(1, len(a) + 1):
        acc -= a[k - 1] * float(x[i + k])
    return acc


def repair_frames(x, rate, frame_ms, order, look, repair):
    """The frame walk declick and declip share, per channel: frames of F = round(frame fs / 1000) samples; frame t is
    repaired once the p samples before it and the look-ahead A after it are in, so the output lags by F + A - 1 samples.
    The channel is walked with p zeros before it and F + A zeros after (the flush), in float32."""
    channels, n = x.shape
    f = max(1, int(round(frame_ms * rate / 1000.0)))
    latency = f + look - 1
    out = np.zeros((channels, n + latency), np.float32)
    for c in range(channels):
        buf = np.concatenate([np.zeros(order, np.float32), x[c], np.zeros(f + look, np.float32)])
        state = {}
        for t in range((n + f - 1) // f):
            start = order + t * f
            repair(buf, start, f, order, start + f + look, order + n, state)
        out[c, latency:] = buf[order:order + n]
    return out, latency


def frame_predictor(buf, start, f, order):
    """The order-p predictor of the frame and the p samples before it, by the autocorrelation method."""
    seg = buf[start - order:start + f].astype(np.float64)
    return levinson(autocorrelation(seg, order), order)


def declick(x, rate, threshold=10.0, length=2.0, order=16, frame=20.0):
    """Clicks found by the residuals of the frame's order-p predictor (frame_predictor): sigma is the forward residual's
    median magnitude over the frame / 0.6745 (at least 1e-7); a sample is a click's when both its forward and its
    backward residual are over threshold · sigma (the forward one spreads p samples after a click, the backward one p
    before it: both are large only on the click). A click starts at such a sample in the frame (not in the first p
    samples of the recording) and runs to the last such sample with no gap over p; one longer than `length` ms is left
    (a transient, not a click). Its samples are replaced by the least-squares AR interpolation, and the forward
    residuals of the p samples after it worked out again. The look-ahead is A = longest + 2p."""
    longest = max(1, int(round(length * rate / 1000.0)))

    def repair(buf, start, f, p, end, real_end, state):
        a = frame_predictor(buf, start, f, p)
        ef = np.zeros(end)
        eb = np.zeros(end)
        for i in range(start, end - p):
            ef[i] = residual(buf, i, a)
            eb[i] = backward(buf, i, a)
        sigma = max(float(np.median(np.abs(ef[start:start + f]))) / 0.6745, QUIET_SIGMA)
        limit = threshold * sigma

        def hit(i):
            return abs(ef[i]) > limit and abs(eb[i]) > limit

        i = low = max(start, 2 * p)
        stop = min(start + f, real_end)
        while i < stop:
            if not hit(i):
                i += 1
                continue
            s = last = i
            k = i + 1
            while k < min(end - p, real_end) and k - last <= p and last - s + 1 <= longest:
                if hit(k):
                    last = k
                k += 1
            while s > low and abs(ef[s - 1]) > limit and last - s + 1 <= longest:
                s -= 1
            while last + 1 < min(end - p, real_end) and abs(eb[last + 1]) > limit and last - s + 1 <= longest:
                last += 1
            if last - s + 1 <= longest:
                buf[s:last + 1] = lsar(buf, s, last, a).astype(np.float32)
                for j in range(last + 1, min(last + p + 1, end - p)):
                    ef[j] = residual(buf, j, a)
            i = low = last + 1

    return repair_frames(x, rate, frame, order, longest + 2 * order, repair)


def clean_predictor(buf, start, top, p, bad):
    """The order-p predictor fitted by least squares (the covariance method) over the rows n in [start, top) whose
    samples n - p .. n are all good (none clipped and left), a ridge of 1e-9 of the mean diagonal added; the frame's
    autocorrelation predictor when fewer than 2p rows are good."""
    rows = [n for n in range(start, top) if not bad[n - p:n + 1].any()]
    if len(rows) < 2 * p:
        return None
    lag = np.array([[float(buf[n - k]) for k in range(p + 1)] for n in rows])
    cov = lag[:, 1:].T @ lag[:, 1:]
    rhs = lag[:, 1:].T @ lag[:, 0]
    cov += np.eye(p) * (1e-9 * np.trace(cov) / p + 1e-30)
    return np.linalg.solve(cov, rhs)


def declip(x, rate, level=None, near=0.05, length=5.0, run=2, order=12, frame=20.0, headroom=0.0):
    """Clipped runs: at least `run` samples in a row of one sign at or over the clip level less `near` dB (and not
    rebuilt already), the clip level the value (dBFS) or, by default, the largest magnitude come in so far (up to the end
    of the look-ahead A = longest + p); a run longer than `length` ms is left. A run's samples are replaced by the
    least-squares AR interpolation, the predictor fitted to the good rows of the frame and its look-ahead
    (clean_predictor), then any that came out under the clip level set to it (a clipped sample was at least that); then
    every sample times 10^(-headroom / 20)."""
    longest = max(1, int(round(length * rate / 1000.0)))
    fixed = None if level is None else db_gain(level)

    def repair(buf, start, f, p, end, real_end, state):
        if "repaired" not in state:
            state["repaired"] = np.zeros(len(buf), bool)
            state["peak"] = 0.0
        done = state["repaired"]
        top = min(end, real_end)
        seen = state.get("seen", p)
        if top > seen:
            state["peak"] = max(state["peak"], float(np.abs(buf[seen:top]).max()))
            state["seen"] = top
        clip = fixed if fixed is not None else state["peak"]
        if clip <= 0:
            return
        bar = clip * db_gain(-near)

        def clipped(i, sign):
            return not done[i] and abs(float(buf[i])) >= bar and (sign == 0 or np.sign(buf[i]) == sign)

        a = None
        i = start
        stop = min(start + f, real_end)
        while i < stop:
            if not clipped(i, 0) or clipped(i - 1, np.sign(buf[i])):
                i += 1
                continue
            sign = np.sign(buf[i])
            s = e = i
            while e + 1 < top and clipped(e + 1, sign) and e + 1 - s < longest:
                e += 1
            m = e - s + 1
            if m >= run and not (e + 1 < top and clipped(e + 1, sign)):
                if a is None:
                    bad = np.zeros(len(buf), bool)
                    for j in range(start - p, top):
                        bad[j] = clipped(j, 0)
                    a = clean_predictor(buf, start, top, p, bad)
                    if a is None:
                        a = frame_predictor(buf, start, f, p)
                y = lsar(buf, s, e, a)
                y = np.where(sign * y < clip, sign * clip, y)
                buf[s:e + 1] = y.astype(np.float32)
                done[s:e + 1] = True
            while e + 1 < top and clipped(e + 1, sign):
                e += 1
            i = e + 1

    out, latency = repair_frames(x, rate, frame, order, longest + order, repair)
    if headroom != 0:
        out = (out.astype(np.float64) * db_gain(-headroom)).astype(np.float32)
    return out, latency


# ------------------------------------------------------------------ deess

def sidechain(mode, hertz, q, rate):
    """The filter that splits the sibilance band off: mode high, the RBJ cookbook's low-pass (the band is x less it:
    everything over the frequency, so x - band is the low-pass itself); mode band, the RBJ band-pass of 0 dB peak gain
    (the band is its output, and x - band the RBJ notch). (b0, b1, b2, a1, a2, gain at 0 Hz), a0 divided out."""
    w0 = 2 * math.pi * hertz / rate
    cos, alpha = math.cos(w0), math.sin(w0) / (2 * q)
    a0 = 1 + alpha
    if mode == "high":
        b0 = (1 - cos) / 2 / a0
        return b0, (1 - cos) / a0, b0, -2 * cos / a0, (1 - alpha) / a0, 1.0
    return alpha / a0, 0.0, -alpha / a0, -2 * cos / a0, (1 - alpha) / a0, 0.0


def curve(level, threshold, ratio, knee):
    over = level - threshold
    if 2 * over <= -knee:
        return 0.0
    if knee > 0 and 2 * abs(over) <= knee:
        t = over + knee / 2
        return (1.0 / ratio - 1.0) * t * t / (2 * knee)
    return over * (1.0 / ratio - 1.0)


def deess(x, rate, threshold=-30.0, mode="high", frequency=5000.0, q=0.7071, ratio=4.0, knee=6.0, attack=1.0, release=50.0, window=5.0, link=True):
    """A split-band compressor on the sibilance band: band = the sidechain filter (settled on the first sample: its input
    memory that sample, its output that sample times its gain at 0 Hz); the level a one-pole mean square (`window`) of the band's largest
    square over the linked channels; the compressor's curve (P2's) on it, the reduction in dB smoothed with `attack`
    (down) and `release` (up); the output x - (1 - 10^(reduction / 20)) · band: the band turned down, the rest as it was."""
    b0, b1, b2, a1, a2, dc = sidechain(mode, frequency, q, rate)
    cw, ca, cr = coef(window, rate), coef(attack, rate), coef(release, rate)
    knee_start = 10 ** ((threshold - knee / 2) / 10)
    channels, n = x.shape
    groups = [list(range(channels))] if link else [[c] for c in range(channels)]
    band = np.zeros((channels, n))
    for c in range(channels):
        x1 = x2 = float(x[c, 0])
        y1 = y2 = x1 * dc
        for i in range(n):
            xi = float(x[c, i])
            yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            band[c, i] = xi - yi if mode == "high" else yi
            x2, x1, y2, y1 = x1, xi, y1, yi
    out = np.zeros_like(x)
    for chans in groups:
        env = red = 0.0
        last, last_gain = 0.0, 1.0
        for i in range(n):
            level = max(band[c, i] * band[c, i] for c in chans)
            env = follow(env, level, cw, SILENT_POWER)
            wanted = curve(10 * math.log10(env), threshold, ratio, knee) if env > knee_start else 0.0
            red = follow(red, wanted, ca if wanted < red else cr, GAIN_SNAP)
            if red != last:
                last, last_gain = red, db_gain(red)
            for c in chans:
                out[c, i] = np.float32(float(x[c, i]) - (1 - last_gain) * band[c, i])
    return out, 0


# ------------------------------------------------------------------ signals

def voiced(rate, n, f0, seed):
    """A voice-like sound: harmonics 1 to 20 of a gliding f0 at 1/k, with a little noise (as the dynamics fixture's)."""
    t = np.arange(n) / rate
    f = f0 * (1 + 0.15 * np.sin(2 * np.pi * 1.3 * t))
    phase = 2 * np.pi * np.cumsum(f) / rate
    s = sum(np.sin(k * phase) / k for k in range(1, min(21, int(rate / 2 / (1.2 * f0)))))
    rng = np.random.default_rng(seed)
    return s / np.abs(s).max() + 0.02 * rng.standard_normal(n)


BURSTS = [(0.5, 0.6, 140), (1.4, 0.5, 180), (2.2, 0.5, 120)]


def bursts(rate, n, level, seed, spans=BURSTS):
    x = np.zeros(n)
    for start, length, f0 in spans:
        a, m = int(start * rate), int(length * rate)
        env = np.sin(np.pi * np.arange(m) / m) ** 0.5
        x[a:a + m] += db_gain(level) * 1.5 * env * voiced(rate, m, f0, seed + a)
    return x


def pink(n, rng):
    spectrum = np.fft.rfft(rng.standard_normal(n))
    spectrum[1:] /= np.sqrt(np.arange(1, len(spectrum)))
    spectrum[0] = 0
    y = np.fft.irfft(spectrum, n)
    return y / np.sqrt(np.mean(y * y))


def speech_in_noise(kind):
    rate, n = 16000, 48000
    rng = np.random.default_rng(3110 if kind == "white" else 3111)
    noise = rng.standard_normal(n) if kind == "white" else pink(n, rng)
    x = bursts(rate, n, -20, 3112) + db_gain(-45) * noise
    return x.astype(np.float32)[None, :], rate


CLICKS = [(1200, 1, 0.6), (3100, 2, -0.5), (5000, 3, 0.4), (7300, 1, -0.7), (9100, 4, 0.5), (11800, 6, -0.4), (13700, 2, 0.8)]


def clicks16k():
    rate, n = 16000, 16000
    t = np.arange(n) / rate
    rng = np.random.default_rng(3113)
    clean = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.15 * np.sin(2 * np.pi * 1250 * t + 0.4) + 0.003 * rng.standard_normal(n)
    x = clean.copy()
    for at, length, height in CLICKS:
        x[at:at + length] += height * np.array([1.0, 0.7, -0.5, 0.4, -0.3, 0.2][:length])
    return x.astype(np.float32)[None, :], clean.astype(np.float32)[None, :], rate


def clipped16k():
    rate, n = 16000, 16000
    t = np.arange(n) / rate
    rng = np.random.default_rng(3114)
    clean = 0.55 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 330 * t + 0.7) + 0.002 * rng.standard_normal(n)
    clean *= 0.9 / np.abs(clean).max()
    return np.clip(clean, -0.5, 0.5).astype(np.float32)[None, :], clean.astype(np.float32)[None, :], rate


def highband(rate, n, low, seed):
    """Noise over `low` Hz (a sibilant), unit RMS."""
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / rate)
    spectrum[f < low] = 0
    y = np.fft.irfft(spectrum, n)
    return y / np.sqrt(np.mean(y * y))


SIBILANTS16 = [(0.35, 0.15), (1.1, 0.2), (1.7, 0.12)]
SIBILANTS48 = [(0.02, 0.08), (0.36, 0.1)]


def sibilant(rate, n, channels, sibilants, spans):
    out = []
    for c in range(channels):
        x = bursts(rate, n, -22 - 4 * c, 3115 + c, spans)
        s = highband(rate, n, 4000, 3117 + c)
        for start, length in sibilants:
            a, m = int(start * rate), int(length * rate)
            x[a:a + m] += db_gain(-12) * np.hanning(m) ** 0.5 * s[a:a + m]
        out.append(x)
    return np.stack(out).astype(np.float32)


def reverb16k():
    rate, n = 16000, 48000
    rng = np.random.default_rng(3118)
    dry = bursts(rate, n, -16, 3119, [(0.3, 0.4, 150), (1.3, 0.4, 170), (2.2, 0.4, 130)])
    m = int(0.8 * rate)
    t = np.arange(m) / rate
    ir = rng.standard_normal(m) * np.exp(-3 * math.log(10) * t / 0.6)
    ir[0] = 1.0
    ir /= np.sqrt(np.sum(ir * ir))
    wet = np.convolve(dry, ir)[:n]
    return wet.astype(np.float32)[None, :], dry.astype(np.float32)[None, :], rate


# ------------------------------------------------------------------ effects

def rms_db(x):
    return 10 * math.log10(max(float(np.mean(np.asarray(x, np.float64) ** 2)), 1e-30))


def spans(rate, items, margin):
    return [(int((s + margin) * rate), int((s + l - margin) * rate)) for s, l in items]


def pauses(rate, n, items, margin, after):
    """The stretches between the bursts (and after the last), `margin` s in from each burst's end and the next's start,
    from `after` s on."""
    edges = [(int(after * rate), None)]
    out = []
    cursor = int(after * rate)
    for s, l, *_ in items:
        a = int((s - margin) * rate)
        if a > cursor:
            out.append((cursor, a))
        cursor = max(cursor, int((s + l + margin) * rate))
    if cursor < n:
        out.append((cursor, n))
    return out


def level(x, ranges):
    return rms_db(np.concatenate([x[a:b] for a, b in ranges]))


def band_level(x, rate, ranges, low, high):
    """The power from `low` to `high` Hz (the bins whose frequency is in [low, high)) of the frames centred in the
    ranges, in dB: the library's STFT (centred, Hann, the 32 ms window), its power summed over the bins, averaged."""
    size = window_size(0.032, rate)
    spec = stft_torch(np.asarray(x, np.float32), size)
    power = spec.real.astype(np.float64) ** 2 + spec.imag.astype(np.float64) ** 2
    bins = [k for k in range(size // 2 + 1) if low <= k * rate / size < high]
    frames = [f for f in range(spec.shape[0]) if any(a <= f * (size // 4) < b for a, b in ranges)]
    return 10 * math.log10(max(float(power[np.ix_(frames, bins)].sum(axis=1).mean()), 1e-30))


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what, file=sys.stderr)
    if not ok:
        raise SystemExit(f"effect not met: {what}")


# ------------------------------------------------------------------ the cases

CASES = [
    # (pipeline text, input, function)
    ("denoise-live", "speechwhite16k", lambda x, r: denoise_live(x, r)),
    ("denoise-live=2 floor=-30 window=0.016 track=1 smooth=30 bias=1.4 frames=3", "speechpink16k",
     lambda x, r: denoise_live(x, r, strength=2, floor=-30, window=0.016, track=1, smooth=30, bias=1.4, frames=3)),
    ("spectral-gate", "speechwhite16k", lambda x, r: spectral_gate(x, r)),
    ("spectral-gate=2 reduce=0.9 quantile=0.3 smooth=30 spread=60 window=0.032", "speechpink16k",
     lambda x, r: spectral_gate(x, r, threshold=2, reduce=0.9, quantile=0.3, smooth=30, spread=60, window=0.032)),
    ("declick", "clicks16k", lambda x, r: declick(x, r)),
    ("declick=8 length=1 order=24 frame=10", "clicks16k", lambda x, r: declick(x, r, threshold=8, length=1, order=24, frame=10)),
    ("declip", "clipped16k", lambda x, r: declip(x, r)),
    ("declip level=-6.0206 near=0.1 length=4 run=3 order=20 frame=30 headroom=6", "clipped16k",
     lambda x, r: declip(x, r, level=-6.0206, near=0.1, length=4, run=3, order=20, frame=30, headroom=6)),
    ("deess", "sibilant16k", lambda x, r: deess(x, r)),
    ("deess=-35 mode=band frequency=7000 q=0.7 ratio=6 knee=0 attack=0.5 release=80 window=2 link=false", "sibilant48k",
     lambda x, r: deess(x, r, threshold=-35, mode="band", frequency=7000, q=0.7, ratio=6, knee=0, attack=0.5, release=80, window=2, link=False)),
    ("dereverb", "reverb16k", lambda x, r: dereverb(x, r)),
    ("dereverb=1.5 t60=0.6 delay=40 floor=-20 window=0.016 smooth=10", "reverb16k",
     lambda x, r: dereverb(x, r, strength=1.5, t60=0.6, delay=40, floor=-20, window=0.016, smooth=10)),
]


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import scipy

    clicks, clicks_clean, _ = clicks16k()
    clipped, clipped_clean, _ = clipped16k()
    wet, dry, _ = reverb16k()
    inputs = {
        "speechwhite16k": speech_in_noise("white"),
        "speechpink16k": speech_in_noise("pink"),
        "clicks16k": (clicks, 16000),
        "clicksclean16k": (clicks_clean, 16000),
        "clipped16k": (clipped, 16000),
        "clippedclean16k": (clipped_clean, 16000),
        "sibilant16k": (sibilant(16000, 32000, 1, SIBILANTS16, [(0.5, 0.5, 150), (1.3, 0.35, 170)]), 16000),
        "sibilant48k": (sibilant(48000, 24000, 2, SIBILANTS48, [(0.12, 0.22, 160)]), 48000),
        "reverb16k": (wet, 16000),
    }
    for name, (x, rate) in inputs.items():
        np.save(os.path.join(args.out, f"{name}.npy"), x)

    cases, outputs = [], {}
    for k, (text, source, run) in enumerate(CASES):
        x, rate = inputs[source]
        y, latency = run(x, rate)
        name = f"{k:02d}-{text.split('=')[0].split(' ')[0]}-{source}.npy"
        np.save(os.path.join(args.out, name), y)
        outputs[text] = (y, latency)
        cases.append({"pipeline": text, "input": f"{source}.npy", "rate": rate, "channels": x.shape[0], "output": name, "latency": latency})
        print(f"{text} on {source}: {y.shape}, latency {latency}", file=sys.stderr)

    effects = []

    def aligned(text):
        y, latency = outputs[text]
        return y[:, latency:]

    # Noise in the pauses down, the bursts' level kept.
    for text, source in [(CASES[0][0], "speechwhite16k"), (CASES[1][0], "speechpink16k"), (CASES[2][0], "speechwhite16k"), (CASES[3][0], "speechpink16k")]:
        x = inputs[source][0][0]
        y = aligned(text)[0]
        quiet = pauses(16000, len(x), BURSTS, 0.1, 0.3)
        loud = spans(16000, [(s, l) for s, l, _ in BURSTS], 0.1)
        down = level(x, quiet) - level(y, quiet)
        kept = level(y, loud) - level(x, loud)
        effects.append({"pipeline": text, "kind": "noise", "pauses": quiet, "bursts": loud, "noise_down_db": down, "burst_change_db": kept})
        check(down >= 10 and abs(kept) <= 1, f"{text} on {source}: the pauses' noise down {down:.2f} dB (at least 10), the bursts by {kept:+.2f} dB (within 1)")

    import noisereduce
    x = inputs["speechwhite16k"][0]
    nr = noisereduce.reduce_noise(y=x[0].astype(np.float64), sr=16000, stationary=True, n_std_thresh_stationary=1.5, prop_decrease=1.0)
    quiet = pauses(16000, x.shape[1], BURSTS, 0.1, 0.3)
    loud = spans(16000, [(s, l) for s, l, _ in BURSTS], 0.1)
    cross = {"tool": "noisereduce.reduce_noise(stationary=True, n_std_thresh_stationary=1.5, prop_decrease=1.0)", "input": "speechwhite16k.npy",
             "noise_down_db": level(x[0], quiet) - level(nr, quiet), "burst_change_db": level(nr, loud) - level(x[0], loud)}
    print(f"noisereduce on speechwhite16k: pauses down {cross['noise_down_db']:.2f} dB, bursts {cross['burst_change_db']:+.2f} dB", file=sys.stderr)

    # The clicks' error energy down against the clean tone.
    clean = inputs["clicksclean16k"][0][0].astype(np.float64)
    near = [(at - 8, at + length + 24) for at, length, _ in CLICKS]
    for text in (CASES[4][0], CASES[5][0]):
        y = aligned(text)[0].astype(np.float64)
        before = level(clicks[0] - clean, near)
        after = level(y - clean, near)
        effects.append({"pipeline": text, "kind": "clicks", "regions": near, "error_before_db": before, "error_after_db": after, "down_db": before - after})
        check(before - after >= 20, f"{text}: the clicks' error {before:.2f} dB before, {after:.2f} dB after (down {before - after:.2f}, at least 20)")
        untouched = [(a, b) for a, b in zip([0] + [e for _, e in near], [s for s, _ in near] + [len(clean)]) if b > a]
        moved = float(np.max(np.abs(np.concatenate([(y - clicks[0])[a:b] for a, b in untouched]))))
        effects[-1]["largest_change_elsewhere"] = moved
        check(moved <= 0.05, f"{text}: away from the clicks the tone moved by at most {moved:.4f} (at most 0.05)")

    # Clipped samples rebuilt near the original.
    clean = inputs["clippedclean16k"][0][0].astype(np.float64)
    hit = np.abs(clipped[0]) >= 0.5 * db_gain(-0.05)
    hit[:int(0.02 * 16000)] = False                                              # the first and last frames: no sound before or after them to fit
    hit[-int(0.02 * 16000):] = False
    for text, gain in ((CASES[6][0], 1.0), (CASES[7][0], db_gain(-6))):
        y = aligned(text)[0].astype(np.float64) / gain
        before = float(np.sqrt(np.mean((clipped[0][hit] - clean[hit]) ** 2)))
        after = float(np.sqrt(np.mean((y[hit] - clean[hit]) ** 2)))
        worst = float(np.max(np.abs(y[hit] - clean[hit])))
        effects.append({"pipeline": text, "kind": "clipping", "gain": gain, "rms_error_before": before, "rms_error_after": after, "largest_error_after": worst,
                        "clipped_samples": int(hit.sum())})
        check(after <= 0.1 * before and worst <= 0.1, f"{text}: the clipped samples' RMS error {before:.4f} before, {after:.4f} after (a tenth or less), the largest {worst:.4f} (at most 0.1)")

    # The sibilance band down, the voice kept.
    for text, source, sib, voice, band in ((CASES[8][0], "sibilant16k", SIBILANTS16, [(0.5, 0.5), (1.3, 0.35)], (4000, 8000)),
                                           (CASES[9][0], "sibilant48k", SIBILANTS48, [(0.12, 0.22)], (6000, 8500))):
        x, rate = inputs[source]
        y = aligned(text)
        regions = spans(rate, sib, 0.02)
        vregions = spans(rate, voice, 0.05)
        band_down = band_level(x[0], rate, regions, *band) - band_level(y[0], rate, regions, *band)
        voice_change = band_level(y[0], rate, vregions, 0, 3000) - band_level(x[0], rate, vregions, 0, 3000)
        effects.append({"pipeline": text, "kind": "sibilance", "sibilants": regions, "voice": vregions, "band": list(band), "band_down_db": band_down, "voice_change_db": voice_change})
        check(band_down >= 6 and abs(voice_change) <= 0.5,
              f"{text}: the sibilants from {band[0]} to {band[1]} Hz down {band_down:.2f} dB (at least 6), the voice under 3 kHz by {voice_change:+.2f} dB (within 0.5)")

    # The late reverberation down.
    x = wet[0].astype(np.float64)
    tails = [(int((s + l + 0.08) * 16000), int((s + l + 0.5) * 16000)) for s, l in [(0.3, 0.4), (1.3, 0.4), (2.2, 0.4)]]
    heads = [(int((s + 0.05) * 16000), int((s + l - 0.05) * 16000)) for s, l in [(0.3, 0.4), (1.3, 0.4), (2.2, 0.4)]]
    for text, least in ((CASES[10][0], 4), (CASES[11][0], 6)):
        y = aligned(text)[0].astype(np.float64)
        down = level(x, tails) - level(y, tails)
        kept = level(y, heads) - level(x, heads)
        effects.append({"pipeline": text, "kind": "reverb", "tails": tails, "speech": heads, "tail_down_db": down, "speech_change_db": kept})
        check(down >= least and kept >= -3, f"{text}: the late reverberation (80 to 500 ms after each burst) down {down:.2f} dB (at least {least}), the bursts by {kept:+.2f} dB (at least -3)")

    files = {n: sha256(os.path.join(args.out, n)) for n in sorted(os.listdir(args.out)) if n.endswith(".npy")}
    versions = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "noisereduce": getattr(noisereduce, "__version__", "3")}
    with open(os.path.join(args.out, "repair.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"cases": cases, "effects": effects, "noisereduce": cross, "versions": versions, "files": files}, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
