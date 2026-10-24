"""The reference for Idrak.Audio's dynamics processors (plan 31, phase P2): gate, compress, limit, agc, loudness,
loudness-live and peaknorm, each written here sample by sample from its equations (the ones the C# comments state), and
the integrated loudness of ITU-R BS.1770-4 / EBU R128 also from pyloudnorm.

    python -m venv audio && audio/bin/pip install numpy scipy pyloudnorm
    audio/bin/python tools/audio/make_audio_dynamics.py tests/Idrak.Tests/data/audio/dynamics

Nothing is downloaded: the signals are generated here from fixed seeds. Writes, into the folder:
- the inputs, float32 [channels, samples] .npy: speech16k (mono, 16 kHz, 2.5 s: voiced bursts at three levels with
  pauses over a quiet noise floor and a loud attack), stereo48k (48 kHz, 0.5 s: a tone and noise bursts left, a lower
  tone with a near full-scale transient and a silent stretch right), tone48k (stereo, 48 kHz, 1 s: a 997 Hz sine at
  -23 dBFS peak in both channels) and noise16k (mono, 16 kHz, 2 s: white noise at -20 dBFS RMS, then at -45);
- for every live case, its output as float32 .npy (a limiter's output is longer by its look-ahead: silence first);
- dynamics.json: each live case (its pipeline text, input and output files), each loudness measurement (this
  script's integrated loudness with the standard's filters and with RBJ's, and pyloudnorm's with each: its "DeMan"
  filter class and its default "K-weighting"; the sample peak and the true peak), each whole-recording case's gain, the versions and every file's
  SHA-256.
Reruns write the same bytes on the same machine and library versions.
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

RMS_FLOOR = 1e-30      # a mean square under it (-300 dB) is silence: the envelope snaps to its target
GAIN_SNAP = 1e-12      # a gain (linear or dB) this close to its target is the target


def coef(ms, rate):
    """The one-pole coefficient of a time constant of `ms` milliseconds: exp(-1 / (tau fs)); 0 (instant) for 0."""
    return 0.0 if ms <= 0 else math.exp(-1000.0 / (ms * rate))


def follow(value, target, a, snap):
    """One step of a one-pole follower: target + (value - target) a, the target itself once within `snap`."""
    diff = (value - target) * a
    return target + diff if abs(diff) >= snap else target


def db_to_gain(db):
    return 10.0 ** (db / 20.0)


def detect(x, i, kind):
    """The detector input at sample i over the channels of x ([channels, n] float32): 'square' and 'peak' take the
    largest over the channels, 'mean' the mean of the squares."""
    if kind == "peak":
        return max(abs(float(x[c, i])) for c in range(x.shape[0]))
    if kind == "square":
        best = 0.0
        for c in range(x.shape[0]):
            v = float(x[c, i])
            best = max(best, v * v)
        return best
    s = 0.0
    for c in range(x.shape[0]):
        v = float(x[c, i])
        s += v * v
    return s / x.shape[0]


def groups(x, link):
    """The channel groups that share a gain: all of them when linked, each alone when not."""
    return [list(range(x.shape[0]))] if link else [[c] for c in range(x.shape[0])]


def apply_gains(x, chans, gains, out):
    for c in chans:
        out[c] = (x[c].astype(np.float64) * gains).astype(np.float32)


# ------------------------------------------------------------------ live processors

def gate(x, rate, threshold=-50.0, attack=1.0, hold=50.0, release=100.0, range_db=60.0, hysteresis=3.0, window=10.0, link=True):
    """A noise gate: the level is a one-pole mean square (time constant `window`) of the detector (the largest square
    over the linked channels); the gate opens when it reaches the threshold, stays open while it is at or above the
    threshold less the hysteresis, and when it falls under that, after `hold`, closes; the gain moves to 1 (open) with
    `attack` and to -range dB (closed) with `release`. It starts closed."""
    out = np.empty_like(x)
    aw, aa, ar = coef(window, rate), coef(attack, rate), coef(release, rate)
    t_open, t_close = 10.0 ** (threshold / 10.0), 10.0 ** ((threshold - hysteresis) / 10.0)
    floor = db_to_gain(-range_db)
    hold_n = int(round(hold * rate / 1000.0))
    for chans in groups(x, link):
        sub = x[chans]
        env, is_open, left, g = 0.0, False, 0, floor
        gains = np.empty(x.shape[1])
        for i in range(x.shape[1]):
            env = follow(env, detect(sub, i, "square"), aw, RMS_FLOOR)
            if env >= t_open:
                is_open, left = True, hold_n
            elif is_open:
                if env >= t_close:
                    left = hold_n
                elif left > 0:
                    left -= 1
                else:
                    is_open = False
            target = 1.0 if is_open else floor
            g = follow(g, target, aa if target > g else ar, GAIN_SNAP)
            gains[i] = g
        apply_gains(x, chans, gains, out)
    return out


def compress_curve(level_db, threshold, ratio, knee):
    """The static curve's gain in dB (<= 0) for a level in dB: none under the knee, a quadratic through the knee, then
    (level - threshold)(1/ratio - 1) (Giannoulis, Massberg and Reiss 2012)."""
    over = level_db - threshold
    if 2 * over <= -knee:
        return 0.0
    if knee > 0 and 2 * abs(over) <= knee:
        t = over + knee / 2
        return (1.0 / ratio - 1.0) * t * t / (2 * knee)
    return over * (1.0 / ratio - 1.0)


def compress(x, rate, threshold=-20.0, ratio=3.0, knee=6.0, attack=10.0, release=100.0, makeup=0.0, detector="rms", window=10.0, link=True):
    """A feed-forward compressor: the level is a one-pole mean square (`window`) of the largest square over the linked
    channels ('rms') or their largest magnitude ('peak'); the static curve's gain in dB is smoothed with `attack` (more
    reduction) or `release` (less), then the makeup added; 'auto' makeup is the curve's reduction at 0 dBFS."""
    out = np.empty_like(x)
    if makeup == "auto":
        makeup = -compress_curve(0.0, threshold, ratio, knee)
    aw, aa, ar = coef(window, rate), coef(attack, rate), coef(release, rate)
    for chans in groups(x, link):
        sub = x[chans]
        env, gs = 0.0, 0.0
        gains = np.empty(x.shape[1])
        for i in range(x.shape[1]):
            if detector == "rms":
                env = follow(env, detect(sub, i, "square"), aw, RMS_FLOOR)
                level = 10 * math.log10(env) if env > 0 else -math.inf
            else:
                p = detect(sub, i, "peak")
                level = 20 * math.log10(p) if p > 0 else -math.inf
            gc = compress_curve(level, threshold, ratio, knee) if level > -math.inf else 0.0
            gs = follow(gs, gc, aa if gc < gs else ar, GAIN_SNAP)
            gains[i] = db_to_gain(gs + makeup)
        apply_gains(x, chans, gains, out)
    return out


def limit(x, rate, ceiling=-1.0, lookahead=5.0, release=50.0, link=True):
    """A look-ahead peak limiter: L = round(lookahead fs) samples. Each sample's needed gain r = min(1, c / peak) (the
    largest magnitude over the linked channels); M[n] the least r over the last L + 1 samples; the envelope follows M
    down at once and back up with `release`; the gain is the mean of the envelope's last L + 1 values, applied to the
    input L samples back (the output delayed by L, silence first, the input followed by L samples of silence), each
    sample then clamped to the largest float not over the ceiling."""
    L = int(round(lookahead * rate / 1000.0))
    c = db_to_gain(ceiling)
    cf = np.float32(c)
    if float(cf) > c:
        cf = np.nextafter(cf, np.float32(0))
    n = x.shape[1]
    padded = np.concatenate([x, np.zeros((x.shape[0], L), np.float32)], axis=1)
    delayed = np.concatenate([np.zeros((x.shape[0], L), np.float32), x], axis=1)
    out = np.empty_like(delayed)
    a = coef(release, rate)
    for chans in groups(x, link):
        sub = padded[chans]
        r = np.ones(n + L)
        for i in range(n + L):
            p = detect(sub, i, "peak")
            r[i] = 1.0 if p <= c else c / p
        hist = np.concatenate([np.ones(L), r])
        env_hist = np.ones(n + 2 * L)
        env = 1.0
        gains = np.empty(n + L)
        for i in range(n + L):
            m = hist[i:i + L + 1].min()
            env = m if m < env else follow(env, m, a, GAIN_SNAP)
            env_hist[L + i] = env
            gains[i] = env_hist[i:i + L + 1].sum() / (L + 1)
        for ch in chans:
            y = (delayed[ch].astype(np.float64) * gains).astype(np.float32)
            out[ch] = np.clip(y, -cf, cf)
    return out


def agc(x, rate, target=-20.0, max_db=20.0, min_db=-20.0, attack=100.0, release=1000.0, window=300.0, gate_db=-50.0, link=True):
    """Automatic gain control: the level is a one-pole mean square (`window`) of the mean square over the linked
    channels; at or above the gate the wanted gain is target - level, within [min, max] dB, else the gain holds; the
    gain in dB follows it with `attack` (down) or `release` (up), from 0 dB."""
    out = np.empty_like(x)
    aw, aa, ar = coef(window, rate), coef(attack, rate), coef(release, rate)
    gate_lin = 10.0 ** (gate_db / 10.0)
    for chans in groups(x, link):
        sub = x[chans]
        env, gdb = 0.0, 0.0
        gains = np.empty(x.shape[1])
        for i in range(x.shape[1]):
            env = follow(env, detect(sub, i, "mean"), aw, RMS_FLOOR)
            wanted = min(max(target - 10 * math.log10(env), min_db), max_db) if env >= gate_lin else gdb
            gdb = follow(gdb, wanted, aa if wanted < gdb else ar, GAIN_SNAP)
            gains[i] = db_to_gain(gdb)
        apply_gains(x, chans, gains, out)
    return out


# ------------------------------------------------------------------ BS.1770

# The K-weighting filters of ITU-R BS.1770-4 for any rate: Brecht De Man's design, which gives the standard's 48 kHz
# coefficients (pyloudnorm's "DeMan" filter class): a high shelf, then the RLB high-pass.
SHELF_GAIN, SHELF_Q, SHELF_HZ = 3.99984385397, 0.7071752369554193, 1681.9744509555319
PASS_Q, PASS_HZ = 0.5003270373253953, 38.13547087613982


def k_weighting(rate, weighting="bs1770"):
    """(shelf, high-pass) biquads (b0, b1, b2, a1, a2): 'bs1770' the standard's filters (De Man's design), 'rbj'
    pyloudnorm's default K-weighting (the RBJ cookbook's high shelf of 4 dB at 1500 Hz, Q 1/sqrt 2, and high-pass at
    38 Hz, Q 0.5)."""
    if weighting == "rbj":
        return rbj_shelf(4.0, 1 / math.sqrt(2), 1500.0, rate), rbj_high(0.5, 38.0, rate)
    k = math.tan(math.pi * SHELF_HZ / rate)
    vh = 10.0 ** (SHELF_GAIN / 20.0)
    vb = vh ** 0.499666774155
    a0 = 1.0 + k / SHELF_Q + k * k
    shelf = ((vh + vb * k / SHELF_Q + k * k) / a0, 2.0 * (k * k - vh) / a0, (vh - vb * k / SHELF_Q + k * k) / a0,
             2.0 * (k * k - 1.0) / a0, (1.0 - k / SHELF_Q + k * k) / a0)
    k = math.tan(math.pi * PASS_HZ / rate)
    a0 = 1.0 + k / PASS_Q + k * k
    high = (1.0, -2.0, 1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / PASS_Q + k * k) / a0)
    return shelf, high


def rbj_shelf(gain, q, hz, rate):
    a = 10 ** (gain / 40.0)
    w0 = 2.0 * math.pi * hz / rate
    alpha, cos = math.sin(w0) / (2.0 * q), math.cos(w0)
    a0 = (a + 1) - (a - 1) * cos + 2 * math.sqrt(a) * alpha
    return (a * ((a + 1) + (a - 1) * cos + 2 * math.sqrt(a) * alpha) / a0, -2 * a * ((a - 1) + (a + 1) * cos) / a0,
            a * ((a + 1) + (a - 1) * cos - 2 * math.sqrt(a) * alpha) / a0, 2 * ((a - 1) - (a + 1) * cos) / a0,
            ((a + 1) - (a - 1) * cos - 2 * math.sqrt(a) * alpha) / a0)


def rbj_high(q, hz, rate):
    w0 = 2.0 * math.pi * hz / rate
    alpha, cos = math.sin(w0) / (2.0 * q), math.cos(w0)
    a0 = 1 + alpha
    return (1 + cos) / 2 / a0, -(1 + cos) / a0, (1 + cos) / 2 / a0, -2 * cos / a0, (1 - alpha) / a0


def biquad(x, b0, b1, b2, a1, a2):
    """Direct form I from rest, in double."""
    y = np.empty(len(x))
    x1 = x2 = y1 = y2 = 0.0
    for i, xi in enumerate(x):
        yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[i] = yi
        x2, x1, y2, y1 = x1, xi, y1, yi
    return y


def channel_weight(c):
    """BS.1770's channel weights in its five-channel order (L, R, C, Ls, Rs): 1.41 for the surrounds, else 1."""
    return 1.41 if c in (3, 4) else 1.0


def weighted_squares(x, rate, weighting="bs1770"):
    """Sum over the channels of each weight times the K-weighted sample squared, per sample."""
    shelf, high = k_weighting(rate, weighting)
    s = np.zeros(x.shape[1])
    for c in range(x.shape[0]):
        z = biquad(biquad(x[c].astype(np.float64), *shelf), *high)
        s += channel_weight(c) * z * z
    return s


def integrated(x, rate, block=400.0, overlap=0.75, absolute=-70.0, relative=-10.0, weighting="bs1770"):
    """BS.1770-4 integrated loudness: blocks of round(block fs) samples every round(block (1 - overlap)) samples
    (complete blocks only; a recording shorter than one block is one block); a block's loudness -0.691 + 10 log10 of
    its weighted mean square; blocks above the absolute gate, then above the relative gate (their mean's loudness plus
    `relative`), averaged. None when no block passes."""
    s = weighted_squares(x, rate, weighting)
    n = x.shape[1]
    size = int(round(block * rate / 1000.0))
    hop = max(1, int(round(size * (1 - overlap))))
    if n < size:
        means = [s.sum() / n]
    else:
        means = [s[j * hop:j * hop + size].sum() / size for j in range((n - size) // hop + 1)]
    loud = [-0.691 + 10 * math.log10(m) if m > 0 else -math.inf for m in means]
    gated = [m for m, l in zip(means, loud) if l > absolute]
    if not gated:
        return None
    gamma = -0.691 + 10 * math.log10(sum(gated) / len(gated)) + relative
    kept = [m for m, l in zip(means, loud) if l > absolute and l > gamma]
    if not kept:
        return None
    return -0.691 + 10 * math.log10(sum(kept) / len(kept))


def loudness_live(x, rate, target=-23.0, window=3000.0, hop=100.0, attack=500.0, release=2000.0, max_db=20.0, min_db=-20.0, gate_db=-50.0, weighting="bs1770"):
    """Loudness steered live: the weighted K-weighted squares summed over hops of round(hop fs) samples; at each hop's
    end the loudness of the last round(window / hop) hops (of those so far at the start) is measured; at or above the
    gate the wanted gain is target - loudness within [min, max] dB, else (and before the first hop) the gain holds; the
    gain in dB follows the wanted gain with `attack` (down) or `release` (up), from 0 dB, per sample."""
    shelf, high = k_weighting(rate, weighting)
    hop_n = max(1, int(round(hop * rate / 1000.0)))
    slots = max(1, int(round(window / hop)))
    z = [biquad(biquad(x[c].astype(np.float64), *shelf), *high) for c in range(x.shape[0])]
    aa, ar = coef(attack, rate), coef(release, rate)
    ring = [0.0] * slots
    at = filled = count = 0
    acc, gdb, wanted, holding = 0.0, 0.0, 0.0, True
    gains = np.empty(x.shape[1])
    for i in range(x.shape[1]):
        s = 0.0
        for c in range(x.shape[0]):
            s += channel_weight(c) * z[c][i] * z[c][i]
        acc += s
        count += 1
        if count == hop_n:
            ring[at] = acc
            at = (at + 1) % slots
            filled = min(filled + 1, slots)
            acc, count = 0.0, 0
            total = 0.0
            for k in range(filled):
                total += ring[k]
            loud = -0.691 + 10 * math.log10(total / (filled * hop_n)) if total > 0 else -math.inf
            holding = loud < gate_db
            if not holding:
                wanted = min(max(target - loud, min_db), max_db)
        if not holding:
            gdb = follow(gdb, wanted, aa if wanted < gdb else ar, GAIN_SNAP)
        gains[i] = db_to_gain(gdb)
    out = np.empty_like(x)
    apply_gains(x, range(x.shape[0]), gains, out)
    return out


# ------------------------------------------------------------------ peaks

def true_peak(x, oversample=4, taps=6):
    """The largest magnitude of the samples and of the sound between them, read at oversample - 1 points between each
    two samples (and after the last) by a Hann-windowed sinc reaching `taps` samples each side: the value at n + p/L is
    the sum over m in (-taps, taps] of x[n + m] sinc(p/L - m) (0.5 + 0.5 cos(pi (p/L - m) / taps)), zeros past the
    ends; float32 sums, as the C#."""
    best = float(np.abs(x).max()) if x.size else 0.0
    n = x.shape[1]
    for p in range(1, oversample):
        f = p / oversample
        ms = np.arange(-taps + 1, taps + 1)
        t = f - ms
        h = (np.sinc(t) * (0.5 + 0.5 * np.cos(np.pi * t / taps))).astype(np.float32)
        for c in range(x.shape[0]):
            padded = np.concatenate([np.zeros(taps, np.float32), x[c], np.zeros(taps, np.float32)])
            acc = np.zeros(n, np.float32)
            for k, m in enumerate(ms):
                acc += h[k] * padded[taps + m:taps + m + n]
            best = max(best, float(np.abs(acc).max()))
    return best


def loudness_gain(x, rate, target=-23.0, ceiling=-1.0, meter="true", oversample=4, taps=6, **gates):
    """The whole-recording loudness gain: target - integrated, lowered so the peak (true or sample) stays under the
    ceiling; None when the recording is silent."""
    level = integrated(x, rate, **gates)
    if level is None:
        return None, None
    gain = db_to_gain(target - level)
    if meter != "none":
        peak = true_peak(x, oversample, taps) if meter == "true" else float(np.abs(x).max())
        c = db_to_gain(ceiling)
        if peak * gain > c:
            gain = c / peak
    return level, gain


def peak_gain(x, target=-1.0, meter="sample", oversample=4, taps=6):
    peak = true_peak(x, oversample, taps) if meter == "true" else float(np.abs(x).max())
    return None if peak == 0 else db_to_gain(target) / peak


# ------------------------------------------------------------------ signals

def voiced(rate, n, f0, seed):
    """A voice-like sound: harmonics 1 to 20 of a gliding f0 at 1/k, with a little noise."""
    t = np.arange(n) / rate
    f = f0 * (1 + 0.15 * np.sin(2 * np.pi * 1.3 * t))
    phase = 2 * np.pi * np.cumsum(f) / rate
    s = sum(np.sin(k * phase) / k for k in range(1, 21))
    rng = np.random.default_rng(seed)
    return s / np.abs(s).max() + 0.05 * rng.standard_normal(n)


def speech16k():
    rate, n = 16000, 40000
    rng = np.random.default_rng(3101)
    x = 10 ** (-60 / 20) * rng.standard_normal(n)
    for start, length, level, f0 in [(0.15, 0.55, -30, 120), (0.95, 0.6, -10, 160), (1.85, 0.5, -20, 140)]:
        a, m = int(start * rate), int(length * rate)
        env = np.sin(np.pi * np.arange(m) / m) ** 2 * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * np.arange(m) / rate) ** 2)
        x[a:a + m] += 10 ** (level / 20) * 2.5 * env * voiced(rate, m, f0, 3102 + a)
    x[int(1.0 * rate):int(1.0 * rate) + 40] += 0.9 * np.hanning(40)          # a loud attack in the loud phrase
    return np.clip(x, -1, 1).astype(np.float32)[None, :], rate


def stereo48k():
    rate, n = 48000, 24000
    t = np.arange(n) / rate
    rng = np.random.default_rng(3103)
    left = 0.3 * np.sin(2 * np.pi * 1000 * t)
    burst = np.zeros(n)
    burst[6000:12000] = rng.standard_normal(6000) * np.hanning(6000)
    left += 0.25 * burst
    right = 0.2 * np.sin(2 * np.pi * 300 * t)
    right[14400:14448] += 0.99 * np.sign(np.sin(np.arange(48) / 3))             # a near full-scale transient
    right[18000:21600] = 0.0                                                     # a silent stretch
    return np.clip(np.stack([left, right]), -0.999, 0.999).astype(np.float32), rate


def tone48k():
    rate, n = 48000, 48000
    s = 10 ** (-23 / 20) * np.sin(2 * np.pi * 997 * np.arange(n) / rate)
    return np.stack([s, s]).astype(np.float32), rate


def noise16k():
    rate, n = 16000, 32000
    rng = np.random.default_rng(3104)
    x = rng.standard_normal(n)
    x[:16000] *= 10 ** (-20 / 20)
    x[16000:] *= 10 ** (-45 / 20)
    return x.astype(np.float32)[None, :], rate


# ------------------------------------------------------------------ the cases

LIVE = [
    # (pipeline text, input, reference)
    ("gate=-40", "speech16k", lambda x, r: gate(x, r, threshold=-40)),
    ("gate=-35 attack=5 hold=0 release=20 range=30 hysteresis=0 window=2", "speech16k",
     lambda x, r: gate(x, r, threshold=-35, attack=5, hold=0, release=20, range_db=30, hysteresis=0, window=2)),
    ("gate=-30 link=false", "stereo48k", lambda x, r: gate(x, r, threshold=-30, link=False)),
    ("compress=-25 ratio=4", "speech16k", lambda x, r: compress(x, r, threshold=-25, ratio=4)),
    ("compress=-30 ratio=8 knee=0 attack=1 release=50 makeup=auto detector=peak", "speech16k",
     lambda x, r: compress(x, r, threshold=-30, ratio=8, knee=0, attack=1, release=50, makeup="auto", detector="peak")),
    ("compress=-18 ratio=3 knee=10 makeup=4 window=20", "stereo48k", lambda x, r: compress(x, r, threshold=-18, ratio=3, knee=10, makeup=4, window=20)),
    ("compress=-18 ratio=2 link=false", "stereo48k", lambda x, r: compress(x, r, threshold=-18, ratio=2, link=False)),
    ("limit=-6", "speech16k", lambda x, r: limit(x, r, ceiling=-6)),
    ("limit=-3 lookahead=2 release=20", "stereo48k", lambda x, r: limit(x, r, ceiling=-3, lookahead=2, release=20)),
    ("limit=-6 lookahead=0 link=false", "stereo48k", lambda x, r: limit(x, r, ceiling=-6, lookahead=0, link=False)),
    ("agc=-18 attack=20 release=200 window=50", "speech16k", lambda x, r: agc(x, r, target=-18, attack=20, release=200, window=50)),
    ("agc max=12 min=-6 quiet=-40 link=false", "stereo48k", lambda x, r: agc(x, r, max_db=12, min_db=-6, gate_db=-40, link=False)),
    ("loudness-live=-20 window=400 attack=50 release=200", "speech16k", lambda x, r: loudness_live(x, r, target=-20, window=400, attack=50, release=200)),
    ("loudness-live hop=50 max=6 weighting=rbj", "stereo48k", lambda x, r: loudness_live(x, r, hop=50, max_db=6, weighting="rbj")),
]

WHOLE = [
    # (pipeline text, input, gain function)
    ("loudness", "speech16k", lambda x, r: loudness_gain(x, r)[1]),
    ("loudness=-16 ceiling=-3 meter=sample", "speech16k", lambda x, r: loudness_gain(x, r, target=-16, ceiling=-3, meter="sample")[1]),
    ("loudness=-14", "stereo48k", lambda x, r: loudness_gain(x, r, target=-14)[1]),
    ("loudness=-20 weighting=rbj absolute=-60", "speech16k", lambda x, r: loudness_gain(x, r, target=-20, weighting="rbj", absolute=-60)[1]),
    ("loudness=-30 meter=none block=300 overlap=0.5 relative=-8", "noise16k",
     lambda x, r: loudness_gain(x, r, target=-30, meter="none", block=300, overlap=0.5, relative=-8)[1]),
    ("loudness=-23 oversample=8 taps=8", "tone48k", lambda x, r: loudness_gain(x, r, oversample=8, taps=8)[1]),
    ("peaknorm", "speech16k", lambda x, r: peak_gain(x)),
    ("peaknorm=-3 meter=true", "stereo48k", lambda x, r: peak_gain(x, target=-3, meter="true")),
    ("peaknorm=-0.5 meter=true oversample=2 taps=3", "tone48k", lambda x, r: peak_gain(x, target=-0.5, meter="true", oversample=2, taps=3)),
]


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import pyloudnorm
    import scipy

    inputs = {"speech16k": speech16k(), "stereo48k": stereo48k(), "tone48k": tone48k(), "noise16k": noise16k()}
    for name, (x, rate) in inputs.items():
        np.save(os.path.join(args.out, f"{name}.npy"), x)

    live = []
    for k, (text, source, run) in enumerate(LIVE):
        x, rate = inputs[source]
        y = run(x, rate)
        name = f"live-{k:02d}-{text.split('=')[0].split(' ')[0]}-{source}.npy"
        np.save(os.path.join(args.out, name), y)
        live.append({"pipeline": text, "input": f"{source}.npy", "rate": rate, "channels": x.shape[0], "output": name})
        print(f"{text} on {source}: {y.shape}", file=sys.stderr)

    loudness = []
    for name, (x, rate) in inputs.items():
        data = x.T.astype(np.float64) if x.shape[0] > 1 else x[0].astype(np.float64)
        deman = pyloudnorm.Meter(rate, filter_class="DeMan").integrated_loudness(data)
        rbj = pyloudnorm.Meter(rate).integrated_loudness(data)
        mine = integrated(x, rate)
        loudness.append({"input": f"{name}.npy", "rate": rate, "channels": x.shape[0], "integrated": mine, "integrated_rbj": integrated(x, rate, weighting="rbj"), "pyloudnorm_deman": deman,
                         "pyloudnorm_kweighting": rbj, "sample_peak": float(np.abs(x).max()), "true_peak": true_peak(x)})
        print(f"{name}: integrated {mine:.4f}, pyloudnorm DeMan {deman:.4f}, K-weighting {rbj:.4f}", file=sys.stderr)

    whole = []
    for text, source, run in WHOLE:
        x, rate = inputs[source]
        whole.append({"pipeline": text, "input": f"{source}.npy", "rate": rate, "channels": x.shape[0], "gain": run(x, rate)})

    files = {n: sha256(os.path.join(args.out, n)) for n in sorted(os.listdir(args.out)) if n.endswith(".npy")}
    versions = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                "pyloudnorm": getattr(pyloudnorm, "__version__", "0.1.1+")}
    with open(os.path.join(args.out, "dynamics.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"live": live, "loudness": loudness, "whole": whole, "versions": versions, "files": files}, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
