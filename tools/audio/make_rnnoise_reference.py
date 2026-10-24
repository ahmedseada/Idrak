"""The reference for the RNNoise plug-in processor (plan 31, phase P5): xiph's RNNoise, its last classic release (tag
v0.1.1, BSD-3-Clause, https://github.com/xiph/rnnoise), built with the system C compiler and run on a speech-like
signal; and RNNoise's classic weights written as a model file the plug-in reads.

    python3 tools/audio/make_rnnoise_reference.py tests/Idrak.Tests/data/audio/rnnoise
    python3 tools/audio/make_rnnoise_reference.py --rnnoise D:/src/rnnoise tests/Idrak.Tests/data/audio/rnnoise
    python3 tools/audio/make_rnnoise_reference.py --model-only ~/.cache/idrak/rnnoise/rnnoise.rnnn

Needs git (unless --rnnoise names a checkout of v0.1.1) and, for the reference (not for --model-only), numpy and a C
compiler (CC, default cc). Nothing else is downloaded.

The model: RNNoise v0.1.1 carries its trained weights only as C arrays (src/rnn_data.c). This script parses them and
writes them in RNNoise's own model file format, the text that its rnnoise_model_from_file reads ("rnnoise-nu model file
version 1", then per layer: inputs, neurons, activation, weights, biases; dense input, VAD GRU, noise GRU, denoise
GRU, denoise output, VAD output). The C driver below reads that file with RNNoise's reader and checks its output is
the built-in model's bit for bit, so the file is the weights the reference ships.

Writes, into the folder:
- rnnoise.rnnn: the model file (also what --model-only writes, alone);
- input.npy: float32 [1, 96000], 2 s at 48 kHz, multiples of 1/32768 (16-bit samples): 0.1 s of digital silence (RNNoise's
  silent path), then a noise floor (low-passed white noise, about -40 dBFS) under three voiced syllables (harmonics of a
  gliding 110-170 Hz pitch shaped by two formants, about -12 dBFS at their peak) and a fricative (noise over 4 kHz);
- output.npy: float32 [1, 95520], RNNoise's denoised samples aligned with the input (its first frame, the 480 samples
  before the input begins, left out as its demo does; the last frame's own samples need the next frame, so the
  output is one frame shorter), RNNoise's int16-scaled floats divided by 32768;
- vad.npy: float32 [200], the voice probability rnnoise_process_frame returned for each frame;
- rnnoise.json: the source commit, the compiler and flags, the layer sizes, the lengths and every file's SHA-256.
The input is multiplied by 32768 before RNNoise and the output divided by it (exact: powers of two). The C code is
built with -O2 -ffp-contract=off (no fused multiply-adds, so a machine with them computes what one without does).
Reruns write the same bytes on the same machine and compiler.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile

np = None  # numpy: imported for the reference run only (--model-only needs Python and git alone)

RATE = 48000
FRAME = 480
TAG = "v0.1.1"
REPOSITORY = "https://github.com/xiph/rnnoise.git"

# The layers in the order the model file holds them: (name, kind, its arrays in the file's order).
LAYERS = [
    ("input_dense", "dense", ["input_dense_weights", "input_dense_bias"]),
    ("vad_gru", "gru", ["vad_gru_weights", "vad_gru_recurrent_weights", "vad_gru_bias"]),
    ("noise_gru", "gru", ["noise_gru_weights", "noise_gru_recurrent_weights", "noise_gru_bias"]),
    ("denoise_gru", "gru", ["denoise_gru_weights", "denoise_gru_recurrent_weights", "denoise_gru_bias"]),
    ("denoise_output", "dense", ["denoise_output_weights", "denoise_output_bias"]),
    ("vad_output", "dense", ["vad_output_weights", "vad_output_bias"]),
]
ACTIVATIONS = {"ACTIVATION_TANH": 0, "ACTIVATION_SIGMOID": 1, "ACTIVATION_RELU": 2}

DRIVER = r"""
/* Runs RNNoise over float32 samples (48 kHz mono, whole frames), writing each frame's output and voice probability. */
#include <stdio.h>
#include <stdlib.h>
#include "rnnoise.h"
#define FRAME_SIZE 480
int main(int argc, char **argv) {
  if (argc != 5) { fprintf(stderr, "usage: %s MODEL|builtin IN.f32 OUT.f32 VAD.f32\n", argv[0]); return 1; }
  RNNModel *model = NULL;
  if (strcmp(argv[1], "builtin") != 0) {
    FILE *m = fopen(argv[1], "r");
    if (!m) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    model = rnnoise_model_from_file(m);
    fclose(m);
    if (!model) { fprintf(stderr, "rnnoise_model_from_file refused %s\n", argv[1]); return 1; }
  }
  DenoiseState *st = rnnoise_create(model);
  FILE *in = fopen(argv[2], "rb"), *out = fopen(argv[3], "wb"), *vad = fopen(argv[4], "wb");
  float x[FRAME_SIZE];
  while (fread(x, sizeof(float), FRAME_SIZE, in) == FRAME_SIZE) {
    for (int i = 0; i < FRAME_SIZE; i++) x[i] *= 32768.f;
    float p = rnnoise_process_frame(st, x, x);
    for (int i = 0; i < FRAME_SIZE; i++) x[i] *= 1.f / 32768.f;
    fwrite(x, sizeof(float), FRAME_SIZE, out);
    fwrite(&p, sizeof(float), 1, vad);
  }
  rnnoise_destroy(st);
  if (model) rnnoise_model_free(model);
  fclose(in); fclose(out); fclose(vad);
  return 0;
}
"""


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def checkout(given, work):
    """A checkout of RNNoise v0.1.1: the one given, else a shallow clone of the tag into `work`."""
    if given:
        return os.path.abspath(given)
    target = os.path.join(work, "rnnoise")
    subprocess.run(["git", "-c", "advice.detachedHead=false", "clone", "--quiet", "--depth", "1", "--branch", TAG, REPOSITORY, target], check=True)
    return target


def commit_of(folder):
    try:
        return subprocess.run(["git", "-C", folder, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_model(source):
    """The classic model's layers from src/rnn_data.c: arrays by name, and each layer's sizes and activation."""
    with open(os.path.join(source, "src", "rnn_data.c"), encoding="utf-8") as f:
        text = f.read()
    arrays = {}
    for name, count, body in re.findall(r"static const rnn_weight (\w+)\[(\d+)\] = \{([^}]*)\};", text):
        values = [int(v) for v in body.replace("\n", " ").split(",") if v.strip()]
        if len(values) != int(count):
            sys.exit(f"{name}: {len(values)} values, declared {count}")
        if any(v < -128 or v > 127 for v in values):
            sys.exit(f"{name}: a weight outside a signed byte")
        arrays[name] = values
    layers = []
    for name, kind, parts in LAYERS:
        struct = "DenseLayer" if kind == "dense" else "GRULayer"
        m = re.search(r"static const " + struct + r" " + name + r" = \{([^}]*)\};", text)
        if not m:
            sys.exit(f"{name}: its {struct} is not in rnn_data.c")
        fields = [f.strip() for f in m.group(1).split(",")]
        inputs, neurons, activation = int(fields[-3]), int(fields[-2]), ACTIVATIONS[fields[-1]]
        gates = 3 if kind == "gru" else 1
        sizes = [inputs * neurons * gates] + ([neurons * neurons * gates] if kind == "gru" else []) + [neurons * gates]
        for part, size in zip(parts, sizes):
            if len(arrays[part]) != size:
                sys.exit(f"{part}: {len(arrays[part])} values, the layer needs {size}")
        layers.append({"name": name, "kind": kind, "inputs": inputs, "neurons": neurons, "activation": activation,
                       "values": [arrays[p] for p in parts]})
    return layers


def write_model(layers, path):
    """RNNoise's model file (what rnnoise_model_from_file reads): the header line, then each layer's numbers."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("rnnoise-nu model file version 1\n")
        for layer in layers:
            f.write(f"{layer['inputs']} {layer['neurons']} {layer['activation']}\n")
            for values in layer["values"]:
                for i in range(0, len(values), 32):
                    f.write(" ".join(str(v) for v in values[i:i + 32]) + "\n")


def make_input():
    """2 s at 48 kHz: silence, then speech-like syllables and a fricative over a noise floor (seeded)."""
    global np
    import numpy as np
    rng = np.random.default_rng(31)
    n = 2 * RATE
    t = np.arange(n) / RATE
    noise = rng.standard_normal(n)
    floor = np.convolve(noise, np.ones(8) / 8, mode="same") * 0.03          # low-passed: about -40 dBFS
    voice = np.zeros(n)
    for start, end, f0a, f0b in [(0.30, 0.70, 110, 150), (0.90, 1.30, 140, 170), (1.50, 1.85, 160, 115)]:
        a, b = int(start * RATE), int(end * RATE)
        u = np.linspace(0, 1, b - a, endpoint=False)
        f0 = f0a + (f0b - f0a) * u
        phase = 2 * np.pi * np.cumsum(f0) / RATE
        tone = np.zeros(b - a)
        for k in range(1, 40):
            fk = k * f0
            shape = 1 / k + 1.5 * np.exp(-((fk - 700) / 250) ** 2) + 0.8 * np.exp(-((fk - 1800) / 400) ** 2)
            tone += np.where(fk < RATE / 2 - 1000, shape, 0) * np.sin(k * phase)
        envelope = np.sin(np.pi * u) ** 0.5
        voice[a:b] = tone * envelope
    voice *= 0.25 / np.max(np.abs(voice))                                      # about -12 dBFS at the peaks
    a, b = int(1.35 * RATE), int(1.45 * RATE)
    hiss = rng.standard_normal(b - a)
    hiss = hiss - np.convolve(hiss, np.ones(6) / 6, mode="same")             # mostly over 4 kHz
    voice[a:b] += 0.08 * hiss * np.sin(np.pi * np.linspace(0, 1, b - a))
    x = voice + floor
    x[: int(0.1 * RATE)] = 0                                                   # digital silence first
    q = np.clip(np.round(x * 32767), -32768, 32767) / 32768
    return q.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", nargs="?", help="the folder to write the reference into")
    parser.add_argument("--rnnoise", help="a checkout of xiph/rnnoise at v0.1.1 (default: cloned from GitHub)")
    parser.add_argument("--model-only", metavar="PATH", help="only write the model file to PATH (no compiler needed)")
    args = parser.parse_args()
    if not args.out and not args.model_only:
        parser.error("give the output folder, or --model-only PATH")

    work = tempfile.mkdtemp(prefix="rnnoise-")
    try:
        source = checkout(args.rnnoise, work)
        layers = parse_model(source)
        if args.model_only:
            write_model(layers, args.model_only)
            print(f"wrote {args.model_only} ({os.path.getsize(args.model_only)} bytes, RNNoise {TAG} {commit_of(source)[:12]})")
            return

        global np
        import numpy as np
        os.makedirs(args.out, exist_ok=True)
        model = os.path.join(args.out, "rnnoise.rnnn")
        write_model(layers, model)

        cc = os.environ.get("CC", "cc")
        flags = ["-O2", "-ffp-contract=off", "-std=gnu99"]
        driver = os.path.join(work, "driver.c")
        with open(driver, "w", encoding="ascii") as f:
            f.write('#include <string.h>\n' + DRIVER)
        sources = [os.path.join(source, "src", s) for s in
                   ["denoise.c", "rnn.c", "rnn_data.c", "rnn_reader.c", "pitch.c", "kiss_fft.c", "celt_lpc.c"]]
        program = os.path.join(work, "rnnoise-reference")
        subprocess.run([cc, *flags, "-I", os.path.join(source, "include"), "-I", os.path.join(source, "src"),
                        driver, *sources, "-o", program, "-lm"], check=True)
        version = subprocess.run([cc, "--version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]

        x = make_input()
        raw = os.path.join(work, "input.f32")
        x.tofile(raw)
        outputs = {}
        for which in ["file", "builtin"]:
            out, vad = os.path.join(work, f"out-{which}.f32"), os.path.join(work, f"vad-{which}.f32")
            subprocess.run([program, model if which == "file" else "builtin", raw, out, vad], check=True)
            outputs[which] = (np.fromfile(out, dtype=np.float32), np.fromfile(vad, dtype=np.float32))
        if outputs["file"][0].tobytes() != outputs["builtin"][0].tobytes() or outputs["file"][1].tobytes() != outputs["builtin"][1].tobytes():
            sys.exit("the model file's output differs from the built-in model's: the conversion is wrong")

        y, vad = outputs["file"]
        frames = len(x) // FRAME
        np.save(os.path.join(args.out, "input.npy"), x.reshape(1, -1))
        np.save(os.path.join(args.out, "output.npy"), y[FRAME:].reshape(1, -1))
        np.save(os.path.join(args.out, "vad.npy"), vad)
        files = ["rnnoise.rnnn", "input.npy", "output.npy", "vad.npy"]
        manifest = {
            "source": {"repository": REPOSITORY, "tag": TAG, "commit": commit_of(source)},
            "compiler": version,
            "flags": flags,
            "machine": platform.machine(),
            "numpy": np.__version__,
            "rate": RATE,
            "frame": FRAME,
            "frames": frames,
            "input_samples": int(len(x)),
            "output_samples": int(len(y) - FRAME),
            "scale": 32768,
            "layers": [{k: layer[k] for k in ("name", "kind", "inputs", "neurons", "activation")} for layer in layers],
            "vad_mean": float(np.mean(vad)),
            "files": {name: {"bytes": os.path.getsize(os.path.join(args.out, name)), "sha256": sha256(os.path.join(args.out, name))} for name in files},
        }
        with open(os.path.join(args.out, "rnnoise.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
        print(f"wrote {args.out}: {frames} frames, the model file equal to the built-in model, VAD mean {np.mean(vad):.3f}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
