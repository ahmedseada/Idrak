"""The outside reference for data-parallel training (plan 23, phase C0).

    python -m venv ddp && ddp/bin/pip install torch transformers tokenizers safetensors numpy jinja2
    ddp/bin/python tools/cluster/make_tiny_ddp.py tests/Idrak.Tests/data/cluster

Nothing is downloaded and nothing under tests/Idrak.Tests/data/pretrain/ is written: the script reads plan 19's tiny model
(pretrain/tiny-llama/) and token file (pretrain/tokens.bin) and trains on the first eight 32-token windows (window i is
ids [32i, 32i + 33)), every parameter trained, the loss the mean token cross entropy over every position. It writes:

- ddp/: PyTorch DistributedDataParallel, gloo backend, two CPU processes on 127.0.0.1, rank r taking windows r, r+2, r+4,
  r+6; per rank per step the rank's own loss and the two ranks' mean; three steps of plain SGD at 0.2 (the averaged
  gradients every rank holds after step 1's backward, before the update, and the weights after step 3), of AdamW and of
  plan 19's AdamW with the bias corrections folded into the step size (the weights and the moments after step 3);
- single/: the same three steps in one process on all eight windows at once (batch 8); the script asserts single/ and
  ddp/ agree within 1e-5 (losses, gradients, weights, moments) and prints the largest differences;
- bf16/: the SGD run of ddp/ again with each rank's gradients rounded to bfloat16 before the all-reduce and the sum done
  in float32 (a DDP communication hook), and the largest differences against ddp/, the tolerance a bf16 wire needs.

Both ranks are asserted to hold the same gradients and weights bit for bit. Reruns write the same bytes (fixed seeds, one
thread per process, deterministic algorithms, sorted JSON) on the same machine and library versions; manifest.json records
the versions and each file's SHA-256. README.md in the output folder (written by hand) says how the C# tests read it.
"""
import argparse
import json
import os
import shutil
import socket
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "vlm"))
from make_tiny import dump_json, sha256, safetensors_index  # noqa: E402
from make_tiny_pretrain import ADAMW, SGD_RATE, STEPS, WINDOW, FoldedAdamW  # noqa: E402

WORLD = 2
WINDOWS = 8
TOLERANCE = 1e-5
RUNS = ("sgd", "adamw", "folded", "bf16")
FOLDERS = {"sgd": "", "adamw": "adamw", "folded": "adamw-folded"}   # under ddp/ and single/; bf16 is its own folder
SAFETENSORS_META = {"format": "pt"}


def deterministic():
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def windows_of(tokens):
    return np.stack([tokens[i * WINDOW: i * WINDOW + WINDOW + 1] for i in range(WINDOWS)]).astype(np.int64)


def load_model(model_dir):
    import torch
    from transformers import LlamaForCausalLM
    from transformers.utils import logging
    logging.disable_progress_bar()
    return LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).train()


def make_optimizer(kind, params):
    import torch
    if kind in ("sgd", "bf16"):
        return torch.optim.SGD(params, lr=SGD_RATE, momentum=0.0)
    if kind == "adamw":
        return torch.optim.AdamW(params, **ADAMW)
    return FoldedAdamW(params, **ADAMW)


def save(tensors, path):
    from safetensors.torch import save_file
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file({k: v.detach().clone().contiguous() for k, v in sorted(tensors.items())}, path, metadata=SAFETENSORS_META)


def train(model, inner, rows, kind, out):
    """STEPS steps on `rows` (windows of WINDOW + 1 ids), every parameter trained; the loss before each update. `model`
    is what runs forward (the DDP wrapper or the model), `inner` the model whose parameters are named."""
    import torch
    x = torch.from_numpy(rows[:, :-1])
    y = torch.from_numpy(rows[:, 1:])
    named = list(inner.named_parameters())
    opt = make_optimizer(kind, [p for _, p in named])
    losses = []
    for step in range(1, STEPS + 1):
        opt.zero_grad(set_to_none=True)
        logits = model(input_ids=x).logits
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss.backward()
        losses.append(loss.item())
        if step == 1 and kind in ("sgd", "bf16"):
            save({n: p.grad for n, p in named}, os.path.join(out, "grads-step1.safetensors"))
        opt.step()
    save({n: p for n, p in named}, os.path.join(out, f"weights-step{STEPS}.safetensors"))
    if kind == "adamw":
        state = {}
        for n, p in named:
            state[n + ".exp_avg"] = opt.state[p]["exp_avg"]
            state[n + ".exp_avg_sq"] = opt.state[p]["exp_avg_sq"]
        save(state, os.path.join(out, f"state-step{STEPS}.safetensors"))
    elif kind == "folded":
        state = {}
        for (n, _), m, v in zip(named, opt.m, opt.v):
            state[n + ".exp_avg"] = m
            state[n + ".exp_avg_sq"] = v
        save(state, os.path.join(out, f"state-step{STEPS}.safetensors"))   # compared, not published
    return losses


def bf16_wire(state, bucket):
    """Each rank's gradient bucket rounded to bfloat16 (what a bf16 wire carries), summed in float32, then averaged."""
    import torch
    import torch.distributed as dist
    wire = bucket.buffer().to(torch.bfloat16).to(torch.float32)
    future = dist.all_reduce(wire, op=dist.ReduceOp.SUM, async_op=True).get_future()
    return future.then(lambda f: f.value()[0].div_(dist.get_world_size()))


def rank_main(rank, model_dir, tokens_path, scratch, port):
    """One DDP process: every run in RUNS on windows rank, rank + WORLD, ...; files and losses under scratch/rank{r}."""
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    deterministic()
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    rows = windows_of(np.fromfile(tokens_path, dtype="<u2"))[rank::WORLD]
    losses = {}
    for kind in RUNS:
        model = load_model(model_dir)
        ddp = DistributedDataParallel(model)
        if kind == "bf16":
            ddp.register_comm_hook(None, bf16_wire)
        losses[kind] = train(ddp, model, rows, kind, os.path.join(scratch, f"rank{rank}", kind))
        dist.barrier()
    dump_json(os.path.join(scratch, f"rank{rank}", "losses.json"), losses)
    dist.barrier()
    dist.destroy_process_group()


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def load(path):
    from safetensors.torch import load_file
    return load_file(path)


def largest(a, b):
    """The largest absolute difference over every tensor, the tensor it is in, and the largest magnitude in a."""
    assert sorted(a) == sorted(b), "tensor names differ"
    worst, where, scale = 0.0, None, 0.0
    for name in sorted(a):
        assert a[name].shape == b[name].shape, f"{name}: shapes differ"
        d = float((a[name].double() - b[name].double()).abs().max())
        scale = max(scale, float(a[name].double().abs().max()))
        if d > worst or where is None:
            worst, where = d, name
    return worst, where, scale


def tensor_files(folder):
    return sorted(n for n in os.listdir(folder) if n.endswith(".safetensors"))


def same_on_every_rank(scratch, kind):
    """The ranks' gradient, weight and moment files are equal bit for bit; returns rank 0's folder."""
    first = os.path.join(scratch, "rank0", kind)
    for r in range(1, WORLD):
        other = os.path.join(scratch, f"rank{r}", kind)
        assert tensor_files(first) == tensor_files(other)
        for name in tensor_files(first):
            a, b = load(os.path.join(first, name)), load(os.path.join(other, name))
            for t in sorted(a):
                assert a[t].dtype == b[t].dtype and a[t].shape == b[t].shape and bool((a[t] == b[t]).all()), \
                    f"{kind}/{name}: rank 0 and rank {r} hold different {t}"
    return first


def compare(label, folder_a, folder_b, losses_a, losses_b, limit):
    """Largest differences between two runs' losses and tensor files; asserted within limit when one is given."""
    found = {"losses": max(abs(a - b) for a, b in zip(losses_a, losses_b))}
    print(f"  {label}: losses {found['losses']:.3e}")
    for name in tensor_files(folder_a):
        d, where, scale = largest(load(os.path.join(folder_a, name)), load(os.path.join(folder_b, name)))
        key = name.replace(".safetensors", "")
        found[key] = d
        found[key + "_largest_magnitude"] = scale
        print(f"  {label}: {name} {d:.3e} (in {where}; largest magnitude {scale:.3e}, {d / scale:.3e} of it)")
    if limit is not None:
        for key, d in found.items():
            if not key.endswith("_largest_magnitude"):
                assert d <= limit, f"{label}: {key} differ by {d:.3e} > {limit:g}"
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--pretrain", default=os.path.join(HERE, "..", "..", "tests", "Idrak.Tests", "data", "pretrain"),
                        help="plan 19's fixture folder (read only): tiny-llama/ and tokens.bin")
    args = parser.parse_args()
    import torch
    import torch.multiprocessing as mp
    import transformers
    import safetensors
    deterministic()

    pretrain = os.path.abspath(args.pretrain)
    model_dir = os.path.join(pretrain, "tiny-llama")
    tokens_path = os.path.join(pretrain, "tokens.bin")
    out = os.path.abspath(args.out)
    assert os.path.commonpath([out, pretrain]) != pretrain, "the output must not be inside plan 19's fixture"
    readme = os.path.join(out, "README.md")
    kept = open(readme, encoding="utf-8").read() if os.path.exists(readme) else None   # written by hand, kept
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    if kept is not None:
        with open(readme, "w", encoding="utf-8", newline="\n") as f:
            f.write(kept)

    windows = windows_of(np.fromfile(tokens_path, dtype="<u2"))
    scratch = tempfile.mkdtemp(prefix="idrak-ddp-")
    try:
        print(f"ddp: {WORLD} gloo processes, runs {', '.join(RUNS)}")
        mp.spawn(rank_main, args=(model_dir, tokens_path, scratch, free_port()), nprocs=WORLD, join=True)
        rank_losses = [json.load(open(os.path.join(scratch, f"rank{r}", "losses.json"))) for r in range(WORLD)]
        for kind in RUNS:
            source = same_on_every_rank(scratch, kind)
            target = os.path.join(out, "bf16") if kind == "bf16" else os.path.join(out, "ddp", FOLDERS[kind])
            os.makedirs(target, exist_ok=True)
            for name in tensor_files(source):
                if kind == "folded" and name.startswith("state-"):
                    continue
                shutil.copyfile(os.path.join(source, name), os.path.join(target, name))

        print("single: one process, batch 8")
        single_losses = {}
        for kind in ("sgd", "adamw", "folded"):
            model = load_model(model_dir)
            single_losses[kind] = train(model, model, windows, kind, os.path.join(scratch, "single", kind))
            target = os.path.join(out, "single", FOLDERS[kind])
            os.makedirs(target, exist_ok=True)
            for name in tensor_files(os.path.join(scratch, "single", kind)):
                if kind == "folded" and name.startswith("state-"):
                    continue
                shutil.copyfile(os.path.join(scratch, "single", kind, name), os.path.join(target, name))

        def mean_losses(kind):
            return [sum(rank_losses[r][kind][s] for r in range(WORLD)) / WORLD for s in range(STEPS)]

        print(f"ddp against single (asserted within {TOLERANCE:g}):")
        vs_single = {kind: compare(f"ddp {kind}", os.path.join(scratch, "rank0", kind),
                                   os.path.join(scratch, "single", kind), mean_losses(kind), single_losses[kind],
                                   TOLERANCE)
                     for kind in ("sgd", "adamw", "folded")}
        print("bf16 wire against ddp (measured):")
        vs_ddp = compare("bf16", os.path.join(scratch, "rank0", "bf16"), os.path.join(scratch, "rank0", "sgd"),
                         mean_losses("bf16"), mean_losses("sgd"), None)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    def runs(kind):
        return {"rank_losses": [rank_losses[r][kind] for r in range(WORLD)], "losses": mean_losses(kind)}

    settings = {
        "window": WINDOW, "windows": WINDOWS, "window_ids": windows.tolist(), "steps": STEPS,
        "loss": "mean cross entropy over every position of every window (targets: the next token)",
        "sgd": {"learning_rate": SGD_RATE, "momentum": 0.0, "weight_decay": 0.0, "max_gradient_norm": 0.0},
        "adamw": {**{k: (list(v) if isinstance(v, tuple) else v) for k, v in ADAMW.items()}, "max_gradient_norm": 0.0},
        "adamw_folded": "AdamW with the bias corrections folded into the step size, as Idrak computes it (plan 19)",
    }
    files = {"sgd": {"grads_step1": "grads-step1.safetensors", "weights_after": f"weights-step{STEPS}.safetensors"},
             "adamw": {"weights_after": f"adamw/weights-step{STEPS}.safetensors",
                       "state_after": f"adamw/state-step{STEPS}.safetensors"},
             "adamw_folded": {"weights_after": f"adamw-folded/weights-step{STEPS}.safetensors"}}
    dump_json(os.path.join(out, "ddp", "ddp.json"), {
        **settings, "world_size": WORLD, "backend": "gloo", "rank_windows": [list(range(r, WINDOWS, WORLD)) for r in range(WORLD)],
        "gradients": "DistributedDataParallel's average: each rank's gradient of its own mean loss, summed and divided by the world size",
        "losses": "losses: the ranks' mean before each update (the global mean, every rank has as many positions); rank_losses[r]: rank r's own",
        "sgd": {**settings["sgd"], **runs("sgd"), "files": files["sgd"]},
        "adamw": {**settings["adamw"], **runs("adamw"), "files": files["adamw"]},
        "adamw_folded": {**runs("folded"), "what": settings["adamw_folded"], "files": files["adamw_folded"]},
        "vs_single": {"tolerance": TOLERANCE, "sgd": vs_single["sgd"], "adamw": vs_single["adamw"],
                      "adamw_folded": vs_single["folded"]},
    })
    dump_json(os.path.join(out, "single", "single.json"), {
        **settings, "batch": WINDOWS,
        "sgd": {**settings["sgd"], "losses": single_losses["sgd"], "files": files["sgd"]},
        "adamw": {**settings["adamw"], "losses": single_losses["adamw"], "files": files["adamw"]},
        "adamw_folded": {"what": settings["adamw_folded"], "losses": single_losses["folded"], "files": files["adamw_folded"]},
    })
    dump_json(os.path.join(out, "bf16", "bf16.json"), {
        **{k: settings[k] for k in ("window", "windows", "window_ids", "steps", "loss")}, "world_size": WORLD,
        "backend": "gloo", "rank_windows": [list(range(r, WINDOWS, WORLD)) for r in range(WORLD)],
        "wire": "each rank's gradient rounded to bfloat16 (round to nearest even), summed in float32, divided by the world size",
        "sgd": {**settings["sgd"], **runs("bf16"), "files": files["sgd"]},
        "vs_ddp": vs_ddp,
    })

    files_index, tensors = {}, {}
    for root, _, names in sorted(os.walk(out)):
        for name in sorted(names):
            path = os.path.join(root, name)
            rel = os.path.relpath(path, out).replace(os.sep, "/")
            if rel in ("manifest.json", "README.md"):
                continue
            files_index[rel] = {"bytes": os.path.getsize(path), "sha256": sha256(path)}
            if name.endswith(".safetensors"):
                tensors[rel] = safetensors_index(path)[0]
    dump_json(os.path.join(out, "manifest.json"), {
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
                     "safetensors": safetensors.__version__, "numpy": np.__version__},
        "sources": {"model": "../pretrain/tiny-llama/", "tokens": "../pretrain/tokens.bin",
                    "model_sha256": sha256(os.path.join(model_dir, "model.safetensors")), "tokens_sha256": sha256(tokens_path)},
        "summary": {"ddp_sgd_losses": mean_losses("sgd"), "single_sgd_losses": single_losses["sgd"],
                    "ddp_adamw_losses": mean_losses("adamw"), "bf16_sgd_losses": mean_losses("bf16")},
        "tensors": tensors, "files": files_index,
    })
    print(f"wrote {out}: SGD losses ddp {mean_losses('sgd')} single {single_losses['sgd']}")


if __name__ == "__main__":
    main()
