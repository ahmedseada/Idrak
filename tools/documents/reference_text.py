"""A reference extractor's text for every PDF in a folder, for the `documents real:` test group (plan 28, D5): pdfminer.six
(pinned in requirements.txt, the reader the samples' reference text comes from) reads each file, and its text is written
as NAME.reference.txt, the pages in order separated by a form feed, at the PDF's path under the output folder (default
test-output/documents/real-reference, which the group reads unless IDRAK_DOCUMENTS_REAL_REFERENCE names another), or
beside each PDF with --beside. The group lists where Idrak's text differs from it, never fails on it: the reference fixes
what the text must say, never how fast Idrak may be.

    python -m pip install -r tools/documents/requirements.txt
    python -I tools/documents/reference_text.py FOLDER
    python -I tools/documents/reference_text.py FOLDER -o test-output/documents/real-reference --timeout 300

Each file is read in a child process with a time limit (and, on Linux, a memory limit), so a broken or hostile file costs
its limit at most: it is listed with its error and no text is written for it (an older one is removed). An encrypted
file's password is the first line of NAME.password.txt beside it (as the group reads it), or --password for every file
without one. The run prints each file as it is read, with the rate and the time left, and writes reference.json (each
file's pages, seconds and error) into the output folder.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True  # no __pycache__ left in the repository's tools folder
sys.path.insert(0, HERE)  # the docfixtures package beside this script (-I leaves the script's folder off the path)

ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_OUT = os.path.join(ROOT, "test-output", "documents", "real-reference")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", nargs="?", help="the folder of PDFs (searched with its subfolders for *.pdf)")
    ap.add_argument("-o", "--out", default=DEFAULT_OUT,
                    help="where NAME.reference.txt files and reference.json go (default: test-output/documents/real-reference)")
    ap.add_argument("--beside", action="store_true", help="write NAME.reference.txt beside each PDF instead (reference.json still goes to -o)")
    ap.add_argument("--password", default=None, help="the password for encrypted files without a NAME.password.txt")
    ap.add_argument("--timeout", type=float, default=120, help="seconds a file may take (default 120)")
    ap.add_argument("--memory-mb", type=int, default=4096, help="the most memory a file may take on Linux, in MiB (default 4096)")
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child is None and args.folder is None:
        ap.error("the folder of PDFs is required")
    return args


def pdfs(folder):
    found = []
    for dirpath, dirnames, names in os.walk(folder):
        dirnames.sort()
        found += [os.path.join(dirpath, n) for n in names if n.lower().endswith(".pdf")]
    return sorted(found, key=lambda p: os.path.relpath(p, folder).replace(os.sep, "/"))


def sidecar_password(path):
    sidecar = os.path.splitext(path)[0] + ".password.txt"
    if not os.path.exists(sidecar):
        return None
    with open(sidecar, encoding="utf-8-sig") as f:
        return f.readline().rstrip("\r\n")


def child(path, password, memory_mb):
    """Reads one file and prints its pages' text as JSON (run in the child process)."""
    if sys.platform != "win32":
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (memory_mb << 20, memory_mb << 20))
    from docfixtures.reference import pdfminer_text

    try:
        out = {"pages": pdfminer_text(path, password)}
    except MemoryError:
        out = {"error": "MemoryError"}
    except RecursionError:
        out = {"error": "RecursionError"}
    except Exception as e:  # noqa: BLE001 - recorded
        out = {"error": f"{type(e).__name__}: {e}"[:400]}
    sys.stdout.write(json.dumps(out))


def duration(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60} min {seconds % 60:02d} s" if seconds >= 60 else f"{seconds} s"


def main():
    args = parse_args()
    if args.child is not None:
        child(args.child, args.password, args.memory_mb)
        return 0

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        print(f"{folder} is not a folder", file=sys.stderr)
        return 2
    files = pdfs(folder)
    if not files:
        print(f"no .pdf file in {folder} or the folders under it", file=sys.stderr)
        return 2
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    print(f"reference text (pdfminer.six) for {len(files)} PDF files in {folder}"
          f" -> {'beside each file' if args.beside else out}", flush=True)
    report, started = [], time.monotonic()
    for i, path in enumerate(files, 1):
        relative = os.path.relpath(path, folder)
        target = (os.path.splitext(path)[0] if args.beside else os.path.join(out, os.path.splitext(relative)[0])) + ".reference.txt"
        password = sidecar_password(path)
        password = password if password is not None else args.password
        cmd = [sys.executable, "-I", os.path.abspath(__file__), "--child", path, "--memory-mb", str(args.memory_mb)]
        if password is not None:
            cmd += ["--password", password]
        begun = time.monotonic()
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=args.timeout)
            text = proc.stdout.decode("utf-8", errors="replace").strip()
            result = json.loads(text) if proc.returncode == 0 and text.startswith("{") else \
                {"error": f"the reader stopped (exit code {proc.returncode})"}
        except subprocess.TimeoutExpired:
            result = {"error": f"no answer within {args.timeout:g} s"}
        seconds = time.monotonic() - begun
        entry = {"file": relative.replace(os.sep, "/"), "seconds": round(seconds, 3)}
        if "pages" in result:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8", newline="") as f:
                f.write("\f".join(result["pages"]))
            entry["pages"] = len(result["pages"])
            entry["reference"] = target
            what = f"{len(result['pages'])} page{'' if len(result['pages']) == 1 else 's'}"
        else:
            if os.path.exists(target):
                os.remove(target)
            entry["error"] = result["error"]
            what = f"no text: {result['error']}"
        report.append(entry)
        spent = time.monotonic() - started
        rate = i / spent if spent > 0 else 0.0
        left = (len(files) - i) / rate if rate > 0 else 0.0
        print(f"[{i}/{len(files)}] {entry['file']}: {what} ({seconds:.2f} s; {rate:.2f} files/s, {duration(left)} left)", flush=True)

    with open(os.path.join(out, "reference.json"), "w", encoding="utf-8") as f:
        json.dump({"folder": folder, "tool": "pdfminer.six", "files": report}, f, indent=2, ensure_ascii=False)
        f.write("\n")
    failed = sum(1 for e in report if "error" in e)
    print(f"{len(report) - failed} files read, {failed} without text, in {duration(time.monotonic() - started)};"
          f" the report: {os.path.join(out, 'reference.json')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
