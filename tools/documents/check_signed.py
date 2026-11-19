"""Checks the signatures of PDF files Idrak signed (plan 28, D12) with the reference validator, pyHanko
(pyhanko-certvalidator): each signature validated with the test root alone as trusted (decision 16), at a fixed time,
nothing fetched (revocation soft-fail); a B-LT signature also with a validation context made from the file's DSS alone,
revocation required (hard-fail), at its timestamp's time. Each file's NAME.json (written by the tests beside NAME.pdf)
says what each signature must check as: valid, covering the whole file or not, timestamped, its revocation, its changes.

    python -I tools/documents/check_signed.py test-output/documents/signed -o test-output/documents/signed-check.json
    python -I tools/documents/check_signed.py test-output/documents/signed-encrypted -o test-output/documents/signed-encrypted-check.json --password reader-2026

A folder is searched for *.pdf (sorted). The PKI (pki.root.cer and the other certificates) is the folder the JSON names
(the samples' d12 folder), or --pki. The exit code is 0 when pyHanko finds every signature as its JSON says, 1 otherwise.
"""
import argparse
import json
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs", nargs="+", help="PDF files or folders (searched for *.pdf)")
    ap.add_argument("-o", "--out", required=True, help="the report (JSON)")
    ap.add_argument("--password", default=None, help="the password to open the files with")
    ap.add_argument("--pki", default=None, help="the test PKI's folder (default: the one each file's JSON names)")
    return ap.parse_args()


def files_of(inputs):
    out = []
    for p in inputs:
        if os.path.isdir(p):
            out += [os.path.join(p, n) for n in os.listdir(p) if n.lower().endswith(".pdf")]
        else:
            out.append(p)
    return sorted(out)


def check(path, password, pki_folder):
    from asn1crypto import x509 as ax509
    from docfixtures.samples_d12 import _level, _quiet_pyhanko

    _quiet_pyhanko()
    from pyhanko.pdf_utils.reader import PdfFileReader
    from pyhanko.sign.validation import DocumentSecurityStore, validate_pdf_signature
    from pyhanko_certvalidator import ValidationContext

    expected_path = os.path.splitext(path)[0] + ".json"
    with open(expected_path, encoding="utf-8") as fh:
        expected = json.load(fh)
    folder = pki_folder or expected["pki"]

    def cert(name):
        with open(os.path.join(folder, name), "rb") as fh:
            return ax509.Certificate.load(fh.read())

    root = cert("pki.root.cer")
    intermediate = cert("pki.intermediate.cer")
    moment = datetime.fromisoformat(expected["validationTime"])
    results, problems = [], []
    with open(path, "rb") as fh:
        reader = PdfFileReader(fh, strict=False)
        if reader.encrypted:
            reader.decrypt(password or "")
        embedded = reader.embedded_signatures
        if len(embedded) != len(expected["signatures"]):
            problems.append(f"{len(embedded)} signatures, {len(expected['signatures'])} expected")
        for want in expected["signatures"]:
            emb = next((e for e in embedded if e.field_name == want["field"]), None)
            if emb is None:
                problems.append(f"{want['field']}: not found")
                continue
            vc = ValidationContext(trust_roots=[root], other_certs=[intermediate], moment=moment, allow_fetching=False,
                                   revocation_mode="soft-fail")
            st = validate_pdf_signature(emb, signer_validation_context=vc, ts_validation_context=vc)
            got = {"field": emb.field_name, "intact": st.intact, "valid": st.valid, "trusted": st.trusted,
                   "coverage": st.coverage.name, "modificationLevel": _level(st.modification_level),
                   "docmdpOk": st.docmdp_ok, "bottomLine": st.bottom_line, "mechanism": st.pkcs7_signature_mechanism,
                   "digest": st.md_algorithm,
                   "timestamp": None if st.timestamp_validity is None else
                   {"intact": st.timestamp_validity.intact, "valid": st.timestamp_validity.valid,
                    "trusted": st.timestamp_validity.trusted, "time": st.timestamp_validity.timestamp.isoformat()}}
            if want["kind"] == "BaselineLt":
                dss = DocumentSecurityStore.read_dss(reader)
                ts_time = st.timestamp_validity.timestamp if st.timestamp_validity else moment
                lt_vc = dss.as_validation_context({"trust_roots": [root], "moment": ts_time, "allow_fetching": False,
                                                   "revocation_mode": "hard-fail"})
                try:
                    lt = validate_pdf_signature(emb, signer_validation_context=lt_vc, ts_validation_context=lt_vc)
                    got["lt"] = {"intact": lt.intact, "valid": lt.valid, "trusted": lt.trusted, "time": ts_time.isoformat()}
                except Exception as e:  # noqa: BLE001 - recorded
                    got["lt"] = {"error": f"{type(e).__name__}: {str(e)[:200]}", "trusted": False}
            results.append(got)
            differs = []
            if (got["intact"] and got["valid"] and got["trusted"]) != want["valid"]:
                differs.append("valid")
            if (got["coverage"] == "ENTIRE_FILE") != want["coversWholeFile"]:
                differs.append("coversWholeFile")
            if want["timestamp"] != bool(got["timestamp"] and got["timestamp"]["valid"] and got["timestamp"]["trusted"]):
                differs.append("timestamp")
            if (want["changes"] != "NotAllowed") != bool(got["docmdpOk"]):
                differs.append("changes")
            if want["revocation"] == "good" and want["kind"] == "BaselineLt" and not got["lt"].get("trusted"):
                differs.append("revocation")
            if differs:
                problems.append({"field": want["field"], "differs": differs})
    return {"file": path, "signatures": results, "problems": problems}


def main():
    args = parse_args()
    files = files_of(args.inputs)
    if not files:
        sys.exit("no PDF file given or found")
    report = []
    failed = 0
    for path in files:
        try:
            result = check(path, args.password, args.pki)
        except Exception as e:  # noqa: BLE001 - recorded
            result = {"file": path, "problems": [f"{type(e).__name__}: {str(e)[:300]}"]}
        report.append(result)
        verdict = "ok" if not result["problems"] else f"DIFFERS: {result['problems']}"
        failed += bool(result["problems"])
        sigs = ", ".join(f"{s['field']} intact={s['intact']} valid={s['valid']} trusted={s['trusted']} "
                         f"coverage={s['coverage']} level={s['modificationLevel']}"
                         + (f" timestamp={s['timestamp']['valid'] and s['timestamp']['trusted']}" if s.get("timestamp") else "")
                         + (f" lt-trusted={s['lt'].get('trusted')}" if s.get("lt") else "")
                         for s in result.get("signatures", []))
        print(f"{os.path.basename(path)}: {verdict} [{sigs}]", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"tool": "pyHanko (pyhanko-certvalidator), trust: pki.root.cer only, no fetching; B-LT also with the "
                           "DSS alone, revocation required, at the timestamp's time", "files": report}, fh, indent=1)
    print(f"{len(files)} files, {failed} differ", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
