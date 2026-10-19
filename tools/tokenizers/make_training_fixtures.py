"""The references for plan 24's trainers (phase T4): what sentencepiece and tokenizers learn from the test corpus, for the
C# trainers to be held to.

    tok/bin/python -I tools/tokenizers/make_training_fixtures.py            # writes tests/Idrak.Tests/data/tokenizer-training/
    tok/bin/python -I tools/tokenizers/make_training_fixtures.py --runs 5   # also: how much Hugging Face's WordPiece runs agree
    tok/bin/python -I tools/tokenizers/make_training_fixtures.py --part unigram --part normalization   # only those files

The training text is data/tokenizers/corpus.txt and data/pretrain/corpus.txt, read as the C# trainers read them (each
line without its "\n", a "\r" before it dropped, as LineEnds "lf" reads it). Files written (the folder is emptied first,
README.md kept):

- unigram.json: per configuration, the options and the pieces (piece, score, type) sentencepiece's unigram trainer learns
  (one thread, no shuffling, the identity normalization unless the configuration names a built-in rule, the Arabic marks
  removed by a user rule where the configuration says so, no line skipped for length unless it gives
  max_sentence_length). The C# trainer is held to these by overlap of the learned (normal) pieces, above a threshold
  measured here and recorded in plans/24-tokenizers.md: the two sum in different orders, so equal is not expected. Some
  configurations also hold the ids sentencepiece's own model gives each line of data/tokenizers/corpus.txt
  ("encodings"), for the C# reader to give from the same pieces and switches.
- whitespace-corpus.txt: made here (a fixed seed): lines of code whose indentation is runs of the literal ▁, the text
  the allow_whitespace_only_pieces configuration learns from (the test corpus has no run of ▁ for it to change).
- normalization.json: what sentencepiece's built-in rules "nfkc" and "nmt_nfkc" make of every code point (U+0001 to
  U+10FFFF but the surrogates; only those they change are written, nmt_nfkc's where it differs from nfkc), of a sample of
  combining sequences (keys of the rules' own maps, base letters with marks in every order of two, Hangul jamo) and of
  every line of the training text (only the lines they change), the charsmap alone (no whitespace handling). Also the
  code points whose Unicode 18 NFKC (from tests/Idrak.Tests/data/unicode/normalization-test.txt.gz, the library's
  tables' version) differs from what sentencepiece's older tables make of them, with both outputs.
- wordpiece.json: the words BERT's normalizer and pre-tokenizer make of the text, then the vocabulary of Hugging Face's
  WordPieceTrainer algorithm (BPE merges with the ## prefix) ported here with its ties in a fixed order (the ##
  characters numbered in code-point order after the alphabet, the words taken in code-point order, the queue as .NET's
  PriorityQueue keeps it), which the C# "frequency" trainer must learn exactly; and the vocabulary tokenizers' own
  WordPieceTrainer learned in this run (it numbers the ## characters in hash order, so another run learns another one:
  the C# trainer is held to it by overlap).
- manifest.json: the library versions and each file's size and SHA-256.

Everything but wordpiece.json's "huggingface" vocabulary is the same on every rerun with the same library versions.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sentencepiece as spm  # noqa: E402
from sentencepiece import sentencepiece_model_pb2 as model_pb2  # noqa: E402
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers  # noqa: E402
import tokenizers  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "tests", "Idrak.Tests", "data")
OUT = os.path.join(DATA, "tokenizer-training")
FILES = ["tokenizers/corpus.txt", "pretrain/corpus.txt"]
SEED = 20261019

# Tatweel and the Arabic diacritics the trainers' Arabic option removes (src/Idrak/Models/Tokenizers/UnigramTrainer.cs,
# ArabicMarks), as a sentencepiece normalization rule TSV: source code point, tab, nothing.
DIACRITICS = [(0x0610, 0x061A), (0x064B, 0x065F), (0x0670, 0x0670), (0x06D6, 0x06DC), (0x06DF, 0x06E4), (0x06E7, 0x06E8),
              (0x06EA, 0x06ED)]
ARABIC_RULES = ["640\t"] + [f"{c:X}\t" for first, last in DIACRITICS for c in range(first, last + 1)]

UNIGRAM = {
    "unigram": dict(vocab_size=1000),
    "unigram-bytes": dict(vocab_size=1000, byte_fallback=True, split_digits=True),
    "unigram-arabic": dict(vocab_size=800, arabic=True, max_sentencepiece_length=8, shrinking_factor=0.8, num_sub_iterations=3),
    # SentencePiece's switches, each against its default (plan 24, what is left, item 3); the whitespace and script
    # switches at 2000, where they change the pieces of this text (at 800 they barely do; split_by_number changes none).
    "unigram-no-whitespace-split": dict(vocab_size=2000, split_by_whitespace=False),
    "unigram-no-script-split": dict(vocab_size=2000, split_by_unicode_script=False),
    "unigram-no-number-split": dict(vocab_size=800, split_by_number=False),
    "unigram-suffix": dict(vocab_size=800, treat_whitespace_as_suffix=True, encodings=True),
    "unigram-whitespace-pieces": dict(vocab_size=300, allow_whitespace_only_pieces=True, corpus="whitespace"),
    "unigram-max-sentence": dict(vocab_size=800, max_sentence_length=200),
    "unigram-nmt-nfkc": dict(vocab_size=800, normalization="nmt_nfkc", encodings=True),
    "unigram-nfkc": dict(vocab_size=800, normalization="nfkc"),
}
NORMALIZATION_RULES = ["nfkc", "nmt_nfkc"]
UNICODE_TEST = os.path.join(DATA, "unicode", "normalization-test.txt.gz")
WORDPIECE = dict(vocab_size=2000, min_frequency=2, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"], lowercase=True)


def lines():
    """Every line of the training files as the C# trainers read them (LineEnds "lf")."""
    return [line for name in FILES for line in corpus_lines(name)]


def corpus_lines(name):
    """The lines of one file under data/, as the C# trainers read them (LineEnds "lf")."""
    out = []
    with open(os.path.join(DATA, name), encoding="utf-8", newline="") as f:
        text = f.read()
    parts = text.split("\n")
    for i, line in enumerate(parts):
        if i == len(parts) - 1 and line == "":
            break
        if i < len(parts) - 1 and line.endswith("\r"):
            line = line[:-1]
        out.append(line)
    return out


def whitespace_corpus():
    """Lines of code indented by runs of the literal ▁ (2, 4 or 8), from a fixed seed: identity normalization keeps them,
    so the allow_whitespace_only_pieces switch has runs of ▁ to learn."""
    import random
    rng = random.Random(SEED)
    names = ["value", "total", "items", "index", "count", "result", "name", "left", "right", "node", "text", "size"]
    calls = ["print", "append", "update", "len", "range", "sorted", "sum", "max", "min", "open"]
    out = []
    for _ in range(600):
        depth = rng.choice([1, 1, 2, 2, 3, 4])
        a, b = rng.choice(names), rng.choice(names)
        line = rng.choice([f"{a} = {rng.choice(calls)}({b})", f"if {a} > {b}:", f"for {a} in {rng.choice(calls)}({b}):",
                           f"return {a} + {b}", f"{a}.{rng.choice(calls)}({b}, {rng.randint(0, 99)})", f"while {a}:", "pass"])
        out.append("\u2581" * (2 * depth if depth < 4 else 8) + line)
    return out


def write_json(path, value):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------------------------------------------------
# Unigram: sentencepiece's trainer.
# ---------------------------------------------------------------------------------------------------------------------
def train_unigram(name, settings, text, work):
    settings = dict(settings)
    encodings = settings.pop("encodings", False)
    corpus = settings.pop("corpus", None)
    normalization = settings.pop("normalization", "identity")
    extra = {}
    if settings.pop("arabic", False):
        rules = os.path.join(work, "arabic-removals.tsv")
        with open(rules, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(ARABIC_RULES) + "\n")
        extra["normalization_rule_tsv"] = rules
    else:
        extra["normalization_rule_name"] = normalization
    if corpus == "whitespace":
        text = whitespace_corpus()
    spm.set_random_generator_seed(SEED)
    import io
    writer = io.BytesIO()
    spm.SentencePieceTrainer.train(sentence_iterator=iter(text), model_writer=writer, num_threads=1, minloglevel=2,
                                   model_type="unigram", shuffle_input_sentence=False, hard_vocab_limit=False,
                                   character_coverage=0.9995, **dict(dict(max_sentence_length=1 << 20), **settings), **extra)
    proto = model_pb2.ModelProto()
    proto.ParseFromString(writer.getvalue())
    options = dict(settings, character_coverage=0.9995, special_tokens=["<unk>", "<s>", "</s>"],
                   arabic=["tatweel", "diacritics"] if extra.get("normalization_rule_tsv") else [])
    if normalization != "identity":
        options["normalization"] = normalization
    if corpus:
        options["corpus"] = corpus
    pieces = [[p.piece, p.score, int(p.type)] for p in proto.pieces]
    print(f"{name}: {len(pieces)} pieces", file=sys.stderr)
    entry = {"options": options, "pieces": pieces}
    if encodings:
        model = spm.SentencePieceProcessor(model_proto=writer.getvalue())
        entry["encodings"] = [model.encode(line) for line in corpus_lines("tokenizers/corpus.txt")]
    return entry


# ---------------------------------------------------------------------------------------------------------------------
# WordPiece: the words, Hugging Face's algorithm with fixed ties, and Hugging Face's own run.
# ---------------------------------------------------------------------------------------------------------------------
def bert_words(text, lowercase):
    normalizer = normalizers.BertNormalizer(lowercase=lowercase)
    pre = pre_tokenizers.BertPreTokenizer()
    counts = {}
    for line in text:
        if not line:
            continue
        for word, _ in pre.pre_tokenize_str(normalizer.normalize_str(line)):
            counts[word] = counts.get(word, 0) + 1
    return counts


class DotNetQueue:
    """System.Collections.Generic.PriorityQueue (a 4-ary min-heap), so ties pop in .NET's order."""

    def __init__(self, compare):
        self.nodes = []
        self.compare = compare

    def __len__(self):
        return len(self.nodes)

    def enqueue(self, element, priority):
        self.nodes.append(None)
        i = len(self.nodes) - 1
        while i > 0:
            parent = (i - 1) >> 2
            if self.compare(priority, self.nodes[parent][1]) < 0:
                self.nodes[i] = self.nodes[parent]
                i = parent
            else:
                break
        self.nodes[i] = (element, priority)

    def dequeue(self):
        root = self.nodes[0]
        last = self.nodes.pop()
        if self.nodes:
            self._down(last, 0)
        return root

    def _down(self, node, i):
        size = len(self.nodes)
        while True:
            first = (i << 2) + 1
            if first >= size:
                break
            child, child_index = self.nodes[first], first
            for j in range(first + 1, min(first + 4, size)):
                if self.compare(self.nodes[j][1], child[1]) < 0:
                    child, child_index = self.nodes[j], j
            if self.compare(node[1], child[1]) <= 0:
                break
            self.nodes[i] = child
            i = child_index
        self.nodes[i] = node


def merge_word(symbols, length, left, right, merged, changes):
    write = read = 0
    while read < length:
        if symbols[read] == left and read + 1 < length and symbols[read + 1] == right:
            if write > 0:
                changes.append(((symbols[write - 1], left), -1))
                changes.append(((symbols[write - 1], merged), 1))
            symbols[write] = merged
            if read + 2 < length:
                changes.append(((right, symbols[read + 2]), -1))
                changes.append(((merged, symbols[read + 2]), 1))
            write += 1
            read += 2
        else:
            symbols[write] = symbols[read]
            write += 1
            read += 1
    return write


def wordpiece_fixed(word_counts, specials, vocab_size, min_frequency, prefix="##"):
    """tokenizers' BpeTrainer.do_train with a continuing-subword prefix (what its WordPieceTrainer runs), its bookkeeping as
    it is (a pair queued twice can merge twice), with the ties fixed: the words in code-point order, the ## characters
    numbered after the alphabet in code-point order, the queue as .NET's."""
    ids, tokens = {}, []

    def add(token):
        if token not in ids:
            ids[token] = len(tokens)
            tokens.append(token)

    for special in specials:
        add(special)
    ordered = sorted(word_counts.items())
    alphabet = {}
    for word, count in ordered:
        for ch in word:
            alphabet[ord(ch)] = alphabet.get(ord(ch), 0) + count
    for code in sorted(alphabet):
        add(chr(code))
    continuing = sorted({ord(ch) for word, _ in ordered for i, ch in enumerate(word) if i > 0 and ch in ids})
    for code in continuing:
        add(prefix + chr(code))
    words, counts = [], []
    for word, count in ordered:
        symbols = []
        for i, ch in enumerate(word):
            if ch in ids:
                symbols.append(ids[prefix + ch] if i > 0 else ids[ch])
        words.append(symbols)
        counts.append(count)
    lengths = [len(w) for w in words]
    pair_counts, where = {}, {}
    for i, symbols in enumerate(words):
        for k in range(lengths[i] - 1):
            pair = (symbols[k], symbols[k + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + counts[i]
            where.setdefault(pair, {})[i] = None

    def compare(x, y):
        if x[0] != y[0]:
            return (y[0] > x[0]) - (y[0] < x[0])
        return (x[1] > y[1]) - (x[1] < y[1])

    queue = DotNetQueue(compare)

    def push_all():
        for pair, positions in where.items():
            count = pair_counts[pair]
            if count > 0:
                queue.enqueue((pair, positions), (count, pair))
        where.clear()

    push_all()
    while len(ids) < vocab_size and len(queue):
        (pair, positions), (count, _) = queue.dequeue()
        current = pair_counts[pair]
        if count != current:
            queue.enqueue((pair, positions), (current, pair))
            continue
        if current < 1 or min_frequency > current:
            break
        left, right = pair
        right_part = tokens[right][len(prefix):] if tokens[right].startswith(prefix) else tokens[right]
        merged = tokens[left] + right_part
        add(merged)
        merged_id = ids[merged]
        for i in positions:
            changes = []
            lengths[i] = merge_word(words[i], lengths[i], left, right, merged_id, changes)
            for changed, change in changes:
                pair_counts[changed] = pair_counts.get(changed, 0) + change * counts[i]
                if change > 0:
                    where.setdefault(changed, {})[i] = None
        push_all()
    return [token for token, _ in sorted(ids.items(), key=lambda kv: kv[1])]


def wordpiece_huggingface(text):
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=WORDPIECE["lowercase"])
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(vocab_size=WORDPIECE["vocab_size"], min_frequency=WORDPIECE["min_frequency"],
                                        special_tokens=WORDPIECE["special_tokens"], show_progress=False)
    tokenizer.train_from_iterator([line for line in text if line], trainer)
    vocab = tokenizer.get_vocab()
    return [token for token, _ in sorted(vocab.items(), key=lambda kv: kv[1])]


# ---------------------------------------------------------------------------------------------------------------------
# Normalization: sentencepiece's built-in rules, the charsmap alone.
# ---------------------------------------------------------------------------------------------------------------------
def is_hangul(c):
    return 0x1100 <= c <= 0x11FF or 0x3130 <= c <= 0x318F or 0xFFA0 <= c <= 0xFFDC or 0xAC00 <= c <= 0xD7A3 \
        or 0xA960 <= c <= 0xA97F or 0xD7B0 <= c <= 0xD7FF


def code_points():
    return [c for c in range(1, 0x110000) if not 0xD800 <= c <= 0xDFFF]


def unicode_nfkc():
    """Each code point's NFKC by the Unicode version of the library's tables (part 1 of its NormalizationTest: every
    character with a decomposition), where it changes."""
    import gzip
    out, part = {}, None
    with gzip.open(UNICODE_TEST, "rt", encoding="utf-8") as f:
        for row in f:
            row = row.rstrip("\n")
            if row.startswith("#") or not row:
                continue
            if row.startswith("@"):
                part = row
                continue
            if part == "@Part1":
                fields = row.split(";")
                source = [int(x, 16) for x in fields[0].split()]
                nfkc = "".join(chr(int(x, 16)) for x in fields[3].split())
                if len(source) == 1 and nfkc != chr(source[0]):
                    out[source[0]] = nfkc
    return out


def sequences(normalizer):
    """Combining sequences: a sample of the rule's own keys (the Hangul ones apart), base letters with every ordered pair
    of ten marks, and Hangul jamo (conjoining, compatibility and halfwidth) in syllable order."""
    import random
    rng = random.Random(SEED)
    keys = sorted({k for k, _ in normalizer.decompile() if len(k) >= 2})
    plain = [k for k in keys if not any(is_hangul(ord(c)) for c in k)]
    hangul = [k for k in keys if any(is_hangul(ord(c)) for c in k)]
    out = rng.sample(plain, 1200) + rng.sample(hangul, 200)
    marks = [0x301, 0x300, 0x302, 0x308, 0x323, 0x327, 0x328, 0x30C, 0x31B, 0x306]
    for base in "aeoAOuU\uff45\u03b1\u0438":
        for m1 in marks:
            for m2 in marks:
                if m1 != m2 and rng.random() < 0.35:
                    out.append(base + chr(m1) + chr(m2))
    for i in range(120):
        lead, vowel, tail = rng.randrange(19), rng.randrange(21), rng.randrange(28)
        out.append(chr(0x1100 + lead) + chr(0x1161 + vowel) + (chr(0x11A7 + tail) if tail else ""))
    out += ["\u3131\u314f", "\uffa1\uffc2", "\u3131\u314f\u11a8", "\u1100\u1161\u11a8\u11a8", "\u3132\u3150"]
    return out


def normalization():
    plain = {r: spm.SentencePieceNormalizer(rule_name=r) for r in NORMALIZATION_RULES}
    points = code_points()
    singles = {}
    for rule, normalizer in plain.items():
        singles[rule] = dict(zip(points, normalizer.normalize([chr(c) for c in points])))
    unicode = unicode_nfkc()
    lacking = sorted(c for c, value in unicode.items() if singles["nfkc"][c] != value)
    changed = {c: v for c, v in singles["nfkc"].items() if v != chr(c)}
    nmt = {c: v for c, v in singles["nmt_nfkc"].items() if v != singles["nfkc"][c]}
    sample = sequences(plain["nfkc"])
    text = lines()
    corpus = {}
    for rule, normalizer in plain.items():
        corpus[rule] = {str(i): out for i, (line, out) in enumerate(zip(text, normalizer.normalize(text))) if out != line}
    print(f"normalization: nfkc changes {len(changed)} code points, nmt_nfkc {len(nmt)} more; {len(sample)} sequences; "
          f"sentencepiece's tables lack {len(lacking)} of Unicode 18's", file=sys.stderr)
    return {
        "nfkc": {f"{c:04X}": v for c, v in sorted(changed.items())},
        "nmt_nfkc": {f"{c:04X}": v for c, v in sorted(nmt.items())},
        "unicode_lacking": {f"{c:04X}": {"unicode": unicode[c], "sentencepiece": singles["nfkc"][c]} for c in lacking},
        "sequences": [[s, plain["nfkc"].normalize(s), plain["nmt_nfkc"].normalize(s)] for s in sample],
        "corpus": corpus,
    }


def overlap(a, b):
    return len(set(a) & set(b)) / max(1, len(set(b)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=int, default=0, help="also train tokenizers' WordPiece this many times and print their agreement")
    parser.add_argument("--part", action="append", choices=["unigram", "wordpiece", "normalization"],
                        help="write only these files (and the manifest), keeping the others; default: every file, the folder emptied first")
    args = parser.parse_args()
    parts = set(args.part or ["unigram", "wordpiece", "normalization"])
    text = lines()
    if not args.part and os.path.isdir(OUT):
        for entry in sorted(os.listdir(OUT)):
            if entry != "README.md":
                path = os.path.join(OUT, entry)
                shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    os.makedirs(OUT, exist_ok=True)
    if "unigram" in parts:
        work = os.path.join(OUT, ".work")
        os.makedirs(work)
        try:
            with open(os.path.join(OUT, "whitespace-corpus.txt"), "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(whitespace_corpus()) + "\n")
            write_json(os.path.join(OUT, "unigram.json"), {name: train_unigram(name, settings, text, work) for name, settings in UNIGRAM.items()})
        finally:
            shutil.rmtree(work)
    if "normalization" in parts:
        write_json(os.path.join(OUT, "normalization.json"), normalization())
    if "wordpiece" in parts:
        wordpiece(text, args.runs)
    manifest()


def wordpiece(text, runs):
    words = bert_words(text, WORDPIECE["lowercase"])
    fixed = wordpiece_fixed(words, WORDPIECE["special_tokens"], WORDPIECE["vocab_size"], WORDPIECE["min_frequency"])
    hf = wordpiece_huggingface(text)
    print(f"wordpiece: {len(words)} words, fixed {len(fixed)} tokens, tokenizers {len(hf)} tokens, overlap {overlap(fixed, hf):.4f}",
          file=sys.stderr)
    for run in range(runs):
        again = wordpiece_huggingface(text)
        print(f"tokenizers run {run + 1}: overlap with the first {overlap(again, hf):.4f}, with fixed {overlap(fixed, again):.4f}",
              file=sys.stderr)
    write_json(os.path.join(OUT, "wordpiece.json"), {
        "options": WORDPIECE,
        "words": len(words),
        "fixed": fixed,
        "huggingface": hf,
    })


def manifest():
    files = {}
    for name in sorted(os.listdir(OUT)):
        if name in ("manifest.json", "README.md"):
            continue
        with open(os.path.join(OUT, name), "rb") as f:
            data = f.read()
        files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    inputs = {}
    for name in FILES + ["unicode/normalization-test.txt.gz"]:
        with open(os.path.join(DATA, name), "rb") as f:
            data = f.read()
        inputs[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    write_json(os.path.join(OUT, "manifest.json"), {
        "versions": {"sentencepiece": spm.__version__, "tokenizers": tokenizers.__version__,
                     "python": sys.version.split()[0]},
        "seed": SEED,
        "inputs": inputs,
        "files": files,
    })


if __name__ == "__main__":
    main()
