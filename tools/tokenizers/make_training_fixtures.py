"""The references for plan 24's trainers (phase T4): what sentencepiece and tokenizers learn from the test corpus, for the
C# trainers to be held to.

    tok/bin/python -I tools/tokenizers/make_training_fixtures.py            # writes tests/Idrak.Tests/data/tokenizer-training/
    tok/bin/python -I tools/tokenizers/make_training_fixtures.py --runs 5   # also: how much Hugging Face's WordPiece runs agree

The training text is data/tokenizers/corpus.txt and data/pretrain/corpus.txt, read as the C# trainers read them (each
line without its "\n", a "\r" before it dropped, as LineEnds "lf" reads it). Files written (the folder is emptied first,
README.md kept):

- unigram.json: per configuration, the options and the pieces (piece, score, type) sentencepiece's unigram trainer learns
  (one thread, no shuffling, the identity normalization, the Arabic marks removed by a user rule where the configuration
  says so). The C# trainer is held to these by overlap of the learned (normal) pieces, above a threshold measured here
  and recorded in plans/24-tokenizers.md: the two sum in different orders, so equal is not expected.
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
}
WORDPIECE = dict(vocab_size=2000, min_frequency=2, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"], lowercase=True)


def lines():
    """Every line of the training files as the C# trainers read them (LineEnds "lf")."""
    out = []
    for name in FILES:
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


def write_json(path, value):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------------------------------------------------
# Unigram: sentencepiece's trainer.
# ---------------------------------------------------------------------------------------------------------------------
def train_unigram(name, settings, text, work):
    settings = dict(settings)
    extra = {}
    if settings.pop("arabic", False):
        rules = os.path.join(work, "arabic-removals.tsv")
        with open(rules, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(ARABIC_RULES) + "\n")
        extra["normalization_rule_tsv"] = rules
    else:
        extra["normalization_rule_name"] = "identity"
    spm.set_random_generator_seed(SEED)
    import io
    writer = io.BytesIO()
    spm.SentencePieceTrainer.train(sentence_iterator=iter(text), model_writer=writer, num_threads=1, minloglevel=2,
                                   model_type="unigram", shuffle_input_sentence=False, hard_vocab_limit=False,
                                   character_coverage=0.9995, max_sentence_length=1 << 20, **settings, **extra)
    proto = model_pb2.ModelProto()
    proto.ParseFromString(writer.getvalue())
    options = dict(settings, character_coverage=0.9995, special_tokens=["<unk>", "<s>", "</s>"],
                   arabic=["tatweel", "diacritics"] if extra.get("normalization_rule_tsv") else [])
    pieces = [[p.piece, p.score, int(p.type)] for p in proto.pieces]
    print(f"{name}: {len(pieces)} pieces", file=sys.stderr)
    return {"options": options, "pieces": pieces}


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


def overlap(a, b):
    return len(set(a) & set(b)) / max(1, len(set(b)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=int, default=0, help="also train tokenizers' WordPiece this many times and print their agreement")
    args = parser.parse_args()
    text = lines()
    if os.path.isdir(OUT):
        for entry in sorted(os.listdir(OUT)):
            if entry != "README.md":
                path = os.path.join(OUT, entry)
                shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    os.makedirs(OUT, exist_ok=True)
    work = os.path.join(OUT, ".work")
    os.makedirs(work)
    try:
        write_json(os.path.join(OUT, "unigram.json"), {name: train_unigram(name, settings, text, work) for name, settings in UNIGRAM.items()})
    finally:
        shutil.rmtree(work)

    words = bert_words(text, WORDPIECE["lowercase"])
    fixed = wordpiece_fixed(words, WORDPIECE["special_tokens"], WORDPIECE["vocab_size"], WORDPIECE["min_frequency"])
    hf = wordpiece_huggingface(text)
    print(f"wordpiece: {len(words)} words, fixed {len(fixed)} tokens, tokenizers {len(hf)} tokens, overlap {overlap(fixed, hf):.4f}",
          file=sys.stderr)
    for run in range(args.runs):
        again = wordpiece_huggingface(text)
        print(f"tokenizers run {run + 1}: overlap with the first {overlap(again, hf):.4f}, with fixed {overlap(fixed, again):.4f}",
              file=sys.stderr)
    write_json(os.path.join(OUT, "wordpiece.json"), {
        "options": WORDPIECE,
        "words": len(words),
        "fixed": fixed,
        "huggingface": hf,
    })

    files = {}
    for name in sorted(os.listdir(OUT)):
        if name in ("manifest.json", "README.md"):
            continue
        with open(os.path.join(OUT, name), "rb") as f:
            data = f.read()
        files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    inputs = {}
    for name in FILES:
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
