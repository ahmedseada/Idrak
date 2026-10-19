"""The references for plan 24's tokenizers (phase T0): small trained tokenizers and what the reference libraries make of a
corpus written to break tokenizers.

    python -m venv tok && tok/bin/pip install -r tools/tokenizers/requirements.txt
    tok/bin/python -I tools/tokenizers/make_fixtures.py              # the committed fixtures (nothing downloaded)
    tok/bin/python -I tools/tokenizers/make_fixtures.py --real       # the real models' references (downloads)
    tok/bin/python -I tools/tokenizers/make_fixtures.py --bench      # the references' speed, as JSON

The default mode writes tests/Idrak.Tests/data/tokenizers/ (every file in it except README.md, written by hand, and
real/, written by --real, is removed first):

- corpus.txt: the hand-written test corpus (the lines below; Arabic with harakat and tatweel, mixed directions with
  numbers, emoji sequences, CJK, accents and combining marks, whitespace runs, tabs, CRLF lines, empty lines, very long
  words, control characters, and ordinary prose in many languages). Its lines are the file split at LF: a CRLF line keeps
  its CR.
- unigram/, unigram-bytes/, bpe/: SentencePiece models trained on corpus.txt plus data/pretrain/corpus.txt
  (tokenizer.model), the tokenizer.json transformers' converters make of them, model.json (the ModelProto as the official
  protobuf reads it, and the precompiled charsmap's trie as an independent Python reader of darts-clone units decodes it),
  expected-spm.json (sentencepiece on every corpus line), expected-hf.json (tokenizers on every corpus line, and pairs),
  normalized.json (the charsmap alone through tokenizers' Precompiled, and sentencepiece's whole normalizer).
- wordpiece-uncased/, wordpiece-cased/: a WordPiece vocabulary learned from the BERT-normalized words of the same text
  (see wordpiece_vocab for why not with tokenizers' WordPieceTrainer), written as BertTokenizerFast would
  (tokenizer.json, vocab.txt, tokenizer_config.json), and expected-hf.json.
- known-differences.json: the corpus lines where sentencepiece and tokenizers disagree on the same model.
- manifest.json: the library and Unicode versions and each file's size and SHA-256.

Reruns write the same bytes (fixed seeds, one thread, sorted JSON, the folder emptied first) with the same library
versions. Offsets are UTF-16 indexes into the text (a character outside the BMP counts two), as C# strings index.
"""
import argparse
import hashlib
import importlib
import io
import json
import os
import shutil
import struct
import sys
import time
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT = os.path.join(ROOT, "tests", "Idrak.Tests", "data", "tokenizers")
PRETRAIN_CORPUS = os.path.join(ROOT, "tests", "Idrak.Tests", "data", "pretrain", "corpus.txt")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=DEFAULT_OUT, help="the fixture folder (default: tests/Idrak.Tests/data/tokenizers)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--real", action="store_true", help="download the real models into the cache, write real/<name>.json")
    mode.add_argument("--bench", action="store_true", help="print the references' tokens a second on the corpus as JSON")
    parser.add_argument("--cache", default=None, help="the cache root for --real (default: IDRAK_CACHE, else ~/.cache/idrak)")
    parser.add_argument("--seconds", type=float, default=1.0, help="--bench: the least time each measurement runs")
    return parser.parse_args()


ARGS = parse_args() if __name__ == "__main__" else None
if ARGS is None or not ARGS.bench:
    # One thread in training and encoding, so reruns write the same bytes (tokenizers reads these when it is imported).
    os.environ["RAYON_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sentencepiece as spm  # noqa: E402
import tokenizers  # noqa: E402
from google.protobuf import __version__ as PROTOBUF_VERSION  # noqa: E402
from sentencepiece import sentencepiece_model_pb2 as model_pb2  # noqa: E402
from tokenizers import AddedToken, Tokenizer, models, normalizers, pre_tokenizers, processors  # noqa: E402

SEED = 2026

# ---------------------------------------------------------------------------------------------------------------------
# The corpus. Each entry is one line; "\r" at the end makes it a CRLF line. Written for this file: nothing is downloaded.
# ---------------------------------------------------------------------------------------------------------------------
CORPUS = [
    # Arabic with harakat, shadda, tanween, dagger alef, and the same words bare.
    "اَلْعِلْمُ نُورٌ وَالْجَهْلُ ظَلَامٌ.",
    "كَتَبَ الطَّالِبُ الدَّرْسَ فِي الدَّفْتَرِ، ثُمَّ قَرَأَهُ عَلَى مُعَلِّمِهِ.",
    "ذَهَبَ الْوَلَدُ إِلَى الْمَدْرَسَةِ صَبَاحًا وَعَادَ مَسَاءً.",
    "هٰذَا كِتَابٌ جَدِيدٌ، وَذٰلِكَ قَلَمٌ قَدِيمٌ.",
    "العلم نور والجهل ظلام. كتب الطالب الدرس في الدفتر.",
    # Tatweel (kashida), alone and inside words.
    "مـــرحـــبـــا بـكـم في الـمـكـتـبـة الـعـامـة",
    "الســـــــلام عليكم ورحمة الله ـ وبركاته ـــ",
    "ـــــــــــــــــــــــــــــــــــــــــــــــــ",
    # Mixed directions with numbers, Arabic-Indic and Persian digits, bidi marks.
    "سعر الكتاب 25.50 ريال، ورقم الطلب ORD-2024-0917 بتاريخ 2024/09/17.",
    "اتصل على +966 50 123 4567 أو راسلنا عبر support@example.com قبل الساعة 10:30 صباحًا.",
    "العدد ١٢٣٤٥ بالأرقام العربية الهندية، و۱۲۳۴۵ بالأرقام الفارسية، و12345 بالأرقام الغربية.",
    "يعمل Windows 11 على معالج Intel Core i7-12700H بذاكرة 16GB و SSD سعته 512GB.",
    "‏نص من اليمين‏ then English ‎left to right‎ ثم عربي مرة أخرى.",
    "‫تضمين من اليمين‬ and ⁧عزل من اليمين⁩ with ⁦isolated LTR⁩.",
    "النسبة 45% من 1,250,000 ريال = 562,500 ريال (تقريبًا).",
    # Emoji: skin tones, flags (regional indicators and a tag sequence), ZWJ sequences, keycaps, variation selectors.
    "👍🏽 👋🏿 🙏🏻 👌🏼 ✊🏾 thumbs up with skin tones",
    "🇸🇦 🇪🇬 🇯🇵 🇺🇸 🇫🇷 🏴󠁧󠁢󠁳󠁣󠁴󠁿 🏴‍☠️ flags",
    "👨‍👩‍👧‍👦 👩🏽‍💻 🧑🏿‍🚀 🏳️‍🌈 ❤️‍🔥 🐦‍🔥 🍋‍🟩 family, coder, astronaut, rainbow, fire",
    "1️⃣ 2️⃣ #️⃣ *️⃣ ☺️ ✌️ ☀︎ keycaps and variation selectors",
    "مرحبا 😀 بالعالم 🌍! Hello 😀 world 🌍! 你好😀世界🌍",
    "🫠🫨🥹🫶🏽😶‍🌫️😮‍💨🤌🏻",
    # CJK, fullwidth and halfwidth forms, a character outside the BMP.
    "今天天气很好，我们去公园散步吧。",
    "机器学习是人工智能的一个分支，它让计算机从数据中学习。",
    "東京は日本の首都です。ひらがなとカタカナと漢字を使います。",
    "안녕하세요, 만나서 반갑습니다. 한국어는 한글로 씁니다.",
    "ＡＢＣ１２３！ｶﾀｶﾅ ﾃｽﾄ　全角スペース　と半角",
    "𠜎𠜱𠝹𠱓 are characters outside the Basic Multilingual Plane, 𝐀𝐁𝐂 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 too.",
    # Latin accents, precomposed and decomposed; other scripts with marks.
    "Café, naïve, façade, coöperate, résumé, Ångström, Øresund, Straße, Ærø, señor, São Paulo.",
    "Café, naïve, façade, résumé, Ångström (decomposed).",
    "Tiếng Việt có dấu thanh: à á ả ã ạ, ầ ấ ẩ ẫ ậ, ừ ứ ử ữ ự.",
    "Z͑ͫ̓a̐̂l͒g̈́o̧ t͡e̕x̢t stacks combining marks.",
    "नमस्ते दुनिया, यह हिंदी में एक वाक्य है।",
    "สวัสดีครับ ภาษาไทยไม่มีช่องว่างระหว่างคำ",
    "שָׁלוֹם עוֹלָם, עברית עם ניקוד ובלי ניקוד.",
    "ﬁ ﬂ ﬃ ligatures, ① ② ③ circled, ㎏ ㎞ ㎡ squared, ½ ¼ fractions, Ⅻ roman, ™ © ®.",
    "­ soft­hyphen and ﬁ and Ⓐⓑ and Ａ.",
    # Whitespace runs, tabs, other spaces, CRLF lines, empty and blank lines.
    "many     spaces   between    words",
    "   leading and trailing spaces   ",
    "col1\tcol2\t\tcol4\t",
    "\t\ttabs first",
    "100 km and 50 kg, en space em space thin space hair space.",
    "日本　語 ideographic space, zero​width space, joiner‍and non‌joiner.",
    "a line that ends with CRLF\r",
    "سطر عربي ينتهي بـ CRLF\r",
    "\r",
    "",
    "",
    "   \t  ",
    "﻿starts with a byte order mark",
    # Very long words.
    "Donaudampfschifffahrtselektrizitätenhauptbetriebswerkbauunterbeamtengesellschaft",
    "pneumonoultramicroscopicsilicovolcanoconiosis supercalifragilisticexpialidocious",
    "a" * 300,
    "أفأستسقيكموها وفسيكفيكهم وأنلزمكموها",
    "https://example.com/a/very/long/path/with-many-segments-and-words/index.html?query=tokenizers&lang=ar&page=12345",
    "0123456789" * 12,
    "ابتثجحخدذرزسشصضطظعغفقكلمنهوي" * 6,
    # Control characters.
    "\x01\x02 bell\x07 escape\x1b[31mred\x1b[0m delete\x7f",
    "vertical\x0btab form\x0cfeed next\x85line line separator paragraph separator",
    "replacement � private use  \U000f0000 and a noncharacter-free end",
    # Special-token text the tokenizers know.
    "<s> </s> <unk> <pad> [CLS] [SEP] [MASK] [UNK] [PAD] <mask> <cls> <sep> <extra_id_0>",
    "text<mask>with<cls>markers</s>inside",
    # Ordinary prose: English.
    "The quick brown fox jumps over the lazy dog.",
    "Tokenizers split text into pieces that a language model can read, one id for each piece.",
    "A tokenizer trained on a small corpus learns the words it sees most often and spells the rest from smaller parts.",
    "The library reads the file once, keeps the vocabulary in flat arrays, and looks pieces up without building strings.",
    "It was a bright cold day in April, and the clocks were striking thirteen.",
    "Numbers like 3.14159, 2,718 and -42 appear in text as often as words do.",
    "She sells sea shells by the sea shore; the shells she sells are surely sea shells.",
    "Unicode gives every character a number, and UTF-8 writes that number in one to four bytes.",
    # Arabic.
    "تقسم أدوات التجزئة النص إلى قطع يقرؤها النموذج اللغوي، لكل قطعة رقم.",
    "تعلم المجزئ المدرب على نص صغير الكلمات الأكثر تكرارا، ويكتب الباقي من أجزاء أصغر.",
    "قرأت المكتبة الملف مرة واحدة، وحفظت المفردات في مصفوفات مسطحة.",
    "اللغة العربية من أكثر اللغات انتشارا في العالم، ويتحدث بها أكثر من أربعمائة مليون إنسان.",
    "في الصباح الباكر خرج المزارعون إلى الحقول، وكانت الشمس تشرق فوق التلال الخضراء.",
    "أرسل الوزير خطابا إلى المحكمة يطلب فيه نسخة من الحكم الصادر في الدعوى رقم 512.",
    # French, German, Spanish, Portuguese, Italian, Turkish, Polish.
    "Le petit prince regardait les étoiles et se demandait où était sa rose.",
    "Ça m'a coûté très cher, mais l'été dernier était inoubliable à la mer.",
    "Die Bibliothek liest die Datei einmal und behält den Wortschatz in flachen Feldern.",
    "Grüße aus München! Über die Brücke gehen wir zur Straße am Fluss.",
    "El niño comió una manzana y después corrió por el jardín con su perro.",
    "¿Dónde está la biblioteca? ¡Está al lado de la estación!",
    "A tokenização divide o texto em pedaços que o modelo consegue ler.",
    "La lingua italiana è bella, e la pizza napoletana è più buona.",
    "Türkçe öğrenmek güzeldir; ılık bir rüzgâr esiyor ve çiçekler açıyor.",
    "Zażółć gęślą jaźń: polskie znaki są trudne dla wielu programów.",
    # Russian, Ukrainian, Greek.
    "Токенизатор делит текст на части, которые модель может прочитать.",
    "Съешь же ещё этих мягких французских булок, да выпей чаю.",
    "Їжак ґречно усміхнувся: це український текст з літерами є, і, ї та ґ.",
    "Η γρήγορη καφέ αλεπού πηδάει πάνω από τον τεμπέλη σκύλο.",
    # Persian, Urdu, Hebrew, Hindi, Bengali.
    "زبان فارسی با الفبای عربی نوشته می‌شود و حروف پ، چ، ژ و گ را دارد.",
    "اردو زبان میں بھی عربی رسم الخط استعمال ہوتا ہے، جیسے یہ جملہ۔",
    "הספרייה קוראת את הקובץ פעם אחת ושומרת את אוצר המילים במערכים.",
    "पुस्तकालय फ़ाइल को एक बार पढ़ता है और शब्दावली को सरणियों में रखता है।",
    "আমি বাংলায় গান গাই, আমি বাংলার গান গাই।",
    # Chinese, Japanese, Korean prose.
    "图书馆只读取一次文件，并把词表保存在平坦的数组中。",
    "自然语言处理需要把文本切分成词元，每个词元对应一个编号。",
    "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。",
    "トークナイザーは文章を小さな単位に分けて、それぞれに番号を付けます。",
    "도서관은 파일을 한 번 읽고 어휘를 평평한 배열에 보관합니다.",
    # Mixed scripts in one sentence.
    "The word كتاب means book, 本 in Japanese, книга in Russian and βιβλίο in Greek.",
    "Version 2.0 من المكتبة يدعم UTF-8 و UTF-16 و Unicode 15.1 بالكامل.",
    "在2024年，Idrak 支持 العربية، English 和 中文。",
]


def corpus_text():
    return "\n".join(CORPUS) + "\n"


def corpus_lines(text):
    """The file split at LF; the final LF ends the last line (a CRLF line keeps its CR)."""
    lines = text.split("\n")
    assert lines[-1] == ""
    return lines[:-1]


def training_lines(text):
    """corpus.txt and the pretraining corpus (data/pretrain/corpus.txt, without its '---' separators), non-empty lines."""
    with open(PRETRAIN_CORPUS, encoding="utf-8", newline="") as f:
        pretrain = f.read()
    lines = [line for line in corpus_lines(text) if line.strip()]
    lines += [line for line in pretrain.split("\n") if line.strip() and line != "---"]
    return lines


# ---------------------------------------------------------------------------------------------------------------------
# Output helpers.
# ---------------------------------------------------------------------------------------------------------------------
def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def render(value, level=0):
    """Sorted JSON with objects one key a line and lists of objects or lists one compact item a line."""
    pad = " " * (level + 1)
    if isinstance(value, dict) and value:
        items = [f"{pad}{json.dumps(k, ensure_ascii=False)}: {render(value[k], level + 1)}" for k in sorted(value)]
        return "{\n" + ",\n".join(items) + "\n" + " " * level + "}"
    if isinstance(value, list) and value and all(isinstance(x, (dict, list)) for x in value):
        return "[\n" + ",\n".join(pad + compact(x) for x in value) + "\n" + " " * level + "]"
    return compact(value)


def write_json(path, value):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(render(value))
        f.write("\n")


def write_bytes(path, data):
    with open(path, "wb") as f:
        f.write(data)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    with open(path, "rb") as f:
        return sha256_bytes(f.read())


def float_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def utf16_index(text):
    """index[i] = the UTF-16 index of code point i (index[len] = the UTF-16 length)."""
    index = [0] * (len(text) + 1)
    for i, ch in enumerate(text):
        index[i + 1] = index[i] + (2 if ord(ch) > 0xFFFF else 1)
    return index


def utf8_to_utf16(text):
    """byte offset -> UTF-16 index, for byte offsets on character boundaries (None inside a character)."""
    mapping = {}
    b = u = 0
    for ch in text:
        mapping[b] = u
        b += len(ch.encode("utf-8"))
        u += 2 if ord(ch) > 0xFFFF else 1
    mapping[b] = u
    return mapping


def versions():
    import transformers

    result = {
        "python": sys.version.split()[0],
        "tokenizers": tokenizers.__version__,
        "sentencepiece": spm.__version__,
        "transformers": transformers.__version__,
        "protobuf": PROTOBUF_VERSION,
        "unicodedata (python)": unicodedata.unidata_version,
    }
    try:
        import huggingface_hub

        result["huggingface_hub"] = huggingface_hub.__version__
    except ImportError:
        pass
    return result


# ---------------------------------------------------------------------------------------------------------------------
# The ModelProto as the official protobuf reads it, and the charsmap's darts-clone trie decoded in Python.
# ---------------------------------------------------------------------------------------------------------------------
def message_dump(message):
    """Every field of a message: its value (the declared default when absent), floats with their bits, bytes as size and
    SHA-256, and the names of the fields present in the file."""
    out = {}
    for field in message.DESCRIPTOR.fields:
        value = getattr(message, field.name)
        if field.is_repeated:
            value = list(value)
        elif field.type == field.TYPE_BYTES:
            value = {"bytes": len(value), "sha256": sha256_bytes(value)}
        elif field.type == field.TYPE_FLOAT:
            out[field.name + "_bits"] = float_bits(value)
        elif field.type == field.TYPE_ENUM:
            value = field.enum_type.values_by_number[value].name
        out[field.name] = value
    out["present"] = sorted(f.name for f, _ in message.ListFields())
    return out


def darts_entries(charsmap):
    """The precompiled charsmap (u32 trie size in bytes, the trie's u32 units, then NUL-terminated normalized strings,
    all little-endian) decoded by walking every path of the double array, as darts-clone lays it out."""
    trie_size = struct.unpack_from("<I", charsmap, 0)[0]
    units = struct.unpack_from(f"<{trie_size // 4}I", charsmap, 4)
    normalized = charsmap[4 + trie_size:]

    def offset(unit):
        return (unit >> 10) << ((unit & (1 << 9)) >> 6)

    def label(unit):
        return unit & ((1 << 31) | 0xFF)

    def has_leaf(unit):
        return (unit >> 8) & 1 == 1

    entries = []
    stack = [(offset(units[0]), b"")]
    while stack:
        node, key = stack.pop()
        for c in range(1, 256):
            child = node ^ c
            if child >= len(units) or label(units[child]) != c:
                continue
            next_node = child ^ offset(units[child])
            if has_leaf(units[child]):
                value = units[next_node] & 0x7FFFFFFF
                end = normalized.index(b"\0", value)
                entries.append((key + bytes([c]), value, normalized[value:end]))
            stack.append((next_node, key + bytes([c])))
    entries.sort()
    listing = "".join(f"{k.hex()}\t{v.hex()}\n" for k, _, v in entries).encode("ascii")
    return {
        "trie_bytes": trie_size,
        "units": len(units),
        "normalized_bytes": len(normalized),
        "entries": len(entries),
        "entries_sha256": sha256_bytes(listing),
        "longest_key": max((len(k) for k, _, _ in entries), default=0),
    }


def model_dump(model_bytes):
    proto = model_pb2.ModelProto()
    proto.ParseFromString(model_bytes)
    piece_type = model_pb2.ModelProto.SentencePiece.Type
    dump = {
        "file": {"bytes": len(model_bytes), "sha256": sha256_bytes(model_bytes)},
        "pieces": [{"piece": p.piece, "score": p.score, "score_bits": float_bits(p.score), "type": piece_type.Name(p.type)}
                   for p in proto.pieces],
        "trainer_spec": message_dump(proto.trainer_spec),
        "normalizer_spec": message_dump(proto.normalizer_spec),
        "denormalizer_spec": message_dump(proto.denormalizer_spec) if proto.HasField("denormalizer_spec") else None,
    }
    charsmap = proto.normalizer_spec.precompiled_charsmap
    dump["charsmap"] = darts_entries(charsmap) if charsmap else None
    return proto, dump


# ---------------------------------------------------------------------------------------------------------------------
# References.
# ---------------------------------------------------------------------------------------------------------------------
def spm_references(model_bytes, lines):
    """sentencepiece on every line: ids, pieces, offsets (its byte offsets turned into UTF-16), type ids (all 0),
    special-tokens mask (control pieces), decoded text. No BOS or EOS: sentencepiece adds none by default."""
    sp = spm.SentencePieceProcessor(model_proto=model_bytes)
    out = []
    for text in lines:
        proto = sp.encode(text, return_type="proto")
        to16 = utf8_to_utf16(text)
        ids = [p.id for p in proto.pieces]
        out.append({
            "text": text,
            "ids": ids,
            "pieces": [p.piece for p in proto.pieces],
            "offsets": [[to16[p.begin], to16[p.end]] for p in proto.pieces],
            "type_ids": [0] * len(ids),
            "special_tokens_mask": [1 if sp.IsControl(i) else 0 for i in ids],
            "decoded": sp.decode(ids),
        })
    return out


def hf_encoding(tokenizer, text, pair=None):
    encoding = tokenizer.encode(text, pair) if pair is not None else tokenizer.encode(text)
    first, second = utf16_index(text), utf16_index(pair) if pair is not None else None
    offsets = []
    for (start, end), sequence in zip(encoding.offsets, encoding.sequence_ids):
        index = second if sequence == 1 else first
        offsets.append([index[start], index[end]])
    record = {
        "ids": encoding.ids,
        "pieces": encoding.tokens,
        "offsets": offsets,
        "type_ids": encoding.type_ids,
        "special_tokens_mask": encoding.special_tokens_mask,
        "sequence_ids": [-1 if s is None else s for s in encoding.sequence_ids],
        "decoded": tokenizer.decode(encoding.ids),
        "decoded_with_special": tokenizer.decode(encoding.ids, skip_special_tokens=False),
    }
    return record


def hf_references(tokenizer, lines):
    """tokenizers on every line (special tokens added by the post-processor), and pairs of consecutive non-empty lines
    from the first 24 (the pair's offsets index its own text)."""
    single = []
    for text in lines:
        record = hf_encoding(tokenizer, text)
        record["text"] = text
        single.append(record)
    texts = [line for line in lines if line.strip()][:24]
    pairs = []
    for i in range(0, len(texts) - 1, 2):
        record = hf_encoding(tokenizer, texts[i], texts[i + 1])
        record["text"], record["pair"] = texts[i], texts[i + 1]
        pairs.append(record)
    return {"lines": single, "pairs": pairs}


def normalized_references(proto, model_bytes, lines):
    """The charsmap alone through tokenizers' Precompiled (by grapheme), and sentencepiece's whole normalizer (charsmap,
    dummy prefix, extra whitespace, ▁) with, for each code point of its output, the UTF-16 index in the text where it
    starts (then the end)."""
    sp = spm.SentencePieceProcessor(model_proto=model_bytes)
    charsmap = proto.normalizer_spec.precompiled_charsmap
    precompiled = normalizers.Precompiled(charsmap) if charsmap else None
    out = []
    for text in lines:
        normalized, offsets = sp.Normalize(text, with_offsets=True)
        index = utf16_index(text)
        record = {"text": text, "spm": normalized, "spm_offsets": [index[o] for o in offsets]}
        if precompiled is not None:
            record["precompiled"] = precompiled.normalize_str(text)
        out.append(record)
    return out


def known_differences(name, spm_refs, hf_refs):
    """Lines where sentencepiece's ids (or decoded text) differ from tokenizers' with the post-processor's tokens left out."""
    out = []
    for i, (a, b) in enumerate(zip(spm_refs, hf_refs["lines"])):
        hf_ids = [t for t, special in zip(b["ids"], b["special_tokens_mask"]) if not special]
        if a["ids"] != hf_ids or a["decoded"] != b["decoded"]:
            out.append({"tokenizer": name, "line": i, "text": a["text"], "spm_ids": a["ids"], "hf_ids": hf_ids,
                        "spm_pieces": a["pieces"], "hf_pieces": [p for p, s in zip(b["pieces"], b["special_tokens_mask"]) if not s],
                        "spm_decoded": a["decoded"], "hf_decoded": b["decoded"],
                        "same_ids": a["ids"] == hf_ids})
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Training and conversion.
# ---------------------------------------------------------------------------------------------------------------------
# A user normalization rule (sentencepiece's TSV: source code points, tab, target code points; an empty target removes):
# tatweel removed, Arabic-Indic and Persian digits to ASCII, alef forms to bare alef, final yeh forms, and two Latin
# ligatures. Compiled by sentencepiece into a small precompiled charsmap (normalizer name "user_defined").
ARABIC_RULES = (
    ["640\t"]
    + [f"{0x660 + d:X}\t{0x30 + d:X}" for d in range(10)]
    + [f"{0x6F0 + d:X}\t{0x30 + d:X}" for d in range(10)]
    + ["622\t627", "623\t627", "625\t627", "671\t627", "649\t64A", "6CC\t64A", "FB01\t66 69", "FB02\t66 6C", "41 41\t61 61"]
)

SPM_MODELS = {
    # T5's layout (pad 0, eos 1, unk 2, no bos), the default NFKC charsmap, control and user-defined pieces.
    "unigram": dict(model_type="unigram", vocab_size=1000, pad_id=0, eos_id=1, unk_id=2, bos_id=-1,
                    control_symbols=["<cls>", "<sep>"], user_defined_symbols=["<mask>"],
                    normalization_rule_name="nmt_nfkc"),
    # Byte fallback, digits split, and the user rule above.
    "unigram-bytes": dict(model_type="unigram", vocab_size=1000, byte_fallback=True, split_digits=True,
                          normalization_rule_tsv="ARABIC_RULES"),
    # Llama-2's layout: BPE, byte fallback, digits split, identity normalization, no extra-whitespace removal.
    "bpe": dict(model_type="bpe", vocab_size=1000, byte_fallback=True, split_digits=True,
                normalization_rule_name="identity", remove_extra_whitespaces=False,
                allow_whitespace_only_pieces=True),
}
T5_EXTRA_IDS = 2


def train_spm(name, settings, lines, work):
    settings = dict(settings)
    if settings.get("normalization_rule_tsv") == "ARABIC_RULES":
        rules = os.path.join(work, "arabic-rules.tsv")
        with open(rules, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(ARABIC_RULES) + "\n")
        # The model keeps the rule file's name (normalizer_spec.normalization_rule_tsv): train from inside the work
        # folder so it is the bare name, the same on every machine.
        settings["normalization_rule_tsv"] = os.path.basename(rules)
    spm.set_random_generator_seed(SEED)
    writer = io.BytesIO()
    here = os.getcwd()
    os.chdir(work)
    try:
        spm.SentencePieceTrainer.train(sentence_iterator=iter(lines), model_writer=writer, num_threads=1,
                                       shuffle_input_sentence=False, hard_vocab_limit=False, minloglevel=2,
                                       character_coverage=0.9995, **settings)
    finally:
        os.chdir(here)
    return writer.getvalue()


class _Slow:
    """What transformers' converters read from a slow tokenizer (transformers 5 has none left): the .model path and a few
    settings. Named after the slow class (T5Tokenizer, LlamaTokenizer), convert_slow_tokenizer picks the converter."""

    def __init__(self, vocab_file, proto, **settings):
        self.vocab_file = vocab_file
        self._pieces = [p.piece for p in proto.pieces]
        self.__dict__.update(settings)

    def convert_tokens_to_ids(self, token):
        extra = [f"<extra_id_{i}>" for i in range(getattr(self, "_extra_ids", 0) - 1, -1, -1)]
        return (self._pieces + extra).index(token)

    def convert_ids_to_tokens(self, index):
        return self._pieces[index]


def special(token):
    return AddedToken(token, single_word=False, lstrip=False, rstrip=False, normalized=False, special=True)


def convert_spm(name, model_path, proto):
    """tokenizer.json the way the Hub's files were made: transformers' converter for the matching slow class, then the
    special tokens the fast class adds (and its post-processor, for Llama)."""
    convert = importlib.import_module("transformers.convert_slow_tokenizer")
    from transformers.tokenization_utils_base import generate_merges

    spec = proto.trainer_spec
    if name == "unigram":
        slow = type("T5Tokenizer", (_Slow,), {})(model_path, proto, _extra_ids=T5_EXTRA_IDS)
        tokenizer = convert.convert_slow_tokenizer(slow)
        extras = [f"<extra_id_{i}>" for i in range(T5_EXTRA_IDS)]
        tokenizer.add_special_tokens([special(t) for t in [spec.pad_piece, spec.eos_piece, spec.unk_piece] + extras])
        return tokenizer
    if name == "unigram-bytes":
        # No Hub class keeps a Unigram model's byte fallback (T5's and ALBERT's converters drop it), so: SpmConverter
        # with byte fallback kept and Llama's decoder sequence, which turns the byte pieces back into text.
        class ByteFallbackUnigramConverter(convert.SpmConverter):
            handle_byte_fallback = True

            def decoder(self, replacement, add_prefix_space):
                return convert.LlamaConverter.decoder(self, replacement, add_prefix_space)

        tokenizer = ByteFallbackUnigramConverter(_Slow(model_path, proto)).converted()
        tokenizer.add_special_tokens([special(t) for t in [spec.unk_piece, spec.bos_piece, spec.eos_piece]])
        return tokenizer

    # LlamaTokenizer. transformers 5's SpmConverter.tokenizer calls SentencePieceExtractor.extract with the scores,
    # which its new signature no longer takes, so the BPE model is built as transformers 4's extractor built it:
    # generate_merges over the pieces and their scores.
    class Llama(convert.LlamaConverter):
        def tokenizer(self, proto):
            vocab_scores = self.vocab(proto)
            vocab = {piece: i for i, (piece, _) in enumerate(vocab_scores)}
            merges = generate_merges(vocab, vocab_scores)
            tokenizer = Tokenizer(models.BPE(vocab, merges, unk_token=proto.trainer_spec.unk_piece, fuse_unk=True,
                                             byte_fallback=True, dropout=None))
            added = [(i, p.piece, p.type == 3) for i, p in enumerate(proto.pieces) if p.type in (3, 4)]
            tokenizer.add_tokens([AddedToken(t, normalized=False, special=s) for _, t, s in sorted(added)])
            return tokenizer

    slow = type("LlamaTokenizer", (_Slow,), {})(model_path, proto, legacy=True, add_prefix_space=True)
    tokenizer = Llama(slow).converted()
    tokenizer.add_special_tokens([special(t) for t in [spec.unk_piece, spec.bos_piece, spec.eos_piece]])
    # LlamaTokenizerFast's post-processor (add_bos_token true, add_eos_token false).
    bos = spec.bos_piece
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{bos}:0 $A:0", pair=f"{bos}:0 $A:0 {bos}:1 $B:1", special_tokens=[(bos, spec.bos_id)])
    return tokenizer


SPM_CONFIGS = {
    "unigram": {"tokenizer_class": "T5Tokenizer", "eos_token": "</s>", "unk_token": "<unk>", "pad_token": "<pad>",
                "extra_ids": T5_EXTRA_IDS, "additional_special_tokens": [f"<extra_id_{i}>" for i in range(T5_EXTRA_IDS)],
                "model_max_length": 512},
    "unigram-bytes": {"tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<s>", "eos_token": "</s>",
                      "unk_token": "<unk>", "model_max_length": 512},
    "bpe": {"tokenizer_class": "LlamaTokenizer", "bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>",
            "add_bos_token": True, "add_eos_token": False, "legacy": True, "model_max_length": 2048},
}

WORDPIECE_SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
WORDPIECE_MODELS = {
    # BertConverter's post-processor (TemplateProcessing), as bert-base-uncased's tokenizer.json has it.
    "wordpiece-uncased": dict(lowercase=True, vocab_size=1500, post="template"),
    # tokenizers' own BertProcessing, as older BERT tokenizer.json files have it.
    "wordpiece-cased": dict(lowercase=False, vocab_size=2000, post="bert"),
}


def wordpiece_vocab(settings, lines):
    """A WordPiece vocabulary, reproducibly. tokenizers' WordPieceTrainer is not: it numbers the "##" alphabet in Rust's
    hash order, which changes every process, and merges between equal counts are broken by those numbers (two runs of
    0.23.2 on this corpus learned 39 different pieces). So the words BertNormalizer and BertPreTokenizer make of the
    training text go through a SentencePiece BPE run instead (deterministic): a piece that starts a word ("▁x") becomes
    "x", any other becomes "##x", every character of the alphabet is kept in both forms, and the special tokens come
    first, as BERT vocabularies lay them out."""
    normalizer = normalizers.BertNormalizer(lowercase=settings["lowercase"])
    pre = pre_tokenizers.BertPreTokenizer()
    words = []
    for line in lines:
        pieces = [w for w, _ in pre.pre_tokenize_str(normalizer.normalize_str(line))]
        if pieces:
            words.append(" ".join(pieces))
    spm.set_random_generator_seed(SEED)
    writer = io.BytesIO()
    spm.SentencePieceTrainer.train(sentence_iterator=iter(words), model_writer=writer, num_threads=1, minloglevel=2,
                                   shuffle_input_sentence=False, hard_vocab_limit=False, character_coverage=0.9995,
                                   model_type="bpe", vocab_size=settings["vocab_size"], normalization_rule_name="identity",
                                   split_by_unicode_script=False, unk_id=0, bos_id=-1, eos_id=-1, pad_id=-1)
    proto = model_pb2.ModelProto()
    proto.ParseFromString(writer.getvalue())
    vocab = {}
    for token in WORDPIECE_SPECIALS:
        vocab.setdefault(token, len(vocab))
    pieces = [p.piece for p in proto.pieces if p.type == model_pb2.ModelProto.SentencePiece.NORMAL]
    for piece in pieces:
        token = piece[1:] if piece.startswith("\u2581") else "##" + piece
        if token and token != "##":
            vocab.setdefault(token, len(vocab))
    for piece in pieces:
        if len(piece) == 1 and piece != "\u2581":
            vocab.setdefault(piece, len(vocab))
            vocab.setdefault("##" + piece, len(vocab))
    return vocab


def train_wordpiece(settings, lines):
    """The vocabulary above, made into the tokenizer BertTokenizerFast makes from its vocab.txt (transformers'
    BertConverter: BertNormalizer, BertPreTokenizer, WordPiece, TemplateProcessing, the WordPiece decoder)."""
    convert = importlib.import_module("transformers.convert_slow_tokenizer")

    lower = settings["lowercase"]
    vocab = wordpiece_vocab(settings, lines)

    class BasicTokenizer:
        tokenize_chinese_chars = True
        strip_accents = None
        do_lower_case = lower

    slow = type("BertTokenizer", (), dict(vocab=vocab, unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]",
                                          cls_token_id=vocab["[CLS]"], sep_token_id=vocab["[SEP]"],
                                          basic_tokenizer=BasicTokenizer()))()
    tokenizer = convert.convert_slow_tokenizer(slow)
    tokenizer.add_special_tokens([special(t) for t in WORDPIECE_SPECIALS])
    if settings["post"] == "bert":
        tokenizer.post_processor = processors.BertProcessing(("[SEP]", vocab["[SEP]"]), ("[CLS]", vocab["[CLS]"]))
    config = {"tokenizer_class": "BertTokenizer", "do_lower_case": lower, "strip_accents": None,
              "tokenize_chinese_chars": True, "unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]",
              "cls_token": "[CLS]", "mask_token": "[MASK]", "model_max_length": 512,
              "clean_up_tokenization_spaces": True}
    return tokenizer, vocab, config


# ---------------------------------------------------------------------------------------------------------------------
# Modes.
# ---------------------------------------------------------------------------------------------------------------------
def clear_folder(out):
    """Removes everything in the fixture folder except the hand-written README.md and --real's real/."""
    os.makedirs(out, exist_ok=True)
    for entry in sorted(os.listdir(out)):
        if entry in ("README.md", "real"):
            continue
        path = os.path.join(out, entry)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def make_fixtures(out):
    clear_folder(out)
    text = corpus_text()
    with open(os.path.join(out, "corpus.txt"), "w", encoding="utf-8", newline="") as f:
        f.write(text)
    lines = corpus_lines(text)
    training = training_lines(text)
    differences = []
    work = os.path.join(out, ".work")
    os.makedirs(work)
    try:
        for name, settings in SPM_MODELS.items():
            folder = os.path.join(out, name)
            os.makedirs(folder)
            model_bytes = train_spm(name, settings, training, work)
            model_path = os.path.join(folder, "tokenizer.model")
            write_bytes(model_path, model_bytes)
            proto, dump = model_dump(model_bytes)
            write_json(os.path.join(folder, "model.json"), dump)
            tokenizer = convert_spm(name, model_path, proto)
            tokenizer.save(os.path.join(folder, "tokenizer.json"), pretty=True)
            write_json(os.path.join(folder, "tokenizer_config.json"), SPM_CONFIGS[name])
            tokenizer = Tokenizer.from_file(os.path.join(folder, "tokenizer.json"))
            spm_refs = spm_references(model_bytes, lines)
            hf_refs = hf_references(tokenizer, lines)
            write_json(os.path.join(folder, "expected-spm.json"), {"model": name, "lines": spm_refs})
            write_json(os.path.join(folder, "expected-hf.json"), dict(model=name, **hf_refs))
            write_json(os.path.join(folder, "normalized.json"),
                       {"model": name, "lines": normalized_references(proto, model_bytes, lines)})
            differences += known_differences(name, spm_refs, hf_refs)
            print(f"{name}: {len(proto.pieces)} pieces, {len(model_bytes):,} bytes", file=sys.stderr)
    finally:
        shutil.rmtree(work)

    for name, settings in WORDPIECE_MODELS.items():
        folder = os.path.join(out, name)
        os.makedirs(folder)
        tokenizer, vocab, config = train_wordpiece(settings, training)
        tokenizer.save(os.path.join(folder, "tokenizer.json"), pretty=True)
        with open(os.path.join(folder, "vocab.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write("".join(token + "\n" for token, _ in sorted(vocab.items(), key=lambda kv: kv[1])))
        write_json(os.path.join(folder, "tokenizer_config.json"), config)
        tokenizer = Tokenizer.from_file(os.path.join(folder, "tokenizer.json"))
        write_json(os.path.join(folder, "expected-hf.json"), dict(model=name, **hf_references(tokenizer, lines)))
        print(f"{name}: {len(vocab)} pieces", file=sys.stderr)

    write_json(os.path.join(out, "known-differences.json"), {"differences": differences})
    write_manifest(out, len(lines), len(training))


def write_manifest(out, lines, training):
    files = {}
    for folder, _, names in os.walk(out):
        for name in names:
            path = os.path.join(folder, name)
            relative = os.path.relpath(path, out).replace(os.sep, "/")
            if relative in ("manifest.json", "README.md") or relative.startswith("real/"):
                continue
            files[relative] = {"bytes": os.path.getsize(path), "sha256": sha256_file(path)}
    write_json(os.path.join(out, "manifest.json"), {
        "corpus_lines": lines,
        "training_lines": training,
        "training_input": {"data/pretrain/corpus.txt": {"bytes": os.path.getsize(PRETRAIN_CORPUS),
                                                       "sha256": sha256_file(PRETRAIN_CORPUS)}},
        "seed": SEED,
        "versions": versions(),
        "files": dict(sorted(files.items())),
    })


REAL_MODELS = [
    "google-t5/t5-small",
    "google/mt5-small",
    "albert/albert-base-v2",
    "FacebookAI/xlm-roberta-base",
    "hf-internal-testing/llama-tokenizer",
    "google-bert/bert-base-uncased",
    "google-bert/bert-base-multilingual-cased",
    "aubmindlab/bert-base-arabertv02",
]
REAL_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "spiece.model",
              "sentencepiece.bpe.model", "tokenizer.model", "vocab.txt"]
SPM_FILES = ["tokenizer.model", "spiece.model", "sentencepiece.bpe.model"]


def cache_root(explicit):
    if explicit:
        return explicit
    if os.environ.get("IDRAK_CACHE"):
        return os.environ["IDRAK_CACHE"]
    return os.path.join(os.path.expanduser("~"), ".cache", "idrak")


def make_real(out, cache):
    """Downloads each real model's tokenizer files into <cache>/tokenizers/<org>--<name>/ and writes only references:
    real/<name>.json (the files' sizes and SHA-256, the versions, sentencepiece's and tokenizers' references on every
    corpus line, and where they disagree)."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    with open(os.path.join(out, "corpus.txt"), encoding="utf-8", newline="") as f:
        lines = corpus_lines(f.read())
    real = os.path.join(out, "real")
    if os.path.isdir(real):
        shutil.rmtree(real)
    os.makedirs(real)
    for repo in REAL_MODELS:
        local = os.path.join(cache, "tokenizers", repo.replace("/", "--"))
        print(f"{repo}: downloading into {local}", file=sys.stderr)
        snapshot_download(repo_id=repo, local_dir=local, allow_patterns=REAL_FILES)
        present = sorted(name for name in REAL_FILES if os.path.isfile(os.path.join(local, name)))
        record = {
            "repo": repo,
            "files": {name: {"bytes": os.path.getsize(os.path.join(local, name)),
                             "sha256": sha256_file(os.path.join(local, name))} for name in present},
            "versions": versions(),
        }
        spm_file = next((name for name in SPM_FILES if name in present), None)
        if os.path.isfile(os.path.join(local, "tokenizer.json")):
            tokenizer, record["hf_source"] = Tokenizer.from_file(os.path.join(local, "tokenizer.json")), "tokenizer.json"
        else:
            tokenizer = AutoTokenizer.from_pretrained(local).backend_tokenizer
            record["hf_source"] = "transformers AutoTokenizer (" + ", ".join(present) + ")"
        hf_refs = hf_references(tokenizer, lines)
        record["hf"] = hf_refs
        if spm_file is not None:
            with open(os.path.join(local, spm_file), "rb") as f:
                model_bytes = f.read()
            proto, dump = model_dump(model_bytes)
            record["spm_file"] = spm_file
            record["model"] = {k: v for k, v in dump.items() if k != "pieces"}
            record["model"]["pieces"] = len(dump["pieces"])
            record["model"]["pieces_sha256"] = sha256_bytes(compact(dump["pieces"]).encode("utf-8"))
            record["spm"] = spm_references(model_bytes, lines)
            record["known_differences"] = known_differences(repo, record["spm"], hf_refs)
        write_json(os.path.join(real, repo.split("/")[-1] + ".json"), record)
        print(f"{repo}: references written", file=sys.stderr)


def make_bench(out, seconds):
    """Tokens a second of each reference on the corpus: one thread (one encode call per text) and all threads
    (sentencepiece's num_threads, tokenizers' encode_batch), the first pass (cold) apart from the best later one."""
    with open(os.path.join(out, "corpus.txt"), encoding="utf-8", newline="") as f:
        lines = [line for line in corpus_lines(f.read()) if line.strip()]
    texts = []
    while sum(len(t) for t in texts) < 200_000:
        texts += lines
    threads = os.cpu_count() or 1

    def measure(run):
        start = time.perf_counter()
        tokens = run()
        cold = tokens / (time.perf_counter() - start)
        best, spent = 0.0, 0.0
        while spent < seconds:
            start = time.perf_counter()
            tokens = run()
            elapsed = time.perf_counter() - start
            spent += elapsed
            best = max(best, tokens / elapsed)
        return {"cold": round(cold), "warm": round(best), "tokens": tokens}

    report = {"characters": sum(len(t) for t in texts), "texts": len(texts), "threads": threads,
              "versions": versions(), "tokenizers": {}}
    for name in list(SPM_MODELS) + list(WORDPIECE_MODELS):
        folder = os.path.join(out, name)
        row = {}
        model = os.path.join(folder, "tokenizer.model")
        if os.path.isfile(model):
            sp = spm.SentencePieceProcessor(model_file=model)
            row["sentencepiece, one thread"] = measure(lambda: sum(len(sp.encode(t)) for t in texts))
            row["sentencepiece, all threads"] = measure(lambda: sum(len(x) for x in sp.encode(texts, num_threads=threads)))
        tokenizer = Tokenizer.from_file(os.path.join(folder, "tokenizer.json"))
        row["tokenizers, one thread"] = measure(lambda: sum(len(tokenizer.encode(t).ids) for t in texts))
        row["tokenizers, all threads"] = measure(lambda: sum(len(e.ids) for e in tokenizer.encode_batch(texts)))
        report["tokenizers"][name] = row
    print(json.dumps(report, indent=1, sort_keys=True, ensure_ascii=False))


def main():
    out = os.path.abspath(ARGS.out)
    if ARGS.real:
        make_real(out, cache_root(ARGS.cache))
    elif ARGS.bench:
        make_bench(out, ARGS.seconds)
    else:
        make_fixtures(out)


if __name__ == "__main__":
    main()
