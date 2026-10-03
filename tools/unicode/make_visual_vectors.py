"""Writes tests/Idrak.Tests/data/arabic/visual-order.jsonl: lines of Arabic mixed with English, numbers, paths,
brackets, lam-alef and vowel marks, each with the visual order an independent implementation of the Unicode
Bidirectional Algorithm and Arabic shaping (GNU FriBidi's fribidi_log2vis, shaping and mirroring on) gives for it.
The idrak tool's renderer (src/Idrak.Cli/Shared/VisualText.cs) is tested against them.

Usage (Linux with libfribidi0 installed): python make_visual_vectors.py OUT.jsonl
Rows: {"text", "dir" (auto, rtl or ltr), "visual"}. FriBidi keeps the length, so the alef of a lam-alef ligature
becomes U+FEFF; it is removed here, as are the direction marks, which the tool's renderer drops too.
"""
import ctypes, json, random, sys

lib = ctypes.CDLL("libfribidi.so.0")
lib.fribidi_log2vis.restype = ctypes.c_int
PAR = {"auto": 0x40, "ltr": 0x110, "rtl": 0x111}


def visual(text, direction):
    cps = [ord(c) for c in text]
    n = len(cps)
    arr = (ctypes.c_uint32 * n)(*cps)
    out = (ctypes.c_uint32 * (n + 1))()
    base = ctypes.c_uint32(PAR[direction])
    r = lib.fribidi_log2vis(arr, n, ctypes.byref(base), out, None, None, None)
    assert r != 0
    result = "".join(chr(out[i]) for i in range(n))
    # fribidi keeps the length: the alef of a lam-alef ligature becomes U+FEFF; marks are dropped by us
    return result.replace("\ufeff", "").replace("\u200e", "").replace("\u200f", "").replace("\u061c", "")


fixed = [
    "مرحبا",
    "لا يوجد جهاز",
    "السلام عليكم",
    "إلا أن الأمر",
    "آلة لآلئ لإنسان",
    "الجهاز cpu جاهز",
    "تعذر فتح الملف C:\\models\\qwen.gguf",
    "الذاكرة 16 GB من 24 GB",
    "النسخة 0.1.7 من idrak",
    "(تجربة) [قوس] {معقوف}",
    "اكتب idrak help لعرض الأوامر.",
    "خطأ: لم يُعثر على النموذج 'qwen'.",
    "السرعة 123.5 رمزًا/ث",
    "شغّل: idrak doctor --android",
    "Error: لا يوجد",
    "vulkan:0 (بطاقة رسومات)",
    "النسبة 45% من 100",
    "١٢٣ و 456",
    "المسار /home/user/.cache/idrak",
    "ـــ تطويل",
    "a (b) ج (د) e",
    "لأ للا ؟",
    "عدد 3-4 أو 5+6",
    "سعر $20 ثم 30€",
    "‏idrak‏ أداة",
]
letters = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي" + "ءآأإؤئةى" + "لالالا"
other = "abc XYZ 0123 .,:;-+/%$#()[]<>\"' ١٢٣ \u064e\u0651\u0640"
random.seed(7)
rows = []
for text in fixed:
    for d in ("auto", "rtl", "ltr"):
        rows.append({"text": text, "dir": d, "visual": visual(text, d)})
for _ in range(300):
    n = random.randint(1, 24)
    text = "".join(random.choice(letters if random.random() < 0.6 else other) for _ in range(n)).strip()
    if not text or text.startswith("\u064e") or text.startswith("\u0651"):
        continue
    d = random.choice(["auto", "rtl", "ltr"])
    rows.append({"text": text, "dir": d, "visual": visual(text, d)})
with open(sys.argv[1], "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(rows))
