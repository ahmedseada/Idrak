"""Makes the Arabic font fixtures of the "shaping:" tests (plan 18): real fonts with large layout tables.

    python -I tools/fonts/make_arabic_font_references.py FOLDER

FOLDER holds Google Fonts' Amiri-Regular.ttf, Cairo[slnt,wght].ttf and ReemKufi[wght].ttf (github.com/google/fonts, ofl/)
and their OFL.txt files as OFL-amiri.txt, OFL-cairo.txt and OFL-reemkufi.txt. Writes under
tests/Idrak.Tests/data/fonts/arabic:
  amiri.ttf, cairo.ttf, reem-kufi.ttf   each font cut to the characters of LINES and the Arabic letters, every layout
                                        feature, variation table and name kept, hinting dropped (the SIL Open Font
                                        License, OFL-*.txt beside them, allows it)
  shaping.json                          HarfBuzz's shaping of every line in each font (glyph, cluster, advances,
                                        offsets), right to left, Arabic script and language, the default cluster level
LINES are random text from idrak data text (-n 92 --unit lines --line-words 2..10 --diacritics 0.33 --seed 11, the
numbers and dates left out) and a few fixed lines. Lines of this kind caught, against these fonts: marks of a lam-kaf ligature on the wrong component (Amiri), marks on the second glyph of
a multiple substitution (Amiri), a mark's cluster apart from its letter's (every font), and a substitution feature's
GPOS lookups left out (Reem Kufi's rclt kerning). Reruns give the same bytes. Needs fonttools and uharfbuzz (pip).
"""

import json
import os
import shutil
import sys

import uharfbuzz as hb
from fontTools import subset
from fontTools.ttLib import TTFont

LINES = [
    "المهندس فهد الفيصل أٍما، ضمن المبلغ، نورة الحربي، بين",
    "ريالات لٍذًا صدرت",
    "وبما يحق الإّحٌالة بين الحي مشيط، يلزم مشروع",
    "بشأن الطْابٌعِة بشأن الزائر فاطمة بنت عَلّي الظاهري طالب الشرط",
    "الدِكٌتًور عائشة الراشدٌ ظهر جلِسَة الظًل، ريالات قرر",
    "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ",
    "لا إله إلا الله",
    "الأُمَّةُ ـــ مـحـمـد",
    "لأ لإ لآ لا",
    "عبدالله بن عبدالعزيز",
    "ﷲ ﷺ",
    "قِسْطَاسٌ مُّسْتَقِيمٌ",
    "فِي ٱلْأَرْضِ",
    "بمٍوٍجب موعد الدائرة بخصوص الورشة الثّالّث إنها.",
    "علي بن منيرة الصاعدي حول المَحرَرُ:",
    "حًضرُة النْشُرة",
    "خلال البيانات بريدة؛ المنتجّ مستشار مع الضوء الطائف الجهة",
    "ولما بين",
    "لطيفة القحطانّيَ؛ إن؛ يجوز الضيافة الالتزام. التصميم",
    "ريال عبر التاريخُ الخرج. هنٌد بن وُلًيد الدوسري عند؟",
    "الحديقة عبر هـ: الإحالة",
    "خلف المنطقة الاعتذًارْ تجاه حّضرَة ضدْ البريدُ",
    "تجاه هـ لصالح الظل المتبقي البريد",
    "على هـ. سوف يَتْعيّنٍ الشرط، دوُنُ الحق.",
    "بًل علًمٍا اقترح مساعد",
    "المؤجر خلفْ هـ؛ حسب.",
    "كان ذكرٌ مادة، منذ الطٍابع.",
    "السٌيٌدٍة طلال.",
    "وقع المشروع المؤقت، إنه سجلت الخبر الرٍسوم المًطْلوُب اجتماع النهائي",
    "هللة. قدم الالتزام القديم.",
    "فوق هـ دوٌن المعرض المحترم",
    "أمِامِ مكة التقرير المْشارٍ من التحديث الباحة حّتى الشركة الورق",
    "مًعْالّيٌ الخدمة. وعليه قدمت الشركة، أيضا تبين الواحة لقد عيُن.",
    "استٌفسْرْ المكتبة النسخة شيخة بن طُلٌال الحْرٍبي؛ بيٌنً.",
    "الشٍرْط الهيئة، اعتمد السيارة الواضح",
    "الشَروِط نْظّام مْعاليً الضريبة. وِافق المراجعة الموْقٍع أحمد",
    "الأستاذ ماجٌد السُالٌمً بشأن هـ",
    "حول هـ لصالح الهيئة ينبع قبل.",
    "منيرة الظاهري الساعة ريال ثامر بّن عبدالله.",
    "على الحٌبر المعٌتًمَدٍ المدير سعد الطويرقَي؟",
    "سجلت الرخصة الثاني بل سلم النسخة الجهِة.",
    "بندر الشٌمري، حًولٌ الرياض استلم إدارة المُذكورُ، كانت نفيدكم معالي.",
    "المكٍتٍبة ريالات هللة في هـ خلال.",
    "رًيالات: لكن استلم العقار، لقد يعتمد مكتب",
    "حٍسبَ القطيف ما؟",
    "دون هـ الصادر البيع، الوثيقة بعد",
    "سوف نٍظم الصًادرَ. المتٌحفً المؤقت إًنُها",
    "فْي المِرْفقّات وافق البضاعة جلًسة لن قدمت الطابع المدير.",
    "الحق طلب المذكورة هللة وإذا.",
    "فاطٍمَة القحطاني، لكن نرفق المنتج؛ بين الموْرد الخبر",
    "الواحة الورق",
    "المدير طلال.",
    "نًحيْل الخطاب الحالي يوسف بنت فاطمة الشمري. حتى",
    "ريالا بَشًأِنٍ الأًحسًاء الماء بخصوص.",
    "بعد هـ",
    "السيد منُصُور العنزي نشرت المَدُة الحق، تجاه النّخُيّل مكتب، الجدول",
    "أمام الباحة",
    "أُجْلَ القصة العاجل؛ وفق الواحة المّختصة أمام هـ",
    "إلى الدعَوة الإيجار ضمُن هـ إلى المبنى",
    "وعليه أكدَ الحقوق أمام الخبر، المهندس ناصر السالم",
    "بموجب الظل الرياض صدرت المعرضً المْتبقٌي، البْرِنامِج",
    "المؤسسة فْوٌقُ هـ و غٍاب مسّاعد حسب المٍسابًقة.",
    "وفق البيع حول تبًوْكْ",
    "الشيخ هشام القّحطٌانيِ لذا اطُلعِ الموقع مٍنصُوًرِ القحطاني:",
    "اشتٌرٍى الملكية إِليه عمر الشمري، غادر",
    "على مبلغ جدة",
    "هللة لمّ خصص الشرط عبِرِ هـ",
    "الخطاب",
    "المدير تّركي",
    "رٌيًال حضرة عن",
    "من الطائف بشأن الهاتفُ المكرمة منذ الأحساء ضٌمِنٍ الآتي",
    "ضد المنورة حيث عن حِائٍلّ؛ عمر بنٌت ظافر العنزي",
    "اجتماع الزائر طلب",
    "عند الشاشة هللة من",
    "أمام عقد القطِيف: منّذّ مندوب سعادة",
    "عضو",
    "هللة أطلٌقتّ نائب المؤتمر تبين الثمن",
    "تجاه الحي، الإِيجارْ",
    "الموظف المؤقت لم اعًتِذر الشركة تقدم النظام الإيجار. بيَنَ",
    "ريّالاتٌ اعتًذَرٌ الملكية المذكور. فهد الغامدي ضيف الخالدي",
    "الكاتب المطلوب. أطلقت.",
    "فّقًط عدل التحديث",
    "حضرة الحيٍ",
    "أما حول المِبَنْى حْتُى هـ إِلِى",
    "صاحب مكتب الكاتبّ سعد الراشد قرر المقترح الحدث الأستاذ شيخة",
    "أيُضا راجِعْ الخادٌمِ؛ ريم.",
    "أيضا من الاجّتّماعَ؟",
    "بعد مْندًوبُ الواضح الشراء عن هـ: فاطمة",
    "الجدول الحي اعتمد الورق الفنية، السيد عمر",
    "عين المراجعة الجدول صرف الآثار الخطاب",
    "وأما نشر فقرة: التصريح المسْتنٌد الأوِل هللة",
    "ضّدٌ سكاكا بموجب العمُيلَ الدمام أّرفق النخيل",
    "نأمل الشريك القصة، وفق القصة",
    "صّدٍرًتّ المنطقة الشرط طلب مدير البريد، عبر",
    "السيدة سٌعِوُد الظاهري خّلْف المدة:",
    "السوق عبر هـ",
    "ضّمن الأصل سكاكا، آخَر مادْة، الإِحُالًة المذكورة السيدة نورة.",
    "جواهر الخالدي الخدمة العامة، بشأن الشّبكٌة",
    "بخصوص المراجعة نورة الفيصل لدى الخطاب الدمام المسابقة",
    "سلم الجدول الشارع شٍيُخة الثبيتي"
]

FONTS = [("Amiri-Regular.ttf", "amiri.ttf", "amiri"), ("Cairo[slnt,wght].ttf", "cairo.ttf", "cairo"), ("ReemKufi[wght].ttf", "reem-kufi.ttf", "reemkufi")]


def main():
    source = sys.argv[1]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "Idrak.Tests", "data", "fonts", "arabic")
    os.makedirs(out, exist_ok=True)
    unicodes = sorted({ord(c) for line in LINES for c in line} | set(range(0x0621, 0x0653)) | {0x20, 0x060C, 0x061B, 0x061F, 0x0640})
    references = []
    for name, cut, family in FONTS:
        options = subset.Options()
        options.layout_features = ["*"]
        options.name_IDs = ["*"]
        options.name_languages = ["*"]
        options.notdef_outline = True
        options.hinting = False
        options.glyph_names = False
        font = TTFont(os.path.join(source, name), recalcTimestamp=False)
        subsetter = subset.Subsetter(options)
        subsetter.populate(unicodes=unicodes)
        subsetter.subset(font)
        path = os.path.join(out, cut)
        font.save(path)
        shutil.copyfile(os.path.join(source, f"OFL-{family}.txt"), os.path.join(out, f"OFL-{family}.txt"))

        hb_font = hb.Font(hb.Face(hb.Blob.from_file_path(path)))
        for line in LINES:
            buf = hb.Buffer()
            buf.add_str(line)
            buf.direction = "rtl"
            buf.script = "Arab"
            buf.language = "ar"
            hb.shape(hb_font, buf)
            glyphs = [[i.codepoint, i.cluster, p.x_advance, p.y_advance, p.x_offset, p.y_offset] for i, p in zip(buf.glyph_infos, buf.glyph_positions)]
            references.append({"font": "arabic/" + cut, "text": line, "script": "Arab", "direction": "rtl", "language": "ar", "glyphs": glyphs})

    with open(os.path.join(out, "shaping.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(references, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")


if __name__ == "__main__":
    main()
