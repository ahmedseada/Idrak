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
LINES are random text from idrak data text (seed 11, harakat on a third of the words) and a few fixed lines. They
caught, against these fonts: marks of a lam-kaf ligature on the wrong component (Amiri), marks on the second glyph of
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
    "ريالات لٍذًا صدرت",
    "بشأن الأْرٌضِ بشأن السداد فاطمة بنت عَلّي الظاهري طالب",
    "خلال الإفادة بريدة؛ عليهّ مستشار مع الأتعاب الخرج",
    "وفق الصلح حول تبوك لْكُن",
    "فوِقٍ الخِبر أن اطلع",
    "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ",
    "لا إله إلا الله",
    "الأُمَّةُ ـــ مـحـمـد",
    "لأ لإ لآ لا",
    "عبدالله بن عبدالعزيز",
    "ﷲ ﷺ",
    "قِسْطَاسٌ مُّسْتَقِيمٌ",
    "فِي ٱلْأَرْضِ",
    "المهندس فهد الفيصل أٍما، ضمن المستأجر، نورة الحربي",
    "وبما يحق المّنٌطقٌة، بخصوص هـ، يلزم",
    "الدِكٌتًور عائشة الراشدٌ ظهر جلِسَة الطًعنْ بشأن الباحة قرر",
    "بمٍوٍجب موعد المركز بخصوص البلاغ الثّالّث إنها",
    "علي بن منيرة الصاعدي حول التَعوَيُض",
    "حًضرُة المْعُاملة المٌؤسٍسٌة",
    "ولما بين الحْكٍم",
    "لطيفة القحطانّيَ؛ إن؛ يجوز الحكم المدة الصك",
    "ريال عبر الثمن الطائف الغرامة عبر المجلس",
    "الكفالة أمام هـ،",
    "خلف البريد الجهة أحمد الصاعديْ أحْمٍدْ بن",
    "تجاه هـ لصالح الطعن النظامي؛ ضّدً",
    "على هـ سوف يَتْعيّنٍ الأجل، دوُنُ",
    "بًل علًمٍا ادعى مساعد",
    "الزواج خلفْ هـ؛ حسب",
    "كان ذكر القسمة فوق الشاهدٍ",
    "السٌيٌدٍة طلال الثبيتي",
    "وقع القضية المؤقت، إنه سجلت الرفض الوٍقف المعني، عبدالله",
    "هللة قدم المدة القديم",
    "فوق هـ دوٌن التظلم المحترم",
    "أمِامِ مكة المحكمة المشار، خالد السالم؛ حدد الطعن التظلم",
    "مًعْالّيٌ الطلب وعليه قدمت المؤسسة، أيضا تبين اليمين لقد",
    "التمٌس الشكوى التعويض نظام لُقٌد أْيٍد",
    "الأٍجْل الحي، صادق الإقرار المعنية",
    "المَوعِد نظام، بموجب الأمَانّة الصندوٌق المُوُظَفّ عنْ",
    "الأستاذ ماجٌد السالم، مستشّار",
    "حول هـ لصالح الحي ينبع قبل",
    "منيرة الظاهري القيمة ريال ثامر بّن عبدالله",
    "على المٌعاينِة النافذ غير صدر الإقرار",
    "سجلت المستند الثاني بل سلم المجلس الصنِدوٌق",
    "بندر الشٌمري، حًولٌ الرياض استلم إدارة المُذكورُ، كانت نفيدكم",
    "المهٍرٍ وفق بريدة خلف المدعي مشيط؛ الحق",
    "رًيالات لكن استلم الورثة، لقد يعتمد محكمة",
    "حٍسبَ القطيف ما دفع؟",
    "دون هـ الإحالة الصلح، الحقوق بعد",
    "سوف ضٍبط الإًحالَة الضمٌان اللازمة وحيث",
    "فْي الدِائْرة؛ وٌلٍيدٌ بن ريم الذيابٍيّ النيابة",
    "الشرطِ حيث صدرت المادة سوف رفض",
    "فاطٍمَة القحطاني، لكن نرفق عليه؛ بين الختم",
    "اليمين القرينة؛ وفق",
    "القاضي طلال الضويحي",
    "نًحيْل الملكية الحالي يوسف بنت فاطمة الشمري حتى",
    "ريالا بَشًأِنٍ الأًحسًاء الكاتب قبُل",
    "بعد هـ عٌبر",
    "السيد منُصُور العنزي قيدت السَاعُة الشرط، تجاه الإّقُرّار محكمة",
    "أمام الباحة الأستاذ",
    "أُجْلَ المخالفة العاجل؛ وفق اليميُنّ المطلوب غير سجلت حضرة",
    "إلى المحَافظة في الوثيقة مع القصُيم إلى الإقامة",
    "وعليه أقرَ الشروط أمام الخبر، المهندس ناصر",
    "بموجب الطعن الرياض صدرت التظلمً المْتبقٌي، الوْكِالة",
    "الموظف فْوٌقُ هـ و غٍاب مسّاعد حسب",
    "الشيخ هشام القّحطٌانيِ لذا اطُلعِ البينة مٍنصُوًرِ القحطاني",
    "اشتٌرٍى التركة إِليه عمر الشمري، غادر",
    "على مبلغ جدة",
    "هللة لمّ خصص الأجل عبِرِ هـ",
    "الملكية ؛ ريالات",
    "القاضي تّركي الظاهري",
    "رٌيًال حضرة عن",
    "من الطائف بشأن الشهادُة المكرمة منذ الأحساء ضمن",
    "ضد المنورة حيث عن حِائٍلّ؛ عمر بنٌت ظافر",
    "حكم السداد طلب",
    "عند المبلغ هللة من",
    "أمام عقد القطِيف منّذّ مندوب سعادة",
    "عضو في",
    "هللة رفعٌت العقار نائب المٌهنْدًس ريم",
    "تجاه الهوية، النِفقة",
    "الجلسة المؤقت لم رًفِض وْزَارة، ضدُ العقد، المدعى",
    "ريّالاتٌ رفًضَ الطلب الابتدائي؛ عين المادة المبلغ",
    "العقوبة المطلوب رفعت الغرامة",
    "فّقًط نقض الإدارة المستحق",
    "حضرة الهوٍية أمام",
    "أما أودع الإِقَامْة حْتُى هـ إِلِى",
    "صاحب محكمة المحامّيٍ خديجة الحّرٍبُيٍ سوف ثبت الحي، عُلما؟",
    "أيُضا نظِرْ المادٌة، عند مادة",
    "أيضا من الحّكّم مكٍة",
    "بعد مْندًوبُ الواضح التحكيم عن هـ",
    "القرار الهوية صادق القرينة الجزائية، السيد عمر",
    "عين التنفيذ القرار صرف الآثار الملكية",
    "وأما حكم فقرة الظرف الواجْباتٌ المبين",
    "ضّدٌ سكاكا بموجب المدُعى نّجرْانِ كان",
    "نأمل الموافقة المخالفة، وفق المخالفة",
    "صدرت المبلغ البلدية في هـ",
    "السيدة سٌعِوُد الظاهري خّلْف الساعة وزُارة",
    "الصورة عبر هـ",
    "ضّمن اللجنة سكاكا، آخر الطلبّ المهر ، بمٌا",
    "جواهر الخالدي الطلب العامة، بشأن اللّائحٌة",
    "بخصوص التنفيذ نورة الفيصل لدى الملكية الدمام القاضي"
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
