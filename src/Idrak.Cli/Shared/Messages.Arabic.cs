// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Cli.Shared;

internal static partial class Messages
{
    /// <summary>
    /// The Arabic catalog: each English message (as written at its call, placeholders included) and its translation.
    /// Translations start with an Arabic word where they can, so the line reads right to left as a whole; they carry
    /// no vowel marks (consoles draw combining marks poorly); command names, options, paths and units stay Latin.
    /// </summary>
    internal static readonly Dictionary<string, string> Arabic = new(StringComparer.Ordinal)
    {
        // idrak help: titles, groups and the common options.
        ["idrak: Idrak's command-line tool"] = "idrak: أداة سطر الأوامر لمكتبة إدراك",
        ["Usage: {0}"] = "الاستخدام: {0}",
        ["Aliases: {0}"] = "الأسماء المختصرة: {0}",
        ["Short forms: {0}"] = "الصيغ القصيرة: {0}",
        ["(The rest of this help is in English.)"] = "(بقية هذه المساعدة بالإنجليزية.)",
        ["Run 'idrak help COMMAND' (or 'idrak COMMAND --help') for a command's options; 'idrak help topics' for concept pages."] =
            "اكتب 'idrak help COMMAND' (أو 'idrak COMMAND --help') لعرض خيارات أمر، و'idrak help topics' لصفحات المفاهيم.",
        ["Other"] = "أخرى",
        ["Setup and health"] = "الإعداد والفحص",
        ["Run models"] = "تشغيل النماذج",
        ["Serve"] = "الخدمة عبر الشبكة",
        ["Models"] = "النماذج",
        ["Train"] = "التدريب",
        ["Data"] = "البيانات",
        ["Retrieval"] = "الاسترجاع",
        ["Measure"] = "القياس",
        ["Design"] = "التصميم",
        ["Developers"] = "للمطورين",
        ["Environment (idrak help env for all):"] = "متغيرات البيئة (اكتب idrak help env لعرضها كلها):",
        ["Every command"] = "كل الأوامر",
        [Help.CommonOptions] =
            "خيارات عامة:\n" +
            "  -d, --device NAME   الجهاز: cpu أو cuda:0 أو vulkan:1 أو hip:0 (الافتراضي: الجهاز الافتراضي)\n" +
            "  -P, --plugin PATH   تحميل مكتبة تسجل صيغا وعائلات نماذج وغير ذلك (يمكن تكراره)\n" +
            "  -j, --json          مخرجات تقرؤها البرامج (بالإنجليزية دائما)\n" +
            "  -q, --quiet         مخرجات أقل؛ و -v, --verbose: مخرجات أكثر\n" +
            "      --cache DIR     مجلد التخزين المؤقت (الافتراضي IDRAK_CACHE أو ~/.cache/idrak)\n" +
            "  -C, --config FILE   ملف الإعدادات (الافتراضي IDRAK_CONFIG أو ~/.idrak/config.json)\n" +
            "      --log FILE      كتابة كل سطر، ومعه السطور التفصيلية، إلى FILE أيضا\n" +
            "      @FILE           قراءة الوسائط من FILE، وسيط في كل سطر (@@text: النص @text كما هو)\n" +
            "  -O, --output FILE   كتابة المخرجات الرئيسية للأمر إلى FILE\n" +
            "      --format F      الصيغة: text أو json أو csv أو md (للجداول)\n" +
            "      --color WHEN    الألوان: auto أو always أو never (و NO_COLOR يغلب)؛ --plain: بلا رموز Unicode ولا سطر تقدم\n" +
            "      --offline       استخدام المخزن فقط؛ وأي تنزيل خطأ\n" +
            "      --threads N     عدد خيوط المعالج؛ --seed N: بذرة واحدة لأخذ العينات والخلط والتهيئة\n" +
            "      --timeout D     التوقف بعد مدة (30s أو 5m أو 1h30m)\n" +
            "      --lang L        لغة الرسائل: en أو ar (و \"lang\" في الإعدادات، IDRAK_LANG)؛ --lang-render auto أو visual أو logical\n" +
            "  -h, --help          هذه المساعدة؛ idrak -V, --version: الإصدارات؛ idrak help topics: صفحات المفاهيم\n",

        // Usage errors and questions.
        ["'{0}' needs a subcommand: {1}."] = "الأمر '{0}' يحتاج إلى أمر فرعي: {1}.",
        ["Unknown command '{0}'. Run 'idrak help' for the commands."] = "أمر غير معروف '{0}'. اكتب 'idrak help' لعرض الأوامر.",
        ["idrak {0}: gave up after --timeout {1} ({2})"] = "توقف الأمر idrak {0} بعد انقضاء --timeout {1} ({2})",
        ["Unknown option {0}."] = "خيار غير معروف {0}.",
        ["{0} needs a value."] = "الخيار {0} يحتاج إلى قيمة.",
        ["{0} takes no value."] = "الخيار {0} لا يأخذ قيمة.",
        ["add --yes (-y) to go ahead, or --dry-run to see what would happen"] = "أضف --yes (-y) للمتابعة، أو --dry-run لمعرفة ما سيحدث",
        ["add --yes (-y) to go ahead"] = "أضف --yes (-y) للمتابعة",
        ["add {0}"] = "أضف {0}",
        ["'{0}' needs an answer, and there is no terminal to ask on{1}; {2}."] = "السؤال '{0}' يحتاج إلى جواب، ولا توجد طرفية لطرحه{1}؛ {2}.",
        ["yes"] = "نعم",
        ["no"] = "لا",
        ["Stopping (Ctrl+C again to end at once)..."] = "جار الإيقاف (اضغط Ctrl+C مرة أخرى للإنهاء فورا)...",

        // idrak devices.
        ["Device"] = "الجهاز",
        ["Backend"] = "الواجهة الخلفية",
        ["Name"] = "الاسم",
        ["Memory"] = "الذاكرة",
        ["CUs"] = "وحدات الحوسبة",
        ["Lanes"] = "المسارات",
        ["Width"] = "العرض",
        ["Matrix"] = "المصفوفات",
        ["In a plain run"] = "في التشغيل العادي",
        ["Note"] = "ملاحظة",
        ["cannot start: {0}"] = "تعذر التشغيل: {0}",
        ["by name"] = "بالاسم",
        ["default device"] = "الجهاز الافتراضي",
        ["none found: {0}"] = "لم يعثر على شيء: {0}",

        // idrak doctor.
        ["ok"] = "سليم",
        ["info"] = "معلومة",
        ["warn"] = "تنبيه",
        ["FAIL"] = "فشل",
        ["fix: {0}"] = "الحل: {0}",
        ["why: {0}"] = "السبب: {0}",
        ["No command fixes what is left; see the fix lines above."] = "لا يوجد أمر يصلح ما تبقى؛ انظر سطور الحل أعلاه.",
        ["To fix:"] = "للإصلاح:",
        [" (safe: --fix --yes does it)"] = " (آمن: الأمر --fix --yes ينفذه)",
        ["Done: {0}"] = "تم: {0}",
        ["{0} check(s) failed, {1} warning(s)."] = "فشل من الفحوص: {0}، والتنبيهات: {1}.",
        ["Everything needed works; {0} warning(s)."] = "كل ما يلزم يعمل؛ والتنبيهات: {0}.",
        ["Everything works."] = "كل شيء يعمل.",
        [".NET runtime"] = "بيئة .NET",
        ["{0} ({1}) on {2}"] = "يعمل {0} ({1}) على {2}",
        ["install the .NET 10 runtime or SDK (dotnet --version prints 10.x)"] = "ثبت بيئة تشغيل .NET 10 أو حزمة SDK (يطبع dotnet --version الرقم 10.x)",
        ["Idrak is built for .NET 10; an older runtime cannot load it."] = "بنيت إدراك لـ .NET 10؛ ولا تستطيع بيئة أقدم تحميلها.",
        ["{0} logical processors, {1} float lanes per vector{2}, {3} memory for the process"] =
            "معالجات منطقية: {0}، ومسارات float في كل متجه: {1}{2}، وذاكرة للعملية: {3}",
        ["The CPU backend always works; its speed follows the vector width and the processor count."] =
            "واجهة المعالج المركزي تعمل دائما؛ وسرعتها تتبع عرض المتجه وعدد المعالجات.",
        ["{0} device(s) found"] = "عدد الأجهزة الموجودة: {0}",
        ["no devices found"] = "لم يعثر على أجهزة",
        ["Vulkan drivers"] = "مشغلات Vulkan",
        ["{0} driver file(s): {1}"] = "ملفات المشغلات ({0}): {1}",
        ["no Vulkan driver (ICD) files found"] = "لم يعثر على ملفات مشغلات Vulkan (ICD)",
        ["install the GPU's Vulkan driver (or a software one for tests); VK_ICD_FILENAMES can point at a driver file"] =
            "ثبت مشغل Vulkan لبطاقة الرسومات (أو مشغلا برمجيا للاختبارات)؛ ويمكن أن يشير VK_ICD_FILENAMES إلى ملف مشغل",
        ["The Vulkan loader reaches GPUs only through the driver (ICD) files it finds; without one there is no Vulkan device."] =
            "لا يصل محمل Vulkan إلى بطاقات الرسومات إلا عبر ملفات المشغلات (ICD) التي يجدها؛ وبدونها لا يوجد جهاز Vulkan.",
        ["install a GPU driver with CUDA support (only for GPUs that have it); the other backends work without it"] =
            "ثبت مشغل بطاقة رسومات يدعم CUDA (للبطاقات التي تدعمه فقط)؛ والواجهات الأخرى تعمل بدونه",
        ["install the Vulkan loader (libvulkan.so.1, vulkan-1.dll) and the GPU's Vulkan driver"] =
            "ثبت محمل Vulkan (libvulkan.so.1 أو vulkan-1.dll) ومشغل Vulkan لبطاقة الرسومات",
        ["install the HIP runtime and hipRTC (ROCm on Linux, the HIP SDK on Windows; HIP_PATH or ROCM_PATH finds them)"] =
            "ثبت بيئة HIP و hipRTC (حزمة ROCm على Linux، و HIP SDK على Windows؛ ويجدهما HIP_PATH أو ROCM_PATH)",
        ["install what the {0} backend needs (see the plug-in that registers it)"] = "ثبت ما تحتاجه الواجهة {0} (انظر الإضافة التي تسجلها)",
        ["CUDA drives GPUs through their CUDA driver; without it those GPUs can still run through Vulkan."] =
            "تشغل CUDA بطاقات الرسومات عبر مشغل CUDA؛ وبدونه يمكن لهذه البطاقات أن تعمل عبر Vulkan.",
        ["Vulkan compute runs Idrak's generated kernels on most GPUs (and on phones); it needs the loader and a driver."] =
            "تشغل حوسبة Vulkan النوى التي تولدها إدراك على معظم بطاقات الرسومات (وعلى الهواتف)؛ وتحتاج إلى المحمل ومشغل.",
        ["HIP runs kernels compiled at run time by hipRTC; without the runtime those GPUs can still run through Vulkan."] =
            "تشغل HIP نوى يترجمها hipRTC أثناء التشغيل؛ وبدون بيئتها يمكن لهذه البطاقات أن تعمل عبر Vulkan.",
        ["A registered backend adds devices of its kind."] = "كل واجهة خلفية مسجلة تضيف أجهزة من نوعها.",
        [", or leave the backend out with {0}"] = "، أو استبعد الواجهة بالمتغير {0}",
        ["update the device's driver{0}"] = "حدث مشغل الجهاز{0}",
        ["A device that is found but cannot start fails every model placed on it."] = "الجهاز الموجود الذي يتعذر تشغيله يفشل معه كل نموذج يوضع عليه.",
        ["{0} compute units"] = "وحدات حوسبة: {0}",
        ["subgroup {0}"] = "المجموعة الفرعية: {0}",
        ["kernels width {0}"] = "عرض النوى: {0}",
        ["matrix units"] = "وحدات مصفوفات",
        ["no matrix units"] = "بلا وحدات مصفوفات",
        ["listed"] = "مدرج",
        ["by name only"] = "بالاسم فقط",
        ["Each device's limits (memory, subgroup size, matrix units) decide which kernels run and what models fit."] =
            "حدود كل جهاز (الذاكرة وحجم المجموعة الفرعية ووحدات المصفوفات) تحدد النوى التي تعمل والنماذج التي تتسع.",
        ["Commands run on the default device unless --device (or the config's \"device\") names another."] =
            "تعمل الأوامر على الجهاز الافتراضي ما لم يسم --device (أو \"device\" في الإعدادات) جهازا آخر.",
        ["cache"] = "التخزين المؤقت",
        ["not created yet"] = "لم ينشأ بعد",
        ["{0} is not writable: {1}"] = "لا يمكن الكتابة في {0}: {1}",
        ["choose another folder with --cache DIR or IDRAK_CACHE"] = "اختر مجلدا آخر بالخيار --cache DIR أو بالمتغير IDRAK_CACHE",
        ["disk space"] = "مساحة القرص",
        ["{0} free for the cache"] = "المساحة الحرة للتخزين المؤقت: {0}",
        ["free some space, or move the cache to a larger disk with --cache DIR or IDRAK_CACHE"] =
            "حرر بعض المساحة، أو انقل التخزين المؤقت إلى قرص أكبر بالخيار --cache DIR أو بالمتغير IDRAK_CACHE",
        ["Models take 0.5 to 20 GB each; a full disk stops downloads part way."] = "يأخذ كل نموذج من 0.5 إلى 20 GB؛ والقرص الممتلئ يوقف التنزيل في منتصفه.",
        ["unknown ({0})"] = "غير معروفة ({0})",
        ["Models take 0.5 to 20 GB each."] = "يأخذ كل نموذج من 0.5 إلى 20 GB.",
        ["Downloaded models, measured kernel choices and compiled kernels are kept in the cache folder."] =
            "تحفظ في مجلد التخزين المؤقت النماذج المنزلة واختيارات النوى المقيسة والنوى المترجمة.",
        ["no existing parent folder"] = "لا يوجد مجلد أب موجود",
        ["config"] = "الإعدادات",
        [" (profile {0})"] = " (الملف الشخصي {0})",
        ["none at {0} (defaults apply; idrak init writes one)"] = "لا يوجد ملف في {0} (تطبق القيم الافتراضية؛ والأمر idrak init يكتب ملفا)",
        ["environment"] = "البيئة",
        ["no Idrak variables set"] = "لم يعين أي متغير من متغيرات إدراك",
        ["set: {0} (idrak env shows them)"] = "المعينة: {0} (يعرضها idrak env)",
        ["Environment variables change devices, tuning and memory for every Idrak program; a forgotten one explains surprises."] =
            "متغيرات البيئة تغير الأجهزة والضبط والذاكرة لكل برامج إدراك؛ ومتغير منسي يفسر المفاجآت.",
        ["The config file gives defaults (device, cache, plug-ins, model aliases) for every command."] =
            "ملف الإعدادات يعطي القيم الافتراضية (الجهاز والتخزين المؤقت والإضافات وأسماء النماذج المختصرة) لكل الأوامر.",
        ["build Turnip with -Dfreedreno-kmds=kgsl and export VK_ICD_FILENAMES=<its icd.d/freedreno_icd*.json> (installation/android-termux.md, step 7)"] =
            "ابن Turnip بالخيار -Dfreedreno-kmds=kgsl ثم export VK_ICD_FILENAMES=<its icd.d/freedreno_icd*.json> (الملف installation/android-termux.md، الخطوة 7)",
        ["On the phone the Vulkan loader finds the GPU only through the Turnip driver file that VK_ICD_FILENAMES names."] =
            "على الهاتف لا يجد محمل Vulkan بطاقة الرسومات إلا عبر ملف مشغل Turnip الذي يسميه VK_ICD_FILENAMES.",
        ["not set"] = "غير معين",
        ["names missing file(s): {0}"] = "يسمي ملفات غير موجودة: {0}",
        ["The GPU driver reaches the phone's GPU through its kernel device node; without access there is no GPU."] =
            "يصل مشغل الرسومات إلى بطاقة الهاتف عبر ملف الجهاز في النواة؛ وبدون صلاحية الوصول لا توجد بطاقة رسومات.",
        ["not found"] = "غير موجود",
        ["run on the phone (Termux, with proot sharing /dev); ls -la /dev/kgsl-3d0 should list it"] =
            "شغل الأداة على الهاتف (في Termux مع proot يشارك /dev)؛ ويجب أن يعرضه ls -la /dev/kgsl-3d0",
        ["not readable and writable: {0}"] = "لا يمكن القراءة منه والكتابة فيه: {0}",
        ["give the user access to the device node (installation/android-termux.md)"] = "امنح المستخدم صلاحية الوصول إلى ملف الجهاز (installation/android-termux.md)",
        ["readable and writable"] = "قابل للقراءة والكتابة",
        ["export DOTNET_GCHeapHardLimit=0x100000000 (4 GiB)"] = "عين المتغير: export DOTNET_GCHeapHardLimit=0x100000000 (4 GiB)",
        ["Android's address space is too small for .NET's default heap reservation; without a limit dotnet fails at start."] =
            "مساحة العناوين في Android أصغر من حجز الذاكرة الافتراضي في .NET؛ وبدون حد يفشل dotnet عند البدء.",
        ["Downloads go through the proxy these variables name (read by .NET's HTTP client); a wrong one stops every download."] =
            "تمر التنزيلات عبر الوسيط الذي تسميه هذه المتغيرات (يقرؤها عميل HTTP في .NET)؛ والوسيط الخاطئ يوقف كل تنزيل.",
        ["proxy"] = "الوسيط",
        ["none set"] = "لم يعين شيء",
        ["set: {0}"] = "المعينة: {0}",
        ["reachability"] = "الوصول",
        ["not checked (--offline)"] = "لم يفحص (--offline)",
        ["With --offline nothing is downloaded."] = "مع --offline لا ينزل شيء.",
        ["model hub"] = "مستودع النماذج",
        ["Models and datasets download from the hub (HF_ENDPOINT names a mirror)."] = "تنزل النماذج ومجموعات البيانات من المستودع (ويسمي HF_ENDPOINT نسخة بديلة).",
        ["repository API"] = "واجهة المستودعات",
        ["github: data sources read repositories through this API (GITHUB_API_URL)."] = "مصادر البيانات github: تقرأ المستودعات عبر هذه الواجهة (GITHUB_API_URL).",
        ["package feed"] = "مصدر الحزم",
        ["idrak update and new projects read the package feed."] = "يقرأ الأمر idrak update والمشاريع الجديدة مصدر الحزم.",
        ["{0} answered {1} in {2} ms"] = "أجاب {0} بالرمز {1} خلال {2} ms",
        ["{0} did not answer: {1}"] = "لم يجب {0}: {1}",
        ["check the connection and the proxy (HTTPS_PROXY); local models and --offline still work"] =
            "افحص الاتصال والوسيط (HTTPS_PROXY)؛ والنماذج المحلية و --offline تعمل مع ذلك",
        ["tokens"] = "رموز الدخول",
        ["Hugging Face: {0}, GitHub: {1}, Kaggle: {2} (idrak login stores one)"] = "رموز الدخول لـ Hugging Face: {0}، و GitHub: {1}، و Kaggle: {2} (يحفظها idrak login)",
        ["Gated and private models, private repositories and Kaggle datasets need a token; public ones do not."] =
            "النماذج المقيدة والخاصة والمستودعات الخاصة ومجموعات بيانات Kaggle تحتاج إلى رمز دخول؛ والعامة لا تحتاج.",
        ["set"] = "معين",

        // idrak chat.
        ["Chatting with {0}. /help lists the commands, /exit ends."] = "محادثة مع {0}. الأمر /help يعرض الأوامر، و /exit ينهي.",
        ["Continuing {0} messages from {1}."] = "متابعة {0} رسالة من {1}.",
        ["(the model asked for tools; give them with --tools FILE.dll)"] = "(طلب النموذج أدوات؛ أعطها بالخيار --tools FILE.dll)",
        ["[stopped]"] = "[توقف]",
        ["Commands:"] = "الأوامر:",
        [Commands.Run.ChatSession.CommandHelp] =
            "  /help                 هذه الأوامر\n" +
            "  /system [TEXT]        عرض موجه النظام أو تعيينه (و /system none يزيله)\n" +
            "  /reset                البدء من جديد (يبقى موجه النظام)\n" +
            "  /save FILE            حفظ المحادثة (بصيغة JSON للمحادثات)؛ /load FILE: متابعة محادثة محفوظة\n" +
            "  /file FILE            إضافة محتوى ملف نصي إلى الرسالة التالية\n" +
            "  /stats                السرعة والرموز والسياق المستخدم والذاكرة\n" +
            "  /think on|off|default وضع التفكير\n" +
            "  /tools                الأدوات التي يمكن للنموذج استدعاؤها\n" +
            "  /set [NAME VALUE]     عرض الإعدادات أو تغييرها: temperature و top-k و top-p و max-tokens و seed و think\n" +
            "  /copy                 نسخ آخر إجابة (إلى الحافظة في الطرفيات التي تسمح بذلك)\n" +
            "  /retry                الإجابة عن آخر رسالة من جديد\n" +
            "  /exit                 إنهاء المحادثة (أو /quit أو Ctrl+D)\n",
        ["A message starting with / is sent as typed when it starts with //."] = "الرسالة التي تبدأ بالرمز / ترسل كما كتبت إذا بدأت بالرمز //.",
        ["System prompt: {0}"] = "موجه النظام: {0}",
        ["No system prompt (set one with /system TEXT)."] = "لا يوجد موجه نظام (عينه بالأمر /system TEXT).",
        ["System prompt removed."] = "أزيل موجه النظام.",
        ["System prompt set."] = "عين موجه النظام.",
        ["Conversation cleared."] = "مسحت المحادثة.",
        ["Saved {0} messages to {1}."] = "حفظت {0} رسالة في {1}.",
        ["No file {0}."] = "لا يوجد ملف {0}.",
        ["Loaded {0} messages from {1}."] = "حملت {0} رسالة من {1}.",
        ["{0} will be sent with the next message."] = "سيرسل الملف {0} مع الرسالة التالية.",
        ["{0} needs a file name, e.g. {0} talk.json"] = "الأمر {0} يحتاج إلى اسم ملف، مثل {0} talk.json",
        ["Reasoning: {0}."] = "التفكير: {0}.",
        ["No tools; give an assembly with --tools FILE.dll."] = "لا توجد أدوات؛ أعط مكتبة بالخيار --tools FILE.dll.",
        ["/set {0} needs a value, e.g. /set temperature 0.7"] = "الأمر /set {0} يحتاج إلى قيمة، مثل /set temperature 0.7",
        ["Nothing to retry yet."] = "لا يوجد ما يعاد بعد.",
        ["Unknown command {0}; /help lists them."] = "أمر غير معروف {0}؛ والأمر /help يعرض الأوامر.",
        ["No answer to copy yet."] = "لا توجد إجابة لنسخها بعد.",
        ["Copied the last answer ({0} characters) to the clipboard."] = "نسخت آخر إجابة ({0} حرفا) إلى الحافظة.",
        ["Model     {0}"] = "النموذج   {0}",
        ["Last      {0} tokens at {1:F1} tokens/s; prompt {2} tokens in {3:F2} s"] = "الأخيرة   {0} رمزا بسرعة {1:F1} tokens/s؛ والموجه {2} رمزا في {3:F2} s",
        ["Context   {0} of {1} tokens used"] = "السياق    استخدم {0} من {1} رمزا",
        ["Session   {0} answers, {1} tokens, {2:F1} tokens/s on average, {3} messages"] = "الجلسة    {0} إجابة، و {1} رمزا، بمتوسط {2:F1} tokens/s، و {3} رسالة",
        ["Memory    process {0} MiB, managed {1} MiB"] = "الذاكرة   العملية {0} MiB، والمدارة {1} MiB",

        // Command summaries (idrak help, idrak help COMMAND).
        ["Datasets: show, count, download, build (and preview, validate, stats, convert, dedupe, split, sample, mix)"] =
            "مجموعات البيانات: عرض وعد وتنزيل وبناء (ومعاينة وتحقق وإحصاءات وتحويل وإزالة تكرار وتقسيم وعينات ومزج)",
        ["First rows, columns and their types of a data file (Parquet, JSON Lines, CSV)"] = "أول صفوف ملف بيانات وأعمدته وأنواعها (Parquet و JSON Lines و CSV)",
        ["Check rows against the shape a command expects (chat, preference or table), with the first bad rows"] =
            "فحص الصفوف مقابل الشكل الذي يتوقعه أمر (محادثة أو تفضيل أو جدول)، مع أول الصفوف الخاطئة",
        ["Row lengths in characters, words and tokens, a length histogram, and the rows over a context length"] =
            "أطوال الصفوف بالحروف والكلمات والرموز، ومدرج للأطوال، والصفوف التي تتجاوز طول سياق",
        ["Convert rows between CSV, JSON Lines and JSON (Parquet is read), and chat layouts to chat rows"] =
            "تحويل الصفوف بين CSV و JSON Lines و JSON (وقراءة Parquet)، وتخطيطات المحادثة إلى صفوف محادثة",
        ["Remove duplicate rows (exact, or near: text compared without case, spacing and punctuation)"] =
            "إزالة الصفوف المكررة (المتطابقة، أو المتقاربة: نص يقارن دون حالة الأحرف والمسافات وعلامات الترقيم)",
        ["Split rows into train, validation and test files with a seed (stratified by a column with -t)"] =
            "تقسيم الصفوف إلى ملفات تدريب وتحقق واختبار ببذرة (مع طبقات حسب عمود بالخيار -t)",
        ["A random (or, with -t, stratified) sample of rows"] = "عينة عشوائية من الصفوف (أو عينة طبقية بالخيار -t)",
        ["Assemble a training set from several sources with weights (a dataset recipe)"] = "تجميع مجموعة تدريب من عدة مصادر بأوزان (وصفة مجموعة بيانات)",
        ["Layers, shapes, parameters, FLOPs and memory of a network.json"] = "طبقات ملف network.json وأشكالها ومعاملاتها و FLOPs والذاكرة",
        ["Design a network and its training setup for a data set"] = "تصميم شبكة وإعداد تدريبها لمجموعة بيانات",
        ["Draw a network.json as text, Mermaid or SVG"] = "رسم ملف network.json نصا أو بصيغة Mermaid أو SVG",
        ["Trains a small built-in sample in seconds on the device, with the speed"] = "تدريب مثال صغير مدمج في ثوان على الجهاز، مع السرعة",
        ["Writes the generated GPU kernels (PTX, SPIR-V, HIP source) for debugging"] = "كتابة نوى بطاقة الرسومات المولدة (PTX و SPIR-V ومصدر HIP) لتصحيح الأخطاء",
        ["Which kernel each operation runs on a device: registered, its own, composed, host fallback or none"] = "النواة التي تشغل كل عملية على جهاز: مسجلة، أو نواة الجهاز، أو مركبة، أو بديل المضيف، أو لا شيء",
        ["Kernels on {0} ({1}): {2}"] = "النوى على {0} ({1}): {2}",
        ["Operation"] = "العملية",
        ["Kernel"] = "النواة",
        ["Without its own"] = "دون نواة خاصة",
        ["registered"] = "مسجلة",
        ["device"] = "الجهاز",
        ["composed"] = "مركبة",
        ["host"] = "المضيف",
        ["none"] = "لا شيء",
        ["Writes a new project: console, webapi, rag or plugin"] = "كتابة مشروع جديد: console أو webapi أو rag أو plugin",
        ["Imports an ONNX model into Idrak layers and saves a model package (.ikm)"] = "استيراد نموذج ONNX إلى طبقات إدراك وحفظ حزمة نموذج (.ikm)",
        ["Exports a model package (.ikm) or a network JSON to ONNX"] = "تصدير حزمة نموذج (.ikm) أو ملف JSON لشبكة إلى ONNX",
        ["Checks an ONNX model: Idrak's import against a round trip and against ONNX Runtime"] = "فحص نموذج ONNX: استيراد إدراك مقابل رحلة ذهاب وعودة ومقابل ONNX Runtime",
        ["An interactive prompt for idrak commands, with history and completion"] = "موجه تفاعلي لأوامر idrak، مع السجل والإكمال",
        ["Runs the library's tests from a source checkout"] = "تشغيل اختبارات المكتبة من نسخة المصدر",
        ["Runs an idrak command with telemetry printed live or written to JSON Lines"] = "تشغيل أمر idrak مع القياسات مطبوعة مباشرة أو مكتوبة بصيغة JSON Lines",
        ["Sizes and paths of the caches (models, tuning, kernels, downloads)"] = "أحجام التخزين المؤقت ومساراته (النماذج والضبط والنوى والتنزيلات)",
        ["Clears one part of the cache (models, tuning, kernels) or all of it"] = "مسح جزء من التخزين المؤقت (النماذج أو الضبط أو النوى) أو كله",
        ["Prints a shell completion script (bash, zsh, fish, pwsh)"] = "طباعة نص الإكمال للصدفة (bash أو zsh أو fish أو pwsh)",
        ["Prints a config value (dotted keys, e.g. aliases.qwen.model)"] = "طباعة قيمة من الإعدادات (مفاتيح بنقاط، مثل aliases.qwen.model)",
        ["Save environment variables for every idrak run, in any terminal (asks for them, with suggested values)"] =
            "حفظ متغيرات البيئة لكل تشغيل للأداة idrak في أي طرفية (يسأل عنها ويقترح قيمها)",
        ["Remove every saved environment variable: all back to their defaults (asks first)"] =
            "إزالة كل متغيرات البيئة المحفوظة لتعود كلها إلى قيمها الافتراضية (يسأل أولا)",
        ["Remove environment variables saved by idrak env set"] = "إزالة متغيرات البيئة التي حفظها idrak env set",
        ["Sets a config value (JSON values such as [\"a.dll\"] or 4 are kept as JSON)"] = "تعيين قيمة في الإعدادات (تحفظ قيم JSON مثل [\"a.dll\"] أو 4 بصيغة JSON)",
        ["Removes a config value"] = "إزالة قيمة من الإعدادات",
        ["Every config value (or a profile's), and the file's path"] = "كل قيم الإعدادات (أو قيم ملف شخصي)، ومسار الملف",
        ["Every device: backend, name, memory, compute units, subgroup size, matrix units, kernel width"] =
            "كل الأجهزة: الواجهة الخلفية والاسم والذاكرة ووحدات الحوسبة وحجم المجموعة الفرعية ووحدات المصفوفات وعرض النوى",
        ["What works on this machine and what to fix: runtime, backends, devices, caches, disk, environment"] =
            "ما يعمل على هذا الجهاز وما يجب إصلاحه: بيئة التشغيل والواجهات الخلفية والأجهزة والتخزين المؤقت والقرص والبيئة",
        ["Every environment variable Idrak reads: value, default and meaning"] = "كل متغيرات البيئة التي تقرؤها إدراك: القيمة والافتراضي والمعنى",
        ["Writes the config file: default device, cache folder, model aliases"] = "كتابة ملف الإعدادات: الجهاز الافتراضي ومجلد التخزين المؤقت وأسماء النماذج المختصرة",
        ["Stores a token for the Hugging Face hub, GitHub or Kaggle where the library reads it"] = "حفظ رمز دخول لمستودع Hugging Face أو GitHub أو Kaggle حيث تقرؤه المكتبة",
        ["Removes a stored token (Hugging Face hub, GitHub or Kaggle)"] = "إزالة رمز دخول محفوظ (مستودع Hugging Face أو GitHub أو Kaggle)",
        ["Everything registered: formats, families, RoPE scalings, tool-call formats, ops, devices"] =
            "كل ما هو مسجل: الصيغ والعائلات وتحجيمات RoPE وصيغ استدعاء الأدوات والعمليات والأجهزة",
        ["Every registered weight, KV cache, checkpoint, dataset and tool-call format"] = "كل الصيغ المسجلة: الأوزان وذاكرة KV ونقاط الحفظ ومجموعات البيانات واستدعاء الأدوات",
        ["A report (Markdown and JSON) of the machine, devices, drivers, tests and benchmarks"] = "تقرير (Markdown و JSON) عن الجهاز والأجهزة والمشغلات والاختبارات وقياسات الأداء",
        ["The Android (Termux) setup steps that are safe to automate, then the phone checks"] = "خطوات إعداد Android (في Termux) الآمنة للأتمتة، ثم فحوص الهاتف",
        ["Whether a newer version of the tool exists, and the command to install it"] = "هل يوجد إصدار أحدث من الأداة، والأمر الذي يثبته",
        ["Speed of a model (tokens per second, GFLOP/s, memory) or of the kernels on a device"] = "سرعة نموذج (رموز في الثانية و GFLOP/s والذاكرة) أو سرعة النوى على جهاز",
        ["Compare token ids, chat templates, logits and greedy output with a transformers reference"] =
            "مقارنة أرقام الرموز وقوالب المحادثة والقيم اللوغاريتمية والمخرجات الجشعة بمرجع transformers",
        ["Answer metrics of a chat model on held-out conversations (accuracy, exact match, F1)"] = "مقاييس إجابات نموذج محادثة على محادثات محجوزة (الدقة والتطابق التام و F1)",
        ["Perplexity of a text under a model (compare weight formats and fine-tunes)"] = "حيرة نص تحت نموذج (لمقارنة صيغ الأوزان والضبط الدقيق)",
        ["Time per layer, operation and kernel of one decoding step"] = "الوقت لكل طبقة وعملية ونواة في خطوة فك ترميز واحدة",
        ["The kernel choices measured on this machine (the tuning caches); --reset clears them"] = "اختيارات النوى المقيسة على هذا الجهاز (مخازن الضبط)؛ والخيار --reset يمسحها",
        ["Give a model (and its weight and KV formats) a short name, kept in the config"] = "إعطاء نموذج (وصيغ أوزانه وذاكرة KV) اسما مختصرا يحفظ في الإعدادات",
        ["The model aliases in the config"] = "أسماء النماذج المختصرة في الإعدادات",
        ["Remove a model alias from the config"] = "إزالة اسم مختصر لنموذج من الإعدادات",
        ["Convert between GGUF and Hugging Face folders (safetensors bf16, f16, f32)"] = "التحويل بين GGUF ومجلدات Hugging Face (ملفات safetensors بصيغة bf16 أو f16 أو f32)",
        ["Which tensors differ between two checkpoints, and by how much"] = "الموترات المختلفة بين نقطتي حفظ، ومقدار الاختلاف",
        ["Supported model families and what each supports (windows, soft-capping, RoPE scalings, experts, GGUF)"] =
            "عائلات النماذج المدعومة وما تدعمه كل منها (النوافذ والتحديد الناعم وتحجيمات RoPE والخبراء و GGUF)",
        ["Tensor names, shapes, types and metadata of a GGUF, safetensors or .ikm file"] = "أسماء الموترات وأشكالها وأنواعها وبياناتها الوصفية في ملف GGUF أو safetensors أو .ikm",
        ["Cached models with sizes, formats and last use"] = "النماذج المخزنة مع أحجامها وصيغها وآخر استخدام",
        ["Memory per weight and KV format at a context length, against each device's memory"] = "الذاكرة لكل صيغة أوزان وذاكرة KV عند طول سياق، مقابل ذاكرة كل جهاز",
        ["Merge a LoRA or DoRA adapter into the base weights"] = "دمج محول LoRA أو DoRA في الأوزان الأساسية",
        ["Download a Hugging Face model or a GGUF file into the cache, with progress and resume"] = "تنزيل نموذج من Hugging Face أو ملف GGUF إلى التخزين المؤقت، مع التقدم والاستئناف",
        ["Pack a model's weights (int8, int4, bf16 or any registered format): size and a quick perplexity check"] =
            "ضغط أوزان نموذج (int8 أو int4 أو bf16 أو أي صيغة مسجلة): الحجم وفحص سريع للحيرة",
        ["Remove a cached model"] = "إزالة نموذج مخزن",
        ["Search the Hugging Face hub for models Idrak can load"] = "البحث في مستودع Hugging Face عن نماذج تستطيع إدراك تحميلها",
        ["Family, parameters, layers, context, vocabulary, RoPE, windows, chat template, files and license of a model"] =
            "عائلة النموذج ومعاملاته وطبقاته وسياقه ومفرداته و RoPE ونوافذه وقالب محادثته وملفاته وترخيصه",
        ["Check a cached model's files: sizes, hashes where the hub gives them, readable tensors"] =
            "فحص ملفات نموذج مخزن: الأحجام، والبصمات حيث يعطيها المستودع، وقابلية قراءة الموترات",
        ["Answer a question from an index's passages with a chat model, citing them"] = "الإجابة عن سؤال من مقاطع فهرس بنموذج محادثة، مع الإشارة إليها",
        ["Retrieval quality of an index (hit rate, MRR) on question/passage pairs"] = "جودة الاسترجاع لفهرس (معدل الإصابة و MRR) على أزواج سؤال ومقطع",
        ["Chunk a folder's text files and build a search index (BM25, and vectors with --model)"] = "تقطيع ملفات نصية في مجلد وبناء فهرس بحث (BM25، ومتجهات مع --model)",
        ["The best passages of an index for a query, with scores (no generation)"] = "أفضل مقاطع فهرس لاستعلام، مع الدرجات (بلا توليد)",
        ["The coding agent: reads, searches, edits files and runs commands in a folder"] = "وكيل البرمجة: يقرأ الملفات ويبحث فيها ويحررها وينفذ الأوامر في مجلد",
        ["Answers many prompts (JSON Lines) with batched generation, resumable"] = "الإجابة عن موجهات كثيرة (JSON Lines) بتوليد على دفعات، مع إمكانية الاستئناف",
        ["Interactive chat with a model, streamed with tokens per second"] = "محادثة تفاعلية مع نموذج، تبث مع عدد الرموز في الثانية",
        ["The same prompt on two models (or two settings of one), answers side by side with speed"] = "الموجه نفسه على نموذجين (أو إعدادين لنموذج)، والإجابات جنبا إلى جنب مع السرعة",
        ["Embeddings of lines or files to JSON or .npy"] = "تضمينات السطور أو الملفات إلى JSON أو .npy",
        ["One answer to a prompt (argument, --input file or piped), then exit"] = "إجابة واحدة عن موجه (وسيط أو ملف --input أو مدخل منقول)، ثم الخروج",
        ["Tokens and ids of a text (--chat renders the chat template first, --count only counts)"] = "رموز نص وأرقامها (الخيار --chat يطبق قالب المحادثة أولا، و --count يعد فقط)",
        ["The chat template, its tool-call format and a rendered sample conversation with a tool call"] = "قالب المحادثة وصيغة استدعاء الأدوات فيه ومحادثة مثال مع استدعاء أداة",
        ["Raw completion of a text, without the chat template"] = "إكمال خام لنص، دون قالب المحادثة",
        ["The tools an assembly registers: names, descriptions and parameters"] = "الأدوات التي تسجلها مكتبة: الأسماء والأوصاف والمعاملات",
        ["Calls each tool of an assembly with sample arguments and shows the results"] = "استدعاء كل أداة في مكتبة بوسائط مثال وعرض النتائج",
        ["Serve the tools of an assembly over MCP (standard input/output)"] = "تقديم أدوات مكتبة عبر MCP (المدخل والمخرج القياسيان)",
        ["Check a running server (Idrak or any compatible one): reachable, APIs, models, latency"] = "فحص خادم يعمل (إدراك أو أي خادم متوافق): الوصول والواجهات والنماذج وزمن الاستجابة",
        ["Models of a running server: loaded or not, memory, context, last use"] = "نماذج خادم يعمل: محملة أو لا، والذاكرة والسياق وآخر استخدام",
        ["Stop a running server; open requests finish first"] = "إيقاف خادم يعمل؛ وتنتهي الطلبات المفتوحة أولا",
        ["Load a model in a running server now, instead of on its first request"] = "تحميل نموذج في خادم يعمل الآن، بدلا من تحميله عند أول طلب",
        ["Unload a model from a running server, freeing its memory"] = "إلغاء تحميل نموذج من خادم يعمل، وتحرير ذاكرته",
        ["Call an endpoint of a running server (for scripts and checks)"] = "استدعاء نقطة نهاية في خادم يعمل (للنصوص البرمجية والفحوص)",
        ["Make an API key for serve (printed once; the config keeps only its hash)"] = "إنشاء مفتاح API للأمر serve (يطبع مرة واحدة؛ وتحفظ الإعدادات بصمته فقط)",
        ["The API keys serve accepts (names and dates; the keys are not stored)"] = "مفاتيح API التي يقبلها serve (الأسماء والتواريخ؛ والمفاتيح لا تحفظ)",
        ["Remove an API key (servers started afterwards no longer accept it)"] = "إزالة مفتاح API (الخوادم التي تبدأ بعد ذلك لا تقبله)",
        ["Serve models over the chat API and the OpenAI-style API on one port"] = "تقديم النماذج عبر واجهة المحادثة والواجهة بأسلوب OpenAI على منفذ واحد",
        ["Serve models and open a small web chat page in the browser"] = "تقديم النماذج وفتح صفحة محادثة صغيرة في المتصفح",
        ["Train a network from a builder JSON on a CSV or an image folder; writes a model package (.ikm)"] =
            "تدريب شبكة من ملف JSON للبناء على ملف CSV أو مجلد صور؛ وكتابة حزمة نموذج (.ikm)",
        ["Continue a training run from its last checkpoint (the remaining epochs, or --epochs N more)"] =
            "متابعة تدريب من آخر نقطة حفظ (الحقب المتبقية، أو --epochs N إضافية)",
        ["Training runs (from their JSON Lines logs): epochs, best epoch and loss, time, status"] = "عمليات التدريب (من سجلاتها بصيغة JSON Lines): الحقب وأفضل حقبة والخسارة والوقت والحالة",
        ["One training run: settings, loss curves (text plot), the epochs, best epoch and time"] = "عملية تدريب واحدة: الإعدادات ومنحنيات الخسارة (رسم نصي) والحقب وأفضل حقبة والوقت",
        ["Training runs side by side: settings, best epoch and loss, time, and their validation curves"] =
            "عمليات تدريب جنبا إلى جنب: الإعدادات وأفضل حقبة والخسارة والوقت ومنحنيات التحقق",
        ["Run a model package (.ikm) on new rows (CSV, JSON Lines, Parquet) or images and write the predictions"] =
            "تشغيل حزمة نموذج (.ikm) على صفوف جديدة (CSV أو JSON Lines أو Parquet) أو صور وكتابة التنبؤات",
        ["Bundle a network, its weights, scalers and tokenizer from a folder into one model package (.ikm)"] =
            "جمع شبكة وأوزانها ومقاييسها ومقطع رموزها من مجلد في حزمة نموذج واحدة (.ikm)",
        ["Distil a teacher model into a student: its token probabilities (on the fly or precomputed) or its answers"] =
            "تقطير نموذج معلم في نموذج طالب: احتمالات رموزه (أثناء التدريب أو محسوبة مسبقا) أو إجاباته",
        ["Fine-tune language models (LoRA, QLoRA, DoRA, DPO/ORPO/SimPO); evaluate, chat, export, download, info"] =
            "الضبط الدقيق لنماذج اللغة (LoRA و QLoRA و DoRA و DPO/ORPO/SimPO)؛ والتقييم والمحادثة والتصدير والتنزيل والمعلومات",
        ["Write a commented tune.json (model, data, adapter and training settings) for idrak tune --config"] =
            "كتابة ملف tune.json بتعليقات (النموذج والبيانات والمحول وإعدادات التدريب) للأمر idrak tune --config",
        ["Versions of the tool, the libraries, the .NET runtime and the device drivers"] = "إصدارات الأداة والمكتبات وبيئة .NET ومشغلات الأجهزة",
    };
}
