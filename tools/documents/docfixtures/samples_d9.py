"""D9, forms: every field kind (text single and multi-line, comb, password, maximum length; check box; radio group;
list box, multi-select list and combo box; push button; signature), filled and empty, with appearance streams and with
/NeedAppearances only; values, flags and DA inherited across a field tree; an XFA form with its datasets; and the
filled form flattened (each widget's appearance drawn into the page, the fields gone), whose look pdfium's drawing of
the filled form gives. The forms completed (2026-11-19): turned widgets and border styles (pdfium makes their
appearances), a rich text value (/RV), fields held inside other objects (object streams, a field's dictionary written in
its parent's /Kids or in /Fields), a field whose text its /DA font lacks, and an XFA form whose template gives sections,
captions, kinds, choices, bindings and scripts.

A field, as `fields` lists it: `name` (full, dotted), `partialName`, `kind` (text, checkbox, radio, list, combo,
pushbutton, signature), `value` (text, the on state's name for a check box or radio group, null when off or empty),
`values` (a multi-select list), `defaultValue`, `options` (`export` and `display`), `selected` (option indices),
`flags` (the names of the flags set) and `flagsValue`, `maxLength`, `alignment` (left, centre, right), `font` and
`fontSize` (from its DA), `tooltip`, `inherited` (the keys it takes from an ancestor), and `widgets`: `page`, `box`
(the page as shown, top-left origin), `rect` (as written), `state` (/AS), `onState`, `hasAppearance`, `appearanceText`
(the text its appearance shows), `rotation` (/MK /R, when set). Also, when set: `border` (`style`: solid, dashed, beveled,
inset, underline; `width`; `dashes`) and `richText` (the runs /RV gives over /DS: `text`, `bold`, `italic`, `underline`,
`strikeOut`, `size`, `colour` [r, g, b], `family`). An XFA sample's `form.xfa.template` gives `sections` (`name`, `path`,
`parent`), `fields` (`name`, `path`, `kind`, `section`, `caption`, `binding`, `value`, `options`, `flags`, `field`: the
field of the document's fields holding its value) and `scripts` (how many, none run)."""
from .common import Sample, base_expected
from .document import Doc
from .fonts import Standard14
from .pdf import N, Stream, String, U16, sha256
from .filters import flate

D9 = ["D9"]

FLAGS = {"readOnly": 1, "required": 2, "noExport": 3, "multiline": 13, "password": 14, "noToggleToOff": 15,
         "radio": 16, "pushbutton": 17, "combo": 18, "edit": 19, "sort": 20, "fileSelect": 21, "multiSelect": 22,
         "doNotSpellCheck": 23, "doNotScroll": 24, "comb": 25, "radiosInUnison": 26, "richText": 26,
         "commitOnSelChange": 27}


def flag_value(*names):
    v = 0
    for n in names:
        v |= 1 << (FLAGS[n] - 1)
    return v


def flag_names(value, kind):
    out = []
    groups = {"text": ("multiline", "password", "fileSelect", "doNotSpellCheck", "doNotScroll", "comb", "richText"),
              "checkbox": ("noToggleToOff", "radio", "pushbutton", "radiosInUnison"),
              "radio": ("noToggleToOff", "radio", "pushbutton", "radiosInUnison"),
              "pushbutton": ("noToggleToOff", "radio", "pushbutton", "radiosInUnison"),
              "list": ("combo", "edit", "sort", "multiSelect", "doNotSpellCheck", "commitOnSelChange"),
              "combo": ("combo", "edit", "sort", "multiSelect", "doNotSpellCheck", "commitOnSelChange"),
              "signature": ()}
    for n in ("readOnly", "required", "noExport") + groups[kind]:
        if value & (1 << (FLAGS[n] - 1)):
            out.append(n)
    return out


class Form:
    """Fields and widgets on a document's pages, their appearance streams, and the expected fields."""

    def __init__(self, doc, appearances=True, need_appearances=False):
        self.doc = doc
        self.appearances = appearances
        self.need_appearances = need_appearances
        self.helv = doc.font(Standard14, "Helvetica")
        self.zadb = doc.font(Standard14, "ZapfDingbats", encoding=None)
        self.fields = []  # top-level field refs
        self.expected = []
        self.dr = {"Font": {"Helv": self.helv.ref, "ZaDb": self.zadb.ref}}

    # -- appearance streams ----------------------------------------------------------------------------------------------

    def _ap(self, w, h, content, fonts=("Helv",)):
        res = {"Font": {k: self.dr["Font"][k] for k in fonts}} if fonts else {}
        return self.doc.pdf.add(Stream({"Type": N("XObject"), "Subtype": N("Form"), "BBox": [0, 0, w, h],
                                        "Resources": res, "Filter": N("FlateDecode")}, flate(content), content))

    def _width(self, text, size):
        return sum(g.width for g in self.helv.glyphs(text)) / 1000 * size

    def _esc(self, text):
        from .pdf import escape_literal

        return escape_literal(text.encode("latin-1"))

    def text_ap(self, w, h, lines, size, align=0, comb=None, colour=b"0 g", border=True):
        out = [b"/Tx BMC", b"q"]
        if border:
            out += [b"0.85 0.9 0.97 rg 0 0 %s %s re f" % (_n(w), _n(h)), b"0.5 G 0.75 w 0.375 0.375 %s %s re S"
                    % (_n(w - 0.75), _n(h - 0.75))]
        out += [b"1 1 %s %s re W n" % (_n(w - 2), _n(h - 2)), b"BT", b"/Helv %s Tf" % _n(size), colour]
        if comb:
            cell = w / comb
            text = lines[0]
            x0 = 0.0
            first = True
            for k, ch in enumerate(text):
                x = k * cell + (cell - self._width(ch, size)) / 2
                y = (h - size * 0.72) / 2
                out.append(b"%s %s Td %s Tj" % (_n(x - x0), _n(y if first else 0), self._esc(ch)))
                x0 = x
                first = False
        else:
            leading = size * 1.15
            y = (h - size * 0.72) / 2 if len(lines) == 1 else h - 2 - size
            x_prev, y_prev = 0.0, 0.0
            for k, line in enumerate(lines):
                tw = self._width(line, size)
                x = 2 if align == 0 else (w - tw) / 2 if align == 1 else w - 2 - tw
                yy = y - k * leading
                out.append(b"%s %s Td %s Tj" % (_n(x - x_prev), _n(yy - y_prev), self._esc(line)))
                x_prev, y_prev = x, yy
        out += [b"ET", b"Q", b"EMC"]
        return self._ap(w, h, b"\n".join(out) + b"\n")

    def list_ap(self, w, h, options, selected, size):
        out = [b"/Tx BMC", b"q", b"1 1 1 rg 0 0 %s %s re f" % (_n(w), _n(h)), b"0.5 G 0.75 w 0.375 0.375 %s %s re S"
               % (_n(w - 0.75), _n(h - 0.75))]
        leading = size * 1.25
        for k in selected:
            out.append(b"0.6 0.75 0.95 rg 1 %s %s %s re f" % (_n(h - 1 - (k + 1) * leading), _n(w - 2), _n(leading)))
        out += [b"BT", b"/Helv %s Tf" % _n(size), b"0 g"]
        y_prev = 0.0
        for k, (_, display) in enumerate(options):
            y = h - 1 - (k + 1) * leading + size * 0.3
            out.append(b"%s %s Td %s Tj" % (_n(3 if k == 0 else 0), _n(y - y_prev), self._esc(display)))
            y_prev = y
        out += [b"ET", b"Q", b"EMC"]
        return self._ap(w, h, b"\n".join(out) + b"\n")

    def check_ap(self, w, h, char, on):
        if not on:
            return self._ap(w, h, b"q 1 1 1 rg 0 0 %s %s re f 0.5 G 0.75 w 0.375 0.375 %s %s re S Q\n"
                            % (_n(w), _n(h), _n(w - 0.75), _n(h - 0.75)), fonts=())
        size = h * 0.8
        content = (b"q 1 1 1 rg 0 0 %s %s re f 0.5 G 0.75 w 0.375 0.375 %s %s re S Q\n"
                   % (_n(w), _n(h), _n(w - 0.75), _n(h - 0.75))
                   + b"q BT /ZaDb %s Tf 0 0.3 0.6 rg %s %s Td (%s) Tj ET Q\n"
                   % (_n(size), _n(w * 0.15), _n(h * 0.18), char.encode("latin-1")))
        return self._ap(w, h, content, fonts=("ZaDb",))

    def button_ap(self, w, h, label, size=11):
        tw = self._width(label, size)
        content = (b"q 0.25 0.45 0.75 rg 0 0 %s %s re f Q\nq BT /Helv %s Tf 1 1 1 rg %s %s Td %s Tj ET Q\n"
                   % (_n(w), _n(h), _n(size), _n((w - tw) / 2), _n((h - size * 0.72) / 2), self._esc(label)))
        return self._ap(w, h, content)

    # -- fields ------------------------------------------------------------------------------------------------------

    def widget(self, page, rect, parent=None, extra=None, ap=None, state=None, mk=None):
        d = {"Type": N("Annot"), "Subtype": N("Widget"), "Rect": list(rect), "F": 4, "P": page.ref}
        if parent is not None:
            d["Parent"] = parent
        if mk:
            d["MK"] = mk
        if ap is not None:
            d["AP"] = {"N": ap}
        if state is not None:
            d["AS"] = N(state)
        d.update(extra or {})
        ref = self.doc.pdf.add(d)
        page.annotations.append(ref)
        return ref

    def widget_json(self, page, rect, state=None, on=None, text=None, button=False):
        """`button`: a check box or radio button, which keeps its appearance dictionary even under /NeedAppearances
        (its on state's name lives there)."""
        p = self.doc.pages.index(page)
        x0, y0, x1, y1 = rect
        return {"page": p, "box": page.display_box([(x0, y0), (x1, y1)]), "rect": list(rect), "state": state,
                "onState": on, "hasAppearance": bool(button or self.appearances), "appearanceText": text}

    def add(self, ref, exp, top=True):
        if top:
            self.fields.append(ref)
        self.expected.append(exp)

    def text(self, page, name, rect, value, size=11, flags=0, max_len=None, align=0, tooltip=None, default=None):
        w, h = rect[2] - rect[0], rect[3] - rect[1]
        comb = max_len if flags & flag_value("comb") else None
        multiline = flags & flag_value("multiline")
        shown = None
        if value is not None:
            shown = "*" * len(value) if flags & flag_value("password") else value
        lines = shown.split("\n") if shown is not None else [""]
        ap = self.text_ap(w, h, lines, size, align, comb) if self.appearances else None
        d = {"FT": N("Tx"), "T": String(name.encode("ascii")), "Ff": flags, "DA": String(b"/Helv %s Tf 0 g" % _n(size))}
        if value is not None:
            d["V"] = String(value.encode("latin-1")) if all(ord(c) < 256 for c in value) else U16(value)
        if default is not None:
            d["DV"] = String(default.encode("latin-1"))
        if max_len is not None:
            d["MaxLen"] = max_len
        if align:
            d["Q"] = align
        if tooltip:
            d["TU"] = String(tooltip.encode("latin-1"))
        d.update({"Type": N("Annot"), "Subtype": N("Widget"), "Rect": list(rect), "F": 4, "P": page.ref})
        if ap is not None:
            d["AP"] = {"N": ap}
        ref = self.doc.pdf.add(d)
        page.annotations.append(ref)
        self.add(ref, {"name": name, "partialName": name, "kind": "text", "value": value, "values": None,
                       "defaultValue": default, "options": None, "selected": None,
                       "flags": flag_names(flags, "text"), "flagsValue": flags, "maxLength": max_len,
                       "alignment": ("left", "centre", "right")[align], "font": "Helv", "fontSize": size,
                       "tooltip": tooltip, "inherited": [],
                       "widgets": [self.widget_json(page, rect, text=("\n".join(lines) if multiline else lines[0])
                                                    if value is not None else "")]})
        return ref

    def checkbox(self, page, name, rect, on, export="Yes", tooltip=None, flags=0):
        w, h = rect[2] - rect[0], rect[3] - rect[1]
        ap = {export: self.check_ap(w, h, "4", True), "Off": self.check_ap(w, h, "4", False)}
        state = export if on else "Off"
        d = {"FT": N("Btn"), "T": String(name.encode("ascii")), "Ff": flags, "V": N(state),
             "DA": String(b"/ZaDb 0 Tf 0 g"), "Type": N("Annot"), "Subtype": N("Widget"), "Rect": list(rect), "F": 4,
             "P": page.ref, "AS": N(state), "MK": {"CA": String(b"4")}}
        if tooltip:
            d["TU"] = String(tooltip.encode("latin-1"))
        if ap is not None:
            d["AP"] = {"N": ap}
        ref = self.doc.pdf.add(d)
        page.annotations.append(ref)
        self.add(ref, {"name": name, "partialName": name, "kind": "checkbox", "value": export if on else None,
                       "values": None, "defaultValue": None, "options": None, "selected": None,
                       "flags": flag_names(flags, "checkbox"), "flagsValue": flags, "maxLength": None,
                       "alignment": None, "font": "ZaDb", "fontSize": 0, "tooltip": tooltip, "inherited": [],
                       "widgets": [self.widget_json(page, rect, state, export, button=True)]})
        return ref

    def radio(self, page, name, rects, states, value, flags=None):
        flags = flag_value("radio", "noToggleToOff") if flags is None else flags
        parent = self.doc.pdf.alloc()
        kids, widgets = [], []
        for rect, st in zip(rects, states):
            w, h = rect[2] - rect[0], rect[3] - rect[1]
            ap = {st: self.check_ap(w, h, "l", True), "Off": self.check_ap(w, h, "l", False)}
            state = st if st == value else "Off"
            kids.append(self.widget(page, rect, parent, ap=ap, state=state, mk={"CA": String(b"l")}))
            widgets.append(self.widget_json(page, rect, state, st, button=True))
        self.doc.pdf.put(parent, {"FT": N("Btn"), "T": String(name.encode("ascii")), "Ff": flags,
                                  "V": N(value) if value else N("Off"), "Kids": kids,
                                  "DA": String(b"/ZaDb 0 Tf 0 g")})
        self.add(parent, {"name": name, "partialName": name, "kind": "radio", "value": value, "values": None,
                          "defaultValue": None, "options": [{"export": s, "display": s} for s in states],
                          "selected": [states.index(value)] if value else [],
                          "flags": flag_names(flags, "radio"), "flagsValue": flags, "maxLength": None,
                          "alignment": None, "font": "ZaDb", "fontSize": 0, "tooltip": None, "inherited": [],
                          "widgets": widgets})
        return parent

    def choice(self, page, name, rect, options, selected, combo=False, multi=False, size=10, pairs=False):
        flags = (flag_value("combo") if combo else 0) | (flag_value("multiSelect") if multi else 0)
        w, h = rect[2] - rect[0], rect[3] - rect[1]
        opts = [(e, d) for e, d in options]
        values = [opts[k][0] for k in selected]
        if self.appearances:
            ap = (self.text_ap(w, h, [opts[selected[0]][1]] if selected else [""], size) if combo
                  else self.list_ap(w, h, opts, selected, size))
        else:
            ap = None
        opt_obj = [[String(e.encode("latin-1")), String(d.encode("latin-1"))] if pairs else String(d.encode("latin-1"))
                   for e, d in opts]
        d = {"FT": N("Ch"), "T": String(name.encode("ascii")), "Ff": flags, "Opt": opt_obj,
             "DA": String(b"/Helv %s Tf 0 g" % _n(size)), "Type": N("Annot"), "Subtype": N("Widget"),
             "Rect": list(rect), "F": 4, "P": page.ref}
        if selected:
            d["V"] = [String(v.encode("latin-1")) for v in values] if multi else String(values[0].encode("latin-1"))
            if not combo:
                d["I"] = list(selected)
        if ap is not None:
            d["AP"] = {"N": ap}
        ref = self.doc.pdf.add(d)
        page.annotations.append(ref)
        kind = "combo" if combo else "list"
        self.add(ref, {"name": name, "partialName": name, "kind": kind,
                       "value": values[0] if selected and not multi else None, "values": values if multi else None,
                       "defaultValue": None, "options": [{"export": e, "display": dd} for e, dd in opts],
                       "selected": list(selected), "flags": flag_names(flags, kind), "flagsValue": flags,
                       "maxLength": None, "alignment": "left", "font": "Helv", "fontSize": size, "tooltip": None,
                       "inherited": [], "widgets": [self.widget_json(page, rect, text=opts[selected[0]][1]
                                                                     if combo and selected else None)]})
        return ref

    def pushbutton(self, page, name, rect, label):
        w, h = rect[2] - rect[0], rect[3] - rect[1]
        flags = flag_value("pushbutton")
        ap = self.button_ap(w, h, label) if self.appearances else None
        d = {"FT": N("Btn"), "T": String(name.encode("ascii")), "Ff": flags, "Type": N("Annot"),
             "Subtype": N("Widget"), "Rect": list(rect), "F": 4, "P": page.ref,
             "MK": {"BG": [0.25, 0.45, 0.75], "CA": String(label.encode("latin-1"))},
             "A": {"S": N("ResetForm")}, "DA": String(b"/Helv 11 Tf 1 g")}
        if ap is not None:
            d["AP"] = {"N": ap}
        ref = self.doc.pdf.add(d)
        page.annotations.append(ref)
        self.add(ref, {"name": name, "partialName": name, "kind": "pushbutton", "value": None, "values": None,
                       "defaultValue": None, "options": None, "selected": None,
                       "flags": flag_names(flags, "pushbutton"), "flagsValue": flags, "maxLength": None,
                       "alignment": None, "font": "Helv", "fontSize": 11, "tooltip": None, "inherited": [],
                       "action": "ResetForm", "caption": label, "widgets": [self.widget_json(page, rect, text=label)]})
        return ref

    def signature(self, page, name, rect):
        w, h = rect[2] - rect[0], rect[3] - rect[1]
        ap = self._ap(w, h, b"q 0.5 G 0.75 w 0.375 0.375 %s %s re S Q\n" % (_n(w - 0.75), _n(h - 0.75)), fonts=()) \
            if self.appearances else None
        d = {"FT": N("Sig"), "T": String(name.encode("ascii")), "Type": N("Annot"), "Subtype": N("Widget"),
             "Rect": list(rect), "F": 4, "P": page.ref}
        if ap is not None:
            d["AP"] = {"N": ap}
        ref = self.doc.pdf.add(d)
        page.annotations.append(ref)
        self.add(ref, {"name": name, "partialName": name, "kind": "signature", "value": None, "values": None,
                       "defaultValue": None, "options": None, "selected": None, "flags": [], "flagsValue": 0,
                       "maxLength": None, "alignment": None, "font": None, "fontSize": None, "tooltip": None,
                       "inherited": [], "signed": False, "widgets": [self.widget_json(page, rect)]})
        return ref

    def finish(self, sig_flags=None, xfa=None):
        acro = {"Fields": self.fields, "DA": String(b"/Helv 0 Tf 0 g"), "DR": self.dr}
        if self.need_appearances:
            acro["NeedAppearances"] = True
        if sig_flags:
            acro["SigFlags"] = sig_flags
        if xfa is not None:
            acro["XFA"] = xfa
        self.doc.catalog_extra["AcroForm"] = self.doc.pdf.add(acro)


def _n(x):
    from .pdf import fmt_num

    return fmt_num(float(x)).encode("ascii")


def _title(doc, page, text, size=13):
    f = doc.fonts[0]
    c = page.content
    c.BT()
    c.Tf(f, size)
    c.Td(30, page.size()[1] - 36)
    c.show(text)
    c.ET()
    page.line(text)


LABELS = [("Name", 700), ("Comments", 640), ("Postcode", 580), ("PIN", 548), ("Order reference", 516),
          ("Newsletter", 484), ("Delivery", 452), ("Colour", 400), ("Extras", 330), ("Country", 270),
          ("Signature", 200)]


def _labels(doc, page, form):
    c = page.content
    c.BT()
    c.Tf(form.helv, 10)
    prev = (0, 0)
    for text, y in LABELS:
        c.Td(30 - prev[0], y + 4 - prev[1])
        prev = (30, y + 4)
        c.show(text)
        page.line(text)
    c.ET()


def _all_fields(form, page, filled):
    form.text(page, "name", [150, 696, 400, 716], "Alex Morgan" if filled else None,
              flags=flag_value("required"), tooltip="Full name")
    form.text(page, "comments", [150, 610, 400, 660],
              "Please deliver after 10 am.\nCall the front desk on arrival." if filled else None,
              size=10, flags=flag_value("multiline"))
    form.text(page, "postcode", [150, 576, 270, 596], "AB1234" if filled else None, flags=flag_value("comb"),
              max_len=6)
    form.text(page, "pin", [150, 544, 230, 564], "4821" if filled else None, flags=flag_value("password"))
    form.text(page, "reference", [150, 512, 300, 532], "PO-2024-17" if filled else None,
              flags=flag_value("readOnly"), max_len=10, align=2)
    form.checkbox(page, "newsletter", [150, 480, 166, 496], filled, tooltip="Send me the monthly newsletter")
    form.radio(page, "delivery", [[150, 448, 166, 464], [230, 448, 246, 464], [310, 448, 326, 464]],
               ["Standard", "Express", "Pickup"], "Express" if filled else None)
    form.choice(page, "colour", [150, 370, 260, 420], [("Red", "Red"), ("Green", "Green"), ("Blue", "Blue")],
                [1] if filled else [])
    form.choice(page, "extras", [150, 300, 300, 350], [("gift", "Gift wrap"), ("card", "Greeting card"),
                                                      ("express", "Express handling")],
                [0, 2] if filled else [], multi=True, pairs=True)
    form.choice(page, "country", [150, 266, 300, 286], [("Canada", "Canada"), ("France", "France"),
                                                       ("Japan", "Japan")], [2] if filled else [], combo=True)
    form.signature(page, "signature", [150, 170, 350, 220])
    form.pushbutton(page, "reset", [300, 110, 400, 136], "Reset form")


def _form_doc(name, filled, appearances=True, need=False, seed="2026", finish=True):
    doc = Doc(name, "1.7", seed)
    page = doc.page(450, 780)
    form = Form(doc, appearances=appearances, need_appearances=need)
    _title(doc, page, "Order form")
    _labels(doc, page, form)
    _all_fields(form, page, filled)
    if finish:
        form.finish(sig_flags=None)
        doc.finish()
    return doc, form


def _forms_sample(doc, form, name, desc, extra=None):
    data, w = doc.write("table")
    exp = base_expected(doc, data, name, "form", D9, desc, w, objects=False)
    exp["fields"] = form.expected
    exp["form"] = {"needAppearances": form.need_appearances, "appearances": form.appearances, "xfa": None}
    exp.update(extra or {})
    return Sample(name, "d9", "form", D9, desc, data, exp, render_forms=True, checks=[check_fields])


def filled(seed):
    doc, form = _form_doc("forms-filled", True, seed=seed)
    desc = ("Every field kind filled, each widget with its appearance stream: text (required, a tooltip), multi-line, "
            "comb (MaxLen 6), password (shown as asterisks), read-only right-aligned with MaxLen, a check box (on), a "
            "radio group (Express), a list box, a multi-select list with export values, a combo box, an empty "
            "signature field and a push button (ResetForm).")
    return _forms_sample(doc, form, "forms-filled", desc)


def empty(seed):
    doc, form = _form_doc("forms-empty", False, seed=seed)
    desc = "The same form with every field empty or off, appearances included."
    return _forms_sample(doc, form, "forms-empty", desc)


def need_appearances(seed):
    doc, form = _form_doc("forms-need-appearances", True, appearances=False, need=True, seed=seed)
    desc = ("The filled form with /NeedAppearances true and no appearance streams for its text, choice, push button "
            "and signature fields: a reader that draws them must make their appearances from the values (pdfium does, "
            "with its form environment). Check boxes and radio buttons keep theirs, as real files do: their on "
            "states' names live in them.")
    return _forms_sample(doc, form, "forms-need-appearances", desc)


def inherited(seed):
    """A field tree: values, flags, DA, MaxLen and the field type inherited from ancestors; one field with two widgets
    on two pages; widgets that are pure kids (no /T)."""
    doc = Doc("forms-inherited", "1.7", seed)
    p1 = doc.page(420, 300)
    p2 = doc.page(420, 300)
    form = Form(doc)
    _title(doc, p1, "Shipping address")
    _title(doc, p2, "Contact")
    pdf = doc.pdf
    address = pdf.alloc()
    da = b"/Helv 10 Tf 0 0.3 0.6 rg"
    kids = []
    for k, (part, value, extra) in enumerate([("street", "12 Garden Lane", {}), ("city", "Springfield", {}),
                                               ("zip", "40213", {"MaxLen": 5})]):
        rect = [100, 220 - 34 * k, 300, 240 - 34 * k]
        w, h = rect[2] - rect[0], rect[3] - rect[1]
        comb = 5 if part == "zip" else None
        ap = form.text_ap(w, h, [value], 10, comb=comb, colour=b"0 0.3 0.6 rg")
        d = {"T": String(part.encode("ascii")), "Parent": address, "V": String(value.encode("ascii")),
             "Type": N("Annot"), "Subtype": N("Widget"), "Rect": rect, "F": 4, "P": p1.ref, "AP": {"N": ap}}
        d.update(extra)
        if part == "zip":
            d["Ff"] = flag_value("comb")
        ref = pdf.add(d)
        p1.annotations.append(ref)
        kids.append(ref)
        inherited_keys = ["FT", "DA"] + ([] if part == "zip" else ["Ff"])
        flags = flag_value("comb") if part == "zip" else flag_value("required")
        form.expected.append({"name": f"address.{part}", "partialName": part, "kind": "text", "value": value,
                              "values": None, "defaultValue": None, "options": None, "selected": None,
                              "flags": flag_names(flags, "text"), "flagsValue": flags,
                              "maxLength": 5 if part == "zip" else None, "alignment": "left", "font": "Helv",
                              "fontSize": 10, "tooltip": None, "inherited": inherited_keys,
                              "widgets": [form.widget_json(p1, rect, text=value)]})
    pdf.put(address, {"T": String(b"address"), "FT": N("Tx"), "Ff": flag_value("required"), "DA": String(da),
                      "Kids": kids})
    form.fields.append(address)
    # One field, two widgets (pure kids, no /T), the value on the field.
    contact = pdf.alloc()
    value = "office@example.com"
    widgets, wjson = [], []
    for page, rect in ((p1, [100, 100, 300, 120]), (p2, [100, 220, 300, 240])):
        ap = form.text_ap(200, 20, [value], 10)
        widgets.append(form.widget(page, rect, contact, ap=ap))
        wjson.append(form.widget_json(page, rect, text=value))
    pdf.put(contact, {"T": String(b"contact"), "FT": N("Tx"), "V": String(value.encode("ascii")),
                      "DA": String(b"/Helv 10 Tf 0 g"), "Kids": widgets})
    form.fields.append(contact)
    form.expected.append({"name": "contact", "partialName": "contact", "kind": "text", "value": value, "values": None,
                          "defaultValue": None, "options": None, "selected": None, "flags": [], "flagsValue": 0,
                          "maxLength": None, "alignment": "left", "font": "Helv", "fontSize": 10, "tooltip": None,
                          "inherited": [], "widgets": wjson})
    # A check box whose value and type come from its parent ("terms"), the widget a kid with its own /T.
    terms = pdf.alloc()
    rect = [100, 160, 116, 176]
    ap = {"Agreed": form.check_ap(16, 16, "4", True), "Off": form.check_ap(16, 16, "4", False)}
    agree = pdf.add({"T": String(b"agree"), "Parent": terms, "Type": N("Annot"), "Subtype": N("Widget"),
                     "Rect": rect, "F": 4, "P": p2.ref, "AP": {"N": ap}, "AS": N("Agreed"), "MK": {"CA": String(b"4")}})
    p2.annotations.append(agree)
    pdf.put(terms, {"T": String(b"terms"), "FT": N("Btn"), "V": N("Agreed"), "Kids": [agree],
                    "DA": String(b"/ZaDb 0 Tf 0 g")})
    form.fields.append(terms)
    form.expected.append({"name": "terms.agree", "partialName": "agree", "kind": "checkbox", "value": "Agreed",
                          "values": None, "defaultValue": None, "options": None, "selected": None, "flags": [],
                          "flagsValue": 0, "maxLength": None, "alignment": None, "font": "ZaDb", "fontSize": 0,
                          "tooltip": None, "inherited": ["FT", "V", "DA"],
                          "widgets": [form.widget_json(p2, rect, "Agreed", "Agreed", button=True)]})
    c = p1.content
    c.BT(); c.Tf(form.helv, 10)
    c.Td(30, 228); c.show("Street"); p1.line("Street")
    c.Td(0, -34); c.show("City"); p1.line("City")
    c.Td(0, -34); c.show("Zip"); p1.line("Zip")
    c.Td(0, -78); c.show("Email"); p1.line("Email")
    c.ET()
    c = p2.content
    c.BT(); c.Tf(form.helv, 10)
    c.Td(30, 228); c.show("Email"); p2.line("Email")
    c.Td(0, -64); c.show("Terms"); p2.line("Terms")
    c.ET()
    form.finish()
    doc.finish()
    desc = ("A field tree: address.street, address.city and address.zip take FT, DA and Ff from 'address' (zip "
            "overrides Ff with comb and adds MaxLen 5); 'contact' is one field with two widgets on two pages (pure "
            "kids, no /T); terms.agree takes FT, V and DA from 'terms'. `inherited` names the keys taken from an "
            "ancestor.")
    return _forms_sample(doc, form, "forms-inherited", desc)


XFA_TEMPLATE = b"""<template xmlns="http://www.xfa.org/schema/xfa-template/3.3/">
 <subform name="order" layout="tb" locale="en_US">
  <pageSet><pageArea name="Page1"><contentArea x="0.25in" y="0.25in" w="8in" h="10.5in"/>
   <medium stock="default" short="8.5in" long="11in"/></pageArea></pageSet>
  <subform name="customer" layout="tb">
   <field name="fullName" w="3in" h="0.3in"><ui><textEdit/></ui><caption><value><text>Name</text></value></caption></field>
   <field name="email" w="3in" h="0.3in"><ui><textEdit/></ui><caption><value><text>Email</text></value></caption></field>
  </subform>
  <subform name="item" layout="tb">
   <field name="product" w="3in" h="0.3in"><ui><textEdit/></ui></field>
   <field name="quantity" w="1in" h="0.3in"><ui><numericEdit/></ui></field>
   <field name="giftWrap" w="0.3in" h="0.3in"><ui><checkButton/></ui>
    <items><integer>1</integer><integer>0</integer></items></field>
  </subform>
 </subform>
</template>
"""

XFA_DATASETS = b"""<xfa:datasets xmlns:xfa="http://www.xfa.org/schema/xfa-data/1.0/">
 <xfa:data>
  <order>
   <customer><fullName>Jordan Lee</fullName><email>jordan.lee@example.com</email></customer>
   <item><product>Desk lamp</product><quantity>3</quantity><giftWrap>1</giftWrap></item>
  </order>
 </xfa:data>
</xfa:datasets>
"""

XFA_CONFIG = b"""<config xmlns="http://www.xfa.org/schema/xci/3.1/"><present><pdf><version>1.7</version></pdf></present></config>
"""


def xfa(seed):
    """An XFA form (template and datasets) with its AcroForm twin fields, as static XFA forms are written."""
    doc = Doc("forms-xfa", "1.7", seed)
    page = doc.page(612, 792)
    form = Form(doc)
    _title(doc, page, "Order (an XFA form)")
    pdf = doc.pdf
    packets = []
    for label, body in (("preamble", b'<xdp:xdp xmlns:xdp="http://ns.adobe.com/xdp/">\n'), ("config", XFA_CONFIG),
                        ("template", XFA_TEMPLATE), ("datasets", XFA_DATASETS), ("postamble", b"</xdp:xdp>\n")):
        packets += [String(label.encode("ascii")), pdf.add(Stream({"Filter": N("FlateDecode")}, flate(body), body))]
    # The AcroForm twins: order[0].customer[0].fullName[0] and so on, holding the same values.
    values = [("order[0].customer[0].fullName[0]", "Jordan Lee"), ("order[0].customer[0].email[0]",
                                                                    "jordan.lee@example.com"),
              ("order[0].item[0].product[0]", "Desk lamp"), ("order[0].item[0].quantity[0]", "3")]
    top = pdf.alloc()
    tree = {}
    leaves = []
    for k, (full, value) in enumerate(values):
        parts = full.split(".")
        rect = [150, 700 - 30 * k, 400, 720 - 30 * k]
        ap = form.text_ap(250, 20, [value], 10)
        parent_key = ".".join(parts[:-1])
        if parent_key not in tree:
            tree[parent_key] = pdf.alloc()
        leaf = pdf.add({"T": String(parts[-1].encode("ascii")), "FT": N("Tx"), "V": String(value.encode("ascii")),
                        "DA": String(b"/Helv 10 Tf 0 g"), "Parent": tree[parent_key], "Type": N("Annot"),
                        "Subtype": N("Widget"), "Rect": rect, "F": 4, "P": page.ref, "AP": {"N": ap}})
        page.annotations.append(leaf)
        leaves.append((parent_key, leaf))
        form.expected.append({"name": full, "partialName": parts[-1], "kind": "text", "value": value, "values": None,
                              "defaultValue": None, "options": None, "selected": None, "flags": [], "flagsValue": 0,
                              "maxLength": None, "alignment": "left", "font": "Helv", "fontSize": 10, "tooltip": None,
                              "inherited": [], "widgets": [form.widget_json(page, rect, text=value)]})
    for key, ref in tree.items():
        parts = key.split(".")
        kids = [leaf for pk, leaf in leaves if pk == key]
        pdf.put(ref, {"T": String(parts[-1].encode("ascii")), "Parent": top, "Kids": kids})
    pdf.put(top, {"T": String(b"order[0]"), "Kids": list(tree.values())})
    form.fields.append(top)
    form.finish(xfa=packets)
    doc.finish()
    desc = ("An XFA form (template, datasets, config, preamble and postamble packets) whose AcroForm twins hold the "
            "same values; `xfa.data` gives the datasets' values by path. A reader reads the data as fields and "
            "reports that it does not draw XFA layout.")
    data_values = {"order.customer.fullName": "Jordan Lee", "order.customer.email": "jordan.lee@example.com",
                   "order.item.product": "Desk lamp", "order.item.quantity": "3", "order.item.giftWrap": "1"}
    sample = _forms_sample(doc, form, "forms-xfa", desc)
    sample.expected["form"]["xfa"] = {"packets": ["preamble", "config", "template", "datasets", "postamble"],
                                      "data": data_values, "datasetsSha256": sha256(XFA_DATASETS),
                                      "issue": "XFA layout is not drawn: the AcroForm twins are"}
    sample.expected["form"]["xfa"]["template"] = {
        "sections": [{"name": "order", "path": "order", "parent": None},
                     {"name": "Page1", "path": "order.Page1", "parent": "order"},
                     {"name": "customer", "path": "order.customer", "parent": "order"},
                     {"name": "item", "path": "order.item", "parent": "order"}],
        "fields": [_tfield("fullName", "order.customer.fullName", "text", "order.customer", "Name",
                           "order.customer.fullName", "Jordan Lee", field="order[0].customer[0].fullName[0]"),
                   _tfield("email", "order.customer.email", "text", "order.customer", "Email", "order.customer.email",
                           "jordan.lee@example.com", field="order[0].customer[0].email[0]"),
                   _tfield("product", "order.item.product", "text", "order.item", None, "order.item.product",
                           "Desk lamp", field="order[0].item[0].product[0]"),
                   _tfield("quantity", "order.item.quantity", "text", "order.item", None, "order.item.quantity", "3",
                           field="order[0].item[0].quantity[0]"),
                   _tfield("giftWrap", "order.item.giftWrap", "checkbox", "order.item", None, "order.item.giftWrap",
                           "1", options=[("1", "1"), ("0", "0")], field="order.item.giftWrap")],
        "scripts": 0}
    return sample


def _tfield(name, path, kind, section, caption, binding, value, options=(), flags=(), field=None):
    """A template field as the expected JSON gives it."""
    return {"name": name, "path": path, "kind": kind, "section": section, "caption": caption, "binding": binding,
            "value": value, "options": [{"display": d, "export": e} for d, e in options], "flags": list(flags),
            "field": field}


def flattened(seed):
    """The filled form flattened: each widget's appearance drawn into the page content where its Rect is (q, the
    BBox mapped onto the Rect, Do, Q), the fields and widgets never written. Its look is pdfium's drawing of
    forms-filled."""
    doc, form = _form_doc("forms-flattened", True, seed=seed, finish=False)
    page = doc.pages[0]
    pdf = doc.pdf
    c = page.content
    k = 0
    for ref in page.annotations:
        annot = pdf.get(ref)
        ap = annot["AP"]["N"]
        if isinstance(ap, dict):
            ap = ap[annot["AS"].value]
        x0, y0 = annot["Rect"][0], annot["Rect"][1]
        k += 1
        c.q()
        c.cm(1, 0, 0, 1, x0, y0)
        c._use("XObject", f"Fw{k}", ap)
        c.emit(f"/Fw{k} Do")
        c.Q()
        del pdf.objects[ref.num]
    for ref in form.fields:
        pdf.objects.pop(ref.num, None)
    page.annotations = []
    doc.finish()
    data, w = doc.write("table")
    desc = ("forms-filled flattened: each widget's appearance (the on state for buttons) drawn into the page content "
            "at its Rect, no AcroForm, fields or widgets. It must look as forms-filled drawn with its fields "
            "(`reference.flatten` compares pdfium's two drawings).")
    exp = base_expected(doc, data, "forms-flattened", "form", D9, desc, w, objects=False)
    exp["fields"] = []
    exp["form"] = {"flattenedFrom": "forms-filled", "needAppearances": False, "appearances": True, "xfa": None}
    return Sample("forms-flattened", "d9", "form", D9, desc, data, exp, checks=[check_flatten])


# -- reference checks ----------------------------------------------------------------------------------------------------

def pdfium_fields(path, password=None):
    """Each widget pdfium's form environment finds: its field's full name, type, value, flags, checked state and
    options (pdfium resolves the inheritance)."""
    import ctypes

    import pypdfium2 as pdfium
    import pypdfium2.raw as raw

    def wide(fn, *args):
        n = fn(*args, None, 0)
        if n <= 2:
            return ""
        buf = (ctypes.c_ushort * (n // 2))()
        fn(*args, buf, n)
        return bytes(buf)[:n - 2].decode("utf-16-le")

    types = {raw.FPDF_FORMFIELD_TEXTFIELD: "text", raw.FPDF_FORMFIELD_CHECKBOX: "checkbox",
             raw.FPDF_FORMFIELD_RADIOBUTTON: "radio", raw.FPDF_FORMFIELD_LISTBOX: "list",
             raw.FPDF_FORMFIELD_COMBOBOX: "combo", raw.FPDF_FORMFIELD_PUSHBUTTON: "pushbutton",
             raw.FPDF_FORMFIELD_SIGNATURE: "signature"}
    doc = pdfium.PdfDocument(path, password=password or None)
    doc.init_forms()
    handle = doc.formenv.raw if doc.formenv else None
    out = []
    try:
        for i in range(len(doc)):
            page = doc[i]
            for k in range(raw.FPDFPage_GetAnnotCount(page)):
                annot = raw.FPDFPage_GetAnnot(page, k)
                if raw.FPDFAnnot_GetSubtype(annot) != raw.FPDF_ANNOT_WIDGET:
                    raw.FPDFPage_CloseAnnot(annot)
                    continue
                kind = types.get(raw.FPDFAnnot_GetFormFieldType(handle, annot), "other")
                entry = {"page": i, "name": wide(raw.FPDFAnnot_GetFormFieldName, handle, annot), "kind": kind,
                         "value": wide(raw.FPDFAnnot_GetFormFieldValue, handle, annot),
                         "flags": raw.FPDFAnnot_GetFormFieldFlags(handle, annot)}
                if kind in ("checkbox", "radio"):
                    entry["checked"] = bool(raw.FPDFAnnot_IsChecked(handle, annot))
                    entry["export"] = wide(raw.FPDFAnnot_GetFormFieldExportValue, handle, annot)
                if kind in ("list", "combo"):
                    n = raw.FPDFAnnot_GetOptionCount(handle, annot)
                    entry["options"] = [wide(raw.FPDFAnnot_GetOptionLabel, handle, annot, j) for j in range(n)]
                    entry["selected"] = [j for j in range(n) if raw.FPDFAnnot_IsOptionSelected(handle, annot, j)]
                out.append(entry)
                raw.FPDFPage_CloseAnnot(annot)
            page.close()
    finally:
        doc.close()
    return out


def check_fields(path, exp, notes):
    """Compares the expected fields with pdfium's widgets: name, kind, value, flags, checked state and options."""
    got = pdfium_fields(path)
    by_name = {}
    for g in got:
        by_name.setdefault(g["name"], []).append(g)
    problems = []
    for f in exp["fields"]:
        gs = by_name.get(f["name"], [])
        if len(gs) != len(f["widgets"]):
            problems.append({"field": f["name"], "widgets": len(f["widgets"]), "pdfium": len(gs)})
            continue
        for g, w in zip(gs, f["widgets"]):
            if g["kind"] != f["kind"]:
                problems.append({"field": f["name"], "kind": [f["kind"], g["kind"]]})
            if g["flags"] != f["flagsValue"]:
                problems.append({"field": f["name"], "flags": [f["flagsValue"], g["flags"]]})
            if f["kind"] == "text" and g["value"] != (f["value"] or ""):
                problems.append({"field": f["name"], "value": [f["value"], g["value"]]})
            if f["kind"] in ("checkbox", "radio"):
                on = w["state"] is not None and w["state"] != "Off"
                if g["checked"] != on:
                    problems.append({"field": f["name"], "checked": [on, g["checked"]]})
            if f["kind"] in ("list", "combo"):
                labels = [o["display"] for o in f["options"]]
                if g["options"] != labels or g["selected"] != f["selected"]:
                    problems.append({"field": f["name"], "options": [labels, f["selected"], g["options"],
                                                                     g["selected"]]})
    extra = sorted(set(by_name) - {f["name"] for f in exp["fields"]})
    if extra:
        problems.append({"pdfiumOnly": extra})
    if problems:
        notes.append(f"{exp['name']}: pdfium reads {len(problems)} field(s) differently")
    return "fields", {"widgets": len(got), "problems": problems}, {}


def check_flatten(path, exp, notes):
    """pdfium's drawing of the flattened file against its drawing of forms-filled with its fields (the same pixels
    are expected; the largest channel difference and the share of differing pixels are recorded)."""
    import os

    import pypdfium2 as pdfium

    from .reference import DPI

    other = os.path.join(os.path.dirname(path), "forms-filled.pdf")
    images = []
    for p, forms in ((path, False), (other, True)):
        doc = pdfium.PdfDocument(p)
        if forms:
            doc.init_forms()
        page = doc[0]
        images.append(page.render(scale=DPI / 72, may_draw_forms=True).to_pil().convert("RGB"))
        page.close()
        doc.close()
    a, b = images
    if a.size != b.size:
        return "flatten", {"comparedWith": "forms-filled.p1.png", "sizes": [a.size, b.size]}, {}
    da, db = a.tobytes(), b.tobytes()
    worst = max(abs(x - y) for x, y in zip(da, db))
    differing = sum(1 for k in range(0, len(da), 3) if da[k:k + 3] != db[k:k + 3])
    # Measured: pdfium anti-aliases text in a widget's appearance differently from the same text in page content, so
    # glyph edges differ (up to about 90 levels on about 1% of the pixels); the shapes and fills are identical.
    if worst > 96 or differing > len(da) // 3 // 50:
        notes.append(f"{exp['name']}: pdfium draws it {worst} levels off forms-filled on {differing} pixels")
    return "flatten", {"comparedWith": "forms-filled.p1.png", "largestDifference": worst,
                       "differingPixels": differing, "pixels": len(da) // 3,
                       "note": "glyph edges only: pdfium anti-aliases text in widget appearances differently"}, {}


# -- the forms completed (2026-11-19) -------------------------------------------------------------------------------

def _text_dict(page, name, rect, value, size=11, flags=0, extra=None, ap=None):
    d = {"FT": N("Tx"), "T": String(name.encode("ascii")), "Ff": flags, "DA": String(b"/Helv %s Tf 0 g" % _n(size)),
         "Type": N("Annot"), "Subtype": N("Widget"), "Rect": list(rect), "F": 4, "P": page.ref}
    if value is not None:
        d["V"] = String(value.encode("latin-1")) if all(ord(c) < 256 for c in value) else U16(value)
    if ap is not None:
        d["AP"] = {"N": ap}
    d.update(extra or {})
    return d


def _text_json(form, page, name, rect, value, size=11, flags=0, text=None, partial=None, appearance=None):
    w = form.widget_json(page, rect, text=text)
    if appearance is not None:
        w["hasAppearance"] = appearance
    return {"name": name, "partialName": partial or name, "kind": "text", "value": value, "values": None,
            "defaultValue": None, "options": None, "selected": None, "flags": flag_names(flags, "text"),
            "flagsValue": flags, "maxLength": None, "alignment": "left", "font": "Helv", "fontSize": size,
            "tooltip": None, "inherited": [], "widgets": [w]}


STYLED = [  # name, rect, /MK, /BS, value, the expected border
    ("vertical", [40, 60, 70, 300], {"R": 90, "BC": [0, 0, 0.6], "BG": [0.94, 0.94, 1]}, None, "Turned a quarter",
     None, 90),
    ("upside", [100, 40, 300, 65], {"R": 180, "BC": [0, 0, 0.6]}, None, "Upside down", None, 180),
    ("downward", [440, 60, 470, 300], {"R": 270, "BC": [0, 0, 0.6]}, None, "Turned three quarters", None, 270),
    ("dashed", [100, 340, 300, 365], {"BC": [0.2, 0.2, 0.2]}, {"S": N("D"), "W": 1, "D": [3, 2]}, "Dashed border",
     {"style": "dashed", "width": 1, "dashes": [3, 2]}, 0),
    ("beveled", [100, 290, 300, 315], {"BC": [0.3, 0.3, 0.3], "BG": [0.85, 0.85, 0.85]}, {"S": N("B"), "W": 1},
     "Beveled box", {"style": "beveled", "width": 1, "dashes": []}, 0),
    ("inset", [100, 240, 300, 265], {"BC": [0.3, 0.3, 0.3], "BG": [0.92, 0.92, 0.92]}, {"S": N("I"), "W": 1},
     "Inset box", {"style": "inset", "width": 1, "dashes": []}, 0),
    ("underline", [100, 190, 300, 215], {"BC": [0, 0, 0]}, {"S": N("U"), "W": 1}, "Underlined",
     {"style": "underline", "width": 1, "dashes": []}, 0),
    ("solid", [100, 140, 300, 165], {"BC": [0.5, 0, 0]}, {"S": N("S"), "W": 2}, "Solid thick border",
     {"style": "solid", "width": 2, "dashes": []}, 0),
]


def styled(seed):
    """Text fields turned by /MK /R (90, 180, 270) and bordered by /BS (dashed, beveled, inset, underline, solid 2),
    filled, with /NeedAppearances and no appearance streams: pdfium makes their appearances (turned and bordered as
    their entries say), which the reference drawing shows."""
    doc = Doc("forms-styled", "1.7", seed)
    page = doc.page(500, 400)
    form = Form(doc, appearances=False, need_appearances=True)
    _title(doc, page, "Styled fields")
    for name, rect, mk, bs, value, border, rotation in STYLED:
        extra = {"MK": mk}
        if bs is not None:
            extra["BS"] = bs
        ref = form.doc.pdf.add(_text_dict(page, name, rect, value, extra=extra))
        page.annotations.append(ref)
        exp = _text_json(form, page, name, rect, value, text=value)
        if rotation:
            exp["widgets"][0]["rotation"] = rotation
        if border is not None:
            exp["border"] = border
        form.add(ref, exp)
    form.finish()
    doc.finish()
    desc = ("Text fields turned a quarter, a half and three quarters (/MK /R) and bordered dashed ([3 2]), beveled, inset, "
            "underlined and solid (2 points) by /BS, filled, under /NeedAppearances with no appearance streams: pdfium "
            "makes the appearances its drawing shows.")
    return _forms_sample(doc, form, "forms-styled", desc)


RICH_RV = ('<?xml version="1.0"?><body xmlns="http://www.w3.org/1999/xhtml" '
           'xmlns:xfa="http://www.xfa.org/schema/xfa-data/1.0/" xfa:spec="2.0.2"><p><b>Dear reader,</b></p><p>The '
           '<span style="font-style:italic;color:#cc0000">quarterly report</span> is <span style="font-size:14pt;'
           "font-family:'Times New Roman'\">ready</span>.</p></body>")
RICH_DS = "font: 11pt Helvetica; color: #000000"


def _run(text, bold=False, italic=False, size=11.0, colour=(0.0, 0.0, 0.0), family="Helvetica"):
    return {"text": text, "bold": bold, "italic": italic, "underline": False, "strikeOut": False, "size": size,
            "colour": list(colour) if colour is not None else None, "family": family}


def rich(seed):
    """A rich text field (/RV XHTML over /DS, /V its plain text), an empty rich text field and a plain one."""
    doc = Doc("forms-rich", "1.7", seed)
    page = doc.page(420, 300)
    form = Form(doc)
    _title(doc, page, "Messages")
    plain = "Dear reader,\nThe quarterly report is ready."
    flags = flag_value("multiline", "richText")
    rect = [40, 150, 380, 230]
    ap = form.text_ap(340, 80, plain.split("\n"), 11)
    ref = doc.pdf.add(_text_dict(page, "message", rect, plain, flags=flags, ap=ap,
                                 extra={"RV": String(RICH_RV.encode("ascii")), "DS": String(RICH_DS.encode("ascii"))}))
    page.annotations.append(ref)
    exp = _text_json(form, page, "message", rect, plain, flags=flags, text=plain)
    exp["richText"] = [_run("Dear reader,", bold=True), _run("\nThe "),
                       _run("quarterly report", italic=True, colour=(0.8, 0.0, 0.0)), _run(" is "),
                       _run("ready", size=14.0, family="Times New Roman"), _run(".")]
    form.add(ref, exp)
    for name, rect, size, flags in (("note", [40, 70, 380, 130], 10, flag_value("multiline", "richText")),
                                    ("title", [40, 30, 380, 52], 11, 0)):
        w, h = rect[2] - rect[0], rect[3] - rect[1]
        ref = doc.pdf.add(_text_dict(page, name, rect, None, size=size, flags=flags,
                                     ap=form.text_ap(w, h, [""], size)))
        page.annotations.append(ref)
        form.add(ref, _text_json(form, page, name, rect, None, size=size, flags=flags, text=""))
    form.finish()
    doc.finish()
    desc = ("A rich text field (multi-line, the rich text flag): /RV is XHTML (a bold paragraph, then italic red, 14 pt "
            "Times New Roman and plain runs) over /DS (11 pt Helvetica, black), /V its plain text; an empty rich text "
            "field and an empty plain one to fill.")
    sample = _forms_sample(doc, form, "forms-rich", desc)
    sample.checks.append(check_plain_values)
    return sample


def held(seed):
    """Fields held inside other objects: every object but the streams in object streams (a cross-reference stream);
    applicant.name written as a dictionary in its parent's /Kids (its widget an object of its own, with no /Parent),
    remarks written in the form's /Fields itself, applicant.email and city ordinary fields (field and widget one
    object)."""
    doc = Doc("forms-held", "1.7", seed)
    page = doc.page(420, 300)
    form = Form(doc)
    _title(doc, page, "Application")
    pdf = doc.pdf
    applicant = pdf.alloc()
    rect = [120, 220, 380, 240]
    w1 = pdf.add({"Type": N("Annot"), "Subtype": N("Widget"), "Rect": rect, "F": 4, "P": page.ref,
                  "AP": {"N": form.text_ap(260, 20, [""], 11)}})
    page.annotations.append(w1)
    name = {"T": String(b"name"), "FT": N("Tx"), "Parent": applicant, "DA": String(b"/Helv 11 Tf 0 g"), "Kids": [w1]}
    rect2 = [120, 190, 380, 210]
    email = pdf.add({"T": String(b"email"), "FT": N("Tx"), "Parent": applicant, "DA": String(b"/Helv 11 Tf 0 g"),
                     "Type": N("Annot"), "Subtype": N("Widget"), "Rect": rect2, "F": 4, "P": page.ref,
                     "AP": {"N": form.text_ap(260, 20, [""], 11)}})
    page.annotations.append(email)
    pdf.put(applicant, {"T": String(b"applicant"), "Kids": [name, email]})
    rect3 = [120, 120, 380, 170]
    w3 = pdf.add({"Type": N("Annot"), "Subtype": N("Widget"), "Rect": rect3, "F": 4, "P": page.ref,
                  "AP": {"N": form.text_ap(260, 50, [""], 10)}})
    page.annotations.append(w3)
    remarks = {"T": String(b"remarks"), "FT": N("Tx"), "Ff": flag_value("multiline"), "DA": String(b"/Helv 10 Tf 0 g"),
               "Kids": [w3]}
    form.expected += [_text_json(form, page, "applicant.name", rect, None, text="", partial="name"),
                      _text_json(form, page, "applicant.email", rect2, None, text="", partial="email"),
                      _text_json(form, page, "remarks", rect3, None, size=10, flags=flag_value("multiline"), text="")]
    form.fields += [applicant, remarks]
    form.text(page, "city", [120, 80, 380, 100], None)
    c = page.content
    c.BT()
    c.Tf(form.helv, 10)
    prev = 0
    for label, y in (("Name", 224), ("Email", 194), ("Remarks", 154), ("City", 84)):
        c.Td(30 if prev == 0 else 0, y - prev)
        prev = y
        c.show(label)
        page.line(label)
    c.ET()
    form.finish()
    doc.finish()
    data, w = doc.write("objstm")
    desc = ("Fields held inside other objects: every object but the streams in object streams; applicant.name a "
            "dictionary in its parent's /Kids (its widget an object of its own without /Parent), remarks a dictionary "
            "in /Fields, applicant.email and city ordinary. All empty, with appearances. pdfium names a widget by its "
            "/Parent, which no dictionary held inside another object can be: it reads those two widgets with no name "
            "(`reference.fields`); a writer that moves them to objects of their own gives them one.")
    exp = base_expected(doc, data, "forms-held", "form", D9, desc, w, objects=False)
    exp["fields"] = form.expected
    exp["form"] = {"needAppearances": False, "appearances": True, "xfa": None, "objectStreams": True,
                   "held": ["applicant.name", "remarks"]}
    return Sample("forms-held", "d9", "form", D9, desc, data, exp, render_forms=True, checks=[check_fields])


def faces(seed):
    """A field whose value its /DA font (Helvetica) cannot show: Cyrillic, its appearance made in an embedded DejaVu
    subset; the form's /DR names that font /F1 (a name a writer adding a font must not take); and a plain field."""
    from .common import DEJAVU
    from .fonts import Type0

    doc = Doc("forms-faces", "1.7", seed)
    page = doc.page(420, 200)
    form = Form(doc)
    _title(doc, page, "Greetings")
    dejavu = doc.font(Type0, DEJAVU)
    form.dr["Font"]["F1"] = dejavu.ref
    value = "Добрый день"
    rect = [40, 110, 380, 134]
    codes = b"".join(g.code for g in dejavu.glyphs(value))
    content = (b"/Tx BMC\nq 1 1 338 22 re W n\nBT\n/F1 12 Tf 0 g\n2 7.5 Td <%s> Tj\nET\nQ\nEMC\n"
               % codes.hex().upper().encode("ascii"))
    ap = doc.pdf.add(Stream({"Type": N("XObject"), "Subtype": N("Form"), "BBox": [0, 0, 340, 24],
                             "Resources": {"Font": {"F1": dejavu.ref}}, "Filter": N("FlateDecode")}, flate(content),
                            content))
    ref = doc.pdf.add(_text_dict(page, "greeting", rect, value, size=12, ap=ap))
    page.annotations.append(ref)
    form.add(ref, _text_json(form, page, "greeting", rect, value, size=12, text=value))
    rect2 = [40, 60, 380, 84]
    ref2 = doc.pdf.add(_text_dict(page, "note", rect2, None, size=12, ap=form.text_ap(340, 24, [""], 12)))
    page.annotations.append(ref2)
    form.add(ref2, _text_json(form, page, "note", rect2, None, size=12, text=""))
    form.finish()
    doc.finish()
    desc = ("A field whose /DA names Helvetica holding a Cyrillic value, its appearance in an embedded DejaVu subset "
            "(/F1 in the form's /DR too), and an empty plain field: a writer filling it in a face its /DA lacks names "
            "a font of that face in /DA and /DR, under a name /DR does not hold.")
    return _forms_sample(doc, form, "forms-faces", desc)


XFA2_TEMPLATE = b"""<template xmlns="http://www.xfa.org/schema/xfa-template/3.3/">
 <subform name="report" layout="tb" locale="en_US">
  <pageSet><pageArea name="Sheet"><contentArea x="0.25in" y="0.25in" w="8in" h="10.5in"/>
   <medium stock="default" short="8.5in" long="11in"/>
   <field name="footer" w="3in" h="0.3in"><ui><textEdit/></ui><bind match="none"/><value><text>Page footer</text></value></field>
  </pageArea></pageSet>
  <subform name="author" layout="tb">
   <field name="fullName" w="3in" h="0.3in"><ui><textEdit/></ui><caption><value><text>Full name</text></value></caption>
    <validate nullTest="error"/></field>
   <field name="mail" w="3in" h="0.3in"><ui><textEdit/></ui><caption><value><text>Email</text></value></caption>
    <bind match="dataRef" ref="$record.contact.email"/>
    <event activity="exit"><script contentType="application/x-javascript">this.rawValue = this.rawValue.trim();</script></event></field>
  </subform>
  <subform name="details" layout="tb">
   <field name="summary" w="6in" h="1in"><ui><textEdit multiLine="1"/></ui><caption><value><text>Summary</text></value></caption></field>
   <field name="pages" w="1in" h="0.3in"><ui><numericEdit/></ui><caption><value><text>Pages</text></value></caption>
    <calculate><script>Sum(1, 2)</script></calculate></field>
   <field name="format" w="2in" h="0.3in"><ui><choiceList/></ui><caption><value><text>Format</text></value></caption>
    <items><text>Printed</text><text>Digital</text></items><items save="1"><text>print</text><text>digital</text></items></field>
   <field name="topics" w="2in" h="1in"><ui><choiceList open="multiSelect"/></ui>
    <items><text>Science</text><text>Technology</text></items></field>
   <exclGroup name="priority"><caption><value><text>Priority</text></value></caption>
    <field name="normal"><ui><checkButton shape="round"/></ui><caption><value><text>Normal</text></value></caption><items><text>normal</text></items></field>
    <field name="urgent"><ui><checkButton shape="round"/></ui><caption><value><text>Urgent</text></value></caption><items><text>urgent</text></items></field>
   </exclGroup>
   <field name="reviewed"><ui><checkButton/></ui><items><integer>1</integer><integer>0</integer></items></field>
   <field name="submit"><ui><button/></ui><caption><value><text>Submit</text></value></caption>
    <event activity="click"><script contentType="application/x-javascript">xfa.host.messageBox("Sent");</script></event></field>
  </subform>
  <subform>
   <field name="revision" access="readOnly"><ui><textEdit/></ui></field>
  </subform>
  <proto><field name="unused"><ui><textEdit/></ui></field></proto>
  <variables><script name="helpers" contentType="application/x-javascript">function round(v) { return Math.round(v); }</script></variables>
 </subform>
</template>
"""

XFA2_DATASETS = b"""<xfa:datasets xmlns:xfa="http://www.xfa.org/schema/xfa-data/1.0/">
 <xfa:data>
  <report>
   <author><fullName>Sam Rivera</fullName></author>
   <contact><email>sam.rivera@example.com</email></contact>
   <details><summary>A short technical report on network latency.</summary><pages>12</pages><format>digital</format><topics>Technology</topics><priority>urgent</priority><reviewed>1</reviewed></details>
   <revision>3</revision>
  </report>
 </xfa:data>
</xfa:datasets>
"""


def xfa_template(seed):
    """An XFA form with no AcroForm fields (its data read as fields of their own) whose template holds a page area with
    a field bound to nothing, captions, a required field, a field bound by reference, a multi-line field, a numeric
    field, a drop-down list with saved values, a multi-select list, a radio group (exclGroup), a check box, a button,
    an unnamed subform's read-only field, a prototype (no field) and four scripts (none run)."""
    doc = Doc("forms-xfa-template", "1.7", seed)
    page = doc.page(612, 792)
    form = Form(doc)
    _title(doc, page, "Report (an XFA form)")
    packets = []
    for label, body in (("preamble", b'<xdp:xdp xmlns:xdp="http://ns.adobe.com/xdp/">\n'), ("config", XFA_CONFIG),
                        ("template", XFA2_TEMPLATE), ("datasets", XFA2_DATASETS), ("postamble", b"</xdp:xdp>\n")):
        packets += [String(label.encode("ascii")), doc.pdf.add(Stream({"Filter": N("FlateDecode")}, flate(body), body))]
    form.finish(xfa=packets)
    doc.finish()
    desc = ("An XFA form whose AcroForm holds no field: its datasets' values are read as fields of their own, and its "
            "template (`xfa.template`) gives its sections, fields, captions, kinds, choices, bindings and four scripts, "
            "none run.")
    sample = _forms_sample(doc, form, "forms-xfa-template", desc)
    data = {"report.author.fullName": "Sam Rivera", "report.contact.email": "sam.rivera@example.com",
            "report.details.summary": "A short technical report on network latency.", "report.details.pages": "12",
            "report.details.format": "digital", "report.details.topics": "Technology",
            "report.details.priority": "urgent", "report.details.reviewed": "1", "report.revision": "3"}
    sample.expected["form"]["xfa"] = {
        "packets": ["preamble", "config", "template", "datasets", "postamble"], "data": data,
        "datasetsSha256": sha256(XFA2_DATASETS), "issue": "XFA layout is not drawn",
        "template": {
            "sections": [{"name": "report", "path": "report", "parent": None},
                         {"name": "Sheet", "path": "report.Sheet", "parent": "report"},
                         {"name": "author", "path": "report.author", "parent": "report"},
                         {"name": "details", "path": "report.details", "parent": "report"}],
            "fields": [
                _tfield("footer", "report.Sheet.footer", "text", "report.Sheet", None, None, "Page footer"),
                _tfield("fullName", "report.author.fullName", "text", "report.author", "Full name",
                        "report.author.fullName", "Sam Rivera", flags=["required"], field="report.author.fullName"),
                _tfield("mail", "report.author.mail", "text", "report.author", "Email", "report.contact.email",
                        "sam.rivera@example.com", field="report.contact.email"),
                _tfield("summary", "report.details.summary", "text", "report.details", "Summary",
                        "report.details.summary", "A short technical report on network latency.",
                        flags=["multiline"], field="report.details.summary"),
                _tfield("pages", "report.details.pages", "text", "report.details", "Pages", "report.details.pages",
                        "12", field="report.details.pages"),
                _tfield("format", "report.details.format", "combo", "report.details", "Format",
                        "report.details.format", "digital", options=[("Printed", "print"), ("Digital", "digital")],
                        field="report.details.format"),
                _tfield("topics", "report.details.topics", "list", "report.details", None, "report.details.topics",
                        "Technology", options=[("Science", "Science"), ("Technology", "Technology")],
                        flags=["multiSelect"], field="report.details.topics"),
                _tfield("priority", "report.details.priority", "radio", "report.details", "Priority",
                        "report.details.priority", "urgent", options=[("Normal", "normal"), ("Urgent", "urgent")],
                        field="report.details.priority"),
                _tfield("reviewed", "report.details.reviewed", "checkbox", "report.details", None,
                        "report.details.reviewed", "1", options=[("1", "1"), ("0", "0")],
                        field="report.details.reviewed"),
                _tfield("submit", "report.details.submit", "pushbutton", "report.details", "Submit",
                        "report.details.submit", None),
                _tfield("revision", "report.revision", "text", "report", None, "report.revision", "3",
                        flags=["readOnly"], field="report.revision")],
            "scripts": 4}}
    return sample


def pdfminer_fields(path):
    """Each terminal field's full name and /V as pdfminer.six's object layer reads them (text decoded, names as
    names); a field's /V not inherited."""
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfparser import PDFParser
    from pdfminer.pdftypes import resolve1
    from pdfminer.psparser import PSLiteral
    from pdfminer.utils import decode_text

    out = {}
    with open(path, "rb") as f:
        doc = PDFDocument(PDFParser(f))
        acro = resolve1(doc.catalog.get("AcroForm"))

        def text(v):
            v = resolve1(v)
            if isinstance(v, bytes):
                return decode_text(v)
            if isinstance(v, PSLiteral):
                return v.name if isinstance(v.name, str) else v.name.decode("latin-1")
            return None if v is None else str(v)

        def walk(fields, prefix):
            for item in resolve1(fields) or []:
                d = resolve1(item)
                part = text(d.get("T"))
                name = prefix if part is None else (part if not prefix else prefix + "." + part)
                kids = [resolve1(k) for k in resolve1(d.get("Kids")) or []]
                if any("T" in k for k in kids):
                    walk(d.get("Kids"), name)
                else:
                    out[name] = text(d.get("V"))

        walk(acro.get("Fields") if acro else [], "")
    return out


def check_plain_values(path, exp, notes):
    """pdfium's and pdfminer.six's reading of each text field's plain value (/V) against the expected (a rich text
    field's plain text among them)."""
    got = {g["name"]: g["value"] for g in pdfium_fields(path) if g["kind"] == "text"}
    miner = pdfminer_fields(path)
    problems = []
    for f in exp["fields"]:
        if f["kind"] != "text":
            continue
        want = f["value"] or ""
        if got.get(f["name"]) != want:
            problems.append({"field": f["name"], "pdfium": got.get(f["name"]), "expected": want})
        if (miner.get(f["name"]) or "") != want:
            problems.append({"field": f["name"], "pdfminer": miner.get(f["name"]), "expected": want})
    if problems:
        notes.append(f"{exp['name']}: pdfium or pdfminer.six reads {len(problems)} plain value(s) differently")
    return "plainValues", {"pdfium": got, "pdfminer": miner, "problems": problems}, {}


def build(seed):
    return [filled(seed), empty(seed), need_appearances(seed), inherited(seed), xfa(seed), flattened(seed), styled(seed),
            rich(seed), held(seed), faces(seed), xfa_template(seed)]

