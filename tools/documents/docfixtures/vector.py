"""Vector content (D6, D7): colour spaces and their RGB, functions (types 0, 2, 3 and 4), and the path recorder the
content writer uses, so every painted path, clip and shading of a sample comes with what a reader must get from the
very operators written.

A painted path, as each expected page lists it (`paths`, content order shared with the runs and images):
- `operators` the construction operators as written (m l c v y h re), `paint` the painting operator (S s f F f* B B* b
  b*), `verbs` (move, line, cubic, close: `v` and `y` become cubics with the current or the end point doubled, `re` a
  move, three lines and a close) and `points` (each verb's points, page space as shown: after the CropBox and /Rotate,
  origin top left, y down; a close has none);
- `fill` (nonzero, evenodd or null), `stroke` (null, or `width` in page points (the line width times the square root
  of the CTM's determinant), `lineWidth` as written, `cap` butt/round/square, `join` miter/round/bevel, `miterLimit`,
  `dash` with `pattern` and `phase` in page points and as written);
- `fillColour` / `strokeColour`: `space` (the family), `components` as given, `rgb` (0-255; see `rgb` below) and
  `tolerance`, or a pattern (`pattern`, `patternType`, `paintType`, and for an uncoloured tiling its colour);
- `clip`: the index of the innermost clip in force (the page's `clips`; each names its `parent`, so the clip is their
  intersection), `alpha` (`fill` ca, `stroke` CA), `blend` (the blend mode), `softMask` (null or its description),
  `ctm`, `bbox` (the points' box, page space).
Colours to RGB, as `ImageData` holds them: grey and RGB round(c * 255); CMYK through 8 bits and (255 - K) - X * (255 -
K) / 255 (the D4 image rule); ICCBased, CalGray and CalRGB by their component count (the profile and calibration not
applied, as D4); Lab through XYZ (D65) to sRGB (tolerance 2); Indexed through its lookup and base; Separation and
DeviceN through their tint transform into the alternate space (tolerance 1).
"""
import math

from .filters import flate
from .pdf import N, Name, Stream, String, fmt_num

CAPS = ("butt", "round", "square")
JOINS = ("miter", "round", "bevel")


def new_vstate():
    return {"fill": GRAY_BLACK, "stroke": GRAY_BLACK, "lw": 1.0, "cap": 0, "join": 0, "miter": 10.0,
            "dash": ([], 0.0), "clip": None, "ca": 1.0, "CA": 1.0, "blend": "Normal", "smask": None}


# -- functions ---------------------------------------------------------------------------------------------------------

class Function:
    """A PDF function: `value(inputs) -> outputs` and the object written."""

    def pdf(self, doc):
        raise NotImplementedError


class Exponential(Function):
    """Type 2: C0 + x^N * (C1 - C0)."""

    def __init__(self, c0, c1, n=1.0, domain=(0, 1)):
        self.c0, self.c1, self.n, self.domain = list(c0), list(c1), n, list(domain)

    def value(self, xs):
        x = min(max(xs[0], self.domain[0]), self.domain[1])
        return [a + (x ** self.n) * (b - a) for a, b in zip(self.c0, self.c1)]

    def pdf(self, doc):
        return {"FunctionType": 2, "Domain": self.domain, "C0": self.c0, "C1": self.c1, "N": self.n}


class Sampled(Function):
    """Type 0, one input, 8 bits a sample, linear interpolation (Order 1)."""

    def __init__(self, samples, outputs, domain=(0, 1)):
        self.samples = samples  # list of tuples (outputs values 0-255)
        self.m = outputs
        self.domain = list(domain)

    def value(self, xs):
        x = min(max(xs[0], self.domain[0]), self.domain[1])
        n = len(self.samples)
        e = (x - self.domain[0]) / (self.domain[1] - self.domain[0]) * (n - 1)
        i = min(int(math.floor(e)), n - 2)
        t = e - i
        out = []
        for k in range(self.m):
            a, b = self.samples[i][k], self.samples[i + 1][k]
            out.append((a + t * (b - a)) / 255)
        return out

    def pdf(self, doc):
        data = bytes(v for s in self.samples for v in s)
        return doc.pdf.add(Stream({"FunctionType": 0, "Domain": self.domain, "Range": [0, 1] * self.m,
                                   "Size": [len(self.samples)], "BitsPerSample": 8, "Filter": N("FlateDecode")},
                                  flate(data), data))


class Stitching(Function):
    """Type 3: k functions over the subdomains `bounds` cut, each input encoded into [Encode]."""

    def __init__(self, functions, bounds, encode, domain=(0, 1)):
        self.functions, self.bounds, self.encode, self.domain = functions, list(bounds), list(encode), list(domain)

    def value(self, xs):
        x = min(max(xs[0], self.domain[0]), self.domain[1])
        edges = [self.domain[0]] + self.bounds + [self.domain[1]]
        k = 0
        while k < len(self.bounds) and x >= self.bounds[k]:
            k += 1
        lo, hi = edges[k], edges[k + 1]
        e0, e1 = self.encode[2 * k], self.encode[2 * k + 1]
        t = e0 if hi == lo else e0 + (x - lo) * (e1 - e0) / (hi - lo)
        return self.functions[k].value([t])

    def pdf(self, doc):
        return {"FunctionType": 3, "Domain": self.domain, "Functions": [f.pdf(doc) for f in self.functions],
                "Bounds": self.bounds, "Encode": self.encode}


class Calculator(Function):
    """Type 4: a PostScript calculator program (the operators the samples use: arithmetic, stack, comparison, if)."""

    def __init__(self, program, inputs, outputs):
        self.program = program
        self.n_in, self.n_out = inputs, outputs
        self.tokens = _ps_parse(program)

    def value(self, xs):
        stack = list(xs)
        _ps_run(self.tokens, stack)
        if len(stack) != self.n_out:
            raise ValueError(f"{self.program}: leaves {len(stack)} values, not {self.n_out}")
        return [min(1.0, max(0.0, v)) for v in stack]

    def pdf(self, doc):
        data = self.program.encode("ascii")
        return doc.pdf.add(Stream({"FunctionType": 4, "Domain": [0, 1] * self.n_in, "Range": [0, 1] * self.n_out},
                                  data))


def _ps_parse(text):
    toks = text.replace("{", " { ").replace("}", " } ").split()
    pos = 0

    def block():
        nonlocal pos
        assert toks[pos] == "{"
        pos += 1
        out = []
        while toks[pos] != "}":
            if toks[pos] == "{":
                out.append(block())
            else:
                t = toks[pos]
                try:
                    out.append(float(t))
                except ValueError:
                    out.append(t)
                pos += 1
        pos += 1
        return out

    return block()


def _ps_run(tokens, s):
    for t in tokens:
        if isinstance(t, float):
            s.append(t)
        elif isinstance(t, list):
            s.append(t)
        elif t == "add":
            b, a = s.pop(), s.pop(); s.append(a + b)
        elif t == "sub":
            b, a = s.pop(), s.pop(); s.append(a - b)
        elif t == "mul":
            b, a = s.pop(), s.pop(); s.append(a * b)
        elif t == "div":
            b, a = s.pop(), s.pop(); s.append(a / b)
        elif t == "neg":
            s.append(-s.pop())
        elif t == "abs":
            s.append(abs(s.pop()))
        elif t == "dup":
            s.append(s[-1])
        elif t == "exch":
            s[-1], s[-2] = s[-2], s[-1]
        elif t == "pop":
            s.pop()
        elif t == "index":
            n = int(s.pop()); s.append(s[-1 - n])
        elif t == "copy":
            n = int(s.pop()); s.extend(s[-n:] if n else [])
        elif t == "sqrt":
            s.append(math.sqrt(s.pop()))
        elif t in ("gt", "lt", "ge", "le"):
            b, a = s.pop(), s.pop()
            s.append({"gt": a > b, "lt": a < b, "ge": a >= b, "le": a <= b}[t])
        elif t == "true":
            s.append(True)
        elif t == "false":
            s.append(False)
        elif t == "if":
            proc, cond = s.pop(), s.pop()
            if cond:
                _ps_run(proc, s)
        elif t == "ifelse":
            p2, p1, cond = s.pop(), s.pop(), s.pop()
            _ps_run(p1 if cond else p2, s)
        else:
            raise ValueError(f"PostScript operator {t} is not used by the samples")


# -- colour spaces -----------------------------------------------------------------------------------------------------

def _b(v):
    return round(min(1.0, max(0.0, v)) * 255)


def cmyk_to_rgb(c, m, y, k):
    from .images import cmyk_rgb

    return cmyk_rgb(_b(c), _b(m), _b(y), _b(k))


class ColourSpace:
    family = "?"
    n = 0
    tolerance = 0

    def rgb(self, comps):
        raise NotImplementedError

    def value(self, doc):
        """The object written in /ColorSpace resources (a name for the device spaces)."""
        return N(self.family)

    def initial(self):
        return [0.0] * self.n


class Device(ColourSpace):
    def __init__(self, family):
        self.family = family
        self.n = {"DeviceGray": 1, "DeviceRGB": 3, "DeviceCMYK": 4}[family]

    def rgb(self, comps):
        if self.n == 1:
            return (_b(comps[0]),) * 3
        if self.n == 3:
            return tuple(_b(c) for c in comps)
        return cmyk_to_rgb(*comps)

    def initial(self):
        return [0.0, 0.0, 0.0, 1.0] if self.n == 4 else [0.0] * self.n


GRAY = Device("DeviceGray")
RGB = Device("DeviceRGB")
CMYK = Device("DeviceCMYK")


class ICCBased(ColourSpace):
    family = "ICCBased"

    def __init__(self, n, profile, alternate):
        self.n, self.profile, self.alternate = n, profile, alternate
        self.ref = None

    def rgb(self, comps):
        return {1: GRAY, 3: RGB, 4: CMYK}[self.n].rgb(comps)

    def value(self, doc):
        if self.ref is None:
            self.ref = doc.pdf.add(Stream({"N": self.n, "Alternate": N(self.alternate), "Filter": N("FlateDecode")},
                                          flate(self.profile), self.profile))
        return [N("ICCBased"), self.ref]


class CalGray(ColourSpace):
    family = "CalGray"
    n = 1

    def rgb(self, comps):
        return GRAY.rgb(comps)

    def value(self, doc):
        return [N("CalGray"), {"WhitePoint": [0.9505, 1, 1.089], "Gamma": 2.2}]


class CalRGB(ColourSpace):
    family = "CalRGB"
    n = 3

    def rgb(self, comps):
        return RGB.rgb(comps)

    def value(self, doc):
        return [N("CalRGB"), {"WhitePoint": [0.9505, 1, 1.089], "Gamma": [2.2, 2.2, 2.2],
                              "Matrix": [0.4124, 0.2126, 0.0193, 0.3576, 0.7152, 0.1192, 0.1805, 0.0722, 0.9505]}]


class Lab(ColourSpace):
    family = "Lab"
    n = 3
    tolerance = 2

    def __init__(self, white=(0.9505, 1, 1.089), rng=(-100, 100, -100, 100)):
        self.white, self.range = list(white), list(rng)

    def rgb(self, comps):
        from .images import lab_rgb

        return lab_rgb(comps[0], comps[1], comps[2], self.white)

    def value(self, doc):
        return [N("Lab"), {"WhitePoint": self.white, "Range": self.range}]

    def initial(self):
        return [0.0, 0.0, 0.0]


class Indexed(ColourSpace):
    family = "Indexed"
    n = 1

    def __init__(self, base, hival, lookup):
        self.base, self.hival, self.lookup = base, hival, bytes(lookup)
        self.tolerance = base.tolerance

    def rgb(self, comps):
        idx = max(0, min(self.hival, round(comps[0])))
        m = self.base.n
        return self.base.rgb([v / 255 for v in self.lookup[idx * m:(idx + 1) * m]])

    def value(self, doc):
        return [N("Indexed"), self.base.value(doc), self.hival, String(self.lookup, hex=True)]


class Separation(ColourSpace):
    family = "Separation"
    n = 1
    tolerance = 1

    def __init__(self, name, alternate, function):
        self.name, self.alternate, self.function = name, alternate, function
        self.obj = None

    def rgb(self, comps):
        return self.alternate.rgb(self.function.value(comps))

    def value(self, doc):
        if self.obj is None:
            self.obj = [N("Separation"), N(self.name), self.alternate.value(doc), self.function.pdf(doc)]
        return self.obj

    def initial(self):
        return [1.0]


class DeviceN(ColourSpace):
    family = "DeviceN"
    tolerance = 1

    def __init__(self, names, alternate, function, attributes=None):
        self.names, self.alternate, self.function = names, alternate, function
        self.n = len(names)
        self.attributes = attributes
        self.obj = None

    def rgb(self, comps):
        return self.alternate.rgb(self.function.value(comps))

    def value(self, doc):
        if self.obj is None:
            self.obj = [N("DeviceN"), [N(n) for n in self.names], self.alternate.value(doc), self.function.pdf(doc)]
            if self.attributes:
                self.obj.append(self.attributes)
        return self.obj

    def initial(self):
        return [1.0] * self.n


class PatternSpace(ColourSpace):
    """/Pattern, or [/Pattern base] for uncoloured tiling patterns (their colour given with the pattern)."""
    family = "Pattern"

    def __init__(self, base=None):
        self.base = base
        self.n = base.n if base else 0

    def value(self, doc):
        return [N("Pattern"), self.base.value(doc)] if self.base else N("Pattern")


class Colour:
    __slots__ = ("space", "components", "pattern")

    def __init__(self, space, components, pattern=None):
        self.space, self.components, self.pattern = space, list(components), pattern

    def describe(self):
        if self.pattern is not None:
            name, pat = self.pattern
            out = {"space": "Pattern", "pattern": name, "patternType": pat.pattern_type,
                   "paintType": getattr(pat, "paint_type", None), "shadingType": getattr(pat, "shading_type", None)}
            if self.space.base is not None:
                out["components"] = [r4(c) for c in self.components]
                out["rgb"] = list(self.space.base.rgb(self.components))
                out["tolerance"] = self.space.base.tolerance
            return out
        return {"space": self.space.family, "components": [r4(c) for c in self.components],
                "rgb": list(self.space.rgb(self.components)), "tolerance": self.space.tolerance}


GRAY_BLACK = Colour(GRAY, [0.0])


def r4(x):
    v = round(x, 4)
    return 0.0 if v == 0 else v


def fmt(*xs):
    return " ".join(fmt_num(float(x)) if not isinstance(x, int) else str(x) for x in xs)


# -- the recorder ------------------------------------------------------------------------------------------------------

class VectorContent:
    """Path, colour, line-state, clipping, transparency and shading operators for `Content` (document.py), each
    emitted as written and recorded as a reader's interpreter must see it."""

    record = True

    # -- line state and colours ----------------------------------------------------------------------------------------

    def w(self, v):
        self.vs["lw"] = float(v)
        self.emit(fmt(v) + " w")

    def J(self, v):
        self.vs["cap"] = int(v)
        self.emit(f"{int(v)} J")

    def j(self, v):
        self.vs["join"] = int(v)
        self.emit(f"{int(v)} j")

    def M(self, v):
        self.vs["miter"] = float(v)
        self.emit(fmt(v) + " M")

    def d(self, pattern, phase):
        self.vs["dash"] = (list(pattern), float(phase))
        self.emit("[" + " ".join(fmt(x) for x in pattern) + "] " + fmt(phase) + " d")

    def g(self, v):
        self.vs["fill"] = Colour(GRAY, [v])
        self.emit(fmt(v) + " g")

    def G(self, v):
        self.vs["stroke"] = Colour(GRAY, [v])
        self.emit(fmt(v) + " G")

    def rg(self, r, g, b):
        self.vs["fill"] = Colour(RGB, [r, g, b])
        self.emit(fmt(r, g, b) + " rg")

    def RG(self, r, g, b):
        self.vs["stroke"] = Colour(RGB, [r, g, b])
        self.emit(fmt(r, g, b) + " RG")

    def k(self, c, m, y, kk):
        self.vs["fill"] = Colour(CMYK, [c, m, y, kk])
        self.emit(fmt(c, m, y, kk) + " k")

    def K(self, c, m, y, kk):
        self.vs["stroke"] = Colour(CMYK, [c, m, y, kk])
        self.emit(fmt(c, m, y, kk) + " K")

    def _space_name(self, name, space):
        if isinstance(space, Device) or (isinstance(space, PatternSpace) and space.base is None):
            return space.family if name is None else name
        self._use("ColorSpace", name, space.value(self.page.doc))
        return name

    def cs(self, name, space):
        """`name` the resource (None for a device space or /Pattern written by its own name)."""
        self.vs["fill"] = Colour(space, space.initial())
        self.vs["fill_space"] = (self._space_name(name, space), space)
        self.emit(f"/{self.vs['fill_space'][0]} cs")

    def CS(self, name, space):
        self.vs["stroke"] = Colour(space, space.initial())
        self.vs["stroke_space"] = (self._space_name(name, space), space)
        self.emit(f"/{self.vs['stroke_space'][0]} CS")

    def sc(self, *comps, op="sc"):
        space = self.vs["fill"].space
        self.vs["fill"] = Colour(space, comps)
        self.emit(fmt(*comps) + f" {op}")

    def SC(self, *comps, op="SC"):
        space = self.vs["stroke"].space
        self.vs["stroke"] = Colour(space, comps)
        self.emit(fmt(*comps) + f" {op}")

    def scn_pattern(self, name, pattern, comps=(), stroke=False):
        key = "stroke" if stroke else "fill"
        space = self.vs[key].space
        self._use("Pattern", name, pattern.ref)
        self.vs[key] = Colour(space, comps, pattern=(name, pattern))
        self.emit((fmt(*comps) + " " if comps else "") + f"/{name} " + ("SCN" if stroke else "scn"))

    def gs(self, name, ext):
        """An ExtGState (an ExtGState object below): its ca, CA, BM, SMask and line entries applied."""
        self._use("ExtGState", name, ext.ref)
        for key, value in ext.entries.items():
            if key == "ca":
                self.vs["ca"] = float(value)
            elif key == "CA":
                self.vs["CA"] = float(value)
            elif key == "BM":
                self.vs["blend"] = value.value if isinstance(value, Name) else value
            elif key == "SMask":
                self.vs["smask"] = ext.smask_describe
            elif key == "LW":
                self.vs["lw"] = float(value)
            elif key == "LC":
                self.vs["cap"] = int(value)
            elif key == "LJ":
                self.vs["join"] = int(value)
            elif key == "ML":
                self.vs["miter"] = float(value)
            elif key == "D":
                self.vs["dash"] = (list(value[0]), float(value[1]))
        self.emit(f"/{name} gs")

    # -- path construction ---------------------------------------------------------------------------------------------

    def _pt(self, x, y):
        from .document import apply

        return apply(self.ctm, x, y)

    def m(self, x, y):
        self.path_ops.append(("m", [self._pt(x, y)]))
        self._cur = (x, y)
        self._start = (x, y)
        self.emit(fmt(x, y) + " m")

    def l(self, x, y):
        self.path_ops.append(("l", [self._pt(x, y)]))
        self._cur = (x, y)
        self.emit(fmt(x, y) + " l")

    def c(self, x1, y1, x2, y2, x3, y3):
        self.path_ops.append(("c", [self._pt(x1, y1), self._pt(x2, y2), self._pt(x3, y3)]))
        self._cur = (x3, y3)
        self.emit(fmt(x1, y1, x2, y2, x3, y3) + " c")

    def v(self, x2, y2, x3, y3):
        self.path_ops.append(("v", [self._pt(*self._cur), self._pt(x2, y2), self._pt(x3, y3)]))
        self._cur = (x3, y3)
        self.emit(fmt(x2, y2, x3, y3) + " v")

    def y(self, x1, y1, x3, y3):
        self.path_ops.append(("y", [self._pt(x1, y1), self._pt(x3, y3), self._pt(x3, y3)]))
        self._cur = (x3, y3)
        self.emit(fmt(x1, y1, x3, y3) + " y")

    def h(self):
        self.path_ops.append(("h", []))
        self._cur = self._start
        self.emit("h")

    def re(self, x, y, w, hh):
        pts = [self._pt(x, y), self._pt(x + w, y), self._pt(x + w, y + hh), self._pt(x, y + hh)]
        self.path_ops.append(("re", pts))
        self._cur = self._start = (x, y)
        self.emit(fmt(x, y, w, hh) + " re")

    def W(self, evenodd=False):
        self._clip_rule = "evenodd" if evenodd else "nonzero"
        self.emit("W*" if evenodd else "W")

    # -- painting ------------------------------------------------------------------------------------------------------

    PAINT = {"S": (None, True), "s": (None, True), "f": ("nonzero", False), "F": ("nonzero", False),
             "f*": ("evenodd", False), "B": ("nonzero", True), "B*": ("evenodd", True), "b": ("nonzero", True),
             "b*": ("evenodd", True), "n": (None, False)}

    def paint(self, op):
        fill, stroke = self.PAINT[op]
        ops = self.path_ops
        if op in ("s", "b", "b*"):
            ops = ops + [("h", [])]
        verbs, points = _verbs(ops)
        page = self.page
        disp = [[r4(c) for c in page.display(*p)] for p in points]
        clip_rule = getattr(self, "_clip_rule", None)
        self._clip_rule = None
        self.emit(op)
        operators = [o for o, _ in self.path_ops]
        self.path_ops = []
        if clip_rule is not None and self.record:
            clips = page.clips
            clips.append({"index": len(clips), "rule": clip_rule, "operators": operators, "verbs": verbs,
                          "points": disp, "parent": self.vs["clip"], "bbox": _bbox(disp)})
            self.vs["clip"] = len(clips) - 1
        elif clip_rule is not None:
            self.vs["clip"] = None
        if op == "n" or not self.record:
            return None
        rec = {"order": page.next_order(), "operators": operators, "paint": op, "verbs": verbs, "points": disp,
               "fill": fill, "stroke": self._stroke() if stroke else None,
               "fillColour": self.vs["fill"].describe() if fill else None,
               "strokeColour": self.vs["stroke"].describe() if stroke else None,
               "clip": self.vs["clip"], "alpha": {"fill": r4(self.vs["ca"]), "stroke": r4(self.vs["CA"])},
               "blend": self.vs["blend"], "softMask": self.vs["smask"], "group": self.vs.get("group"),
               "ctm": [r4(v) for v in self.ctm], "bbox": _bbox(disp)}
        page.paths.append(rec)
        return rec

    def _stroke(self):
        a, b, c, d = self.ctm[:4]
        scale = math.sqrt(abs(a * d - b * c))
        pattern, phase = self.vs["dash"]
        return {"width": r4(self.vs["lw"] * scale), "lineWidth": r4(self.vs["lw"]), "cap": CAPS[self.vs["cap"]],
                "join": JOINS[self.vs["join"]], "miterLimit": r4(self.vs["miter"]),
                "dash": {"pattern": [r4(x * scale) for x in pattern], "phase": r4(phase * scale),
                         "patternWritten": [r4(x) for x in pattern], "phaseWritten": r4(phase)}}

    def S(self):
        return self.paint("S")

    def f(self):
        return self.paint("f")

    def B(self):
        return self.paint("B")

    def n(self):
        return self.paint("n")

    # -- shadings ------------------------------------------------------------------------------------------------------

    def sh(self, name, shading):
        self._use("Shading", name, shading.ref)
        self.emit(f"/{name} sh")
        if not self.record:
            return None
        rec = {"order": self.page.next_order(), "resource": name, "shadingType": shading.shading_type,
               "colourSpace": shading.space.family, "object": [shading.ref.num, 0], "clip": self.vs["clip"],
               "alpha": {"fill": r4(self.vs["ca"]), "stroke": r4(self.vs["CA"])}, "blend": self.vs["blend"],
               "softMask": self.vs["smask"], "group": self.vs.get("group"),
               "ctm": [r4(v) for v in self.ctm], "samples": shading.samples(self.page, self.ctm)}
        self.page.shadings.append(rec)
        return rec

    def vector_extras(self):
        """What a run or an image also needs for drawing: its colour (runs), the clip and the transparency."""
        return {"fillColour": self.vs["fill"].describe(), "strokeColour": self.vs["stroke"].describe(),
                "clip": self.vs["clip"], "alpha": {"fill": r4(self.vs["ca"]), "stroke": r4(self.vs["CA"])},
                "blend": self.vs["blend"], "group": self.vs.get("group")}

    def form_clip(self, bbox, matrix):
        """A form XObject's BBox clips what it draws: recorded as a clip (rule nonzero) under the clip in force."""
        from .document import apply, mul

        if not self.record or not self.page.doc.vector:
            return None
        m = mul(matrix, self.ctm)
        x0, y0, x1, y1 = bbox
        pts = [apply(m, x0, y0), apply(m, x1, y0), apply(m, x1, y1), apply(m, x0, y1)]
        disp = [[r4(c) for c in self.page.display(*p)] for p in pts]
        clips = self.page.clips
        clips.append({"index": len(clips), "rule": "nonzero", "operators": ["BBox"],
                      "verbs": ["move", "line", "line", "line", "close"], "points": disp, "parent": self.vs["clip"],
                      "bbox": _bbox(disp), "form": True})
        return len(clips) - 1


def _verbs(ops):
    verbs, points = [], []
    for op, pts in ops:
        if op == "m":
            verbs.append("move"); points += pts
        elif op == "l":
            verbs.append("line"); points += pts
        elif op in ("c", "v", "y"):
            verbs.append("cubic"); points += pts
        elif op == "h":
            verbs.append("close")
        elif op == "re":
            verbs += ["move", "line", "line", "line", "close"]
            points += pts
    return verbs, points


def _bbox(points):
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [r4(min(xs)), r4(min(ys)), r4(max(xs) - min(xs)), r4(max(ys) - min(ys))]


# -- resources ----------------------------------------------------------------------------------------------------------

class ExtGState:
    def __init__(self, doc, entries, smask_describe=None):
        self.entries = entries
        self.smask_describe = smask_describe
        self.ref = doc.pdf.add(dict(entries))


class Shading:
    """A shading dictionary (types 1-7) and a few points where its colour is known exactly (`samples`: page point and
    RGB, for the drawer's check; meshes give their vertices' colours)."""

    def __init__(self, doc, shading_type, space, entries, probes=(), stream=None):
        self.shading_type = shading_type
        self.space = space
        self.probes = probes  # (user x, user y, components)
        d = {"ShadingType": shading_type, "ColorSpace": space.value(doc), **entries}
        if stream is not None:
            data = stream
            d["Filter"] = N("FlateDecode")
            self.ref = doc.pdf.add(Stream(d, flate(data), data))
        else:
            self.ref = doc.pdf.add(d)

    def samples(self, page, ctm):
        from .document import apply

        out = []
        for x, y, comps in self.probes:
            px, py = page.display(*apply(ctm, x, y))
            out.append({"point": [r4(px), r4(py)], "components": [r4(c) for c in comps],
                        "rgb": list(self.space.rgb(comps)), "tolerance": max(2, self.space.tolerance)})
        return out


class ShadingPattern:
    pattern_type = 2

    def __init__(self, doc, shading, matrix=None):
        self.shading = shading
        self.shading_type = shading.shading_type
        d = {"Type": N("Pattern"), "PatternType": 2, "Shading": shading.ref}
        if matrix:
            d["Matrix"] = list(matrix)
        self.ref = doc.pdf.add(d)


class TilingPattern:
    pattern_type = 1

    def __init__(self, doc, paint_type, bbox, xstep, ystep, content, resources=None, matrix=None):
        self.paint_type = paint_type
        d = {"Type": N("Pattern"), "PatternType": 1, "PaintType": paint_type, "TilingType": 1, "BBox": list(bbox),
             "XStep": xstep, "YStep": ystep, "Resources": resources or {}, "Filter": N("FlateDecode")}
        if matrix:
            d["Matrix"] = list(matrix)
        self.ref = doc.pdf.add(Stream(d, flate(content), content))


# -- the reference check: pdfium's path objects --------------------------------------------------------------------------

def pdfium_paths(path, password=None):
    """Every path object pdfium finds on each page (through form XObjects), its points in PDF user space, draw mode,
    stroke width, colours and dash, in content order."""
    import ctypes

    import pypdfium2 as pdfium
    import pypdfium2.raw as raw

    def matrix_of(obj):
        m = raw.FS_MATRIX()
        if not raw.FPDFPageObj_GetMatrix(obj, m):
            return (1, 0, 0, 1, 0, 0)
        return (m.a, m.b, m.c, m.d, m.e, m.f)

    def mulm(m, n):
        a, b, c, d, e, f = m
        A, B, C, D, E, F = n
        return (a * A + b * C, a * B + b * D, c * A + d * C, c * B + d * D, e * A + f * C + E, e * B + f * D + F)

    def walk(get_count, get_obj, outer, out):
        for i in range(get_count()):
            obj = get_obj(i)
            kind = raw.FPDFPageObj_GetType(obj)
            m = mulm(matrix_of(obj), outer)
            if kind == raw.FPDF_PAGEOBJ_FORM:
                walk(lambda o=obj: raw.FPDFFormObj_CountObjects(o), lambda k, o=obj: raw.FPDFFormObj_GetObject(o, k),
                     m, out)
                continue
            if kind != raw.FPDF_PAGEOBJ_PATH:
                continue
            segs = []
            for k in range(raw.FPDFPath_CountSegments(obj)):
                seg = raw.FPDFPath_GetPathSegment(obj, k)
                x, y = ctypes.c_float(), ctypes.c_float()
                raw.FPDFPathSegment_GetPoint(seg, x, y)
                px, py = x.value * m[0] + y.value * m[2] + m[4], x.value * m[1] + y.value * m[3] + m[5]
                segs.append((raw.FPDFPathSegment_GetType(seg), px, py, bool(raw.FPDFPathSegment_GetClose(seg))))
            mode, stroke = ctypes.c_int(), ctypes.c_int()
            raw.FPDFPath_GetDrawMode(obj, mode, stroke)
            width = ctypes.c_float()
            raw.FPDFPageObj_GetStrokeWidth(obj, width)
            cols = []
            for getter in (raw.FPDFPageObj_GetFillColor, raw.FPDFPageObj_GetStrokeColor):
                r, g, b, a = (ctypes.c_uint() for _ in range(4))
                ok = getter(obj, r, g, b, a)
                cols.append([r.value, g.value, b.value, a.value] if ok else None)
            n = raw.FPDFPageObj_GetDashCount(obj)
            dash = (ctypes.c_float * max(n, 1))()
            if n > 0:
                raw.FPDFPageObj_GetDashArray(obj, dash, n)
            out.append({"segments": segs, "fill": {0: None, 1: "evenodd", 2: "nonzero"}.get(mode.value),
                        "stroke": bool(stroke.value), "width": width.value, "matrixScale": math.sqrt(abs(
                            m[0] * m[3] - m[1] * m[2])), "fillColour": cols[0], "strokeColour": cols[1],
                        "dash": [dash[k] for k in range(n)] if n > 0 else [],
                        "cap": raw.FPDFPageObj_GetLineCap(obj), "join": raw.FPDFPageObj_GetLineJoin(obj)})

    doc = pdfium.PdfDocument(path, password=password or None)
    pages = []
    try:
        for i in range(len(doc)):
            page = doc[i]
            out = []
            walk(lambda p=page: raw.FPDFPage_CountObjects(p), lambda k, p=page: raw.FPDFPage_GetObject(p, k),
                 (1, 0, 0, 1, 0, 0), out)
            pages.append(out)
            page.close()
    finally:
        doc.close()
    return pages


def check_paths(exp_pages, pdfium_pages, geometry):
    """Matches each expected path to pdfium's path objects in order and measures the largest point difference (page
    points), and whether the fill rule, stroking, width, dash, cap, join and device colours agree."""
    report = []
    for exp, got in zip(exp_pages, pdfium_pages):
        paths = exp.get("paths") or []
        if not paths and not got:
            continue
        geom = geometry(exp)
        worst = 0.0
        matched = 0
        differences = []
        k = 0
        for p in paths:
            want = [tuple(pt) for pt in p["points"]]
            found = None
            for j in range(k, len(got)):
                cand = got[j]
                pts = [geom(x, y) for t, x, y, c in cand["segments"]]
                if len(pts) >= len(want) and _close(pts, want):
                    found = (cand, pts)
                    k = j + 1
                    break
            if found is None:
                differences.append({"order": p["order"], "problem": "no pdfium path object matched"})
                continue
            cand, pts = found
            matched += 1
            worst = max(worst, _worst(pts, want))
            if (cand["fill"] or None) != p["fill"]:
                differences.append({"order": p["order"], "fill": [p["fill"], cand["fill"]]})
            if cand["stroke"] != (p["stroke"] is not None):
                differences.append({"order": p["order"], "stroke": [p["stroke"] is not None, cand["stroke"]]})
            elif p["stroke"] is not None:
                if abs(cand["width"] - p["stroke"]["lineWidth"]) > 0.01:
                    differences.append({"order": p["order"], "lineWidth": [p["stroke"]["lineWidth"], cand["width"]]})
                if CAPS[cand["cap"]] != p["stroke"]["cap"] or JOINS[cand["join"]] != p["stroke"]["join"]:
                    differences.append({"order": p["order"], "capJoin": [p["stroke"]["cap"], p["stroke"]["join"],
                                                                         CAPS[cand["cap"]], JOINS[cand["join"]]]})
                if [round(x, 3) for x in cand["dash"]] != [round(x, 3) for x in p["stroke"]["dash"]["patternWritten"]]:
                    differences.append({"order": p["order"], "dash": [p["stroke"]["dash"]["patternWritten"],
                                                                      cand["dash"]]})
            for key, col in (("fillColour", cand["fillColour"]), ("strokeColour", cand["strokeColour"])):
                want_col = p[key]
                if want_col and want_col.get("rgb") and col and want_col["space"] in ("DeviceGray", "DeviceRGB"):
                    diff = max(abs(a - b) for a, b in zip(want_col["rgb"], col[:3]))
                    if diff > 1:
                        differences.append({"order": p["order"], key: [want_col["rgb"], col[:3]]})
        report.append({"page": exp["index"], "paths": len(paths), "pdfiumPaths": len(got), "matched": matched,
                       "largestPointDifference": round(worst, 4), "differences": differences})
    return report


def _close(pts, want, tol=0.05):
    if len(pts) < len(want):
        return False
    return _worst(pts, want) <= tol


def _worst(pts, want):
    """pdfium may add the closing point of a rectangle or of a closed subpath: compare the expected points in order
    against pdfium's, skipping pdfium points that repeat a subpath's start."""
    i = 0
    worst = 0.0
    for w in want:
        best = None
        while i < len(pts):
            d = max(abs(pts[i][0] - w[0]), abs(pts[i][1] - w[1]))
            i += 1
            if d <= 0.05 or best is None:
                best = d
                if d <= 0.05:
                    break
        if best is None:
            return float("inf")
        worst = max(worst, best)
    return worst
