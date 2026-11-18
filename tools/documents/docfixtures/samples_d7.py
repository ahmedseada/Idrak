"""D7, vector content and drawing: pages using every path operator, both fill rules, caps, joins, miter limits, dashes
with phases, clipping (W, W*, nested, a form's BBox), every colour space (device, ICCBased, Indexed, CalGray, CalRGB,
Lab, Separation and DeviceN through tint transforms of each function type), shadings (axial, radial, function-based,
free-form, lattice, Coons and tensor meshes), tiling patterns (coloured and uncoloured) and shading patterns,
transparency (ca, CA, blend modes, a luminosity soft mask, isolated and knockout groups), text through an embedded Type
1 font (made here), Type 3 glyphs, TrueType and CFF fonts, images placed by turned and skewed matrices, and a turned and
a cropped page. Each page lists its paths, clips and shadings (vector.py gives the format) and pdfium draws it at the
recorded DPI; pdfium's own path objects are matched against the expected paths."""
import math

from .common import DEJAVU, FONTS, Sample, base_expected, helvetica
from .document import Doc
from .fontprograms import CffSimple, CffType0, Type1Embedded, cff_outlines
from .fonts import TrueTypeSimple, Type3
from .images import Image, gray_icc, pattern, srgb_icc
from .pdf import N
from .samples_d3 import TYPE3_PROCS
from .vector import (CMYK, GRAY, RGB, CalGray, CalRGB, Calculator, DeviceN, Exponential, ExtGState, ICCBased, Indexed,
                     Lab, PatternSpace, Sampled, Separation, Shading, ShadingPattern, Stitching, TilingPattern)

D7 = ["D7"]
DEJAVU_CFF = f"{FONTS}/dejavu-sans-cff.otf"
W, H = 400, 300


def _doc(name, version="1.7", seed="2026"):
    doc = Doc(name, version, seed)
    doc.vector = True
    return doc


def _title(doc, p, text, y=None):
    f = helvetica(doc)
    c = p.content
    c.BT()
    c.Tf(f, 11)
    c.Td(14, y if y is not None else p.size()[1] - 18)
    c.show(text)
    c.ET()
    p.line(text)
    return f


def _sample(doc, data, w, name, desc, files=None, extra=None):
    exp = base_expected(doc, data, name, "vector", D7, desc, w, objects=False)
    exp.update(extra or {})
    return Sample(name, "d7", "vector", D7, desc, data, exp, files or {})


def _finish(doc, kind="table"):
    doc.finish()
    return doc.write(kind)


# -- paths -----------------------------------------------------------------------------------------------------------------

def paths(seed):
    doc = _doc("vector-paths", "1.4", seed)
    p = doc.page(W, H)
    _title(doc, p, "Path construction and painting")
    c = p.content
    c.q()
    c.w(2)
    c.RG(0.1, 0.3, 0.7)
    c.m(20, 230); c.l(60, 262); c.l(100, 230); c.l(140, 262); c.S()              # m l, S (open)
    c.m(160, 230); c.c(180, 280, 220, 200, 240, 252); c.S()                      # c
    c.m(255, 230); c.v(295, 290, 320, 232); c.S()                                # v (first control = current point)
    c.m(330, 230); c.y(350, 290, 385, 242); c.S()                                # y (second control = end point)
    c.RG(0.6, 0.1, 0.4)
    c.m(20, 150); c.l(80, 150); c.l(50, 200); c.paint("s")                       # s closes, then strokes
    c.m(100, 150); c.l(160, 150); c.l(130, 200); c.h(); c.S()                   # h, S
    c.rg(0.9, 0.5, 0.1); c.re(180, 150, 60, 45); c.f()                           # re f
    c.rg(0.2, 0.6, 0.3); c.re(250, 150, 60, 45); c.paint("F")                    # F (as f)
    c.rg(0.95, 0.85, 0.3); c.RG(0.3, 0.2, 0.1)
    c.re(320, 150, 60, 45); c.B()                                                # B
    c.m(20, 40); c.l(100, 40); c.l(100, 120); c.l(20, 120); c.paint("b")         # b closes, fills and strokes
    star = [(140 + 40 * math.sin(k * 4 * math.pi / 5), 80 + 40 * math.cos(k * 4 * math.pi / 5)) for k in range(5)]
    c.rg(0.5, 0.7, 0.95)
    c.m(*star[0])
    for pt in star[1:]:
        c.l(*pt)
    c.paint("b*")                                                                # b* (even-odd)
    c.m(200, 40); c.l(260, 40); c.l(260, 100); c.l(200, 100); c.h()              # two subpaths in one path
    c.m(230, 70); c.l(290, 70); c.l(290, 130); c.l(230, 130); c.h()
    c.paint("B*")                                                                # B* (even-odd: the overlap empty)
    c.m(310, 40); c.c(310, 140, 390, 140, 390, 40); c.c(370, 70, 330, 70, 310, 40); c.f()   # closed by f (implicit)
    c.Q()
    data, w = _finish(doc)
    desc = ("Every path construction operator (m l c v y h re) and every painting operator (S s f F f* B B* b b*), "
            "an implicitly closed fill, a path of two subpaths painted with B*.")
    return _sample(doc, data, w, "vector-paths", desc)


def fill_rules(seed):
    doc = _doc("vector-fill-rules", "1.4", seed)
    p = doc.page(W, H)
    _title(doc, p, "Fill rules: nonzero (left of each pair) and even-odd (right)")
    c = p.content

    def star(cx, cy, r):
        pts = [(cx + r * math.sin(k * 4 * math.pi / 5), cy + r * math.cos(k * 4 * math.pi / 5)) for k in range(5)]
        c.m(*pts[0])
        for pt in pts[1:]:
            c.l(*pt)
        c.h()

    def squares(x, y, reverse_inner):
        c.m(x, y); c.l(x + 70, y); c.l(x + 70, y + 70); c.l(x, y + 70); c.h()
        inner = [(x + 20, y + 20), (x + 50, y + 20), (x + 50, y + 50), (x + 20, y + 50)]
        if reverse_inner:
            inner = inner[::-1]
        c.m(*inner[0])
        for pt in inner[1:]:
            c.l(*pt)
        c.h()

    def overlap(x, y):
        c.re(x, y, 50, 50)
        c.re(x + 25, y + 25, 50, 50)

    c.rg(0.15, 0.45, 0.8)
    star(50, 220, 38); c.f()
    star(140, 220, 38); c.paint("f*")
    c.rg(0.8, 0.35, 0.15)
    squares(200, 185, False); c.f()          # same direction: no hole with nonzero
    squares(290, 185, False); c.paint("f*")  # a hole with even-odd
    c.rg(0.2, 0.6, 0.35)
    squares(20, 70, True); c.f()             # opposite direction: a hole with both rules
    squares(110, 70, True); c.paint("f*")
    c.rg(0.55, 0.3, 0.7)
    overlap(200, 70); c.f()
    overlap(290, 70); c.paint("f*")
    c.rg(0.9, 0.7, 0.1)
    c.m(30, 20); c.c(90, 60, 90, -10, 150, 30); c.c(90, 70, 90, 0, 30, 20); c.f()   # a self-crossing curve
    c.m(180, 20); c.c(240, 60, 240, -10, 300, 30); c.c(240, 70, 240, 0, 180, 20); c.paint("f*")
    data, w = _finish(doc)
    desc = ("Both fill rules on the same shapes: a self-intersecting star, nested squares turning the same way and the "
            "opposite way, two overlapping rectangles in one path, and a self-crossing curve.")
    return _sample(doc, data, w, "vector-fill-rules", desc)


def strokes(seed):
    doc = _doc("vector-strokes", "1.4", seed)
    p = doc.page(W, H)
    _title(doc, p, "Caps, joins, miter limits, dashes and widths")
    c = p.content
    c.RG(0.1, 0.1, 0.1)
    c.w(12)
    for k in range(3):                              # caps
        c.J(k)
        c.m(30, 250 - 26 * k); c.l(120, 250 - 26 * k); c.S()
    c.J(0)
    c.w(10)
    for k in range(3):                              # joins
        c.j(k)
        x = 150 + 80 * k
        c.m(x, 190); c.l(x + 30, 250); c.l(x + 60, 190); c.S()
    c.j(0)
    c.w(4)
    for k, limit in enumerate((1.5, 10)):           # miter limit: a sharp angle bevelled under 1.5, mitred under 10
        c.M(limit)
        x = 30 + 50 * k
        c.m(x, 110); c.l(x + 15, 165); c.l(x + 30, 110); c.S()
    c.M(10)
    c.w(3)
    c.RG(0.1, 0.35, 0.75)
    for k, (pat, phase) in enumerate([([6, 3], 0), ([6, 3], 4), ([12, 4, 2, 4], 0), ([2], 1)]):
        c.d(pat, phase)
        c.m(150, 150 - 14 * k); c.l(380, 150 - 14 * k); c.S()
    c.J(1)
    c.d([0, 7], 0)                                  # zero-length dashes with round caps: dots
    c.m(150, 88); c.l(380, 88); c.S()
    c.J(0)
    c.d([8, 4], 2)
    c.m(150, 20); c.c(200, 80, 260, -20, 300, 60); c.S()   # a dashed curve
    c.rg(0.9, 0.9, 0.7)
    c.re(310, 20, 70, 40); c.B()                    # a dashed closed rectangle, filled and stroked
    c.d([], 0)
    c.RG(0.7, 0.1, 0.1)
    for k, width in enumerate((0, 0.5, 3)):         # widths: 0 is the thinnest line a device draws
        c.w(width)
        c.m(30, 80 - 15 * k); c.l(120, 80 - 15 * k); c.S()
    c.q()
    c.cm(2, 0, 0, 1, 0, 0)                          # a non-uniform CTM: the stroke is wider across than along
    c.w(2)
    c.m(15, 20); c.l(45, 20); c.l(45, 35); c.S()
    c.Q()
    data, w = _finish(doc)
    desc = ("Line caps (butt, round, square), joins (miter, round, bevel), a sharp angle under miter limits 1.5 "
            "(bevelled) and 10 (mitred), dash patterns with phases, round-capped zero-length dashes, a dashed curve "
            "and rectangle, widths 0, 0.5 and 3, and a stroke under a non-uniform CTM.")
    return _sample(doc, data, w, "vector-strokes", desc)


def _circle(c, cx, cy, r):
    k = 0.5523 * r
    c.m(cx + r, cy)
    c.c(cx + r, cy + k, cx + k, cy + r, cx, cy + r)
    c.c(cx - k, cy + r, cx - r, cy + k, cx - r, cy)
    c.c(cx - r, cy - k, cx - k, cy - r, cx, cy - r)
    c.c(cx + k, cy - r, cx + r, cy - k, cx + r, cy)
    c.h()


def clipping(seed):
    doc = _doc("vector-clipping", "1.5", seed)
    p = doc.page(W, H)
    _title(doc, p, "Clipping: W, W*, nested clips, a form's BBox")
    c = p.content
    c.q()
    c.re(20, 170, 80, 80); c.W(); c.n()             # a rectangle clip
    c.rg(0.2, 0.5, 0.85)
    _circle(c, 100, 210, 55); c.f()
    c.Q()
    c.q()
    _circle(c, 160, 210, 35); _circle(c, 200, 210, 35); c.W(evenodd=True); c.n()   # even-odd: the lens left out
    c.rg(0.85, 0.4, 0.2)
    c.re(110, 160, 140, 100); c.f()
    c.Q()
    c.q()
    c.re(270, 160, 110, 100); c.W(); c.n()          # nested: a rectangle, then a circle inside it
    c.q()
    _circle(c, 325, 165, 60); c.W(); c.n()
    c.rg(0.3, 0.7, 0.3)
    c.re(250, 140, 150, 140); c.f()
    c.Q()
    c.rg(0.95, 0.85, 0.2)
    c.re(270, 255, 110, 30); c.f()                  # under the rectangle alone
    c.Q()
    c.rg(0.6, 0.6, 0.6)
    c.re(10, 135, 380, 6); c.f()                    # no clip (restored by Q)
    with c.form("Fm1", [0, 0, 80, 60], matrix=(1, 0, 0, 1, 30, 30)) as fc:   # a form clipped by its BBox
        fc.rg(0.55, 0.25, 0.65)
        _circle(fc, 40, 30, 45); fc.f()
    c.q()
    c.re(160, 30, 90, 80); c.W(); c.n()
    img = Image(doc, 24, 16, 8, "DeviceRGB", 3, pattern(24, 16, 3, 255, 3), label="an image under a rectangle clip")
    rec = c.image("Im1", img, (140, 0, 0, 100, 140, 20))
    c.Q()
    c.q()
    c.BT()
    f = helvetica(doc)
    c.Tf(f, 40)
    c.Tr(7)                                         # text as a clip (render mode 7)
    c.Td(270, 40)
    run = c.show("Clip")
    c.ET()
    # The text's glyphs become the clip at ET: recorded as a clip naming the run (no path points).
    p.clips.append({"index": len(p.clips), "rule": "nonzero", "operators": ["Tr 7"], "verbs": [], "points": [],
                    "parent": c.vs["clip"], "bbox": p.display_box(run["quads"]), "textRun": run["order"]})
    c.vs["clip"] = len(p.clips) - 1
    c.rg(0.1, 0.4, 0.6)
    c.re(260, 30, 130, 60); c.f()
    c.Q()
    p.line("Clip")
    files = {"vector-clipping.p1.im1.png": img.png()}
    rec["pixels"]["file"] = "vector-clipping.p1.im1.png"
    data, w = _finish(doc)
    desc = ("A rectangle clip (W n), an even-odd clip of two overlapping circles (W* n), a circle clip nested in a "
            "rectangle clip and the rectangle alone after the inner Q, a path after Q with no clip, a form clipped by "
            "its BBox, an image under a clip, and text used as a clip (render mode 7, not drawn: its run is hidden; "
            "its clip names the run).")
    return _sample(doc, data, w, "vector-clipping", desc, files)


# -- colour --------------------------------------------------------------------------------------------------------------

def colour_spaces(seed):
    doc = _doc("vector-colour-spaces", "1.5", seed)
    p = doc.page(W, H)
    _title(doc, p, "Colour spaces, filled and stroked")
    c = p.content
    icc_rgb = ICCBased(3, srgb_icc(), "DeviceRGB")
    icc_gray = ICCBased(1, gray_icc(), "DeviceGray")
    lut = bytes(v for i in range(4) for v in ((i * 80) % 256, (200 - i * 50) % 256, (60 + i * 60) % 256))
    lut_cmyk = bytes([0, 0, 0, 0, 200, 30, 0, 0, 0, 180, 220, 0, 20, 20, 20, 120])
    fills = [
        ("g", None, None, [0.45]),
        ("rg", None, None, [0.85, 0.25, 0.1]),
        ("k", None, None, [0.1, 0.6, 0.9, 0.05]),
        ("cs", "CS1", icc_rgb, [0.2, 0.7, 0.4]),
        ("cs", "CS2", icc_gray, [0.7]),
        ("cs", "CS3", Indexed(RGB, 3, lut), [2]),
        ("cs", "CS4", Indexed(CMYK, 3, lut_cmyk), [1]),
        ("cs", "CS5", CalGray(), [0.3]),
        ("cs", "CS6", CalRGB(), [0.9, 0.8, 0.2]),
        ("cs", "CS7", Lab(), [62, 40, -30]),
        ("cs", "CS8", Separation("Spot Orange", CMYK, Exponential([0, 0, 0, 0], [0, 0.55, 0.95, 0], 1.0)), [0.8]),
        ("cs", "CS9", Separation("Spot Teal", RGB, Sampled([(255, 255, 255), (120, 200, 190), (0, 110, 120)], 3)),
         [0.7]),
        ("cs", "CS10", Separation("Spot Brown", RGB, Calculator("{ dup 0.6 mul exch dup 0.35 mul exch 0.15 mul }", 1,
                                                                3)), [0.9]),
        ("cs", "CS11", DeviceN(["Cyan", "Spot Gold"], CMYK, Calculator("{ dup 0.25 mul exch 0 }", 2, 4)), [0.6, 0.8]),
        ("cs", "CS12", DeviceN(["Mix A", "Mix B"], RGB, Stitching([Exponential([1, 1, 1], [0.9, 0.2, 0.2]),
                                                                  Exponential([0.9, 0.2, 0.2], [0.3, 0, 0])],
                                                                 [0.5], [0, 1, 0, 1])), [0.75, 0.2]),
    ]
    c.RG(0, 0, 0)
    c.w(1.5)
    for k, (op, name, space, comps) in enumerate(fills):
        x = 20 + (k % 5) * 75
        y = 210 - (k // 5) * 80
        if op == "g":
            c.g(*comps)
        elif op == "rg":
            c.rg(*comps)
        elif op == "k":
            c.k(*comps)
        else:
            c.cs(name, space)
            c.sc(*comps, op="scn" if k % 2 else "sc")
        c.re(x, y, 60, 45)
        c.B()
    c.CS("CS7", Lab())
    c.SC(40, -50, 40, op="SC")
    c.w(5)
    c.m(20, 18); c.l(180, 18); c.S()
    c.CS("CS8", fills[10][2])
    c.SC(0.5, op="SCN")
    c.m(220, 18); c.l(380, 18); c.S()
    data, w = _finish(doc)
    desc = ("Rectangles filled (and stroked in black) in DeviceGray, DeviceRGB, DeviceCMYK, ICCBased (RGB and grey "
            "profiles), Indexed over RGB and CMYK, CalGray, CalRGB, Lab, Separation through a type 2, a type 0 "
            "(sampled) and a type 4 (calculator) tint transform, DeviceN through a type 4 and a type 3 (stitching) "
            "transform; strokes in Lab and Separation. Each path's colour is given as written and as RGB.")
    return _sample(doc, data, w, "vector-colour-spaces", desc)


def _affine_colour(x, y, x0, y0, w, h):
    u, v = (x - x0) / w, (y - y0) / h
    return [0.15 + 0.7 * u, 0.2 + 0.6 * v, 0.9 - 0.5 * u - 0.3 * v]


def _mesh_bytes(items):
    out = bytearray()
    for kind, value in items:
        if kind == "flag":
            out.append(value)
        elif kind == "xy":
            out += round(value[0]).to_bytes(2, "big") + round(value[1]).to_bytes(2, "big")
        elif kind == "rgb":
            out += bytes(round(v * 255) for v in value)
    return bytes(out)


def _mesh_entries(bits_flag=True):
    d = {"BitsPerCoordinate": 16, "BitsPerComponent": 8, "Decode": [0, 65535, 0, 65535, 0, 1, 0, 1, 0, 1]}
    if bits_flag:
        d["BitsPerFlag"] = 8
    return d


def _q(v):
    """A colour as the 8-bit mesh data holds it."""
    return [round(x * 255) / 255 for x in v]


def shadings(seed):
    doc = _doc("vector-shadings", "1.6", seed)
    p = doc.page(W, H)
    _title(doc, p, "Shadings: axial, radial, function-based, meshes of every type")
    c = p.content
    # Axial (type 2) with a stitching function, extended both ways, clipped to a rectangle.
    fn = Stitching([Exponential([0.9, 0.1, 0.1], [0.95, 0.85, 0.1]), Exponential([0.95, 0.85, 0.1], [0.1, 0.3, 0.9])],
                   [0.5], [0, 1, 0, 1])
    x0, x1, y = 30, 110, 225
    axial = Shading(doc, 2, RGB, {"Coords": [x0, y, x1, y], "Function": fn.pdf(doc), "Extend": [True, True]},
                    probes=[(x0 + t * (x1 - x0), y, fn.value([t])) for t in (0.25, 0.5, 0.75)])
    c.q(); c.re(20, 200, 100, 50); c.W(); c.n(); c.sh("Sh1", axial); c.Q()
    # Radial (type 3), concentric, extended.
    cx, cy, R = 185, 225, 30
    rfn = Exponential([1, 1, 1], [0.2, 0.4, 0.8], 1.0)
    radial = Shading(doc, 3, RGB, {"Coords": [cx, cy, 0, cx, cy, R], "Function": rfn.pdf(doc), "Extend": [True, True]},
                     probes=[(cx + t * R, cy, rfn.value([t])) for t in (0.0, 0.5, 0.9)])
    c.q(); c.re(140, 200, 90, 50); c.W(); c.n(); c.sh("Sh2", radial); c.Q()
    # Function-based (type 1): (u, v) -> (u, v, 0.5) over its domain mapped onto a rectangle.
    ffn = Calculator("{ 0.5 }", 2, 3)
    fx, fy, fw, fh = 250, 200, 130, 50
    func = Shading(doc, 1, RGB, {"Domain": [0, 1, 0, 1], "Matrix": [fw, 0, 0, fh, fx, fy], "Function": ffn.pdf(doc)},
                   probes=[(fx + u * fw, fy + v * fh, [u, v, 0.5]) for u, v in ((0.25, 0.25), (0.5, 0.5), (0.75, 0.6))])
    c.sh("Sh3", func)
    # Free-form triangle mesh (type 4): two triangles, then one more sharing an edge (flag 1).
    mx, my, mw, mh = 20, 110, 110, 70
    tri = [(mx, my), (mx + mw, my), (mx, my + mh), (mx + mw, my + mh)]
    col = [_q(_affine_colour(x, yy, mx, my, mw, mh)) for x, yy in tri]
    data4 = _mesh_bytes([("flag", 0), ("xy", tri[0]), ("rgb", col[0]), ("flag", 0), ("xy", tri[1]), ("rgb", col[1]),
                         ("flag", 0), ("xy", tri[2]), ("rgb", col[2]), ("flag", 1), ("xy", tri[3]), ("rgb", col[3])])
    probes4 = []
    for a, b, cc in ((0, 1, 2), (1, 2, 3)):
        px = (tri[a][0] + tri[b][0] + tri[cc][0]) / 3
        py = (tri[a][1] + tri[b][1] + tri[cc][1]) / 3
        probes4.append((px, py, [(col[a][k] + col[b][k] + col[cc][k]) / 3 for k in range(3)]))
    mesh4 = Shading(doc, 4, RGB, _mesh_entries(), probes=probes4, stream=data4)
    c.sh("Sh4", mesh4)
    # Lattice (type 5): 3 by 2 vertices, colours affine in position (the same on either diagonal).
    lx, ly, lw, lh = 150, 110, 100, 70
    verts = [(lx + i * lw / 2, ly + j * lh) for j in range(2) for i in range(3)]
    lcol = [_q(_affine_colour(x, yy, lx, ly, lw, lh)) for x, yy in verts]
    data5 = _mesh_bytes([e for v, cl in zip(verts, lcol) for e in (("xy", v), ("rgb", cl))])
    lattice = Shading(doc, 5, RGB, {**_mesh_entries(False), "VerticesPerRow": 3}, stream=data5,
                      probes=[(lx + 0.25 * lw, ly + 0.5 * lh, _affine_colour(lx + 0.25 * lw, ly + 0.5 * lh, lx, ly, lw, lh)),
                              (lx + 0.7 * lw, ly + 0.4 * lh, _affine_colour(lx + 0.7 * lw, ly + 0.4 * lh, lx, ly, lw, lh))])
    c.sh("Sh5", lattice)
    # Coons (type 6) and tensor (type 7) patches on rectangles, straight edges at thirds (so linear in u and v).
    for k, (stype, px0) in enumerate(((6, 270), (7, 20))):
        py0 = 110 if stype == 6 else 30
        pw, ph = (110, 70) if stype == 6 else (360, 60)

        def P(i, j):
            return (px0 + i * pw / 3, py0 + j * ph / 3)

        order = [P(0, 0), P(0, 1), P(0, 2), P(0, 3), P(1, 3), P(2, 3), P(3, 3), P(3, 2), P(3, 1), P(3, 0), P(2, 0),
                 P(1, 0)]
        if stype == 7:
            order += [P(1, 1), P(1, 2), P(2, 2), P(2, 1)]
        corners = [P(0, 0), P(0, 3), P(3, 3), P(3, 0)]
        ccol = [_q(_affine_colour(x, yy, px0, py0, pw, ph)) for x, yy in corners]
        items = [("flag", 0)] + [("xy", pt) for pt in order] + [("rgb", cl) for cl in ccol]
        probes = [(px0 + u * pw, py0 + v * ph, _affine_colour(px0 + u * pw, py0 + v * ph, px0, py0, pw, ph))
                  for u, v in ((0.5, 0.5), (0.2, 0.7), (0.8, 0.3))]
        patch = Shading(doc, stype, RGB, _mesh_entries(), probes=probes, stream=_mesh_bytes(items))
        c.sh(f"Sh{6 + k}", patch)
    data, w = _finish(doc)
    desc = ("Shadings painted by sh: axial (a type 3 stitching function, extended, clipped), radial (concentric, "
            "extended, clipped), function-based (a type 4 function of two inputs), a free-form triangle mesh (flags 0 "
            "and 1), a lattice, a Coons patch and a tensor patch. Each shading lists points whose colour is exact "
            "(`samples`): along the axis, from the centre, the domain mapped, mesh colours affine in position so any "
            "triangulation gives them.")
    return _sample(doc, data, w, "vector-shadings", desc)


def patterns(seed):
    doc = _doc("vector-patterns", "1.6", seed)
    p = doc.page(W, H)
    _title(doc, p, "Tiling patterns (coloured, uncoloured) and shading patterns")
    c = p.content
    cell = b"1 0.2 0.2 rg 0 0 10 10 re f 0.2 0.3 0.9 rg 10 10 10 10 re f 0.95 0.8 0.1 rg 15 5 m 20 0 l 20 5 l h f\n"
    coloured = TilingPattern(doc, 1, [0, 0, 20, 20], 20, 20, cell)
    stripes = TilingPattern(doc, 2, [0, 0, 12, 12], 12, 12, b"0 0 12 4 re f 0 6 m 12 12 l 12 10 l 2 0 l h f\n")
    turned = TilingPattern(doc, 1, [0, 0, 16, 16], 16, 16, b"0.1 0.6 0.4 rg 0 0 8 8 re f 8 8 8 8 re f\n",
                           matrix=[0.7071, 0.7071, -0.7071, 0.7071, 0, 0])
    sfn = Exponential([0.95, 0.95, 0.3], [0.6, 0.1, 0.5])
    sh = Shading(doc, 2, RGB, {"Coords": [0, 0, 120, 0], "Function": sfn.pdf(doc), "Extend": [True, True]})
    spat = ShadingPattern(doc, sh, matrix=[1, 0, 0, 1, 260, 0])
    patterns_json = {
        "P1": {"patternType": 1, "paintType": 1, "bbox": [0, 0, 20, 20], "xStep": 20, "yStep": 20, "matrix": None,
               "object": [coloured.ref.num, 0]},
        "P2": {"patternType": 1, "paintType": 2, "bbox": [0, 0, 12, 12], "xStep": 12, "yStep": 12, "matrix": None,
               "object": [stripes.ref.num, 0]},
        "P3": {"patternType": 1, "paintType": 1, "bbox": [0, 0, 16, 16], "xStep": 16, "yStep": 16,
               "matrix": [0.7071, 0.7071, -0.7071, 0.7071, 0, 0], "object": [turned.ref.num, 0]},
        "P4": {"patternType": 2, "shadingType": 2, "matrix": [1, 0, 0, 1, 260, 0], "object": [spat.ref.num, 0]},
    }
    c.cs(None, PatternSpace())
    c.scn_pattern("P1", coloured)
    c.re(20, 170, 110, 90); c.f()
    c.cs("CSp", PatternSpace(RGB))
    c.scn_pattern("P2", stripes, comps=(0.8, 0.15, 0.15))
    c.re(150, 170, 100, 90); c.f()
    c.scn_pattern("P2", stripes, comps=(0.1, 0.45, 0.8))
    _circle(c, 315, 215, 45); c.f()
    c.cs(None, PatternSpace())
    c.scn_pattern("P3", turned)
    c.re(20, 40, 110, 100); c.f()
    c.scn_pattern("P4", spat)
    _circle(c, 315, 90, 50); c.f()
    c.CS(None, PatternSpace())
    c.scn_pattern("P1", coloured, stroke=True)
    c.w(14)
    c.m(150, 60); c.l(240, 130); c.S()
    data, w = _finish(doc)
    desc = ("Tiling patterns: coloured (PaintType 1), uncoloured (PaintType 2) under [/Pattern /DeviceRGB] painted in "
            "two colours, one turned by its /Matrix; a shading pattern (an axial shading placed by the pattern's "
            "matrix) filling a circle; a thick line stroked with a pattern. `patterns` lists each pattern's cell.")
    exp_extra = {}
    sample = _sample(doc, data, w, "vector-patterns", desc, extra=exp_extra)
    sample.expected["pages"][0]["patterns"] = patterns_json
    return sample


def transparency(seed):
    doc = _doc("vector-transparency", "1.7", seed)
    p = doc.page(W, H)
    _title(doc, p, "Transparency: ca, CA, blend modes, a soft mask, groups")
    c = p.content
    for k in range(8):                                     # the backdrop: coloured bars
        c.rg(*[(0.9, 0.2, 0.2), (0.2, 0.7, 0.3), (0.2, 0.3, 0.9), (0.95, 0.8, 0.1)][k % 4])
        c.re(10 + 48 * k, 150, 46, 80); c.f()
    half = ExtGState(doc, {"Type": N("ExtGState"), "ca": 0.5, "CA": 0.5})
    c.q()
    c.gs("GS1", half)
    for k, col in enumerate([(1, 0, 0), (0, 0.8, 0), (0, 0, 1)]):
        c.rg(*col)
        _circle(c, 60 + 30 * k, 250 - 12 * (k % 2), 30); c.f()
    c.RG(0, 0, 0)
    c.w(8)
    c.m(170, 270); c.l(390, 270); c.S()
    c.Q()
    modes = ["Normal", "Multiply", "Screen", "Overlay", "Darken", "Lighten", "ColorDodge", "ColorBurn", "HardLight",
             "SoftLight", "Difference", "Exclusion", "Hue", "Saturation", "Color", "Luminosity"]
    for k, mode in enumerate(modes):
        gsm = ExtGState(doc, {"Type": N("ExtGState"), "BM": N(mode)})
        c.q()
        c.gs(f"BM{k}", gsm)
        c.rg(0.5, 0.25, 0.75)
        c.re(14 + 24 * k, 160 + (k % 2) * 30, 20, 30); c.f()
        c.Q()
    # A luminosity soft mask: a grey axial shading in a group, white keeping the paint, black removing it.
    mfn = Exponential([0], [1])
    with c.form("SM1", [0, 0, 400, 300], extra={"Group": {"Type": N("Group"), "S": N("Transparency"),
                                                             "CS": N("DeviceGray")}}, draw=False) as mask:
        mask_shading = Shading(doc, 2, GRAY, {"Coords": [20, 0, 180, 0], "Function": mfn.pdf(doc),
                                               "Extend": [True, True]})
        mask.sh("Sh1", mask_shading)
    smask = ExtGState(doc, {"Type": N("ExtGState"), "SMask": {"Type": N("Mask"), "S": N("Luminosity"),
                                                            "G": mask.ref, "BC": [0]}},
                      smask_describe={"type": "Luminosity", "group": [mask.ref.num, 0], "backdrop": [0],
                                      "shading": "axial grey from x 20 (black) to x 180 (white) in user space"})
    c.q()
    c.gs("GS2", smask)
    c.rg(0.1, 0.3, 0.8)
    c.re(20, 20, 160, 110); c.f()
    c.Q()
    groups = []
    for k, (iso, ko) in enumerate(((True, True), (False, False))):
        gi = len(groups)
        groups.append({"index": gi, "isolated": iso, "knockout": ko, "form": f"Fm{k + 1}"})
        with c.form(f"Fm{k + 1}", [0, 0, 400, 300], extra={"Group": {"Type": N("Group"), "S": N("Transparency"),
                                                                     "I": iso, "K": ko, "CS": N("DeviceRGB")}}) as g:
            g.vs["group"] = gi
            g.gs("GS1", half)
            g.rg(0.9, 0.3, 0.1)
            _circle(g, 240 + 95 * k, 85, 35); g.f()
            g.rg(0.1, 0.5, 0.9)
            _circle(g, 265 + 95 * k, 60, 35); g.f()
        groups[gi]["object"] = [g.ref.num, 0]
        # The circles' overlap: orange then blue at ca 0.5 over the white page. A knockout group keeps the blue alone
        # (it knocks out the orange beneath it within the group); otherwise blue is composed over the orange.
        ox, oy = 252 + 95 * k, 72
        rgb = [140, 191, 242] if ko else [134, 147, 185]
        groups[gi]["probes"] = [{"point": [ox, H - oy], "rgb": rgb, "tolerance": 2,
                                 "note": "the overlap of the two circles"}]
    data, w = _finish(doc)
    desc = ("Constant alpha (ExtGState ca 0.5 for fills, CA 0.5 for a stroke) over a backdrop; the 16 blend modes over "
            "the backdrop; a luminosity soft mask (a grey axial group) over a fill; an isolated knockout group and a "
            "non-isolated non-knockout group, each holding two overlapping half-transparent circles (`groups`).")
    sample = _sample(doc, data, w, "vector-transparency", desc)
    sample.expected["pages"][0]["groups"] = groups
    return sample


# -- text and images -------------------------------------------------------------------------------------------------------

TEXT = "Type 1 outlines: Quarterly report, 2024."


def fonts(seed):
    doc = _doc("vector-fonts", "1.6", seed)
    p = doc.page(W, H)
    _title(doc, p, "Text through embedded fonts of every kind")
    c = p.content
    t1 = doc.font(Type1Embedded, DEJAVU, TEXT)
    t3 = doc.font(Type3, TYPE3_PROCS)
    tt = doc.font(TrueTypeSimple, DEJAVU)
    cff = doc.font(CffSimple, DEJAVU_CFF)
    cid = doc.font(CffType0, DEJAVU_CFF)
    lines = [(t1, TEXT, "Type 1 (FontFile, eexec)"), (t3, "BEAD CAB ✓", "Type 3"),
             (tt, "TrueType outlines: delivery on time.", "TrueType"),
             (cff, "CFF outlines: the budget is approved.", "Type1C"),
             (cid, "CFF composite: new office in May.", "CIDFontType0")]
    c.BT()
    y = 245
    for k, (font, text, _) in enumerate(lines):
        c.Tf(font, 16)
        c.Tm(1, 0, 0, 1, 20, y - 30 * k)
        if k == 1:
            c.rg(0.1, 0.45, 0.2)
        c.show(text)
        c.rg(0, 0, 0)
        p.line(text)
    c.Tf(tt, 18)
    c.Tm(1, 0, 0, 1, 20, 95)
    c.Tr(1)
    c.RG(0.7, 0.1, 0.1)
    c.w(0.8)
    c.show("Outlined text")
    p.line("Outlined text")
    c.Tm(1, 0, 0, 1, 190, 95)
    c.Tr(2)
    c.rg(0.95, 0.8, 0.2)
    c.show("Filled and outlined")
    p.line("Filled and outlined")
    c.Tr(0)
    c.rg(0.2, 0.3, 0.8)
    c.Tm(0.9659, 0.2588, -0.2588, 0.9659, 230, 15)
    c.show("Turned 15 degrees")
    p.line("Turned 15 degrees")
    c.ET()
    data, w = _finish(doc)
    programs = {"type1": t1.describe_program(),
                "type1c": {"kind": "type1c", "object": [cff.program_ref.num, 0], "fontName": cff.base_font},
                "opentypeCff": {"kind": "opentype-cff", "object": [cid.program_ref.num, 0], "fontName": cid.base_font,
                                "unitsPerEm": cid.upem,
                                "glyphs": cff_outlines(doc.pdf.get(cid.program_ref).decoded,
                                                       sorted({cid.order[g] for g in cid.cid_gid.values()}))}}
    desc = ("Text in an embedded Type 1 font made from DejaVu Sans outlines (FontFile with Length1-3, eexec, built-in "
            "StandardEncoding), Type 3 glyphs in a colour, embedded TrueType, a bare CFF (FontFile3 /Type1C) and an "
            "OpenType CFF under CIDFontType0 (FontFile3 /OpenType); outlined (render mode 1), filled and outlined (2), "
            "and turned text. `fontPrograms` gives the Type 1 glyphs as fontTools reads the program (1000 units), and "
            "the CFF composite's glyphs (font units).")
    return _sample(doc, data, w, "vector-fonts", desc, extra={"fontPrograms": programs})


def images(seed):
    doc = _doc("vector-images", "1.5", seed)
    p = doc.page(W, H)
    _title(doc, p, "Images placed by turned and skewed matrices")
    c = p.content
    files = {}
    a = math.radians(30)
    specs = [
        ("turned 30 degrees", (80 * math.cos(a), 80 * math.sin(a), -50 * math.sin(a), 50 * math.cos(a), 60, 150), None),
        ("skewed", (90, 0, 35, 55, 160, 170), None),
        ("mirrored and stretched", (-120, 0, 0, 40, 380, 200), None),
        ("turned 150 degrees and skewed", (-50, 30, 25, -45, 345, 140), None),
        ("under ca 0.5", (100, 0, 0, 60, 30, 20), "half"),
        ("with a soft mask", (100, 0, 0, 60, 150, 20), "smask"),
        ("a stencil painted orange", (100, 0, 0, 60, 270, 20), "stencil"),
    ]
    half = ExtGState(doc, {"Type": N("ExtGState"), "ca": 0.5})
    for k, (label, m, kind) in enumerate(specs):
        if kind == "stencil":
            img = Image(doc, 24, 16, 1, None, 1, pattern(24, 16, 1, 1, k), image_mask=True, label=label)
        elif kind == "smask":
            soft = Image(doc, 24, 16, 8, "DeviceGray", 1, [[(x * 255) // 23 for x in range(24)] for _ in range(16)],
                         label="soft mask: a ramp")
            img = Image(doc, 24, 16, 8, "DeviceRGB", 3, pattern(24, 16, 3, 255, k), smask=soft, label=label)
        else:
            img = Image(doc, 24, 16, 8, "DeviceRGB", 3, pattern(24, 16, 3, 255, k), label=label)
        c.q()
        if kind == "half":
            c.gs("GS1", half)
        if kind == "stencil":
            c.rg(0.95, 0.5, 0.1)
        rec = c.image(f"Im{k + 1}", img, m)
        c.Q()
        base = f"vector-images.p1.im{k + 1}"
        rec["pixels"]["file"] = base + ".png"
        files[base + ".png"] = img.png()
        if img.smask is not None:
            rec["softMask"]["pixels"]["file"] = base + ".smask.png"
            files[base + ".smask.png"] = img.smask.png()
    data, w = _finish(doc)
    desc = ("Images placed by matrices turned by 30 degrees, skewed, mirrored and stretched, turned and skewed; one "
            "under ca 0.5, one with a soft mask (a ramp), a stencil painted in the fill colour. Each image's `ctm` "
            "places its unit square; its expected pixels are beside it.")
    return _sample(doc, data, w, "vector-images", desc, files)


def turned_page(seed):
    doc = _doc("vector-turned-pages", "1.4", seed)
    for rotate, crop in ((90, None), (0, [40, 30, 360, 270])):
        p = doc.page(W, H, rotate=rotate, crop=crop)
        _title(doc, p, f"Rotate {rotate}" + (", cropped" if crop else ""), y=250)
        c = p.content
        c.rg(0.2, 0.5, 0.8)
        c.re(60, 60, 120, 80); c.f()
        c.q()
        c.re(200, 60, 120, 120); c.W(); c.n()
        c.rg(0.9, 0.4, 0.1)
        _circle(c, 260, 120, 80); c.f()
        c.Q()
        c.RG(0.1, 0.1, 0.1)
        c.w(3)
        c.d([10, 5], 0)
        c.m(50, 200); c.c(120, 260, 200, 160, 330, 220); c.S()
    data, w = _finish(doc)
    desc = ("Paths, a clip and a dashed curve on a page turned by /Rotate 90 and on a cropped page (CropBox [40 30 "
            "360 270]): every point is given in the page as shown.")
    return _sample(doc, data, w, "vector-turned-pages", desc)


def build(seed):
    return [paths(seed), fill_rules(seed), strokes(seed), clipping(seed), colour_spaces(seed), shadings(seed),
            patterns(seed), transparency(seed), fonts(seed), images(seed), turned_page(seed)]
