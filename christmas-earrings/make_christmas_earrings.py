#!/usr/bin/env python3
"""
Stocking Stuffer Earrings: ten multicolor Christmas earrings for 3D printing.

Every design is a stack of color bands:

    band 1  0.0-1.2 mm   base color (the full silhouette)
    band 2  1.2-1.8 mm
    band 3  1.8-2.4 mm
    band 4  2.4-3.0 mm

Each band is one color, so the top of every spot shows the color of the
highest band there. Each design is drawn as the region that should *show*
each color; the script grows the lower bands underneath the upper ones so
nothing floats, then cuts the 1.8 mm hook hole through everything.

Outputs, one folder per design:

    NN-name/NN-name.3mf   left + mirrored right earring on one plate; every
                          part already assigned to a tool (T1-T4), ready for
                          Snapmaker Orca / OrcaSlicer / Bambu Studio
    NN-name/preview.svg   the layered artwork

Requires: pip install numpy shapely trimesh mapbox_earcut
Run:      python3 make_christmas_earrings.py
"""
import io
import json
import math
import os
import sys
import zipfile

import numpy as np
import trimesh
from shapely import affinity
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------- settings
BANDS = [(0.0, 1.2), (1.2, 1.8), (1.8, 2.4), (2.4, 3.0)]
HOLE_R = 0.9            # 1.8 mm hook hole
TAB_R = 2.3             # round tab around the hole where a design needs one
PLA_DENSITY = 1.24      # g/cm^3
PLATE_CENTER = (135.0, 135.0)   # Snapmaker U1 plate is 270 x 270 mm

PALETTE = {
    'santa_red':   ('Santa red',         '#C21F32'),
    'snow_white':  ('Snow white',        '#F6F4EE'),
    'holly_green': ('Holly green',       '#1F7A45'),
    'pine_green':  ('Pine green',        '#154B30'),
    'gold':        ('Gold',              '#D9A834'),
    'gingerbread': ('Gingerbread brown', '#A0602F'),
    'chocolate':   ('Chocolate brown',   '#4B2E20'),
    'cream':       ('Antler cream',      '#E9D8B4'),
    'charcoal':    ('Charcoal',          '#2E3134'),
    'silver':      ('Silver',            '#BFC5CB'),
    'carrot':      ('Carrot orange',     '#EC7A1E'),
    'ice_blue':    ('Ice blue',          '#9ED5EA'),
}


# ----------------------------------------------------------------- 2D helpers
def C(x, y, r, q=24):
    return Point(x, y).buffer(r, quad_segs=q)


def E(x, y, rx, ry, rot=0.0, q=24):
    g = affinity.scale(Point(0, 0).buffer(1.0, quad_segs=q), rx, ry, origin=(0, 0))
    g = affinity.rotate(g, rot, origin=(0, 0))
    return affinity.translate(g, x, y)


def S(pts, w, cap='round'):
    """A stroke of width w along a polyline."""
    return LineString(pts).buffer(w / 2.0, quad_segs=12, cap_style=cap, join_style='round')


def P(pts):
    return Polygon(pts).buffer(0)


def U(*gs):
    flat = []
    for g in gs:
        if isinstance(g, (list, tuple)):
            flat.extend(g)
        else:
            flat.append(g)
    return unary_union([g for g in flat if g is not None and not g.is_empty])


def rrect(x0, y0, x1, y1, r):
    if r <= 0:
        return box(x0, y0, x1, y1)
    return box(x0 + r, y0 + r, x1 - r, y1 - r).buffer(r, quad_segs=12)


def bez(p0, p1, p2, p3, n=28):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = (np.array(p, float) for p in (p0, p1, p2, p3))
    pts = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
    return [tuple(p) for p in pts]


def arc(cx, cy, r, a0, a1, n=24):
    a = np.radians(np.linspace(a0, a1, n))
    return list(zip(cx + r * np.cos(a), cy + r * np.sin(a)))


def star(cx, cy, R, r, n=5, rot=90.0):
    pts = []
    for i in range(2 * n):
        a = math.radians(rot + i * 180.0 / n)
        rr = R if i % 2 == 0 else r
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return P(pts)


def hexagon(cx, cy, r, rot=0.0):
    return P([(cx + r * math.cos(math.radians(rot + 60 * i)),
               cy + r * math.sin(math.radians(rot + 60 * i))) for i in range(6)])


def smooth(g, r_out=0.0, r_in=0.0):
    """Round convex corners by r_out and fill concave corners by r_in."""
    if r_in > 0:
        g = g.buffer(r_in, quad_segs=12).buffer(-r_in, quad_segs=12)
    if r_out > 0:
        g = g.buffer(-r_out, quad_segs=12).buffer(r_out, quad_segs=12)
    return g


def mirror_x(g, cx):
    return affinity.scale(g, -1, 1, origin=(cx, 0))


def holly_leaf(cx, cy, L, W, rot, bites=3):
    rx, ry = L / 2.0, W / 2.0
    leaf = E(0, 0, rx, ry)
    b = 0.42 * L / bites
    cut = []
    for i in range(bites):
        x = -rx + L * (i + 0.5) / bites
        y = ry * math.sqrt(max(0.0, 1 - (x / rx) ** 2)) + 0.5 * b
        cut += [C(x, y, b, 12), C(x, -y, b, 12)]
    leaf = leaf.difference(U(cut))
    leaf = affinity.rotate(leaf, rot, origin=(0, 0))
    return affinity.translate(leaf, cx, cy)


def zigzag(p0, p1, amp, n):
    """Zigzag between p0 and p1 with n segments and the given amplitude."""
    p0, p1 = np.array(p0, float), np.array(p1, float)
    d = p1 - p0
    nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    pts = []
    for i in range(n + 1):
        t = i / n
        s = amp if i % 2 else -amp
        pts.append(tuple(p0 + d * t + nrm * s))
    return pts


def top_y(region, x):
    line = LineString([(x, -1000), (x, 1000)]).intersection(region)
    return line.bounds[3]


def polys(g):
    if g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    return [p for p in getattr(g, 'geoms', []) if isinstance(p, Polygon)]


def clean(g, min_area=0.12, min_hole=0.6):
    """Drop crumbs, fill pinholes too small to print, and remove pinch points."""
    g = g.buffer(0.01, quad_segs=4).buffer(-0.01, quad_segs=4)
    keep = []
    for p in polys(g):
        if p.area < min_area:
            continue
        holes = [r for r in p.interiors if Polygon(r).area >= min_hole]
        keep.append(Polygon(p.exterior, holes))
    return unary_union(keep).simplify(0.008) if keep else Polygon()


# ----------------------------------------------------------------- designs
# Each design returns a dict:
#   layers: list of (palette key, part label, region that should show it), bottom band first
#   hole:   (x, y) center of the hook hole
#   right:  optional {band index: palette key} color override for the right earring
DESIGNS = []


def design(slug, name, blurb, tip):
    def wrap(fn):
        DESIGNS.append(dict(slug=slug, name=name, blurb=blurb, tip=tip, fn=fn))
        return fn
    return wrap


@design('ugly-sweater', 'Ugly Christmas Sweater',
        'A tiny Fair Isle sweater with ribbed cuffs, a hem and a crew-neck collar. Pixel trees '
        'with gold stars run across the chest band.',
        'The rib lines in the hem are 0.8 mm windows. Use 0.12 to 0.16 mm layers so they stay open.')
def ugly_sweater():
    cx = 14.0
    neck = bez((17.4, 24.2), (16.6, 22.2), (11.4, 22.2), (10.6, 24.2), 20)
    right = [(14, 0), (22, 0), (22, 13.2), (24.3, 4.6), (28.3, 6.0), (21.4, 22.6), (17.4, 24.2)]
    left = [(2 * cx - x, y) for x, y in reversed(right)]
    body = P(right + neck[1:-1] + left[1:])
    body = smooth(body, r_out=0.55, r_in=0.35)
    bail = S([(cx, 22.0), (cx, 25.0)], 2 * 2.4)

    collar = S(neck, 1.8).intersection(body)
    hem = rrect(6.0, 0.0, 22.0, 2.6, 0.0).intersection(body)
    hem = hem.difference(U([box(x, 0.55, x + 0.8, 2.05) for x in np.arange(7.4, 21.0, 2.0)]))

    def cuff(a, b):
        a, b = np.array(a), np.array(b)
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0]])
        q = [tuple(a), tuple(b), tuple(b + n * 2.4), tuple(a + n * 2.4)]
        return P(q).intersection(body)

    cuffs = U(cuff((24.3, 4.6), (28.3, 6.0)), cuff((-0.3, 6.0), (3.7, 4.6)))
    chest = box(-2, 9.4, 30, 15.2).intersection(body)
    flakes = []
    for x in (8.6, 14.0, 19.4):
        for a in (90, 30, 150):
            dx, dy = 1.45 * math.cos(math.radians(a)), 1.45 * math.sin(math.radians(a))
            flakes.append(S([(x - dx, 18.7 - dy), (x + dx, 18.7 + dy)], 0.78))
    zig = S(zigzag((6.9, 5.8), (21.1, 5.8), 0.75, 12), 0.85, cap='flat').intersection(body)
    white = U(collar, hem, cuffs, chest, flakes, zig)

    trees, stars = [], []
    for x in (9.0, 14.0, 19.0):
        trees += [smooth(P([(x - 2.2, 10.6), (x + 2.2, 10.6), (x, 13.2)]), r_out=0.2),
                  smooth(P([(x - 1.6, 12.0), (x + 1.6, 12.0), (x, 14.2)]), r_out=0.2),
                  box(x - 0.5, 9.85, x + 0.5, 10.8)]
        stars.append(star(x, 14.35, 1.25, 0.62))
    for sx in (24.1, 2 * cx - 24.1):
        trees.append(affinity.rotate(box(sx - 1.0, 11.3, sx + 1.0, 13.3), 45, origin=(sx, 12.3)))
    green = U(trees)
    return dict(
        layers=[('santa_red', 'sweater', U(body, bail)),
                ('snow_white', 'ribbing and pattern', white),
                ('holly_green', 'trees', green),
                ('gold', 'stars', U(stars))],
        hole=(cx, 25.0))


@design('gingerbread', 'Gingerbread Cookie',
        'A chunky gingerbread person with white icing at the wrists and ankles, a red bow tie '
        'and three gumdrop buttons set in dabs of icing.',
        'Matte light-brown PLA looks most like a baked cookie. Use a glossy red for the gumdrops.')
def gingerbread():
    cx = 13.0
    head = C(cx, 26.5, 6.0)
    torso = S([(cx, 20.0), (cx, 12.0)], 10.4)
    arms = S([(2.6, 17.2), (cx, 19.2), (23.4, 17.2)], 5.4)
    legs = S([(6.8, 2.9), (cx, 13.0), (19.2, 2.9)], 6.0)
    body = smooth(U(head, torso, arms, legs), r_in=1.3)
    tab = C(cx, 33.4, TAB_R)
    base = smooth(U(body, tab), r_in=0.8)
    inner = body.buffer(-0.55)

    icing = [C(10.8, 27.6, 0.85), C(15.2, 27.6, 0.85),
             S(arc(cx, 26.6, 2.7, 218, 322, 16), 0.85)]
    for x in (5.0, 2 * cx - 5.0):
        icing.append(S(zigzag((x, 14.6), (x - 0.25 if x < cx else x + 0.25, 20.0), 0.6, 5), 0.8))
    for foot, sgn in (((6.8, 2.9), 1), ((19.2, 2.9), -1)):
        d = np.array([cx - foot[0], 13.0 - foot[1]])
        d /= np.linalg.norm(d)
        p = np.array(foot) + d * 2.9
        n = np.array([-d[1], d[0]])
        icing.append(S(zigzag(tuple(p - n * 3.4), tuple(p + n * 3.4), 0.55, 6), 0.8))
    for y in (17.4, 14.4, 11.4):
        icing.append(C(cx, y, 1.6))
    white = U(icing).intersection(inner)

    bow = U(P([(cx, 20.9), (10.3, 22.3), (10.3, 19.5)]),
            P([(cx, 20.9), (15.7, 22.3), (15.7, 19.5)]))
    bow = U(smooth(bow, r_out=0.35), C(cx, 20.9, 0.85))
    red = U(bow, [C(cx, y, 1.15) for y in (17.4, 14.4, 11.4)])
    return dict(
        layers=[('gingerbread', 'cookie', base),
                ('snow_white', 'icing', white),
                ('santa_red', 'bow tie and gumdrops', red)],
        hole=(cx, 33.4))


@design('candy-cane', 'Candy Cane',
        'A peppermint cane with diagonal red stripes and a sprig of holly where the hook begins. '
        'The pair mirrors so both hooks curl outward.',
        'Print the white in a bright, opaque PLA so the red stripes stay crisp at the edges.')
def candy_cane():
    acx, acy, ar = 8.6, 22.0, 5.4
    center = [(14.0, 3.4), (14.0, acy)] + arc(acx, acy, ar, 0, 180, 40)[1:] + [(acx - ar, 19.4)]
    cane = S(center, 5.6)
    stripes = []
    for i in range(-12, 14):
        s = box(-40, i * 3.7, 40, i * 3.7 + 1.85)
        stripes.append(affinity.rotate(s, 38, origin=(9, 15)))
    red = U(stripes).intersection(cane)
    hx = cane.centroid.x
    th = math.degrees(math.acos(max(-1, min(1, (hx - acx) / ar))))
    hole = (acx + ar * math.cos(math.radians(th)), acy + ar * math.sin(math.radians(th)))
    hc = (14.4, 19.6)
    leaves = U(holly_leaf(hc[0] + 2.9 * math.cos(math.radians(28)), hc[1] + 2.9 * math.sin(math.radians(28)), 6.0, 2.9, 28),
               holly_leaf(hc[0] + 2.9 * math.cos(math.radians(152)), hc[1] + 2.9 * math.sin(math.radians(152)), 6.0, 2.9, 152))
    berries = U(C(hc[0] - 0.9, hc[1] - 0.35, 0.95), C(hc[0] + 0.9, hc[1] - 0.35, 0.95), C(hc[0], hc[1] + 0.85, 0.95))
    return dict(
        layers=[('snow_white', 'cane', cane),
                ('santa_red', 'stripes', red),
                ('holly_green', 'holly', leaves),
                ('santa_red', 'berries', berries)],
        hole=hole)


@design('santa-hat', 'Santa Hat',
        'A floppy red Santa hat with a bumpy white brim and pom-pom, plus a holly sprig tucked '
        'into the brim.',
        'For a plush look, add Fuzzy Skin to only the white part in Snapmaker Orca: right-click it and add the setting.')
def santa_hat():
    cone = P(bez((3.6, 4.5), (5.0, 13.0), (8.0, 24.5), (14.0, 27.2))
             + bez((14.0, 27.2), (19.5, 29.0), (24.4, 25.0), (24.8, 19.2))[1:]
             + bez((24.8, 19.2), (23.0, 22.6), (20.0, 22.0), (19.0, 17.8))[1:]
             + bez((19.0, 17.8), (18.4, 13.0), (21.0, 8.5), (22.4, 4.5))[1:])
    cone = smooth(cone, r_out=0.6)
    brim = rrect(1.0, 0.3, 25.0, 5.4, 2.4)
    fluff = [C(x, 5.25, 0.95) for x in np.arange(2.6, 24.0, 2.15)]
    fluff += [C(x, 0.45, 0.95) for x in np.arange(3.6, 23.0, 2.15)]
    brim = U(brim, fluff)
    pc = (24.6, 17.6)
    pom = U(C(pc[0], pc[1], 2.7), [C(pc[0] + 2.45 * math.cos(math.radians(a)), pc[1] + 2.45 * math.sin(math.radians(a)), 1.05)
                                   for a in range(0, 360, 45)])
    base = smooth(U(cone, brim, pom), r_in=0.5)
    hx = 15.6
    hole = (hx, top_y(base, hx) - 2.35)
    leaves = U(holly_leaf(5.4, 3.1, 5.4, 2.6, 162), holly_leaf(9.8, 3.4, 5.4, 2.6, 18))
    berries = U(C(7.3, 2.6, 0.85), C(8.1, 4.0, 0.85), C(6.6, 4.1, 0.85))
    return dict(
        layers=[('santa_red', 'hat', base),
                ('snow_white', 'brim and pom-pom', U(brim, pom)),
                ('holly_green', 'holly', leaves),
                ('santa_red', 'berries', berries)],
        hole=hole)


@design('snowman', 'Top-Hat Snowman',
        'A two-ball snowman in a black top hat with a red hat band, a knit scarf, coal eyes and '
        'buttons, twig arms and a carrot nose.',
        'The coal dots are 1.1 mm wide. Turn on thin-wall detection so the mouth comes out as dots, not blobs.')
def snowman():
    cx = 12.5
    body = C(cx, 9.0, 8.2)
    head = C(cx, 21.2, 5.8)
    snow = smooth(U(body, head), r_in=0.9)
    brim = rrect(5.6, 26.0, 19.4, 27.6, 0.7)
    crown = rrect(7.6, 26.8, 17.4, 35.2, 0.9)
    hat = U(brim, crown)
    arm_l = U(S([(5.0, 11.3), (0.8, 14.4)], 1.25), S([(2.5, 13.1), (1.9, 15.9)], 1.05))
    arm_r = mirror_x(arm_l, cx)
    eyes = [C(10.3, 22.7, 0.85), C(14.7, 22.7, 0.85)]
    mouth = [C(x, y, 0.56) for x, y in arc(cx, 20.9, 2.7, 222, 318, 4)]
    buttons = [C(cx, y, 1.05) for y in (12.4, 8.9, 5.4)]
    black = U(hat, arm_l, arm_r, eyes, mouth, buttons)
    band = box(7.0, 27.5, 18.0, 29.0).intersection(crown)
    scarf = smooth(P(bez((6.4, 16.9), (9.5, 15.4), (15.5, 15.4), (18.6, 16.9), 16)
                     + bez((18.6, 15.0), (15.5, 13.6), (9.5, 13.6), (6.4, 15.0), 16)), r_out=0.6)
    tail = S([(15.8, 14.6), (17.6, 7.6)], 2.7, cap='flat')
    d = np.array([17.6 - 15.8, 7.6 - 14.6])
    d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    stripes = U([S([tuple(np.array([17.6, 7.6]) - d * t - n * 2), tuple(np.array([17.6, 7.6]) - d * t + n * 2)], 0.75, cap='flat')
                 for t in (1.5, 3.0)])
    red = U(band, scarf, tail.difference(stripes))
    nose = smooth(P([(12.7, 21.95), (12.7, 20.25), (18.0, 20.55)]), r_out=0.25)
    base = smooth(U(snow, hat, arm_l, arm_r), r_in=0.4)
    return dict(
        layers=[('snow_white', 'snowman', base),
                ('charcoal', 'hat and coal', black),
                ('santa_red', 'scarf and hat band', red),
                ('carrot', 'nose', nose)],
        hole=(cx, 33.0))


@design('reindeer', 'Red-Nosed Reindeer',
        'A round reindeer face with branching cream antlers, a pale muzzle and a big red nose '
        'on top. The eyes and smile are windows that show the dark brown underneath.',
        'Print the nose in red glitter or silk PLA so it catches the light.')
def reindeer():
    cx = 14.0
    head = E(cx, 13.5, 7.2, 8.6)
    ear_l = E(5.4, 19.6, 4.0, 1.9, -25)
    ear_r = mirror_x(ear_l, cx)
    antler_l = U(S([(10.6, 20.6), (9.4, 25.0), (6.6, 28.6), (4.4, 31.0)], 2.0),
                 S([(9.4, 25.4), (10.9, 29.4)], 1.8),
                 S([(6.8, 28.3), (7.6, 32.0)], 1.7),
                 S([(5.8, 29.5), (2.6, 28.6)], 1.7))
    antler_r = mirror_x(antler_l, cx)
    antlers = U(antler_l, antler_r)
    tab = C(cx, 23.2, TAB_R)
    base = smooth(U(head, ear_l, ear_r, antlers, tab), r_in=0.6)
    eyes = U(C(11.2, 16.2, 1.05), C(16.8, 16.2, 1.05))
    face = U(head.buffer(-0.7), ear_l.buffer(-0.55), ear_r.buffer(-0.55)).difference(eyes)
    muzzle = E(cx, 8.8, 5.0, 4.2)
    smile = S(arc(cx, 8.6, 2.0, 205, 335, 14), 0.8)
    inner_ears = U(E(5.6, 19.5, 2.5, 0.85, -25), mirror_x(E(5.6, 19.5, 2.5, 0.85, -25), cx))
    cream = U(muzzle.difference(smile), inner_ears, antlers)
    nose = E(cx, 10.7, 2.7, 2.05)
    return dict(
        layers=[('chocolate', 'outline and eyes', base),
                ('gingerbread', 'face', face),
                ('cream', 'muzzle, ears and antlers', cream),
                ('santa_red', 'nose', nose)],
        hole=(cx, 23.2))


@design('holly-wreath', 'Holly Wreath',
        'An open wreath of two-tone holly leaves with red berry clusters and a red bow with a '
        'gold knot. The open center keeps it light.',
        'Silk gold PLA for the knot and beads makes this one look like real jewelry.')
def holly_wreath():
    cx, cy = 15.0, 15.0
    leaves = []
    for i in range(13):
        a = 90 + i * 360 / 13
        r = 10.6
        tilt = 16 if i % 2 else -16
        leaves.append(holly_leaf(cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)), 6.2, 2.9, a + 90 + tilt))
    for i in range(11):
        a = 90 + (i + 0.5) * 360 / 11
        r = 8.5
        tilt = -14 if i % 2 else 14
        leaves.append(holly_leaf(cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)), 5.4, 2.6, a + 90 + tilt))
    leaves = U(leaves)
    ring = C(cx, cy, 11.3).difference(C(cx, cy, 7.6))
    bow_loops = U(E(11.9, 3.9, 3.1, 1.85, 14), E(18.1, 3.9, 3.1, 1.85, -14))
    tails = U(S([(14.0, 3.4), (12.2, -1.6)], 1.7, cap='flat'), S([(16.0, 3.4), (17.8, -1.6)], 1.7, cap='flat'))
    knot = C(cx, 3.9, 1.3)
    bow = U(bow_loops, tails, knot.buffer(0.3))
    base = U(ring, leaves.buffer(0.45), bow.buffer(0.45))
    base = smooth(base, r_in=0.3)
    berries = []
    for a in (35, 145, 200, 340, 260 - 70, 280 + 70):
        bx, by = cx + 9.7 * math.cos(math.radians(a)), cy + 9.7 * math.sin(math.radians(a))
        berries += [C(bx - 0.8, by - 0.45, 0.85), C(bx + 0.8, by - 0.45, 0.85), C(bx, by + 0.85, 0.85)]
    gold = U(knot, [C(cx + 9.6 * math.cos(math.radians(a)), cy + 9.6 * math.sin(math.radians(a)), 0.8) for a in (0, 180, 60, 120)])
    return dict(
        layers=[('pine_green', 'backing', base),
                ('holly_green', 'leaves', leaves),
                ('santa_red', 'bow and berries', U(berries, bow)),
                ('gold', 'knot and beads', gold)],
        hole=(cx, 24.8))


@design('christmas-tree', 'Trimmed Tree',
        'A three-tier tree with scalloped branches, gold garland swags, red ornaments and a gold '
        'star that doubles as the hanging loop.',
        'Swap the green for white or pale green for a flocked tree. Nothing else in the file changes.')
def christmas_tree():
    cx = 14.0

    def tier(y0, x0, x1, apex):
        tri = smooth(P([(x0, y0), (x1, y0), (cx, apex)]), r_out=0.7)
        n = int(round((x1 - x0 - 2.0) / 2.3)) + 1
        sc = [C(x, y0 + 0.05, 1.2) for x in np.linspace(x0 + 1.5, x1 - 1.5, n)]
        return U(tri, sc)

    t1 = tier(5.2, 1.0, 27.0, 16.6)
    t2 = tier(12.6, 3.8, 24.2, 23.2)
    t3 = tier(19.8, 6.6, 21.4, 29.4)
    tree = U(t1, t2, t3)
    trunk = rrect(11.4, 1.0, 16.6, 6.4, 0.5)
    st = star(cx, 30.4, 3.6, 1.5)
    tab = C(cx, 34.0, TAB_R)
    topper = smooth(U(st, tab), r_in=0.7)
    base = smooth(U(tree, trunk, topper), r_in=0.4)
    garland = U(S(bez((3.2, 7.4), (9.0, 5.4), (17.0, 9.8), (24.4, 11.4)), 0.95).intersection(t1.buffer(-0.35)),
                S(bez((21.8, 14.4), (16.0, 14.4), (11.0, 18.6), (6.4, 19.2)), 0.95).intersection(t2.buffer(-0.35)),
                S(bez((9.2, 21.4), (12.0, 21.0), (16.0, 24.4), (19.0, 24.8)), 0.95).intersection(t3.buffer(-0.35)))
    ornaments = [C(x, y, 1.15) for x, y in ((6.0, 9.6), (12.6, 10.8), (20.4, 7.6), (22.6, 13.0),
                                             (9.4, 15.4), (16.6, 19.6), (19.2, 15.6), (11.6, 23.4), (16.2, 26.0))]
    return dict(
        layers=[('chocolate', 'trunk', base),
                ('holly_green', 'branches', tree),
                ('gold', 'star and garland', U(topper, garland)),
                ('santa_red', 'ornaments', U(ornaments))],
        hole=(cx, 34.0))


@design('snowflake', 'Layered Snowflake',
        'Two six-arm snowflakes stacked and turned 30 degrees, so together they read as a '
        'twelve-point flake. A silver gem sits in the middle.',
        'Translucent or glow-in-the-dark filament on the bottom flake looks great. The arms are 1.7 to 1.9 mm wide.')
def snowflake():
    cx, cy = 15.5, 15.5

    def flake(L, w, branches, rot, hub):
        parts = [hexagon(cx, cy, hub, rot + 30)]
        for k in range(6):
            a = math.radians(rot + 60 * k)
            d = np.array([math.cos(a), math.sin(a)])
            o = np.array([cx, cy])
            parts.append(S([tuple(o), tuple(o + d * L)], w))
            for frac, bl, bw in branches:
                p = o + d * L * frac
                for s in (1, -1):
                    b = math.radians(rot + 60 * k + s * 52)
                    parts.append(S([tuple(p), tuple(p + np.array([math.cos(b), math.sin(b)]) * bl)], bw))
        return U(parts)

    big = flake(14.2, 1.9, [(0.47, 4.2, 1.55), (0.76, 2.8, 1.45)], 90, 3.0)
    small = flake(9.4, 1.75, [(0.58, 2.5, 1.35)], 120, 2.7)
    tab = C(cx, cy + 15.6, TAB_R)
    base = smooth(U(big, tab), r_in=0.5)
    gem = U(hexagon(cx, cy, 1.65, 0), [C(cx + 9.4 * math.cos(math.radians(120 + 60 * k)),
                                          cy + 9.4 * math.sin(math.radians(120 + 60 * k)), 0.9) for k in range(6)])
    return dict(
        layers=[('ice_blue', 'large flake', base),
                ('snow_white', 'small flake', small),
                ('silver', 'gem and tips', gem)],
        hole=(cx, cy + 15.6))


@design('holiday-lights', 'Mismatched Holiday Lights',
        'Old-fashioned C9 string-light bulbs with ridged silver sockets and a glint of shine. '
        'The left bulb prints red and the right prints green in the same job.',
        'Translucent red and green filament make the bulbs glow when light hits them.')
def holiday_lights():
    cx = 8.0
    bulb = U(C(cx, 12.8, 6.7), C(cx, 1.5, 1.35)).convex_hull
    socket = rrect(4.1, 17.2, 11.9, 26.4, 0.9)
    cap = rrect(5.3, 25.6, 10.7, 27.8, 0.6)
    tab = C(cx, 29.3, TAB_R)
    base = smooth(U(bulb, socket, cap, tab), r_in=0.6)
    glass = bulb.difference(socket.buffer(0.15))
    grooves = U([box(2, y, 14, y + 0.75) for y in (19.4, 21.7, 24.0)])
    metal = U(socket, cap).difference(grooves)
    glint = U(S(arc(cx, 12.6, 4.4, 160, 212, 14), 1.1), C(4.9, 6.7, 0.6))
    return dict(
        layers=[('charcoal', 'socket core and loop', base),
                ('santa_red', 'bulb', glass),
                ('silver', 'socket and glint', U(metal, glint))],
        hole=(cx, 29.3),
        right={1: 'holly_green'})


# ----------------------------------------------------------------- build
def build_design(d):
    spec = d['fn']()
    layers = spec['layers']
    assert len(layers) <= 4
    hole = C(spec['hole'][0], spec['hole'][1], HOLE_R, 32)
    regions = [clean(r) for _, _, r in layers]
    solids = [None] * len(regions)
    acc = Polygon()
    for k in range(len(regions) - 1, -1, -1):
        acc = U(acc, regions[k])
        solids[k] = clean(acc.difference(hole))
    # thin-feature report (anything narrower than ~0.7 mm)
    for k, s in enumerate(solids):
        lost = s.area - s.buffer(-0.33).buffer(0.33).area
        if lost > 0.6:
            print(f"  note: {d['slug']} band {k + 1} has {lost:.1f} mm^2 narrower than 0.66 mm")
    return spec, solids


def extrude(geom, z0, z1):
    meshes = []
    for p in polys(geom):
        m = trimesh.creation.extrude_polygon(p, z1 - z0)
        m.apply_translation([0, 0, z0])
        meshes.append(m)
    m = trimesh.util.concatenate(meshes)
    m.merge_vertices()
    return m


# The "base loadout": these colors always land on the same toolhead, so the
# U1 can keep red, white, green and gold loaded and swap only what a design adds.
PREFERRED_TOOL = {'santa_red': 0, 'snow_white': 1, 'holly_green': 2, 'gold': 3}


def tool_plan(spec):
    """Map each color to a toolhead (0-based). Base-loadout colors keep their usual tool."""
    colors = []
    for key, _, _ in spec['layers']:
        if key not in colors:
            colors.append(key)
    for key in spec.get('right', {}).values():
        if key not in colors:
            colors.append(key)
    assert len(colors) <= 4, colors
    slot = {}
    for key in colors:
        if key in PREFERRED_TOOL:
            slot[key] = PREFERRED_TOOL[key]
    free = [t for t in range(4) if t not in slot.values()]
    for key in colors:
        if key not in slot:
            slot[key] = free.pop(0)
    return slot


def write_3mf(path, title, objects, tools):
    """objects: list of (name, [(part name, trimesh, tool index 0-based)], (x, y) plate position)."""
    model = io.StringIO()
    model.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    model.write('<model unit="millimeter" xml:lang="en-US" '
                'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n')
    model.write(f' <metadata name="Title">{title}</metadata>\n')
    model.write(' <metadata name="Designer">Stocking Stuffer Earrings</metadata>\n')
    model.write(' <resources>\n  <basematerials id="1">\n')
    by_slot = {t: k for k, t in tools.items()}
    for t in range(4):
        key = by_slot.get(t)
        name, hexc = PALETTE[key] if key else ('Unused', '#808080')
        model.write(f'   <base name="{name}" displaycolor="{hexc}FF"/>\n')
    model.write('  </basematerials>\n')
    next_id = 2
    tops = []
    for oname, parts, pos in objects:
        ids = []
        for pname, mesh, tool in parts:
            oid = next_id
            next_id += 1
            ids.append((oid, pname, tool))
            model.write(f'  <object id="{oid}" name="{pname}" type="model" pid="1" pindex="{tool}">\n'
                        '   <mesh>\n    <vertices>\n')
            for v in mesh.vertices:
                model.write(f'     <vertex x="{v[0]:.4f}" y="{v[1]:.4f}" z="{v[2]:.4f}"/>\n')
            model.write('    </vertices>\n    <triangles>\n')
            for f in mesh.faces:
                model.write(f'     <triangle v1="{f[0]}" v2="{f[1]}" v3="{f[2]}"/>\n')
            model.write('    </triangles>\n   </mesh>\n  </object>\n')
        tid = next_id
        next_id += 1
        model.write(f'  <object id="{tid}" name="{oname}" type="model">\n   <components>\n')
        for oid, _, _ in ids:
            model.write(f'    <component objectid="{oid}"/>\n')
        model.write('   </components>\n  </object>\n')
        tops.append((tid, oname, ids, pos))
    model.write(' </resources>\n <build>\n')
    for tid, _, _, pos in tops:
        model.write(f'  <item objectid="{tid}" transform="1 0 0 0 1 0 0 0 1 {pos[0]:.3f} {pos[1]:.3f} 0"/>\n')
    model.write(' </build>\n</model>\n')

    # Per-part tool assignment, read by OrcaSlicer, Snapmaker Orca and Bambu Studio.
    cfg = io.StringIO()
    cfg.write('<?xml version="1.0" encoding="UTF-8"?>\n<config>\n')
    for tid, oname, ids, _ in tops:
        cfg.write(f'  <object id="{tid}">\n')
        cfg.write(f'    <metadata key="name" value="{oname}"/>\n')
        cfg.write(f'    <metadata key="extruder" value="{ids[0][2] + 1}"/>\n')
        for oid, pname, tool in ids:
            cfg.write(f'    <part id="{oid}" subtype="normal_part">\n')
            cfg.write(f'      <metadata key="name" value="{pname}"/>\n')
            cfg.write('      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n')
            cfg.write(f'      <metadata key="extruder" value="{tool + 1}"/>\n')
            cfg.write('    </part>\n')
        cfg.write('  </object>\n')
    cfg.write('</config>\n')

    content_types = ('<?xml version="1.0" encoding="UTF-8"?>\n'
                     '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
                     ' <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
                     ' <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
                     ' <Default Extension="config" ContentType="text/xml"/>\n'
                     '</Types>\n')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
            ' <Relationship Target="/3D/3dmodel.model" Id="rel0" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n'
            '</Relationships>\n')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', content_types)
        z.writestr('_rels/.rels', rels)
        z.writestr('3D/3dmodel.model', model.getvalue())
        z.writestr('Metadata/model_settings.config', cfg.getvalue())


# ----------------------------------------------------------------- SVG preview
def darker(hex_color, f=0.68):
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return '#%02x%02x%02x' % (int(r * f), int(g * f), int(b * f))


def svg_path(geom, fx):
    out = []
    for p in polys(geom):
        for ring in [p.exterior] + list(p.interiors):
            pts = list(ring.coords)[:-1]
            out.append('M' + ' L'.join(f'{fx(x):.2f} {fx.y(y):.2f}' for x, y in pts) + 'Z')
    return ' '.join(out)


class Frame:
    def __init__(self, minx, maxy):
        self.minx, self.maxy = minx, maxy

    def __call__(self, x):
        return x - self.minx

    def y(self, y):
        return self.maxy - y


def svg_earring(slug, solids, keys, hole, frame, dx=0.0):
    fx = frame
    hx, hy = fx(hole[0]) + dx, fx.y(hole[1])
    ring_c = hy - 1.55
    ring_top = ring_c - 2.5
    gid = f'metal-{slug}-{int(dx)}'
    s = [f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="1">'
         f'<stop offset="0" stop-color="#e9e3d6"/><stop offset="0.5" stop-color="#b9ae98"/>'
         f'<stop offset="1" stop-color="#8e8471"/></linearGradient></defs>']
    s.append(f'<path d="M{hx:.2f} {ring_top - 0.55:.2f} L{hx:.2f} {ring_top - 4.15:.2f} '
             f'C{hx:.2f} {ring_top - 9.75:.2f} {hx + 6.4:.2f} {ring_top - 9.75:.2f} {hx + 6.4:.2f} {ring_top - 3.95:.2f} '
             f'L{hx + 6.4:.2f} {ring_top - 0.15:.2f}" fill="none" stroke="url(#{gid})" stroke-width="0.62" stroke-linecap="round"/>')
    s.append(f'<circle cx="{hx:.2f}" cy="{ring_top - 0.75:.2f}" r="0.75" fill="url(#{gid})"/>')
    for k, (geom, key) in enumerate(zip(solids, keys)):
        color = PALETTE[key][1]
        g = affinity.translate(geom, dx, 0)
        d = svg_path(g, fx)
        off = '0.35 0.5' if k == 0 else '0.2 0.32'
        op = '0.22' if k == 0 else '0.3'
        s.append(f'<g class="lay" data-k="{k + 1}">'
                 f'<path d="{d}" transform="translate({off})" fill="#1b0d05" fill-opacity="{op}" fill-rule="evenodd"/>'
                 f'<path d="{d}" fill="{color}" fill-rule="evenodd" stroke="{darker(color)}" stroke-width="0.12" stroke-linejoin="round"/>'
                 '</g>')
    s.append(f'<circle cx="{hx:.2f}" cy="{ring_c:.2f}" r="2.5" fill="none" stroke="url(#{gid})" stroke-width="0.5"/>')
    return s, ring_top - 9.75


def build_svg(slug, name, solids, keys_left, keys_right, hole, pair=False):
    minx, miny, maxx, maxy = solids[0].bounds
    frame = Frame(minx, maxy)
    w, h = maxx - minx, maxy - miny
    gap = 4.0
    body, top = svg_earring(slug, solids, keys_left, hole, frame)
    if pair:
        right = [mirror_x(s, (minx + maxx) / 2) for s in solids]
        hole_r = (minx + maxx - hole[0], hole[1])
        b2, _ = svg_earring(slug, right, keys_right, hole_r, frame, dx=w + gap)
        body += b2
        w = 2 * w + gap
    vb_top = min(top - 0.6, -1.0)
    pad = 3.2
    vb = f'{-pad:.2f} {vb_top:.2f} {w + 2 * pad + 3.2:.2f} {h - vb_top + 1.6:.2f}'
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" role="img" '
            f'aria-label="{name} earring preview">' + ''.join(body) + '</svg>')


# ----------------------------------------------------------------- main
def main():
    manifest = []
    for i, d in enumerate(DESIGNS, 1):
        tag = f'{i:02d}-{d["slug"]}'
        print(tag)
        spec, solids = build_design(d)
        tools = tool_plan(spec)
        keys_left = [k for k, _, _ in spec['layers']]
        keys_right = [spec.get('right', {}).get(k, key) for k, key in enumerate(keys_left)]
        labels = [lab for _, lab, _ in spec['layers']]

        minx, miny, maxx, maxy = solids[0].bounds
        mx, my = (minx + maxx) / 2, (miny + maxy) / 2
        width, height = maxx - minx, maxy - miny
        sep = width / 2 + 5.0

        objects = []
        volume = 0.0
        for side, keys, sgn in (('Left', keys_left, -1), ('Right', keys_right, 1)):
            parts = []
            for k, (geom, key) in enumerate(zip(solids, keys)):
                g = geom if side == 'Left' else mirror_x(geom, mx)
                g = affinity.translate(g, -mx, -my)
                z0, z1 = BANDS[k]
                mesh = extrude(g, z0, z1)
                if not mesh.is_watertight:
                    print(f'  WARNING: {tag} {side} band {k + 1} not watertight')
                if side == 'Left':
                    volume += mesh.volume
                pname = f'{PALETTE[key][0]} {labels[k]} (T{tools[key] + 1})'
                parts.append((pname, mesh, tools[key]))
            objects.append((f'{d["name"]} {side}', parts, (PLATE_CENTER[0] + sgn * sep, PLATE_CENTER[1])))

        folder = os.path.join(OUT_DIR, tag)
        os.makedirs(folder, exist_ok=True)
        tmf = os.path.join(folder, f'{tag}.3mf')
        write_3mf(tmf, d['name'], objects, tools)

        pair = bool(spec.get('right'))
        svg = build_svg(d['slug'], d['name'], solids, keys_left, keys_right, spec['hole'], pair=pair)
        with open(os.path.join(folder, 'preview.svg'), 'w') as f:
            f.write(svg)

        grams = volume / 1000.0 * PLA_DENSITY
        thick = BANDS[len(solids) - 1][1]
        manifest.append(dict(
            tag=tag, slug=d['slug'], name=d['name'], blurb=d['blurb'], tip=d['tip'],
            size=[round(width, 1), round(height, 1), thick], grams=round(grams, 1),
            bands=[dict(k=k + 1, key=key, color=PALETTE[key][0], hex=PALETTE[key][1],
                        label=labels[k], tool=tools[key] + 1,
                        z=[BANDS[k][0], BANDS[k][1]],
                        right=(None if keys_right[k] == key else dict(
                            key=keys_right[k], color=PALETTE[keys_right[k]][0], hex=PALETTE[keys_right[k]][1],
                            tool=tools[keys_right[k]] + 1)))
                   for k, key in enumerate(keys_left)],
            tools=[dict(t=t + 1, key=k, color=PALETTE[k][0], hex=PALETTE[k][1]) for k, t in sorted(tools.items(), key=lambda kv: kv[1])],
            pair_preview=pair, file=f'{tag}/{tag}.3mf', svg=svg,
            bytes=os.path.getsize(tmf)))
        print(f'  {width:.1f} x {height:.1f} x {thick} mm, {grams:.2f} g each, tools {len(tools)}, '
              f'{os.path.getsize(tmf) / 1024:.0f} KB')

    with open(os.path.join(OUT_DIR, 'designs.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    print('done ->', OUT_DIR)


if __name__ == '__main__':
    main()
