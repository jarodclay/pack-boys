#!/usr/bin/env python3
"""
Thanksgiving turkey earring generator.

Builds a flat, front-facing turkey earring: a red / orange / yellow tail fan
behind a brown body, an orange beak and feet, a red wattle, and a loop at the
top for a jump ring. The back is flat. On the front each feather row steps up
a little, so the earring looks layered.

Exports (one earring per STL, all sharing the same origin):

  turkey_earring.3mf             a pair of earrings, 4 parts each (brown, red,
                                 orange, yellow), ready for a 4-tool printer
  turkey_earring_brown.stl       body + head
  turkey_earring_red.stl         outer feathers + loop + wattle
  turkey_earring_orange.stl      middle feathers + beak + feet
  turkey_earring_yellow.stl      inner feathers
  turkey_eyes_black.stl          8 eye inserts (4 + spares) to print in black
                                 and glue into the eye holes

Requires: pip install numpy trimesh manifold3d
Run:      python3 make_turkey_earring.py
All sizes are in millimetres; tweak the PARAMETERS block and re-run.
"""
import io
import os
import zipfile

import numpy as np
import trimesh
from manifold3d import CrossSection, JoinType, Manifold, OpType

# ----------------------------------------------------------------- PARAMETERS
# Heights of each layer. The back is flat; the front steps up toward the body.
H_RED = 1.6          # outer feather row (and the loop)
H_ORANGE = 2.0       # middle feather row (and the feet)
H_YELLOW = 2.4       # inner feather row
H_BODY = 2.8         # body + head (and the eye inserts)
H_FACE = 3.2         # beak and wattle stand a little proud of the head

LOOP_HOLE = 2.0      # hole for the jump ring
LOOP_RING = 1.6      # width of plastic around the hole

EYE_HOLE = 2.8       # eye holes through the head
EYE_CLEARANCE = 0.1  # per side; the printed eye is EYE_HOLE - 2 * this
EYE_COUNT = 8        # eyes on the insert plate (a pair of earrings needs 4)

GROOVE_WIDTH = 0.7   # engraved detail lines on the front (feathers, wings)
GROOVE_DEPTH = 0.4

SEGMENTS = 96        # curve smoothness
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLATE_CENTER = (135.0, 135.0)   # Snapmaker U1 bed is 270 x 270 mm
# ---------------------------------------------------------------------------

FAN_CENTER = np.array([0.0, -3.0])
# (count, first angle, last angle, length, width) of each feather row.
# Angles are measured from straight up, positive to the right.
ROW_RED = (9, -100.0, 100.0, 20.0, 8.4)
ROW_ORANGE = (8, -87.5, 87.5, 16.0, 7.6)
ROW_YELLOW = (9, -100.0, 100.0, 12.2, 6.2)


def circle(r, x=0.0, y=0.0):
    return CrossSection.circle(r, SEGMENTS).translate([x, y])


def ellipse(x, y, rx, ry, rot=0.0):
    return CrossSection.circle(1.0, SEGMENTS).scale([rx, ry]).rotate(rot).translate([x, y])


def stroke(points, width):
    """A round-ended line of the given width through a list of points."""
    r = width / 2.0
    caps = [circle(r, *p) for p in points]
    return union([CrossSection.batch_hull([a, b]) for a, b in zip(caps, caps[1:])])


def union(shapes):
    out = shapes[0]
    for s in shapes[1:]:
        out = out + s
    return out


def close(cs, r):
    """Round off inside corners (fillet) with radius r."""
    return cs.offset(r, JoinType.Round, 2.0, SEGMENTS).offset(-r, JoinType.Round, 2.0, SEGMENTS)


def soften(cs, r):
    """Round off outside corners with radius r."""
    return cs.offset(-r, JoinType.Round, 2.0, SEGMENTS).offset(r, JoinType.Round, 2.0, SEGMENTS)


def row_angles(row):
    n, a0, a1, _, _ = row
    return np.linspace(a0, a1, n)


def direction(deg):
    t = np.radians(deg)
    return np.array([np.sin(t), np.cos(t)])


def feather_row(row):
    _, _, _, length, width = row
    feathers = []
    for a in row_angles(row):
        c = FAN_CENTER + direction(a) * length / 2.0
        feathers.append(ellipse(c[0], c[1], width / 2.0, length / 2.0, -a))
    return union(feathers)


def feather_veins(row, start, visible):
    """Centre line of every feather in a row, from `start` out toward the tip,
    kept only where that feather shows. Stubs shorter than ~2 mm are dropped."""
    _, _, _, length, _ = row
    inside = visible.offset(-0.8, JoinType.Round, 2.0, SEGMENTS)
    lines = []
    for a in row_angles(row):
        d = direction(a)
        vein = stroke([FAN_CENTER + d * start, FAN_CENTER + d * (length - 1.6)],
                      GROOVE_WIDTH) ^ inside
        lines += [v for v in vein.decompose() if v.area() > GROOVE_WIDTH * 2.0]
    return union(lines) if lines else CrossSection()


def build_2d():
    red_row = feather_row(ROW_RED)
    orange_row = feather_row(ROW_ORANGE)
    yellow_row = feather_row(ROW_YELLOW)

    body = ellipse(0.0, -6.0, 8.6, 8.9)
    head = circle(5.0, 0.0, 5.6)
    brown = close(body + head, 2.0)

    beak = soften(CrossSection([np.array([(-1.7, 4.7), (0.0, 1.9), (1.7, 4.7), (0.0, 5.4)])]), 0.35)
    wattle = CrossSection.batch_hull([circle(1.25, 1.35, 0.6), circle(0.75, 0.55, 3.4)])

    feet = []
    for s in (-1.0, 1.0):
        x = 3.3 * s
        feet += [stroke([(x, -13.0), (x, -16.0)], 1.4),
                 stroke([(x, -16.0), (x - 1.6, -17.6)], 1.2),
                 stroke([(x, -16.0), (x, -18.2)], 1.2),
                 stroke([(x, -16.0), (x + 1.6, -17.6)], 1.2)]
    feet = union(feet)

    tip_y = FAN_CENTER[1] + ROW_RED[3]
    loop_r = LOOP_HOLE / 2.0 + LOOP_RING
    loop_c = (0.0, tip_y + loop_r - 1.4)
    loop = circle(loop_r, *loop_c)
    loop_hole = circle(LOOP_HOLE / 2.0, *loop_c)
    red_back = close(red_row + loop, 0.6)

    eye_holes = union([circle(EYE_HOLE / 2.0, s * 2.05, 7.1) for s in (-1.0, 1.0)])

    # Front-to-back stacking: each shape only keeps what is not covered by
    # something in front of it, so the colour regions never overlap.
    stack = [
        ('beak', beak),
        ('wattle', wattle),
        ('brown', brown),
        ('feet', feet),
        ('yellow', yellow_row),
        ('orange', orange_row),
        ('red', red_back),
    ]
    covered = eye_holes + loop_hole
    regions = {}
    for name, shape in stack:
        regions[name] = shape - covered
        covered = covered + shape

    # Engraved detail on the front.
    grooves = {
        'red': feather_veins(ROW_RED, 12.0, regions['red']),
        'orange': feather_veins(ROW_ORANGE, 9.0, regions['orange']),
        'yellow': feather_veins(ROW_YELLOW, 6.0, regions['yellow']),
    }
    wings = []
    for s in (-1.0, 1.0):
        w = ellipse(5.0 * s, -6.4, 3.1, 5.6, 8.0 * s)
        wings.append(w - w.offset(-GROOVE_WIDTH))
    grooves['brown'] = union(wings) ^ regions['brown'].offset(-0.9)
    return regions, grooves


def extrude(cs, h):
    return Manifold.extrude(cs, h)


def build_3d():
    regions, grooves = build_2d()

    def layer(name, h):
        m = extrude(regions[name], h)
        if name in grooves and not grooves[name].is_empty():
            cut = extrude(grooves[name], GROOVE_DEPTH + 1.0).translate([0, 0, h - GROOVE_DEPTH])
            m = m - cut
        return m

    parts = {
        'brown': layer('brown', H_BODY),
        'red': Manifold.batch_boolean([layer('red', H_RED), extrude(regions['wattle'], H_FACE)],
                                      OpType.Add),
        'orange': Manifold.batch_boolean([layer('orange', H_ORANGE), extrude(regions['beak'], H_FACE),
                                          extrude(regions['feet'], H_ORANGE)], OpType.Add),
        'yellow': layer('yellow', H_YELLOW),
    }

    eye_d = EYE_HOLE - 2.0 * EYE_CLEARANCE
    cols = 4
    eyes = []
    for i in range(EYE_COUNT):
        x = (i % cols - (cols - 1) / 2.0) * (eye_d + 4.0)
        y = (i // cols - (EYE_COUNT // cols - 1) / 2.0) * (eye_d + 4.0)
        eyes.append(Manifold.cylinder(H_BODY, eye_d / 2.0, eye_d / 2.0, SEGMENTS).translate([x, y, 0]))
    parts['eyes'] = Manifold.batch_boolean(eyes, OpType.Add)
    return regions, parts


def to_trimesh(m):
    mesh = m.to_mesh()
    return trimesh.Trimesh(vertices=np.asarray(mesh.vert_properties)[:, :3],
                           faces=np.asarray(mesh.tri_verts), process=True)


def write_3mf(path, title, parts, placements):
    """parts: list of (name, trimesh, '#RRGGBB') forming one object.
    placements: list of (x, y) plate positions, one earring at each."""
    out = io.StringIO()
    out.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    out.write('<model unit="millimeter" xml:lang="en-US" '
              'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n')
    out.write(f' <metadata name="Title">{title}</metadata>\n')
    out.write(' <resources>\n  <basematerials id="1">\n')
    for name, _, color in parts:
        out.write(f'   <base name="{name}" displaycolor="{color}FF"/>\n')
    out.write('  </basematerials>\n')
    for idx, (name, tm, _) in enumerate(parts):
        oid = idx + 2
        out.write(f'  <object id="{oid}" name="{name}" type="model" pid="1" pindex="{idx}">\n'
                  '   <mesh>\n    <vertices>\n')
        for v in tm.vertices:
            out.write(f'     <vertex x="{v[0]:.4f}" y="{v[1]:.4f}" z="{v[2]:.4f}"/>\n')
        out.write('    </vertices>\n    <triangles>\n')
        for f in tm.faces:
            out.write(f'     <triangle v1="{f[0]}" v2="{f[1]}" v3="{f[2]}"/>\n')
        out.write('    </triangles>\n   </mesh>\n  </object>\n')
    top = len(parts) + 2
    out.write(f'  <object id="{top}" name="{title}" type="model">\n   <components>\n')
    for idx in range(len(parts)):
        out.write(f'    <component objectid="{idx + 2}"/>\n')
    out.write('   </components>\n  </object>\n </resources>\n <build>\n')
    for x, y in placements:
        out.write(f'  <item objectid="{top}" transform="1 0 0 0 1 0 0 0 1 {x:.3f} {y:.3f} 0"/>\n')
    out.write(' </build>\n</model>\n')

    content_types = ('<?xml version="1.0" encoding="UTF-8"?>\n'
                     '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
                     ' <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
                     ' <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
                     '</Types>\n')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
            ' <Relationship Target="/3D/3dmodel.model" Id="rel0" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n'
            '</Relationships>\n')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', content_types)
        z.writestr('_rels/.rels', rels)
        z.writestr('3D/3dmodel.model', out.getvalue())


COLORS = {
    'brown': ('1 Brown - body', '#7A4A26'),
    'red': ('2 Red - outer feathers, loop, wattle', '#C62A1E'),
    'orange': ('3 Orange - middle feathers, beak, feet', '#F07F12'),
    'yellow': ('4 Yellow - inner feathers', '#F7C51E'),
}


def main():
    regions, parts = build_3d()
    meshes = {k: to_trimesh(v) for k, v in parts.items()}

    total = 0.0
    for name, tm in meshes.items():
        bodies = len(parts[name].decompose())
        print(f"{name:7s} watertight={tm.is_watertight} tris={len(tm.faces):6d} "
              f"vol={tm.volume:7.1f} mm^3 bodies={bodies}")
        if name != 'eyes':
            total += tm.volume
    allb = np.array([meshes[k].bounds for k in COLORS])
    lo, hi = allb[:, 0].min(axis=0), allb[:, 1].max(axis=0)
    size = hi - lo
    print(f"earring size {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} mm, "
          f"~{total * 1.24e-3:.1f} g in PLA at 100% infill")

    for key in COLORS:
        meshes[key].export(os.path.join(OUT_DIR, f'turkey_earring_{key}.stl'))
    meshes['eyes'].export(os.path.join(OUT_DIR, 'turkey_eyes_black.stl'))

    gap = 8.0
    dx = (size[0] + gap) / 2.0
    cx, cy = PLATE_CENTER
    write_3mf(os.path.join(OUT_DIR, 'turkey_earring.3mf'), 'Turkey Earring',
              [(COLORS[k][0], meshes[k], COLORS[k][1]) for k in COLORS],
              [(cx - dx, cy), (cx + dx, cy)])
    print("done ->", OUT_DIR)


if __name__ == '__main__':
    main()
