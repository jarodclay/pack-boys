#!/usr/bin/env python3
"""
Baseball hoodie bead generator.

Builds a baseball-shaped drawstring bead (white ball + red V stitches) and
exports print-ready files:

  baseball_bead.3mf               two-part model (white ball + red stitches) for
                                  multi-color printers (AMS / MMU / tool changer)
  baseball_bead_white_ball.stl    just the white ball (stitch pockets cut out)
  baseball_bead_red_stitches.stl  just the red stitches (load as a 2nd part)
  baseball_bead_single_color.stl  one solid piece with raised stitches, for
                                  single-color printers (paint the stitches red)

Requires: pip install numpy trimesh manifold3d
Run:      python3 make_baseball_bead.py
All sizes are in millimetres; tweak the PARAMETERS block and re-run.
"""
import io
import os
import zipfile

import numpy as np
import trimesh
from manifold3d import Manifold, Mesh

# ----------------------------------------------------------------- PARAMETERS
BALL_DIAMETER = 24.0      # outside diameter of the ball
HOLE_DIAMETER = 7.0       # drawstring hole (fits typical hoodie cords)
FLAT_HEIGHT = 20.8        # ball is trimmed flat top & bottom to this height
HOLE_CHAMFER = 0.6        # 45 deg chamfer on the hole edges

STITCH_COUNT = 44         # number of V stitches around the seam
STITCH_WIDTH = 0.85       # thickness of each stitch "thread"
STITCH_RAISE = 0.45       # how far stitches stand proud of the ball surface
STITCH_DEPTH = 0.8        # how far stitches sink into the ball (anchoring)
STITCH_INNER = 0.38       # distance of the V's tip from the seam line
STITCH_SPREAD = 1.15      # how far each arm reaches away from the seam
STITCH_TRAIL = 0.9        # how far each arm sweeps back along the seam

SEAM_SHAPE = 0.30         # baseball-curve shape factor (0.25..0.40)
SEAM_GROOVE_RADIUS = 0.35 # shallow seam groove between the two "leather" halves
SEAM_GROOVE_DEPTH = 0.30

SEGMENTS = 192            # sphere smoothness
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
# ---------------------------------------------------------------------------

R = BALL_DIAMETER / 2.0


def seam_curve(n):
    """Classic baseball/tennis seam curve; lies exactly on the unit sphere."""
    b = SEAM_SHAPE
    a = 1.0 - b
    c = 2.0 * np.sqrt(a * b)
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts = np.stack([a * np.cos(t) + b * np.cos(3 * t),
                    a * np.sin(t) - b * np.sin(3 * t),
                    c * np.sin(2 * t)], axis=1)
    # Rotate so the centre of one leather "lobe" (1,1,0)/sqrt2 points up (+Z).
    # The hole goes through the two lobe centres, far from any stitching.
    rot = np.array([[0.0, 0.0, 1.0],
                    [1 / np.sqrt(2), -1 / np.sqrt(2), 0.0],
                    [1 / np.sqrt(2), 1 / np.sqrt(2), 0.0]])
    return pts @ rot.T


def resample_closed(pts, n):
    """Resample a closed polyline to n points evenly spaced by arc length."""
    seg = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    closed = np.vstack([pts, pts[:1]])
    targets = np.linspace(0.0, total, n, endpoint=False)
    out = np.empty((n, 3))
    for k in range(3):
        out[:, k] = np.interp(targets, s, closed[:, k])
    return out / np.linalg.norm(out, axis=1, keepdims=True), total


def frames(p):
    """Tangent and side (binormal) unit vectors for points on the unit sphere."""
    t = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    t -= (t * p).sum(1, keepdims=True) * p
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    side = np.cross(p, t)
    return t, side


def to_manifold(verts, faces):
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32)
    # make sure the triangles face outward
    v0, v1, v2 = (verts[faces[:, i]].astype(np.float64) for i in range(3))
    if np.einsum('ij,ij->i', v0, np.cross(v1, v2)).sum() < 0:
        faces = faces[:, ::-1].copy()
    return Manifold(Mesh(vert_properties=verts, tri_verts=faces))


def seam_groove(p, side):
    """Closed tube following the seam, used to carve a shallow groove."""
    rt = SEAM_GROOVE_RADIUS
    rc = R + rt - SEAM_GROOVE_DEPTH
    m = 16
    phi = np.linspace(0, 2 * np.pi, m, endpoint=False)
    n = len(p)
    rings = (rc * p)[:, None, :] + rt * (np.cos(phi)[None, :, None] * p[:, None, :]
                                         + np.sin(phi)[None, :, None] * side[:, None, :])
    verts = rings.reshape(-1, 3)
    faces = []
    for i in range(n):
        i2 = (i + 1) % n
        for j in range(m):
            j2 = (j + 1) % m
            a, b_, c, d = i * m + j, i2 * m + j, i2 * m + j2, i * m + j2
            faces += [(a, b_, c), (a, c, d)]
    return to_manifold(verts, faces)


def stitches(p, tangent, side, idx, trail):
    """Union of all V stitches (two short rounded bars per stitch).

    p/tangent/side are a finely, evenly sampled seam; idx are the sample
    indices where stitches sit; each arm's outer end is anchored `trail`
    samples back along the seam so the V's follow the curve in tight bends.
    """
    rs = STITCH_WIDTH / 2.0
    ball = trimesh.creation.icosphere(subdivisions=2, radius=rs).vertices
    n = len(p)
    bars = []
    for i in idx:
        j = (i - trail) % n
        for sgn in (1.0, -1.0):
            ends = []
            for k, across in ((i, STITCH_INNER), (j, STITCH_INNER + STITCH_SPREAD)):
                d = p[k] + sgn * across * side[k] / R
                ends.append(d / np.linalg.norm(d))
            cloud = []
            for e in ends:
                cloud.append(ball + e * (R + STITCH_RAISE - rs))  # rounded top
                cloud.append(ball + e * (R - STITCH_DEPTH))      # anchor
            bars.append(Manifold.hull_points(np.vstack(cloud)))
    return Manifold.batch_boolean(bars, _op_add())


def _op_add():
    from manifold3d import OpType
    return OpType.Add


def build():
    dense = seam_curve(20000)
    groove_pts, length = resample_closed(dense, 720)
    g_t, g_side = frames(groove_pts)

    fine_n = STITCH_COUNT * 100
    fine, _ = resample_closed(dense, fine_n)
    f_t, f_side = frames(fine)
    step = length * R / fine_n                     # mm between fine samples
    idx = np.arange(STITCH_COUNT) * 100 + 50       # half-pitch offset
    trail = int(round(STITCH_TRAIL / step))
    print(f"seam length {length * R:.1f} mm, stitch pitch {length * R / STITCH_COUNT:.2f} mm")

    h = FLAT_HEIGHT / 2.0
    sphere = Manifold.sphere(R, SEGMENTS)
    slab = Manifold.cube([3 * R, 3 * R, FLAT_HEIGHT], center=True)
    body = sphere ^ slab

    hole = Manifold.cylinder(4 * R, HOLE_DIAMETER / 2, HOLE_DIAMETER / 2, 96, center=True)
    c = HOLE_CHAMFER
    cham_top = Manifold.cylinder(c + 0.01, HOLE_DIAMETER / 2, HOLE_DIAMETER / 2 + c + 0.01, 96)
    cham_top = cham_top.translate([0, 0, h - c])
    cham_bot = cham_top.mirror([0, 0, 1])
    cutters = hole + cham_top + cham_bot

    groove = seam_groove(groove_pts, g_side)
    red = stitches(fine, f_t, f_side, idx, trail)
    red = (red ^ slab) - cutters

    shell = body - cutters - groove
    white = shell - red
    single = shell + red
    return white, red, single


def to_trimesh(m):
    mesh = m.to_mesh()
    tm = trimesh.Trimesh(vertices=np.asarray(mesh.vert_properties)[:, :3],
                         faces=np.asarray(mesh.tri_verts), process=True)
    return tm


def write_3mf(path, parts):
    """parts: list of (name, trimesh, '#RRGGBB'). One object, one part each."""
    out = io.StringIO()
    out.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    out.write('<model unit="millimeter" xml:lang="en-US" '
              'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n')
    out.write(' <metadata name="Title">Baseball Hoodie Bead</metadata>\n')
    out.write(' <resources>\n  <basematerials id="1">\n')
    for name, _, color in parts:
        out.write(f'   <base name="{name}" displaycolor="{color}FF"/>\n')
    out.write('  </basematerials>\n')
    for idx, (name, tm, _) in enumerate(parts):
        oid = idx + 2
        out.write(f'  <object id="{oid}" name="{name}" type="model" pid="1" pindex="{idx}">\n'
                  '   <mesh>\n    <vertices>\n')
        for v in tm.vertices:
            out.write(f'     <vertex x="{v[0]:.5f}" y="{v[1]:.5f}" z="{v[2]:.5f}"/>\n')
        out.write('    </vertices>\n    <triangles>\n')
        for f in tm.faces:
            out.write(f'     <triangle v1="{f[0]}" v2="{f[1]}" v3="{f[2]}"/>\n')
        out.write('    </triangles>\n   </mesh>\n  </object>\n')
    top = len(parts) + 2
    out.write(f'  <object id="{top}" name="Baseball Bead" type="model">\n   <components>\n')
    for idx in range(len(parts)):
        out.write(f'    <component objectid="{idx + 2}"/>\n')
    out.write('   </components>\n  </object>\n </resources>\n')
    # sit the bead on the bed, centred on a 256 mm plate
    zmin = min(tm.bounds[0][2] for _, tm, _ in parts)
    out.write(f' <build>\n  <item objectid="{top}" '
              f'transform="1 0 0 0 1 0 0 0 1 128 128 {-zmin:.5f}"/>\n </build>\n</model>\n')

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


def main():
    white, red, single = build()
    lift = FLAT_HEIGHT / 2.0  # put the flat bottom on z = 0
    tw, tr, ts = (to_trimesh(m.translate([0, 0, lift])) for m in (white, red, single))
    for name, tm in (('white ball', tw), ('red stitches', tr), ('single color', ts)):
        print(f"{name:13s} watertight={tm.is_watertight} tris={len(tm.faces)} "
              f"vol={tm.volume:.1f} mm^3 bodies={len(tm.split(only_watertight=False))}")
    tw.export(os.path.join(OUT_DIR, 'baseball_bead_white_ball.stl'))
    tr.export(os.path.join(OUT_DIR, 'baseball_bead_red_stitches.stl'))
    ts.export(os.path.join(OUT_DIR, 'baseball_bead_single_color.stl'))
    write_3mf(os.path.join(OUT_DIR, 'baseball_bead.3mf'),
              [('White Ball', tw, '#FFFFFF'), ('Red Stitches', tr, '#C8102E')])
    print("done ->", OUT_DIR)


if __name__ == '__main__':
    main()
