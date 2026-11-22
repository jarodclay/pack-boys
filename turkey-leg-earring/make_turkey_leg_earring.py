#!/usr/bin/env python3
"""
Turkey leg earring generator.

Builds a cartoon turkey drumstick earring (brown meat + white bone with a
hanging loop at the top) and exports print-ready files:

  turkey_leg_earring_pair.3mf     two earrings, each one object with two parts
                                  (Bone -> filament 1, Meat -> filament 2), for
                                  multi-color printers such as the Snapmaker U1
  turkey_leg_bone_white.stl       just the white bone + loop
  turkey_leg_meat_brown.stl       just the brown meat (load with the bone STL)
  turkey_leg_single_color.stl     one solid piece, for single-color printing

The back is flat so it prints flat on the bed with no supports. The bone runs
on into the meat as a hidden tab (with brown above and below it), so the two
colors are locked together rather than only touching side by side.

Requires: pip install numpy scipy scikit-image trimesh manifold3d
Run:      python3 make_turkey_leg_earring.py
All sizes are in millimetres; tweak the PARAMETERS block and re-run.
"""
import io
import os
import zipfile

import numpy as np
import trimesh
from manifold3d import Manifold, Mesh
from skimage.measure import marching_cubes

# ----------------------------------------------------------------- PARAMETERS
# Meat (drumstick bulb + neck). Y runs up the earring, Z is thickness.
BULB_CENTER_Y = 10.2
BULB_RADIUS_X = 9.2        # half-width of the meat at its widest
BULB_RADIUS_Y = 10.4
NECK_START_Y = 16.5        # neck = rounded cone from the bulb up to the bone
NECK_START_R = 6.2
NECK_END_Y = 25.6
NECK_END_R = 3.6
BULB_THICKNESS = 0.62      # Z flattening of the bulb (0.62 * 9.2 = 5.7 mm tall)
NECK_THICKNESS = 0.92      # Z flattening at the neck end (so it covers the bone)
MEAT_LUMPS = 0.32          # height of the big soft lumps on the meat
MEAT_CRINKLE = 0.10        # height of the fine "roasted skin" texture
MEAT_LEAN = 0.7            # pushes one side of the bulb out a bit, so it looks less like clip art
NOISE_SEED = 7

# Bone
SHAFT_START_Y = 19.0       # bone starts hidden inside the meat
SHAFT_END_Y = 34.6
SHAFT_R = 2.2              # bone is 4.4 mm wide, 2.2 mm tall
KNOB_X = 2.35              # the two knuckle knobs at the bone end
KNOB_Y = 36.4
KNOB_R = 2.6

# Hanging loop (hole axis goes through the earring, front to back)
LOOP_CENTER_Y = 40.9
LOOP_OUTER_R = 3.5         # 7.0 mm outside diameter
LOOP_HOLE_R = 1.5          # 3.0 mm hole for a jump ring or ear hook
LOOP_HEIGHT = 2.3          # loop thickness
LOOP_EDGE_ROUND = 0.7

TAB_START_Y = 19.5         # hidden bone tab that locks into the meat
TAB_FLOOR_Z = 0.6          # meat left under the tab (3 layers at 0.2 mm)

VOXEL = 0.1                # marching-cubes grid size (smaller = smoother)
SIMPLIFY_TOL = 0.006       # mesh simplification tolerance (mm)

PAIR_SPACING = 26.0        # X distance between the two earrings in the 3MF
BED_CENTER = (135.0, 135.0)  # Snapmaker U1 bed is 270 x 270 mm
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
# ---------------------------------------------------------------------------


def smin(a, b, k):
    """Polynomial smooth minimum (smooth union of two distance fields)."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def sd_ellipsoid(x, y, z, rx, ry, rz):
    k0 = np.sqrt((x / rx) ** 2 + (y / ry) ** 2 + (z / rz) ** 2)
    k1 = np.sqrt((x / rx ** 2) ** 2 + (y / ry ** 2) ** 2 + (z / rz ** 2) ** 2)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def sd_round_cone_y(x, y, z, r1, r2, h):
    """Rounded cone along +Y: radius r1 at y=0, radius r2 at y=h."""
    b = (r1 - r2) / h
    a = np.sqrt(1.0 - b * b)
    qx = np.sqrt(x * x + z * z)
    qy = y
    k = -b * qx + a * qy
    d_body = a * qx + b * qy - r1
    d_bot = np.sqrt(qx * qx + qy * qy) - r1
    d_top = np.sqrt(qx * qx + (qy - h) ** 2) - r2
    return np.where(k < 0.0, d_bot, np.where(k > a * h, d_top, d_body))


def sd_capsule_y(x, y, z, y0, y1, r):
    t = np.clip(y, y0, y1)
    return np.sqrt(x * x + (y - t) ** 2 + z * z) - r


def gradient_noise(x, y, z, cell, seed):
    """Smooth 3D gradient (Perlin-style) noise, roughly in [-1, 1], with
    features about `cell` mm across. The axes are rotated so the bumps don't
    line up with the X/Y grid."""
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(32, 32, 32, 3)).astype(np.float32)
    g /= np.linalg.norm(g, axis=-1, keepdims=True)
    rot, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    px = (rot[0, 0] * x + rot[0, 1] * y + rot[0, 2] * z) / cell + 11.3
    py = (rot[1, 0] * x + rot[1, 1] * y + rot[1, 2] * z) / cell + 11.7
    pz = (rot[2, 0] * x + rot[2, 1] * y + rot[2, 2] * z) / cell + 11.1
    ix, iy, iz = np.floor(px).astype(int), np.floor(py).astype(int), np.floor(pz).astype(int)
    fx, fy, fz = px - ix, py - iy, pz - iz
    ux, uy, uz = (f * f * f * (f * (f * 6 - 15) + 10) for f in (fx, fy, fz))
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                gg = g[(ix + dx) % 32, (iy + dy) % 32, (iz + dz) % 32]
                dot = gg[..., 0] * (fx - dx) + gg[..., 1] * (fy - dy) + gg[..., 2] * (fz - dz)
                w = ((ux if dx else 1 - ux) * (uy if dy else 1 - uy) * (uz if dz else 1 - uz))
                out = out + w * dot
    return out * 1.6


def meat_sdf(x, y, z):
    # flatter at the round end, fuller at the neck so it covers the bone
    s = BULB_THICKNESS + (NECK_THICKNESS - BULB_THICKNESS) * smoothstep(9.0, NECK_END_Y, y)
    zz = z / s
    bulb = sd_ellipsoid(x - MEAT_LEAN * smoothstep(-2.0, 12.0, y) * (1 - smoothstep(12.0, 22.0, y)),
                        y - BULB_CENTER_Y, zz, BULB_RADIUS_X, BULB_RADIUS_Y, BULB_RADIUS_X)
    neck = sd_round_cone_y(x, y - NECK_START_Y, zz, NECK_START_R, NECK_END_R,
                           NECK_END_Y - NECK_START_Y)
    d = smin(bulb, neck, 4.5)
    # soft lumps plus fine crinkles, like roasted skin
    bumps = (MEAT_LUMPS * gradient_noise(x, y, z, 3.4, NOISE_SEED)
             + MEAT_CRINKLE * gradient_noise(x, y, z, 1.2, NOISE_SEED + 1))
    fade = 1.0 - smoothstep(NECK_END_Y - 2.0, NECK_END_Y + 2.0, y)   # keep the bone exit clean
    return d - bumps * fade


def bone_sdf(x, y, z):
    shaft = sd_capsule_y(x, y, z, SHAFT_START_Y, SHAFT_END_Y, SHAFT_R)
    knob_l = np.sqrt((x + KNOB_X) ** 2 + (y - KNOB_Y) ** 2 + z * z) - KNOB_R
    knob_r = np.sqrt((x - KNOB_X) ** 2 + (y - KNOB_Y) ** 2 + z * z) - KNOB_R
    d = smin(shaft, smin(knob_l, knob_r, 0.6), 1.8)

    # flat ring with rounded edges
    rr = np.sqrt(x * x + (y - LOOP_CENTER_Y) ** 2)
    mid = 0.5 * (LOOP_OUTER_R + LOOP_HOLE_R)
    half_w = 0.5 * (LOOP_OUTER_R - LOOP_HOLE_R)
    d2 = np.abs(rr - mid) - half_w
    e = LOOP_EDGE_ROUND
    wx = d2 + e
    wy = np.abs(z) - LOOP_HEIGHT + e
    ring = (np.minimum(np.maximum(wx, wy), 0.0)
            + np.sqrt(np.maximum(wx, 0.0) ** 2 + np.maximum(wy, 0.0) ** 2) - e)
    d = smin(d, ring, 0.9)
    # re-open the hole in case the fillet crept into it
    hole = rr - LOOP_HOLE_R
    return smax(d, -hole, 0.4)


def field_to_manifold(fn, lo, hi):
    """Sample fn on a grid and turn its zero level set into a Manifold."""
    # offset the grid so the flat cuts made later (z=0, the tab planes) never
    # land exactly on a row of grid vertices, which would leave slivers
    lo = tuple(v + 0.371 * VOXEL for v in lo)
    xs = np.arange(lo[0], hi[0] + VOXEL, VOXEL)
    ys = np.arange(lo[1], hi[1] + VOXEL, VOXEL)
    zs = np.arange(lo[2], hi[2] + VOXEL, VOXEL)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    f = fn(X, Y, Z).astype(np.float32)
    # close the volume on every side of the grid
    f[0, :, :] = f[-1, :, :] = 1.0
    f[:, 0, :] = f[:, -1, :] = 1.0
    f[:, :, 0] = f[:, :, -1] = 1.0
    verts, faces, _, _ = marching_cubes(f, level=0.0, spacing=(VOXEL,) * 3)
    verts += np.array(lo)
    tm = trimesh.Trimesh(verts, faces, process=True)
    if tm.volume < 0:
        tm.invert()
    m = Manifold(Mesh(vert_properties=np.asarray(tm.vertices, np.float32),
                      tri_verts=np.asarray(tm.faces, np.uint32)))
    if m.status().name != 'NoError':
        raise RuntimeError(f"bad mesh: {m.status()}")
    return m


def box(x0, x1, y0, y1, z0, z1):
    return Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def build():
    # sample a little below z=0, then cut flat at z=0 for the bed side
    meat_full = field_to_manifold(meat_sdf, (-11.0, -2.0, -0.6), (11.0, 31.0, 7.0))
    bone_full = field_to_manifold(bone_sdf, (-6.5, 17.0, -0.6), (6.5, 45.5, 3.5))

    top = box(-50, 50, -50, 100, 0.0, 50)
    meat = meat_full ^ top
    bone = bone_full ^ top

    tab = bone ^ box(-10, 10, TAB_START_Y, NECK_END_Y + NECK_END_R + 1.0, TAB_FLOOR_Z, 50)
    bone_part = (bone - meat) + tab
    meat_part = meat - bone_part
    single = meat + bone

    parts = [p.simplify(SIMPLIFY_TOL) for p in (bone_part, meat_part, single)]
    # put the bottom of the meat at y=0 and the flat back on z=0
    y0 = parts[2].bounding_box()[1]
    return [p.translate([0, -y0, 0]) for p in parts]


def to_trimesh(m):
    mesh = m.to_mesh()
    return trimesh.Trimesh(vertices=np.asarray(mesh.vert_properties)[:, :3],
                           faces=np.asarray(mesh.tri_verts), process=True)


def mesh_xml(oid, name, pindex, tm):
    out = io.StringIO()
    out.write(f'  <object id="{oid}" name="{name}" type="model" pid="1" pindex="{pindex}">\n'
              '   <mesh>\n    <vertices>\n')
    for v in tm.vertices:
        out.write(f'     <vertex x="{v[0]:.4f}" y="{v[1]:.4f}" z="{v[2]:.4f}"/>\n')
    out.write('    </vertices>\n    <triangles>\n')
    for f in tm.faces:
        out.write(f'     <triangle v1="{f[0]}" v2="{f[1]}" v3="{f[2]}"/>\n')
    out.write('    </triangles>\n   </mesh>\n  </object>\n')
    return out.getvalue()


def write_3mf(path, parts, copies):
    """parts: list of (name, trimesh, '#RRGGBB', filament). Each copy is one
    object made of all the parts; Metadata/model_settings.config (read by
    Orca-based slicers such as Snapmaker Orca and Bambu Studio) names the parts
    and assigns each one its filament."""
    model = io.StringIO()
    model.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    model.write('<model unit="millimeter" xml:lang="en-US" '
                'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n')
    model.write(' <metadata name="Title">Turkey Leg Earring</metadata>\n')
    model.write(' <resources>\n  <basematerials id="1">\n')
    for name, _, color, _ in parts:
        model.write(f'   <base name="{name}" displaycolor="{color}FF"/>\n')
    model.write('  </basematerials>\n')
    for i, (name, tm, _, _) in enumerate(parts):
        model.write(mesh_xml(i + 2, name, i, tm))

    xmin = min(tm.bounds[0][0] for _, tm, _, _ in parts)
    xmax = max(tm.bounds[1][0] for _, tm, _, _ in parts)
    ymin = min(tm.bounds[0][1] for _, tm, _, _ in parts)
    ymax = max(tm.bounds[1][1] for _, tm, _, _ in parts)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2

    config = io.StringIO()
    config.write('<?xml version="1.0" encoding="UTF-8"?>\n<config>\n')
    build = io.StringIO()
    first_obj = len(parts) + 2
    for c in range(copies):
        oid = first_obj + c
        label = f"Turkey Leg Earring {c + 1}"
        model.write(f'  <object id="{oid}" name="{label}" type="model">\n   <components>\n')
        for i in range(len(parts)):
            model.write(f'    <component objectid="{i + 2}"/>\n')
        model.write('   </components>\n  </object>\n')
        tx = BED_CENTER[0] + (c - (copies - 1) / 2) * PAIR_SPACING - cx
        ty = BED_CENTER[1] - cy
        build.write(f'  <item objectid="{oid}" transform="1 0 0 0 1 0 0 0 1 '
                    f'{tx:.4f} {ty:.4f} 0" printable="1"/>\n')
        config.write(f'  <object id="{oid}">\n'
                     f'    <metadata key="name" value="{label}"/>\n'
                     f'    <metadata key="extruder" value="1"/>\n')
        for i, (name, _, _, filament) in enumerate(parts):
            config.write(f'    <part id="{i + 2}" subtype="normal_part">\n'
                         f'      <metadata key="name" value="{name}"/>\n'
                         f'      <metadata key="extruder" value="{filament}"/>\n'
                         f'    </part>\n')
        config.write('  </object>\n')
    model.write(' </resources>\n <build>\n')
    model.write(build.getvalue())
    model.write(' </build>\n</model>\n')
    config.write('</config>\n')

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
        z.writestr('Metadata/model_settings.config', config.getvalue())


def main():
    bone, meat, single = build()
    tb, tm_, ts = (to_trimesh(m) for m in (bone, meat, single))
    for name, t in (('bone', tb), ('meat', tm_), ('single color', ts)):
        ext = t.bounds[1] - t.bounds[0]
        print(f"{name:13s} watertight={t.is_watertight} tris={len(t.faces)} "
              f"vol={t.volume:.0f} mm^3 bodies={len(t.split(only_watertight=False))} "
              f"size={ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f} mm")
    print(f"weight in PLA at 100% infill: {ts.volume * 1.24e-3:.2f} g")
    tb.export(os.path.join(OUT_DIR, 'turkey_leg_bone_white.stl'))
    tm_.export(os.path.join(OUT_DIR, 'turkey_leg_meat_brown.stl'))
    ts.export(os.path.join(OUT_DIR, 'turkey_leg_single_color.stl'))
    write_3mf(os.path.join(OUT_DIR, 'turkey_leg_earring_pair.3mf'),
              [('Bone (white)', tb, '#F4F1EA', 1), ('Meat (brown)', tm_, '#7A4321', 2)],
              copies=2)
    print("done ->", OUT_DIR)


if __name__ == '__main__':
    main()
