# Thanksgiving Turkey Earring

A flat, front-facing turkey earring for Thanksgiving. It has a red, orange and yellow tail fan, a brown body, an orange beak and feet, a red wattle, black eyes, and a loop at the top for your jump ring.

![preview](preview.png)

**Size:** 39.4 mm wide × 39.6 mm tall (about 1.5 in), including the loop. Each earring weighs about 2.3 g.
**Thickness:** The back is flat. On the front, each row of feathers steps up 0.4 mm, from 1.6 mm at the outer red feathers to 2.8 mm at the body. The beak and wattle stand 0.4 mm proud of the head, at 3.2 mm. The feathers and wings have shallow engraved lines.
**Loop:** The hole is 2.0 mm with a 1.6 mm wall. It fits standard jump rings.

## Colors

The earring uses 4 colors and prints in one job on the Snapmaker U1. The eyes are a 5th color. You print them separately and glue them in.

| Tool | Color | Part name in the slicer | What it prints |
|---|---|---|---|
| 1 | Brown | `1 Brown - body` | body and head |
| 2 | Red | `2 Red - outer feathers, loop, wattle` | outer feathers, loop, wattle |
| 3 | Orange | `3 Orange - middle feathers, beak, feet` | middle feathers, beak, feet |
| 4 | Yellow | `4 Yellow - inner feathers` | inner feathers |
| separate print | Black | `turkey_eyes_black.stl` | 8 eye plugs (a pair of earrings needs 4) |

## Which file to print

| File | How |
|---|---|
| `turkey_earring.3mf` | Open it in Snapmaker Orca. You get a pair of earrings, each one object with 4 parts. In the object list, set each part's filament to match the table above. |
| `turkey_earring_brown.stl`, `_red.stl`, `_orange.stl`, `_yellow.stl` | If the 3MF doesn't come in as 4 parts, use these instead. Select all 4 files and import them together. When the slicer asks whether to load them as a single object with multiple parts, choose **Yes**. Set each part's filament, then add a second copy for the other earring. |
| `turkey_eyes_black.stl` | Print in black in a separate job. |

## Print settings

- Print it flat side down, the way it loads. No supports needed.
- Use a 0.2 mm layer height. All the step heights are multiples of 0.4 mm, so each color lands exactly on a layer. A 0.1 mm layer height also works and gives finer edges.
- PLA works well.
- Keep the U1's default prime and wipe settings for multi-color prints.
- No brim is needed for the earrings. For the eye plugs, add a 2–3 mm brim. They are tiny (2.6 mm across) and stick better with one.
- Optional: in the slicer, mirror one earring left to right so the wattles hang to opposite sides, making a matched pair.

## Adding the eyes and hardware

1. Snap the eye plugs off the brim. Press one into each eye hole. The holes go all the way through, so the eyes show on both sides.
2. If a plug is loose, add a tiny drop of super glue (CA) with a toothpick. If a plug is too tight, roll it on fine sandpaper. You can also change `EYE_CLEARANCE` in the script and rebuild.
3. Open a jump ring (6–8 mm works well) and put it through the loop. Hang it on an ear wire, then close the jump ring.

## Changing the size

- **Scale in the slicer:** Scale X and Y only. Keep Z at 100% so the layers still line up. Scale the eye plugs by the same percentage. At 85% the earring is about 33.5 mm, the loop hole is 1.7 mm, and the toes are 1.0 mm wide. Don't go much smaller than that.
- **Change one detail:** Edit the numbers at the top of `make_turkey_earring.py`, such as `LOOP_HOLE`, `EYE_HOLE`, `EYE_CLEARANCE` or the layer heights. Then rebuild:

  ```bash
  pip install numpy trimesh manifold3d
  python3 make_turkey_earring.py
  ```
