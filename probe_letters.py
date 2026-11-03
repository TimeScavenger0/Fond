"""Probe path structure of several letters to see whether the A-specific stroke rules generalize."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import pydiffvg
from optimizer.optimize import Config

FONT = "gwfonts/Arimo-Regular.ttf"
LETTERS = ["A", "B", "O", "X", "K", "R", "H", "I", "i"]


def analyze(letter):
    svg = f"tmp_{letter}.svg"
    Config.create_svg_from_font(FONT, letter, output_path=svg, char_size=150, size=200)
    w, h, shapes, groups = pydiffvg.svg_to_scene(svg)
    os.remove(svg)
    print(f"\n=== letter '{letter}': {len(shapes)} path(s) ===")
    for i, s in enumerate(shapes):
        p = s.points
        xs, ys = p[:, 0], p[:, 1]
        wd = (xs.max() - xs.min()).item()
        ht = (ys.max() - ys.min()).item()
        # simple heuristic: counter candidates are small, interior, not the bounding box
        print(
            f"  shape {i}: pts={p.shape[0]:3d}  x[{xs.min():5.1f},{xs.max():5.1f}] "
            f"y[{ys.min():5.1f},{ys.max():5.1f}]  w={wd:4.1f} h={ht:4.1f}"
        )


for L in LETTERS:
    analyze(L)
