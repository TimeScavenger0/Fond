# -*- coding: utf-8 -*-
"""Compute new metrics for brush local editing."""
import os, sys, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from PIL import Image, ImageDraw
import pydiffvg
import xml.etree.ElementTree as ET

letters = ['A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'S', 'T']

def create_brush_mask(size=200, cx=100, cy=120, radius=40):
    mask = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius], fill=255)
    return np.array(mask) > 127

def get_control_points_from_svg(svg_path):
    """Extract control points from SVG file."""
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_path)
    all_points = []
    for s in shapes:
        if hasattr(s, 'points'):
            pts = s.points.detach().cpu().numpy()
            all_points.append(pts)
    return np.vstack(all_points) if all_points else np.array([])

def compute_brush_metrics():
    brush_mask = create_brush_mask()
    H, W = brush_mask.shape
    
    results = []
    for letter in letters:
        init_svg = f"output/brush_test_{letter}/svg-init/init.svg"
        final_svg = f"output/brush_test_{letter}/output-svg/output.svg"
        
        if not os.path.exists(init_svg) or not os.path.exists(final_svg):
            print(f"{letter}: missing SVG files")
            continue
        
        init_pts = get_control_points_from_svg(init_svg)
        final_pts = get_control_points_from_svg(final_svg)
        
        if len(init_pts) != len(final_pts):
            print(f"{letter}: point count mismatch {len(init_pts)} vs {len(final_pts)}")
            continue
        
        # Determine which points are inside/outside brush mask
        inside_mask = []
        outside_mask = []
        for i, (x, y) in enumerate(init_pts):
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < H and 0 <= xi < W and brush_mask[yi, xi]:
                inside_mask.append(i)
            else:
                outside_mask.append(i)
        
        # Compute displacement for each point
        displacements = np.linalg.norm(final_pts - init_pts, axis=1)
        
        # Inside points (should move)
        inside_disp = displacements[inside_mask] if inside_mask else np.array([0])
        # Outside points (should be frozen)
        outside_disp = displacements[outside_mask] if outside_mask else np.array([0])
        
        row = {
            "letter": letter,
            "total_points": len(init_pts),
            "inside_points": len(inside_mask),
            "outside_points": len(outside_mask),
            "inside_avg_disp": round(float(np.mean(inside_disp)), 4),
            "outside_avg_disp": round(float(np.mean(outside_disp)), 4),
            "inside_max_disp": round(float(np.max(inside_disp)), 4),
            "outside_max_disp": round(float(np.max(outside_disp)), 4),
        }
        results.append(row)
        print(f"{letter}: inside_disp={row['inside_avg_disp']:.2f}, outside_disp={row['outside_avg_disp']:.4f}")
    
    # Compute averages
    avg_inside = np.mean([r["inside_avg_disp"] for r in results])
    avg_outside = np.mean([r["outside_avg_disp"] for r in results])
    avg_frozen_ratio = avg_outside / avg_inside if avg_inside > 0 else 0
    
    summary = {
        "per_letter": results,
        "avg_inside_disp": round(float(avg_inside), 4),
        "avg_outside_disp": round(float(avg_outside), 4),
        "frozen_ratio": round(float(avg_frozen_ratio), 4),
        "note": "inside_disp = average displacement of points inside brush mask; outside_disp = average displacement of points outside brush mask (should be near 0); frozen_ratio = outside/inside (lower = better freezing)"
    }
    
    with open("output/brush_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Summary ===")
    print(f"Average inside displacement: {avg_inside:.4f}")
    print(f"Average outside displacement: {avg_outside:.4f}")
    print(f"Frozen ratio (outside/inside): {avg_frozen_ratio:.4f}")
    print(f"Outside points are {avg_frozen_ratio*100:.1f}% as mobile as inside points")

if __name__ == "__main__":
    compute_brush_metrics()
