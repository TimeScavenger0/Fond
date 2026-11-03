# -*- coding: utf-8 -*-
import json, csv, os

def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

brush_metrics = load_json(r"D:\VScode\VScodeProjects\Fond\output\brush_metrics.json")
brush_loss = load_json(r"D:\VScode\VScodeProjects\Fond\output\brush_experiment.json")

out = r"D:\VScode\VScodeProjects\Fond\output\brush_metrics_results.csv"
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["Letter", "Final Loss", "Inside Points", "Outside Points", 
                "Inside Avg Displacement", "Outside Avg Displacement", "Frozen Ratio"])
    
    for r in brush_loss:
        letter = r["letter"]
        # Find matching metrics
        m = next((x for x in brush_metrics.get("per_letter", []) if x["letter"] == letter), {})
        w.writerow([
            letter, 
            round(r["final_loss"], 6),
            m.get("inside_points", ""),
            m.get("outside_points", ""),
            m.get("inside_avg_disp", ""),
            m.get("outside_avg_disp", ""),
            round(m.get("outside_avg_disp", 0) / m.get("inside_avg_disp", 1), 4) if m.get("inside_avg_disp", 0) > 0 else ""
        ])
    
    # Summary row
    avg_loss = sum(r["final_loss"] for r in brush_loss) / len(brush_loss)
    w.writerow([
        "Average",
        round(avg_loss, 6),
        "",
        "",
        brush_metrics.get("avg_inside_disp", ""),
        brush_metrics.get("avg_outside_disp", ""),
        brush_metrics.get("frozen_ratio", "")
    ])
    
    w.writerow([])
    w.writerow(["说明", "圈内位移 = 画笔圈内控制点平均位移（越大说明局部编辑越强）"])
    w.writerow(["", "圈外位移 = 画笔圈外控制点平均位移（越小说明冻结效果越好，理想为0）"])
    w.writerow(["", "冻结比率 = 圈外位移/圈内位移（越低越好，0表示圈外完全冻结）"])
    w.writerow(["", "Baseline（FontCLIP）没有画笔功能，所有控制点全局移动，圈外位移=圈内位移，冻结比率=1.0"])

print(f"Saved to {out}")
print(f"Inside displacement: {brush_metrics.get('avg_inside_disp', 'N/A')}")
print(f"Outside displacement: {brush_metrics.get('avg_outside_disp', 'N/A')}")
print(f"Frozen ratio: {brush_metrics.get('frozen_ratio', 'N/A')}")
