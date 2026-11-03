# -*- coding: utf-8 -*-
import json, csv, os

def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

our_loss = load_json(r"D:\VScode\VScodeProjects\Fond\output\eval_baseline.json")
base_loss = load_json(r"D:\VScode\VScodeProjects\Fond\output\baseline_eval.json")
quality = load_json(r"D:\VScode\VScodeProjects\Fond\output\quality_metrics.json")
ablation_quality = load_json(r"D:\VScode\VScodeProjects\Fond\output\ablation_quality_metrics.json")

ablation_data = {
    "Full Method (Ours)": {"loss": our_loss, "quality": quality, "type": "our"},
    "- w/o two-stage LR": {"loss": load_json(r"D:\VScode\VScodeProjects\Fond\output\ablation_no_two_stage_lr.json"), "quality": ablation_quality.get("no_two_stage_lr", {}), "type": "ablation"},
    "- w/o best-of-3": {"loss": load_json(r"D:\VScode\VScodeProjects\Fond\output\ablation_no_best_of_3.json"), "quality": ablation_quality.get("no_best_of_3", {}), "type": "ablation"},
    "- w/o structure_reg": {"loss": load_json(r"D:\VScode\VScodeProjects\Fond\output\ablation_no_structure_reg.json"), "quality": ablation_quality.get("no_structure_reg", {}), "type": "ablation"},
    "- w/o VGG loss": {"loss": load_json(r"D:\VScode\VScodeProjects\Fond\output\ablation_no_vgg_loss.json"), "quality": ablation_quality.get("no_vgg_loss", {}), "type": "ablation"},
    "- w/o anchor gradient scale": {"loss": load_json(r"D:\VScode\VScodeProjects\Fond\output\ablation_no_anchor_scale.json"), "quality": ablation_quality.get("no_anchor_scale", {}), "type": "ablation"},
    "Baseline (FontCLIP)": {"loss": base_loss, "quality": quality, "type": "baseline"},
}

out = r"D:\VScode\VScodeProjects\Fond\output\ablation_results.csv"
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["Config", "Letter", "Final Loss", "Time (s)", "RMSE↓", "PSNR↑", "SSIM↑"])
    
    for model_name, data in ablation_data.items():
        loss_data = data["loss"]
        q_data = data["quality"]
        q_type = data["type"]
        
        if not loss_data:
            w.writerow([model_name, "—", "TBD", "TBD", "—", "—", "—"])
            continue
        
        per_letter_q = {}
        if q_type == "our":
            for r in q_data:
                per_letter_q[r["letter"]] = r
        elif q_type == "baseline":
            for r in q_data:
                per_letter_q[r["letter"]] = r
        elif q_type == "ablation" and "per_letter" in q_data:
            for r in q_data["per_letter"]:
                per_letter_q[r["letter"]] = r
        
        for r in loss_data:
            letter = r["letter"]
            q = per_letter_q.get(letter, {})
            if q_type == "our":
                rmse = q.get("our_rmse", "")
                psnr = q.get("our_psnr", "")
                ssim = q.get("our_ssim", "")
            elif q_type == "baseline":
                rmse = q.get("base_rmse", "")
                psnr = q.get("base_psnr", "")
                ssim = q.get("base_ssim", "")
            else:
                rmse = q.get("rmse", "")
                psnr = q.get("psnr", "")
                ssim = q.get("ssim", "")
            w.writerow([model_name, letter, round(r["final_loss"], 6), round(r["time"], 1), rmse, psnr, ssim])
        
        avg_loss = sum(r["final_loss"] for r in loss_data) / len(loss_data)
        avg_time = sum(r["time"] for r in loss_data) / len(loss_data)
        
        if q_type == "our":
            avg_rmse = sum(q.get("our_rmse", 0) for q in q_data) / len(q_data)
            avg_psnr = sum(q.get("our_psnr", 0) for q in q_data) / len(q_data)
            avg_ssim = sum(q.get("our_ssim", 0) for q in q_data) / len(q_data)
        elif q_type == "baseline":
            avg_rmse = sum(q.get("base_rmse", 0) for q in q_data) / len(q_data)
            avg_psnr = sum(q.get("base_psnr", 0) for q in q_data) / len(q_data)
            avg_ssim = sum(q.get("base_ssim", 0) for q in q_data) / len(q_data)
        else:
            avg_rmse = q_data.get("avg_rmse", "")
            avg_psnr = q_data.get("avg_psnr", "")
            avg_ssim = q_data.get("avg_ssim", "")
        
        w.writerow([model_name, "Average", round(avg_loss, 6), round(avg_time, 1), avg_rmse, avg_psnr, avg_ssim])
        w.writerow([])

print(f"Saved to {out}")
for name, data in ablation_data.items():
    if data["loss"]:
        avg = sum(r["final_loss"] for r in data["loss"]) / len(data["loss"])
        print(f"{name}: avg_loss={avg:.6f}")
