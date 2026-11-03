# -*- coding: utf-8 -*-
import json, csv, os

def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

our_loss = load_json(r"D:\VScode\VScodeProjects\Fond\output\eval_baseline.json")
base_loss = load_json(r"D:\VScode\VScodeProjects\Fond\output\baseline_eval.json")
clip = load_json(r"D:\VScode\VScodeProjects\Fond\output\clip_score_results.json")
quality = load_json(r"D:\VScode\VScodeProjects\Fond\output\quality_metrics.json")

clip_map = {r["letter"]: r for r in clip}
qual_map = {r["letter"]: r for r in quality}

out = r"D:\VScode\VScodeProjects\Fond\output\baseline_results.csv"
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["Model", "Letter", "Final Loss", "Time (s)", "CLIP Score Delta", "RMSE↓", "PSNR↑", "SSIM↑"])
    
    for r in our_loss:
        letter = r["letter"]
        c = clip_map.get(letter, {})
        q = qual_map.get(letter, {})
        w.writerow(["Our Method", letter, round(r["final_loss"], 6), round(r["time"], 1),
                    c.get("our_delta", ""), q.get("our_rmse", ""), q.get("our_psnr", ""), q.get("our_ssim", "")])
    avg_loss = sum(r["final_loss"] for r in our_loss) / len(our_loss)
    avg_time = sum(r["time"] for r in our_loss) / len(our_loss)
    avg_clip = sum(c.get("our_delta", 0) for c in clip) / len(clip)
    avg_rmse = sum(q.get("our_rmse", 0) for q in quality) / len(quality)
    avg_psnr = sum(q.get("our_psnr", 0) for q in quality) / len(quality)
    avg_ssim = sum(q.get("our_ssim", 0) for q in quality) / len(quality)
    w.writerow(["Our Method", "Average", round(avg_loss, 6), round(avg_time, 1),
                round(avg_clip, 4), round(avg_rmse, 2), round(avg_psnr, 2), round(avg_ssim, 4)])
    
    for r in base_loss:
        letter = r["letter"]
        c = clip_map.get(letter, {})
        q = qual_map.get(letter, {})
        w.writerow(["FontCLIP Baseline", letter, round(r["final_loss"], 6), round(r["time"], 1),
                    c.get("base_delta", ""), q.get("base_rmse", ""), q.get("base_psnr", ""), q.get("base_ssim", "")])
    avg_loss_b = sum(r["final_loss"] for r in base_loss) / len(base_loss)
    avg_time_b = sum(r["time"] for r in base_loss) / len(base_loss)
    avg_clip_b = sum(c.get("base_delta", 0) for c in clip) / len(clip)
    avg_rmse_b = sum(q.get("base_rmse", 0) for q in quality) / len(quality)
    avg_psnr_b = sum(q.get("base_psnr", 0) for q in quality) / len(quality)
    avg_ssim_b = sum(q.get("base_ssim", 0) for q in quality) / len(quality)
    w.writerow(["FontCLIP Baseline", "Average", round(avg_loss_b, 6), round(avg_time_b, 1),
                round(avg_clip_b, 4), round(avg_rmse_b, 2), round(avg_psnr_b, 2), round(avg_ssim_b, 4)])
    
    w.writerow(["MSD-Font (论文引用)", "Average", "—", "—", "—", 0.2439, 12.61, 0.7110])
    w.writerow([])
    w.writerow(["备注", "相同 prompt: 'italic, slanted font'，相同步数: 300，相同初始字体: Arimo-Regular，10个大写字母(A/B/C/E/H/K/M/O/S/T)。Our Method vs FontCLIP Baseline 为本环境实测；MSD-Font 为 CVPR 2024 论文引用数字。消融实验结果见 ablation_results.csv"])

print(f"Saved to {out}")
