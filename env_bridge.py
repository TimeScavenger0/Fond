# -*- coding: utf-8 -*-
"""跨环境调度桥（fontclip 主进程 -> fontdiffuser 子进程）。

FontDiffuser 依赖（diffusers/transformers/pygame）与 fontclip 环境冲突，
统一界面跑在 fontclip，通过 subprocess 调用 fontdiffuser 解释器执行
fd_batch_generate.py 完成批量扩散生成。
"""
import os
import sys
import json
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
FD_PYTHON = r"D:\Anaconda-3\envs\fontdiffuser\python.exe"
BATCH_SCRIPT = os.path.join(ROOT, "fd_batch_generate.py")


def run_batch(config):
    """运行 FontDiffuser 批量生成。生成器，逐行 yield 进度字典。

    输入 config 与 fd_batch_generate.py 的 JSON 字段一致。
    yield: {"ratio": float, "status": str}
    最后一次 yield: {"ratio": 1.0, "status": ..., "done": True, "manifest": dict}
    """
    cfg_path = os.path.join(config["out_dir"], "_bridge_config.json")
    os.makedirs(config["out_dir"], exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    assert os.path.exists(FD_PYTHON), FD_PYTHON
    proc = subprocess.Popen(
        [FD_PYTHON, "-u", BATCH_SCRIPT, cfg_path],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        encoding="utf-8", errors="replace", bufsize=1,
    )

    n_total = max(len(config.get("chars", [])), 1)
    manifest = None
    for raw in proc.stdout:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("PROGRESS"):
            parts = line.split()
            try:
                idx = int(parts[1].split("/")[0])
            except (IndexError, ValueError):
                idx = 0
            ratio = min(idx / n_total, 0.99)
            yield {"ratio": ratio, "status": line}
        elif line.startswith("BATCH_DONE"):
            yield {"ratio": 0.99, "status": line}
        elif line.startswith("MODEL_LOADED"):
            yield {"ratio": 0.0, "status": line}

    proc.wait()
    mpath = os.path.join(config["out_dir"], "manifest.json")
    if os.path.exists(mpath):
        with open(mpath, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    yield {"ratio": 1.0, "status": "batch finished (exit=%d)" % proc.returncode,
           "done": True, "manifest": manifest, "returncode": proc.returncode}
