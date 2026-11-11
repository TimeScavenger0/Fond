"""
构建云端训练代码补丁包 cloud_code_patch.zip（不含字体/数据，仅代码与脚本）。
在云端 Fond 根目录解压，覆盖 #60 改动的三个核心文件并新增训练入口与 cloud_deploy 脚本。
用 zipfile 保持目录结构与 LF 换行，不改变文件内容。
"""
import os
import zipfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloud_code_patch.zip")

FILES = [
    "train_cn_mixed.py",
    "models/aux_modules.py",
    "trains/config_fine_tune.py",
    "trains/trainer.py",
    "utils/transform_image.py",
    "utils/initialize_font_data.py",
    "evals/evaluate_tools.py",
    "cloud_deploy/download_checkpoint.py",
    "cloud_deploy/setup_cloud.sh",
    "cloud_deploy/run_train.sh",
    "cloud_deploy/requirements-cloud.txt",
]

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for rel in FILES:
        full = os.path.join(BASE, rel)
        assert os.path.exists(full), full
        z.write(full, rel)

with zipfile.ZipFile(OUT, "r") as z:
    bad = z.testzip()
    assert bad is None, bad
    print("entries:", len(z.namelist()))
    for n in z.namelist():
        print(" ", n)
print("code patch size: %.2f MB" % (os.path.getsize(OUT) / 1048576))
print("CODE PATCH BUILD OK")
