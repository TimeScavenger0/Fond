# -*- coding: utf-8 -*-
"""验证中文训练数据集的完整性与一致性。"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "dataset")
IMG_DIR = os.path.join(OUT, "cn_images")

with open(os.path.join(OUT, "cn_font_to_attribute_values.json"),
          "r", encoding="utf-8") as f:
    all_attr = json.load(f)

splits = {}
for name in ("train", "validation", "test"):
    with open(os.path.join(OUT, name + "_font_to_attribute_values.json"),
              "r", encoding="utf-8") as f:
        splits[name] = set(json.load(f).keys())

imgs = {os.path.splitext(p)[0]
        for p in os.listdir(IMG_DIR) if p.endswith(".png")}
keys = set(all_attr.keys())

print("total fonts in all json:", len(keys))
print("images on disk:", len(imgs))
print("split sizes:", {k: len(v) for k, v in splits.items()})
print("sum of splits:", sum(len(v) for v in splits.values()))
print("splits disjoint:", splits["train"].isdisjoint(splits["validation"])
      and splits["train"].isdisjoint(splits["test"])
      and splits["validation"].isdisjoint(splits["test"]))
print("union of splits == all:", set().union(*splits.values()) == keys)
print("images match labels:", imgs == keys)
print("all have 30 attrs:", all(len(v) == 30 for v in all_attr.values()))
blank_attrs = {"bad", "capitals", "cursive", "display",
               "fresh", "friendly", "gentle"}
print("no blank attrs present:",
      all(blank_attrs.isdisjoint(v) for v in all_attr.values()))
