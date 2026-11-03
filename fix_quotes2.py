# -*- coding: utf-8 -*-
from docx import Document
doc = Document(r"D:\VScode\VScodeProjects\Fond\FontCLIP画笔优化开发日志.docx")
for p in doc.paragraphs:
    for run in p.runs:
        if '"' in run.text:
            # Replace remaining ASCII double quotes with Chinese closing quote
            run.text = run.text.replace('"', '”')
doc.save(r"D:\VScode\VScodeProjects\Fond\FontCLIP画笔优化开发日志.docx")
print("fixed")
