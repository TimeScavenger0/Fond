# -*- coding: utf-8 -*-
from docx import Document
doc = Document(r"D:\VScode\VScodeProjects\Fond\FontCLIP画笔优化开发日志.docx")
for p in doc.paragraphs:
    for run in p.runs:
        if '"' in run.text:
            run.text = run.text.replace('STATE["', 'STATE[“').replace('"]=True', '”]=True').replace('"]=输出', '”]=输出')
doc.save(r"D:\VScode\VScodeProjects\Fond\FontCLIP画笔优化开发日志.docx")
print("fixed")
