from docx import Document
doc = Document(r"D:\VScode\VScodeProjects\Fond\FontCLIP融合系统评估指标体系.docx")
for p in doc.paragraphs:
    if p.text.strip():
        print(p.text)
for table in doc.tables:
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        print(" | ".join(cells))
