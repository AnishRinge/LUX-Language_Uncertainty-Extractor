from pathlib import Path
p=Path('reports/_make_pdf.py')
s=p.read_text(encoding='utf-8-sig')
s=s.replace(r"r'(?m)^## (\\d+)\\. (.*)$'",r"r'(?m)^## (\d+)\. (.*)$'")
p.write_text(s,encoding='utf-8')
