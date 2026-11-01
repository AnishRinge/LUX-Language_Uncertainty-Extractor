from pathlib import Path
p=Path('reports/_make_pdf.py')
s=p.read_text(encoding='utf-8-sig')
s=s.replace("re.split(r'(?m)^## (\\d+)\\. (.+)$',src)","re.split(r'(?m)^## (\\d+)\\. (.*)$',src)")
p.write_text(s,encoding='utf-8')
