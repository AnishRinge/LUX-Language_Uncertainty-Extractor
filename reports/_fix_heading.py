from pathlib import Path
p=Path('reports/LUX_Project_Report_10_Pages.md')
s=p.read_text(encoding='utf-8-sig')
s=s.replace('## 7. \n\nWe built dedicated evaluation runners','## 7. Evaluation workflow and statistical interpretation\n\nWe built dedicated evaluation runners')
p.write_text(s,encoding='utf-8')
