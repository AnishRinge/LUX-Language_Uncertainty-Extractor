from pathlib import Path
p=Path('reports/LUX_Project_Report_10_Pages.md')
s=p.read_text(encoding='utf-8-sig')
start=s.index('## 7. Evaluation workflow and statistical interpretation')
end=s.index('## 8. Inference software, architecture, and interface')
block=s[start:end]
s=s[:start]+s[end:]
# Keep assessment methodology in section 6; place statistical interpretation before interface architecture.
pos=s.index('## 8. Inference software, architecture, and interface')
block=block.replace('## 7. Evaluation workflow and statistical interpretation','### Evaluation workflow and statistical interpretation',1)
s=s[:pos]+block+s[pos:]
p.write_text(s,encoding='utf-8')
