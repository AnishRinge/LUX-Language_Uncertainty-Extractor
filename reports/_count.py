import re
from pathlib import Path
s=Path('reports/LUX_Project_Report_10_Pages.md').read_text(encoding='utf-8-sig')
p=re.split(r'(?m)^## (\d+)\. (.*)$',s)
print('split items',len(p),'sections', (len(p)-1)//3)
