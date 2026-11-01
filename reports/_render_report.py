import re, html, sys
from pathlib import Path
src=Path('reports/LUX_Project_Report_10_Pages.md').read_text(encoding='utf-8-sig')
parts=re.split(r'(?m)^## (\d+)\. (.+)$',src)
sections=[(parts[i],parts[i+1],parts[i+2]) for i in range(1,len(parts),3)]
def inline(x):
 x=html.escape(x); x=re.sub(r'`([^`]+)`',r'<code>\1</code>',x); x=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',x); x=re.sub(r'\*(.+?)\*',r'<i>\1</i>',x); return x
def conv(body):
 out=[]; ul=False; table=[]; code=False; code_lines=[]
 def ftable():
  nonlocal table
  if table:
   rows=[]
   for l in table:
    cells=[c.strip() for c in l.strip().strip('|').split('|')]
    if all(re.fullmatch(r':?-{3,}:?',c) for c in cells): continue
    rows.append(cells)
   out.append('<table>'+''.join('<tr>'+''.join(('<th>'+inline(c)+'</th>' if j==0 else '<td>'+inline(c)+'</td>') for c in row)+'</tr>' for j,row in enumerate(rows))+'</table>')
  table=[]
 def ful():
  nonlocal ul
  if ul: out.append('</ul>'); ul=False
 for l in body.strip().splitlines():
  if l.startswith('```'):
   ful(); ftable()
   if code: out.append('<pre>'+html.escape('\n'.join(code_lines).replace('flowchart LR','SYSTEM FLOW').replace('flowchart TD','EVALUATION FLOW'))+'</pre>'); code_lines=[]; code=False
   else: code=True
   continue
  if code: code_lines.append(l); continue
  if l.startswith('|'): ful(); table.append(l); continue
  ftable()
  if not l.strip(): ful(); continue
  m=re.match(r'^(#{2,4})\s+(.*)',l)
  if m: ful(); out.append(f'<h{min(4,len(m.group(1))-1)}>{inline(m.group(2))}</h{min(4,len(m.group(1))-1)}>'); continue
  if l.startswith('- '):
   if not ul: out.append('<ul>'); ul=True
   out.append('<li>'+inline(l[2:])+'</li>'); continue
  ful()
  if l.startswith('> '): out.append('<blockquote>'+inline(l[2:])+'</blockquote>')
  else: out.append('<p>'+inline(l)+'</p>')
 ful(); ftable(); return ''.join(out)
css='''<style>@page{size:A4;margin:15mm 17mm 17mm}body{font:10pt/1.24 Arial,sans-serif;color:#233047}h1{font-size:31pt;color:#123159;margin:0 0 14pt}h2{font-size:18pt;color:#123159;border-bottom:2px solid #20a7a0;padding-bottom:6pt;margin:0 0 11pt}h3{font-size:12.5pt;color:#126b78;margin:8pt 0 4pt}h4{font-size:10.5pt;color:#126b78;margin:7pt 0 3pt}p{margin:0 0 6pt}ul{margin:3pt 0 7pt;padding-left:17pt}li{margin-bottom:2pt}table{border-collapse:collapse;width:100%;margin:6pt 0 8pt;font-size:8.8pt}th{background:#123159;color:white;padding:5pt;border:1px solid #d0dae5;text-align:left}td{padding:4pt;border:1px solid #d0dae5}pre{font:7.4pt/1.14 Consolas,monospace;background:#eef4f8;color:#123159;border:1px solid #cbd9e5;padding:8pt;margin:5pt 0 9pt;white-space:pre-wrap}blockquote{border-left:3px solid #20a7a0;padding-left:8pt;color:#31506c;font-style:italic}code{font:9pt Consolas;background:#eef4f8}.cover{page-break-after:always;text-align:center;padding-top:48pt}.section{page-break-after:always}.section:last-child{page-break-after:auto}.kicker{color:#12827f;font-size:11pt;letter-spacing:2px}.team{width:80%;margin:34pt auto}.foot{font-size:9pt;color:#607185;margin-top:24pt}</style>'''
html_doc=['<!doctype html><html><head><meta charset="utf-8">'+css+'</head><body>']
html_doc.append('<section class="cover"><div class="kicker">ACADEMIC PROJECT REPORT</div><h1>LUX</h1><h2 style="border:0">Pre-Generation Hallucination Risk Analysis</h2><p>Completed project report · 30 October 2026</p><h3 style="margin-top:48pt">Project team</h3><table class="team"><tr><th>Student</th><th>Registration number</th></tr><tr><td>Anish Mukul Ringe</td><td>23BDS0082</td></tr><tr><td>Meet Dinesh Patil</td><td>23BDS0083</td></tr><tr><td>Abhigyan Prakash Tiwari</td><td>23BDS0119</td></tr></table><p class="foot">Language Uncertainty Extractor</p></section>')
for n,t,b in sections: html_doc.append(f'<section class="section"><h2>{n}. {t}</h2>{conv(b)}</section>')
html_doc.append('</body></html>')
Path('reports/LUX_Project_Report_10_Pages.html').write_text(''.join(html_doc),encoding='utf-8')
