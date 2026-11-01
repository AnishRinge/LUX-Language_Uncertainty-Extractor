import re
from pathlib import Path
from textwrap import wrap
from pypdf import PdfReader

src=Path('reports/LUX_Project_Report_10_Pages.md').read_text(encoding='utf-8-sig')
p=re.split(r'(?m)^## (\d+)\. (.*)$',src)
sections=[(p[i],p[i+1],p[i+2]) for i in range(1,len(p),3)]
assert len(sections)==10
W,H=595,842; ML,MR=52,52; TOP,BOT=770,52; CW=W-ML-MR
NAVY=(.07,.18,.32); TEAL=(.08,.55,.54); INK=(.15,.20,.28); MID=(.37,.43,.51); PALE=(.93,.96,.98); GRID=(.80,.85,.89)
font='/Windows/Fonts/arial.ttf'; bold='/Windows/Fonts/arialbd.ttf'; mono='/Windows/Fonts/consola.ttf'; monob='/Windows/Fonts/consolab.ttf'
pages=[]
def esc(s):
 return s.replace('\\','\\\\').replace('(','\\(').replace(')','\\)').replace('\r','').replace('\n',' ')
def wrapw(text,size,width,bolded=False,monoed=False):
 # whitespace width approximations based on chosen embedded fonts
 factor=.60 if monoed else (.56 if bolded else .52)
 maxc=max(1,int(width/(size*factor)))
 return wrap(text,width=maxc,break_long_words=True,break_on_hyphens=True) or ['']
class Page:
 def __init__(self,num,title=None):
  self.num=num; self.ops=[]; self.y=TOP
  self.rect(0,0,W,H,(1,1,1),stroke=None)
  if num>1:
   self.text(ML, H-30, 'LUX  /  LANGUAGE UNCERTAINTY EXTRACTOR',8,MID,bolded=True)
   self.line(ML,H-39,W-MR,H-39,GRID,.7)
  if title:
   self.text(ML,self.y,title,18,NAVY,bolded=True); self.y-=30
 def text(self,x,y,s,size=9,color=INK,bolded=False,monoed=False):
  f='F4' if monoed and bolded else 'F3' if monoed else 'F2' if bolded else 'F1'
  self.ops.append(f'q {color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg BT /{f} {size:.2f} Tf 1 0 0 1 {x:.2f} {y:.2f} Tm ({esc(s)}) Tj ET Q')
 def line(self,x1,y1,x2,y2,color=GRID,lw=1): self.ops.append(f'q {color[0]:.3f} {color[1]:.3f} {color[2]:.3f} RG {lw} w {x1:.1f} {y1:.1f} m {x2:.1f} {y2:.1f} l S Q')
 def rect(self,x,y,w,h,fill,stroke=GRID,lw=.7):
  if stroke: self.ops.append(f'q {fill[0]:.3f} {fill[1]:.3f} {fill[2]:.3f} rg {stroke[0]:.3f} {stroke[1]:.3f} {stroke[2]:.3f} RG {lw} w {x} {y} {w} {h} re B Q')
  else: self.ops.append(f'q {fill[0]:.3f} {fill[1]:.3f} {fill[2]:.3f} rg {x} {y} {w} {h} re f Q')
 def para(self,text,size=9.1,gap=5,color=INK,bolded=False):
  lines=wrapw(text,size,CW,bolded)
  for line in lines:
   self.text(ML,self.y,line,size,color,bolded); self.y-=size*1.32
  self.y-=gap
 def heading(self,s,level=3):
  size={3:11.3,4:10.2}.get(level,11); self.y-=2
  self.text(ML,self.y,s,size,TEAL,bolded=True); self.y-=size*1.45
 def bullets(self,items,size=7.8):
  for t in items:
   lines=wrapw(t,size,CW-17)
   self.text(ML+3,self.y,'-',size,TEAL,bolded=True)
   self.text(ML+16,self.y,lines[0],size,INK)
   self.y-=size*1.17
   for l in lines[1:]: self.text(ML+16,self.y,l,size,INK); self.y-=size*1.17
   self.y-=1.5
 def table(self,rows,weights=None,size=7.9):
  if not rows:return
  n=len(rows[0]); weights=weights or [1]*n; widths=[CW*w/sum(weights) for w in weights]; pad=5; lineh=size*1.22
  for ri,row in enumerate(rows):
   wrapped=[wrapw(str(cell),size,widths[j]-pad*2,ri==0) for j,cell in enumerate(row)]
   maxlines=max(map(len,wrapped)); rh=max(22,maxlines*lineh+pad*2)
   if ri==0:
    self.rect(ML,self.y-rh,CW,rh,NAVY,stroke=None)
   else: self.rect(ML,self.y-rh,CW,rh,PALE if ri%2==1 else (1,1,1),GRID,.45)
   x=ML
   for j,celllines in enumerate(wrapped):
    for k,txt in enumerate(celllines): self.text(x+pad,self.y-pad-size-k*lineh,txt,size,(1,1,1) if ri==0 else INK,bolded=ri==0)
    x+=widths[j]
   self.y-=rh
  self.y-=8
 def footer(self):
  self.line(ML,32,W-MR,32,GRID,.6); self.text(ML,19,'ACADEMIC PROJECT REPORT  ·  30 OCTOBER 2026',7.5,MID); self.text(W-MR-42,19,f'{self.num} / 10',7.5,MID)
 def box(self,x,y,w,h,label,sub='',fill=PALE):
  self.rect(x,y-h,w,h,fill,GRID,.8)
  ls=wrapw(label,7.6,w-10,True)
  yy=y-12
  for l in ls:self.text(x+5,yy,l,7.6,NAVY,bolded=True); yy-=9
  if sub:
   for l in wrapw(sub,6.2,w-10):self.text(x+5,yy-1,l,6.2,MID); yy-=7.5
 def arrow(self,x1,y1,x2,y2):
  self.line(x1,y1,x2,y2,TEAL,1.2)
  # horizontal/vertical arrow tip
  if abs(x2-x1)>=abs(y2-y1):
   d=1 if x2>x1 else -1; self.line(x2,y2,x2-5*d,y2+3,TEAL,1.2); self.line(x2,y2,x2-5*d,y2-3,TEAL,1.2)
  else:
   d=1 if y2>y1 else -1; self.line(x2,y2,x2-3,y2-5*d,TEAL,1.2); self.line(x2,y2,x2+3,y2-5*d,TEAL,1.2)

def unmd(s):
 s=re.sub(r'\[([^\]]+)\]\([^)]*\)',r'\1',s); s=s.replace('**','').replace('`','').replace('*',''); return s.strip()
def clean_block(body):
 return body.strip().splitlines()

# Cover page
p0=Page(0)
p0.text(ML,700,'ACADEMIC PROJECT REPORT',11,TEAL,bolded=True)
p0.text(ML,650,'LUX',40,NAVY,bolded=True)
p0.text(ML,615,'Pre-Generation Hallucination',24,NAVY,bolded=True)
p0.text(ML,585,'Risk Analysis',24,NAVY,bolded=True)
p0.line(ML,560,W-MR,560,TEAL,3)
p0.text(ML,530,'Completed project report  ·  30 October 2026',11,MID)
p0.text(ML,470,'PROJECT TEAM',12,TEAL,bolded=True)
rows=[['Student','Registration number'],['Anish Mukul Ringe','23BDS0082'],['Meet Dinesh Patil','23BDS0083'],['Abhigyan Prakash Tiwari','23BDS0119']]
p0.y=445;p0.table(rows,[2,1],10)
p0.text(ML,260,'Language Uncertainty Extractor',13,NAVY,bolded=True)
p0.text(ML,238,'Research on pre-generation signals for answer reliability',9,MID)
p0.footer(); pages.append(p0)

for idx,(number,title,body) in enumerate(sections, start=1):
 pg=Page(idx,title)
 if idx==9:
  pg.heading('Checks recorded in project logs')
  pg.para('We created tests for dataset schemas, manifests, hashing, audit rules, generation pilots, evaluator schemas and retries, target eligibility, predictor baselines, production pipelines, risk inference, saved OOF analysis, and live API behavior.',7.5,3)
  pg.table([['Verification record','Recorded result'],['Baseline inference implementation','355 tests passed in the recorded suite'],['Live interface regression','433 tests passed; one deprecation warning'],['Live API targeted checks','12 tests passed; HTTP 200 on valid requests; HTTP 422 on invalid inputs'],['Saved-OOF checks','486 aligned targets; five prompt-level folds; 2,000 paired bootstrap resamples'],['Artifact and provenance','Versioned JSON, source hashes, checkpoints, evaluator-version isolation']], [1.3,1.7], 7.0)
  pg.heading('Reproducibility',3)
  pg.para('We saved dataset manifests, split assignments, generation records, evaluation provenance, fold predictions, model comparisons, verification summaries, and baseline artifacts. Evaluation and generation runners persist checkpoints so a saved run can be inspected and resumed.',7.5,3)
  pg.para('We kept implementation checks separate from predictive validity. Successful API responses establish that the assessment path executes; they do not measure accuracy. The scientific evidence remains specific to SQuAD v2, the frozen Qwen revision and decoding setup, and evaluator-derived targets.',7.5,3)
  pg.heading('Interface validation',3)
  pg.bullets(['Valid requests returned assessments for prompt-only, hidden-only, fusion, mean, and median options.','The hidden representation measured 29 × 2,048; assessments run before generation.','Invalid questions and unknown predictor selections returned 422 without a score.'],7.5)
  pg.footer(); pages.append(pg); continue
 lines=clean_block(body)
 i=0
 while i<len(lines):
  line=lines[i].strip()
  if not line: i+=1; continue
  if line.startswith('```mermaid'):
   i+=1; diagram=[]
   while i<len(lines) and not lines[i].startswith('```'): diagram.append(lines[i].strip()); i+=1
   i+=1
   if 'flowchart LR' in '\n'.join(diagram):
    # three row architecture flow with branches
    y=pg.y-15; bw=78; gap=8; x0=ML
    boxes=[('SQuAD v2','source'),('Canonicalize','schema'),('Audit','integrity'),('Select','500 prompts'),('Features','question + context'),('Frozen Qwen3','prompt forward'),('Generation','10 seeds'),('Evaluation','versioned labels'),('Risk target','UNRELIABLE / 10')]
    row1=boxes[:5]; row2=boxes[5:]
    for r,arr in enumerate([row1,row2]):
     yy=y-r*82
     for j,(lab,sub) in enumerate(arr):
      xx=x0+j*(bw+gap); pg.box(xx,yy,bw,43,lab,sub,(.91,.96,.96) if r==0 else PALE)
      if j<len(arr)-1:pg.arrow(xx+bw,yy-22,xx+bw+gap-2,yy-22)
    # arrows to split/model comparison and downstream
    yy=y-82
    pg.text(ML,yy-68,'Prompt-only  |  Hidden-only  |  Fusion',8.4,NAVY,bolded=True)
    pg.arrow(ML+230,yy-57,ML+230,yy-70)
    pg.box(ML+120,yy-77,220,40,'Cross-validation and saved artifacts','prompt identity folds')
    pg.arrow(ML+230,yy-117,ML+230,yy-129)
    pg.box(ML+120,yy-136,220,39,'Assessment and conservative routing','pre-generation interface')
    pg.y=yy-189
   else:
    y=pg.y-12
    pg.box(ML+20,y,150,44,'Question + context','ten generated answers',(.91,.96,.96)); pg.box(ML+215,y,150,44,'Structured evaluation','versioned categories'); pg.arrow(ML+170,y-22,ML+215,y-22)
    pg.box(ML+20,y-74,150,44,'Human review','meaning and ambiguity',PALE); pg.box(ML+215,y-74,150,44,'Complete outcomes','risk = count / 10',(.91,.96,.96)); pg.arrow(ML+170,y-96,ML+215,y-96)
    pg.arrow(ML+95,y-44,ML+95,y-74); pg.arrow(ML+290,y-44,ML+290,y-74)
    pg.y=y-143
   continue
  if line.startswith('```'):
   i+=1; code=[]
   while i<len(lines) and not lines[i].startswith('```'): code.append(lines[i]); i+=1
   i+=1
   for l in code:
    if l.startswith('flowchart') or l.startswith('  '):continue
    pg.para(unmd(l),7.4,gap=1,color=MID)
   continue
  if line.startswith('|'):
   tbl=[]
   while i<len(lines) and lines[i].strip().startswith('|'):
    cells=[unmd(c) for c in lines[i].strip().strip('|').split('|')]
    if not all(re.fullmatch(r':?-{3,}:?',c.strip()) for c in cells):tbl.append(cells)
    i+=1
   if tbl:
    n=len(tbl[0]); weights=[2]+[1]*(n-1) if n>1 else [1]
    if n>=4:weights=[1.6,1,1,1]
    pg.table(tbl,weights,6.4 if n>=4 else 6.9)
   continue
  if line.startswith('#'):
   level=len(line)-len(line.lstrip('#')); pg.heading(unmd(line.lstrip('# ')),min(4,max(3,level))); i+=1; continue
  if line.startswith('- '):
   items=[]
   while i<len(lines) and lines[i].strip().startswith('- '):items.append(unmd(lines[i].strip()[2:])); i+=1
   pg.bullets(items,7.5);continue
  if line.startswith('> '):
   pg.para(unmd(line[2:]),9.2,7,TEAL,True);i+=1;continue
  # collect ordinary paragraph until next block
  para=[unmd(line)];i+=1
  while i<len(lines) and lines[i].strip() and not lines[i].strip().startswith(('#','- ','|','```','> ')):
   para.append(unmd(lines[i].strip())); i+=1
  pg.para(' '.join(para),7.35,gap=2)
 pg.footer(); pages.append(pg)

# balanced page content size can run close; print cursors to check
print('section bottom y:',[(x.num,round(x.y,1)) for x in pages])
# PDF 1.4 with embedded TrueType fonts and ToUnicode mapping for accurate extraction
objects=[]
def add(obj):objects.append(obj);return len(objects)
font_ids={}
for name,path in [('F1','C:/Windows/Fonts/arial.ttf'),('F2','C:/Windows/Fonts/arialbd.ttf'),('F3','C:/Windows/Fonts/consola.ttf'),('F4','C:/Windows/Fonts/consolab.ttf')]:
 data=Path(path).read_bytes(); fontfile=add(b'<< /Length '+str(len(data)).encode()+b' /Length1 '+str(len(data)).encode()+b' >>\nstream\n'+data+b'\nendstream')
 # use built-in Type1 fonts instead to avoid font parsing dependencies
 font_ids[name]=None
# Standard Helvetica and Courier are reliable; symbols use basic ASCII except bullets; replace bullet glyph.
font_ids={'F1':add(b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>'),
'F2':add(b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>'),
'F3':add(b'<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>'),
'F4':add(b'<< /Type /Font /Subtype /Type1 /BaseFont /Courier-Bold /Encoding /WinAnsiEncoding >>')}
# fix non-winansi bullet
for pg in pages: pg.ops=[op.replace('•','-') for op in pg.ops]
# Put the cover block on page one and combine it with the objective page.
pages[1].ops = pages[0].ops[1:-1] + pages[1].ops
pages.pop(0)
pages[0].num=1
page_ids=[]; content_ids=[]
for pg in pages:
 stream=('\n'.join(pg.ops)).encode('cp1252','replace')
 cid=add(b'<< /Length '+str(len(stream)).encode()+b' >>\nstream\n'+stream+b'\nendstream');content_ids.append(cid)
 pid=add(b'');page_ids.append(pid)
# page tree/catalog
pages_root=add(b''); catalog=add(b'')
for j,pid in enumerate(page_ids):
 resources=b'<< /Font << '+b' '.join(b'/'+k.encode()+b' '+str(v).encode()+b' 0 R' for k,v in font_ids.items())+b' >> >>'
 objects[pid-1]=b'<< /Type /Page /Parent '+str(pages_root).encode()+b' 0 R /MediaBox [0 0 595 842] /Resources '+resources+b' /Contents '+str(content_ids[j]).encode()+b' 0 R >>'
objects[pages_root-1]=b'<< /Type /Pages /Kids ['+b' '.join(str(x).encode()+b' 0 R' for x in page_ids)+b'] /Count '+str(len(page_ids)).encode()+b' >>'
objects[catalog-1]=b'<< /Type /Catalog /Pages '+str(pages_root).encode()+b' 0 R >>'
pdf=bytearray(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n'); offsets=[0]
for n,obj in enumerate(objects,1):
 offsets.append(len(pdf));pdf.extend(f'{n} 0 obj\n'.encode());pdf.extend(obj);pdf.extend(b'\nendobj\n')
xref=len(pdf);pdf.extend(f'xref\n0 {len(objects)+1}\n'.encode());pdf.extend(b'0000000000 65535 f \n')
for off in offsets[1:]:pdf.extend(f'{off:010d} 00000 n \n'.encode())
pdf.extend(f'trailer\n<< /Size {len(objects)+1} /Root {catalog} 0 R >>\nstartxref\n{xref}\n%%EOF'.encode())
out=Path('reports/LUX_Project_Report_10_Pages.pdf');out.write_bytes(pdf)
r=PdfReader(str(out));print('written',out.stat().st_size,'pages',len(r.pages)); print('page chars',[len(x.extract_text() or '') for x in r.pages]); text='\n'.join(x.extract_text() or '' for x in r.pages);print('names',[n in text for n in ['Anish Mukul Ringe','Meet Dinesh Patil','Abhigyan Prakash Tiwari','23BDS0082','23BDS0083','23BDS0119']]); print(text[:500])









