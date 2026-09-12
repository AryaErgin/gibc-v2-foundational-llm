"""Render recorded evidence only. Requires existing Pillow; never imports model/tasks.

PNG and SVG share all coordinates. Font files and input/output hashes are recorded.
No network, dataset loading, benchmark requests or model execution.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import re
from pathlib import Path

INK='#152c40'; MUTED='#53687a'; BLUE='#226aa1'; TEAL='#087f83'; GOLD='#bd7414'; RED='#ab4348'
BG='#f5f8fa'; GRID='#d7e1e7'; WHITE='#ffffff'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Canvas:
    def __init__(self, title, subtitle, regular, bold):
        from PIL import Image, ImageDraw, ImageFont
        self.image=Image.new('RGB',(2560,1440),BG)
        self.draw=ImageDraw.Draw(self.image)
        self.fonts={}
        self.paths=(regular,bold)
        self.fontloader=ImageFont.truetype
        self.svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720" role="img">',f'<title>{html.escape(title)}</title>',f'<desc>{html.escape(subtitle)}</desc>',f'<rect width="1280" height="720" fill="{BG}"/>']
        self.text(48,27,title,32,bold=True)
        self.text(48,75,subtitle,18,MUTED)

    def text(self,x,y,text,size=20,color=INK,bold=False):
        key=(size,bold)
        if key not in self.fonts: self.fonts[key]=self.fontloader(str(self.paths[int(bold)]),size*2)
        self.draw.text((x*2,y*2),str(text),font=self.fonts[key],fill=color,anchor='lt')
        self.svg.append(f'<text x="{x}" y="{y+size*.8}" font-family="Segoe UI, DejaVu Sans, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{html.escape(str(text))}</text>')

    def line(self,points,color=GRID,width=2):
        pts=[(int(x*2),int(y*2)) for x,y in points]
        self.draw.line(pts,fill=color,width=width*2)
        self.svg.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="{width}"/>')

    def rect(self,x,y,w,h,color=WHITE):
        self.draw.rectangle((x*2,y*2,(x+w)*2,(y+h)*2),fill=color)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}"/>')

    def dot(self,x,y,color,r=4):
        self.draw.ellipse(((x-r)*2,(y-r)*2,(x+r)*2,(y+r)*2),fill=color)
        self.svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"/>')

    def save(self,path):
        from PIL import Image
        self.image.resize((1280,720),Image.Resampling.LANCZOS).save(path.with_suffix('.png'))
        path.with_suffix('.svg').write_text('\n'.join(self.svg+['</svg>'])+'\n',encoding='utf-8',newline='\n')


def axes(c,x,y,w,h,xmax,ymin,ymax,xticks,yticks):
    def point(a,b): return x+w*a/xmax,y+h-h*(b-ymin)/(ymax-ymin)
    for t in yticks:
        yy=point(0,t)[1];c.line([(x,yy),(x+w,yy)])
        c.text(x-48,yy-10,f'{t:g}',16,MUTED)
    for t in xticks:
        xx=point(t,ymin)[0];c.text(xx-12,y+h+12,f'{t:g}',16,MUTED)
    c.line([(x,y),(x,y+h),(x+w,y+h)],MUTED,2)
    return point


def build(repo,out,regular,bold,evidence):
    d=json.loads(evidence.read_text());v=d['validation'];s=d['summary']
    c=Canvas('EXP-020 | Full-horizon learning, then model freeze','Recorded internal validation is separate from official benchmark evaluation.',regular,bold)
    c.text(80,132,'Training loss · mean of each 1,000 updates',20,bold=True)
    p=axes(c,85,184,490,333,7.2,2.5,5.6,[0,2.4,4.8,7.2],[2.5,3,4,5])
    c.line([p(b['tokens']/1e9,b['mean_train_loss']) for b in d['train_loss_bins']],BLUE,3)
    c.text(120,552,'Prediction tokens (billions)',18,MUTED)
    c.text(715,132,'Frozen General / Edu validation',20,bold=True)
    p=axes(c,720,184,490,333,7.2,2.7,3.5,[0,2.4,4.8,7.2],[2.8,3,3.2,3.4])
    for key,color in [('general',BLUE),('edu',TEAL),('combined',GOLD)]:
        points=[p(r['tokens']/1e9,r[key]) for r in v[1:]]
        c.line(points,color,3)
        for x,y in points:c.dot(x,y,color)
    for i,(label,color) in enumerate([('General',BLUE),('Edu',TEAL),('Combined',GOLD)]):
        c.rect(720+165*i,561,12,12,color);c.text(739+165*i,556,label,18,color)
    c.text(80,604,'Final Combined NLL 2.935286 · terminal checkpoint selected before its official scoring',23,bold=True)
    c.text(48,651,'NLL in nats. Validation plot omits step-0 NLL 9.147 to show trained range; full values remain in the source.',16,MUTED)
    c.text(48,680,'Source: results/exp020-submission-evidence.json ← frozen training metrics and summary.',15,MUTED)
    c.save(out/'trajectory')

    old=(repo/'results/EXP-001D-summary.md').read_text()
    tasks=[('HellaSwag','hellaswag','acc_norm'),('ARC-Easy','arc_easy','acc_norm'),('PIQA','piqa','acc_norm'),('WinoGrande','winogrande','acc')]
    c=Canvas('EXP-001 → EXP-012 → EXP-020','Historical progression, not a one-variable ablation: model size, data and horizon changed.',regular,bold)
    c.text(60,127,'Scores (%) · first three: acc_norm; WinoGrande: acc · shared 0–70 scale',19,bold=True)
    for i,(label,key,metric) in enumerate(tasks):
        row=next(line for line in old.splitlines() if line.startswith('- '+label+':'))
        baseline=float(re.search(r'`'+metric+r',none` ([\d.]+)',row).group(1))
        vals=[baseline,d['historical_exp012']['official_results'][key][metric],d['official_results'][key]['metrics'][metric+',none']]
        y=190+i*102;c.text(60,y+8,label,19,bold=True)
        for j,(value,col) in enumerate(zip(vals,[MUTED,BLUE,TEAL])):
            yy=y+j*25;c.rect(208,yy,600*value/.70,16,col);c.text(220+600*value/.70,yy-3,f'{100*value:.3f}',16,col)
    for i,(label,col) in enumerate([('EXP-001 · 8.39M / 100M tokens',MUTED),('EXP-012 · 49.86M / 2.4B',BLUE),('EXP-020 · 49.86M / 7.2B',TEAL)]):
        c.rect(60,616+i*23,10,10,col);c.text(78,612+i*23,label,16,col)
    c.rect(902,157,327,427)
    c.text(923,183,'Held-out token PPL ↓',23,bold=True)
    wiki=d['official_results']['wikitext103']['metrics']['perplexity']
    c.text(923,236,'EXP-012   35.939',25,BLUE)
    c.text(923,281,f'EXP-020   {wiki:.3f}',25,TEAL,bold=True)
    c.text(923,345,'HellaSwag / ARC: better',19)
    c.text(923,382,'PIQA: roughly stable',19)
    c.text(923,419,'WinoGrande: regressed',19,RED)
    c.text(923,478,'Single-run comparison.',17,MUTED)
    c.text(923,506,'No significance claim.',17,MUTED)
    c.text(48,692,'Sources: EXP-001D-summary.md and submission digest. Old harness word/byte PPL is not plotted.',14,MUTED)
    c.save(out/'progression')

    cwd=json.loads((repo/'provenance/exp019-closure.json').read_text())
    qk=json.loads((repo/'provenance/exp018-closure.json').read_text())
    c=Canvas('Evidence funnel | Keep the controls, keep the failures','Each decision uses its own preregistered horizon/control—not a pooled method leaderboard.',regular,bold)
    rows=[('Recipe development','2:1 mixture + SwiGLU','RETAINED',TEAL),('WSD · EXP-013 → 017','Replicated 300M win; 2.4B gate missed','REJECTED AT SCALE',RED),('LLR / curriculum / Magma','Frozen internal gates not met','NOT PROMOTED',RED),('QK-Norm · EXP-018',f'1.5B Δ NLL {qk["decision"]["combined_improvement_nll"]:.5f}; required ≤ −0.015','NOT PROMOTED',RED),('CWD · EXP-019',f'Intermediate win → 1.5B Δ {cwd["terminal_delta_cwd_minus_adamw"]["combined_nll"]:+.5f}','REJECTED',RED)]
    for i,(name,detail,decision,col) in enumerate(rows):
        y=126+i*82;c.rect(48,y,1184,68);c.rect(48,y,6,68,col)
        c.text(70,y+13,name,22,bold=True);c.text(440,y+17,detail,20);c.text(994,y+20,decision,16,col,bold=True)
    c.rect(48,565,1184,67,INK);c.text(70,584,'EXP-020: Recipe-v3 + ordinary AdamW + full-horizon cosine · 7.2B unique-data stream',24,WHITE,bold=True)
    c.text(48,653,'Innovation: controlled evidence, transparent gates and horizon sensitivity—not invention of standard components.',18,MUTED)
    c.text(48,687,'Sources: RESULTS.md; EXP-004/008/013–016 summaries; EXP-017/018/019 closure records.',15,MUTED)
    c.save(out/'decisions')

    c=Canvas('One reproducible path | Data → training → freeze → reporting','No pretrained weights. Required benchmark scores cannot change the frozen final model.',regular,bold)
    boxes=[(48,139,367,186,'01  DATA',[('2:1 FineWeb / FineWeb-Edu',22),('Frozen revisions + scratch tokenizer',18),('Exact-document dedup + 13-gram screen',17),('7,750,968 unique selected documents',18)]),(457,139,367,186,'02  MODEL',[('49,860,480 parameters',25),('9 × width 640; 20 heads; context 512',18),('SwiGLU · RoPE · pre-RMSNorm',18),('Tied 8,192-token embedding / head',18)]),(866,139,367,186,'03  TRAIN',[('7,199,981,568 prediction tokens',22),('Fresh seed 42 · AdamW · cosine',19),('219,726 updates · BF16 / FP32 state',18),('One laptop GPU · 37.94 h incl. pacing',18)])]
    for x,y,w,h,title,lines in boxes:
        c.rect(x,y,w,h);c.text(x+17,y+14,title,20,TEAL,bold=True)
        for i,(line,size) in enumerate(lines):c.text(x+17,y+55+i*30,line,size)
    c.line([(418,223),(453,223)],BLUE,4);c.line([(827,223),(862,223)],BLUE,4)
    c.text(48,361,'Internal General/Edu validation',25,bold=True)
    c.text(48,403,'Frozen milestones; late-checkpoint rule',21)
    c.text(48,438,'Lowest Combined; later within 0.003 NLL',21)
    c.line([(523,410),(608,410)],BLUE,4)
    c.rect(623,357,610,122,INK)
    c.text(646,376,'04  FREEZE: terminal step 219,726',25,WHITE,bold=True)
    c.text(646,420,'Checkpoint / config / tokenizer identities verified',21,WHITE)
    c.rect(48,528,1184,91,WHITE)
    c.text(70,543,'05  OFFICIAL REPORTING ONLY · CPU FP32 · zero-shot',25,TEAL,bold=True)
    c.text(70,585,'HellaSwag · ARC-Easy · PIQA · WinoGrande · separate held-out WikiText-103 token PPL',22)
    c.text(48,652,'Historical benchmark exposure and public-text exclusion screening are disclosed; final selection precedes its scoring.',17,MUTED)
    c.text(48,687,'Sources: frozen config; corpus manifest; training summary; preregistration; completed official task metadata.',15,MUTED)
    c.save(out/'pipeline')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--evidence',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--font',type=Path,required=True)
    p.add_argument('--bold-font',type=Path,required=True)
    a=p.parse_args();e=a.evidence or a.repo/'results/exp020-submission-evidence.json'
    if a.output.exists() and any(a.output.glob('*.png')):raise FileExistsError('Use a fresh figure directory')
    a.output.mkdir(parents=True,exist_ok=True)
    build(a.repo,a.output,a.font,a.bold_font,e)
    import PIL
    names=['results/EXP-001D-summary.md','provenance/exp018-closure.json','provenance/exp019-closure.json']
    meta={'evidence_sha256':sha(e),'renderer_sha256':sha(Path(__file__)),'pillow_version':PIL.__version__,
          'font_sha256':sha(a.font),'bold_font_sha256':sha(a.bold_font),'font_note':'Locally installed fonts; not redistributed. PNG rendered at 2x and downsampled.',
          'inputs':{name:sha(a.repo/name) for name in names},
          'outputs':{p.name:sha(p) for p in sorted(a.output.iterdir()) if p.suffix in ('.png','.svg')}}
    (a.output/'figure-provenance.json').write_text(json.dumps(meta,indent=2)+'\n',newline='\n')
    print('Rendered four PNG/SVG pairs from saved evidence; no model or benchmark execution.')


if __name__=='__main__':main()
