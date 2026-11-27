"""Reproduce the two Phase 2 figures. Research code and submission stay untouched.

Dependencies: reportlab, pypdf, pypdfium2. Run with Python from the project directory.
SVG and PDF are drawn from the same geometric primitives; PDF text is embedded.
"""
from pathlib import Path
from html import escape
import hashlib
import json
import math

from pypdf import PdfReader
import pypdfium2 as pdfium
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import toColor

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'docs/submissions/phase2/DAV_Phase2_Submission.tex'
FONT_DIR = Path('C:/Windows/Fonts')
pdfmetrics.registerFont(TTFont('FigureSans', str(FONT_DIR / 'arial.ttf')))
pdfmetrics.registerFont(TTFont('FigureSansBold', str(FONT_DIR / 'arialbd.ttf')))

C = dict(ink='#203047', muted='#55657A', line='#62748A',
         blue='#315C8B', bluefill='#EEF3FA', green='#547B43',
         greenfill='#F0F6EC', orange='#A26536', orangefill='#FBF1E9',
         purple='#70548B', purplefill='#F3EFF8', gray='#F5F7FA')


class Figure:
    def __init__(self, name, height, title, description):
        self.name, self.w, self.h = name, 1080, height
        self.scale = (180 / 25.4 * 72) / self.w  # full two-column width, 180 mm
        self.pdf = canvas.Canvas(str(ROOT / (name + '.pdf')),
                                 pagesize=(self.w*self.scale, self.h*self.scale))
        self.pdf.setTitle(title)
        self.pdf.setAuthor('DAV Phase 2: proposed research')
        self.pdf.setSubject(description)
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="180mm" '
                      f'height="{height/self.w*180:.3f}mm" viewBox="0 0 {self.w} {height}" '
                      'role="img" aria-labelledby="title desc">',
                      f'<title id="title">{escape(title)}</title>',
                      f'<desc id="desc">{escape(description)}</desc>',
                      '<rect width="100%" height="100%" fill="white"/>']
        self.texts = []
        self.rect(0, 0, self.w, self.h, 'white', 'white', 0)

    def rect(self, x, y, w, h, fill, stroke, r=10, dash=False, lw=1.4):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
                          f'fill="{fill}" stroke="{stroke}" stroke-width="{lw}"'
                          + (' stroke-dasharray="6 4"' if dash else '') + '/>')
        p, s = self.pdf, self.scale
        p.setFillColor(toColor(fill)); p.setStrokeColor(toColor(stroke))
        p.setLineWidth(lw*s); p.setDash([6*s, 4*s] if dash else [])
        p.roundRect(x*s, (self.h-y-h)*s, w*s, h*s, r*s, fill=1, stroke=1)
        p.setDash([])

    def text(self, x, y, label, size=18, bold=False, color=None, anchor='middle', width=None):
        color = color or C['ink']
        font = 'FigureSansBold' if bold else 'FigureSans'
        tw = pdfmetrics.stringWidth(label, font, size)
        if width is not None and tw > width:
            raise ValueError(f'{self.name}: text too wide: {label!r}: {tw:.1f}>{width}')
        assert 0 <= x <= self.w and 0 <= y <= self.h
        self.texts.append(dict(text=label, x=x, y=y, size=size, width=tw))
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
                          f'font-size="{size}" font-weight="{"700" if bold else "400"}" '
                          f'fill="{color}" text-anchor="{anchor}">{escape(label)}</text>')
        p, s = self.pdf, self.scale
        p.setFillColor(toColor(color)); p.setFont(font, size*s)
        fn = {'middle': p.drawCentredString, 'start': p.drawString, 'end': p.drawRightString}[anchor]
        fn(x*s, (self.h-y)*s, label)

    def path(self, pts, color=None, dash=False, arrow=False, lw=1.6):
        color = color or C['line']
        commands = 'M ' + ' L '.join(f'{x},{y}' for x,y in pts)
        self.parts.append(f'<path d="{commands}" fill="none" stroke="{color}" '
                          f'stroke-width="{lw}" stroke-linejoin="round"'
                          + (' stroke-dasharray="6 4"' if dash else '') + '/>')
        p,s=self.pdf,self.scale
        p.setStrokeColor(toColor(color));p.setLineWidth(lw*s)
        p.setDash([6*s,4*s] if dash else [])
        q=p.beginPath();q.moveTo(pts[0][0]*s,(self.h-pts[0][1])*s)
        for x,y in pts[1:]:q.lineTo(x*s,(self.h-y)*s)
        p.drawPath(q);p.setDash([])
        if arrow:
            x,y=pts[-1]; px,py=pts[-2]
            dx,dy=x-px,y-py;norm=math.hypot(dx,dy);dx/=norm;dy/=norm
            verts=[(x,y),(x-8*dx+3.8*dy,y-8*dy-3.8*dx),
                   (x-8*dx-3.8*dy,y-8*dy+3.8*dx)]
            self.parts.append('<polygon points="'+' '.join(f'{a},{b}' for a,b in verts)+
                              f'" fill="{color}"/>')
            p.setFillColor(toColor(color));q=p.beginPath()
            q.moveTo(verts[0][0]*s,(self.h-verts[0][1])*s)
            for a,b in verts[1:]:q.lineTo(a*s,(self.h-b)*s)
            q.close();p.drawPath(q,fill=1,stroke=0)

    def group(self, y, h, heading, key):
        self.rect(26,y,1028,h,C[key+'fill'],C[key],16)
        tw=pdfmetrics.stringWidth(heading,'FigureSansBold',20)
        self.rect(39,y-1,tw+28,33,C[key],C[key],6)
        self.text(53,y+23,heading,20,True,'#FFFFFF','start')

    def card(self,x,y,w,h,heading,lines,key='blue',dash=False):
        self.rect(x,y,w,h,'#FFFFFF',C[key],10,dash)
        self.text(x+w/2,y+27,heading,19,True,width=w-20)
        for i,line in enumerate(lines):
            self.text(x+w/2,y+53+i*23,line,18,width=w-20)

    def node(self,x,y,w,h,heading,lines,key='blue'):
        self.rect(x,y,w,h,'#FFFFFF',C[key],6,lw=1.5)
        self.text(x+w/2,y+27,heading,21.5,True,width=w-20)
        for i,line in enumerate(lines):
            self.text(x+w/2,y+53+i*24,line,20.5,width=w-20)

    def panel(self,x,y,w,h,heading,key='blue',dash=False):
        self.rect(x,y,w,h,C[key+'fill'],C[key],8,dash,lw=1.1)
        self.text(x+16,y+28,heading,22.5,True,C[key],anchor='start',width=w-32)

    def ellipse(self,x,y,w,h,fill,stroke):
        self.parts.append(f'<ellipse cx="{x+w/2}" cy="{y+h/2}" rx="{w/2}" ry="{h/2}" '
                          f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        p,s=self.pdf,self.scale
        p.setFillColor(toColor(fill));p.setStrokeColor(toColor(stroke));p.setLineWidth(1.5*s)
        p.ellipse(x*s,(self.h-y-h)*s,(x+w)*s,(self.h-y)*s,fill=1,stroke=1)

    def data_store(self,x,y,w,h,heading,lines,key='blue'):
        e=24;fill='#FFFFFF';stroke=C[key]
        self.parts.append(f'<rect x="{x}" y="{y+e/2}" width="{w}" height="{h-e}" fill="{fill}"/>')
        p,s=self.pdf,self.scale;p.setFillColor(toColor(fill))
        p.rect(x*s,(self.h-y-h+e/2)*s,w*s,(h-e)*s,fill=1,stroke=0)
        self.ellipse(x,y+h-e,w,e,fill,stroke)
        self.path([(x,y+e/2),(x,y+h-e/2)],color=stroke,lw=1.5)
        self.path([(x+w,y+e/2),(x+w,y+h-e/2)],color=stroke,lw=1.5)
        self.ellipse(x,y,w,e,fill,stroke)
        self.text(x+w/2,y+53,heading,21.5,True,width=w-16)
        for i,line in enumerate(lines):
            self.text(x+w/2,y+91+i*32,line,20.5,width=w-16)

    def decision(self,cx,cy,w,h,lines):
        pts=[(cx,cy-h/2),(cx+w/2,cy),(cx,cy+h/2),(cx-w/2,cy)]
        self.parts.append('<polygon points="'+' '.join(f'{x},{y}' for x,y in pts)+
                          f'" fill="#FFFFFF" stroke="{C["orange"]}" stroke-width="1.5"/>')
        p,s=self.pdf,self.scale;p.setFillColor(toColor('#FFFFFF'))
        p.setStrokeColor(toColor(C['orange']));p.setLineWidth(1.5*s)
        q=p.beginPath();q.moveTo(pts[0][0]*s,(self.h-pts[0][1])*s)
        for x,y in pts[1:]:q.lineTo(x*s,(self.h-y)*s)
        q.close();p.drawPath(q,fill=1,stroke=1)
        for i,line in enumerate(lines):self.text(cx,cy-4+i*24,line,20.5,width=w-25)

    def save(self):
        self.parts.append('</svg>')
        (ROOT/(self.name+'.svg')).write_text('\n'.join(self.parts),encoding='utf-8')
        self.pdf.showPage();self.pdf.save()
        reader=PdfReader(ROOT/(self.name+'.pdf'))
        assert not list(reader.pages[0].images), 'PDF must contain vector shapes and text'
        doc=pdfium.PdfDocument(str(ROOT/(self.name+'.pdf')))
        page=doc[0]
        page.render(scale=300/72).to_pil().save(ROOT/(self.name+'.png'))
        page.render(scale=1.8).to_pil().save(ROOT/(self.name+'_inspection.png'))
        page.close();doc.close()
        return dict(name=self.name,width_mm=180,height_mm=round(self.h/self.w*180,3),
                    minimum_font_pt=round(min(t['size'] for t in self.texts)*self.scale,2),
                    raster_images_in_pdf=0,text_items=len(self.texts))


def overview():
    f=Figure('methodology_overview',1000,'Research workflow and experimental design',
             'Original research workflow: twelve prescribed stages; numbered operations, '
             'explicit paired alternatives and complementary analyses. No measured outcomes.')
    C['line']='#374151'
    f.text(32,35,'Research workflow and experimental design',28,True,anchor='start')
    f.text(32,65,'Proposed study | cohort-specific evidence reliability',20.5,
           color=C['muted'],anchor='start')
    f.panel(20,88,1040,164,'a  Prepare observations and representations | Steps 1-4','blue')
    xs=[32,294,556,818]
    details=[('1 Research input',['Risk target + clocks','Cohort / seed units']),
             ('2 Data collection',['Controlled streams','Natural case studies']),
             ('3 Preprocessing',['Fit before final testing','Freeze partitions']),
             ('4 Representation',['Marginal + joint scores','Fixed feature map'])]
    for x,(heading,lines) in zip(xs,details):f.node(x,134,230,94,heading,lines,key='blue')
    for x in xs[:-1]:f.path([(x+230,181),(x+262,181)],arrow=True)

    f.panel(20,296,1040,219,'b  Specify the method and matched comparison | Steps 5-7','green')
    f.node(32,347,310,94,'5 Evidence method',['Input alarm + loss interval','Eligibility before verdict'],key='green')
    f.node(385,347,310,94,'6 Matched baselines',['Harm gate + proxy rules','Shared evidence access'],key='green')
    f.node(738,347,310,94,'7 Experimental design',['2 models; 6 mechanisms','30 seeds; paired streams'],key='green')
    f.path([(342,394),(385,394)],arrow=True);f.path([(695,394),(738,394)],arrow=True)
    f.path([(540,252),(540,294)],arrow=True)
    f.rect(32,461,1016,38,'#FFFFFF',C['green'],4)
    f.text(540,488,'Paired alternatives: fixed / rolling reference | label delay d = 0, 2, 5',20.5,
           color=C['green'],width=992)

    f.panel(20,560,1040,182,'c  Assess the same experiment records | Steps 8-11','purple')
    f.node(32,620,230,90,'8 Evaluate',['Correct / wrong yield','Coverage + availability'],key='purple')
    f.node(328,620,218,90,'9 Statistics',['Seed uncertainty','Truth bounds'],key='purple')
    f.node(578,620,218,90,'10 Ablations',['One change at a time','Same data + clocks'],key='purple')
    f.node(828,620,220,90,'11 Failure analysis',['Enumerate all cases','Versioned traces'],key='purple')
    f.path([(540,515),(540,558)],arrow=True)
    f.path([(262,665),(280,665),(280,608),(938,608)])
    for cx in [437,687,938]:f.path([(cx,608),(cx,620)],arrow=True)

    f.panel(20,794,1040,144,'d  Synthesize evidence and limits | Step 12','purple')
    f.node(32,842,1016,86,'12 Final research output',
           ['Comparative evidence + failure catalogue + cohort-aligned report',
            'Phase 2 specifies future experiments; no results are claimed'],key='purple')
    for cx in [147,437,687,938]:f.path([(cx,710),(cx,768)])
    f.path([(147,768),(938,768)]);f.path([(540,768),(540,842)],arrow=True)
    f.path([(32,970),(69,970)],arrow=True)
    f.text(82,977,'Dependency; a branch denotes complementary analyses',20.5,anchor='start',color=C['muted'])
    f.text(1048,977,'Proposed protocol',20.5,anchor='end',color=C['muted'])
    return f.save()


def architecture():
    f=Figure('proposed_framework_architecture',1000,'Cohort-aligned monitoring data flow',
             'Original swimlane data-flow architecture. Rectangles are computations, '
             'cylinders retain records, and a diamond tests label eligibility. '
             'A read-only assessment boundary isolates generator truth from the monitor.')
    C['line']='#374151'
    f.text(32,35,'Proposed framework | cohort-aligned data flow',28,True,anchor='start')
    f.text(32,65,'Frozen preparation | parallel evidence channels | independent assessment',20.5,
           color=C['muted'],anchor='start')
    f.node(32,105,290,125,'Fit and freeze',['Classifier f; transform phi','Training / development'],key='blue')
    f.node(395,105,290,125,'Calibrate input scores',['Null trajectories','Scales + threshold tau'],key='blue')
    f.node(738,105,310,125,'Fix risk reference W0',['Historical labelled losses','Same risk target throughout'],key='blue')

    f.panel(228,300,636,151,'Input channel | label-free','green')
    f.node(244,350,190,88,'Input reference',['Fixed / rolling','Earlier inputs'],key='green')
    f.node(466,350,172,88,'Score change',['KS / JS + MMD','Joint + marginal'],key='green')
    f.node(668,350,184,88,'Input alarm A',['Threshold score','Rank marginals'],key='green')
    f.data_store(32,347,166,350,'Test cohort',
                 ['x_i, f(x_i)','ID, t_i, r_i','Released y_i','b = 1,000'],key='blue')
    f.data_store(894,347,154,350,'Ledger',
                 ['A, B','Label fraction','Cohort age','Reference ID','Validity'],key='purple')

    f.text(244,470,'Invalid records suspend the run for repair.',20.5,anchor='start',color=C['muted'],width=620)
    f.panel(228,476,636,227,'Loss channel | released labels','orange')
    f.node(244,528,190,85,'Join labels',['Released y only','Valid ID join'],key='orange')
    f.decision(552,570,144,100,['All labels','released?'])
    f.node(668,528,184,85,'Harm verdict B',['Excess-loss CI','Else unresolved'],key='orange')
    f.rect(466,649,172,44,'#FFFFFF',C['orange'],6,lw=1.5)
    f.text(552,678,'Unresolved',21.5,True,width=152)
    # One uniform information-dependency arrow; role is encoded by labelled lanes.
    f.path([(177,230),(177,271),(115,271),(115,347)],arrow=True)
    f.path([(540,230),(540,267),(620,267),(620,350)],arrow=True)
    # Route W0 between nodes; a small underpass prevents an accidental junction.
    f.path([(893,230),(893,264),(654,264),(654,386)])
    f.path([(654,394),(654,509),(760,509),(760,528)],arrow=True)
    f.path([(198,390),(218,390),(218,338),(552,338),(552,350)],arrow=True)
    f.path([(434,390),(466,390)],arrow=True);f.path([(638,390),(668,390)],arrow=True)
    f.path([(852,390),(894,390)],arrow=True)
    f.path([(198,570),(244,570)],arrow=True);f.path([(434,570),(480,570)],arrow=True)
    f.path([(624,570),(668,570)],arrow=True)
    f.text(646,558,'yes',20.5,width=42)
    f.path([(552,620),(552,649)],arrow=True);f.text(578,643,'no',20.5,width=35)
    f.path([(852,570),(894,570)],arrow=True)
    f.path([(638,671),(894,671)],arrow=True)

    f.panel(32,744,1016,216,'Reporting and synthetic assessment | read-only','purple',dash=True)
    f.rect(52,790,614,36,'#FFFFFF',C['purple'],4)
    f.text(359,816,'One shared cohort evidence record',20.5,width=594)
    f.node(52,850,284,88,'Aligned evidence display',['Cohort / time / status','Input + loss evidence'],key='purple')
    f.node(382,850,284,88,'Compare verdict policies',['Separated vs harm-gated','Correct / wrong yield'],key='purple')
    f.node(712,790,316,148,'Independent risk truth',
           ['Generator-only samples','Synthetic evaluation only','Never used by monitor'],key='purple')
    f.path([(971,697),(971,780),(646,780),(646,790)],arrow=True)
    f.path([(359,826),(359,838),(194,838),(194,850)],arrow=True)
    f.path([(359,838),(524,838),(524,850)],arrow=True)
    f.path([(712,894),(666,894)],arrow=True);f.text(689,882,'truth',20.5,width=45)
    f.text(32,987,'Box: computation | cylinder: records | diamond: eligibility | crossings do not join',20.5,
           color=C['muted'],anchor='start',width=1016)
    return f.save()


if __name__ == '__main__':
    records=[overview(),architecture()]
    manifest=dict(source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  figures=records,design='Research-informed workflow and swimlane data-flow architecture; 180 mm; semantic shapes; uniform arrows',
                  svg_text='editable Arial text',pdf_text='embedded TrueType Arial',
                  png_dpi=300,source_modified=False)
    (ROOT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))
