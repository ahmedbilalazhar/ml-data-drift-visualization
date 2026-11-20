"""Render the editable Markdown report using bundled reportlab; also render QA pages."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import time
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from PIL import Image as PILImage
import pypdfium2 as pdfium


def inline(text):
    text=escape(text)
    return re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',text)


def render(source):
    started=time.perf_counter();source=Path(source)
    target=source.with_suffix('.pdf');qa=source.parent/'qa';qa.mkdir(exist_ok=True)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle('BodyIDM',fontName='Helvetica',fontSize=9.3,leading=13,
                              textColor=colors.HexColor('#253e3a'),spaceAfter=8))
    styles.add(ParagraphStyle('H1IDM',fontName='Helvetica-Bold',fontSize=20,leading=25,
                              textColor=colors.HexColor('#145e55'),spaceAfter=18))
    styles.add(ParagraphStyle('H2IDM',fontName='Helvetica-Bold',fontSize=13,leading=17,spaceAfter=10,
                              textColor=colors.HexColor('#145e55')))
    styles.add(ParagraphStyle('H3IDM',fontName='Helvetica-Bold',fontSize=10,leading=14,spaceBefore=6,spaceAfter=6))
    styles.add(ParagraphStyle('CellIDM',fontName='Helvetica',fontSize=7.6,leading=10,spaceAfter=0,
                              textColor=colors.HexColor('#253e3a')))
    width=A4[0]-84
    flow=[];lines=source.read_text(encoding='utf-8').splitlines();i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line: i+=1;continue
        if line=='<!-- PAGE -->':flow.append(PageBreak());i+=1;continue
        image_match=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',line)
        if image_match:
            p=source.parent/image_match.group(2);w,h=PILImage.open(p).size
            scale=min(width/w,315/h)
            flow.extend([Image(str(p),width=w*scale,height=h*scale),Spacer(1,10)])
            i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                row=[v.strip() for v in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',v) for v in row):
                    rows.append(row)
                i+=1
            n=len(rows[0]);weights=[max(7,min(45,max(len(r[j]) for r in rows))) for j in range(n)]
            total=sum(weights); widths=[width*v/total for v in weights]
            cells=[[Paragraph(inline(v),styles['CellIDM']) for v in r] for r in rows]
            t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e2ede6')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),
                ('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),
                ('BOTTOMPADDING',(0,0),(-1,-1),6),
                ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#a6bcb0')),
                ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#dce5de'))]))
            flow.extend([t,Spacer(1,10)]);continue
        heading=re.match(r'^(#{1,3}) (.*)',line)
        if heading:
            flow.append(Paragraph(inline(heading.group(2)),styles[f'H{len(heading.group(1))}IDM']))
            i+=1;continue
        parts=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','!','<!--')):
            parts.append(lines[i].strip());i+=1
        flow.append(Paragraph(inline(' '.join(parts)),styles['BodyIDM']))
    def footer(canvas,doc):
        canvas.saveState();canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#74877d'))
        canvas.drawString(42,25,'IDM-Viz | Phase 3 | 15 November 2026')
        canvas.drawRightString(A4[0]-42,25,str(doc.page));canvas.restoreState()
    document=SimpleDocTemplate(str(target),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=40,bottomMargin=43,
        title='IDM-Viz: Phase 3 Implementation and Experimental Evaluation',
        author='Umar Behram (23I-2604); Ahmed Bilal (23I-2581)')
    document.build(flow,onFirstPage=footer,onLaterPages=footer)
    pdf=pdfium.PdfDocument(str(target));texts=[]
    for i in range(len(pdf)):
        page=pdf[i];page.render(scale=1.5).to_pil().save(qa/f'page-{i+1:02}.png')
        texts.append(page.get_textpage().get_text_range())
    (qa/'extracted_text.txt').write_text('\n\n'.join(texts),encoding='utf-8')
    record=dict(pages=len(pdf),render_seconds=time.perf_counter()-started,
        pdf_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        page_text_characters=[len(v) for v in texts])
    (source.parent/'pdf_render_record.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);a=p.parse_args();render(a.source)
