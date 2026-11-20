"""Render the Phase 3 report PDF and save page images for visual review."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from xml.sax.saxutils import escape

from PIL import Image as PILImage
import pypdfium2 as pdfium
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Preformatted, Table, TableStyle)


def inline(source: str) -> str:
    text = escape(source)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def render(source: Path):
    started = time.perf_counter()
    source = Path(source)
    target = source.with_suffix(".pdf")
    qa_dir = source.parent / "qa"
    qa_dir.mkdir(exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("BodyPhase3", fontName="Helvetica", fontSize=9.2, leading=13,
                              textColor=colors.HexColor("#253e3a"), spaceAfter=8))
    styles.add(ParagraphStyle("H1Phase3", fontName="Helvetica-Bold", fontSize=20, leading=25,
                              textColor=colors.HexColor("#145e55"), spaceAfter=16))
    styles.add(ParagraphStyle("H2Phase3", fontName="Helvetica-Bold", fontSize=13, leading=17,
                              spaceAfter=9, textColor=colors.HexColor("#145e55")))
    styles.add(ParagraphStyle("H3Phase3", fontName="Helvetica-Bold", fontSize=10, leading=14,
                              spaceBefore=6, spaceAfter=6))
    styles.add(ParagraphStyle("CellPhase3", fontName="Helvetica", fontSize=7.6, leading=10,
                              spaceAfter=0, textColor=colors.HexColor("#253e3a")))
    styles.add(ParagraphStyle("CodePhase3", fontName="Courier", fontSize=7.0, leading=9,
                              leftIndent=6, rightIndent=6, borderPadding=5,
                              backColor=colors.HexColor("#f1f5f2"), spaceAfter=8))
    width = A4[0] - 84
    flow = []
    lines = source.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line == "<!-- PAGE -->":
            flow.append(PageBreak())
            index += 1
            continue
        if line.startswith("```"):
            code = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(lines[index].rstrip())
                index += 1
            if index < len(lines):
                index += 1
            flow.extend([Preformatted("\n".join(code), styles["CodePhase3"]), Spacer(1, 5)])
            continue
        image_match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line)
        if image_match:
            path = source.parent / image_match.group(2)
            image_width, image_height = PILImage.open(path).size
            scale = min(width / image_width, 330 / image_height)
            flow.extend([Image(str(path), width=image_width * scale, height=image_height * scale),
                         Spacer(1, 8)])
            index += 1
            continue
        if line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                row = [value.strip() for value in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", value) for value in row):
                    rows.append(row)
                index += 1
            count = len(rows[0])
            weights = [max(7, min(45, max(len(row[column]) for row in rows))) for column in range(count)]
            column_widths = [width * value / sum(weights) for value in weights]
            cells = [[Paragraph(inline(value), styles["CellPhase3"]) for value in row] for row in rows]
            table = Table(cells, colWidths=column_widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2ede6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#a6bcb0")),
                ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#dce5de"))]))
            flow.extend([table, Spacer(1, 8)])
            continue
        heading = re.match(r"^(#{1,3}) (.*)", line)
        if heading:
            flow.append(Paragraph(inline(heading.group(2)),
                                  styles[f"H{len(heading.group(1))}Phase3"]))
            index += 1
            continue
        parts = [line]
        index += 1
        while index < len(lines) and lines[index].strip() and not lines[index].startswith(("#", "|", "!", "<!--")):
            parts.append(lines[index].strip())
            index += 1
        flow.append(Paragraph(inline(" ".join(parts)), styles["BodyPhase3"]))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#74877d"))
        canvas.drawString(42, 25, "Data Analysis and Visualization | Phase 3")
        canvas.drawRightString(A4[0] - 42, 25, str(document.page))
        canvas.restoreState()

    document = SimpleDocTemplate(str(target), pagesize=A4, leftMargin=42, rightMargin=42,
        topMargin=40, bottomMargin=43,
        title="Phase 3: Research Implementation and Experimental Evaluation",
        author="Ahmed Bilal (23I-2581); Umar Behram (23I-2604)")
    document.build(flow, onFirstPage=footer, onLaterPages=footer)
    pdf = pdfium.PdfDocument(str(target))
    page_text = []
    for page_number in range(len(pdf)):
        page = pdf[page_number]
        page.render(scale=1.5).to_pil().save(qa_dir / f"page-{page_number + 1:02}.png")
        page_text.append(page.get_textpage().get_text_range())
    (qa_dir / "extracted_text.txt").write_text("\n\n".join(page_text), encoding="utf-8")
    record = dict(pages=len(pdf), render_seconds=time.perf_counter() - started,
        pdf_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        page_text_characters=[len(text) for text in page_text])
    (source.parent / "pdf_render_record.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    render(args.source)
