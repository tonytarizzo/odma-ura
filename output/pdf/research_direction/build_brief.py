"""Make a one-page supervisor handout from verified historical results."""

from pathlib import Path
import json

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from pypdf import PdfReader

OUT = Path(__file__).resolve().parent
INK, MUTED, GREEN = colors.HexColor("#172b3a"), colors.HexColor("#5c6978"), colors.HexColor("#22846b")


def paragraph(c, text, x, top, width, size=10, leading=13, color=INK):
    style = ParagraphStyle("body", fontName="Helvetica", fontSize=size, leading=leading, textColor=color)
    p = Paragraph(text, style); _, height = p.wrap(width, 1000)
    p.drawOn(c, x, top-height)
    return top-height


def main():
    evidence = json.loads((OUT / "plot_data.json").read_text())
    assert len(evidence["production"]) == 48
    path = OUT / "supervisor_brief.pdf"
    width, height = landscape(A4)
    c = canvas.Canvas(str(path), pagesize=(width, height))
    c.setTitle("Why the project now focuses on polar multiuser decoding")
    c.setAuthor("ODMA-URA research")
    c.setFillColor(INK); c.setFont("Helvetica-Bold", 22)
    c.drawString(29, height-43, "Why we now focus on polar multiuser decoding")
    paragraph(c, "The largest measured gain came from better inference. A modest energy correction then brought polar close to dense.",
              30, height-56, width-60, size=11, leading=14)
    c.drawImage(str(OUT / "research_direction.png"), 22, 236, width=width-44, height=(width-44)*5.4/17.5, mask="auto")
    top, gap = 211, 22
    colwidth = (width-60-2*gap)/3
    columns = [
        ("Pause the hash search", "Hash J8 retains nearly the same mean PUPE as its stored-amplitude parent: <b>0.29361 vs 0.29331</b>, with 64x fewer amplitude values at B14. It still needs a multiuser inverse. This is a priority decision: compact encoding alone did not solve recovery."),
        ("Keep polar; simplify energy shaping", "Learned polar energies reach <b>0.28776 vs dense 0.28526</b> over the full B14 grid, and improve every seed at B12/B14. Balancing explains much of the gain. The shared mixer worsened these runs. <b>Placements stayed fixed</b>; learning or removing ODMA remains untested."),
        ("Spend the next effort on inference", "On identical polar waveforms, D2 cuts stress-point PUPE <b>0.559 to 0.137</b>. A later observation-only refinement roughly halves D2's remaining error for dense and shaped polar. The current work derives a joint polar receiver with explicit accuracy and cost accounting."),
    ]
    bottoms = []
    for idx, (heading, text) in enumerate(columns):
        x = 30+idx*(colwidth+gap)
        c.setFillColor(GREEN); c.setFont("Helvetica-Bold", 11.5); c.drawString(x, top, heading)
        bottoms.append(paragraph(c, text, x, top-11, colwidth, size=10, leading=13.5))
    assert min(bottoms) > 63, bottoms
    c.setStrokeColor(colors.HexColor("#d8dfe4")); c.line(30, 59, width-30, 59)
    paragraph(c, "Scope: historical small-B adaptations; real AWGN; unit-energy waveforms; known K/noise. D2 and refinement enumerate the alphabet. "
              "Close means do not prove equivalence or optimality. Polar gives useful structure; scalable multiuser decoding remains the research problem.",
              30, 51, width-60, size=8.3, leading=10, color=MUTED)
    c.setFillColor(MUTED); c.setFont("Helvetica", 7.5)
    c.drawString(30, 14, "Sources: Jobs 032-034; original grids plus 10 Oct paired diagnostic. Plot data and source hashes saved alongside. | 18 October 2026")
    c.showPage(); c.save()
    reader = PdfReader(str(path)); assert len(reader.pages) == 1
    text = " ".join(reader.pages[0].extract_text().split())
    assert all(s in text for s in ("Pause the hash search", "Placements stayed fixed", "0.29361", "0.28776"))
    print(path)


if __name__ == "__main__": main()
