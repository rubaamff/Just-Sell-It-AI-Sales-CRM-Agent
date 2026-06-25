"""
proposal_pdf.py
================
Converts a generated proposal (plain text from the Proposal Agent) into a
clean, branded PDF file the user can download.
"""

import re
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable
)

# Section headings the agent is instructed to produce (see proposal_agent.py)
SECTION_HEADINGS = [
    "Introduction",
    "Discovered Challenges",
    "Proposed Solutions from Beam Data",
    "Pricing",
    "Next Step",
]


def _build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ProposalTitle",
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#3a2d7a"),
        spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="ProposalSubtitle",
        fontName="Helvetica",
        fontSize=11,
        textColor=colors.HexColor("#666666"),
        spaceAfter=20,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading",
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=colors.HexColor("#667eea"),
        spaceBefore=18,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="ProposalBody",
        fontName="Helvetica",
        fontSize=10.5,
        leading=16,
        textColor=colors.HexColor("#222222"),
        spaceAfter=6,
    ))
    return styles


def _escape(text: str) -> str:
    """Minimal escaping for ReportLab's mini-HTML paragraph markup."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _parse_sections(raw_text: str):
    """
    Splits the agent's plain-text proposal into (heading, body) pairs based
    on the known section headings. Falls back to a single 'Proposal' section
    if no headings are recognised.
    """
    pattern = r"(?im)^\s*#{0,3}\s*(" + "|".join(SECTION_HEADINGS) + r")\s*:?\s*$"
    matches = list(re.finditer(pattern, raw_text))

    if not matches:
        return [("Proposal", raw_text.strip())]

    sections = []
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        body = raw_text[start:end].strip()
        sections.append((heading, body))
    return sections


def generate_proposal_pdf(company_name: str, proposal_text: str, output_path: str) -> str:
    """
    Renders the agent-generated proposal text into a styled PDF file.

    Parameters
    ----------
    company_name : Name of the target company (used in the title).
    proposal_text : Raw text returned by the Proposal Agent.
    output_path   : Where to write the .pdf file.

    Returns the output_path for convenience.
    """
    styles = _build_styles()
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=0.85 * inch,
        bottomMargin=0.85 * inch,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
    )

    story = []
    story.append(Paragraph("Beam Data", styles["ProposalTitle"]))
    story.append(Paragraph(f"Proposal prepared for {_escape(company_name)}", styles["ProposalSubtitle"]))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#cccccc"), thickness=0.7))
    story.append(Spacer(1, 12))

    for heading, body in _parse_sections(proposal_text):
        story.append(Paragraph(_escape(heading), styles["SectionHeading"]))
        for para in body.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            # Convert simple "- " bullet lines into one paragraph with line breaks
            para_html = _escape(para).replace("\n", "<br/>")
            story.append(Paragraph(para_html, styles["ProposalBody"]))

    doc.build(story)
    return output_path
