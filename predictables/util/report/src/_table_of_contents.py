from __future__ import annotations

from pathlib import Path
from reportlab.lib.styles import getSampleStyleSheet  # type: ignore[import-untyped]
from reportlab.platypus import Paragraph, SimpleDocTemplate, TableOfContents  # type: ignore[import-untyped]


def generate_table_of_contents(pdf_file: str, sections: list) -> SimpleDocTemplate:
    # Check if the PDF file already exists
    if Path(pdf_file).exists():
        # If it exists, create a new PDF document with the TOC as the second page
        doc = SimpleDocTemplate(pdf_file)
        doc.build([])
        doc.add_page()
    else:
        # If it doesn't exist, create a new PDF document
        doc = SimpleDocTemplate(pdf_file)

    # Create a list to hold the table of contents entries
    toc_entries: list[SimpleDocTemplate] = []

    # Create a stylesheet for the table of contents
    styles = getSampleStyleSheet()
    toc_style = styles["TOCHeading1"]

    # Create a TableOfContents object
    toc = TableOfContents()

    # Add the table of contents to the document
    doc.build([*toc_entries, toc])

    # Create a function to add entries to the table of contents
    def add_entry(text: str, level: int) -> None:
        """Add an entry to the table of contents."""
        entry = Paragraph(text, toc_style)
        toc.addEntry(level, entry, 1)

    # Generate the table of contents entries
    for section in sections:
        add_entry(section["title"], section["level"])

    # Return the generated table of contents
    return doc
