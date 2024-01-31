import os

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, TableOfContents


def generate_table_of_contents(pdf_file, sections):
    # Check if the PDF file already exists
    if os.path.exists(pdf_file):
        # If it exists, create a new PDF document with the TOC as the second page
        doc = SimpleDocTemplate(pdf_file)
        doc.build([])
        doc.add_page()
    else:
        # If it doesn't exist, create a new PDF document
        doc = SimpleDocTemplate(pdf_file)

    # Create a list to hold the table of contents entries
    toc_entries = []

    # Create a stylesheet for the table of contents
    styles = getSampleStyleSheet()
    toc_style = styles["TOCHeading1"]

    # Create a TableOfContents object
    toc = TableOfContents()

    # Add the table of contents to the document
    doc.build(toc_entries + [toc])

    # Create a function to add entries to the table of contents
    def add_entry(text, level):
        entry = Paragraph(text, toc_style)
        toc.addEntry(level, entry, 1)

    # Generate the table of contents entries
    for section in sections:
        add_entry(section["title"], section["level"])

    # Return the generated table of contents
    return doc
