from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter, inch

import pandas as pd
import polars as pl

from typing import Union, List

from PredicTables.util import to_pd_df


class Report:
    def __init__(self, filename: str, margins: List[float] = [0.5, 0.5, 0.5, 0.5]):
        self.doc = SimpleDocTemplate(filename, pagesize=letter)
        self.elements = []
        self.styles = getSampleStyleSheet()
        self.doc.leftMargin = margins[0] * inch
        self.doc.rightMargin = margins[1] * inch
        self.doc.topMargin = margins[2] * inch
        self.doc.bottomMargin = margins[3] * inch

    def set(self, **kwargs):
        """Sets the document properties of the pdf document. Does not by itself make any visible changes to the document."""
        for k, v in kwargs.items():
            if hasattr(self.doc, k):
                setattr(self.doc, k, v)
            else:
                SyntaxWarning(
                    f"Attribute {k} is not a valid attribute of the document. Ignoring."
                )

        return self

    def footer(self, *args):
        """
        Sets the footer of the pdf document. Every page except for the first page will have this footer, which
        takes the passed args and formats them left to right in the footer.
        """

    def heading(self, level: int, text: str, return_self: bool = True):
        self.elements.append(Paragraph(text, self.styles["Heading" + str(level)]))
        if return_self:
            return self

    def h1(self, text: str):
        """Adds a h1-style heading to the document that says, `text`."""
        self.heading(1, text, return_self=False)
        return self

    def h2(self, text: str):
        """Adds a h2-style heading to the document that says, `text`."""
        self.heading(2, text, return_self=False)
        return self

    def h3(self, text: str):
        """Adds a h3-style heading to the document that says, `text`."""
        self.heading(3, text, return_self=False)
        return self

    def h4(self, text: str):
        """Adds a h4-style heading to the document that says, `text`."""
        self.heading(4, text, return_self=False)
        return self

    def h5(self, text: str):
        """Adds a h5-style heading to the document that says, `text`."""
        self.heading(5, text, return_self=False)
        return self

    def p(self, text: str):
        """Adds a paragraph to the document that says, `text`."""
        self.elements.append(Paragraph(text, self.styles["Normal"]))
        return self

    def inner_a(self, text: str, inner_link: str):
        self.elements.append(Paragraph(text, self.styles["Normal"]))
        return self

    def text(self, text: str):
        """Alias for `p`. Adds a paragraph to the document that says, `text`."""
        return self.p(text)

    def ul(self, text: List[str]):
        """Adds an unordered list to the document. For each item in `text`, a bullet point is added to the list."""
        for t in text:
            bullet_char = "\u2022"
            self.elements.append(Paragraph(f"{bullet_char} {t}", self.styles["Normal"]))
        return self

    def ol(self, text: List[str]):
        """Adds an ordered list to the document. For each item in `text`, an item number is added to the list."""
        for i, t in enumerate(text):
            self.elements.append(Paragraph(f"{i+1}. {t}", self.styles["Normal"]))
        return self

    def code(self, text: str):
        """Adds a code block to the document. Used for displaying code snippets."""
        self.elements.append(Paragraph(text, self.styles["Code"]))
        return self

    def math(self, mathjax: str):
        """Adds a math block to the document. Used for displaying math equations."""
        self.elements.append(Paragraph(mathjax, self.styles["Math"]))
        return self

    def spacer(self, height: float):
        """Adds a spacer to the document. Used for adding vertical space between elements. Height is in inches."""
        self.elements.append(Spacer(1, height))
        return self

    def image_no_caption(self, filename: str, width: float, height: float):
        """Adds an image to the document. Used for adding images to the document. Width and height are in inches."""
        self.elements.append(Image(filename, width, height))
        return self

    def plot_no_caption(self, func, width: float, height: float):
        """Adds a plot to the document. Used for adding plots to the document. Width and height are in inches."""
        self.elements.append(func(width, height))
        return self

    def page_break(self):
        """Adds a page break to the document. Used for adding a page break to the document."""
        self.elements.append(PageBreak())
        return self

    def table(
        self, df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame], style: TableStyle
    ):
        df = to_pd_df(df)
        data = [df.columns[:,].tolist()] + df.values.tolist()
        t = Table(data)
        t.setStyle(style)
        self.elements.append(t)
        return self

    def build(self):
        """Builds the pdf document and saves it to the filename specified in the constructor. This is the final command that must be called to generate the pdf document."""
        self.doc.build(self.elements)

    def restyle(self, tag, **kwargs):
        """Restyles the pdf document by updating the stylesheet with the keyword arguments passed in. This is used to change the font family, font size, etc. of the document."""
        for k, v in kwargs.items():
            if hasattr(self.styles.get(tag), k):
                setattr(self.styles.get(tag), k, v)
            else:
                SyntaxWarning(
                    f"Attribute {k} is not a valid attribute of the stylesheet. Ignoring."
                )

        return self

    def title(self, text: str):
        """Sets the title metadata attribute of the pdf document. Does not by itself make any visible changes to the document."""
        self.doc.title = text
        return self

    def author(self, text: str):
        """Sets the author metadata attribute of the pdf document. Does not by itself make any visible changes to the document."""
        self.doc.author = text
        return self

    def subject(self, text: str):
        """Sets the subject metadata attribute of the pdf document. Does not by itself make any visible changes to the document."""
        self.doc.subject = text
        return self

    def keywords(self, text: str):
        """Sets the keywords metadata attribute of the pdf document. Does not by itself make any visible changes to the document."""
        self.doc.keywords = text
        return self
