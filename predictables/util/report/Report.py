import copy
from typing import List, Union

import pandas as pd
import polars as pl
from reportlab.lib.pagesizes import inch, letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from predictables.util import to_pd_df


class Report:
    """
    A PredicTables Report object is used to create a pdf document. It is a wrapper around the reportlab library.
    It defines standard styles for headings, paragraphs, code blocks, and math blocks. It also defines a standard
    html-like syntax for creating links and lists. It is meant to be used in a chain, where each method returns
    the Report object itself. The Report object is then built by calling the `build` method.

    Methods
    -------
    set(**kwargs)
        Sets the document properties of the pdf document. Does not by itself make any visible changes to the document.
        Similar in use to `style`, but `set` is used to set document properties, while `style` is used to set stylesheet
        properties.
    style(tag, **kwargs)
        Styles the pdf document by updating the stylesheet with the keyword arguments passed in. This is used to
        change the font family, font size, etc. of the document. Similar in use to `set`, but `style` is used to set
        stylesheet properties, while `set` is used to set document properties.
    footer(*args) - NOT IMPLEMENTED
        Sets the footer of the pdf document. Every page except for the first page will have this footer, which
        takes the passed args and formats them left to right in the footer.
    heading(level, text, return_self=True)
        Adds a heading to the document that says, `text`. The heading is styled according to the level passed in.
        If `return_self` is True, returns the Report object itself. This is used by the h1, h2, h3, h4, and h5 methods,
        and is not meant to be called directly.
    h1(text, element_id=None)
        Adds a h1-style heading to the document that says, `text`. If an `element_id` is passed, creates a bookmark
        location to return to with an inner link.
    h2(text, element_id=None)
        Adds a h2-style heading to the document that says, `text`. If an `element_id` is passed, creates a bookmark
        location to return to with an inner link.
    h3(text, element_id=None)
        Adds a h3-style heading to the document that says, `text`. If an `element_id` is passed, creates a bookmark
        location to return to with an inner link.
    h4(text, element_id=None)
        Adds a h4-style heading to the document that says, `text`. If an `element_id` is passed, creates a bookmark
        location to return to with an inner link.
    h5(text, element_id=None)
        Adds a h5-style heading to the document that says, `text`. If an `element_id` is passed, creates a bookmark
        location to return to with an inner link.
    h6(text, element_id=None)
        Adds a h6-style heading to the document that says, `text`. If an `element_id` is passed, creates a bookmark
        location to return to with an inner link.
    p(text)
        Adds a paragraph to the document that says, `text`.
    text(text)
        Alias for `p`. Adds a paragraph to the document that says, `text`.
    paragraph(text)
        Alias for `p`. Adds a paragraph to the document that says, `text`.
    inner_link(text, inner_link)
        Creates a link to a defined inner location in the document. The link will say `text` and link to the inner
        location `inner_link`.
    ul(text)
    """

    def __init__(
        self,
        filename: str,
        margins: List[float] = None,
        pagesize=letter,
    ):
        if margins is None:
            margins = [0.5, 0.5, 0.5, 0.5]
        if margins is None:
            margins = [0.5, 0.5, 0.5, 0.5]
        if margins is None:
            margins = [0.5, 0.5, 0.5, 0.5]
        self.doc = SimpleDocTemplate(filename, pagesize=pagesize)
        self.elements = []
        self.styles = getSampleStyleSheet()
        self.doc.leftMargin = margins[0] * inch
        self.doc.rightMargin = margins[1] * inch
        self.doc.topMargin = margins[2] * inch
        self.doc.bottomMargin = margins[3] * inch

    def __copy__(self) -> "Report":
        new_report = self.__class__()
        new_report.elements = copy.copy(self.elements)

        return new_report

    def copy(self) -> "Report":
        return self.__copy__()

    def __deepcopy__(self, memo) -> "Report":
        new_report = self.__class__()
        new_report.elements = copy.deepcopy(self.elements)

        return new_report

    def deepcopy(self) -> "Report":
        return self.__deepcopy__()

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

    def style(self, tag, **kwargs):
        """
        Styles the pdf document by updating the stylesheet with the keyword
        arguments passed in. This is used to change the font family, font
        size, etc. of the document."""
        for k, v in kwargs.items():
            if hasattr(self.styles.get(tag), k):
                setattr(self.styles.get(tag), k, v)
            else:
                SyntaxWarning(
                    f"Attribute {k} is not a valid attribute of the stylesheet. Ignoring."
                )

        return self

    def footer(self, *args):
        """
        Sets the footer of the pdf document. Every page except for the first page will have this footer, which
        takes the passed args and formats them left to right in the footer.
        """
        # TODO: Implement footer
        return self

    def heading(self, level: int, text: str, return_self: bool = True):
        self.elements.append(Paragraph(text, self.styles["Heading" + str(level)]))
        if return_self:
            return self

    def h1(self, text: str, element_id: str = None) -> "Report":
        """
        Adds a h1-style heading to the document that says, `text`. If an `element_id`
        is passed, creates a bookmark location to return to with an inner link.

        Parameters
        ----------
        text : str
            The text to display in the heading.
        element_id : str, optional
            The id of the element to link to, by default None

        Returns
        -------
        Report
            The current Report object. This method should be called in a chain.
        """
        # Add element to chain
        self.heading(1, text, return_self=False)

        # Create bookmark, if necessary
        if element_id is not None:
            self.elements[-1].addBookmark(element_id, relative=1, level=0)
        return self

    def h2(self, text: str, element_id: str = None) -> "Report":
        """
        Adds a h2-style heading to the document that says, `text`. If an `element_id`
        is passed, creates a bookmark location to return to with an inner link.

        Parameters
        ----------
        text : str
            The text to display in the heading.
        element_id : str, optional
            The id of the element to link to, by default None

        Returns
        -------
        Report
            The current Report object. This method should be called in a chain.
        """
        # Add element to chain
        self.heading(2, text, return_self=False)

        # Create bookmark, if necessary
        if element_id is not None:
            self.elements[-1].addBookmark(element_id, relative=1, level=0)
        return self

    def h3(self, text: str, element_id: str = None) -> "Report":
        """
        Adds a h3-style heading to the document that says, `text`. If an `element_id`
        is passed, creates a bookmark location to return to with an inner link.

        Parameters
        ----------
        text : str
            The text to display in the heading.
        element_id : str, optional
            The id of the element to link to, by default None

        Returns
        -------
        Report
            The current Report object. This method should be called in a chain.
        """
        # Add element to chain
        self.heading(3, text, return_self=False)

        # Create bookmark, if necessary
        if element_id is not None:
            self.elements[-1].addBookmark(element_id, relative=1, level=0)
        return self

    def h4(self, text: str, element_id: str = None) -> "Report":
        """
        Adds a h4-style heading to the document that says, `text`. If an `element_id`
        is passed, creates a bookmark location to return to with an inner link.

        Parameters
        ----------
        text : str
            The text to display in the heading.
        element_id : str, optional
            The id of the element to link to, by default None

        Returns
        -------
        Report
            The current Report object. This method should be called in a chain.
        """
        # Add element to chain
        self.heading(4, text, return_self=False)

        # Create bookmark, if necessary
        if element_id is not None:
            self.elements[-1].addBookmark(element_id, relative=1, level=0)
        return self

    def h5(self, text: str, element_id: str = None) -> "Report":
        """
        Adds a h5-style heading to the document that says, `text`. If an `element_id`
        is passed, creates a bookmark location to return to with an inner link.

        Parameters
        ----------
        text : str
            The text to display in the heading.
        element_id : str, optional
            The id of the element to link to, by default None

        Returns
        -------
        Report
            The current Report object. This method should be called in a chain.

        See Also
        --------

        """
        # Add element to chain
        self.heading(5, text, return_self=False)

        # Create bookmark, if necessary
        if element_id is not None:
            self.elements[-1].addBookmark(element_id, relative=1, level=0)
        return self

    def h6(self, text: str, element_id: str = None) -> "Report":
        """
        Adds a h6-style heading to the document that says, `text`. If an `element_id`
        is passed, creates a bookmark location to return to with an inner link.

        Parameters
        ----------
        text : str
            The text to display in the heading.
        element_id : str, optional
            The id of the element to link to, by default None

        Returns
        -------
        Report
            The current Report object. This method should be called in a chain.

        See Also
        --------

        """
        # Add element to chain
        self.heading(6, text, return_self=False)

        # Create bookmark, if necessary
        if element_id is not None:
            self.elements[-1].addBookmark(element_id, relative=1, level=0)
        return self

    def p(self, text: str) -> "Report":
        """
        Adds a paragraph to the document that says, `text`.

        Parameters
        ----------
        text : str
            The text to display in the paragraph.

        Returns
        -------
        Report
            The current Report object. This method should be called in a chain.

        See Also
        --------
        text : Alias for `p`. Adds a paragraph to the document that says, `text`.

        Example
        -------
        >>> from PredicTables.util import Report
        >>> (
        ...    Report("test.pdf")
        ...     .p("This is a paragraph.")
        ...     .p("This is another paragraph.")
        ...     .build()
        ... ) # Will create a pdf called test.pdf with two paragraphs.
        """
        # Add element to chain
        self.elements.append(Paragraph(text, self.styles["Normal"]))
        return self

    def text(self, text: str) -> "Report":
        """Alias for `p`. Adds a paragraph to the document that says, `text`."""
        return self.p(text)

    def paragraph(self, text: str) -> "Report":
        """Alias for `p`. Adds a paragraph to the document that says, `text`."""
        return self.p(text)

    def inner_link(self, text: str, inner_link: str):
        """
        Creates a link to a defined inner location in the document. The link will say
        `text` and link to the inner location `inner_link`.

        Parameters
        ----------
        text : str
            The text to display in the link.
        inner_link : str
            The id of the element to link to.

        Returns
        -------
        Report
            The current Report object. This method should be called in a chain.

        Raises
        ------
        ValueError
            If the inner_link does not exist.

        Notes
        -----
        1. The inner_link need not be defined before the link is created. However,
           the inner_link must be defined before the pdf is built.
        2. The inner_link is styled as a hyperlink, so it will be blue and underlined.
        """
        # Check if inner_link exists
        if inner_link not in self.doc._namedDests:
            raise ValueError(
                f"inner_link {inner_link} does not exist. Please define it before using it."
            )

        # Add element to chain
        self.elements.append(
            Paragraph(f'<a href="#{inner_link}">{text}</a>', self.styles["Hyperlink"])
        )
        return self

    def ul(self, text: List[str], bullet_char: str = "\u2022") -> "Report":
        """
        Adds an unordered list to the document. For each item in `text`, a bullet point is added to the list.
        If a different bullet point character is desired, it can be passed in as `bullet_char`. Will default
        to the unicode bullet point character.

        Parameters
        ----------
        text : List[str]
            The items to add to the list.
        bullet_char : str, optional
            The character to use for the bullet point, by default "\u2022" (bullet point in unicode)

        Returns
        -------
        Report
            The current Report object. This method should be called in a chain.

        Example
        -------
        >>> from PredicTables.util import Report
        >>> (
        ...    Report("test.pdf")
        ...     .ul(["Item 1", "Item 2", "Item 3"])
        ...     .build()
        ... ) # Will create a pdf called test.pdf with an unordered list with three items.
        """
        for t in text:
            self.elements.append(Paragraph(f"{bullet_char} {t}", self.styles["Normal"]))
        return self

    def _number_style(self, n: int, style: str):
        """
        Returns an iterator of numbers in the given style.

        Parameters
        ----------
        n : int
            The number of numbers to return.
        style : str
            The style of the numbers. Must be one of "decimal", "lower-roman", "upper-roman", "lower-alpha", or "upper-alpha".

        Returns
        -------
        Iterator[str]
            An iterator of numbers in the given style.
        """
        import itertools

        def decimal_to_roman(number, is_upper=True):
            roman_symbols = [
                (1000, "M"),
                (900, "CM"),
                (500, "D"),
                (400, "CD"),
                (100, "C"),
                (90, "XC"),
                (50, "L"),
                (40, "XL"),
                (10, "X"),
                (9, "IX"),
                (5, "V"),
                (4, "IV"),
                (1, "I"),
            ]

            result = ""
            for value, symbol in roman_symbols:
                while number >= value:
                    result += symbol if is_upper else symbol.lower()
                    number -= value

            return result

        def decimal_to_abc(number, is_upper=True):
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

            result = ""
            while number > 0:
                number -= 1
                result += (
                    alphabet[number % 26] if is_upper else alphabet[number % 26].lower()
                )
                number //= 26

            return result[::-1]

        decimal_numb = itertools.count(1)

        if style == "decimal":
            return decimal_numb
        elif style == "lower-roman":
            return map(lambda x: decimal_to_roman(x, False), decimal_numb)
        elif style == "upper-roman":
            return map(lambda x: decimal_to_roman(x, True), decimal_numb)
        elif style == "lower-alpha":
            return map(lambda x: decimal_to_abc(x, False), decimal_numb)
        elif style == "upper-alpha":
            return map(lambda x: decimal_to_abc(x, True), decimal_numb)
        else:
            raise ValueError(f"Style {style} is not a valid number style.")

    def ol(self, text: List[str], number_style: str = "decimal") -> "Report":
        """
        Adds an ordered list to the document. For each item in `text`, an item number is added to the list.
        If a different number style is desired, it can be passed in as `number_style`. Will default
        to the decimal number style.

        Parameters
        ----------
        text : List[str]
            The items to add to the list.
        number_style : str, optional
            The style to use for the item numbers, by default "decimal", but also accepts "lower-roman",
            "upper-roman", "lower-alpha", and "upper-alpha".

        """
        styled_numbers = self._number_style(len(text), number_style)
        for _i, t in enumerate(text):
            self.elements.append(
                Paragraph(f"{next(styled_numbers)}. {t}", self.styles["Normal"])
            )
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
