from __future__ import annotations

import copy
import datetime
import itertools
from pathlib import Path
import typing
import uuid
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import pygments
from reportlab.lib.colors import black, lightgrey, white  # type: ignore[import-untyped]
from reportlab.lib.enums import TA_CENTER  # type: ignore[import-untyped]
from reportlab.lib.pagesizes import inch, letter  # type: ignore[import-untyped]
from reportlab.lib.styles import (  # type: ignore[import-untyped]
    ParagraphStyle,
    getSampleStyleSheet,
)
from reportlab.platypus import (  # type: ignore[import-untyped]
    Flowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from predictables.util.src._to_pd import to_pd_df


class Report:
    """Create an html-styled pdf document.

    A predictables Report object is used to create a pdf document. It is a
    wrapper around the reportlab library. It defines standard styles for
    headings, paragraphs, code blocks, and math blocks. It also defines a
    standard html-like syntax for creating links and lists. It is meant to
    be used in a chain, where each method returns the Report object itself.
    The Report object is then built by calling the `build` method.

    Attributes
    ----------
    filename : str
        The name of the pdf file to create.
    margins : list[float]
        The margins of the pdf document. Defaults to [0.5, 0.5, 0.5, 0.5]
        if not specified. The order is [left, right, top, bottom], and
        the units are inches.
    doc : SimpleDocTemplate
        The reportlab SimpleDocTemplate object that is used to create
        the pdf.
    elements : list[Flowable]
        The list of elements that will be added to the pdf document. Starts
        blank, and elements are added to it as the Report object is built.
    styles : SampleStyleSheet
        The reportlab SampleStyleSheet object that is used to style the
        pdf document. Starts with default styles, and styles are added to
        it as the Report object is built.


    Methods
    -------
    set(**kwargs)
        Sets the document properties of the pdf document. Does not by
        itself make any visible changes to the document.
        Similar in use to `style`, but `set` is used to set document
        properties, while `style` is used to set stylesheet properties.
    style(tag, **kwargs)
        Styles the pdf document by updating the stylesheet with the keyword
        arguments passed in. This is used to change the font family, font
        size, etc. of the document. Similar in use to `set`, but `style`
        is used to set stylesheet properties, while `set` is used to set
        document properties.
    footer(*args) - NOT IMPLEMENTED
        Sets the footer of the pdf document. Every page except for the
        first page will have this footer, which takes the passed args
        and formats them left to right in the footer.
    heading(level, text, return_self=True)
        Adds a heading to the document that says, `text`. The heading
        is styled according to the level passed in. If `return_self` is
        True, returns the Report object itself. This is used by the h1,
        h2, h3, h4, and h5 methods, and is not meant to be called directly.
    h1(text, element_id=None)
        Adds a h1-style heading to the document that says, `text`. If an
        `element_id` is passed, creates a bookmark location to return to
        with an inner link.
    h2(text, element_id=None)
        Adds a h2-style heading to the document that says, `text`. If an
        `element_id` is passed, creates a bookmark location to return to
        with an inner link.
    h3(text, element_id=None)
        Adds a h3-style heading to the document that says, `text`. If an
        `element_id` is passed, creates a bookmark location to return to
        with an inner link.
    h4(text, element_id=None)
        Adds a h4-style heading to the document that says, `text`. If an
        `element_id` is passed, creates a bookmark location to return to
        with an inner link.
    h5(text, element_id=None)
        Adds a h5-style heading to the document that says, `text`. If an
        `element_id` is passed, creates a bookmark location to return to
        with an inner link.
    h6(text, element_id=None)
        Adds a h6-style heading to the document that says, `text`. If an
        `element_id` is passed, creates a bookmark location to return to
        with an inner link.
    p(text)
        Adds a paragraph to the document that says, `text`.
    text(text)
        Alias for `p`. Adds a paragraph to the document that says, `text`.
    paragraph(text)
        Alias for `p`. Adds a paragraph to the document that says, `text`.
    inner_link(text, inner_link)
        Creates a link to a defined inner location in the document. The link
        will say `text` and link to the inner location `inner_link`.
    ul(text)
    """

    def __init__(
        self,
        filename: str,
        margins: list[float] | None = None,
        pagesize: tuple = letter,
        dpi: int = 200,
    ):
        """Create a Report object that can be used to create a pdf document.

        Parameters
        ----------
        filename : str
            The name of the pdf file to create.
        margins : list[float], optional
            The margins of the pdf document. Defaults to [0.5, 0.5, 0.5, 0.5]
            if not specified. The order is [left, right, top, bottom], and
            the units are inches.
        pagesize : tuple, optional
            The size of the pages in the pdf document. Defaults to letter
            size if not specified. The units are inches.
        dpi : int, optional
            The dpi of the images in the pdf document. Defaults to 200 if
            not specified.

        Returns
        -------
        None. Initializes the Report object, but need to call `build` to
        actually create the pdf document.
        """
        self.filename = filename
        self.pagesize = pagesize
        self.dpi = dpi

        if margins is None:
            margins = [0.5, 0.5, 0.5, 0.5]
        self.doc = SimpleDocTemplate(filename, pagesize=pagesize)
        self.elements: list[Flowable] = []
        self.styles = getSampleStyleSheet()
        self.doc.leftMargin = margins[0] * inch
        self.doc.rightMargin = margins[1] * inch
        self.doc.topMargin = margins[2] * inch
        self.doc.bottomMargin = margins[3] * inch

    def __copy__(self) -> "Report":
        """Create a shallow copy of the Report object."""
        new_report = self.__class__(
            filename=f"{self.filename.replace('.pdf', '')}-COPY.pdf"
        )
        new_report.elements = copy.copy(self.elements)

        return new_report

    def copy(self) -> "Report":
        """Create a shallow copy of the Report object."""
        return self.__copy__()

    def __deepcopy__(self, memo: dict) -> "Report":
        """Create a deep copy of the Report object."""
        new_report = self.__class__(
            filename=f"{self.filename.replace('.pdf', '')}-COPY.pdf"
        )
        new_report.elements = copy.deepcopy(self.elements)

        return new_report

    def deepcopy(self) -> "Report":
        """Create a deep copy of the Report object."""
        return self.__deepcopy__({})

    def set(self, **kwargs) -> "Report":
        """Set the document properties of the pdf document.

        Does not by itself make any visible changes to the document.
        """
        for k, v in kwargs.items():
            if hasattr(self.doc, k):
                setattr(self.doc, k, v)
            else:
                warnings.warn(
                    f"Attribute {k} is not a valid attribute of the document. "
                    "Ignoring.",
                    SyntaxWarning,
                    stacklevel=2,
                )

        return self

    def style(self, tag: str, **kwargs) -> "Report":
        """Apply styles to the pdf document.

        Styles the pdf document by updating the stylesheet with the keyword
        arguments passed in. This is used to change the font family, font
        size, etc. of the document.
        """
        for k, v in kwargs.items():
            if hasattr(self.styles.get(tag), k):
                setattr(self.styles.get(tag), k, v)
            else:
                warnings.warn(
                    f"Attribute {k} is not a valid attribute of the stylesheet. "
                    "Ignoring.",
                    SyntaxWarning,
                    stacklevel=2,
                )

        return self

    def footer(self) -> "Report":
        """Set the footer of the pdf document.

        Every page except for the first page will have this footer,
        which takes the passed args and formats them left to right
        in the footer.
        """
        # TODO(@<aaweaver-actuary>):  Implement footer method
        # https://github.com/predictables/predictables/issues/70
        return self

    def heading(
        self, level: int, text: str, return_self: bool = True
    ) -> "Report" | None:
        """Add a heading to the document that says, `text`.

        The heading is styled according to the level passed in.

        Parameters
        ----------
        level : int
            The level of the heading. Must be between 1 and 6, inclusive.
        text : str
            The text to display in the heading.
        return_self : bool, optional
            Whether or not to return the Report object itself. If True,
            returns the Report object itself. If False, returns None.
            Defaults to True.

        Returns
        -------
        Optional[Report]
            The current Report object. This method should be called in a
            chain.
        """
        self.elements.append(Paragraph(text, self.styles[f"Heading{level}"]))
        return self if return_self else None

    def h1(self, text: str, element_id: str | None = None) -> "Report":
        """Add a h1-style heading to the document that says, `text`.

        If an `element_id` is passed, creates a bookmark location to return to
        with an inner link.

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
        heading : Adds a heading to the document that says, `text`.

        Examples
        --------
        >>> from predictables.util import Report
        >>> (
        ...     Report("test.pdf")
        ...     .h1("This is a heading.")
        ...     .h1("This is another heading.")
        ...     .build()
        ... )  # Will create a pdf called test.pdf with two headings.

        >>> (
        ...     Report("test_with_bookmark.pdf")
        ...     .h1("This is a heading.", element_id="test")
        ...     .h1("This is another heading.")
        ...     .inner_link("Go to the first heading", "test")
        ...     .build()
        ... )  # Will create a pdf called test_with_bookmark.pdf with two
              # headings and a link at the end pointing to the first
              # heading.
        """
        # Add element to chain
        self.heading(1, text, return_self=False)

        # Create bookmark, if necessary
        if element_id is not None:
            self.elements[-1].addBookmark(element_id, relative=1, level=0)
        return self

    def h2(self, text: str, element_id: str | None = None) -> "Report":
        """Add a h2-style heading to the document that says, `text`.

        If an `element_id` is passed, creates a bookmark location to return to
        with an inner link.

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
        heading : Adds a heading to the document that says, `text`.

        Example
        -------
        >>> from predictables.util import Report
        >>> (
        ...     Report("test.pdf")
        ...     .h1("Top-level heading.")
        ...     .h2("Second-level heading.")
        ...     .h3("Third-level heading.")
        ...     .h4("Fourth-level heading.")
        ...     .h5("Fifth-level heading.")
        ...     .h6("Sixth-level heading.")
        ...     .build()
        ... )  # Will create a pdf called test.pdf with six headings.
        """
        # Add element to chain
        self.heading(2, text, return_self=False)

        # Create bookmark, if necessary
        if element_id is not None:
            self.elements[-1].addBookmark(element_id, relative=1, level=0)
        return self

    def h3(self, text: str, element_id: str | None = None) -> "Report":
        """Add a h3-style heading to the document that says, `text`.

        If an `element_id` is passed, creates a bookmark location to return to with
        an inner link.

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
        heading : Adds a heading to the document that says, `text`.

        Example
        -------
        >>> from predictables.util import Report
        >>> (
        ...     Report("test.pdf")
        ...     .h1("Top-level heading.")
        ...     .h2("Second-level heading.")
        ...     .h3("Third-level heading.")
        ...     .h4("Fourth-level heading.")
        ...     .h5("Fifth-level heading.")
        ...     .h6("Sixth-level heading.")
        ...     .build()
        ... )  # Will create a pdf called test.pdf with six headings.
        """
        # Add element to chain
        self.heading(3, text, return_self=False)

        # Create bookmark, if necessary
        if element_id is not None:
            self.elements[-1].addBookmark(element_id, relative=1, level=0)
        return self

    def h4(self, text: str, element_id: str | None = None) -> "Report":
        """Add a h4-style heading to the document that says, `text`.

        If an `element_id` is passed, creates a bookmark location to return to with an inner link.

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
        heading : Adds a heading to the document that says, `text`.

        Example
        -------
        >>> from predictables.util import Report
        >>> (
        ...     Report("test.pdf")
        ...     .h1("Top-level heading.")
        ...     .h2("Second-level heading.")
        ...     .h3("Third-level heading.")
        ...     .h4("Fourth-level heading.")
        ...     .h5("Fifth-level heading.")
        ...     .h6("Sixth-level heading.")
        ...     .build()
        ... )  # Will create a pdf called test.pdf with six headings.
        """
        # Add element to chain
        self.heading(4, text, return_self=False)

        # Create bookmark, if necessary
        if element_id is not None:
            self.elements[-1].addBookmark(element_id, relative=1, level=0)
        return self

    def h5(self, text: str, element_id: str | None = None) -> "Report":
        """Add a h5-style heading to the document that says, `text`.

        If an `element_id` is passed, creates a bookmark location to return to with an inner link.

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
        heading : Adds a heading to the document that says, `text`.

        Example
        -------
        >>> from predictables.util import Report
        >>> (
        ...     Report("test.pdf")
        ...     .h1("Top-level heading.")
        ...     .h2("Second-level heading.")
        ...     .h3("Third-level heading.")
        ...     .h4("Fourth-level heading.")
        ...     .h5("Fifth-level heading.")
        ...     .h6("Sixth-level heading.")
        ...     .build()
        ... )  # Will create a pdf called test.pdf with six headings.
        """
        # Add element to chain
        self.heading(5, text, return_self=False)

        # Create bookmark, if necessary
        if element_id is not None:
            self.elements[-1].addBookmark(element_id, relative=1, level=0)
        return self

    def h6(self, text: str, element_id: str | None = None) -> "Report":
        """Add a h6-style heading to the document that says, `text`.

        If an `element_id` is passed, creates a bookmark location to
        return to with an inner link.

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
        heading : Adds a heading to the document that says, `text`.

        Example
        -------
        >>> from predictables.util import Report
        >>> (
        ...     Report("test.pdf")
        ...     .h1("Top-level heading.")
        ...     .h2("Second-level heading.")
        ...     .h3("Third-level heading.")
        ...     .h4("Fourth-level heading.")
        ...     .h5("Fifth-level heading.")
        ...     .h6("Sixth-level heading.")
        ...     .build()
        ... )  # Will create a pdf called test.pdf with six headings.
        """
        # Add element to chain
        self.heading(6, text, return_self=False)

        # Create bookmark, if necessary
        if element_id is not None:
            self.elements[-1].addBookmark(element_id, relative=1, level=0)
        return self

    def p(self, text: str) -> "Report":
        """Add a paragraph to the document that says, `text`.

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
        paragraph : Alias for `p`. Adds a paragraph to the document that says, `text`.

        Example
        -------
        >>> from predictables.util import Report
        >>> (
        ...     Report("test.pdf")
        ...     .p("This is a paragraph.")
        ...     .p("This is another paragraph.")
        ...     .text("This is a third paragraph with a different method.")
        ...     .paragraph("This is a fourth paragraph with a different method.")
        ...     .build()
        ... )  # Will create a pdf called test.pdf with four paragraphs that are
              # all the same, even though they were created with different methods.
        """
        # Add element to chain
        self.elements.append(Paragraph(text, self.styles["Normal"]))
        return self

    def text(self, text: str) -> "Report":
        """Define an alias for `p`.

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
        p : `text` is an alias for `p`.
            Adds a paragraph to the document that says, `text`.
        paragraph : Another alias for `p`.
            Adds a paragraph to the document that says, `text`.

        Example
        -------
        >>> from predictables.util import Report
        >>> (
        ...     Report("test.pdf")
        ...     .p("This is a paragraph.")
        ...     .p("This is another paragraph.")
        ...     .text("This is a third paragraph with a different method.")
        ...     .paragraph("This is a fourth paragraph with a different method.")
        ...     .build()
        ... )  # Will create a pdf called test.pdf with four paragraphs that are
              # all the same, even though they were created with different methods.
        """
        return self.p(text)

    def paragraph(self, text: str) -> "Report":
        """Define an alias for `p`.

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
        p : `text` is an alias for `p`.
            Adds a paragraph to the document that says, `text`.
        paragraph : Another alias for `p`.
            Adds a paragraph to the document that says, `text`.

        Example
        -------
        >>> from predictables.util import Report
        >>> (
        ...     Report("test.pdf")
        ...     .p("This is a paragraph.")
        ...     .p("This is another paragraph.")
        ...     .text("This is a third paragraph with a different method.")
        ...     .paragraph("This is a fourth paragraph with a different method.")
        ...     .build()
        ... )  # Will create a pdf called test.pdf with four paragraphs that are
              # all the same, even though they were created with different methods.
        """
        return self.p(text)

    def inner_link(self, text: str, inner_link: str) -> "Report":
        """Create a link to a defined inner location in the document.

        The link will say `text` and link to the inner location `inner_link`.

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
        if inner_link not in self.doc._namedDests:  # noqa: SLF001
            raise ValueError(
                f"inner_link {inner_link} does not exist. "
                "Please define it before using it."
            )

        # Add element to chain
        self.elements.append(
            Paragraph(f'<a href="#{inner_link}">{text}</a>', self.styles["Hyperlink"])
        )
        return self

    def ul(self, text: list[str], bullet_char: str = "\u2022") -> "Report":
        """Add an unordered list to the document.

        For each item in `text`, a bullet point is added to the list. If
        a different bullet point character is desired, it can be passed in
        as `bullet_char`. Will default to the unicode bullet point character.

        Parameters
        ----------
        text : list[str]
            The items to add to the list.
        bullet_char : str, optional
            The character to use for the bullet point, by default "\u2022"
            (bullet point in unicode)

        Returns
        -------
        Report
            The current Report object. This method should be called in a chain.

        See Also
        --------
        ol : Adds an ordered list to the document. For each item in `text`, an
             item number is added to the list.

        Example
        -------
        >>> from predictables.util import Report
        >>> (
        ...     Report("test.pdf").ul(["Item 1", "Item 2", "Item 3"]).build()
        ... )  # Will create a pdf called test.pdf with an unordered
              # list with three items.
        """
        for t in text:
            self.elements.append(Paragraph(f"{bullet_char} {t}", self.styles["Normal"]))
        return self

    @staticmethod
    def decimal_to_roman(number: int, is_upper: bool = True) -> str:
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

    @staticmethod
    def decimal_to_abc(number: int, is_upper: bool = True) -> str:
        """Convert a number to an alphabetic representation."""
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        result = ""
        while number > 0:
            number -= 1
            result += (
                alphabet[number % 26] if is_upper else alphabet[number % 26].lower()
            )
            number //= 26

        return result[::-1]

    def _number_style(self, style: str) -> typing.Iterator[str]:
        """Create an iterator of numbers in the given style.

        This function relies on the itertools.count function to create an
        iterator of numbers in the given style. The style must be one of
        "decimal", "lower-roman", "upper-roman", "lower-alpha", or
        "upper-alpha".

        Parameters
        ----------
        n : int
            The number of numbers to return.
        style : str
            The style of the numbers. Must be one of "decimal", "lower-roman",
            "upper-roman", "lower-alpha", or "upper-alpha".

        Returns
        -------
        Iterator[str]
            An iterator of numbers in the given style.
        """
        decimal_numb = itertools.count(1)

        if style == "decimal":
            return map(str, decimal_numb)
        elif style == "lower-roman":
            return (Report.decimal_to_roman(x, False) for x in decimal_numb)
        elif style == "upper-roman":
            return (Report.decimal_to_roman(x, True) for x in decimal_numb)
        elif style == "lower-alpha":
            return (Report.decimal_to_abc(x, False) for x in decimal_numb)
        elif style == "upper-alpha":
            return (Report.decimal_to_abc(x, True) for x in decimal_numb)
        else:
            raise ValueError(f"Style {style} is not a valid number style.")

    def ol(self, text: list[str], number_style: str = "decimal") -> "Report":
        """Add an ordered list to the document.

        For each item in `text`, an item number is added to the list.
        If a different number style is desired, it can be passed in as
        `number_style`. Will default to the decimal number style.

        Parameters
        ----------
        text : list[str]
            The items to add to the list.
        number_style : str, optional
            The style to use for the item numbers, by default "decimal", but also
            accepts "lower-roman", "upper-roman", "lower-alpha", and "upper-alpha".

        Returns
        -------
        Report
            The current Report object. This method should be called in a chain.

        See Also
        --------
        ul : Adds an unordered list to the document. For each item in `text`, a
             bullet point is added to the list.

        Example
        -------
        >>> from predictables.util import Report
        >>> (
        ...     Report("test.pdf").ol(["Item 1", "Item 2", "Item 3"]).build()
        ... )  # Will create a pdf called test.pdf with an ordered list with three items.
        """
        styled_numbers = self._number_style(len(text), number_style)
        for t in text:
            self.elements.append(
                Paragraph(f"{next(styled_numbers)}. {t}", self.styles["Normal"])
            )
        return self

    def code(self, text: str, language: str | None = None) -> "Report":
        """Add a code block to the document.

        Used for displaying code snippets.

        Parameters
        ----------
        text : str
            The code to display in the code block.
        language : str, optional
            The language of the code, by default None. If a language is provided,
            it must be one of "py", "r", "c", "cpp", "java", "js", "html", "css",
            "xml", "json", "yaml", "sql", "bash", "sh", "powershell", "dockerfile",
            "makefile", "cmake", "latex", "markdown", "plaintext", None. This
            functionality is provided by the Pygments library.

            Note that passing no language to this argument is functionally
            equivalent to passing "plaintext", in that either way a plaintext
            code block with no syntax highlighting will be generated.

        Returns
        -------
        Report
            The current Report object. This method should be called in a chain.

        See Also
        --------
        math : Adds a math block to the document. Used for displaying math equations.

        Example
        -------
        >>> from predictables.util import Report
        >>> (
        ...     Report("test.pdf").code("print('Hello, world!')").build()
        ... )  # Will create a pdf called test.pdf with a plaintext code block that
              # says print('Hello, world!')

        >>> (
        ...     Report("test-python.pdf")
        ...     .code("print('Hello, world!')", language="py")
        ...     .build()
        ... )  # Will create a pdf called test-python.pdf with a python code block
              # that says print('Hello, world!'), and will be syntax highlighted
              # as python code.
        """
        if language is None:
            self.elements.append(Paragraph(text, self.styles["Code"]))
        else:
            self.elements.append(
                Paragraph(
                    pygments.highlight(
                        text, pygments.lexers.get_lexer_by_name(language)
                    ),
                    self.styles["Code"],
                )
            )
        return self

    def math(self, mathjax: str) -> "Report":
        """Add a math block to the document. Used for displaying math equations."""
        self.elements.append(Paragraph(mathjax, self.styles["Math"]))
        return self

    def spacer(self, height: float) -> "Report":
        """Add a spacer to the document.

        Used for adding vertical space between elements. Height is in inches.

        Parameters
        ----------
        height : float
            The height of the spacer in inches.

        Returns
        -------
        Report
            The current Report object. This method should be called in a chain.
        """
        self.elements.append(Spacer(1, height * inch))
        return self

    def image(self, filename: str, width: float = 7, height: float = 7) -> "Report":
        """Add an image to the document.

        Used to add a saved image to the document. Width and
        height are in inches.

        Parameters
        ----------
        filename : str
            The filename of the image to add.
        width : float
            The width of the image in inches.
        height : float
            The height of the image in inches.

        Returns
        -------
        Report
            The current Report object. This method should be called in a chain.

        Example
        -------
        >>> report = Report("test.pdf")
        >>> report.image("test.png", 6, 4).build()
        """
        self.elements.append(Image(filename, width * inch, height * inch))
        return self

    def plot(
        self, func: typing.Callable, width: float = 7, height: float = 7
    ) -> "Report":
        """Add a plot to the document.

        The plot is generated by the provided callable `func`. The plot
        is saved as a temporary image file and then added to the PDF
        document.

        Parameters
        ----------
        func : callable
            A function that generates and returns a matplotlib plot.
        width : float, optional
            Width of the plot in inches.
        height : float, optional
            Height of the plot in inches.

        Returns
        -------
        Report
            The current Report object. This method should be called in a chain.

        Example
        -------
        >>> def my_plot():
        ...     fig, ax = plt.subplots()
        ...     ax.plot([1, 2, 3], [1, 4, 9])
        ...     return fig, ax

        >>> report = Report("test.pdf")
        >>> report.plot(my_plot, 6, 4, "Example plot").build()
        """
        temp_filename = f"temp_plot_{uuid.uuid4()}.png"

        # Generate the plot and save it as a temporary file
        ax = func()
        fig = ax.get_figure()
        fig.savefig(temp_filename, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        # Add the plot image to the report
        return self.image(temp_filename, width, height)

    def page_break(self) -> "Report":
        """Add a page break to the document.

        This method is used to end a page.
        """
        self.elements.append(PageBreak())
        return self

    def caption(self, text: str, width: float = 7) -> "Report":
        """Add a caption to the document.

        The caption is centered and has no top margin or padding.

        Parameters
        ----------
        text : str
            The text to display as a caption.
        width : float, optional
            The width of the plot/image/table/etc that the caption is for,
            by default 6. This is used to ensure the caption is centered
            under the plot.

        Returns
        -------
        Report
            The current Report object. This method should be called in a chain.

        Example
        -------
        >>> report = Report("test.pdf")
        >>> report.caption("This is a caption.").build()
        """
        # Calculate the left and right margins to center the caption under the plot
        page_width = self.pagesize[0] / inch  # Convert page width to inches
        total_margin = page_width - width
        left_margin = total_margin / 2  # Equal margins on both sides

        # Define a custom style for the caption
        caption_style = ParagraphStyle(
            "CaptionStyle",
            parent=self.styles["Normal"],
            alignment=TA_CENTER,
            spaceBefore=0,
            spaceAfter=0,
            leftIndent=left_margin * inch,
            rightIndent=left_margin * inch,
            fontSize=10,  # Adjust font size as needed
        )

        # Add the caption to the report
        self.elements.append(Paragraph(text, caption_style))

        return self

    def table(
        self, df: pd.DataFrame | pl.DataFrame | pl.LazyFrame, style: TableStyle = None
    ) -> "Report":
        def create_table_style(
            font: str = "Helvetica", fontsize: int = 10
        ) -> TableStyle:
            return TableStyle(
                [
                    # Background color of the first row
                    ("BACKGROUND", (0, 0), (-1, 0), lightgrey),
                    # Text color of the first row
                    ("TEXTCOLOR", (0, 0), (-1, 0), black),
                    # Font style of the first row
                    ("FONTNAME", (0, 0), (-1, 0), f"{font}-Bold"),
                    # Font size of the first row
                    ("FONTSIZE", (0, 0), (-1, 0), fontsize),
                    # Double line under the first row
                    ("LINEBELOW", (0, 0), (-1, 0), 1, black),
                    # Background color of the remaining rows
                    ("BACKGROUND", (0, 1), (-1, -1), white),
                    # Text color of the remaining rows
                    ("TEXTCOLOR", (0, 1), (-1, -1), black),
                    # Font style of the remaining rows
                    ("FONTNAME", (0, 1), (-1, -1), f"{font}"),
                    # Font size of the remaining rows
                    ("FONTSIZE", (0, 1), (-1, -1), fontsize),
                    # Single line under the remaining rows
                    ("LINEBELOW", (0, 1), (-1, -1), 1, black),
                    # Single line above the remaining rows
                    ("LINEABOVE", (0, 1), (-1, -1), 0.5, black),
                    # Single line before the first column
                    ("LINEBEFORE", (0, 0), (0, -1), 1, black),
                    # Background color of the first column
                    ("BACKGROUND", (0, 0), (0, -1), lightgrey),
                    # Text color of the first column
                    ("TEXTCOLOR", (0, 0), (0, -1), black),
                    # Font style of the first column
                    ("FONTNAME", (0, 0), (0, -1), f"{font}-Bold"),
                    # Font size of the first column
                    ("FONTSIZE", (0, 0), (0, -1), fontsize),
                    # Single line after the last column:
                    ("LINEAFTER", (-1, 0), (-1, -1), 1, black),
                    # Single line before the first column:
                    ("LINEBEFORE", (0, 0), (0, -1), 1, black),
                    # Thick black line around the entire table
                    ("BOX", (0, 0), (-1, -1), 2, black),
                ]
            )

        if style is None:
            style = create_table_style()

        df = to_pd_df(df)
        data = [["", *df.columns.tolist()]] + [
            [df.index.tolist()[i]] + df.to_numpy().tolist()[i]
            for i, _ in enumerate(df.to_numpy().tolist())
        ]
        t = Table(data)
        if style is not None:
            t.setStyle(style)
        self.elements.append(t)
        return self

    def build(self) -> None:
        """Build the pdf document and saves it to the filename specified in the constructor.

        This is the final command that must be called to generate the pdf document.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Builds the pdf document and saves it to the filename specified in
            the constructor.
        """
        self.doc.build(self.elements)

        # Remove temporary plot files
        for file in Path.cwd().iterdir():
            if file.startswith("temp_plot_"):
                Path.unlink(file)

    def title(self, text: str) -> "Report":
        """Set the title metadata attribute of the pdf document.

        Does not by itself make any visible changes to the document.

        Parameters
        ----------
        text : str
            The title to add to the document metadata.

        Returns
        -------
        Report
            The current Report object. This method should be called in a chain.
        """
        self.doc.title = text
        return self

    def author(self, text: str) -> "Report":
        """Set the author metadata attribute of the pdf document.

        Does not by itself make any visible changes to the document.

        Parameters
        ----------
        text : str
            The author to add to the document metadata.

        Returns
        -------
        Report
            The current Report object. This method should be called in a chain.
        """
        self.doc.author = text
        return self

    def subject(self, text: str) -> "Report":
        """Set the subject metadata attribute of the pdf document.

        Does not by itself make any visible changes to the document.

        Parameters
        ----------
        text : str
            The subject to add to the document metadata.

        Returns
        -------
        Report
            The current Report object. This method should be called in a chain.
        """
        self.doc.subject = text
        return self

    def date(self, date: datetime.date | None = None) -> None:
        """Set the date metadata attribute of the pdf document.

        Does not by itself make any visible changes to the document.

        Parameters
        ----------
        date : Optional[datetime.date], optional
            The date to add to the document metadata. If not specified, defaults
            to the current date.

        Returns
        -------
        Report
            The current Report object. This method should be called in a chain.
        """
        if date is None:
            date = datetime.datetime.now()

        self.doc.date = date.strftime("%Y-%m-%d")

    def date_(self, date: datetime.date | None = None) -> str:
        """Alias `date`, and return the date added to the document metadata.

        This allows you to ensure they are the same.

        Parameters
        ----------
        date : Optional[datetime.date], optional
            The date to add to the document metadata. If not specified, defaults
            to the current date.

        Returns
        -------
        str
            The date added to the document metadata.
        """
        self.date(date)
        return self.doc.date

    def keywords(self, text: str) -> "Report":
        """Set the keywords metadata attribute of the pdf document.

        Does not by itself make any visible changes to the document.

        Parameters
        ----------
        text : str
            The keywords to add to the document metadata.

        Returns
        -------
        Report
            The current Report object. This method should be called in a chain.
        """
        self.doc.keywords = text
        return self
