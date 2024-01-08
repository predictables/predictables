"""
This library allows users to create technical PDF reports that can include text, tables, plots, images, and lists. It is designed to be flexible and easy to use, with several options for customization.

Classes
-------
Report
    Manages the layout and rendering of elements across multiple pages in the PDF. It handles page breaks and organizes elements into a coherent structure.
Element
    The base class for all report elements. Defines common attributes and a method for rendering.
Text
    Represents a styled block of text in the report, adaptable for various layouts.
Heading
    Represents a heading in the report, styled with a larger and bold font typically used for titles or section headers.
Subheading
    Represents a subheading in the report, styled to be visually subordinate to the main heading.
Plot
    For including matplotlib plots in the report. Handles rendering of plot images.
Table
    For including pandas DataFrames in the report. Handles rendering of tables.
Image
    To insert images into the report, either from a file path or a buffer.
Column
    Facilitates the creation of columnar layouts, allowing for different alignments in each column.
BulletList
    Represents a bullet list in the report, allowing for customized bullet points and text.
NumberedList
    Represents a numbered list in the report, providing automatic numbering for list items.
Line
    Represents a horizontal line in the report, with customizable properties such as color, thickness, and style.
Spacer
    Represents a spacer in the report, used to consume vertical space.

PageBreak
    Represents a page break in the report, used to trigger the start of a new page.

Examples
--------
Adding a text block:
>>> text_block = "This is a sample text block."
>>> text_element = Text(text_block, width=450, height=100)
>>> report.add_element(text_element)

Adding a plot:
>>> def example_plot(ax):
>>>     ax.plot([0, 1, 2], [10, 20, 15], marker='o')
>>>     ax.set_title('Sample Plot')
>>> plot_element = Plot(example_plot, width=450, height=300)
>>> report.add_element(plot_element)

Adding an image:
>>> image_path = 'path/to/image.png'
>>> image_element = Image(image_path, width=450, height=300)
>>> report.add_element(image_element)

Adding a bullet list:
>>> bullet_list_items = ["First item", "Second item", "Third item"]
>>> bullet_list = BulletList(bullet_list_items, width=450)
>>> report.add_element(bullet_list)

Adding a numbered list:
>>> numbered_list_items = ["First item", "Second item", "Third item"]
>>> numbered_list = NumberedList(numbered_list_items, width=450)
>>> report.add_element(numbered_list)

Creating a page break:
>>> report.add_element(PageBreak())

Generating the report:
>>> report.generate()

These examples demonstrate how to add various types of elements to a report and how to generate the final PDF document using the Report class.
Each class supports various customization options, such as setting the width and height of elements, as well as font styles and colors for text.

For more detailed examples and options, refer to the user guide or API documentation.
"""

import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    Image as PlatypusImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table as PlatypusTable,
    TableStyle,
)


class Page:
    """
    Represents a page in the report, and is structured as a list of elements. The page is rendered by
    iterating through the elements and rendering them sequentially. A Page object keeps track
    of the current page number and the current y-coordinate, which is updated after each element is rendered.

    If you attempt to add an element to the page that would cause it to exceed the page height,
    a PageBreak is automatically inserted and the element is added to a new page.

    Parameters
    ----------
    paper_size : tuple of float
        The width and height of the page in inches.
    x_margin : float
        The horizontal margin of the page in inches. The left and right sides of the page will be
        offset by this amount.
    y_margin : float
        The vertical margin of the page in inches. The top and bottom sides of the page will be
        offset by this amount.

    Attributes
    ----------
    elements : list of Element
        The elements on the page.
    page_number : int
        The current page number. Starts at 1.
    y : float
        The current y-coordinate of the page. Starts at the beginning of the content area, not
        the top of the page. This value begins at 0 and is updated after each element is rendered.
    max_y : float
        The maximum y-coordinate of the page. This is the y-coordinate of the bottom of the content
        area, not the bottom of the page. This value is calculated based on the page height and
        margins. If adding an element would cause the page to exceed this value, a PageBreak is
        automatically inserted and the element is added to a new page.
    default_width : float
        The default width of elements on the page. This is the full width of the page minus the
        left and right margins. This defaults to the full width of the page, but can be overridden
        by setting the width attribute of an element.
    default_height : float
        The default height of elements on the page. This defaults to one inch, but can be overridden
        by setting the height attribute of an element.

    Methods
    -------
    add_element(element)
        Adds an element to the page, managing page breaks based on space availability.
    """

    def __init__(self, paper_size=letter, x_margin=1 * inch, y_margin=1 * inch):
        """
        Initializes the Page with a specific paper size and margins.

        Parameters
        ----------
        paper_size : tuple of float, optional
            The width and height of the page in inches. Default is letter size.
        x_margin : float, optional
            The horizontal margin of the page in inches. The left and right sides of the page will be
            offset by this amount. Default is 1 inch.
        y_margin : float, optional
            The vertical margin of the page in inches. The top and bottom sides of the page will be
            offset by this amount. Default is 1 inch.
        """
        self.paper_size = paper_size
        self.x_margin = x_margin
        self.y_margin = y_margin
        self.width = paper_size[0] - (2 * x_margin)
        self.height = paper_size[1] - (2 * y_margin)
        self.elements = []
        self.current_y = y_margin  # Start at the top of the content area
        self.max_y = self.height - y_margin
        self.default_width = self.width
        self.default_height = 1 * inch

    def has_space_for(self, element) -> bool:
        """
        Determines if there's enough space on the page for the element.

        Parameters
        ----------
        element : Element
            The element to be added to the page.

        Returns
        -------
        bool
            True if there's enough space for the element, False otherwise.
        """
        element_height = (
            element.height if element.height is not None else self.default_height
        )
        return self.current_y - element_height >= self.y_margin

    def add_element(self, element, **kwargs):
        """
        Adds an element to the page, managing page breaks based on space availability.

        If adding the element would cause the page to exceed the maximum y-coordinate, a PageBreak
        is automatically inserted and the element is added to a new page.

        Parameters
        ----------
        element : Element
            The element to be added to the page.
        **kwargs : dict, optional
            Keyword arguments that are passed to the Element base class.
        """

        element_height = (
            kwargs.get("height", element.height)
            if hasattr(element, "height") and element.height is not None
            else self.default_height
        )

        # Check if adding the element would cause the page to exceed the maximum y-coordinate
        if self.current_y + element_height < self.max_y:
            # If so, add a PageBreak and add the element to a new page
            return False

        # Update the element's attributes
        self.elements.append(element)
        self.current_y -= element_height
        return True  # Element added successfully

    def render(self, canvas):
        """
        Renders the page and its elements onto the given canvas.

        This method is responsible for placing each element within the page margins
        and calling each element's render method.

        Parameters
        ----------
        canvas : Canvas
            The canvas object on which to draw the page's elements.
        """
        for element in self.elements:
            element.render(canvas, self.x_margin, self.current_y)
            if hasattr(element, "height"):
                self.current_y -= (
                    element.height
                )  # Update the y-coordinate after rendering


class Report:
    """
    Manages the layout and rendering of elements in the PDF across multiple pages.

    Parameters
    ----------
    filename : str
        The path to the PDF file to be generated.

    Attributes
    ----------
    canvas : Canvas
        The ReportLab Canvas object where elements are drawn.
    pages : list of list
        Each sublist contains elements for a specific page.

    Methods
    -------
    add_element(element)
        Adds an element to the report, managing page breaks based on space availability.
    generate()
        Finalizes the PDF and saves it to the specified path.
    """

    def __init__(
        self, filename, paper_size=letter, x_margin=1 * inch, y_margin=1 * inch
    ):
        """
        Initializes the Report with a specific filename.

        Sets up the ReportLab Canvas object for drawing elements and prepares
        the page structure.

        Parameters
        ----------
        filename : str
            The path to the PDF file to be generated.
        paper_size : tuple of float, optional
            The width and height of the page in inches. Default is letter size.
        x_margin : float, optional
            The horizontal margin of the page in inches. The left and right sides of the page will be
            offset by this amount. Default is 1 inch.
        y_margin : float, optional
            The vertical margin of the page in inches. The top and bottom sides of the page will be
            offset by this amount. Default is 1 inch.

        """
        self.filename = filename
        self.canvas = canvas.Canvas(filename, pagesize=paper_size)
        self.paper_size = paper_size
        self.x_margin = x_margin
        self.y_margin = y_margin
        self.pages = [
            Page(paper_size, x_margin, y_margin)
        ]  # Start with one empty Page object
        self._update_current_page()

    def _update_current_page(self):
        """
        Updates the current page to the last page in the list.

        This method is called after adding an element to the report to ensure that
        the current page is always the last page in the list.
        """
        self.current_page = self.pages[-1]

    def add_element(self, element):
        """
        Adds an element to the report, managing page breaks based on space availability.

        If adding the element would cause the page to exceed the maximum y-coordinate, a PageBreak
        is automatically inserted and the element is added to a new page.

        Parameters
        ----------
        element : Element
            The element to be added to the report.
        """
        current_page = self.pages[-1]
        if not current_page.has_space_for(element):
            # Not enough space, create a new page
            new_page = Page(self.paper_size, self.x_margin, self.y_margin)
            self.pages.append(new_page)
            current_page = new_page

        current_page.add_element(element)

    def generate(self):
        """
        Finalizes the PDF by rendering elements and saving the canvas.

        This method iterates through each page and its elements, rendering them
        sequentially and adding new pages as necessary.
        """
        for page in self.pages:
            page.render(self.canvas)  # Render the entire page
            self.canvas.showPage()  # End the current page

        self.canvas.save()  # Finalize the PDF


class Element:
    """
    The base class for all report elements.

    This class should be subclassed to create different types of elements
    that can be added to the report.

    Parameters
    ----------
    x : float
        The x-coordinate of the element's position in the report.
    y : float
        The y-coordinate of the element's position in the report.
    width : float, optional
        The width of the element. Default is None, which means the element will
        take up the full width of the page.
    height : float, optional
        The height of the element. Default is None, which means the element will
        take up the full height of the page.
    style : dict, optional
        A dictionary of style settings.

    Methods
    -------
    render(canvas)
        Abstract method to render the element on a canvas.

    See Also
    --------
    Text : A subclass of Element that represents a styled block of text in the report.
    Heading : A subclass of Text that represents a heading in the report.
    Subheading : A subclass of Text that represents a subheading in the report.
    Plot : A subclass of Element that represents a plot or chart in the report.
    Table : A subclass of Element that represents a table in the report.
    Image : A subclass of Element that represents an image in the report.
    Column : A subclass of Element that represents a row of elements in the report.
    BulletList : A subclass of Element that represents a bullet list in the report.
    NumberedList : A subclass of Element that represents a numbered list in the report.
    Line : A subclass of Element that represents a horizontal line in the report.
    Spacer : A subclass of Element that represents a spacer in the report.
    PageBreak : A subclass of Element that represents a page break in the report.

    Examples
    --------
    >>> text_content = "This is a sample text block."
    >>> text = Text(text_content, width=400, height=50)
    >>> report.add_element(text)
    """

    def __init__(self, x=0, y=0, width=None, height=None, style=None):
        self.x = x
        self.y = y
        self.width = width or canvas._pagesize[0]
        self.height = height or canvas._pagesize[1]
        self.style = style or {}

    def render(self, canvas):
        raise NotImplementedError("Subclasses must implement this method.")


class Text(Element):
    """
    Represents a styled block of text in the report, flexible enough to be used
    both as a standalone element and within complex structures.

    Inherits from the Element class and allows for adding customized text to the report,
    with support for detailed styling and integration into complex layouts.

    Parameters
    ----------
    text : str
        The text content of the element.
    font_name : str, optional
        The name of the font to use for rendering the text.
    font_size : int, optional
        The size of the font.
    font_color : colors.Color, optional
        The color of the font.
    alignment : str, optional
        The alignment of the text within the element.
        Options are 'left', 'center', or 'right'.
    **kwargs : dict, optional
        Keyword arguments that are passed to the Element base class.

    Methods
    -------
    render(canvas)
        Renders the styled text element onto the given canvas, adaptable for
        various layout contexts.

    See Also
    --------
    Element : The base class for all report elements.

    Examples
    --------
    >>> text_content = "This is a sample text block."
    >>> text = Text(text_content, width=400, height=50)
    >>> report.add_element(text)
    """

    def __init__(
        self,
        text,
        x=0,
        y=0,
        width=None,
        height=None,
        style=None,
        font_name="Helvetica",
        font_size=12,
        font_color=colors.black,
        alignment="left",
        **kwargs,
    ):
        """
        Constructs all the necessary attributes for the Text object.

        Parameters
        ----------
        text : str
            The text content of the element.
        font_name : str, optional
            The name of the font to use for rendering the text.
        font_size : int, optional
            The size of the font.
        font_color : colors.Color, optional
            The color of the font.
        alignment : str, optional
            The alignment of the text within the element.
        **kwargs : dict, optional
            Keyword arguments that are passed to the Element base class.
        """
        super().__init__(x=x, y=y, width=width, height=height, style=style, **kwargs)
        self.text = text
        self.font_name = font_name
        self.font_size = font_size
        self.font_color = font_color
        self.alignment = alignment

    def render(self, canvas):
        """
        Renders the styled text element onto the given canvas.

        This method creates a ReportLab Paragraph object with the specified
        style attributes and adds it to the canvas at the specified location.
        It is designed to be flexible for use in various layout contexts.

        Parameters
        ----------
        canvas : Canvas
            The canvas on which to render the text element.
        """
        # Create a custom paragraph style
        style = ParagraphStyle(
            name="CustomTextStyle",
            fontName=self.font_name,
            fontSize=self.font_size,
            textColor=self.font_color,
            alignment={"left": TA_LEFT, "center": TA_CENTER, "right": TA_RIGHT}.get(
                self.alignment, TA_LEFT
            ),
        )

        # Create and render the paragraph
        paragraph = Paragraph(self.text, style)
        paragraph.wrapOn(
            canvas,
            self.width or canvas._pagesize[0],
            self.height or canvas._pagesize[1],
        )
        paragraph.drawOn(canvas, self.x, self.y)


class Heading(Text):
    """
    Represents a heading in the report, styled with a larger and bold font typically used for titles or section headers.

    Inherits from the Text class and customizes the style to be more prominent than other text.

    Parameters
    ----------
    text : str
        The text content of the heading.
    **kwargs : dict, optional
        Keyword arguments that allow further customization of the heading's style.

    See Also
    --------
    Text : The base class for text elements in the report.

    Examples
    --------
    >>> heading_text = "This is a heading."
    >>> heading = Heading(heading_text, width=400, height=50)
    >>> report.add_element(heading)
    """

    def __init__(self, text, **kwargs):
        """
        Constructs all the necessary attributes for the Heading object.

        The style is predefined but can be overridden by passing a custom style
        through the kwargs.

        Parameters
        ----------
        text : str
            The text content of the heading.
        **kwargs : dict, optional
            Keyword arguments that allow further customization of the heading's style.
        """
        # kwargs["style"] = ParagraphStyle(
        #     "Heading",
        #     parent=kwargs.get("style", getSampleStyleSheet()["Heading1"]),
        #     fontSize=14,
        #     leading=16,
        #     spaceAfter=12,
        #     bold=True,
        # )
        # super().__init__(text, **kwargs)

        # Define default style for Subheading
        default_style = ParagraphStyle(
            "Heading",
            parent=getSampleStyleSheet()["Heading1"],
            fontSize=14,
            leading=16,
            spaceAfter=12,
            bold=True,
        )
        # Override default style with any provided custom styles
        custom_style = kwargs.pop("style", {})
        for key, value in custom_style.items():
            if hasattr(default_style, key):
                setattr(default_style, key, value)

        super().__init__(text, style=default_style, **kwargs)

    # def render(self, canvas):
    #     """
    #     Renders the heading element onto the given canvas using the predefined style.

    #     Overrides the render method from the Text class to apply the heading style.

    #     Parameters
    #     ----------
    #     canvas : Canvas
    #         The canvas object on which to draw the heading.
    #     """
    #     styles = getSampleStyleSheet()
    #     style = self.style or styles["Heading1"]
    #     for key, value in self.style.items():
    #         if hasattr(style, key):
    #             setattr(style, key, value)
    #     paragraph = Paragraph(self.text, style)
    #     paragraph.wrapOn(canvas, self.width or canvas._pagesize[0], self.height)
    #     paragraph.drawOn(canvas, self.x, self.y)


class Subheading(Text):
    """
    Represents a subheading in the report, styled to be visually subordinate to the main heading.

    Inherits from the Text class and customizes the style to differentiate from the main heading.

    Parameters
    ----------
    text : str
        The text content of the subheading.
    **kwargs : dict, optional
        Keyword arguments that allow further customization of the subheading's style.

    See Also
    --------
    Text : The base class for text elements in the report.

    Examples
    --------
    >>> subheading_text = "This is a subheading."
    >>> subheading = Subheading(subheading_text, width=400, height=50)
    >>> report.add_element(subheading)
    """

    def __init__(self, text, **kwargs):
        """
        Constructs all the necessary attributes for the Subheading object.

        The style is predefined but can be overridden by passing a custom style
        through the kwargs.

        Parameters
        ----------
        text : str
            The text content of the subheading.
        **kwargs : dict, optional
            Keyword arguments that allow further customization of the subheading's style.
        """
        # Define default style for Subheading
        default_style = ParagraphStyle(
            "Subheading",
            parent=getSampleStyleSheet()["Heading2"],
            fontSize=12,
            spaceAfter=6,
        )
        # Override default style with any provided custom styles
        custom_style = kwargs.pop("style", {})
        for key, value in custom_style.items():
            if hasattr(default_style, key):
                setattr(default_style, key, value)

        super().__init__(text, style=default_style, **kwargs)

    # def render(self, canvas):
    #     """
    #     Renders the subheading element onto the given canvas using the predefined style.

    #     Overrides the render method from the Text class to apply the subheading style.

    #     Parameters
    #     ----------
    #     canvas : Canvas
    #         The canvas object on which to draw the subheading.
    #     """
    #     styles = getSampleStyleSheet()
    #     style = styles["Heading2"]
    #     for key, value in self.style.items():
    #         if hasattr(style, key):
    #             setattr(style, key, value)
    #     paragraph = Paragraph(self.text, style)
    #     paragraph.wrapOn(canvas, self.width or canvas._pagesize[0], self.height)
    #     paragraph.drawOn(canvas, self.x, self.y)


class Plot(Element):
    """
    Represents a plot or chart in the report, typically created using matplotlib.

    Parameters
    ----------
    plot_function : callable
        A function that takes a matplotlib axes object and adds a plot to it.
    **kwargs : dict, optional
        Keyword arguments that are passed to the Element base class.

    Methods
    -------
    render(canvas)
        Renders the plot onto the given canvas using the plot function.
    """

    def __init__(self, plot_function, **kwargs):
        super().__init__(**kwargs)
        self.plot_function = plot_function  # A function that creates a matplotlib plot

    def render(self, canvas):
        # Create a matplotlib figure
        fig, ax = plt.subplots(figsize=(self.width / inch, self.height / inch))
        # Associate the figure with the FigureCanvas
        FigureCanvas(fig)
        # Call the user-defined function to populate the plot
        self.plot_function(ax)
        # Save the plot to a bytes buffer
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format="png", bbox_inches="tight")
        img_buffer.seek(0)
        # Place the image on the PDF canvas
        canvas.drawImage(
            ImageReader(img_buffer),
            self.x,
            self.y - self.height,
            width=self.width,
            height=self.height,
        )
        img_buffer.close()


class Table(Element):
    """
    Represents a table in the report, rendered from a pandas DataFrame.

    Inherits from the Element class and uses ReportLab's Table object to render
    the DataFrame as a table in the PDF.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The pandas DataFrame to be rendered as a table.
    style : TableStyle, optional
        The style to be applied to the table. If not provided, a default style is used.
    **kwargs : dict, optional
        Keyword arguments that are passed to the base Element class.
    """

    def __init__(self, dataframe, style=None, **kwargs):
        """
        Constructs all the necessary attributes for the Table object.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The pandas DataFrame to be rendered as a table.
        style : TableStyle, optional
            The style to be applied to the table. If not provided, a default style is used.
        **kwargs : dict, optional
            Keyword arguments that are passed to the base Element class.
        """
        super().__init__(**kwargs)
        self.dataframe = dataframe
        self.data = [dataframe.columns.tolist()] + dataframe.values.tolist()
        self.style = style if style is not None else self.default_style()

    def default_style(self):
        """
        Defines the default style for the table.

        Returns
        -------
        TableStyle
            The default TableStyle object with predefined settings.
        """
        return TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ]
        )

    def render(self, canvas):
        """
        Renders the table element onto the given canvas.

        Parameters
        ----------
        canvas : Canvas
            The canvas object on which to draw the table.
        """
        table = PlatypusTable(self.data, style=self.style)
        table.wrapOn(canvas, self.width, self.height)
        table.drawOn(canvas, self.x, self.y - table._height)


class Image(Element):
    """
    Represents an image in the report, sourced from a file path or a buffer.

    Parameters
    ----------
    image_path_or_buffer : str or io.BytesIO
        The source of the image, either a path to an image file or a buffer object.
    **kwargs : dict, optional
        Keyword arguments that are passed to the Element base class.

    Methods
    -------
    render(canvas)
        Renders the image onto the given canvas.
    """

    def __init__(self, image_path_or_buffer, **kwargs):
        super().__init__(**kwargs)
        self.image_source = image_path_or_buffer

    def render(self, canvas):
        if isinstance(self.image_source, io.BytesIO):
            # If the source is a buffer, reset the pointer to the beginning
            self.image_source.seek(0)
            image = ImageReader(self.image_source)
        else:
            # If the source is a path, use it directly
            image = self.image_source
        canvas.drawImage(
            image,
            self.x,
            self.y - self.height,
            width=self.width,
            height=self.height,
            mask="auto",
        )


class Column(Element):
    """
    Represents a row of elements in the report, allowing for different alignments in each element.

    Inherits from the Element class and is used to create a row with multiple elements,
    each potentially having different content and alignment.

    Parameters
    ----------
    elements : list of Element
        A list of Element objects to be arranged in a row.
    **kwargs : dict, optional
        Keyword arguments for the base Element class.

    Methods
    -------
    render(canvas)
        Renders the elements in a row onto the given canvas.
    """

    def __init__(self, elements, **kwargs):
        super().__init__(**kwargs)
        self.elements = elements

    def render(self, canvas):
        current_x = self.x
        for element in self.elements:
            element.x = current_x
            element.y = self.y
            element.render(canvas)
            current_x += element.width


class BulletList(Element):
    """
    Represents a bullet list in the report.

    Inherits from the Element class and uses the Text class within a Column layout
    to display a list of items with bullet points. The bullet point is aligned with
    the left side of a text block, and the text is slightly indented to the right.

    Parameters
    ----------
    bullet_list_items : list of str
        A list of strings, each representing a bullet list item.
    **kwargs : dict, optional
        Keyword arguments for the base Element class.

    Methods
    -------
    render(canvas)
        Renders the bullet list onto the given canvas.
    """

    def __init__(self, bullet_list_items, **kwargs):
        """
        Initializes the BulletList object with list items.

        Parameters
        ----------
        bullet_list_items : list of str
            A list of strings, each representing a bullet list item.
        **kwargs : dict, optional
            Keyword arguments for the base Element class.
        """
        super().__init__(**kwargs)
        self.bullet_list_items = bullet_list_items

    def render(self, canvas):
        """
        Renders the bullet list onto the given canvas.

        The bullet points are aligned with the left side of the text block,
        and the text is slightly indented to the right of the bullet point.

        Parameters
        ----------
        canvas : Canvas
            The canvas object on which to draw the bullet list.
        """

        current_y = self.y
        bullet_style = ParagraphStyle(
            name="BulletStyle",
            leftIndent=20,
            bulletIndent=-20,
            spaceAfter=5,
            bulletFontName="Symbol",
        )

        for item in self.bullet_list_items:
            bullet_text = Text("•", font_name="Symbol", x=self.x, y=current_y, width=20)
            item_text = Text(item, x=self.x + 20, y=current_y, width=self.width - 20)

            bullet_column = Column(
                [bullet_text, item_text], x=self.x, y=current_y, width=self.width
            )
            bullet_column.render(canvas)

            # Create a temporary Paragraph to calculate height
            temp_paragraph = Paragraph(item, bullet_style)
            _, item_height = temp_paragraph.wrap(self.width - 20, canvas._pagesize[1])
            current_y -= item_height + bullet_style.spaceAfter


class NumberedList(Element):
    """
    Represents a numbered list in the report.

    Inherits from the Element class and uses the Text class within a Column layout
    to display a list of items with numbering. Each number is aligned with
    the left side of a text block, and the text is slightly indented to the right.

    Parameters
    ----------
    numbered_list_items : list of str
        A list of strings, each representing a numbered list item.
    **kwargs : dict, optional
        Keyword arguments for the base Element class.

    Methods
    -------
    render(canvas)
        Renders the numbered list onto the given canvas.
    """

    def __init__(self, numbered_list_items, **kwargs):
        """
        Initializes the NumberedList object with list items.

        Parameters
        ----------
        numbered_list_items : list of str
            A list of strings, each representing a numbered list item.
        **kwargs : dict, optional
            Keyword arguments for the base Element class.
        """
        super().__init__(**kwargs)
        self.numbered_list_items = numbered_list_items

    def render(self, canvas):
        """
        Renders the numbered list onto the given canvas.

        The numbers are aligned with the left side of the text block,
        and the text is slightly indented to the right of the number.

        Parameters
        ----------
        canvas : Canvas
            The canvas object on which to draw the numbered list.
        """
        current_y = self.y
        number_style = ParagraphStyle(name="NumberStyle", leftIndent=20, spaceAfter=5)

        for index, item in enumerate(self.numbered_list_items, start=1):
            number_text = Text(f"{index}.", x=self.x, y=current_y, width=20)
            item_text = Text(item, x=self.x + 20, y=current_y, width=self.width - 20)

            number_column = Column(
                [number_text, item_text], x=self.x, y=current_y, width=self.width
            )
            number_column.render(canvas)

            # Create a temporary Paragraph to calculate height
            temp_paragraph = Paragraph(item, number_style)
            _, item_height = temp_paragraph.wrap(self.width - 20, canvas._pagesize[1])
            current_y -= item_height + number_style.spaceAfter


class Line(Element):
    """
    Represents a horizontal line in the report, with customizable properties such as color, thickness, and style.

    Inherits from the Element class and is used to draw a styled horizontal line in the report.

    Parameters
    ----------
    **kwargs : dict, optional
        Keyword arguments for the base Element class and line style, such as 'x', 'y', 'width',
        'color', 'thickness', and 'style'.

    Attributes
    ----------
    color : reportlab.lib.colors.Color, optional
        The color of the line. Default is black.
    thickness : float, optional
        The thickness of the line. Default is 1.
    style : list, optional
        The dash style of the line. List of numbers describing the dash pattern.

    Methods
    -------
    render(canvas)
        Renders the line onto the given canvas with the specified style.
    """

    def __init__(self, **kwargs):
        """
        Initializes the Line object with given style attributes.

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments for the base Element class and line style.
        """
        super().__init__(**kwargs)
        self.color = kwargs.get("color", colors.black)
        self.thickness = kwargs.get("thickness", 1)
        self.style = kwargs.get("style", None)

    def render(self, canvas):
        """
        Renders the line element onto the given canvas with specified style.

        Draws a styled horizontal line on the canvas starting from the (x, y) position and extending
        horizontally for the length specified by the 'width' attribute.

        Parameters
        ----------
        canvas : Canvas
            The canvas object on which to draw the line.
        """
        canvas.setStrokeColor(self.color)
        canvas.setLineWidth(self.thickness)
        if self.style:
            canvas.setDash(self.style)
        canvas.line(self.x, self.y, self.x + self.width, self.y)


class Spacer(Element):
    """
    Represents a spacer in the report, used to consume vertical space.

    Inherits from the Element class and is utilized to create gaps or breaks
    between other elements in the report without rendering any visible content.

    The spacer's 'height' attribute determines the amount of vertical space it consumes.

    Parameters
    ----------
    **kwargs : dict, optional
        Keyword arguments for the base Element class, such as 'height'.

    Methods
    -------
    render(canvas)
        Consumes vertical space in the report layout without rendering anything.
    """

    def __init__(self, **kwargs):
        """
        Initializes the Spacer object with given style attributes.

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments for the base Element class.
        """
        default_spacer_height = 0.25 * inch
        height = kwargs.get("height", None)
        self.height = height if height is not None else default_spacer_height
        super().__init__(**kwargs)

    # def render(self, canvas):
    #     """
    #     Consumes vertical space in the report layout.

    #     The render method for Spacer does not draw anything on the canvas but
    #     is used to adjust the layout by consuming vertical space as specified
    #     by its 'height' attribute.

    #     Parameters
    #     ----------
    #     canvas : Canvas
    #         The canvas object on which the spacer would 'occupy' space.
    #     """
    # Implementation intentionally left blank to consume space


class PageBreak(Element):
    """
    Represents a page break in the report.

    Inherits from the Element class and is used to trigger the start of a new page
    in the PDF document. When added to the layout manager and rendered, it causes
    the subsequent elements to appear on a new page.

    Parameters
    ----------
    **kwargs : dict, optional
        Keyword arguments for the base Element class, though they are not
        specifically used for the PageBreak functionality.

    Methods
    -------
    render(canvas)
        Renders the page break by starting a new page in the PDF document.
    """

    def __init__(self, **kwargs):
        """
        Initializes the PageBreak object.

        Accepts keyword arguments for consistency with the Element interface,
        but these arguments do not impact the page break functionality.

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments for the base Element class.
        """
        super().__init__(**kwargs)

    def render(self, canvas):
        """
        Renders the page break onto the given canvas.

        This method instructs the canvas to start a new page, effectively
        placing subsequent elements on the next page of the PDF document.

        Parameters
        ----------
        canvas : Canvas
            The canvas object on which to apply the page break.
        """
        canvas.showPage()
