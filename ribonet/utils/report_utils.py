"""
report_utils.py - ReportLab wrapper utilities for easy PDF report generation

Usage:
    from ribonet.utils.report_utils import *
    
    create_report("output.pdf")
    add_heading("Main Title")
    add_subheading("Subtitle")
    add_text("Body text here")
    add_table(df)
    add_plot(fig)
    add_html("path/to/file.html")
    finalize_report()
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm, cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    Paragraph, Spacer, Table, TableStyle, 
    PageBreak, Image
)
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.platypus.frames import Frame
from reportlab.pdfgen.canvas import Canvas
from PIL import Image as PILImage
import io
from typing import Optional, List
import pandas as pd
from pathlib import Path
import os
from bs4 import BeautifulSoup
import logging

# Configure fontTools to only log warnings/errors, not routine font subsetting operations
# This prevents INFO spam from font embedding during PDF generation
logging.getLogger('fontTools.subset').setLevel(logging.WARNING)

log = logging.getLogger(__name__)

# Global variables to store the document state
_doc = None
_elements = []
_styles = None
_filename = None
_max_width = None
_max_height = None
_toc = None

class NumberedPageCanvas(Canvas):
    """
    http://code.activestate.com/recipes/546511-page-x-of-y-with-reportlab/
    http://code.activestate.com/recipes/576832/
    http://www.blog.pythonlibrary.org/2013/08/12/reportlab-how-to-add-page-numbers/
    """

    def __init__(self, *args, **kwargs):
        """Constructor"""
        super().__init__(*args, **kwargs)
        self.pages = []

    def showPage(self):
        """
        On a page break, add information to the list
        """
        self.pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        """
        Add the page number to each page (page x of y)
        """
        page_count = len(self.pages)

        for page in self.pages:
            self.__dict__.update(page)
            self.draw_page_number(page_count)
            super().showPage()

        super().save()

    def draw_page_number(self, page_count):
        """
        Add the page number
        """
        page = "Page %s of %s" % (self._pageNumber, page_count)
        self.setFont("Helvetica", 9)
        self.drawRightString(190 * mm, 10 * mm, page)

class MyDocTemplate(BaseDocTemplate):

    def __init__(self, filename, **kw):
        BaseDocTemplate.__init__(self, filename, **kw)
        frame = Frame(self.leftMargin, self.bottomMargin, 
                      self.width, self.height, id='normal')
        template = PageTemplate(id='simple', frames=[frame])
        self.addPageTemplates(template)

    def afterFlowable(self, flowable):
        "Registers TOC entries."
        if flowable.__class__.__name__ == 'Paragraph':
            text = flowable.getPlainText()
            style = flowable.style.name
            if style == 'CustomHeading':
                self.notify('TOCEntry', (0, text, self.page))
            elif style == 'CustomSubheading':
                self.notify('TOCEntry', (1, text, self.page))

USABLE_WIDTH = A4[0] / inch

def create_report(filename: str, 
                  title: Optional[str] = None):
    """
    Initialize a new PDF report.
    
    Args:
        filename: Output PDF filename
        title: Document title metadata
    
    Example:
        create_report("my_report.pdf", title="Sales Analysis")
    """
    global _doc, _elements, _styles, _filename, _max_width, _max_height, _toc
    
    _filename = filename
    
    _doc = MyDocTemplate(filename, pagesize=A4, title=title,
                         leftMargin=2*cm, rightMargin=2*cm, 
                         topMargin=2*cm, bottomMargin=2*cm)
    
    # Calculate maximum dimensions for images/plots
    # Subtract an extra 0.5cm safety margin on each axis
    page_width, page_height = A4
    _max_width = (page_width - 2*cm - 2*cm - 0.5*cm) / inch
    _max_height = (page_height - 2*cm - 2*cm - 0.5*cm) / inch
    
    _elements = []
    _styles = getSampleStyleSheet()
    _toc = None
    
    # Add custom heading style
    _styles.add(ParagraphStyle(
        name='CustomHeading',
        parent=_styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12,
        leading=22
    ))
    
    # Add custom subheading style
    _styles.add(ParagraphStyle(
        name='CustomSubheading',
        parent=_styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=10,
        spaceBefore=10,
        leading=18
    ))

    _styles.add(ParagraphStyle(
        name='CustomSubheading2',
        parent=_styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=8,
        spaceBefore=8,
        leading=16
    ))
    
    # Add compact style for table cells with minimal line spacing
    _styles.add(ParagraphStyle(
        name='CompactCell',
        parent=_styles['Normal'],
        fontSize=5,
        leading=6,  # Minimal line spacing
        spaceAfter=0,
        spaceBefore=0,
        alignment=TA_CENTER
    ))
    
    # Add justified text style for body text
    _styles.add(ParagraphStyle(
        name='JustifiedText',
        parent=_styles['Normal'],
        alignment=TA_JUSTIFY
    ))
    
    # Add title heading at the beginning if provided
    if title:
        heading = Paragraph(title, _styles['CustomHeading'])
        _elements.append(heading)
        _elements.append(Spacer(1, 12))
    
    # Add intro
    add_image(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, "data", "img", "ribonet.png"), width=4)
    intro_text = """This report has been generated by RiboNet, a deep learning analysis toolkit for ribosome profiling. <br/><br/>
    If you use RiboNet in your research, please cite:<br/>
    Hernandez-Alias X, Kaya S, Doychev K, Schroeder F, Hermann B, Madran B, Majidian S, Gagneur J, Nedialkova DD (2026).
    Metabolic transitions reprogram translation through sequence-dependent regulation. <i>bioRxiv</i>.
    doi: <a href="https://doi.org/10.64898/2026.11.15.766763" color="blue">10.64898/2026.11.15.766763</a>"""
    add_text(intro_text)

def add_heading(text: str):
    """
    Add a main heading to the report.
    
    Args:
        text: Heading text
    
    Example:
        add_heading("Executive Summary")
    """
    global _elements, _styles
    _check_initialized()
    
    heading = Paragraph(text, _styles['CustomHeading'])
    
    _elements.append(heading)
    _elements.append(Spacer(1, 6))


def add_subheading(text: str):
    """
    Add a subheading to the report.
    
    Args:
        text: Subheading text
    
    Example:
        add_subheading("Q1 Results")
    """
    global _elements, _styles
    _check_initialized()
    
    subheading = Paragraph(text, _styles['CustomSubheading'])
    
    _elements.append(subheading)
    _elements.append(Spacer(1, 6))

def add_subsubheading(text: str):
    """
    Add a subsubheading to the report.
    
    Args:
        text: Subsubheading text
    
    Example:
        add_subheading("Q1 Results")
    """
    global _elements, _styles
    _check_initialized()
    
    subheading = Paragraph(text, _styles['CustomSubheading2'])
    _elements.append(subheading)
    _elements.append(Spacer(1, 4))

def add_toc():
    """
    Add a table of contents showing all headings and subheadings.
    
    The TOC includes page numbers and links to each section.
    First-level headings (from add_heading) are shown in bold.
    Second-level headings (from add_subheading) are indented and not bold.
    
    Note: This should be called early in the report, typically after the title
    and before the main content.
    
    Example:
        create_report("report.pdf", title="My Report")
        add_toc()
        add_heading("Introduction")
        add_text("Content here...")
    """
    global _elements, _styles, _toc
    _check_initialized()
    
    # Enable TOC tracking and create TOC object
    _toc = TableOfContents()
    _toc.levelStyles = [
        ParagraphStyle(name='TOCHeading1', fontSize=10, fontName='Helvetica-Bold', 
                      leftIndent=0, firstLineIndent=0, spaceBefore=2, spaceAfter=2,
                      leading=12),
        ParagraphStyle(name='TOCHeading2', fontSize=10, fontName='Helvetica',
                      leftIndent=20, firstLineIndent=0, spaceBefore=2, spaceAfter=2,
                      leading=12),
    ]
    
    # Add TOC title
    toc_title = Paragraph('<b>Table of Contents</b>', _styles['CustomHeading'])
    _elements.append(toc_title)
    _elements.append(Spacer(1, 12))
    
    # Add the TableOfContents object
    _elements.append(_toc)
    _elements.append(Spacer(1, 12))
    _elements.append(PageBreak())


def add_text(text: str, style: str = 'JustifiedText'):
    """
    Add a paragraph of text. Supports basic HTML tags.
    
    Supported HTML tags: <b>, <i>, <u>, <sub>, <super>, <br/>, <font color="...">
    
    Args:
        text: Text content (can include basic HTML tags)
        style: Style name (default: 'JustifiedText' - justified except last line)
    
    Example:
        add_text("This is <b>bold</b> and <i>italic</i> text.")
        add_text("Chemical formula: H<sub>2</sub>O")
    """
    global _elements, _styles
    _check_initialized()
    
    para = Paragraph(text, _styles[style])
    _elements.append(para)
    _elements.append(Spacer(1, 12))


def _make_image(src, width, height, dpi: int = 150):
    """
    Build a ReportLab Image flowable, downsampling the source to roughly the
    display resolution and flattening any alpha channel onto white.

    ReportLab embeds the full-resolution pixel data of every image regardless of
    the on-page display size, and RGBA images additionally emit a separate
    soft-mask stream. In reports with thousands of small motif logos this both
    bloats the PDF (tens of MB) and inflates the image-embedding allocation burst
    during multiBuild. Pre-shrinking to the display size and dropping the alpha
    channel keeps the file small and the allocations modest, with no visible
    quality loss at the sizes these images are rendered.

    Args:
        src: Path to an image file or a file-like object (e.g. BytesIO).
        width: Display width in points (1/72 inch).
        height: Display height in points.
        dpi: Target pixel density for the embedded raster (default: 150).

    Returns:
        reportlab.platypus.Image sized to (width, height) points.
    """
    try:
        with PILImage.open(src) as im:
            im.load()
            # Flatten transparency onto white -> RGB, so reportlab emits no SMask.
            if im.mode in ('RGBA', 'LA', 'P'):
                im = im.convert('RGBA')
                bg = PILImage.new('RGBA', im.size, (255, 255, 255, 255))
                im = PILImage.alpha_composite(bg, im).convert('RGB')
            else:
                im = im.convert('RGB')
            # Downsample (never upsample) to the on-page display resolution.
            target_px = (max(1, int(width / inch * dpi)),
                         max(1, int(height / inch * dpi)))
            if im.width > target_px[0] or im.height > target_px[1]:
                im.thumbnail(target_px, PILImage.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format='PNG', optimize=True)
            buf.seek(0)
        return Image(buf, width=width, height=height)
    except Exception as e:
        # Fall back to direct embedding if PIL processing fails for any reason.
        log.warning("Could not pre-process image for embedding (%s); embedding as-is.", e)
        if hasattr(src, 'seek'):
            src.seek(0)
        return Image(src, width=width, height=height)


def add_plot(plot_obj,
             width: Optional[float] = None,
             height: Optional[float] = None,
             dpi: int = 150):
    """
    Add a matplotlib figure to the report.
    
    Args:
        plot_obj: Matplotlib figure object or axes object
        width: Width in inches (optional, maintains aspect ratio)
        height: Height in inches (optional, maintains aspect ratio)
        dpi: Resolution in dots per inch
    
    Example:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        add_plot(fig)
        
        # Or directly with axes:
        ax = plt.gca()
        add_plot(ax)
    """
    global _elements, _max_width, _max_height
    _check_initialized()
    
    # Handle both Figure and Axes objects
    if hasattr(plot_obj, 'figure'):
        # It's an Axes object
        fig = plot_obj.figure
    else:
        # It's a Figure object
        fig = plot_obj
    
    # Save plot to buffer
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=dpi)
    img_buffer.seek(0)
    
    # Get original size from figure
    fig_width, fig_height = fig.get_size_inches()
    
    # Determine final dimensions
    if width is None and height is None:
        # Use figure size, but constrain to max dimensions
        width = fig_width
        height = fig_height
    elif width is not None and height is None:
        # Width specified, calculate height maintaining aspect ratio
        height = width * (fig_height / fig_width)
    elif width is None and height is not None:
        # Height specified, calculate width maintaining aspect ratio
        width = height * (fig_width / fig_height)
    # else: both specified, use as-is (but will still be constrained below)
    
    # Constrain to maximum page dimensions while maintaining aspect ratio.
    # Reserve the spacer height (12 points = 12/72 inches) so the spacer that
    # follows always fits on the same page and doesn't create a blank page.
    _effective_max_height = _max_height - 12 / 72
    if width > _max_width:
        height = height * (_max_width / width)
        width = _max_width
    if height > _effective_max_height:
        width = width * (_effective_max_height / height)
        height = _effective_max_height
    
    img = _make_image(img_buffer, width*inch, height*inch, dpi=dpi)
    _elements.append(img)
    _elements.append(Spacer(1, 12))


def add_table(df: pd.DataFrame, 
              include_index: bool = False,
              header_color: str = '#4472C4',
              alternate_rows: bool = True,
              col_widths: Optional[List[float]] = None):
    """
    Add a pandas DataFrame as a formatted table.
    
    Args:
        df: Pandas DataFrame to add
        include_index: Whether to include the DataFrame index
        header_color: Hex color for header row
        alternate_rows: Whether to alternate row colors
        col_widths: Optional list of column widths in inches
    
    Example:
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        add_table(df, header_color='#2c3e50')
    """
    global _elements
    _check_initialized()
    
    # Convert DataFrame to list of lists
    if include_index:
        data = [[''] + list(df.columns)]
        data += [[str(idx)] + [str(val) for val in row] for idx, row in df.iterrows()]
    else:
        data = [list(df.columns)]
        data += df.values.tolist()
    
    # Convert all values to strings
    data = [[str(cell) for cell in row] for row in data]
    
    # Create table with optional column widths
    if col_widths:
        col_widths_pts = [w * inch for w in col_widths]
        table = Table(data, colWidths=col_widths_pts)
    else:
        table = Table(data)
    
    # Style the table
    style = [
        # Header row
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(header_color)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
        ('TOPPADDING', (0, 0), (-1, 0), 5),
        # Data rows
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]
    
    # Add alternating row colors if requested
    if alternate_rows:
        style.append(('ROWBACKGROUNDS', (0, 1), (-1, -1), 
                     [colors.white, colors.HexColor('#f2f2f2')]))
    
    table.setStyle(TableStyle(style))
    _elements.append(table)
    _elements.append(Spacer(1, 12))


def add_html(html_path: str, 
             scale_images: float = 0.4,
             image_aspect_ratio: float = 3.33):
    """
    Add HTML table content to the report as a ReportLab table.
    
    Parses HTML tables and creates native ReportLab tables with images.
    Tables automatically split across pages as needed.
    
    Args:
        html_path: Path to HTML file containing a table
        scale_images: Scale factor for image size (0.0-1.0, default: 0.4)
        image_aspect_ratio: Default width/height ratio for images (default: 3.33)
    
    Example:
        add_html("table.html")
        add_html("large_table.html", scale_images=0.3)
    """
    global _elements, _max_width, _max_height
    _check_initialized()
    
    if not os.path.exists(html_path):
        raise FileNotFoundError(f"HTML file not found: {html_path}")
    
    # Read and parse HTML
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table')
    
    if not table:
        log.warning(f"No table found in {html_path}")
        return
    
    # Get base directory for resolving relative image paths
    base_dir = os.path.dirname(os.path.abspath(html_path))
    
    # Extract table components
    thead = table.find('thead')
    tbody = table.find('tbody')
    
    if not tbody:
        log.warning(f"Warning: No tbody found in table")
        return
    
    # Parse the table into data (optimized version)
    header_data = _parse_table_header(thead)
    rows_data = _parse_table_rows(tbody, base_dir, scale_images, image_aspect_ratio)
    
    # Combine header with all rows
    table_data = [header_data] + rows_data
    
    # Create ReportLab table
    t = Table(table_data)
    
    # Style the table
    style = [
        # Header row
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 6),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 1),
        ('TOPPADDING', (0, 0), (-1, 0), 1),
        # Data rows
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), 
         [colors.white, colors.HexColor('#f2f2f2')]),
        ('LEFTPADDING', (0, 0), (-1, -1), 1),
        ('RIGHTPADDING', (0, 0), (-1, -1), 1),
        ('TOPPADDING', (0, 1), (-1, -1), 1),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 1),
    ]
    
    t.setStyle(TableStyle(style))
    
    # Enable table splitting across pages
    t._splitRowIndex = 0
    
    _elements.append(t)
    _elements.append(Spacer(1, 12))


def add_page_break():
    """
    Add a page break.
    
    Example:
        add_page_break()
    """
    global _elements
    _check_initialized()
    
    _elements.append(PageBreak())


def _parse_table_header(thead):
    """
    Fast parsing of HTML table header into ReportLab format.
    
    Args:
        thead: BeautifulSoup thead element
    
    Returns:
        List of header cell contents
    """
    if not thead:
        return []
    
    header_row = thead.find('tr')
    if not header_row:
        return []
    
    header_cells = header_row.find_all(['th', 'td'])
    return [Paragraph(f'<b>{cell.get_text(strip=True)}</b>', _styles['CompactCell']) 
            for cell in header_cells]


def _parse_table_rows(tbody, base_dir: str, scale_images: float, image_aspect_ratio: float):
    """
    Fast parsing of HTML table body rows into ReportLab format.
    Skip PIL image loading for significant speed improvement.
    
    Args:
        tbody: BeautifulSoup tbody element
        base_dir: Base directory for resolving image paths
        scale_images: Image scaling factor
        image_aspect_ratio: Width/height ratio for images
    
    Returns:
        List of row data, where each row is a list of cell contents
    """
    rows = tbody.find_all('tr')
    rows_data = []
    
    # Pre-calculate image dimensions
    target_height = 0.6 * inch * scale_images
    target_width = target_height * image_aspect_ratio
    
    for row in rows:
        cells = row.find_all(['td', 'th'])
        row_data = []
        
        for cell in cells:
            img = cell.find('img')
            
            if img and img.get('src'):
                img_src = img['src']
                
                # Handle 'NA' or invalid images
                if 'NA' in img_src or img_src == 'NA':
                    row_data.append(Paragraph('N/A', _styles['CompactCell']))
                    continue
                
                # Resolve image path
                if not img_src.startswith(('http://', 'https://', 'file://')):
                    if not os.path.isabs(img_src):
                        img_path = os.path.join(base_dir, img_src)
                    else:
                        img_path = img_src
                else:
                    # Handle file:// URLs
                    img_path = img_src.replace('file://', '')
                
                # Check if image exists and add it
                if os.path.exists(img_path):
                    try:
                        img_obj = _make_image(img_path, target_width, target_height)
                        row_data.append(img_obj)
                    except:
                        row_data.append(Paragraph('N/A', _styles['CompactCell']))
                else:
                    row_data.append(Paragraph('N/A', _styles['CompactCell']))
            else:
                # Plain text cell - handle NaN
                cell_text = cell.get_text(strip=True)
                if cell_text.lower() == 'nan':
                    cell_text = 'N/A'
                row_data.append(Paragraph(cell_text, _styles['CompactCell']))
        
        rows_data.append(row_data)
    
    return rows_data


def add_spacer(height: float = 12):
    """
    Add vertical space.
    
    Args:
        height: Height in points (default: 12)
    
    Example:
        add_spacer(24)  # Add 24 points of space
    """
    global _elements
    _check_initialized()
    
    _elements.append(Spacer(1, height))


def add_image(image_path: str, 
              width: Optional[float] = None, 
              height: Optional[float] = None):
    """
    Add an image from a file.
    
    Args:
        image_path: Path to image file
        width: Width in inches (optional, maintains aspect ratio)
        height: Height in inches (optional, maintains aspect ratio)
    
    Example:
        add_image("chart.png", width=5)
        add_image("logo.png")  # Auto-sized with aspect ratio preserved
    """
    global _elements, _max_width, _max_height
    _check_initialized()
    
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Create a temporary Image object to get original dimensions
    temp_img = Image(image_path)
    orig_width = temp_img.drawWidth / inch
    orig_height = temp_img.drawHeight / inch
    
    # Determine final dimensions
    if width is None and height is None:
        # Use original size, but constrain to max dimensions
        width = orig_width
        height = orig_height
    elif width is not None and height is None:
        # Width specified, calculate height maintaining aspect ratio
        height = width * (orig_height / orig_width)
    elif width is None and height is not None:
        # Height specified, calculate width maintaining aspect ratio
        width = height * (orig_width / orig_height)
    # else: both specified, use as-is (but will still be constrained below)
    
    # Constrain to maximum page dimensions while maintaining aspect ratio
    if width > _max_width:
        height = height * (_max_width / width)
        width = _max_width
    if height > _max_height:
        width = width * (_max_height / height)
        height = _max_height
    
    img = _make_image(image_path, width*inch, height*inch)

    _elements.append(img)
    _elements.append(Spacer(1, 12))


def finalize_report():
    """
    Build and save the PDF report. Must be called at the end.
    
    Example:
        finalize_report()
    """
    global _doc, _elements, _filename
    _check_initialized()
    
    # Build the PDF with page numbers
    # Use multiBuild for TOC support (allows multiple passes to resolve page numbers)
    _doc.multiBuild(_elements, canvasmaker=NumberedPageCanvas)
    
    log.info(f"PDF report created: {_filename}")
    
    # Reset global state
    _reset_state()


def _check_initialized():
    """Check if report has been initialized."""
    if _doc is None:
        raise RuntimeError(
            "Report not initialized. Call create_report() first."
        )


def _reset_state():
    """Reset global state after finalizing report."""
    global _doc, _elements, _styles, _filename, _max_width, _max_height, _toc
    _doc = None
    _elements = []
    _styles = None
    _filename = None
    _max_width = None
    _max_height = None
    _toc = None
