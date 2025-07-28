from __future__ import annotations

import io
import re
from typing import Any
from typing import List
from typing import Literal

from api.models.docs_gen import Page
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Response
from pydantic import BaseModel
from pydantic import Field
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak
from reportlab.platypus import Paragraph
from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Spacer
from shared.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


class documentationExportRequest(BaseModel):
    """
    Model for requesting a documentation export.
    """
    repo_name: str = Field(..., description='Name of the repository')
    pages: List[Page] = Field(..., description='List of documentation pages to export')
    format: Literal['md', 'pdf'] = Field(..., description='Export format (md, pdf)')


@router.post('/export/format')
async def export_documentation(request: documentationExportRequest):
    """
    Export documentation content as Markdown or JSON.

    Args:
        request: The export request containing documentation pages and format

    Returns:
        A downloadable file in the requested format
    """
    try:
        logger.info(f"Exporting documentation for {request.repo_name} in {request.format} format")

        repo_name = request.repo_name
        pages = request.pages
        content: Any = None
        if request.format == 'md':
            # Generate Markdown content
            content = generate_markdown_export(repo_name, pages)
            filename = f"{repo_name}_doc.md"
            media_type = 'text/markdown'

        elif request.format == 'pdf':
            # Generate PDF content
            content = generate_pdf_export(repo_name, pages)
            filename = f"{repo_name}_doc.pdf"
            media_type = 'application/pdf'

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")

        # Create response with appropriate headers for file download
        response = Response(
            content=content,
            media_type=media_type,
            headers={
                'Content-Disposition': f"attachment; filename={filename}",
            },
        )

        return response

    except Exception as e:
        error_msg = f"Error exporting documentation: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


def generate_markdown_export(repo_url: str, pages: List[Page]) -> str:
    """
    Generate Markdown export of documentation pages.

    Args:
        repo_url: The repository URL
        pages: List of documentation pages

    Returns:
        Markdown content as string
    """
    # Start with metadata
    markdown = f"# Documentation for {repo_url}\n\n"

    # Add table of contents
    markdown += '## Table of Contents\n\n'
    for page in pages:
        markdown += f"- [{page.page_title}](#{page.page_id})\n"
    markdown += '\n'

    # Add each page
    for page in pages:
        markdown += f"<a id='{page.page_id}'></a>\n\n"
        markdown += f"## {page.page_title}\n\n"

        # Add page content
        markdown += f"{page.content}\n\n"
        markdown += '---\n\n'

    return markdown


def generate_pdf_export(repo_name: str, pages: List[Page]) -> bytes:
    """
    Generate PDF export of documentation pages.

    Args:
        repo_name: The repository name
        pages: List of documentation pages

    Returns:
        PDF content as bytes
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=1 * inch)

    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1,  # Center alignment
    )
    heading_style = styles['Heading1']
    subheading_style = styles['Heading2']
    normal_style = styles['Normal']

    story = []

    # Add title
    story.append(Paragraph(f'Documentation for {repo_name}', title_style))
    story.append(Spacer(1, 0.5 * inch))

    # Add table of contents
    story.append(Paragraph('Table of Contents', heading_style))
    story.append(Spacer(1, 0.2 * inch))

    for i, page in enumerate(pages, 1):
        story.append(Paragraph(f"{i}. {page.page_title}", normal_style))

    story.append(PageBreak())

    # Add each page
    for i, page in enumerate(pages, 1):
        # Add page title
        story.append(Paragraph(f"{i}. {page.page_title}", heading_style))
        story.append(Spacer(1, 0.2 * inch))

        # Process content
        content_lines = page.content.split('\n')
        current_paragraph = ''

        for line in content_lines:
            line = line.strip()
            if line:
                # Handle headers
                if line.startswith('###'):
                    if current_paragraph:
                        story.append(Paragraph(clean_markdown_text(current_paragraph), normal_style))
                        current_paragraph = ''
                    story.append(Paragraph(line.replace('###', '').strip(), subheading_style))
                elif line.startswith('##'):
                    if current_paragraph:
                        story.append(Paragraph(clean_markdown_text(current_paragraph), normal_style))
                        current_paragraph = ''
                    story.append(Paragraph(line.replace('##', '').strip(), subheading_style))
                elif line.startswith('#'):
                    if current_paragraph:
                        story.append(Paragraph(clean_markdown_text(current_paragraph), normal_style))
                        current_paragraph = ''
                    story.append(Paragraph(line.replace('#', '').strip(), subheading_style))
                else:
                    # Regular content
                    if current_paragraph:
                        current_paragraph += ' ' + line
                    else:
                        current_paragraph = line
            else:
                # Empty line, add current paragraph if exists
                if current_paragraph:
                    story.append(Paragraph(clean_markdown_text(current_paragraph), normal_style))
                    story.append(Spacer(1, 0.1 * inch))
                    current_paragraph = ''

        # Add remaining paragraph
        if current_paragraph:
            story.append(Paragraph(clean_markdown_text(current_paragraph), normal_style))

        # Add page break except for last page
        if i < len(pages):
            story.append(PageBreak())

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def clean_markdown_text(text: str) -> str:
    """
    Clean markdown formatting from text for PDF generation.

    Args:
        text: Text with markdown formatting

    Returns:
        Cleaned text
    """
    # Remove markdown formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)  # Bold
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)      # Italic
    text = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', text)  # Code
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)      # Links

    return text
