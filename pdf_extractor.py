"""
Module 1: PDF Text Extraction (Lightweight)
Extracts text and tables from digital/searchable PDFs using pdfplumber.
Scanned PDFs are detected and flagged for pre-processing.
"""

import pdfplumber
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ExtractionResult:
    """Container for PDF extraction results."""
    success: bool
    text: str
    tables: list[list[list[str]]]  # List of tables, each table is list of rows
    page_count: int
    is_scanned: bool
    message: str


SCANNED_PDF_HELP = """
SCANNED PDF DETECTED - Text cannot be extracted directly.

Please convert to a searchable PDF first using one of these free tools:
  - Google Drive: Upload PDF, right-click > Open with Google Docs (auto-OCR)
  - Microsoft OneDrive: Upload, open in Word Online
  - Adobe Acrobat Online: https://www.adobe.com/acrobat/online/pdf-to-word.html
  - NAPS2 (Windows): Free desktop app with batch OCR

After conversion, re-run this tool with the searchable PDF.
"""


def extract_text_and_tables(pdf_path: str) -> tuple[str, list]:
    """Extract text and tables from PDF using pdfplumber."""
    full_text = []
    all_tables = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            full_text.append(page_text)
            
            tables = page.extract_tables()
            if tables:
                all_tables.extend(tables)
    
    return "\n\n".join(full_text), all_tables


def is_scanned_pdf(text: str, page_count: int, min_chars_per_page: int = 50) -> bool:
    """Check if PDF appears to be scanned (minimal extractable text)."""
    if not text.strip():
        return True
    return len(text.strip()) < (min_chars_per_page * page_count)


def extract_pdf(pdf_path: str) -> ExtractionResult:
    """
    Main extraction function for digital PDFs.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        ExtractionResult with text, tables, and metadata
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if path.suffix.lower() != '.pdf':
        raise ValueError(f"Not a PDF file: {pdf_path}")
    
    # Get page count
    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
    
    # Extract content
    text, tables = extract_text_and_tables(pdf_path)
    
    # Check if scanned
    if is_scanned_pdf(text, page_count):
        return ExtractionResult(
            success=False,
            text="",
            tables=[],
            page_count=page_count,
            is_scanned=True,
            message=SCANNED_PDF_HELP
        )
    
    return ExtractionResult(
        success=True,
        text=text,
        tables=tables,
        page_count=page_count,
        is_scanned=False,
        message=f"Extracted {len(text)} characters and {len(tables)} tables from {page_count} pages."
    )


def tables_to_dicts(tables: list[list[list[str]]]) -> list[list[dict]]:
    """
    Convert table data to list of dictionaries using first row as headers.
    Useful for structured police report forms.
    """
    result = []
    for table in tables:
        if len(table) < 2:
            continue
        headers = [str(h).strip() if h else f"col_{i}" for i, h in enumerate(table[0])]
        rows = []
        for row in table[1:]:
            row_dict = {}
            for i, cell in enumerate(row):
                key = headers[i] if i < len(headers) else f"col_{i}"
                row_dict[key] = str(cell).strip() if cell else ""
            rows.append(row_dict)
        result.append(rows)
    return result


# === CLI for testing ===
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py <pdf_file>")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    
    print(f"Extracting: {pdf_file}")
    print("-" * 50)
    
    try:
        result = extract_pdf(pdf_file)
        
        if not result.success:
            print(result.message)
            sys.exit(1)
        
        print(f"Pages: {result.page_count}")
        print(f"Tables found: {len(result.tables)}")
        print(result.message)
        print("-" * 50)
        print("TEXT PREVIEW (first 1000 chars):")
        print(result.text[:1000])
        
        if result.tables:
            print("-" * 50)
            print("FIRST TABLE:")
            for row in result.tables[0][:5]:
                print(row)
                
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)