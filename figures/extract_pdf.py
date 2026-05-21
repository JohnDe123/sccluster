"""Extract text from reference PDF."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

pdf_path = r"C:\Users\25605\Downloads\3733006.3733013.pdf"

# Try pymupdf (fitz)
try:
    import fitz
    doc = fitz.open(pdf_path)
    print(f"Pages: {len(doc)}")
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            print(f"\n=== Page {i+1} ===")
            print(text[:3000])
    doc.close()
    sys.exit(0)
except ImportError:
    pass

# Try pdfplumber
try:
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Pages: {len(pdf.pages)}")
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                print(f"\n=== Page {i+1} ===")
                print(text[:3000])
    sys.exit(0)
except ImportError:
    pass

# Try PyPDF2
try:
    from PyPDF2 import PdfReader
    reader = PdfReader(pdf_path)
    print(f"Pages: {len(reader.pages)}")
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            print(f"\n=== Page {i+1} ===")
            print(text[:3000])
    sys.exit(0)
except ImportError:
    pass

print("No PDF library available. Installing pymupdf...")
