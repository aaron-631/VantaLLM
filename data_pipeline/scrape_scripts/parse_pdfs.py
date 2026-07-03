import os
import json
import fitz  # PyMuPDF
import re
from pathlib import Path

# Try importing OCR tools for image-only PDFs
try:
    import pytesseract
    from PIL import Image
    import io
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

PDF_DIR = Path("data_pipeline/raw/pdfs")
OUTPUT_FILE = Path("data_pipeline/raw/pretrain/conference_pdfs.jsonl")

def clean_text(text):
    """Strips null bytes and excessive whitespace that can break tokenizers."""
    text = text.replace('\x00', '')
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def process_pdf(pdf_path):
    """Extracts text while respecting layout, with an OCR fallback for images."""
    doc = fitz.open(pdf_path)
    full_text = ""
    ocr_triggered = False
    
    for page in doc:
        # sort=True forces PyMuPDF to read top-to-bottom, column-by-column
        page_text = page.get_text("text", sort=True)
        
        # Fallback: If the page yields almost no text, it's likely a scanned image
        if len(page_text.strip()) < 50 and OCR_AVAILABLE:
            try:
                # Render the PDF page to a 150 DPI image in memory
                pix = page.get_pixmap(dpi=150)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                # Run OCR to extract the text
                page_text = pytesseract.image_to_string(img)
                ocr_triggered = True
            except Exception:
                pass
                
        full_text += page_text + "\n"
        
    return clean_text(full_text), ocr_triggered

def extract_pdfs():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(PDF_DIR.rglob("*.pdf"))
    print(f"[INFO] Found {len(pdf_files)} PDFs to process...")
    
    if not pdf_files:
        print(f"[WARN] No PDFs found in {PDF_DIR}. Put files there first.")
        return

    count = 0
    ocr_count = 0
    
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        for pdf_path in pdf_files:
            try:
                extracted_text, used_ocr = process_pdf(pdf_path)
                
                # Only keep documents with actual substance
                if len(extracted_text) > 200:
                    record = {
                        "text": extracted_text,
                        "source": "conference_pdf_archive",
                        "url": f"local://pdfs/{pdf_path.name}",
                        "type": "conference_paper"
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                    if used_ocr: ocr_count += 1
                    
            except Exception as e:
                print(f"  [ERROR] Skipping {pdf_path.name}: {e}")
                
    print("=" * 50)
    print(f"[SUCCESS] Parsed {count} PDFs into {OUTPUT_FILE.name}")
    if ocr_count > 0:
        print(f"[INFO] OCR Engine successfully recovered {ocr_count} image-only PDFs.")
    print("=" * 50)

if __name__ == "__main__":
    extract_pdfs()
