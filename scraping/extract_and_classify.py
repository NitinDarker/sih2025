import os
import shutil
import subprocess
from typing import Optional

import fitz  # PyMuPDF
from transformers import pipeline

# ------------------ Zero-Shot Classifier ------------------

# Initialize once globally
print("🔎 Loading zero-shot classification model...")
classifier = pipeline(
    "zero-shot-classification",
    model="microsoft/mdeberta-v3-base"
)

LABELS = [
    "static data",
    "dynamic data"
]

def classify_text(text: str) -> str:
    """Classify text as static or dynamic using zero-shot classification."""
    if not text.strip():
        return "unknown"

    result = classifier(text, LABELS, multi_label=False)
    label = result["labels"][0]
    score = result["scores"][0]
    print(f"✅ Classified as: {label} (score: {score:.2f})")
    return label


# ------------------ PDF Processing Functions ------------------

def is_digital(pdf_path: str) -> bool:
    """Check if a PDF is digital (has extractable text)."""
    try:
        with fitz.open(pdf_path) as doc:
            text_pages = sum(1 for page in doc if page.get_text().strip())
            return text_pages >= doc.page_count / 2
    except Exception:
        return False


def ghostscript_repair(pdf_path: str, repaired_dir: str) -> str:
    """Repair a digital PDF using Ghostscript."""
    os.makedirs(repaired_dir, exist_ok=True)
    out_path = os.path.join(repaired_dir, os.path.basename(pdf_path))

    gs_candidates = ["gswin64c", "gswin32c", "gs"]
    gs = None
    for cand in gs_candidates:
        try:
            subprocess.run([cand, "-v"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            gs = cand
            break
        except Exception:
            continue

    if gs is None:
        raise RuntimeError("Ghostscript not found. Please install it.")

    subprocess.run(
        [gs, "-o", out_path, "-sDEVICE=pdfwrite", "-dPDFSETTINGS=/prepress", pdf_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out_path


def ocr_extract(pdf_path: str, ocr_dir: str, reader, dpi: int = 200) -> str:
    """OCR a scanned PDF and save extracted text."""
    os.makedirs(ocr_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    txt_path = os.path.join(ocr_dir, f"{base}_ocr.txt")

    with fitz.open(pdf_path) as doc, open(txt_path, "w", encoding="utf-8") as txt_file:
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)
            ocr_result = reader.readtext(pix.tobytes(), detail=0, paragraph=True)
            txt_file.write(f"Page {page_num + 1}:\n")
            txt_file.write(" ".join(ocr_result))
            txt_file.write("\n" + "-" * 80 + "\n")

    return txt_path


def process_pdf(pdf_path: str, repaired_dir: str, ocr_dir: str, error_dir: Optional[str] = None, reader=None) -> None:
    """Process a single PDF: repair or OCR + classify + organize."""
    try:
        if is_digital(pdf_path):
            print(f"📄 Processing digital PDF: {os.path.basename(pdf_path)}")
            repaired_path = ghostscript_repair(pdf_path, repaired_dir)

            # Extract text from repaired PDF
            with fitz.open(repaired_path) as doc:
                text_content = "\n".join(page.get_text() for page in doc)

            label = classify_text(text_content)
            classified_dir = os.path.join(repaired_dir, label.replace(" ", "_"))
            os.makedirs(classified_dir, exist_ok=True)
            shutil.move(repaired_path, os.path.join(classified_dir, os.path.basename(repaired_path)))

        else:
            print(f"📄 Processing scanned PDF: {os.path.basename(pdf_path)}")
            if reader is None:
                import easyocr
                reader = easyocr.Reader(["hi", "en"])

            txt_path = ocr_extract(pdf_path, ocr_dir, reader)

            # Classify extracted text
            with open(txt_path, "r", encoding="utf-8") as f:
                text_content = f.read()

            label = classify_text(text_content)
            classified_dir = os.path.join(ocr_dir, label.replace(" ", "_"))
            os.makedirs(classified_dir, exist_ok=True)
            shutil.move(txt_path, os.path.join(classified_dir, os.path.basename(txt_path)))

    except Exception as e:
        print(f"Error processing {os.path.basename(pdf_path)}: {e}")
        if error_dir:
            os.makedirs(error_dir, exist_ok=True)
            try:
                shutil.copy(pdf_path, os.path.join(error_dir, os.path.basename(pdf_path)))
            except Exception:
                pass


def main(
    input_dir: str = "./data/pdfs",
    repaired_dir: str = "./sorted_data/repaired",
    ocr_dir: str = "./sorted_data/ocr_texts",
    error_dir: Optional[str] = "./sorted_data/error",
    limit: Optional[int] = 50,
):
    """Main function to process all PDFs and classify content."""
    if not os.path.isdir(input_dir):
        print(f"Input directory '{input_dir}' does not exist.")
        return

    os.makedirs(repaired_dir, exist_ok=True)
    os.makedirs(ocr_dir, exist_ok=True)
    if error_dir:
        os.makedirs(error_dir, exist_ok=True)

    # Initialize EasyOCR reader (once)
    reader = None
    try:
        import easyocr
        reader = easyocr.Reader(["hi", "en"], gpu=True)
    except Exception:
        reader = None

    count = 0
    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith(".pdf"):
            continue
        if limit is not None and count >= limit:
            break
        pdf_path = os.path.join(input_dir, filename)
        process_pdf(pdf_path, repaired_dir, ocr_dir, error_dir=error_dir, reader=reader)
        count += 1

    print("✅ Processing completed.")


# ------------------ Entry Point ------------------

def run_extract():
    main()



