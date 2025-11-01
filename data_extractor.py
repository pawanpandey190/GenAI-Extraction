import os
import pymupdf as fitz  # PyMuPDF
try:
    import fitz  # PyMuPDF
except ImportError:
    import pymupdf as fitz
import pytesseract
from PIL import Image
from io import BytesIO
import concurrent.futures
import pdfplumber


# Pdf extraction
"""
Optimized Context-Aware PDF Extractor
-------------------------------------
Extracts text + tables in reading order.
Preserves table context with placeholders in text.
Automatically switches to OCR for scanned PDFs.
Uses concurrent execution for performance.
"""



class ContextAwarePDFExtractor:
    def __init__(self, pdf_path: str, ocr: bool = True):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")
        self.pdf_path = pdf_path
        self.ocr = ocr

    # ------------------ Type Detection ------------------ #
    def _is_text_based(self) -> bool:
        """Detect if PDF contains selectable text."""
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages[:3]:
                if page.extract_text():
                    return True
        return False

    # ------------------ OCR Extraction ------------------ #
    def _extract_text_with_ocr(self):
        """Extract text from scanned (image-based) PDFs."""
        text = []
        with fitz.open(self.pdf_path) as doc:
            for page_index, page in enumerate(doc):
                pix = page.get_pixmap(dpi=300)
                img = Image.open(BytesIO(pix.tobytes()))
                ocr_text = pytesseract.image_to_string(img)
                text.append(f"\n\n--- Page {page_index + 1} ---\n{ocr_text}")
        return "\n".join(text)

    # ------------------ Context-Aware Extraction ------------------ #
    def _extract_text_with_context(self):
        """
        Extract text and tables in reading order.
        Inserts tables inline as formatted text blocks.
        """
        full_content = []

        with pdfplumber.open(self.pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_content = []
                elements = []

                # Extract words + bounding boxes
                for word in page.extract_words():
                    elements.append({
                        "type": "text",
                        "y0": word["top"],
                        "y1": word["bottom"],
                        "content": word["text"]
                    })

                # Extract tables + bounding boxes
                tables = page.find_tables()
                for table in tables:
                    elements.append({
                        "type": "table",
                        "y0": table.bbox[1],
                        "y1": table.bbox[3],
                        "content": table.extract()
                    })

                # Sort by vertical position (y0)
                elements = sorted(elements, key=lambda e: e["y0"])

                # Merge text + tables in order
                for element in elements:
                    if element["type"] == "text":
                        page_content.append(element["content"])
                    elif element["type"] == "table":
                        table_text = self._format_table_as_text(element["content"])
                        page_content.append("\n\n[Table]\n" + table_text + "\n[/Table]\n")

                full_page_text = " ".join(page_content)
                full_content.append(f"\n\n--- Page {page_num} ---\n{full_page_text}")

        return "\n".join(full_content)

    # ------------------ Helpers ------------------ #
    def _format_table_as_text(self, table):
        """Convert table (list of lists) to readable text."""
        if not table:
            return ""
        col_widths = [max(len(str(cell)) for cell in col) for col in zip(*table)]
        rows = []
        for row in table:
            formatted_row = " | ".join(
                str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)
            )
            rows.append(formatted_row)
        return "\n".join(rows)

    # ------------------ Metadata ------------------ #
    def extract_metadata(self):
        """Extract metadata (author, title, etc.) using PyMuPDF."""
        with fitz.open(self.pdf_path) as doc:
            return doc.metadata

    # ------------------ Main Extraction ------------------ #
    def extract_all(self):
        """Extract full PDF with context + metadata."""
        results = {}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            metadata_future = executor.submit(self.extract_metadata)

            # Detect PDF type once
            if self._is_text_based():
                text_future = executor.submit(self._extract_text_with_context)
            else:
                text_future = executor.submit(self._extract_text_with_ocr)

            results["text_with_context"] = text_future.result()
            results["metadata"] = metadata_future.result()

        return results


# ------------------ Example Usage ------------------ #
# if __name__ == "__main__":
#     pdf_path = r"C:\Users\Pawan Pandey\Documents\HPCL document\HPL Docs\HPL Docs\Inputs\TenderDocumentTS1471.pdf"  # change to your path
#     extractor = ContextAwarePDFExtractor(pdf_path)

#     data = extractor.extract_all()

#     print("\n📄 Metadata:\n", data["metadata"])
#     print("\n🧾 Extracted Content (first 1000 chars):\n")
#     print(data["text_with_context"][:3000])
