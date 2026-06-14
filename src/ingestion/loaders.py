import os
import csv
from bs4 import BeautifulSoup
from pypdf import PdfReader

class DocumentLoader:
    """
    Handles native text extraction for multiple file extensions (.txt, .html, .pdf, .csv)
    and formats them uniformly into unified extraction payloads.
    """
    
    @staticmethod
    def load_txt(file_path: str) -> str:
        """Reads raw content out of a standard plaintext file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def load_html(file_path: str) -> str:
        """Parses HTML markup natively and pulls clean structural text elements."""
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator="\n").strip()

    @staticmethod
    def load_pdf(file_path: str) -> str:
        """Extracts native text page-by-page from a binary PDF layer."""
        extracted_text = []
        reader = PdfReader(file_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text.append(text)
        return "\n".join(extracted_text).strip()

    @staticmethod
    def load_csv_cells(file_path: str) -> str:
        """
        Reads tabular data row-by-row and extracts long unstructured text columns 
        (like patient notes) into a combined corpus for chunking.
        """
        combined_text_blocks = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            # Dynamically look for common unstructured text columns in healthcare datasets
            text_column = None
            
            # DEFENSIVE TYPE GUARD: Ensure fieldnames is not None before iterating
            if reader.fieldnames:
                for col in reader.fieldnames:
                    if col.lower() in ["pn_history", "patient_notes", "text", "content", "notes"]:
                        text_column = col
                        break
            
            # If no target text column matches, fallback to combining the whole row text
            for row in reader:
                if text_column and row[text_column]:
                    combined_text_blocks.append(row[text_column])
                else:
                    # Fallback string combination
                    combined_text_blocks.append(" | ".join(str(val) for val in row.values() if val))
                    
        return "\n\n--- DOCUMENT BREAK ---\n\n".join(combined_text_blocks)

    def load_file(self, file_path: str) -> dict:
        """
        Dynamically detects file extensions, routes processing rules, 
        and maps fields into a uniform payload structure.
        """
        filename = os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower()
        
        if ext == ".txt":
            raw_content = self.load_txt(file_path)
        elif ext in [".html", ".htm"]:
            raw_content = self.load_html(file_path)
        elif ext == ".pdf":
            raw_content = self.load_pdf(file_path)
        elif ext == ".csv":
            raw_content = self.load_csv_cells(file_path)
        else:
            raise ValueError(f"Unsupported file extension encountered: {ext}")
            
        return {
            "filename": filename,
            "file_type": ext,
            "raw_text": raw_content
        }

    def load_directory(self, directory_path: str) -> list[dict]:
        """Scans a target directory path and extracts text from all supported assets."""
        documents = []
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                try:
                    documents.append(self.load_file(file_path))
                except ValueError as e:
                    print(f"Skipping file {file}: {e}")
        return documents