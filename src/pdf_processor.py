import pypdf
from typing import List, Dict

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, any]]:
    """
    Reads a PDF file and extracts text page by page.
    
    Args:
        pdf_path (str): The path to the PDF file.
        
    Returns:
        List[Dict[str, any]]: A list of dictionaries, where each dict has:
                              - 'page_number': the 1-based page index
                              - 'text': the raw text extracted from that page
    """
    pages_data = []
    
    # Open and read the PDF
    reader = pypdf.PdfReader(pdf_path)
    
    # Loop through all the pages
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        pages_data.append({
            "page_number": i + 1,
            "text": text or ""  # Ensure we don't store None if extraction is empty
        })
        
    return pages_data

if __name__ == "__main__":
    # Test block
    import sys
    import os
    
    # We will test this with a sample PDF in the next step
    print("PDF Processor module loaded successfully!")
