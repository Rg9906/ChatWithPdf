import os
import re
import sys
from typing import List, Dict

# Add the src directory to sys.path so we can import local modules regardless of where we run the script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf_processor import extract_text_from_pdf

def clean_text(text: str) -> str:
    """
    Cleans raw text by normalizing whitespace, removing excess newlines,
    and trimming spaces while preserving semantic content.
    """
    # Replace multiple spaces/newlines with a single space
    cleaned = re.sub(r'\s+', ' ', text)
    return cleaned.strip()

class Document:
    """A simple container for text and metadata, compatible with LangChain splitters."""
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

def prepare_chunks(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Reads a PDF, cleans the text, and splits it into semantic chunks
    while preserving page number metadata.
    """
    # Step 1: Extract text from PDF (Page by Page)
    pages_data = extract_text_from_pdf(pdf_path)
    
    # Step 2: Clean text and convert to Document objects
    documents = []
    for page in pages_data:
        cleaned_text = clean_text(page["text"])
        # Only process pages that contain actual text
        if cleaned_text:
            doc = Document(
                page_content=cleaned_text,
                metadata={"page_number": page["page_number"]}
            )
            documents.append(doc)
            
    # Step 3: Initialize the RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Step 4: Split documents into chunks
    # split_documents automatically propagates metadata (page_number) to the chunks
    chunks = text_splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":
    # Define target test PDF path (expects pdf inside data/ folder at project root)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_dir = os.path.join(project_root, "data")
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")] if os.path.exists(pdf_dir) else []
    
    if not pdf_files:
        print(f"Error: Please place a PDF file inside the 'data/' folder at the project root: {pdf_dir}")
    else:
        test_pdf = os.path.join(pdf_dir, pdf_files[0])
        print(f"Processing PDF: {test_pdf}...\n")
        
        # Run chunking
        chunks = prepare_chunks(test_pdf, chunk_size=1000, chunk_overlap=200)
        
        # Print Results as required
        print(f"--- CHUNKING SUMMARY ---")
        print(f"Total number of chunks: {len(chunks)}")
        print(f"------------------------\n")
        
        # Print first 3 chunks and their lengths
        print("--- FIRST 3 CHUNKS ---")
        for i, chunk in enumerate(chunks[:3]):
            print(f"Chunk #{i + 1} (Page {chunk.metadata.get('page_number')}) | Length: {len(chunk.page_content)} characters")
            print(f"Content Preview:")
            print(f"\"{chunk.page_content[:200]}...\"")
            print("-" * 50)
