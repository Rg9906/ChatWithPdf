import os
import sys
from sentence_transformers import SentenceTransformer

# Add the src directory to sys.path so we can import local modules regardless of where we run the script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chunking import prepare_chunks, Document

def generate_embeddings(chunks: list[Document], model_name: str = "all-MiniLM-L6-v2"):
    """
    Generates semantic vector embeddings for a list of text chunks.
    
    AN EMBEDDING IS:
    A high-dimensional vector (list of floating-point numbers) that represents 
    the semantic meaning of a piece of text. 
    - Text segments with similar meanings (e.g., "AI and deep learning" and "neural networks")
      will be placed close to each other in this high-dimensional vector space.
    - Text segments with different meanings will be placed far apart.
    - This allows us to perform "semantic search" rather than just keyword matching.
    """
    # 1. Extract the raw text from each Document chunk
    texts = [chunk.page_content for chunk in chunks]
    
    # 2. Load the Sentence Transformer model
    # 'all-MiniLM-L6-v2' maps sentences/paragraphs into a 384-dimensional vector space.
    print(f"Loading embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)
    
    # 3. Generate embeddings
    # The model converts each text chunk into a 384-dimensional dense vector.
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    return embeddings, model

if __name__ == "__main__":
    # Define target test PDF path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_dir = os.path.join(project_root, "data")
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")] if os.path.exists(pdf_dir) else []
    
    if not pdf_files:
        print(f"Error: Please place a PDF file inside the 'data/' folder at the project root: {pdf_dir}")
    else:
        test_pdf = os.path.join(pdf_dir, pdf_files[0])
        
        # 1. Run Chunking
        print("--- Step 1: Chunking PDF ---")
        chunks = prepare_chunks(test_pdf, chunk_size=1000, chunk_overlap=200)
        
        if not chunks:
            print("No text chunks found to embed.")
        else:
            # 2. Generate Embeddings
            print("\n--- Step 2: Generating Embeddings ---")
            embeddings, model = generate_embeddings(chunks)
            
            # Print Results as required
            print("\n--- EMBEDDING GENERATION SUMMARY ---")
            print(f"Number of embeddings generated: {len(embeddings)}")
            print(f"Shape/dimension of one embedding: {embeddings[0].shape}")
            print(f"First 10 values of the first embedding:\n{embeddings[0][:10]}")
            print("------------------------------------\n")
