import os
import sys
import pickle
import numpy as np
import faiss

# Add the src directory to sys.path so we can import local modules regardless of where we run the script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chunking import Document
from embedding_generation import generate_embeddings, prepare_chunks

def create_vector_store(embeddings: np.ndarray, chunks: list[Document]):
    """
    Builds a FAISS index from the generated embeddings.
    
    FAISS (Facebook AI Similarity Search) is a library optimized for search in 
    dense vector spaces. Because FAISS *only* stores vectors and does not know 
    about the underlying text or metadata, we maintain a separate mapped list of chunks.
    """
    # FAISS requires input vectors to be in float32 format
    embeddings_array = np.array(embeddings).astype('float32')
    
    # Get the embedding dimension (for all-MiniLM-L6-v2, this is 384)
    dimension = embeddings_array.shape[1]
    
    # We use IndexFlatL2 which measures similarity using L2 (Euclidean) distance.
    # It performs an exact brute-force search, which is highly accurate and perfect
    # for small-to-medium datasets.
    index = faiss.IndexFlatL2(dimension)
    
    # Add the embeddings to the vector database index
    index.add(embeddings_array)
    
    return index

def save_vector_store(index: faiss.Index, chunks: list[Document], folder_path: str = "vectorstore"):
    """
    Saves the FAISS index and the corresponding text chunks to disk.
    
    We save:
    1. The FAISS index structure containing the vectors (.faiss)
    2. The original document chunks serialized with pickle (.pkl) so we can map
       retrieved vector IDs back to their text contents.
    """
    # Create output directory if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    
    index_path = os.path.join(folder_path, "index.faiss")
    chunks_path = os.path.join(folder_path, "chunks.pkl")
    
    # Save the FAISS index file
    faiss.write_index(index, index_path)
    
    # Save the text chunks list using Python serialization (pickle)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
        
    print(f"FAISS index saved successfully to: {index_path}")
    print(f"Chunk metadata saved successfully to: {chunks_path}")

def load_vector_store(folder_path: str = "vectorstore"):
    """
    Loads both the FAISS index and text chunks from disk.
    """
    index_path = os.path.join(folder_path, "index.faiss")
    chunks_path = os.path.join(folder_path, "chunks.pkl")
    
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Vector store files not found in folder: {folder_path}")
        
    # Read the FAISS index file
    index = faiss.read_index(index_path)
    
    # Deserialize the chunks list
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
        
    return index, chunks

if __name__ == "__main__":
    # Test block
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_dir = os.path.join(project_root, "data")
    vectorstore_dir = os.path.join(project_root, "vectorstore")
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")] if os.path.exists(pdf_dir) else []
    
    if not pdf_files:
        print(f"Error: Please place a PDF file inside the 'data/' folder at the project root: {pdf_dir}")
    else:
        test_pdf = os.path.join(pdf_dir, pdf_files[0])
        
        # 1. Run Chunking
        print("--- Step 1: Chunking PDF ---")
        chunks = prepare_chunks(test_pdf, chunk_size=1000, chunk_overlap=200)
        
        # 2. Run Embeddings
        print("\n--- Step 2: Generating Embeddings ---")
        embeddings, _ = generate_embeddings(chunks)
        
        # 3. Create FAISS Database
        print("\n--- Step 3: Creating FAISS Index ---")
        index = create_vector_store(embeddings, chunks)
        
        print(f"Number of vectors stored: {index.ntotal}")
        print(f"Embedding dimension: {index.d}")
        
        # 4. Save Database
        print("\n--- Step 4: Saving Vector Store to disk ---")
        save_vector_store(index, chunks, folder_path=vectorstore_dir)
        
        # 5. Load Database to Verify
        print("\n--- Step 5: Loading Vector Store to Verify ---")
        loaded_index, loaded_chunks = load_vector_store(folder_path=vectorstore_dir)
        print(f"Verified! Loaded index has {loaded_index.ntotal} vectors with dimension {loaded_index.d}.")
        print(f"Verified! Loaded {len(loaded_chunks)} chunks from pickle file.")
        print("-------------------------------------------\n")
