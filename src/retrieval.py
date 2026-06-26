import os
import sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Add the src directory to sys.path so we can import local modules regardless of where we run the script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chunking import Document
from faiss_vector_store import load_vector_store

def retrieve_relevant_chunks(
    query: str, 
    index: faiss.Index, 
    chunks: list[Document], 
    model: SentenceTransformer, 
    k: int = 5
) -> list[tuple[Document, float]]:
    """
    Embeds the user's query and searches the FAISS index for the top k most similar chunks.
    
    Args:
        query (str): The question asked by the user.
        index (faiss.Index): The loaded FAISS vector database.
        chunks (list[Document]): The loaded text chunk document list.
        model (SentenceTransformer): The embedding model.
        k (int): Number of chunks to retrieve.
        
    Returns:
        list[tuple[Document, float]]: A list of tuples containing (Document, distance score)
    """
    # 1. Embed the query using the SAME model as the chunks
    # We pass the query inside a list so it outputs a 2D array of shape (1, 384)
    query_embedding = model.encode([query])
    query_embedding_array = np.array(query_embedding).astype('float32')
    
    # 2. Search the FAISS index
    # distances: L2 distances (smaller is closer/more similar)
    # indices: The list indexes corresponding to the stored vectors
    distances, indices = index.search(query_embedding_array, k)
    
    # 3. Map retrieved indexes back to the original text chunks
    retrieved_results = []
    
    # FAISS returns a 2D list. Since we only searched 1 query, we read from index 0
    for idx, dist in zip(indices[0], distances[0]):
        # FAISS might return -1 if there aren't enough vectors in the database
        if idx == -1:
            continue
            
        chunk = chunks[idx]
        retrieved_results.append((chunk, dist))
        
    return retrieved_results

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vectorstore_dir = os.path.join(project_root, "vectorstore")
    
    # Verify that the vector database files exist
    if not os.path.exists(vectorstore_dir):
        print(f"Error: Vector store folder not found at: {vectorstore_dir}. Please run faiss_vector_store.py first to build the index.")
    else:
        # Load index and chunks
        print("Loading FAISS index and chunk metadata...")
        index, chunks = load_vector_store(vectorstore_dir)
        
        # Load embedding model
        print("Loading embedding model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Accept a sample question
        print("\n" + "="*50)
        query = input("Ask a question about your PDF: ")
        if not query.strip():
            query = "What is the main topic of the document?"
            print(f"Using default question: '{query}'")
            
        print("="*50 + "\n")
        
        # Perform retrieval
        print(f"Searching for top 5 relevant chunks...")
        results = retrieve_relevant_chunks(query, index, chunks, model, k=5)
        
        # Print Results
        print("\n" + "="*50)
        print(f"USER QUESTION: {query}")
        print("="*50)
        
        if not results:
            print("No relevant chunks found.")
        else:
            for rank, (chunk, distance) in enumerate(results):
                print(f"\n[Rank {rank + 1}] | Page: {chunk.metadata.get('page_number')} | FAISS L2 Distance: {distance:.4f}")
                print(f"Content Preview:")
                print(f"\"{chunk.page_content[:200]}...\"")
                print("-" * 50)
