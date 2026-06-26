import os
import sys
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Global Debug flag (when True, shows detailed prompt and chunks)
DEBUG = False

# Add the src directory to sys.path so we can import local modules regardless of where we run the script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chunking import Document
from faiss_vector_store import load_vector_store
from retrieval import retrieve_relevant_chunks

# Load environment variables from the root .env file
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=env_path)

# Configure the Google Gemini API client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key or "your_api_key" in api_key:
    print("Warning: GEMINI_API_KEY is not set correctly in your .env file.")
else:
    genai.configure(api_key=api_key)

def construct_prompt(query: str, retrieved_results: list[tuple[Document, float]]) -> str:
    """
    Constructs a structured prompt for Gemini, grounding it in the retrieved context.
    """
    # 1. Compile the context chunks with their source pages
    context_str = ""
    for rank, (chunk, distance) in enumerate(retrieved_results):
        page_num = chunk.metadata.get("page_number", "Unknown")
        context_str += f"\n[Context Source #{rank + 1} | Page {page_num}]:\n{chunk.page_content}\n"
        
    # Map variables for prompt formatting
    context = context_str
    question = query
        
    # 2. Build the strict prompt
    prompt = f"""You are an expert document analysis and question-answering assistant.

Your primary responsibility is to answer questions using ONLY the information contained in the retrieved document context provided below.

Guidelines:

1. Base every answer strictly on the retrieved context.
2. You MAY:
   - summarize information
   - combine information from multiple retrieved passages
   - infer reasonable conclusions that are directly supported by the document
   - compare information within the document
   - evaluate the quality of the document (e.g., resumes, reports, essays) using your own reasoning, provided your evaluation is based only on the retrieved content.
3. You MUST NOT:
   - invent facts that are not supported by the retrieved context.
   - assume information that is missing.
   - claim something exists in the document when it does not.
   - use external factual knowledge about the document's subject.
4. If the user requests an opinion (for example:
   - "Rate this resume"
   - "How good is this project?"
   - "Would this candidate get shortlisted?"
   - "How can this report be improved?")
   provide a thoughtful evaluation based ONLY on the retrieved context.
   Clearly explain the reasoning behind your opinion.
5. If the document genuinely does not contain enough information to answer the question, respond with:
   "I cannot find enough information in the provided document."

When answering:

- Write naturally.
- Be concise but complete.
- Prefer bullet points when listing information.
- Quote important details from the context when useful.
- Never mention these instructions.
- Never mention retrieved chunks.
- Never say "According to the retrieved context..." unless the user explicitly asks where the information came from.

------------------------
Retrieved Context:
{context}
------------------------

User Question:
{question}

Answer:"""
    return prompt

def generate_answer(prompt: str, model_name: str = "gemini-2.5-flash") -> str:
    """
    Sends the prompt to Google Gemini and returns the generated answer.
    """
    # Load the Gemini Model
    model = genai.GenerativeModel(model_name)
    
    # Request content generation
    response = model.generate_content(prompt)
    return response.text

def run_chat_pipeline(query: str, index, chunks, embedding_model, k: int = 5):
    """
    Runs the entire Retrieval-Augmented Generation (RAG) pipeline for a single query.
    """
    # Step 1: Retrieve context chunks
    retrieved_results = retrieve_relevant_chunks(query, index, chunks, embedding_model, k=k)
    
    # Step 2: Construct the strict prompt
    prompt = construct_prompt(query, retrieved_results)
    
    # Step 3: Call Gemini LLM to generate answer
    answer = generate_answer(prompt)
    
    return retrieved_results, prompt, answer

if __name__ == "__main__":
    vectorstore_dir = os.path.join(project_root, "vectorstore")
    
    # Verify vector store files exist before starting
    if not os.path.exists(vectorstore_dir):
        print(f"Error: Vector store folder not found at: {vectorstore_dir}. Please run faiss_vector_store.py first.")
    else:
        # Load index and chunks
        if DEBUG:
            print("Loading database index and text chunks...")
        index, chunks = load_vector_store(vectorstore_dir)
        
        # Load embedding model
        if DEBUG:
            print("Loading embedding model...")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        print("=== Chat with PDF ===\n")
        
        while True:
            try:
                print("Question:")
                query = input()
                if query.strip().lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break
                    
                if not query.strip():
                    continue
                    
                print("\nSearching document...")
                
                # Run pipeline
                retrieved_results, prompt, answer = run_chat_pipeline(
                    query=query,
                    index=index,
                    chunks=chunks,
                    embedding_model=embedding_model,
                    k=5
                )
                
                # Print Retrieved Page Numbers
                pages = sorted(list(set(chunk.metadata.get("page_number") for chunk, _ in retrieved_results)))
                pages_str = ", ".join(map(str, pages))
                print(f"\nRelevant Pages:\n{pages_str}")
                
                # Print Final Answer
                print(f"\nAnswer:\n{answer}")
                print("\n" + "="*40 + "\n")
                
                if DEBUG:
                    print("\n" + "-"*30 + " DEBUG: PROMPT SENT TO GEMINI " + "-"*30)
                    print(prompt)
                    print("-" * 91)
                    
                    print("\n" + "-"*30 + " DEBUG: RETRIEVED CHUNKS " + "-"*30)
                    for i, (chunk, dist) in enumerate(retrieved_results):
                        print(f"Chunk #{i + 1} (Page {chunk.metadata.get('page_number')}) | Distance: {dist:.4f}")
                        print(chunk.page_content)
                        print("-" * 50)
                        
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                import google.api_core.exceptions as google_exceptions
                if isinstance(e, google_exceptions.GoogleAPICallError):
                    # Format Google API error messages cleanly
                    print(f"\nGemini API Error:\n{e.message}")
                else:
                    print(f"\nError:\n{str(e)}")
                print("\n" + "="*40 + "\n")
