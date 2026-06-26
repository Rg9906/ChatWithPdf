# 📄 Chat with PDF (RAG)

Ever wished you could just throw a PDF at an AI and ask it questions instead of scrolling through 100 pages looking for that *one* paragraph?

Yeah, same.

This project is my attempt at building exactly that while actually understanding **how Retrieval-Augmented Generation (RAG)** works instead of hiding everything behind a single library call.

The entire pipeline—from reading PDFs to semantic search and finally generating answers—was built step by step with the goal of learning what happens under the hood.

---

## ✨ What it does

* 📄 Reads PDF documents
* ✂️ Cleans and chunks the extracted text
* 🧠 Converts every chunk into semantic embeddings
* 📦 Stores embeddings inside a FAISS vector database
* 🔍 Finds the most relevant chunks for a user's question
* 🤖 Uses Google Gemini to answer using only the retrieved document context
* 🚫 Prevents hallucinations by grounding responses in the PDF

Instead of searching for exact keywords, the application performs **semantic search**, meaning it understands *meaning*, not just matching words.

So asking:

> "What experience does the candidate have?"

will still retrieve the right section even if the resume never literally says "experience."

---

# 🏗️ Project Pipeline

The application follows a pretty standard RAG pipeline:

```text
PDF
 │
 ▼
Extract Text
 │
 ▼
Clean Text
 │
 ▼
Chunk Text
 │
 ▼
Generate Embeddings
 │
 ▼
Store in FAISS
 │
 ▼
User Question
 │
 ▼
Generate Query Embedding
 │
 ▼
Similarity Search
 │
 ▼
Retrieve Relevant Chunks
 │
 ▼
Gemini
 │
 ▼
Answer
```

Every stage lives in its own module so the project stays easy to understand and extend.

---

# 📁 Project Structure

```text
ChatWithPdf/
│
├── data/                  # Raw PDF files used for processing
├── vectorstore/           # Directory where the FAISS index and chunk metadata are saved
│   ├── index.faiss        # Stored FAISS vector index
│   └── chunks.pkl         # Pickled text chunks list matching index IDs
│
├── src/                   # Python source modules
│   ├── pdf_processor.py   # PDF text extraction
│   ├── chunking.py        # Clean text & chunking logic
│   ├── embedding_generation.py # Sentence Transformers embedding logic
│   ├── faiss_vector_store.py   # FAISS Index building and serialization
│   ├── retrieval.py       # Query embedding and vector search
│   └── chat.py            # Grounded prompt construction & Gemini synthesis (CLI app)
│
├── .env                   # Configuration file containing API credentials (GEMINI_API_KEY)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

# ⚙️ Tech Stack

* Python
* Google Gemini API (using the official Python SDK)
* Sentence Transformers (`all-MiniLM-L6-v2`)
* FAISS (`faiss-cpu` for vector similarity search)
* LangChain (`langchain-text-splitters` for chunking text)
* PyPDF (for parsing and extraction)
* NumPy

---

# 🧠 Why FAISS?

Imagine having thousands of document chunks.

Searching every single one every time someone asks a question would be painfully slow.

Instead, every chunk is converted into a vector (embedding) and stored inside FAISS.

When a question comes in:

1. The question is embedded.
2. FAISS finds the closest vectors.
3. Those chunks become the context for Gemini.

This makes retrieval extremely fast while also understanding semantic similarity instead of exact keywords.

---

# 🧩 Why chunking?

LLMs can't read entire books every time you ask a question.

Instead, the document is broken into overlapping chunks.

The overlap helps preserve context between neighbouring sections and improves retrieval quality.

---

# 💬 Example

```text
Question

"What experience does the candidate have?"

↓

Relevant Pages

1

↓

Answer

• Undergraduate Research Assistant
• Information Technology Support Specialist
• Artificial Intelligence Research Assistant
```

---

# 🎯 Goal of this project

The goal wasn't simply to make a chatbot.

It was to understand the complete Retrieval-Augmented Generation pipeline by implementing each stage individually.

Rather than relying on high-level abstractions, this project intentionally exposes the moving pieces:

* PDF parsing
* Text preprocessing
* Chunking
* Embeddings
* Vector databases
* Similarity search
* Prompt construction
* LLM integration

Building each component separately made the overall architecture much easier to understand.

---

# 🚀 Future Improvements

There are plenty of ideas I'd love to add:

* 🌐 FastAPI backend
* ⚛️ React frontend
* 📤 Drag & drop PDF upload
* 📚 Page citations in responses
* 💬 Conversation memory
* 📄 Multiple PDF support
* ⚡ Streaming responses
* 🔄 Hybrid Search (BM25 + Vector Search)
* 📈 Reranking
* 🐳 Docker deployment

---

# 📚 What I learned

This project ended up teaching me a lot more than I expected.

Some of the topics I got hands-on experience with:

* Retrieval-Augmented Generation (RAG)
* Semantic search
* Sentence embeddings
* Vector databases
* FAISS indexing
* Prompt engineering
* LLM APIs
* Document preprocessing
* Information retrieval pipelines

It also gave me a much better appreciation for what's actually happening when modern AI assistants answer questions about documents.

Turns out there's quite a bit going on before the LLM even sees your prompt.

---

# 🤝 Contributing

Found a bug?

Have an idea?

Want to make the retrieval smarter?

Feel free to open an issue or submit a pull request. I'm always happy to learn from better approaches.

---

## Thanks for stopping by!

If you found this project interesting (or if it saved you from manually scrolling through a 200-page PDF 😭), consider giving it a ⭐.

Happy building!
