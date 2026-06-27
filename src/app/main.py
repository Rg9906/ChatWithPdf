from fastapi import FastAPI
from src.app.routers import documents

app = FastAPI(
    title="ChatWithPdf API",
    description="Backend API for Retrieval-Augmented Generation (RAG) over PDF documents",
    version="1.0.0"
)

# Register routers
app.include_router(documents.router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the ChatWithPdf RAG API!",
        "status": "healthy",
        "docs_url": "/docs"
    }
