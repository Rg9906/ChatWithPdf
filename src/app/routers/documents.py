import os
import shutil
from fastapi import APIRouter, File, UploadFile, HTTPException, status, Depends
from pymongo.database import Database
from datetime import datetime
from bson import ObjectId
from typing import List, Dict

# Import local modules
from src.pdf_processor import extract_text_from_pdf
from src.app.database import get_db
from src.app.schemas.document import DocumentResponse

router = APIRouter(
    prefix="/documents",
    tags=["documents"]
)

# Directory to save uploaded PDFs
UPLOAD_DIR = os.path.join(os.getcwd(), "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload", response_model=DocumentResponse)
async def upload_pdf(file: UploadFile = File(...), db: Database = Depends(get_db)):
    """
    Uploads a PDF, saves it to disk, extracts its text, and stores its metadata in MongoDB.
    """
    # 1. Validate file type
    if not file.filename.endswith(".pdf") and file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed."
        )
    
    # Create safe file path
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        # 2. Save the uploaded file to disk asynchronously
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 3. Extract text from the saved PDF
        pages_data = extract_text_from_pdf(file_path)
        file_size = os.path.path.getsize(file_path) if hasattr(os.path, 'path') else os.path.getsize(file_path)
        
        # 4. Save metadata to MongoDB
        doc_metadata = {
            "filename": file.filename,
            "file_path": file_path,
            "file_size_bytes": file_size,
            "page_count": len(pages_data),
            "upload_timestamp": datetime.utcnow(),
            "status": "processed"  # Since text extraction is currently synchronous
        }
        
        # Insert into documents collection
        result = db.documents.insert_one(doc_metadata)
        
        # Add stringified id for the response model mapping
        doc_metadata["id"] = str(result.inserted_id)
        
        return doc_metadata
        
    except Exception as e:
        # Clean up file if anything failed
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing PDF: {str(e)}"
        )

@router.get("/", response_model=List[DocumentResponse])
async def list_documents(db: Database = Depends(get_db)):
    """
    Retrieves all documents stored in MongoDB.
    """
    documents = []
    # Find all documents in the collection
    for doc in db.documents.find():
        doc["id"] = str(doc["_id"])
        documents.append(doc)
    return documents
