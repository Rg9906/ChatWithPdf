import os
from dotenv import load_dotenv

# Load environment variables
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=env_path)

class Settings:
    PROJECT_NAME: str = "ChatWithPdf API"
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    DB_NAME: str = os.getenv("DB_NAME", "chat_with_pdf_db")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")

settings = Settings()
