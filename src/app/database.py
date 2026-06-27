from pymongo import MongoClient
from src.app.config import settings

# Initialize PyMongo client
# In production, you'd typically manage pool size, connection timeouts, etc.
client = MongoClient(settings.MONGO_URI)

def get_db():
    """
    Dependency function to get the MongoDB database instance.
    FastAPI's dependency injection will call this function.
    """
    db = client[settings.DB_NAME]
    try:
        yield db
    finally:
        # If we need to perform any cleanup per request, we do it here.
        # For PyMongo, we don't need to close the client after each request
        # as it handles connection pooling automatically.
        pass
