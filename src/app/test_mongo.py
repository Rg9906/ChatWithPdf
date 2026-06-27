import sys
from pymongo import MongoClient
from src.app.config import settings

def test_mongodb_connection():
    print("Connecting to MongoDB...")
    print(f"URI: {settings.MONGO_URI}")
    print(f"Database: {settings.DB_NAME}")
    
    try:
        # Initialize client
        client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=5000)
        
        # 1. Ping the server
        client.admin.command('ping')
        print("[SUCCESS] Successfully pinged MongoDB!")
        
        # 2. Test write/read operations
        db = client[settings.DB_NAME]
        test_collection = db["connection_test"]
        
        test_doc = {"test": "connection", "timestamp": "now"}
        insert_result = test_collection.insert_one(test_doc)
        print(f"[SUCCESS] Successfully inserted document with ID: {insert_result.inserted_id}")
        
        found_doc = test_collection.find_one({"_id": insert_result.inserted_id})
        print(f"[SUCCESS] Successfully retrieved document: {found_doc}")
        
        # Clean up
        test_collection.delete_one({"_id": insert_result.inserted_id})
        print("[SUCCESS] Cleanup successful!")
        print("\n=== MongoDB configuration is correct and ready for the backend! ===")
        
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_mongodb_connection()
