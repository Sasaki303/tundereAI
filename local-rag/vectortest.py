from get_vector_db import get_vector_db

try:
    db = get_vector_db()
    print("ChromaDB Object:", db)
    print("Collection Name:", db._collection.name if db._collection else "No collection")
    print("Persist Directory:", db._persist_directory)
except Exception as e:
    print("Error:", str(e))
