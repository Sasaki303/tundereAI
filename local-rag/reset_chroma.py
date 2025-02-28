from get_vector_db import get_vector_db

db = get_vector_db()
db.delete_collection()  # ChromaDB のデータを削除
print("🚮 ChromaDB のデータをすべて削除しました！")
