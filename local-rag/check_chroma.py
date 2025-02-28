from get_vector_db import get_vector_db

# ChromaDB に接続
db = get_vector_db()

# 保存されているドキュメントを取得
docs = db.get()
print(f"📂 ChromaDB に保存されているドキュメント数: {len(docs['documents'])}")

# 先頭のデータを表示
if docs["documents"]:
    for i, doc in enumerate(docs["documents"][:5]):  # 最初の5つだけ表示
        print(f"\n📄 ドキュメント {i+1}:")
        print(doc)
else:
    print("⚠️ ChromaDB にデータがありません！")