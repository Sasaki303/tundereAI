import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# 環境変数から設定値を取得
CHROMA_PATH = os.getenv('CHROMA_PATH', 'chroma')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'local-rag')
TEXT_EMBEDDING_MODEL = os.getenv('TEXT_EMBEDDING_MODEL', 'nomic-embed-text')

print(f"CHROMA_PATH: {CHROMA_PATH}")
print(f"COLLECTION_NAME: {COLLECTION_NAME}")
print(f"TEXT_EMBEDDING_MODEL: {TEXT_EMBEDDING_MODEL}")

# ベクトルデータベースを取得する関数
def get_vector_db():
    try:
        print("Initializing Ollama Embeddings...")
        embedding = OllamaEmbeddings(model=TEXT_EMBEDDING_MODEL)
        print("Embedding model initialized.")

        print("Initializing ChromaDB...")
        db = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_PATH,
            embedding_function=embedding
        )
        print("ChromaDB initialized successfully.")
        
        return db
    except Exception as e:
        print("Error in get_vector_db:", str(e))
        return None
