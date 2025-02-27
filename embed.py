import os
from datetime import datetime
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from get_vector_db import get_vector_db

# 環境変数の読み込み
TEMP_FOLDER = os.getenv('TEMP_FOLDER', './_temp')
os.makedirs(TEMP_FOLDER, exist_ok=True)

# アップロードファイルの拡張子チェック
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf'}

# ファイルを保存する関数
def save_file(file):
    ct = datetime.now()
    ts = ct.timestamp()
    filename = str(ts) + "_" + secure_filename(file.filename)
    file_path = os.path.join(TEMP_FOLDER, filename)
    file.save(file_path)
    return file_path

# PDFデータをロード＆分割する関数
def load_and_split_data(file_path):
    loader = UnstructuredPDFLoader(file_path=file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    return chunks

# メイン処理
def embed(file):
    if file.filename != '' and file and allowed_file(file.filename):
        file_path = save_file(file)
        chunks = load_and_split_data(file_path)
        
        # ベクトルDBに追加
        db = get_vector_db()
        db.add_documents(chunks)
        # db.persist()

        # 一時ファイル削除
        os.remove(file_path)
        return True
    return False
