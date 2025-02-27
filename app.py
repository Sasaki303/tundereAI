import os
import json
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from embed import embed  
from query import query
from get_vector_db import get_vector_db
import time  # ⏳ 処理時間計測用

# 環境変数のロード
load_dotenv()

# 一時フォルダの設定
TEMP_FOLDER = os.getenv('TEMP_FOLDER', './_temp')
CHROMA_PATH = os.getenv('CHROMA_PATH', 'chroma')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'local-rag')
LLM_MODEL = os.getenv('LLM_MODEL', 'tundere-ai:Q4_K_M')  # ✅ LLM を修正
TEXT_EMBEDDING_MODEL = os.getenv('TEXT_EMBEDDING_MODEL', 'nomic-embed-text')
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Flaskアプリの作成
app = Flask(__name__)

@app.route('/embed', methods=['POST'])
def route_embed():
    if 'file' not in request.files:
        return jsonify({"error": "ファイルが見つかりません"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "選択されたファイルがありません"}), 400

    try:
        start_time = time.time()  # ⏳ 計測開始
        embedded = embed(file)
        elapsed_time = time.time() - start_time  # ⏳ 計測終了
        print(f"⏳ [ログ] エンベディング処理が完了しました（処理時間: {elapsed_time:.2f} 秒）")  # ⏳ ログ出力

        if embedded:
            return jsonify({"message": "PDF のエンベディングが完了しました"}), 200
        return jsonify({"error": "エンベディングに失敗しました"}), 400
    except Exception as e:
        print(f"⚠️ [エラー] エンベディング処理中にエラーが発生しました: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def route_query():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "質問が見つかりません"}), 400

        start_time = time.time()  # ⏳ 計測開始
        response = query(data['query'])
        elapsed_time = time.time() - start_time  # ⏳ 計測終了
        print(f"⏳ [ログ] クエリ処理が完了しました（処理時間: {elapsed_time:.2f} 秒）")  # ⏳ ログ出力

        if response:
            response_json = json.dumps({"message": response}, ensure_ascii=False)  # ✅ 日本語をそのまま表示
            print(f"📝 [ログ] AI の回答: {response}")  # ✅ AI の返答内容をログに出力
            return response_json, 200, {"Content-Type": "application/json; charset=utf-8"}  # ✅ UTF-8 エンコード指定
        return jsonify({"error": "クエリ処理に失敗しました"}), 400
    except Exception as e:
        print(f"⚠️ [エラー] クエリ処理中にエラーが発生しました: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 [ログ] Flask サーバーを起動しました！")
    print(f"📂 [ログ] データ保存フォルダ: {CHROMA_PATH}")
    print(f"📄 [ログ] コレクション名: {COLLECTION_NAME}")
    print(f"🤖 [ログ] 使用する LLM モデル: {LLM_MODEL}")
    
    app.run(host="0.0.0.0", port=8080, debug=True)
