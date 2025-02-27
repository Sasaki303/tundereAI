from transformers import AutoModelForCausalLM, AutoTokenizer

# モデル名
model_name = "elyza/Llama-3-ELYZA-JP-8B"

# モデルとトークナイザーをローカルに保存
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("✅ モデルのダウンロードが完了しました！")
