from transformers import PreTrainedTokenizerFast

# tokenizer.json のパス
json_path = "C:/Users/sasak/tundereAI/merged-tundere-ai/tokenizer.json"
# tokenizer.model を保存するパス
model_path = "C:/Users/sasak/tundereAI/merged-tundere-ai/tokenizer.model"

# トークナイザーをロード
tokenizer = PreTrainedTokenizerFast(tokenizer_file=json_path)

# tokenizer.model を保存
tokenizer.save_pretrained("C:/Users/sasak/tundereAI/merged-tundere-ai")

print(f"tokenizer.model を {model_path} に保存しました！")
