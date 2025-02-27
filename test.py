import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_text(model_name, prompt):
    print(f"モデル `{model_name}` で推論中...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt = "ツンデレAIとは"

print("ベースモデルの出力:")
print(generate_text("elyza/Llama-3-ELYZA-JP-8B", prompt))

print("\n統合済みモデルの出力:")
print(generate_text("./merged-tundere-ai", prompt))
