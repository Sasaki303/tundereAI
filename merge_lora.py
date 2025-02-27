import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# モデルの設定
MODEL_NAME = "elyza/Llama-3-ELYZA-JP-8B"  # ベースモデル
LORA_PATH = "./lora-tundere-ai"  # LoRA の重み
OUTPUT_DIR = "./merged-tundere-ai"  # 統合モデルの保存先

# モデルと LoRA の読み込み
print("モデルをロード中...", flush=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,  # VRAM 節約
    device_map="cpu"  # すべての処理を CPU で実行
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("LoRA を統合中...", flush=True)
lora_model = PeftModel.from_pretrained(base_model, LORA_PATH)

# `merge_and_unload()` の実行
try:
    print("LoRA の統合を試行中...", flush=True)
    merged_model = lora_model.merge_and_unload()
    print("LoRA 統合成功！", flush=True)

    # 統合済みモデルを保存
    print(f"統合モデルを `{OUTPUT_DIR}` に保存中...", flush=True)
    merged_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("LoRA の統合が完了しました！フルモデルが保存されました。", flush=True)

except Exception as e:
    print(f"エラーが発生しました: {e}", flush=True)
