import os
import torch
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# `.env` の読み込み
load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME", "elyza/Llama-3-ELYZA-JP-8B")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# デバイスの確
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 使用デバイス: {device}")

# モデルとトークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HUGGINGFACE_TOKEN)
tokenizer.pad_token = tokenizer.eos_token  # `pad_token` を `eos_token` に設定

# `to_empty()` を適用して `meta` デバイスから GPU へ移動
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HUGGINGFACE_TOKEN,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.to_empty(device=device)

# LoRA 設定（強化版）
lora_config = LoraConfig(
    r=8,                 # 学習可能な行列の次元数 (r=8で揃える)
    lora_alpha=32,        # LoRA のスケーリング係数
    lora_dropout=0.1,     # 汎化性能向上のためのドロップアウト
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "down_proj", "up_proj"] # FFN層もLoRA適用
)

# LoRA の適用
model = get_peft_model(model, lora_config)
model.train()

# データセットの読み込み
dataset = load_dataset("json", data_files={"train": "tsundere_responses.jsonl"})["train"]

# ユーザーとツンデレAIの自然な会話プロンプト
def generate_prompt(data_point):
    return f"ユーザー: {data_point['input']}\nツンデレAI: {data_point['output']}"

dataset = dataset.map(lambda x: {"text": generate_prompt(x)})

# 学習パラメータ
training_arguments = TrainingArguments(
    output_dir="./train_logs",
    per_device_train_batch_size=6,  # バッチサイズを増やして安定化
    gradient_accumulation_steps=2,
    optim="adamw_torch",
    learning_rate=5e-5,  # 低めの学習率で安定
    max_steps=300,  # しっかり学習
    logging_steps=50,
    save_steps=100,
    fp16=True,
    report_to="none"
)

# LoRA 学習の設定
trainer = SFTTrainer(
    model=model,  
    tokenizer=tokenizer,  
    train_dataset=dataset,  
    dataset_text_field="text",
    peft_config=lora_config,  
    args=training_arguments,  
    max_seq_length=1024,
    packing=False
)

# 学習開始
trainer.train()

# 学習済みのモデルを保存
trainer.model.save_pretrained("./lora-tundere-ai")
tokenizer.save_pretrained("./lora-tundere-ai")

print("LoRA の重みを `./lora-tundere-ai` に保存しました。")
