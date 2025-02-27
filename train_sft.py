import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

#VRAM不足で実行できない可能性があります。
# VRAM をリセット
torch.cuda.empty_cache()

# GPU の認識ログを出力
print("🚀 PyTorch バージョン:", torch.__version__)
print("🔍 CUDA 利用可能:", torch.cuda.is_available())

if torch.cuda.is_available():
    device = "cuda"
    print("使用中の GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("使用可能な VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GB")
else:
    device = "cpu"
    print("CUDA が使用できません。CPU で実行します。")

# モデルとトークナイザーの読み込み
model_name = "elyza/Llama-3-ELYZA-JP-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # エラー修正

# モデルの読み込み（VRAM節約）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16  # FP16 → BF16 に変更
).to(device)

# メモリ節約のためチェックポイントを有効化
model.gradient_checkpointing_enable()

# データセットの読み込み（JSONL形式）
dataset = load_dataset("json", data_files={"train": "data.jsonl"})

# データの前処理（トークナイズ）
def preprocess_function(examples):
    inputs = ["ツンデレなキャラクターとして以下の質問に答えてください。\n\nQ: " + q + "\nA:" for q in examples["input"]]
    outputs = [a for a in examples["output"]]

    # トークナイズ
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(outputs, padding="max_length", truncation=True, max_length=512)

    # `labels` の追加（教師データとして使う）
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# データのトークナイズ
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 学習設定（VRAM節約）
training_args = TrainingArguments(
    output_dir="./sft-model",
    eval_strategy="no",
    save_strategy="epoch",
    per_device_train_batch_size=2,  # バッチサイズを小さくする
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,  # メモリ節約のため累積勾配を使用
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    bf16=True  # `fp16` ではなく `bf16` に変更（VRAM節約）
)

# Trainer の設定
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

# VRAM情報を出力
if device == "cuda":
    print("🔍 学習前の VRAM 使用量:", round(torch.cuda.memory_allocated() / 1024**3, 2), "GB")

# 学習の実行
trainer.train()

# 学習後の VRAM情報を出力
if device == "cuda":
    print("学習後の VRAM 使用量:", round(torch.cuda.memory_allocated() / 1024**3, 2), "GB")

# 学習済みモデルの保存
model.save_pretrained("./sft-model")
tokenizer.save_pretrained("./sft-model")
