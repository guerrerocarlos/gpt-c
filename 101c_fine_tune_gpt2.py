# fine_tune_gpt2.py
import os
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer

# 1) Load a toy dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_texts = dataset['train']['text'][:1000]       # toy subset for quick runs
val_texts   = dataset['validation']['text'][:100]   # toy subset

# 2) Load a GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT2 doesn't have a native PAD token

# 3) Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length"
    )

train_dataset = dataset["train"].select(range(1000)).map(tokenize_function, batched=True)
val_dataset   = dataset["validation"].select(range(100)).map(tokenize_function, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask"])

# 4) Load GPT-2 model (PyTorch by default)
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 5) Training arguments
training_args = TrainingArguments(
    output_dir="my-gpt2-toy",
    num_train_epochs=1,                      # toy example, do more for better results
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10
)

# 6) Data collator
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # GPT-2 is a causal (left-to-right) LM, not masked
)

# 7) Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

# 8) Fine-tune!
trainer.train()

# 9) Save
trainer.save_model("my-gpt2-toy")
tokenizer.save_pretrained("my-gpt2-toy")