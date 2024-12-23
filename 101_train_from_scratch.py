# train_gpt2_from_scratch.py
import os
from pathlib import Path
from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# 1) Load your custom text dataset
#    If you have a local .txt, we can wrap it in a dataset using "text" features
dataset = load_dataset('text', data_files={'train': 'small_dataset.txt'})

# 2) Create a GPT-2 config from scratch
config = GPT2Config(
    vocab_size=50257,       # standard GPT-2 vocab size
    n_positions=1024,       # max sequence length
    n_ctx=1024, 
    n_embd=768,             # hidden size
    n_layer=12,             # number of layers
    n_head=12,              # number of attention heads
    # you can tweak dropout, activation function, etc. if you wish
)

# Bonus: Making GPT-2 “Tiny” For Demonstration
# config = GPT2Config(
#     n_embd=128,   # smaller hidden size
#     n_layer=4,    # fewer layers
#     n_head=4,     # fewer heads
#     vocab_size=50257,
#     n_positions=256
# )

# 3) Create a GPT2LMHeadModel with random initialization
model = GPT2LMHeadModel(config)

# 4) Create or load a GPT-2 tokenizer
#    - Even if we do "from_pretrained('gpt2')", that only gives us the *tokenizer merges + vocab*, not the model weights.
#    - If you want a *completely* scratch tokenizer, you'd have to train your own merges. 
#      But typically, we reuse GPT-2 merges. It's up to you.
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # reusing merges + vocab
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a real pad token

# 5) Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=128,
        # or 512, 1024, depending on your dataset
        padding="max_length"  # to keep it simple
    )

tokenized_dataset = dataset['train'].map(tokenize_function, batched=True)
tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask'])

# 6) Prepare for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT-2 is a causal LM, not masked
)

# 7) Split into train/val (tiny in this example)
train_val = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_val['train']
val_dataset   = train_val['test']

# 8) Training arguments
training_args = TrainingArguments(
    output_dir='gpt2-scratch',
    overwrite_output_dir=True,
    num_train_epochs=1,               # very small for a toy example
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_steps=10,
    learning_rate=1e-4,
    push_to_hub=False,
)

# 9) Create a Trainer
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

# 10) Train
trainer.train()

# 11) Save the model + tokenizer
trainer.save_model("gpt2-scratch")
tokenizer.save_pretrained("gpt2-scratch")