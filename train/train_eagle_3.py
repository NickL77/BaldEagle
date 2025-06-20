import re
import glob
import json
import os
import torch
import wandb
import random
import dotenv

from safetensors import safe_open

from transformers.models.llama.configuration_llama import LlamaConfig

from transformers import AutoTokenizer, TrainingArguments

from modules.model.llama_eagle_3 import LlamaForCausalLMEagle3
from modules.data.data import list_local_files, list_hf_files, list_r2_files, \
    Eagle3LocalDataset, Eagle3HFDataset, Eagle3R2Dataset, \
    DataCollatorWithPadding, get_r2_fs

from modules.trainer.trainer_eagle_3 import EagleTrainer

dotenv.load_dotenv()

wandb.init(project="BaldEagle3-3090-tests")
wandb_run_name = wandb.run.name

path = "models/llama-8b/"

# -------------------------------- Load original Llama weights --------------------------------

with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
    index_json = json.loads(f.read())
    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
    lm_head_path = index_json["weight_map"]["lm_head.weight"]

with safe_open(os.path.join(path, emb_path), framework="pt", device="cpu") as f:
    tensor_slice = f.get_slice("model.embed_tokens.weight")
    vocab_size, hidden_dim = tensor_slice.get_shape()
    tensor = tensor_slice[:, :hidden_dim]

with safe_open(os.path.join(path, lm_head_path), framework="pt", device="cpu") as f:
    lm_head_weights = f.get_slice("lm_head.weight")[:, :]


# -------------------------------- Create draft model + tokenizer + head --------------------------------

tokenizer = AutoTokenizer.from_pretrained(path)
tokenizer.pad_token = tokenizer.eos_token

model_args = LlamaConfig(vocab_size=vocab_size,
                         hidden_size=hidden_dim,
                         intermediate_size=14336,
                         num_hidden_layers=1,
                         bos_token_id=128000,
                         eos_token_id=[128001, 128008, 128009],
                         num_key_value_heads=8,
                         num_attention_heads=32,
                         tie_word_embeddings=False,
                         draft_vocab_size=vocab_size)

draft_model = LlamaForCausalLMEagle3(model_args)
draft_model.load_embedding_weights(tensor)
for param in draft_model.model.embed_tokens.parameters():
    param.requires_grad = False 

draft_model.to("cuda:0")

# Load head
head = torch.nn.Linear(model_args.hidden_size, model_args.vocab_size, bias=False)
with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
    index_json = json.loads(f.read())

    head_path = index_json["weight_map"]["lm_head.weight"]
with safe_open(os.path.join(path, head_path),
                framework="pt",
                device="cpu") as f:
    tensor_slice = f.get_slice("lm_head.weight")
    vocab_size, hidden_dim = tensor_slice.get_shape()
    tensor = tensor_slice[:, :hidden_dim].float()

head.weight.data = tensor
head.to("cuda:0")
head.eval()
for param in head.parameters():
    param.requires_grad = False

# -------------------------------- Load data --------------------------------

# sharegpt_datapaths = list_hf_files("datasets/NickL77/llama8b-eagle-sharegpt/")
# sharegpt_datapaths = list_local_files("/mnt/ssd4tb/sharegpt_grouped_5k/")

# sharegpt_datapaths = list_local_files("/mnt/hdd16tb/llama31_8b_eagle_3_sharegpt_0_67999_mufp16/")
# ultra_chat_datapaths = list_local_files("/mnt/hdd16tb/llama31_8b_eagle_3_ultrachat_0_199999_mufp16/")

# combined_data_paths = sharegpt_datapaths[:-50] + ultra_chat_datapaths[:-50]
# random.Random(42).shuffle(combined_data_paths)
# eval_data_paths = sharegpt_datapaths[-50:] + ultra_chat_datapaths[-50:]

# eagle_train_dataset = Eagle3LocalDataset(combined_data_paths, max_len=1024)
# eagle_test_dataset = Eagle3LocalDataset(eval_data_paths, max_len=1024)

# eagle_collator = DataCollatorWithPadding()

sharegpt_datapaths = list_hf_files("datasets/NickL77/Llama3.1-8B-BaldEagle3-ShareGPT")
ultra_chat_datapaths = list_hf_files("datasets/NickL77/Llama3.1-8B-BaldEagle3-Ultrachat")

sharegpt_datapaths, sharegpt_datapaths_eval = sharegpt_datapaths[:-50], sharegpt_datapaths[-50:]
ultra_chat_datapaths, ultra_chat_datapaths_eval = ultra_chat_datapaths[:-50], ultra_chat_datapaths[-50:]
combined_data_paths = sharegpt_datapaths + ultra_chat_datapaths
combined_data_paths_eval = sharegpt_datapaths_eval + ultra_chat_datapaths_eval
random.Random(42).shuffle(combined_data_paths)

eagle_train_dataset = Eagle3HFDataset(combined_data_paths, max_len=1024)
eagle_test_dataset = Eagle3HFDataset(combined_data_paths_eval, max_len=1024)
eagle_collator = DataCollatorWithPadding()

# -------------------------------- Train --------------------------------

training_args = TrainingArguments(
    output_dir=f"./hf_trainer_output_dir/{wandb_run_name}/",

    num_train_epochs=10,
    
    # More memory saving
    gradient_checkpointing=False, # TODO: check why eagle 3 doesn't support gradient checkpointing
    optim="adamw_8bit", 

    # For on H100
    gradient_accumulation_steps=4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    dataloader_num_workers=32,
    dataloader_prefetch_factor=7,
    logging_steps=128,
    eval_steps=256,

    # For on 3090
    # gradient_accumulation_steps=16,
    # per_device_train_batch_size=1,
    # per_device_eval_batch_size=4,
    # dataloader_num_workers=8,
    # dataloader_prefetch_factor=2,
    # logging_steps=16,
    # eval_steps=16,

    remove_unused_columns=False,
    bf16=True,
    fp16=False,
    
    warmup_ratio=0.01,
    learning_rate=1e-4, # 1e-3
    lr_scheduler_type="constant",  # Placeholder, we override it in the trainer

    max_grad_norm=1.5, # 1
    adam_beta1=0.9, # 0.9
    adam_beta2=0.95, # 0.999
    weight_decay=1e-4,

    eval_strategy="steps",

    save_strategy="steps",
    save_steps=0.02, # saves every 2% of training
    save_total_limit=10,
)

trainer = EagleTrainer(
    model=draft_model,
    head=head,
    args=training_args,
    train_dataset=eagle_train_dataset,
    eval_dataset=eagle_test_dataset,
    data_collator=eagle_collator,

    num_shifts=3, # Number of training time test steps to make
    min_lr_ratio=0.5,  # Pass as a separate argument
)

trainer.train()
# trainer.train(resume_from_checkpoint=True)