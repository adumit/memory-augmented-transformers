import datetime
import os

from tqdm import tqdm

import mlflow
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from torch.utils.data import DataLoader
from torch.optim import AdamW

from .memory_augmented_transformers.gpt2.augmented_model import GPT2WithMemory
from .memory_augmented_transformers.gpt2.lm_with_memory import GPT2LMHeadModelWithMemory
from .memory_augmented_transformers.gpt2.chunked_dataset import GPT2Dataset 
from .memory_augmented_transformers.train_and_validate import run_epoch, run_validation


device = "cuda"
# device = "cpu"
model_id = "distilgpt2"

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


def run_experiment(
    run_name,
    epochs=2,
    lr=5e-4,
    warmup_steps=1e2,
    epsilon=1e-8,
    batch_size=2,
    num_memories=3
):

    mlflow.end_run()
    mlflow.end_run()

    rand_state = 1
    torch.manual_seed(rand_state)

    mlflow.set_registry_uri("./mlruns")
    mlflow.set_experiment("Memory-augmented-transformer")
    mlflow.start_run(run_name=run_name)

    device = "cuda"
    # Currently only have gpt2 coded up and only the distilled version fits well on my GPU
    model_id = "distilgpt2"
    epochs = epochs
    learning_rate = lr
    warmup_steps = warmup_steps
    epsilon = epsilon
    batch_size = batch_size
    num_memories = num_memories

    mlflow.log_params({
        "epochs": epochs,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "epsilon": epsilon,
        "batch_size": batch_size,
        "num_memories": num_memories,
        "rand_state": rand_state
    })


    base_model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    memory_model = GPT2WithMemory(
        base_model.transformer, 
        memory_layer_inds=(3,), 
        num_mems_retrieved=num_memories, 
        only_use_mem_attn=True
    )

    model = GPT2LMHeadModelWithMemory(config=base_model.config, transformer_with_memory=memory_model)
    model = model.to(device)

    all_texts = []
    for fname in os.listdir("./data/PG-19/sample/"):
        if "train" in fname:
            continue
        if not "txt" in fname:
            continue
        with open(f"./data/PG-19/sample/{fname}", "r") as f:
            all_texts.append(f.read())

    dataset = GPT2Dataset(all_texts, tokenizer, max_length=768)
    print("Train dataset length:", len(dataset))

    valid_texts = []
    for fname in os.listdir("./data/PG-19/sample/"):
        if not "txt" in fname:
            continue
        if "train" in fname:
            with open(f"./data/PG-19/sample/{fname}", "r") as f:
                valid_texts.append(f.read())

    valid_dataset = GPT2Dataset(valid_texts, tokenizer, max_length=768)
    print("Valid dataset length:", len(valid_dataset))

    train_dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle=False  # Do not shuffle because we need to see the book in order
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False  # Do not shuffle because we need to see the book in order
    )

    model.train()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of trainable parameters:", params)

    optimizer = AdamW(
        model.parameters(),
        lr = learning_rate,
        eps = epsilon
    )

    with memory_model.knn_memories_context(batch_size = train_dataloader.batch_size) as knn_memories:
        for epoch in range(epochs):
            run_epoch(
                model=model, 
                optimizer=optimizer, 
                knn_memories=knn_memories, 
                train_dataloader=train_dataloader, 
                epoch_ind=epoch
            )
            run_validation(
                model=model,
                valid_dataloader=valid_dataloader,
                knn_memories=knn_memories,
                epoch=epoch
            )
    mlflow.end_run()
    mlflow.end_run()


if __name__ == "__main__":
    rand_state = 1
    torch.manual_seed(rand_state)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    run_experiment(
        "test-run"
    )

    