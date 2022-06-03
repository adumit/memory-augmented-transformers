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


device = "cuda"
# device = "cpu"
model_id = "distilgpt2"

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


def run_epoch(model, optimizer, knn_memories, train_dataloader, epoch_ind):

    total_train_loss = 0.

    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)

        model.zero_grad()        

        outputs = model(
            b_input_ids,
            labels=b_labels, 
            attention_mask = b_masks,
            token_type_ids=None,
            knn_memories=knn_memories
        )

        loss = outputs[0] 

        batch_loss = loss.item()
        total_train_loss += batch_loss
        mlflow.log_metric("Train loss", loss.detach().cpu().data, step=step)

        loss.backward()

        optimizer.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)
    mlflow.log_metric("Avg epoch loss", avg_train_loss, step=step+(epoch_ind * len(train_dataloader)))


def run_validation(model, valid_dataloader, knn_memories, epoch=-1):
    print("Running Validation...")

    model.eval()

    total_eval_loss = 0

    # Evaluate data for one epoch
    for batch in valid_dataloader:

        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)

        with torch.no_grad():        

            outputs  = model(
                b_input_ids, 
#                            token_type_ids=None, 
                attention_mask = b_masks,
                labels=b_labels,
                knn_memories=knn_memories)

            loss = outputs[0]  

        batch_loss = loss.item()
        total_eval_loss += batch_loss        

    avg_val_loss = total_eval_loss / len(valid_dataloader)

    mlflow.log_metric("Validation loss", avg_val_loss, step=epoch+1)

    return


if __name__ == "__main__":
    rand_state = 1
    torch.manual_seed(rand_state)

    mlflow.set_registry_uri("./mlruns")
    mlflow.set_experiment("Memory-augmented-transformers")

    epochs = 2
    learning_rate = 5e-4
    warmup_steps = 1e2
    epsilon = 1e-8
    batch_size = 2
    num_memories = 3

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
        batch_size=batch_size,
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


    