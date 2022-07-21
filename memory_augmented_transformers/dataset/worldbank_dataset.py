from datasets import load_dataset
import random

from memory_augmented_transformers.dataset.gpt2_dataset import GPT2Dataset

def get_worldbank_project_document_dataset(tokenizer, batch_size, sample_perc=.01):
    random.seed(1)
    dataset = load_dataset("lukesjordan/worldbank-project-documents")
    
    num_docs = int(len(dataset["train"]) * sample_perc)
        
    all_ids = list(range(num_docs))
    
    valid_ids = set(random.choices(all_ids, k=int(.2*num_docs)))
    
    train_ds = GPT2Dataset(txt_list=[dataset["train"][i]['document_text'] for i in all_ids if i not in valid_ids], 
                           tokenizer=tokenizer, 
                           batch_size=batch_size)
    valid_ds = GPT2Dataset(txt_list=[dataset["train"][i]['document_text'] for i in all_ids if i in valid_ids], 
                           tokenizer=tokenizer, 
                           batch_size=batch_size)
    return train_ds, valid_ds
