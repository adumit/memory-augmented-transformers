from datasets import load_dataset
import random

from memory_augmented_transformers.dataset.gpt2_dataset import GPT2Dataset

def get_worldbank_project_document_dataset(tokenizer, batch_size, only_sample=False, sample_perc=.01, clean_texts=False, profile_sample=False):
    random.seed(1)
    dataset = load_dataset("lukesjordan/worldbank-project-documents")
    
    num_docs = len(dataset["train"])
    
    if only_sample:
        num_docs = int(num_docs * sample_perc)

    print(num_docs)
        
    all_ids = list(range(num_docs))
    
    valid_ids = set(random.choices(all_ids, k=int(.2*num_docs)))
    if num_docs < 50:
        print("Valid_ids:", valid_ids)
        print("Train_ids:", [i for i in all_ids if i not in valid_ids])
    
    train_ds = GPT2Dataset(txt_list=[dataset["train"][i]['document_text'] for i in all_ids if i not in valid_ids], 
                           tokenizer=tokenizer,
                           batch_size=batch_size,
                           clean_texts=clean_texts,
                           fill_in_padded_batches=True)
    valid_ds = GPT2Dataset(txt_list=[dataset["train"][i]['document_text'] for i in all_ids if i in valid_ids], 
                           tokenizer=tokenizer, 
                           batch_size=batch_size,
                           clean_texts=clean_texts)
    return train_ds, valid_ds