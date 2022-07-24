from tqdm import tqdm
from collections import defaultdict

from torch.utils.data import Dataset
import torch


class GPT2Dataset(Dataset):
    def __init__(self, txt_list, tokenizer, batch_size, gpt2_type="gpt2", max_length=768):

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.input_ids = []
        self.attn_masks = []
        self.item_index_to_document_index = {}

        for doc_ind, txt in tqdm(enumerate(txt_list), desc="dataset_loader", total=len(txt_list)):
            input_batches, attn_batches = self.chunk_inputs(txt + '<|endoftext|>', tokenizer, max_length)
            input_batch_length = len(input_batches)
            current_index = len(self.input_ids)
            self.input_ids.extend(input_batches)
            self.attn_masks.extend(attn_batches)
            for i in range(input_batch_length):
                self.item_index_to_document_index[current_index+i] = doc_ind
        
        #####
        # The following logic handles grouping the documents into batches
        # where we only serially process each document in subsequent batches
        #####
        document_to_item_index = defaultdict(list)
        for item_ind, doc_ind in self.item_index_to_document_index.items():
            document_to_item_index[doc_ind].append(item_ind)
        
        self.batch_index_to_item_indexes = defaultdict(lambda: [-1 for _ in range(batch_size)])
        self.batch_index_to_document_indexes = defaultdict(set)
        self.document_to_batch_placement = {}
        filled_batches = {-1}
        current_batch_index = 0
        current_doc_index = 0
        for item_ind, doc_ind in sorted(self.item_index_to_document_index.items(), key=lambda x: x[1]):
            if doc_ind != current_doc_index:
                current_doc_index = doc_ind
                current_batch_index = max(filled_batches) + 1
            placement_found = False
            while not placement_found:
                if (-1 in self.batch_index_to_item_indexes[current_batch_index] and 
                        doc_ind not in self.batch_index_to_document_indexes[current_batch_index]):
                    placement_found = True
                else:
                    current_batch_index += 1
            # Set the document index the first time we're adding a document to a
            # batch. This ensures that the document is always at the same index
            # in the batch dimension
            if doc_ind not in self.document_to_batch_placement:
              batch_placement = [i for i,x in enumerate(self.batch_index_to_item_indexes[current_batch_index]) if x == -1][0]
              self.document_to_batch_placement[doc_ind] = batch_placement
            batch_placement = self.document_to_batch_placement[doc_ind]
            self.batch_index_to_item_indexes[current_batch_index][batch_placement] = item_ind
            self.batch_index_to_document_indexes[current_batch_index].add(doc_ind)

        self.item_index_iterator = []
        for batch_index, item_indexes in sorted(self.batch_index_to_item_indexes.items(), key=lambda x: x[0]):
            # # Just skipping over batches that don't have enough data. Ideally, we'd actually make use
            # # of this data, but it should be a small portion of the overall dataset anyways.
            # if len(self.batch_index_to_document_indexes[batch_index]) < self.batch_size:
            #     continue
            for item_index in item_indexes:
                self.item_index_iterator.append(item_index)
                            
    
    def chunk_inputs(self, long_text, tokenizer, max_length_of_seq):
        inputs_no_trunc = tokenizer(long_text, max_length=None, return_tensors='pt', truncation=False)

        chunk_start = 0
        chunk_end = max_length_of_seq
        inputs_batch_lst = []
        attn_masks = []
        while chunk_start <= len(inputs_no_trunc['input_ids'][0]):
            inputs_batch = inputs_no_trunc['input_ids'][0][chunk_start:chunk_end]
            attn_mask_batch = inputs_no_trunc['attention_mask'][0][chunk_start:chunk_end]
            if len(inputs_batch) < max_length_of_seq:
                num_tokens_to_add = max(max_length_of_seq - len(inputs_batch), 0)
                inputs_batch = torch.cat([inputs_batch, torch.tensor([self.tokenizer.unk_token_id] * num_tokens_to_add)])
                attn_mask_batch = torch.cat([attn_mask_batch, torch.tensor([0] * num_tokens_to_add)])
            inputs_batch = torch.unsqueeze(inputs_batch, 0)
            inputs_batch_lst.append(inputs_batch)
            attn_masks.append(attn_mask_batch)
            chunk_start += max_length_of_seq
            chunk_end += max_length_of_seq
        assert all(in_batch.shape[1] == max_length_of_seq for in_batch in inputs_batch_lst)
        return inputs_batch_lst, attn_masks
    
    def __len__(self):
        return len(self.item_index_iterator)

    def __getitem__(self, idx):
        ind_to_get = self.item_index_iterator[idx]
        # If the index in -1, that means we are operating on a part of a batch
        # that we don't have a document to fill in for. For now, just pass
        # a item_mask of 0, which we'll multiply the loss by
        if ind_to_get == -1:
          item_mask = torch.tensor(0)
        else:
          item_mask = torch.tensor(1) 
        return self.input_ids[ind_to_get], self.attn_masks[ind_to_get], item_mask