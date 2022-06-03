from torch.utils.data import Dataset
import torch


class GPT2Dataset(Dataset):
    def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=768):

        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        for txt in txt_list:
            input_batches, attn_batches = self.chunk_inputs(txt + '<|endoftext|>', tokenizer, max_length)
            if len(input_batches) % 2 != 0:
                self.input_ids.extend(input_batches[1:])
                self.attn_masks.extend(attn_batches[1:])
            else:
                self.input_ids.extend(input_batches)
                self.attn_masks.extend(attn_batches)
    
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
        return inputs_batch_lst, attn_masks
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]
