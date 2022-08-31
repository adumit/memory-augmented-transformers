"""

Largely copied over from the memorizing-transformers-pytorch 
repo with modifications by me.

Source:
https://github.com/lucidrains/memorizing-transformers-pytorch

"""

import torch
import faiss
import numpy as np
from pathlib import Path
from functools import wraps

from contextlib import ExitStack, contextmanager

from einops import rearrange
from einops_exts import rearrange_with_anon_dims, check_shape

# multiprocessing

from joblib import Parallel, delayed, cpu_count

import math

import torch
import torch.nn.functional as F
from einops import rearrange


DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY = './.tmp/knn.memories'


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_list(val):
    return val if isinstance(val, list) else [val]

def all_el_unique(arr):
    return len(set(arr)) == len(arr)

def l2norm(t):
    return F.normalize(t, dim = -1)


def exists(val):
    return val is not None


def stable_softmax(t, dim = -1):
    t = t - t.amax(dim = dim, keepdim = True).detach()
    return F.softmax(t, dim = dim)


class KNNCust():
    def __init__(
        self,
        dim,
        max_num_entries,
        cap_num_entries = False,
        M = 15,
        keep_stats = False
    ):
        index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        self.index = index
        self.max_num_entries = max_num_entries
        self.cap_num_entries = cap_num_entries
        self.is_trained = False
        self.keep_stats = keep_stats

        self.reset()

    def __del__(self):
        if hasattr(self, 'index'):
            del self.index

    def reset(self):
        self.ids = np.empty((0,), dtype = np.int32)

        if self.keep_stats:
            self.hits = np.empty((0,), dtype = np.int32)
            self.age_num_iterations = np.empty((0,), dtype = np.int32)
            self.ages_since_last_hit = np.empty((0,), dtype = np.int32)

        self.index.reset()
        self.is_trained = False

    def train(self, x):
        self.index.train(x)
        self.is_trained = True

    def add(self, x, ids):
        if not self.is_trained:
            self.train(x)

        self.ids = np.concatenate((ids, self.ids))

        if self.keep_stats:
            self.hits = np.concatenate((np.zeros_like(ids), self.hits))
            self.age_num_iterations = np.concatenate((np.zeros_like(ids), self.age_num_iterations))
            self.ages_since_last_hit = np.concatenate((np.zeros_like(ids), self.ages_since_last_hit))

        if self.cap_num_entries and len(self.ids) > self.max_num_entries:
            self.reset()

        return self.index.add(x)

    def search(
        self,
        x,
        topk,
        nprobe = 8,
        return_distances = False,
        increment_hits = False,
        increment_age = True
    ):
        if not self.is_trained:
            return np.full((x.shape[0], topk), -1)

        distances, indices = self.index.search(x, k = topk)

        if increment_hits and self.keep_stats:
            hits = count_intersect(self.ids, rearrange(indices, '... -> (...)'))
            self.hits += hits

            self.ages_since_last_hit += 1
            self.ages_since_last_hit *= (hits == 0)

        if increment_age and self.keep_stats:
            self.age_num_iterations += 1

        if return_distances:
            return indices, distances

        return indices


class KNNMemoryListCust(list):
    def cleanup(self):
        for memory in self:
            del memory

    @classmethod
    def create_memories(
        self,
        *,
        batch_size,
        num_memory_layers,
        memories_directory = DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY
    ):
        memories_path = Path(memories_directory)
        memories_path.mkdir(exist_ok = True, parents = True)

        def inner(*args, **kwargs):
            return self([KNNMemoryCust(
                *args, 
                num_indices = batch_size, 
                memmap_filename = str(memories_path / f'knn.memory.layer.{ind + 1}.memmap'), **kwargs)
            for ind in range(num_memory_layers)])
        return inner

    @contextmanager
    def at_batch_indices(
        self,
        indices
    ):
        knn_batch_indices_contexts = [memory.at_batch_indices(indices) for memory in self]
        with multi_context(*knn_batch_indices_contexts):
            yield

    def clear_memory(
        self,
        batch_indices = None,
        memory_indices = None
    ):
        memory_indices = default(memory_indices, tuple(range(len(self))))

        for memory_index in memory_indices:
            memory = self[memory_index]
            memory.clear(batch_indices)
        

class KNNMemoryListCustPerHead(list):
    def cleanup(self):
        for memory in self:
            del memory

    @classmethod
    def create_memories(
        self,
        *,
        batch_size,
        num_memory_layers,
        memories_directory = DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY,
        num_heads=12
    ):
        memories_path = Path(memories_directory)
        memories_path.mkdir(exist_ok = True, parents = True)

        def inner(*args, **kwargs):
            per_layer_knn_mems = []
            for layer_ind in range(num_memory_layers):
                per_layer_knn_mems.append(
                    self([
                        KNNMemoryCust(
                            *args, 
                            num_indices = batch_size, 
                            memmap_filename = str(memories_path / f'knn.memory.layer.{layer_ind + 1}.head_{head + 1}.memmap'), **kwargs
                        ) for head in range(num_heads)
                    ]))
            return per_layer_knn_mems
        return inner

    @contextmanager
    def at_batch_indices(
        self,
        indices
    ):
        knn_batch_indices_contexts = [memory.at_batch_indices(indices) for memory in self]
        with multi_context(*knn_batch_indices_contexts):
            yield

    def clear_memory(
        self,
        batch_indices = None,
        memory_indices = None
    ):
        memory_indices = default(memory_indices, tuple(range(len(self))))

        for memory_index in memory_indices:
            memory = self[memory_index]
            memory.clear(batch_indices)

        
class KNNMemoryCust():
    def __init__(
        self,
        dim,
        max_memories = 16000,
        num_indices = 1,
        memmap_filename = './knn.memory.memmap',
        multiprocessing = True,
        cap_num_entries_in_knn=False
    ):
        self.dim = dim
        self.num_indices = num_indices
        self.scoped_indices = list(range(num_indices))

        self.max_memories = max_memories
        self.shape = (num_indices, max_memories, 2, dim)
        self.db_offsets = np.zeros(num_indices, dtype = np.int32)

        self.db = np.memmap(memmap_filename, mode = 'w+', dtype = np.float32, shape = self.shape)
        self.knns = [KNNCust(dim = dim, max_num_entries = max_memories, cap_num_entries = cap_num_entries_in_knn) for _ in range(num_indices)]
    
        self.n_jobs = cpu_count() if multiprocessing else 1

        self.tokens_at_index = np.array([[None for _ in range(max_memories)] for batch_ind in self.scoped_indices])
        self.searched_to_retrieved_tokens = defaultdict(list)

    def set_scoped_indices(self, indices):
        indices = list(indices)
        assert all_el_unique(indices), f'all scoped batch indices must be unique, received: {indices}'
        assert all([0 <= i < self.num_indices for i in indices]), f'each batch index must be between 0 and less than {self.num_indices}: received {indices}'
        self.scoped_indices = indices

    @contextmanager
    def at_batch_indices(self, indices):
        prev_indices = self.scoped_indices
        self.set_scoped_indices(indices)
        yield self
        self.set_scoped_indices(prev_indices)

    def clear(self, batch_indices = None):
        if not exists(batch_indices):
            batch_indices = list(range(self.num_indices))

        batch_indices = cast_list(batch_indices)

        for index in batch_indices:
            knn = self.knns[index]
            knn.reset()

        self.db_offsets[batch_indices] = 0

    def add(self, memories, tokens=None):
        check_shape(memories, 'b n kv d', d = self.dim, kv = 2, b = len(self.scoped_indices))

        memories = memories.detach().cpu().numpy()
        memories = memories[:, -self.max_memories:]
        num_memories = memories.shape[1]

        knn_insert_ids = np.arange(num_memories)

        keys = np.ascontiguousarray(memories[..., 0, :])
        knns = [self.knns[i] for i in self.scoped_indices]
        db_offsets = [self.db_offsets[i] for i in self.scoped_indices]

        # use joblib to insert new key / value memories into faiss index

        @delayed
        def knn_add(knn, key, db_offset):
            knn.add(key, ids = knn_insert_ids + db_offset)

        def knn_add_non_delayed(knn, key, db_offset):
            knn.add(key, ids = knn_insert_ids + db_offset)

        # with tqdm_joblib(tqdm(desc="My calculation", total=len(knns))) as progress_bar:
        Parallel(n_jobs = self.n_jobs)(knn_add(*args) for args in zip(knns, keys, db_offsets))
        # [knn_add_non_delayed(*args) for args in zip(knns, keys, db_offsets)]

        # add the new memories to the memmap "database"

        add_indices = (rearrange(np.arange(num_memories), 'j -> 1 j') + rearrange(self.db_offsets[list(self.scoped_indices)], 'i -> i 1')) % self.max_memories
        self.db[rearrange(np.array(self.scoped_indices), 'i -> i 1'), add_indices] = memories
        if tokens is not None:
            self.tokens_at_index[rearrange(np.array(self.scoped_indices), 'i -> i 1'), add_indices] = tokens
        self.db.flush()

        self.db_offsets += num_memories

    def search(
        self,
        queries,
        topk,
        nprobe = 8,
        increment_hits = True,
        increment_age = True,
        ordered_tokens = None  # This is a [Batch, Seq_len] array of tokens
    ):
        _, *prec_dims, _ = queries.shape
        check_shape(queries, 'b ... d', d = self.dim, b = len(self.scoped_indices))
        queries = rearrange(queries, 'b ... d -> b (...) d')

        device = queries.device
        queries = queries.detach().cpu().numpy()

        all_masks = []
        all_key_values = []

        knns = [self.knns[i] for i in self.scoped_indices]

        # parallelize faiss search

        @delayed
        def knn_search(knn, query):
            return knn.search(query, topk, nprobe, increment_hits = increment_hits, increment_age = increment_age)

        fetched_indices = Parallel(n_jobs = self.n_jobs)(knn_search(*args) for args in zip(knns, queries))

        # get all the memory key / values from memmap 'database'
        # todo - remove for loop below

        for batch_index, indices in zip(self.scoped_indices, fetched_indices):
            mask = indices !=  -1
            db_indices = np.where(mask, indices, 0)

            all_masks.append(torch.from_numpy(mask))

            key_values = self.db[batch_index, db_indices % self.max_memories]
            all_key_values.append(torch.from_numpy(key_values))

            if ordered_tokens is not None:
                tokens_for_batch = ordered_tokens[batch_index, :]
                for i, token in enumerate(tokens_for_batch):
                    self.searched_to_retrieved_tokens[token].append(list(self.tokens_at_index[batch_index][indices[i]]))

        all_masks = torch.stack(all_masks)
        all_key_values = torch.stack(all_key_values)
        all_key_values = all_key_values.masked_fill(~rearrange(all_masks, '... -> ... 1 1'), 0.)

        all_key_values = rearrange_with_anon_dims(all_key_values, 'b (...p) ... -> b ...p ...', p = prec_dims)
        all_masks = rearrange_with_anon_dims(all_masks, 'b (...p) ... -> b ...p ...', p = prec_dims)
        return all_key_values.to(device), all_masks.to(device)

    def __del__(self):
        if hasattr(self, 'knns'):
            for knn in self.knns:
                del knn
        del self.db