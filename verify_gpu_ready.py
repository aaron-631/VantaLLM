"""
verify_gpu_ready.py — Final Data Pipeline Validation
===================================================
Verifies shard integrity and training loop compatibility.
"""

import os
import bisect
import torch
import logging
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("validation")

SHARD_DIR = "data_pipeline/shards"

class ShardedDataset(torch.utils.data.Dataset):
    def __init__(self, shard_paths=None, seq_len=None):
        if shard_paths is None:
            if not os.path.exists(SHARD_DIR):
                raise FileNotFoundError(
                    f"Shard directory not found: {SHARD_DIR}. "
                    "Shard mode requires generated .bin files. "
                    "Please run 'python3 data_pipeline/shard_data.py' first."
                )

            shard_paths = sorted([
                os.path.join(SHARD_DIR, f) for f in os.listdir(SHARD_DIR)
                if f.endswith(".bin")
            ])

            if not shard_paths:
                raise FileNotFoundError(
                    f"No .bin shards found in {SHARD_DIR}. "
                    "Note: .gitkeep is not valid shard data. "
                    "Please run 'python3 data_pipeline/shard_data.py' first."
                )
            
        self.shard_paths = shard_paths
        self.seq_len = seq_len

        # Memory-map every shard once. mmap=True keeps the tensor storage backed
        # by the file on disk instead of copying it into RAM, so:
        #   * __init__ no longer fully materialises every shard just to read its
        #     length (it only touches metadata), and
        #   * __getitem__ can random-access any row of any shard without ever
        #     re-loading a whole shard from disk on a boundary cross.
        # This is what makes shuffle=True across shards cheap instead of thrashing
        # (Finding F08). Handles are held open for the dataset's lifetime; the OS
        # page cache manages resident memory.
        self._shards = []
        self.lengths = []
        for p in shard_paths:
            # weights_only=False: preserve backward-compatible loading
            t = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
            if t.ndim != 2:
                raise ValueError(
                    f"Shard '{p}' must have shape [num_sequences, sequence_length]; "
                    f"found {tuple(t.shape)}."
                )
            if self.seq_len is not None and t.shape[1] != self.seq_len:
                raise ValueError(
                    f"Shard '{p}' stores {t.shape[1]}-token sequences, but "
                    f"TRAIN_SEQ_LEN={self.seq_len}. Repack the shards or set "
                    "TRAIN_SEQ_LEN to the stored sequence length; refusing to "
                    "silently truncate or pad fixed-width training data."
                )
            self._shards.append(t)
            self.lengths.append(len(t))
        self.total_len = sum(self.lengths)
        self.cumulative_lengths = [0] + torch.cumsum(torch.tensor(self.lengths), dim=0).tolist()

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < 0:
            idx += self.total_len
        if idx < 0 or idx >= self.total_len:
            raise IndexError("Index out of bounds")

        # Binary-search the shard containing this global index.
        shard_idx = bisect.bisect_right(self.cumulative_lengths, idx) - 1

        local_idx = idx - self.cumulative_lengths[shard_idx]
        # Row-level read from the memory-mapped shard: only this sequence's pages
        # are faulted in, not the whole shard.
        ids = self._shards[shard_idx][local_idx]
        
        # Match train.py contract: (input_ids, labels)
        # Defensively cast to long (int64) — nn.Embedding requires int64.
        # Guards against existing shards built with dtype=torch.int32.
        ids = ids.long()
        labels = ids.clone()
        labels[labels == 0] = -100  # Mask padding tokens in loss
        return ids, labels

# Operational bring-up sequence length used for
# training/validation workflows.
# Does NOT modify architectural max_seq_len capability.
DEFAULT_TRAIN_SEQ_LEN = 2048

def validate():
    if not os.path.exists(SHARD_DIR):
        raise FileNotFoundError(
            f"Shard directory not found: {SHARD_DIR}. "
            "Shard mode requires generated .bin files. "
            "Please run 'python3 data_pipeline/shard_data.py' first."
        )

    shards = sorted([
        os.path.join(SHARD_DIR, f) for f in os.listdir(SHARD_DIR)
        if f.endswith(".bin")
    ])

    if not shards:
        raise FileNotFoundError(
            f"No .bin shards found in {SHARD_DIR}. "
            "Note: .gitkeep is not valid shard data. "
            "Please run 'python3 data_pipeline/shard_data.py' first."
        )

    log.info(f"Found {len(shards)} shards.")

    # 1. Verify Dataset Loading
    dataset = ShardedDataset(shards, seq_len=DEFAULT_TRAIN_SEQ_LEN)
    log.info(f"Total samples across all shards: {len(dataset):,}")

    # 2. Verify Sequence Integrity
    sample_ids, sample_labels = dataset[0]
    log.info(f"Sample sequence shape: {sample_ids.shape}")
    log.info(f"Sample sequence dtype: {sample_ids.dtype}")

    if sample_ids.shape[0] != DEFAULT_TRAIN_SEQ_LEN:
        log.error(f"INVALID SEQUENCE LENGTH: Expected {DEFAULT_TRAIN_SEQ_LEN}, got {sample_ids.shape[0]}")
        return



    # 3. Verify Token Validity
    # Vocab size is 151,003. All tokens should be within [0, 151003).
    max_token = torch.max(sample_ids).item()
    min_token = torch.min(sample_ids).item()
    log.info(f"Token range in first sample: [{min_token}, {max_token}]")
    
    if max_token >= 151003:
        log.error(f"TOKEN OVERFLOW: Found token {max_token} >= 151003")
        return

    # 4. Verify Dataloader Compatibility
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch_ids, batch_labels = next(iter(dataloader))
    log.info(f"Dataloader batch shape: {batch_ids.shape}")
    
    log.info("FINAL VERIFICATION SUCCESSFUL: Pipeline outputs are training-ready.")

if __name__ == "__main__":
    validate()
