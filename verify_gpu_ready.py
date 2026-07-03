"""
verify_gpu_ready.py — Final Data Pipeline Validation
===================================================
Verifies shard integrity and training loop compatibility.
"""

import os
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
        self.current_shard_idx = -1
        self.data = None
        self.seq_len = seq_len
        
        # Calculate total length without loading everything
        self.lengths = []
        for p in shard_paths:
            # weights_only=False: preserve backward-compatible loading
            t = torch.load(p, map_location="cpu", weights_only=False)
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
            self.lengths.append(len(t))
        self.total_len = sum(self.lengths)
        self.cumulative_lengths = [0] + torch.cumsum(torch.tensor(self.lengths), dim=0).tolist()

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # Find which shard this index belongs to
        shard_idx = -1
        for i in range(len(self.cumulative_lengths) - 1):
            if idx >= self.cumulative_lengths[i] and idx < self.cumulative_lengths[i+1]:
                shard_idx = i
                break
        
        if shard_idx == -1:
            raise IndexError("Index out of bounds")
            
        if self.current_shard_idx != shard_idx:
            log.info(f"Loading shard {self.shard_paths[shard_idx]} into RAM...")
            # weights_only=False: preserve backward-compatible loading
            self.data = torch.load(self.shard_paths[shard_idx], map_location="cpu", weights_only=False)
            self.current_shard_idx = shard_idx
            
        local_idx = idx - self.cumulative_lengths[shard_idx]
        ids = self.data[local_idx]
        
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
