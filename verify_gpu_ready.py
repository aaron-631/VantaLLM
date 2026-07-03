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
        
        # Optional alignment to model's expected sequence length
        if self.seq_len:
            if ids.size(0) > self.seq_len:
                ids = ids[:self.seq_len]
            elif ids.size(0) < self.seq_len:
                pad_tensor = torch.zeros(self.seq_len - ids.size(0), dtype=ids.dtype, device=ids.device)
                ids = torch.cat([ids, pad_tensor])
        
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
    dataset = ShardedDataset(shards)
    log.info(f"Total samples across all shards: {len(dataset):,}")

    # 2. Verify Sequence Integrity
    sample_ids, sample_labels = dataset[0]
    log.info(f"Sample sequence shape: {sample_ids.shape}")
    log.info(f"Sample sequence dtype: {sample_ids.dtype}")

    if sample_ids.shape[0] != DEFAULT_TRAIN_SEQ_LEN:
        log.error(f"INVALID SEQUENCE LENGTH: Expected {DEFAULT_TRAIN_SEQ_LEN}, got {sample_ids.shape[0]}")
        return



    # 3. Verify Token Validity
    # Vocab size is 151,003. All tokens should be within [0, 151000).
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

def _smoke_test_no_shards():
    """
    Minimal GPU smoke test when no shards exist.
    Loads model in vanta_567m config, runs one forward+backward pass on synthetic data,
    and confirms loss is finite. Does NOT test the data pipeline.
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        import torch
        from vantallm.model.transformer import load_model, LLMConfig
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Smoke test device: {device}")

        model = load_model().to(device)
        if torch.cuda.is_available():
            model = model.to(torch.bfloat16)
        model.train()

        B, S = 2, 128
        ids    = torch.randint(0, 151003, (B, S), device=device)
        p_ids  = torch.zeros((B, 3), dtype=torch.long, device=device)
        labels = ids.clone()

        out = model(ids, persona_ids=p_ids, labels=labels)
        loss = out["loss"]

        assert not torch.isnan(loss), f"NaN loss in smoke test: {loss.item()}"
        assert not torch.isinf(loss), f"Inf loss in smoke test: {loss.item()}"
        loss.backward()

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e6
            peak  = torch.cuda.max_memory_allocated() / 1e6
            log.info(f"Smoke test VRAM | Alloc: {alloc:.0f}MB | Peak: {peak:.0f}MB")

        log.info(f"SMOKE TEST PASSED — Loss: {loss.item():.4f}. Model is healthy.")
        log.info("NOTE: Data pipeline NOT validated. Run full shard pipeline before long training runs.")

    except Exception as e:
        log.error(f"SMOKE TEST FAILED: {e}")
        raise

if __name__ == "__main__":
    validate()
