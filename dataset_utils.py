import csv
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

MAX_SAFE_CEILING = 10000
NUM_WORKERS      = 4


class SlideDatasetPT(Dataset):
    """
    Loads pre-extracted patch embeddings (.pt files) for each slide.

    Each .pt file is expected to contain:
        - features : Tensor[N, 1024]  — UNI ViT-L patch embeddings
        - coords   : Tensor[N, 2]     — (x, y) patch coordinates on the WSI grid

    Args:
        labels_csv : path to CSV with columns [patient_id, label]
                     label must be in {0, 1, 2} corresponding to G1, G2, G3
        pt_dir     : directory containing <patient_id>.pt files
        split_ids  : list of patient_id strings for this split (train/val/test)
    """
    def __init__(self, labels_csv, pt_dir, split_ids):
        self.pt_dir = Path(pt_dir)
        with open(labels_csv) as f:
            all_rows = list(csv.DictReader(f))
        split_set = set(split_ids)
        self.samples = []
        for row in all_rows:
            pid = row["patient_id"]
            if pid not in split_set:
                continue
            raw_label = int(row["label"])
            assert raw_label in {0, 1, 2}, f"Corrupt label {raw_label} for {pid}"
            pt_path = self.pt_dir / f"{pid}.pt"
            if pt_path.exists():
                self.samples.append({
                    "patient_id": pid,
                    "label":      raw_label,
                    "pt_path":    pt_path,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s    = self.samples[idx]
        data = torch.load(s["pt_path"], weights_only=True)
        return data["features"], s["label"], data["coords"], s["patient_id"]


def train_collate_fn(batch):
    """
    Stochastic pseudo-bag collation (DTFD strategy).

    For slides with > MAX_SAFE_CEILING patches, samples a random subset each call.
    This acts as data augmentation — the model sees a different bag view every epoch.
    Pads all bags in the batch to MAX_SAFE_CEILING with a boolean attention mask.
    Coords and patient IDs are discarded (not needed during training).
    """
    features, labels, _, __ = zip(*batch)
    B, D = len(features), features[0].shape[1]
    padded = np.zeros((B, MAX_SAFE_CEILING, D), dtype=np.float32)
    masks  = np.zeros((B, MAX_SAFE_CEILING),    dtype=bool)
    for i, f in enumerate(features):
        n = f.shape[0]
        if n > MAX_SAFE_CEILING:
            idx       = torch.randperm(n)[:MAX_SAFE_CEILING].numpy()
            padded[i] = f.numpy()[idx]
            masks[i]  = True
        else:
            padded[i, :n] = f.numpy()
            masks[i,  :n] = True
    return padded, np.array(labels, dtype=np.int32), masks


def val_collate_fn(batch):
    """
    Deterministic collation for validation and inference.

    Always takes the first MAX_SAFE_CEILING patches (no shuffling) so that
    predictions are reproducible and ensemble softmax averaging is valid.
    Returns coords and patient IDs for attention heatmap generation.
    """
    features, labels, coords, pids = zip(*batch)
    B, D = len(features), features[0].shape[1]
    padded     = np.zeros((B, MAX_SAFE_CEILING, D), dtype=np.float32)
    masks      = np.zeros((B, MAX_SAFE_CEILING),    dtype=bool)
    coords_out = []
    for i, (f, c) in enumerate(zip(features, coords)):
        n = f.shape[0]
        if n > MAX_SAFE_CEILING:
            padded[i]  = f.numpy()[:MAX_SAFE_CEILING]
            masks[i]   = True
            coords_out.append(c.numpy()[:MAX_SAFE_CEILING])
        else:
            padded[i, :n] = f.numpy()
            masks[i,  :n] = True
            coords_out.append(c.numpy())
    return padded, np.array(labels, dtype=np.int32), masks, coords_out, list(pids)
