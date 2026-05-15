import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

# --------------------------
# --- Data Loading (Standard) ---
# --------------------------

FEATURE_ROOT_DIR = ''
TRAIN_CSV = ''
VAL_CSV = ''

LABEL_TO_ID = {'ADI': 0, 'BACK': 1, 'DEB': 2, 'LYM': 3, 'MUC': 4, 'MUS': 5, 'NORM': 6, 'STR': 7, 'TUM': 8}

num_classes = len(LABEL_TO_ID)

class PrecomputedFeatureDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.label_to_id = LABEL_TO_ID
        self.dir_v = os.path.join(root_dir, 'virchow')
        self.dir_u = os.path.join(root_dir, 'uni')
        self.dir_h = os.path.join(root_dir, 'hibou')

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        orig_path = row['path'].replace('\\', '/')
        base_name = os.path.splitext(orig_path)[0]
        pt_filename = base_name + ".pt"
        try:
            f_v = torch.load(os.path.join(self.dir_v, pt_filename), map_location='cpu', weights_only=True)
            f_u = torch.load(os.path.join(self.dir_u, pt_filename), map_location='cpu', weights_only=True)
            f_h = torch.load(os.path.join(self.dir_h, pt_filename), map_location='cpu', weights_only=True)
            label = self.label_to_id[row['label']]
            return {'v': f_v, 'u': f_u, 'h': f_h, 'label': label}
        except Exception as e:
            return None

def collate_fn_masked(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None, None
    out = {}
    for key in ['v', 'u', 'h']:
        tensors = [item[key] for item in batch]
        padded = pad_sequence(tensors, batch_first=True, padding_value=0.0)
        lengths = torch.tensor([t.size(0) for t in tensors])
        max_len = padded.size(1)
        mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
        full_name = {'v': 'virchow', 'u': 'uni', 'h': 'hibou'}[key]
        out[full_name] = padded
        out[f'mask_{key}'] = mask
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    return out, labels

def crc100k_dataloader():

    train_ds = PrecomputedFeatureDataset(TRAIN_CSV, FEATURE_ROOT_DIR)
    val_ds = PrecomputedFeatureDataset(VAL_CSV, FEATURE_ROOT_DIR)

    return train_ds, val_ds