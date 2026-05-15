import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import f1_score
from module.KCNet import GraphKANFusion
from datasets.CRC100K import collate_fn_masked, crc100k_dataloader

# --------------------------
# --- Configuration ---
# --------------------------

NUM_CLASSES = 16
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 200

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

LOG_DIR = 'lightning_logs/'
LOG_NAME = 'CRC100K7K'

# --------------------------
# --- Lightning Module ---
# --------------------------

class GraphFusionModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        # Simplified weights
        self.w_aux = 0.3
        self.w_orth = 0.3
        self.w_con = 0.3

        self.validation_step_outputs = []

    def compute_orthogonal_loss(self, shared_list, private_list):
        loss = 0.0
        count = 0
        for s, p in zip(shared_list, private_list):
            s_flat = s.contiguous().view(-1, s.shape[-1])
            p_flat = p.contiguous().view(-1, p.shape[-1])
            num_samples = min(2000, s_flat.size(0))
            if num_samples > 0:
                idx = torch.randperm(s_flat.size(0))[:num_samples]
                sim = F.cosine_similarity(s_flat[idx], p_flat[idx], dim=-1)
                loss += torch.abs(sim).mean()
                count += 1
        return loss / max(count, 1)

    def compute_infonce_loss(self, feat_a, feat_b, temperature=0.5):
        batch_size = feat_a.shape[0]
        feat_a = F.normalize(feat_a, dim=1)
        feat_b = F.normalize(feat_b, dim=1)
        logits = torch.matmul(feat_a, feat_b.T) / temperature
        labels = torch.arange(batch_size).to(feat_a.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    def training_step(self, batch, batch_idx):
        features, labels = batch
        if features is None: return None

        final_logits, expert_logits, logits_correction, \
        shared_list, private_list, shared_pool_list = self.model(features)

        # 1. Main Loss
        loss_main = self.criterion(final_logits, labels)

        # 2. Aux Loss
        loss_aux = 0
        for logits in expert_logits:
            loss_aux += self.criterion(logits, labels)

        # 3. Regularization
        loss_orth = self.compute_orthogonal_loss(shared_list, private_list)

        loss_con = 0
        s_v, s_u, s_h = shared_pool_list
        loss_con += self.compute_infonce_loss(s_v, s_u)
        loss_con += self.compute_infonce_loss(s_u, s_h)
        loss_con += self.compute_infonce_loss(s_v, s_h)
        loss_con = loss_con / 3.0

        # 4. Sparsity Penalty on Correction (Optional)
        # We want the council to only intervene when necessary, not override everything.
        loss_sparsity = torch.norm(logits_correction, p=1) * 1e-4

        # Total
        loss = loss_main + \
               self.w_aux * loss_aux + \
               self.w_orth * loss_orth + \
               self.w_con * loss_con + \
               loss_sparsity

        # Logging
        preds = final_logits.argmax(dim=1)
        acc = (preds == labels).float().mean()

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        # Monitor how much the council is "speaking up"
        self.log('correction_mag', logits_correction.abs().mean())

        return loss

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        final_logits, _, _, _, _, _ = self.model(features)
        loss = self.criterion(final_logits, labels)
        preds = final_logits.argmax(dim=1)

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.append({'preds': preds, 'labels': labels})
        return {'preds': preds, 'labels': labels}

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if not outputs: return
        all_preds = torch.cat([x['preds'] for x in outputs])
        all_labels = torch.cat([x['labels'] for x in outputs])
        val_f1 = f1_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='macro')
        self.log('val_f1', val_f1, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

def run_training():
    print("\n=== Launching Training: Residual Council Fusion (Stable) ===")
    model = GraphKANFusion(num_classes=NUM_CLASSES)
    pl_module = GraphFusionModule(model)

    train_ds, val_ds = crc100k_dataloader()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=8, collate_fn=collate_fn_masked, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=8, collate_fn=collate_fn_masked, persistent_workers=True)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1', mode='max', save_top_k=1, filename='residual_fusion-{epoch}-{val_f1:.4f}'
    )

    trainer = Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        callbacks=[checkpoint_callback, EarlyStopping('val_f1', patience=30, mode='max')],
        logger=TensorBoardLogger(LOG_DIR, name=LOG_NAME),
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm"
    )
    trainer.fit(pl_module, train_loader, val_loader)

if __name__ == '__main__':
    run_training()