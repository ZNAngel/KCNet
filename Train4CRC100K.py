import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import f1_score
from module.KCNet import GraphKANFusion
from module.losses import KCNetObjective
from datasets.CRC100K import LABEL_TO_ID, collate_fn_masked, crc100k_dataloader

# --------------------------
# --- Configuration ---
# --------------------------

NUM_CLASSES = len(LABEL_TO_ID)
# Existing example-run settings; this methods-only update does not reproduce
# the manuscript's experimental sampling / optimization protocol.
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
    def __init__(self, model, *, alpha_aux=0.3, alpha_orth=0.3, alpha_con=0.3, lambda_sp=1e-4):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.objective = KCNetObjective(alpha_aux, alpha_orth, alpha_con, lambda_sp)
        self.save_hyperparameters(ignore=['model'])
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        features, labels = batch
        if features is None: return None

        outputs = self.model(features)
        final_logits, _, logits_correction, _, _, _ = outputs
        losses = self.objective(self.model, outputs, labels, self.model.get_masks(features))
        loss = losses['total']

        # Logging
        preds = final_logits.argmax(dim=1)
        acc = (preds == labels).float().mean()

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        for name in ('main', 'aux', 'orth', 'con', 'sparsity'):
            self.log(f'train_{name}', losses[name])
        self.log('correction_mag', logits_correction.abs().mean())

        return loss

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        if features is None: return None
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
    print("\n=== Launching Training: KCNet (CFC + GFR + RCF) ===")
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
