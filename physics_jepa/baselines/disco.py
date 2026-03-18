from torch.utils.data import DataLoader
import torch
from torch.nn import MSELoss
import wandb
from pathlib import Path
import argparse

from physics_jepa.utils.model_utils import CosineLRScheduler
from physics_jepa.data import DISCOLatentDataset
from physics_jepa.utils.data_utils import normalize_labels
from physics_jepa.attentive_pooler import AttentiveClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

data_path = Path(args.data_path)

NUM_CLASSES = {
    "disco_inference_gray_scott": 2,
    "disco_inference_shear_flow": 2,
    "disco_inference_rayleigh_benard": 2,
    "disco_inference_active_matter": 2,
}
STATS = {
    "disco_inference_gray_scott": {
        "means": [0.086, 0.058], # F, k
        "stds": [0.1, 0.005]
        # "mins": [0.014, 0.051],
        # "maxes": [0.098, 0.065],
    },
    "disco_inference_active_matter": {
        "means": [9.0, -3.0], # zeta, alpha
        "stds": [5.16, 1.41],
        # "mins": [1, -5], # zeta, alpha (ignoring L which is always the same)
        # "maxes": [17, -1],
    },
    "disco_inference_shear_flow": {
        "means": [4.85, 2.69], # rayleigh, schmidt
        "stds": [0.61, 3.38],
        "compression": ["log", None],
    },
    "disco_inference_rayleigh_benard": {
        "means": [8.0, 2.69], # rayleigh, prandtl
        "stds": [1.41, 3.38],
        "compression": ["log", None],
    },
}

label_stats = STATS[data_path.stem]
#TODO: need to add normalization for shear flow/active matter
NUM_EPOCHS = 100

train_dataset = DISCOLatentDataset(
    data_path,
    split="train",
)
val_dataset = DISCOLatentDataset(
    data_path,
    split="valid",
)
print(f"dataset size: {len(train_dataset)}, embedding shape: {train_dataset[0][0].shape}")
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

head = AttentiveClassifier(
    embed_dim=train_dataset[0][0].shape[-1], # should be 1x384
    num_classes=NUM_CLASSES[data_path.stem],
    num_heads=8,
)
print(head)
head.to('cuda')
optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)

lr_scheduler = CosineLRScheduler(
    optimizer,
    base_value=1e-3,
    final_value=0,
    epochs=NUM_EPOCHS,
    niter_per_ep=len(train_dataloader),
    warmup_epochs=2,
)

wandb.init(project="disco-inference", name=data_path.name)

for epoch in range(NUM_EPOCHS):
    head.train()
    for i, batch in enumerate(train_dataloader):
        if 'active_matter' in data_path.stem:
            batch[1] = batch[1][:, 1:] # ignore L, it's always the same

        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        preds = head(batch[0].to('cuda'))
        labels = normalize_labels(batch[1], stats=label_stats).to('cuda')
        loss = MSELoss()(preds, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i == 0:
            print(f"batch {i} preds: {preds[:5]}")
            print(f"batch {i} labels: {labels[:5]}")
            print(f"batch {i} loss: {loss.item()}")

        if i % 100 == 0:
            wandb.log({
                "loss": loss.item(),
                "lr": lr_scheduler.get_last_lr()[0]
            })

    head.eval()
    all_val_loss = 0
    for i, batch in enumerate(val_dataloader):
        with torch.no_grad():
            if 'active_matter' in data_path.stem:
                batch[1] = batch[1][:, 1:] # ignore L, it's always the same

            preds = head(batch[0].to('cuda'))
            labels = normalize_labels(batch[1], stats=label_stats).to('cuda')
            loss = MSELoss()(preds, labels)
        all_val_loss += loss.item()
    all_val_loss /= len(val_dataloader)
    wandb.log({
        "val/loss": all_val_loss
    })
    print(f"epoch {epoch} val loss: {all_val_loss}")