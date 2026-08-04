"""
train_gesture_model.py

Fine-tunes EfficientNetV2-S (ImageNet-pretrained) on a 13-class ASL-letter
ImageFolder dataset, remapping ASL letters to robot command names so the
saved model integrates directly with gesture_inference.py.

Two-stage training:
  Stage 1 — backbone frozen, only the new classifier head trains (fast,
            stabilizes the head before touching pretrained features).
  Stage 2 — full network unfrozen and fine-tuned end-to-end at a lower LR.

Usage:
    python train_gesture_model.py
"""

import os
import copy

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join("dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
OUT_DIR = "models"
OUT_PATH = os.path.join(OUT_DIR, "best_gesture_model.pth")

ARCHITECTURE = "efficientnet_v2_s"
IMG_SIZE = 224
BATCH_SIZE = 32

FREEZE_EPOCHS = 5      # Stage 1: classifier head only
FINETUNE_EPOCHS = 30   # Stage 2: full network, subject to early stopping
HEAD_LR = 3e-4
FINETUNE_LR = 1e-5     # much lower — we're nudging pretrained features, not relearning them
WEIGHT_DECAY = 1e-4
STEP_SIZE = 7          # StepLR: decay LR every N epochs
GAMMA = 0.5            # StepLR: decay factor
EARLY_STOP_PATIENCE = 5
NUM_WORKERS = min(4, os.cpu_count() or 1)

# ASL letter -> robot command name. ImageFolder sorts class folders
# alphabetically, so this dict's key order defines the expected folder
# layout; the VALUES are what the model actually learns to predict.
LETTER_TO_COMMAND = {
    "A": "SHOULDER_FORWARD",
    "B": "STOP",
    "C": "SHOULDER_BACKWARD",
    "D": "FORWARD",
    "G": "LEFT",
    "H": "YAW_LEFT",
    "I": "ELBOW_DOWN",
    "K": "RIGHT",
    "L": "BACKWARD",
    "O": "GRIPPER_OPEN",
    "S": "GRIPPER_CLOSE",
    "U": "YAW_RIGHT",
    "Y": "ELBOW_UP",
}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, val_tf


def build_model(num_classes: int) -> nn.Module:
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = efficientnet_v2_s(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for param in model.features.parameters():
        param.requires_grad = trainable


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        pbar = tqdm(loader, desc="train" if train else "val", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()

            preds = outputs.argmax(1)
            total_loss += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc, all_preds, all_labels


def train_stage(model, train_loader, val_loader, criterion, optimizer, scheduler,
                 device, num_epochs, stage_name, best_val_acc, best_state,
                 epochs_without_improvement):
    """Runs one training stage (frozen-head or full fine-tune), sharing the
    same best-checkpoint/early-stopping state across stages."""
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        print(f"[{stage_name} | Epoch {epoch}/{num_epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            print(f"  -> New best val_acc={best_val_acc:.4f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"[INFO] Early stopping — no improvement for {EARLY_STOP_PATIENCE} epochs.")
                break

    return best_val_acc, best_state, epochs_without_improvement


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    train_tf, val_tf = build_transforms()
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=val_tf)

    # ImageFolder assigns class indices alphabetically by folder name.
    # Verify the dataset actually has exactly the 13 expected letter
    # folders — a missing folder would otherwise silently train on fewer
    # classes with no warning.
    expected_letters = sorted(LETTER_TO_COMMAND.keys())
    if train_ds.classes != expected_letters:
        raise ValueError(
            f"Dataset classes are {train_ds.classes}\nExpected {expected_letters}"
        )
    if val_ds.classes != expected_letters:
        raise ValueError(
            f"Validation dataset classes are {val_ds.classes}\nExpected {expected_letters}"
        )

    command_classes = [LETTER_TO_COMMAND[ltr] for ltr in train_ds.classes]
    print(f"[INFO] Classes (ASL -> command): {list(zip(train_ds.classes, command_classes))}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    model = build_model(len(command_classes)).to(device)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    epochs_without_improvement = 0

    # --- Stage 1: freeze backbone, train classifier head only ---
    print(f"\n[INFO] Stage 1: training classifier head only ({FREEZE_EPOCHS} epochs, backbone frozen)")
    set_backbone_trainable(model, trainable=False)
    head_optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    head_scheduler = StepLR(head_optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    best_val_acc, best_state, epochs_without_improvement = train_stage(
        model, train_loader, val_loader, criterion, head_optimizer, head_scheduler,
        device, FREEZE_EPOCHS, "Stage1-Head", best_val_acc, best_state, epochs_without_improvement,
    )

    # --- Stage 2: unfreeze backbone, fine-tune end-to-end at low LR ---
    print(f"\n[INFO] Stage 2: fine-tuning full network (up to {FINETUNE_EPOCHS} epochs, early stopping active)")
    set_backbone_trainable(model, trainable=True)
    finetune_optimizer = AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    finetune_scheduler = StepLR(finetune_optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    epochs_without_improvement = 0  # reset patience for the new stage
    best_val_acc, best_state, epochs_without_improvement = train_stage(
        model, train_loader, val_loader, criterion, finetune_optimizer, finetune_scheduler,
        device, FINETUNE_EPOCHS, "Stage2-FineTune", best_val_acc, best_state, epochs_without_improvement,
    )

    # Restore best weights found across both stages, then report on the
    # val set one final time using those exact restored weights.
    model.load_state_dict(best_state)
    _, _, final_preds, final_labels = run_epoch(model, val_loader, criterion, finetune_optimizer, device, train=False)

    print("\n[INFO] Classification report (best model, validation set):")
    print(classification_report(final_labels, final_preds, target_names=command_classes))
    print("[INFO] Confusion matrix:")
    print(confusion_matrix(final_labels, final_preds))

    torch.save({
        "model_state_dict": model.state_dict(),   # guaranteed to match the restored best model
        "classes": command_classes,               # index -> robot command name
        "letter_to_command": LETTER_TO_COMMAND,
        "best_val_acc": best_val_acc,
        "optimizer": finetune_optimizer.state_dict(),
        "scheduler": finetune_scheduler.state_dict(),
        "architecture": ARCHITECTURE,
        "image_size": IMG_SIZE,
        "mean": MEAN,
        "std": STD,
    }, OUT_PATH)
    print(f"\n[INFO] Saved best model (val_acc={best_val_acc:.4f}) to {OUT_PATH}")


if __name__ == "__main__":
    main()
