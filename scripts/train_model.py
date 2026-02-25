"""
Unified training script for PhoBERT and ViSoBERT Fake News Detection models.

Usage:
    python scripts/train_model.py phobert      # Train PhoBERT only
    python scripts/train_model.py visobert     # Train ViSoBERT only
    python scripts/train_model.py all          # Train both sequentially
"""

import sys
import os
import argparse

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix,
)
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, logging as hf_logging
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.dataset import FakeNewsDetectionDatasetCached
from src.model import HybridModel
from src.augmentations import TextAugmentations

hf_logging.set_verbosity_error()

# ─────────────────────── CONFIG ───────────────────────
MODELS = {
    "phobert": {
        "name": "vinai/phobert-base-v2",
        "max_len": 256,           # max_position_embeddings=258
        "save": "checkpoints/phobert_best.pth",
        "curves_png": "checkpoints/phobert_training_curves.png",
        "cm_png": "checkpoints/phobert_confusion_matrix.png",
    },
    "visobert": {
        "name": "uitnlp/visobert",
        "max_len": 512,           # max_position_embeddings=514
        "save": "checkpoints/visobert_best.pth",
        "curves_png": "checkpoints/visobert_training_curves.png",
        "cm_png": "checkpoints/visobert_confusion_matrix.png",
    },
}

PICKLE_PATH = "src/cache_dataset.pkl"
BATCH_SIZE = 16
NUM_EPOCHS = 5
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 200
TRAIN_SPLIT = 0.85
STYLE_DIM = 10
NUM_LABELS = 2
HIDDEN_DIM = 256
DROPOUT = 0.3
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────── HELPERS ───────────────────────────
def train_epoch(model, loader, criterion, optimizer, scheduler, scaler):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="  Train"):
        ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        style = batch["style_vector"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        if scaler is not None:
            with autocast():
                logits = model(ids, mask, style)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(ids, mask, style)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        all_preds.extend(torch.argmax(logits, -1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), accuracy_score(all_labels, all_preds)


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="  Val  "):
        ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        style = batch["style_vector"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)

        logits = model(ids, mask, style)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        all_preds.extend(torch.argmax(logits, -1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    prec = precision_score(all_labels, all_preds, average="weighted")
    rec = recall_score(all_labels, all_preds, average="weighted")
    return total_loss / len(loader), acc, f1, prec, rec, all_preds, all_labels


def plot_results(history, final_preds, final_labels, val_acc, val_f1, cfg, tag):
    """Save training curves + confusion matrix PNGs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history["train_loss"], marker="o")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title=f"{tag} Training Loss")
    axes[0].grid(True)
    axes[1].plot(history["train_acc"], marker="o", color="green")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title=f"{tag} Training Accuracy")
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(cfg["curves_png"], dpi=150)
    plt.close()

    cm = confusion_matrix(final_labels, final_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"{tag} Confusion Matrix\nAcc: {val_acc:.4f} | F1: {val_f1:.4f}")
    plt.tight_layout()
    plt.savefig(cfg["cm_png"], dpi=150)
    plt.close()
    print(f"  Plots saved → {cfg['curves_png']}, {cfg['cm_png']}")


# ──────────────────── MAIN TRAIN ──────────────────────
def train_one(key: str):
    cfg = MODELS[key]
    tag = key.upper()
    print("\n" + "=" * 70)
    print(f"  TRAINING {tag}  —  {cfg['name']}  (max_len={cfg['max_len']})")
    print("=" * 70)

    # ── Dataset ──
    aug = TextAugmentations(
        p_style_drop=0.1, p_bm25_drop=0.1, p_white_space=0.1,
        p_punctuation_noise=0.2, p_case_noise=0.15,
        p_teencode=0.3, p_accent_drop=0.2, p_features_drop=0.1,
    )

    full_ds = FakeNewsDetectionDatasetCached(
        pickle_path=PICKLE_PATH, text_augmentations=aug,
        tokenizer_name=cfg["name"], max_len=cfg["max_len"],
    )
    train_n = int(TRAIN_SPLIT * len(full_ds))
    val_n = len(full_ds) - train_n
    train_ds, _ = random_split(full_ds, [train_n, val_n],
                               generator=torch.Generator().manual_seed(SEED))

    val_ds_clean = FakeNewsDetectionDatasetCached(
        pickle_path=PICKLE_PATH, text_augmentations=None,
        tokenizer_name=cfg["name"], max_len=cfg["max_len"],
    )
    _, val_idx = random_split(val_ds_clean, [train_n, val_n],
                              generator=torch.Generator().manual_seed(SEED))
    val_ds = torch.utils.data.Subset(val_ds_clean, val_idx.indices)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Model ──
    model = HybridModel(cfg["name"], STYLE_DIM, NUM_LABELS, HIDDEN_DIM, DROPOUT)
    tok = AutoTokenizer.from_pretrained(cfg["name"])
    model.bert.resize_token_embeddings(len(tok))
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)
    scaler = GradScaler() if DEVICE.type == "cuda" else None

    # ── Train ──
    history = {"train_loss": [], "train_acc": []}
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n  Epoch {epoch}/{NUM_EPOCHS}")
        loss, acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler)
        history["train_loss"].append(loss)
        history["train_acc"].append(acc)
        print(f"  Loss: {loss:.4f} | Acc: {acc:.4f}")

    # ── Validate ──
    print("\n  Validating...")
    val_loss, val_acc, val_f1, val_prec, val_rec, preds, labels = validate(model, val_loader, criterion)
    print(f"  Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
    print(f"  Precision: {val_prec:.4f} | Recall: {val_rec:.4f}")
    print(classification_report(labels, preds, target_names=["Fake", "Real"]))

    # ── Save ──
    os.makedirs(os.path.dirname(cfg["save"]), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_f1": val_f1,
        "history": {**history, "val_loss": val_loss, "val_acc": val_acc,
                     "val_f1": val_f1, "val_precision": val_prec, "val_recall": val_rec},
    }, cfg["save"])
    print(f"  Model saved → {cfg['save']}")

    # ── Plot ──
    plot_results(history, preds, labels, val_acc, val_f1, cfg, tag)

    # Cleanup GPU memory before next model
    del model, optimizer, scheduler, scaler, criterion
    torch.cuda.empty_cache()

    return val_acc, val_f1


# ──────────────────────── CLI ─────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fake News Detection models")
    parser.add_argument("target", choices=["phobert", "visobert", "all"],
                        help="Which model(s) to train")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs("checkpoints", exist_ok=True)

    results = {}
    if args.target in ("phobert", "all"):
        acc, f1 = train_one("phobert")
        results["phobert"] = {"acc": acc, "f1": f1}

    if args.target in ("visobert", "all"):
        acc, f1 = train_one("visobert")
        results["visobert"] = {"acc": acc, "f1": f1}

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE — SUMMARY")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {k.upper():>10}: Acc = {v['acc']:.4f}  |  F1 = {v['f1']:.4f}")
    print("=" * 70)
