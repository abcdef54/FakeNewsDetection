"""
Test PhoBERT Fake News Detection Model
Load trained model and run inference on cached test samples
"""

import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from dataset import FakeNewsDetectionDatasetCached
from model import HybridModel
from transformers import AutoTokenizer

# Fix OpenMP duplicate library issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Configuration — must match training
MODEL_NAME = "vinai/phobert-base-v2"
STYLE_DIM = 10
NUM_LABELS = 2
HIDDEN_DIM = 256
DROPOUT = 0.3
MAX_LEN = 256          # PhoBERT-v2 max_position_embeddings = 258
BATCH_SIZE = 32
PICKLE_PATH = "src/CACHES/DATASET_CACHE/cached_ds.pkl"
MODEL_PATH = "src/checkpoints/phobert_best.pth"
TRAIN_SPLIT = 0.8

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("PHOBERT FAKE NEWS DETECTION - MODEL TESTING")
    print("=" * 70)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # ── 1. Load model ──────────────────────────────────────────────
    print("\n[1/4] Loading trained model...")
    try:
        model = HybridModel(
            model_name=MODEL_NAME,
            style_dim=STYLE_DIM,
            num_labels=NUM_LABELS,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT
        )
        # Resize embeddings to match tokenizer (same as training)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model.bert.resize_token_embeddings(len(tokenizer))

        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        print(f"  Model loaded from '{MODEL_PATH}'")
        if 'val_f1' in checkpoint:
            print(f"  Validation F1 Score: {checkpoint['val_f1']:.4f}")
        if 'history' in checkpoint:
            h = checkpoint['history']
            if 'train_acc' in h and h['train_acc']:
                print(f"  Final Train Acc: {h['train_acc'][-1]:.4f}")
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}' not found!")
        print("Please train the model first using train_phobert.ipynb")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # ── 2. Load test dataset (cached, no augmentation) ─────────────
    print("\n[2/4] Loading test dataset...")
    full_dataset = FakeNewsDetectionDatasetCached(
        pickle_path=PICKLE_PATH,
        text_augmentations=None,   
        tokenizer_name=MODEL_NAME,
        max_len=MAX_LEN
    )
    print(f"  Total samples: {len(full_dataset)}")

    # Same 80/20 split with seed 42 as training
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size

    _, test_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Quick label balance check
    sample_labels = []
    for i in range(min(100, len(test_dataset))):
        sample = test_dataset[i]
        sample_labels.append(sample['label'].item())

    n_fake_sample = sum(1 for l in sample_labels if l == 0)
    n_real_sample = sum(1 for l in sample_labels if l == 1)
    print(f"  Test set: {len(test_dataset)} samples")
    print(f"  Sample check (first 100): FAKE: {n_fake_sample}, REAL: {n_real_sample}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0, 
        pin_memory=True
    )

    print(f"Test dataset ready (same split as validation)")

    # Run inference
    print("\n[3/4] Running inference...")
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            style_vector = batch['style_vector'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask, style_vector)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    print("\n[4/4] Calculating metrics...")
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    # Print results
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Test Samples: {len(all_labels)}")
    print(f"Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1 Score:     {f1:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print("=" * 70)

    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Fake', 'Real'], digits=4))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Fake  Real")
    print(f"Actual Fake   {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"       Real   {cm[1][0]:4d}  {cm[1][1]:4d}")

    # Show some example predictions
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS (First 10)")
    print("=" * 70)
    for i in range(min(10, len(all_labels))):
        true_label = "FAKE" if all_labels[i] == 0 else "REAL"
        pred_label = "FAKE" if all_preds[i] == 0 else "REAL"
        confidence = all_probs[i][all_preds[i]] * 100
        
        status = "✓" if all_labels[i] == all_preds[i] else "✗"
        print(f"{i+1:2d}. {status} True: {true_label:4s} | Pred: {pred_label:4s} (confidence: {confidence:5.2f}%)")

    # Show confident wrong predictions
    print("\n" + "=" * 70)
    print("CONFIDENT MISTAKES (Top 5)")
    print("=" * 70)
    wrong_indices = np.where(all_preds != all_labels)[0]
    if len(wrong_indices) > 0:
        confidences = all_probs[wrong_indices, all_preds[wrong_indices]]
        top_mistakes = wrong_indices[np.argsort(-confidences)[:5]]
        
        for idx, i in enumerate(top_mistakes):
            true_label = "FAKE" if all_labels[i] == 0 else "REAL"
            pred_label = "FAKE" if all_preds[i] == 0 else "REAL"
            confidence = all_probs[i][all_preds[i]] * 100
            print(f"{idx+1}. Sample {i}: True={true_label}, Predicted={pred_label} (confidence: {confidence:.2f}%)")
    else:
        print("No mistakes found! Perfect predictions!")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'], 
                yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - Test Set\nAccuracy: {accuracy:.4f} | F1: {f1:.4f}')
    plt.tight_layout()
    cm_path = os.path.join('src/checkpoints', 'test_confusion_matrix_test.png')
    plt.savefig(cm_path, dpi=150)
    print(f"\nConfusion matrix saved to '{cm_path}'")

    # Plot confidence distribution
    plt.figure(figsize=(10, 5))
    correct_mask = all_preds == all_labels
    correct_confs = all_probs[np.arange(len(all_preds)), all_preds][correct_mask]
    wrong_confs = all_probs[np.arange(len(all_preds)), all_preds][~correct_mask]
    plt.hist(correct_confs, bins=30, alpha=0.6, label=f'Correct ({len(correct_confs)})', color='green')
    plt.hist(wrong_confs, bins=30, alpha=0.6, label=f'Wrong ({len(wrong_confs)})', color='red')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Prediction Confidence Distribution')
    plt.legend()
    plt.tight_layout()
    dist_path = os.path.join('src/checkpoints', 'test_confidence_distribution.png')
    plt.savefig(dist_path, dpi=150)
    print(f"Confidence distribution saved to '{dist_path}'")

    print("\n" + "=" * 70)
    print("TESTING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
