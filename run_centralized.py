"""
Centralized baseline experiment: train a single DNN model on ALL data
(no FL, no partitioning, no NSAC, no DP noise).

Purpose: Show that even with all data concentrated, ML performance on
IoTID20 (93.6% attack / 6.4% normal) is inherently limited —
proving FL's AUC ≈ 0.6 is NOT a framework flaw, but a data challenge.

Usage:
  python run_centralized.py --seed 1
  python run_centralized.py --seed 1 2 3 4 5

This uses the SAME:
  - Dataset (IoTID20) & test split
  - Model (4-layer DNN, input_dim=79, hidden_dims=[64,32,16])
  - Learning rate η=0.01, batch_size=32
  - Evaluation metrics (AUC, macro-F1, Acc, optimal threshold)
  - 5 seeds for fair comparison

No FL overhead: no partitioning, no NSAC compression, no DP noise.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

# Add project root to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models.dnn import build_model


# ============================================================
# Centralized Training
# ============================================================
def centralized_train(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    seed: int = 1,
) -> dict:
    """
    Train a single model on ALL training data centrally.
    
    No FL, no partitioning, no NSAC, no DP noise.
    Uses plain CrossEntropyLoss (same as v6 FL).
    
    Returns dict with final metrics + per-epoch history.
    """
    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Build model (fresh initialization per seed)
    input_dim = X_train.shape[1]
    model = build_model(input_dim)
    d = model.num_parameters
    
    print(f"\n{'='*60}")
    print(f"Centralized Baseline on IoTID20 (seed={seed})")
    print(f"Model: 4-layer DNN, d={d}, input_dim={input_dim}")
    print(f"η={learning_rate}, batch_size={batch_size}, epochs={epochs}")
    print(f"Train: {len(y_train)} samples (class0={np.sum(y_train==0)}, "
          f"class1={np.sum(y_train==1)})")
    print(f"Test:  {len(y_test)} samples")
    print(f"{'='*60}\n")
    
    # Data
    train_data = torch.tensor(X_train, dtype=torch.float32)
    train_labels = torch.tensor(y_train, dtype=torch.long)
    dataset = TensorDataset(train_data, train_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Plain CE (same as v6 FL — no class weights, no oversampling)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    history = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        
        # Evaluate every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            metrics = evaluate_model(model, X_test, y_test)
            print(f"Epoch {epoch}/{epochs}: "
                  f"loss={avg_loss:.4f}, "
                  f"F1={metrics['f1']:.4f}, "
                  f"macro-F1={metrics['f1_macro']:.4f}, "
                  f"Acc={metrics['accuracy']:.4f}, "
                  f"AUC={metrics['auc_roc']:.4f}, "
                  f"thresh={metrics['optimal_threshold']:.2f}")
            history.append({'epoch': epoch, 'loss': avg_loss, **metrics})
    
    # Final evaluation
    final_metrics = evaluate_model(model, X_test, y_test)
    print(f"\nFinal: F1={final_metrics['f1']:.4f}, "
          f"macro-F1={final_metrics['f1_macro']:.4f}, "
          f"Acc={final_metrics['accuracy']:.4f}, "
          f"AUC={final_metrics['auc_roc']:.4f}, "
          f"optimal_threshold={final_metrics['optimal_threshold']:.2f}")
    
    return {
        'final_metrics': final_metrics,
        'epoch_history': history,
        'config': {
            'method': 'centralized',
            'dataset': 'iotid20',
            'seed': seed,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'd': d,
            'input_dim': input_dim,
            'num_train': len(y_train),
            'num_test': len(y_test),
            'class0_train': int(np.sum(y_train == 0)),
            'class1_train': int(np.sum(y_train == 1)),
        }
    }


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set with threshold optimization (same logic as FL simulator)."""
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_t)
        probs = torch.softmax(outputs, dim=1)
        preds_default = outputs.argmax(dim=1).numpy()
    
    y_true = y_test
    y_prob = probs[:, 1].numpy()  # P(class=1)
    
    # Threshold optimization: find threshold that maximizes F1
    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.05, 0.95, 0.05):
        preds_t = (y_prob >= thresh).astype(int)
        f1_t = f1_score(y_true, preds_t, zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = thresh
    
    preds_optimal = (y_prob >= best_thresh).astype(int)
    f1_optimal = best_f1
    acc_optimal = accuracy_score(y_true, preds_optimal)
    
    f1_default = f1_score(y_true, preds_default, zero_division=0)
    acc_default = accuracy_score(y_true, preds_default)
    
    f1_macro = f1_score(y_true, preds_optimal, average='macro', zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0
    
    # Diagnostic print
    n_cls0 = int(np.sum(y_true == 0))
    n_cls1 = int(np.sum(y_true == 1))
    print(f"  [Diag] Test: total={len(y_true)}, "
          f"class0={n_cls0}({n_cls0/len(y_true)*100:.1f}%), "
          f"class1={n_cls1}({n_cls1/len(y_true)*100:.1f}%)")
    print(f"  [Diag] Prob: P(cls1)_mean={y_prob.mean():.4f}, "
          f"std={y_prob.std():.4f}, "
          f"min={y_prob.min():.4f}, max={y_prob.max():.4f}")
    
    return {
        'f1': f1_optimal,
        'f1_default': f1_default,
        'f1_macro': f1_macro,
        'accuracy': acc_optimal,
        'accuracy_default': acc_default,
        'auc_roc': auc,
        'optimal_threshold': best_thresh,
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Run centralized baseline experiment on IoTID20')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                        help='Random seeds to run (default: 1 2 3 4 5)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs (default: 50, matching FL rounds T)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate η (default: 0.01, same as FL)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32, same as FL)')
    parser.add_argument('--output_dir', type=str, default='results_centralized',
                        help='Output directory for results')
    args = parser.parse_args()
    
    # Load the SAME train/test split as FL experiments
    processed_dir = str(ROOT / 'data' / 'processed')
    train_file = os.path.join(processed_dir, 'iotid20_train.npz')
    test_file = os.path.join(processed_dir, 'iotid20_test.npz')
    
    if not os.path.exists(train_file):
        print(f"❌ Train file not found: {train_file}")
        print("Please run preprocess first: python data/preprocess.py")
        sys.exit(1)
    
    train_npz = np.load(train_file)
    X_train = train_npz['X']
    y_train = train_npz['y']
    
    test_npz = np.load(test_file)
    X_test = test_npz['X']
    y_test = test_npz['y']
    
    print(f"[Data] Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"[Data] Feature dim: {X_train.shape[1]}")
    print(f"[Data] Train class distribution: "
          f"class0={np.sum(y_train==0)}({np.sum(y_train==0)/len(y_train)*100:.1f}%), "
          f"class1={np.sum(y_train==1)}({np.sum(y_train==1)/len(y_train)*100:.1f}%)")
    
    # Run for each seed
    all_finals = []
    
    for seed in args.seeds:
        model = None  # will be created inside centralized_train
        results = centralized_train(
            model, X_train, y_train, X_test, y_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=seed,
        )
        
        # Save per-seed results
        exp_name = f"centralized_iotid20_seed{seed}"
        out_path = os.path.join(args.output_dir, exp_name)
        os.makedirs(out_path, exist_ok=True)
        
        with open(os.path.join(out_path, 'results.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Quick summary
        fm = results['final_metrics']
        print(f"\n📊 {exp_name} Summary:")
        print(f"   F1 = {fm['f1']:.4f}  |  macro-F1 = {fm['f1_macro']:.4f}  |  "
              f"AUC = {fm['auc_roc']:.4f}  |  thresh = {fm['optimal_threshold']:.2f}")
        
        all_finals.append(fm)
    
    # Aggregate across seeds
    if len(all_finals) > 0:
        f1s = [m['f1'] for m in all_finals]
        f1_macros = [m['f1_macro'] for m in all_finals]
        accs = [m['accuracy'] for m in all_finals]
        aucs = [m['auc_roc'] for m in all_finals]
        
        summary = {
            'method': 'centralized',
            'dataset': 'iotid20',
            'num_seeds': len(all_finals),
            'f1_mean': np.mean(f1s),
            'f1_std': np.std(f1s),
            'f1_macro_mean': np.mean(f1_macros),
            'f1_macro_std': np.std(f1_macros),
            'accuracy_mean': np.mean(accs),
            'auc_mean': np.mean(aucs),
            'auc_std': np.std(aucs),
            'per_seed_results': all_finals,
        }
        
        print(f"\n{'='*60}")
        print(f"📊 Aggregated Centralized Baseline (IoTID20, {len(all_finals)} seeds):")
        print(f"   F1 = {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
        print(f"   macro-F1 = {summary['f1_macro_mean']:.4f} ± {summary['f1_macro_std']:.4f}")
        print(f"   Acc = {summary['accuracy_mean']:.4f}")
        print(f"   AUC = {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
        print(f"{'='*60}")
        
        # Save aggregate
        agg_file = os.path.join(args.output_dir,
                                f"aggregate_centralized_iotid20.json")
        with open(agg_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Results saved to {args.output_dir}/")
        print(f"   Per-seed: centralized_iotid20_seed{args.seeds[0]}/ ...")
        print(f"   Aggregate: aggregate_centralized_iotid20.json")


if __name__ == '__main__':
    main()
