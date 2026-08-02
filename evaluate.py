"""
Evaluation utilities: metrics computation, MIA attack, and result tables.
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Optional
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix
)
from pathlib import Path


# ============================================================
# Standard metrics
# ============================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray = None) -> Dict:
    """Compute all classification metrics."""
    metrics = {
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auc_roc'] = 0.0
    return metrics


# ============================================================
# Membership Inference Attack (MIA)
# ============================================================
class MembershipInferenceAttack:
    """
    Membership inference attack evaluation (Table 6).

    Attack model: neural network that predicts whether a sample
    was in the training set based on model output probabilities.
    """

    def __init__(self, target_model: torch.nn.Module,
                 attack_model_type: str = 'nn'):
        self.target_model = target_model
        self.attack_model_type = attack_model_type

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract membership features from target model outputs."""
        self.target_model.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.target_model(X_t)
            probs = torch.softmax(outputs, dim=1).numpy()
        # Features: class probabilities, entropy, max prob, loss
        entropy = -np.sum(probs * np.log2(probs + 1e-10), axis=1)
        max_prob = probs.max(axis=1)
        features = np.column_stack([probs, entropy, max_prob])
        return features

    def run_attack(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        test_X: np.ndarray,
        test_y: np.ndarray,
        attack_train_ratio: float = 0.5,
    ) -> Dict:
        """
        Run membership inference attack and evaluate defense.

        Returns: {attack_success_rate, auc_roc, precision, recall}
        """
        from sklearn.neural_network import MLPClassifier

        # Create membership labels: 1 = member (train), 0 = non-member (test)
        train_features = self._extract_features(train_X)
        test_features = self._extract_features(test_X)

        # Split target's train set for attack model training
        n_attack_train = int(len(train_X) * attack_train_ratio)
        n_attack_test_remaining = len(train_X) - n_attack_train

        # Attack training data: half of train (members) + half of test (non-members)
        attack_X = np.concatenate([
            train_features[:n_attack_train],
            test_features[:n_attack_train]
        ])
        attack_y = np.concatenate([
            np.ones(n_attack_train),   # members
            np.zeros(n_attack_train),  # non-members
        ])

        # Attack test data: remaining train (members) + remaining test (non-members)
        test_attack_X = np.concatenate([
            train_features[n_attack_train:n_attack_train + n_attack_test_remaining],
            test_features[n_attack_train:n_attack_train + n_attack_test_remaining]
        ])
        test_attack_y = np.concatenate([
            np.ones(n_attack_test_remaining),
            np.zeros(len(test_features) - n_attack_train),
        ])

        # Train attack model
        attack_model = MLPClassifier(hidden_layer_sizes=[64, 32],
                                      max_iter=200, random_state=42)
        attack_model.fit(attack_X, attack_y)

        # Evaluate attack
        attack_pred = attack_model.predict(test_attack_X)
        attack_prob = attack_model.predict_proba(test_attack_X)[:, 1]

        success_rate = accuracy_score(test_attack_y, attack_pred)
        try:
            attack_auc = roc_auc_score(test_attack_y, attack_prob)
        except ValueError:
            attack_auc = 0.5

        precision = precision_score(test_attack_y, attack_pred, zero_division=0)
        recall = recall_score(test_attack_y, attack_pred, zero_division=0)

        return {
            'attack_success_rate': success_rate,
            'attack_auc_roc': attack_auc,
            'attack_precision': precision,
            'attack_recall': recall,
        }


# ============================================================
# Result Table Generation
# ============================================================
def generate_table4(results_dir: str, seeds: list = [1,2,3,4,5]) -> str:
    """Generate Table 4 (anomaly detection F1/Accuracy/AUC)."""
    datasets = ['iotid20', 'cicids2017']
    alphas = [0.5, 0.1]
    methods = ['fedavg', 'fedprox', 'hierfed', 'dpfedavg', 'dpfed6g',
               'topk_qsgd', 'safeliot', 'fedprox_dp', 'hierfed_dp', 'proposed']

    table = "| Method |"
    for ds in datasets:
        for alpha in alphas:
            if alpha == 0.5:
                table += f" F1 ({ds}, α=0.5) | Acc | AUC |"
            else:
                table += f" F1 (α=0.1) |"
    table += "\n|--------|"

    # ... (simplified; full table generation from aggregate files)
    # This would read aggregate JSON files and format the table
    return table


def load_all_aggregates(results_dir: str) -> Dict:
    """Load all aggregate result files."""
    aggregates = {}
    for f in Path(results_dir).glob('aggregate_*.json'):
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        key = f"{data['method']}_{data['dataset']}_alpha{data['alpha']}"
        aggregates[key] = data
    return aggregates
