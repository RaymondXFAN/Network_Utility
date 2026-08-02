"""
Main experiment runner: one command to run all experiments across
datasets, seeds, alpha values, and methods (proposed + baselines).

Usage:
  python run_experiment.py --dataset iotid20 --alpha 0.5 --seed 1
  python run_experiment.py --dataset both --alpha both --method all
  bash run_all_seeds.sh   # 5 seeds × 2 datasets × 2 alphas
"""

import os
import sys
import json
import argparse
import yaml
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from core.simulator import HierFedMatterSimulator
from baselines.runner import run_baseline


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_single_experiment(
    dataset: str,
    alpha: float,
    seed: int,
    method: str = 'proposed',
    config_base: dict = None,
    config_nsac: dict = None,
    output_dir: str = 'results',
) -> dict:
    """Run a single experiment (one seed, one dataset, one method)."""

    if config_base is None:
        config_base = load_config(str(ROOT / 'configs' / 'base.yaml'))
    if config_nsac is None:
        config_nsac = load_config(str(ROOT / 'configs' / 'nsac.yaml'))

    # Merge configs
    config = {**config_base, **config_nsac}
    config['seed'] = seed
    config['alpha'] = alpha
    config['dataset_name'] = dataset
    config['processed_dir'] = str(ROOT / 'data' / 'processed')

    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{method}_{dataset}_alpha{alpha}_seed{seed}"

    print(f"\n{'='*70}")
    print(f"Experiment: {exp_name}")
    print(f"Timestamp:  {timestamp}")
    print(f"{'='*70}\n")

    if method == 'proposed':
        # Full HierFed-Matter-NSAC-DPBA
        try:
            sim = HierFedMatterSimulator(config)
            results = sim.run()
        except FileNotFoundError as e:
            print(f"⚠ Dataset not found, skipping: {e}")
            return {}
    else:
        # Baseline method
        try:
            results = run_baseline(method, config)
        except FileNotFoundError as e:
            print(f"⚠ Dataset not found, skipping: {e}")
            return {}

    # Save results
    out_path = os.path.join(output_dir, exp_name)
    os.makedirs(out_path, exist_ok=True)

    with open(os.path.join(out_path, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    # Quick summary
    if 'final_metrics' in results:
        fm = results['final_metrics']
        print(f"\n📊 {exp_name} Summary:")
        print(f"   F1 = {fm['f1']:.4f}  |  macro-F1 = {fm['f1_macro']:.4f}  |  "
              f"AUC = {fm['auc_roc']:.4f}  |  thresh = {fm['optimal_threshold']:.2f}")
        if 'communication_stats' in results:
            cs = results['communication_stats']
            print(f"   Speedup = {cs['speedup']:.1f}×  |  "
                  f"Core traffic = {cs['core_traffic_MB']:.1f} MB")

    return results


def aggregate_results(output_dir: str, dataset: str, alpha: float,
                      method: str = 'proposed', seeds: list = [1,2,3,4,5]):
    """Aggregate results across 5 seeds for Table 4/5/6 format."""
    f1s, accs, aucs, speedups = [], [], [], []

    for seed in seeds:
        exp_name = f"{method}_{dataset}_alpha{alpha}_seed{seed}"
        result_file = os.path.join(output_dir, exp_name, 'results.json')
        if not os.path.exists(result_file):
            print(f"  ⚠ Missing: {result_file}")
            continue
        with open(result_file, 'r', encoding='utf-8') as f:
            r = json.load(f)
        fm = r['final_metrics']
        f1s.append(fm['f1'])
        accs.append(fm['accuracy'])
        aucs.append(fm['auc_roc'])
        if 'communication_stats' in r:
            speedups.append(r['communication_stats']['speedup'])

    summary = {
        'method': method,
        'dataset': dataset,
        'alpha': alpha,
        'f1_mean': np.mean(f1s) if f1s else 0,
        'f1_std': np.std(f1s) if f1s else 0,
        'accuracy_mean': np.mean(accs) if accs else 0,
        'auc_mean': np.mean(aucs) if aucs else 0,
        'speedup_mean': np.mean(speedups) if speedups else 0,
        'num_seeds': len(f1s),
    }
    print(f"\n📊 Aggregated ({method}, {dataset}, α={alpha}):")
    print(f"   F1 = {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    print(f"   Acc = {summary['accuracy_mean']:.4f}")
    print(f"   AUC = {summary['auc_mean']:.4f}")
    if speedups:
        print(f"   Speedup = {summary['speedup_mean']:.1f}×")

    # Save aggregate
    agg_file = os.path.join(output_dir,
        f"aggregate_{method}_{dataset}_alpha{alpha}.json")
    with open(agg_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description='Run FL experiments')
    parser.add_argument('--dataset', choices=['iotid20', 'cicids2017', 'both'],
                        default='iotid20')
    parser.add_argument('--alpha', type=float, nargs='+', default=[0.5, 0.1])
    parser.add_argument('--seed', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--method', choices=['proposed', 'fedavg', 'fedprox',
                        'dpfedavg', 'hierfed', 'hierfed_dp', 'topk_qsgd',
                        'safeliot', 'dpfed6g', 'fedprox_dp', 'all'],
                        default='proposed')
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--config', default='configs/base.yaml')
    parser.add_argument('--nsac_config', default='configs/nsac.yaml')
    parser.add_argument('--aggregate_only', action='store_true',
                        help='Only aggregate existing results, skip running.')
    args = parser.parse_args()

    config_base = load_config(args.config)
    config_nsac = load_config(args.nsac_config)
    datasets = ['iotid20', 'cicids2017'] if args.dataset == 'both' else [args.dataset]
    alphas = args.alpha
    seeds = args.seed

    if args.method == 'all':
        methods = ['proposed', 'fedavg', 'fedprox', 'dpfedavg',
                   'hierfed', 'hierfed_dp', 'topk_qsgd']
    else:
        methods = [args.method]

    for dataset in datasets:
        for alpha in alphas:
            for method in methods:
                if not args.aggregate_only:
                    for seed in seeds:
                        run_single_experiment(
                            dataset, alpha, seed, method,
                            config_base, config_nsac, args.output_dir
                        )
                # Aggregate across seeds
                aggregate_results(args.output_dir, dataset, alpha, method, seeds)


if __name__ == '__main__':
    main()
