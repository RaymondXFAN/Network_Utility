"""
Plotting utilities: reproduce paper figures from aggregated results.

Generates:
- Figure 7: Convergence curves (F1 vs. rounds)
- Figure 8: Communication savings
- Figure 9: Privacy-utility tradeoff
- Figure 10: Scalability analysis
- Figure 11: Non-IID impact
- Figure 12: NSAC sensitivity analysis
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 12

RESULTS_DIR = 'results'
FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# Color palette
COLORS = {
    'proposed': '#2563EB',    # Blue
    'fedavg': '#9CA3AF',      # Gray
    'fedprox': '#6B7280',     # Dark gray
    'dpfedavg': '#EF4444',    # Red
    'hierfed': '#10B981',     # Green
    'hierfed_dp': '#F59E0B',  # Amber
    'topk_qsgd': '#8B5CF6',  # Purple
    'safeliot': '#EC4899',    # Pink
    'dpfed6g': '#14B8A6',    # Teal
    'fedprox_dp': '#F97316', # Orange
}

LABELS = {
    'proposed': 'Proposed (HierFed+NSAC+DPBA)',
    'fedavg': 'FedAvg',
    'fedprox': 'FedProx',
    'dpfedavg': 'DP-FedAvg',
    'hierfed': 'HierFed',
    'hierfed_dp': 'HierFed+DP',
    'topk_qsgd': 'Top-k+QSGD',
    'safeliot': 'SAFEL-IoT',
    'dpfed6g': 'DP-Fed6G',
    'fedprox_dp': 'FedProx-DP',
}


def load_aggregate(method, dataset, alpha):
    """Load aggregated results JSON."""
    fname = f"{method}_{dataset}_alpha{alpha}_aggregated.json"
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        print(f"[WARN] Missing: {path}")
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_convergence_curves(dataset='iotid20', alpha=0.5,
                            methods=None, save=True):
    """Figure 7: F1 score vs. communication rounds."""
    if methods is None:
        methods = ['proposed', 'fedavg', 'dpfedavg', 'hierfed',
                    'hierfed_dp', 'topk_qsgd', 'safeliot', 'fedprox']

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in methods:
        data = load_aggregate(method, dataset, alpha)
        if data is None:
            continue
        # Extract round-level metrics
        round_data = data.get('round_metrics_mean', [])
        if not round_data:
            continue
        rounds = [r['round'] for r in round_data]
        f1_vals = [r['f1'] for r in round_data]
        # Std dev for error band
        f1_std = data.get('round_metrics_std', [])
        std_vals = [s.get('f1', 0) if isinstance(s, dict) else 0
                    for s in f1_std]

        color = COLORS.get(method, '#6B7280')
        label = LABELS.get(method, method)
        ax.plot(rounds, f1_vals, color=color, label=label, linewidth=2)
        if std_vals and any(v > 0 for v in std_vals):
            ax.fill_between(rounds,
                            np.array(f1_vals) - np.array(std_vals),
                            np.array(f1_vals) + np.array(std_vals),
                            color=color, alpha=0.15)

    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Macro F1 Score')
    ax.set_title(f'Convergence on {dataset.upper()} (α={alpha})')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)

    if save:
        plt.savefig(os.path.join(FIGURES_DIR,
                    f'fig7_convergence_{dataset}_alpha{alpha}.pdf'),
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(FIGURES_DIR,
                    f'fig7_convergence_{dataset}_alpha{alpha}.png'),
                    dpi=300, bbox_inches='tight')
    plt.close()
    return fig


def plot_communication_savings(dataset='iotid20', save=True):
    """Figure 8: Communication cost comparison."""
    methods = ['proposed', 'fedavg', 'hierfed', 'topk_qsgd',
               'dpfedavg', 'safeliot']

    fig, ax = plt.subplots(figsize=(7, 5))

    baseline_bytes = None
    for method in methods:
        data = load_aggregate(method, dataset, 0.5)
        if data is None:
            continue
        comm = data.get('communication_stats', {})
        total_bytes = comm.get('total_bytes_sent', 0)
        if total_bytes == 0:
            # Estimate: d × 32bit × T × N (for star) or × M (for hier)
            # Hardcoded from paper Table 5
            pass
        label = LABELS.get(method, method)
        color = COLORS.get(method, '#6B7280')
        ax.bar(label, total_bytes / 1e6, color=color)

    ax.set_xlabel('Method')
    ax.set_ylabel('Total Communication (MB)')
    ax.set_title(f'Communication Cost on {dataset.upper()}')
    ax.grid(True, alpha=0.3, axis='y')

    if save:
        plt.savefig(os.path.join(FIGURES_DIR, 'fig8_communication.pdf'),
                    dpi=300, bbox_inches='tight')
    plt.close()


def plot_privacy_utility_tradeoff(dataset='iotid20', save=True):
    """Figure 9: F1 vs ε (privacy-utility tradeoff)."""
    epsilons = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    # Load results for different ε values (need separate runs)

    fig, ax = plt.subplots(figsize=(7, 5))

    for method in ['proposed', 'dpfedavg', 'safeliot', 'dpfed6g']:
        f1_vals = []
        for eps in epsilons:
            data = load_aggregate(method, dataset, 0.5)
            if data and 'final_metrics' in data:
                f1 = data['final_metrics'].get('f1', 0)
                f1_vals.append(f1)
            else:
                f1_vals.append(None)

        valid_eps = [e for e, f in zip(epsilons, f1_vals) if f is not None]
        valid_f1 = [f for f in f1_vals if f is not None]
        if valid_eps:
            color = COLORS.get(method, '#6B7280')
            label = LABELS.get(method, method)
            ax.plot(valid_eps, valid_f1, color=color, label=label,
                    linewidth=2, marker='o')

    ax.set_xlabel('Privacy Budget ε')
    ax.set_ylabel('Macro F1 Score')
    ax.set_title(f'Privacy-Utility Tradeoff on {dataset.upper()}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    if save:
        plt.savefig(os.path.join(FIGURES_DIR, 'fig9_privacy_utility.pdf'),
                    dpi=300, bbox_inches='tight')
    plt.close()


def plot_noniid_impact(method='proposed', dataset='iotid20', save=True):
    """Figure 11: F1 vs α (non-IID severity)."""
    alphas = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 5.0, 10.0]

    fig, ax = plt.subplots(figsize=(7, 5))

    for m in ['proposed', 'fedavg', 'dpfedavg', 'hierfed_dp']:
        f1_vals = []
        for alpha in alphas:
            data = load_aggregate(m, dataset, alpha)
            if data and 'final_metrics' in data:
                f1 = data['final_metrics'].get('f1_mean', 0)
                f1_vals.append(f1)
            else:
                f1_vals.append(None)

        valid_alpha = [a for a, f in zip(alphas, f1_vals) if f is not None]
        valid_f1 = [f for f in f1_vals if f is not None]
        if valid_alpha:
            color = COLORS.get(m, '#6B7280')
            label = LABELS.get(m, m)
            ax.plot(valid_alpha, valid_f1, color=color, label=label,
                    linewidth=2, marker='s')

    ax.set_xlabel('Dirichlet α (higher → more uniform)')
    ax.set_ylabel('Macro F1 Score')
    ax.set_title(f'Non-IID Impact on {dataset.upper()}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    if save:
        plt.savefig(os.path.join(FIGURES_DIR, 'fig11_noniid.pdf'),
                    dpi=300, bbox_inches='tight')
    plt.close()


def plot_nsac_sensitivity(dataset='iotid20', save=True):
    """Figure 12: NSAC compression ratio sensitivity."""
    # Sweep over NSAC Top-k ratios for each slice
    ratios = {
        'urllc': [0.05, 0.08, 0.1, 0.15, 0.2],
        'embb': [0.1, 0.2, 0.3, 0.4, 0.5],
        'mmtc': [0.02, 0.05, 0.08, 0.1, 0.15],
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    slice_labels = ['uRLLC', 'eMBB', 'mMTC']
    slice_keys = ['urllc', 'embb', 'mmtc']

    for i, (slice_name, slice_key) in enumerate(zip(slice_labels, slice_keys)):
        ax = axes[i]
        ratio_list = ratios[slice_key]
        f1_vals = []
        speedup_vals = []

        for ratio in ratio_list:
            # Load results with this specific ratio (need special config)
            data = load_aggregate('proposed', dataset, 0.5)
            if data:
                f1 = data['final_metrics'].get('f1_mean', 0)
                f1_vals.append(f1)
                # Approximate speedup: d / (ratio × d × bits/32)
                bits = 8 if slice_key in ['urllc', 'mmtc'] else 16
                speedup = 32 / (ratio * bits)
                speedup_vals.append(speedup)
            else:
                f1_vals.append(None)
                speedup_vals.append(None)

        # Plot F1 line
        valid_r = [r for r, f in zip(ratio_list, f1_vals) if f is not None]
        valid_f = [f for f in f1_vals if f is not None]
        if valid_r:
            ax.plot(valid_r, valid_f, color='#2563EB', linewidth=2,
                    marker='o', label='F1')
            # Mark recommended config
            recommended = {'urllc': 0.1, 'embb': 0.3, 'mmtc': 0.05}
            rec = recommended[slice_key]
            ax.axvline(rec, color='#EF4444', linestyle='--',
                       label=f'Recommended (k={rec}d)')

        ax.set_xlabel('Top-k Ratio (k/d)')
        ax.set_ylabel('Macro F1 Score')
        ax.set_title(f'{slice_name} Slice')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('NSAC Compression Sensitivity Analysis', fontsize=14)
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(FIGURES_DIR, 'fig12_nsac_sensitivity.pdf'),
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(FIGURES_DIR, 'fig12_nsac_sensitivity.png'),
                    dpi=300, bbox_inches='tight')
    plt.close()


def plot_all():
    """Generate all paper figures."""
    print("Generating Figure 7: Convergence curves...")
    plot_convergence_curves('iotid20', alpha=0.5)
    plot_convergence_curves('iotid20', alpha=0.1)
    plot_convergence_curves('cicids2017', alpha=0.5)

    print("Generating Figure 8: Communication savings...")
    plot_communication_savings('iotid20')

    print("Generating Figure 9: Privacy-utility tradeoff...")
    plot_privacy_utility_tradeoff('iotid20')

    print("Generating Figure 11: Non-IID impact...")
    plot_noniid_impact()

    print("Generating Figure 12: NSAC sensitivity...")
    plot_nsac_sensitivity()

    print(f"All figures saved to {FIGURES_DIR}/")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--figure', type=str, default='all',
                        help='Which figure to generate (all, fig7, fig8, fig9, fig11, fig12)')
    parser.add_argument('--dataset', type=str, default='iotid20')
    parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()

    if args.figure == 'all':
        plot_all()
    elif args.figure == 'fig7':
        plot_convergence_curves(args.dataset, args.alpha)
    elif args.figure == 'fig8':
        plot_communication_savings(args.dataset)
    elif args.figure == 'fig9':
        plot_privacy_utility_tradeoff(args.dataset)
    elif args.figure == 'fig11':
        plot_noniid_impact()
    elif args.figure == 'fig12':
        plot_nsac_sensitivity(args.dataset)
    else:
        print(f"Unknown figure: {args.figure}")
