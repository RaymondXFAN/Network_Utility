# HierFed-Matter-NSAC-DPBA: Open-Source Implementation

> Companion code for the paper: *Privacy-Utility Tradeoff via Dynamic Privacy Budgeting for Matter-Enabled Smart Home Anomaly Detection in B5G/6G Large-Scale Residential IoT*

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets (requires Kaggle CLI or manual download)
bash data/download_data.sh

# 3. Preprocess data
python data/preprocess.py --dataset both --alpha 0.5 --N 50 --seed 1

# 4. Run a single experiment
python run_experiment.py --dataset iotid20 --alpha 0.5 --seed 1 --method proposed

# 5. Run all experiments (5 seeds × 2 datasets × 2 alphas)
bash run_all_seeds.sh

# 6. Aggregate results for paper tables
python run_experiment.py --aggregate_only --method proposed --dataset both
```

## Architecture

```
HierFed-Matter-NSAC-DPBA = HierFed-Matter + NSAC + DPBA-FIM

┌─────────────┐     ┌──────────────────┐     ┌──────────┐
│  IoT Device │────▶│  Matter Gateway  │────▶│   Cloud  │
│  (local     │     │  (edge aggregate │     │ (global  │
│   training) │     │  + NSAC compress │     │ aggregate│
│   + DPBA    │     │  + DPBA noise)   │     │  server) │
└─────────────┘     └──────────────────┘     └──────────┘
```

### Three Core Components

1. **HierFed-Matter** (§4.1): Reuses Matter protocol's device→border-router→cloud topology for hierarchical FL aggregation. Core network traffic reduced from O(N) to O(M).

2. **NSAC** (§4.2): Network-Slice-Aware Compression adapts gradient compression to 6G slice QoS:
   - uRLLC: Top-k(0.1d) + 8-bit → 40× compression, latency ≤200 ms
   - eMBB: Top-k(0.3d) + 16-bit → ~7× compression
   - mMTC: Top-k(0.05d) + 8-bit → 80× compression

3. **DPBA-FIM** (§4.3): Dynamic Privacy Budget Allocation via Fisher Information Matrix:
   - FIM scalar EMA (β_F=0.9) for per-layer importance tracking
   - Device sensitivity: S_i = 0.5σ_i + 0.3(1/n_i) + 0.2H_i
   - Budget allocation: ε_{i,l}(t) = (ε_total/T) × (I_l/ΣI_l) × (S_i/ΣS_j)
   - (ε≤3, δ=10⁻⁵)-DP with only 2.1% accuracy loss

## Key Hyperparameters

| Parameter | Value | Note |
|-----------|-------|------|
| T (rounds) | 100 | ε_r = ε_total/T = 0.03 |
| K (devices/gateway) | 10 | N/K = M gateways |
| ε_total | 3.0 | Total DP budget |
| δ | 10⁻⁵ | DP failure probability |
| C (clip norm) | 1.0 | Gradient clipping bound |
| β_F | 0.9 | FIM EMA smoothing |
| η (learning rate) | 0.01 | **Paper does not specify; adjust if needed** |
| E_local (epochs) | 5 | **Paper does not specify; adjust if needed** |
| batch_size | 32 | **Paper does not specify; adjust if needed** |

⚠️ **η, E_local, and batch_size are not explicitly given in the paper.** The values above are reasonable defaults. If your original experiment used different values, update `configs/base.yaml`.

⚠️ **Parameter count discrepancy**: Paper claims d=12,498 but the described architecture (Input(80)→FC128→FC64→FC32→FC16→FC2) actually has d=21,266 (including biases). Code uses `model.num_parameters` for the true count. This affects NSAC's k_s calculation. Verify with your original code.

## Expected Results (Paper Targets)

| Dataset | Method | F1 (α=0.5) | Speedup | MIA Success |
|---------|--------|-------------|----------|-------------|
| IoTID20 | Proposed | 0.924 ± 0.006 | 19.3× | 14.2% |
| CICIDS2017 | Proposed | 0.891 ± 0.008 | — | — |
| IoTID20 | DP-FedAvg | 0.798 | 1× | 18.3% |
| IoTID20 | HierFed+DP | 0.915 | 10× | 15.7% |

## Baselines Implemented

| Method | Description |
|--------|-------------|
| FedAvg | Standard star-topology FL |
| FedProx | Proximal-regularized FL |
| HierFed | Hierarchical FL (no DP/NSAC) |
| DP-FedAvg | Star-topology + uniform DP (ε=3) |
| DP-Fed6G | Star-topology + adaptive DP |
| Top-k+QSGD | Star-topology + uniform compression |
| SAFEL-IoT | Adaptive DP (no FIM weighting) |
| FedProx-DP | FedProx + uniform DP |
| **HierFed+DP** | **Hierarchical + uniform DP** (key ablation) |

## Project Structure

```
├── README.md
├── requirements.txt
├── configs/         # YAML configs for hyperparameters
├── data/            # Download + preprocess scripts
├── models/          # 5-layer DNN
├── core/            # NSAC, DPBA-FIM, HierFed, Simulator
├── baselines/       # 9 baseline methods
├── run_experiment.py  # Main entry point
├── evaluate.py        # Metrics + MIA evaluation
├── run_all_seeds.sh   # Batch runner
└── 00_implementation_plan.md  # Detailed implementation spec
```

## Citation

If you use this code, please cite our paper:

```bibtex
@article{hierfed_matter_nsac_dpba_2024,
  title={Privacy-Utility Tradeoff via Dynamic Privacy Budgeting for Matter-Enabled Smart Home Anomaly Detection in B5G/6G Large-Scale Residential IoT},
  ...
}
```

## License

MIT License
