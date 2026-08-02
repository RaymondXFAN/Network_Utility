#!/bin/bash
# Run all experiments: 5 seeds × 2 datasets × 2 alphas × proposed method
# Total: 20 experiment runs (~2-6 hours each depending on hardware)

set -e
echo "Starting full experiment suite..."

# Step 1: Preprocess data (if not already done)
if [ ! -f "data/processed/iotid20_train.npz" ]; then
    echo "Preprocessing IoTID20..."
    python data/preprocess.py --dataset iotid20 --alpha 0.5 --N 50 --seed 1
fi

if [ ! -f "data/processed/cicids_train.npz" ]; then
    echo "Preprocessing CICIDS2017..."
    python data/preprocess.py --dataset cicids2017 --alpha 0.5 --N 50 --seed 1
fi

# Step 2: Run proposed method across all seeds and configs
SEEDS="1 2 3 4 5"
DATASETS="iotid20 cicids2017"
ALPHAS="0.5 0.1"

for dataset in $DATASETS; do
    for alpha in $ALPHAS; do
        for seed in $SEEDS; do
            # Preprocess for this alpha/seed combination
            python data/preprocess.py --dataset $dataset --alpha $alpha \
                --N 50 --seed $seed

            echo "Running proposed method: $dataset, α=$alpha, seed=$seed"
            python run_experiment.py \
                --dataset $dataset \
                --alpha $alpha \
                --seed $seed \
                --method proposed \
                --output_dir results
        done
    done
done

# Step 3: Run key baselines (optional; uncomment to run)
# for method in fedavg fedprox dpfedavg hierfed hierfed_dp topk_qsgd; do
#     for dataset in $DATASETS; do
#         for alpha in 0.5; do
#             for seed in $SEEDS; do
#                 echo "Running baseline: $method, $dataset, α=0.5, seed=$seed"
#                 python run_experiment.py \
#                     --dataset $dataset --alpha 0.5 --seed $seed \
#                     --method $method --output_dir results
#             done
#         done
#     done
# done

# Step 4: Aggregate results
echo "Aggregating results..."
python run_experiment.py --aggregate_only --method proposed \
    --dataset both --alpha 0.5 0.1 --output_dir results

echo "All experiments complete! Check results/ directory."
