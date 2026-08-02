#!/bin/bash
# Download IoTID20 and CICIDS2017 datasets
# Run: bash data/download_data.sh

set -e
RAW_DIR="data/raw"
mkdir -p "$RAW_DIR"

echo "============================================================"
echo "  Downloading IoTID20 dataset"
echo "============================================================"
echo ""
echo "Official source: https://sites.google.com/view/iot-network-intrusion-dataset"
echo "Kaggle mirror:   https://www.kaggle.com/datasets/winthumin/iot-ids-preprocessed-datasets-win-thu"
echo ""
echo "Please manually download IoTID20.csv to $RAW_DIR/"
echo "  - From Kaggle: download and place as $RAW_DIR/IoTID20.csv"
echo "  - From official site: download CSV, rename to IoTID20.csv"
echo ""

# Check if Kaggle CLI is available for automated download
if command -v kaggle &>/dev/null; then
    echo "[Auto] Using Kaggle CLI to download IoTID20..."
    kaggle datasets download -d winthumin/iot-ids-preprocessed-datasets-win-thu \
        -p "$RAW_DIR" --unzip
    # Find the CSV and rename
    IOTID_FILE=$(find "$RAW_DIR" -name "*IoTID*" -name "*.csv" | head -1)
    if [ -n "$IOTID_FILE" ]; then
        mv "$IOTID_FILE" "$RAW_DIR/IoTID20.csv"
        echo "[OK] IoTID20.csv saved to $RAW_DIR/"
    fi
else
    echo "[Manual] Kaggle CLI not found. Download IoTID20.csv manually:"
    echo "  1. Visit https://www.kaggle.com/datasets/winthumin/iot-ids-preprocessed-datasets-win-thu"
    echo "  2. Download CSV file"
    echo "  3. Place as $RAW_DIR/IoTID20.csv"
fi

echo ""
echo "============================================================"
echo "  Downloading CICIDS2017 dataset"
echo "============================================================"
echo ""
echo "Official source: https://www.unb.ca/cic/datasets/ids-2017.html (需注册)"
echo "Kaggle mirror:   https://www.kaggle.com/datasets/sampadab17/cicids2017"
echo ""

if command -v kaggle &>/dev/null; then
    echo "[Auto] Using Kaggle CLI to download CICIDS2017..."
    mkdir -p "$RAW_DIR/CICIDS2017"
    kaggle datasets download -d sampadab17/cicids2017 \
        -p "$RAW_DIR/CICIDS2017" --unzip
    echo "[OK] CICIDS2017 files saved to $RAW_DIR/CICIDS2017/"
else
    echo "[Manual] Kaggle CLI not found. Download CICIDS2017 manually:"
    echo "  1. Visit https://www.kaggle.com/datasets/sampadab17/cicids2017"
    echo "  2. Download CSV files (Tuesday-WorkingHours..., Wednesday-WorkingHours...)"
    echo "  3. Place all CSV files in $RAW_DIR/CICIDS2017/"
fi

echo ""
echo "============================================================"
echo "  Verifying data files..."
echo "============================================================"

if [ -f "$RAW_DIR/IoTID20.csv" ]; then
    echo "[OK] IoTID20.csv found"
    ROWS=$(wc -l < "$RAW_DIR/IoTID20.csv")
    echo "     Rows: ~$ROWS (expected: ~625,784)"
else
    echo "[MISSING] IoTID20.csv not found in $RAW_DIR/"
fi

if [ -d "$RAW_DIR/CICIDS2017" ]; then
    CSV_COUNT=$(find "$RAW_DIR/CICIDS2017" -name "*.csv" | wc -l)
    echo "[OK] CICIDS2017 directory found with $CSV_COUNT CSV files"
else
    echo "[MISSING] CICIDS2017 directory not found in $RAW_DIR/"
fi

echo ""
echo "Next step: python data/preprocess.py --dataset both"
