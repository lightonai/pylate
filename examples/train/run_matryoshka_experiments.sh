#!/bin/bash
# Run all three matryoshka doc token selection experiments sequentially.
# Each trains a GTE-ModernColBERT model with a different token selection strategy.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# echo "============================================================"
# echo "Experiment 1/3: MatryoshkaImportanceLoss (Learned Importance + STE)"
# echo "============================================================"
# python "$SCRIPT_DIR/gte_modern_colbert_matryoshka_importance.py"

# echo ""
# echo "============================================================"
# echo "Experiment 2/3: MatryoshkaSoftTopKLoss (Soft Top-K Gating)"
# echo "============================================================"
# python "$SCRIPT_DIR/gte_modern_colbert_matryoshka_soft_topk.py"

echo ""
echo "============================================================"
echo "Experiment 3/3: MatryoshkaHierarchicalPoolingLoss (Hierarchical Pooling)"
echo "============================================================"
python "$SCRIPT_DIR/gte_modern_colbert_matryoshka_hierarchical_pooling.py"

echo ""
echo "============================================================"
echo "All experiments complete."
echo "============================================================"
