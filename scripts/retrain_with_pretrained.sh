#!/bin/bash
# Retrain all 15 systems with PCP pre-trained weights and evaluate
cd /home/bcheng/PRISM
export CUDA_VISIBLE_DEVICES=1,2,3
export LD_LIBRARY_PATH=/home/bcheng/.conda/pkgs/libstdcxx-15.2.0-h39759b7_7/lib/:$(python -c "import site,os; print(os.path.join(site.getsitepackages()[0],'nvidia','cusparselt','lib'))"):$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================================"
echo "  Retraining with PCP pre-trained weights"
echo "  $(date)"
echo "  GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================================"

python -u scripts/retrain_with_pretrained.py

echo ""
echo "============================================================"
echo "  Complete: $(date)"
echo "============================================================"
