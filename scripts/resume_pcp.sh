#!/bin/bash
cd /home/bcheng/PRISM
export CUDA_VISIBLE_DEVICES=1,2,3
export LD_LIBRARY_PATH=/home/bcheng/.conda/pkgs/libstdcxx-15.2.0-h39759b7_7/lib/:$(python -c "import site,os; print(os.path.join(site.getsitepackages()[0],'nvidia','cusparselt','lib'))"):$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================================"
echo "  Resuming PCP Pre-training"
echo "  $(date)"
echo "  GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================================"

python -u scripts/pretrain_pcp.py \
  --resume checkpoints/pretrain/pcp_epoch_1.pt \
  --n-epochs 10 \
  --perts-per-batch 24

echo ""
echo "============================================================"
echo "  PCP Pre-training complete: $(date)"
echo "============================================================"
