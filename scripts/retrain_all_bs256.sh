#!/bin/bash
# Retrain all systems that were NOT trained with bs=256
# Only skin was trained with bs=256 on 4 GPUs. Everything else needs retraining.
# Using GPUs 0,2,3 (GPU 1 in use by another process)

set -e

export CUDA_VISIBLE_DEVICES=0,2,3
export LD_LIBRARY_PATH=/home/bcheng/.conda/pkgs/libstdcxx-15.2.0-h39759b7_7/lib/:$(python -c "import site,os; print(os.path.join(site.getsitepackages()[0],'nvidia','cusparselt','lib'))"):$LD_LIBRARY_PATH

cd /home/bcheng/PRISM

SYSTEMS=(
    pancreas
    cortex
    hsc
    cardiac
    intestine
    lung
    neural_crest
    oligo
    thcell
    paul
    nestorowa
    sadefeldman
    tirosh_melanoma
    neftel_gbm
)

echo "============================================================"
echo "  Retraining ${#SYSTEMS[@]} systems with batch_size=256"
echo "  GPUs: 0,2,3 (3x A100 80GB)"
echo "  Started: $(date)"
echo "============================================================"
echo ""

for sys in "${SYSTEMS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  STARTING: $sys  ($(date))"
    echo "============================================================"

    python run_prism.py \
        --system "configs/${sys}.yaml" \
        --stage all \
        --batch-size 256 \
        --device cuda:0 \
        2>&1 | tee "logs/retrain_${sys}.log"

    echo ""
    echo "  FINISHED: $sys  ($(date))"
    echo "============================================================"
done

echo ""
echo "============================================================"
echo "  ALL RETRAINING COMPLETE  ($(date))"
echo "============================================================"
