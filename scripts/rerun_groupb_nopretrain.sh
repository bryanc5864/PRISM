#!/bin/bash
# Rerun Group B systems with unified config (12L/512d/512ff, gene_vocab=60697)
# NO pretraining — just the architecture change to match Group A

set -e

export LD_LIBRARY_PATH=$(python3 -c "import site,os; print(os.path.join(site.getsitepackages()[0],'nvidia','cusparselt','lib'))"):/home/bcheng/.conda/pkgs/libstdcxx-15.2.0-h39759b7_7/lib/:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

RESULTS_FILE="checkpoints/groupb_rerun.txt"

echo "============================================================" | tee $RESULTS_FILE
echo "  Group B Rerun: unified config, NO pretraining" | tee -a $RESULTS_FILE
echo "  Config: 12L/512d/512ff/gene_vocab=60697" | tee -a $RESULTS_FILE
echo "  Date: $(date)" | tee -a $RESULTS_FILE
echo "============================================================" | tee -a $RESULTS_FILE

SYSTEMS=(
    thcell
    neural_crest
    cardiac
    intestine
    lung
    oligo
    tirosh_melanoma
)

for system in "${SYSTEMS[@]}"; do
    echo "" | tee -a $RESULTS_FILE
    echo "======================================" | tee -a $RESULTS_FILE
    echo "  System: $system" | tee -a $RESULTS_FILE
    echo "  Started: $(date)" | tee -a $RESULTS_FILE
    echo "======================================" | tee -a $RESULTS_FILE

    python3 run_prism.py \
        --system "configs/${system}.yaml" \
        --stage train \
        2>&1 | tee -a $RESULTS_FILE

    python3 run_prism.py \
        --system "configs/${system}.yaml" \
        --stage baselines \
        2>&1 | tee -a $RESULTS_FILE

    echo "  Finished: $(date)" | tee -a $RESULTS_FILE
done

echo "" | tee -a $RESULTS_FILE
echo "============================================================" | tee -a $RESULTS_FILE
echo "  All systems complete: $(date)" | tee -a $RESULTS_FILE
echo "============================================================" | tee -a $RESULTS_FILE
