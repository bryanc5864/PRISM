#!/bin/bash
# Evaluate scGPT-pretrained PCP on all 15 systems
# Runs train + baselines for each system with pretrained weights

set -e

export LD_LIBRARY_PATH=$(python3 -c "import site,os; print(os.path.join(site.getsitepackages()[0],'nvidia','cusparselt','lib'))"):/home/bcheng/.conda/pkgs/libstdcxx-15.2.0-h39759b7_7/lib/:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

PRETRAINED="checkpoints/pretrain_scgpt/pcp_final.pt"
RESULTS_FILE="checkpoints/pretrain_scgpt/downstream_results.txt"

echo "============================================================" | tee $RESULTS_FILE
echo "  Downstream Evaluation: scGPT-pretrained PCP" | tee -a $RESULTS_FILE
echo "  Checkpoint: $PRETRAINED" | tee -a $RESULTS_FILE
echo "  Date: $(date)" | tee -a $RESULTS_FILE
echo "============================================================" | tee -a $RESULTS_FILE

SYSTEMS=(
    skin
    pancreas
    cortex
    hsc
    thcell
    neural_crest
    cardiac
    intestine
    lung
    oligo
    tirosh_melanoma
    neftel_gbm
    paul
    nestorowa
    sadefeldman
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
        --pretrained "$PRETRAINED" \
        2>&1 | tee -a $RESULTS_FILE

    # Run baselines for comparison
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
