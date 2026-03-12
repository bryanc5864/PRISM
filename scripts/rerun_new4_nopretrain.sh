#!/bin/bash
# Rerun the 4 newest systems without pretraining (their results were overwritten)
set -e

export LD_LIBRARY_PATH=$(python3 -c "import site,os; print(os.path.join(site.getsitepackages()[0],'nvidia','cusparselt','lib'))"):/home/bcheng/.conda/pkgs/libstdcxx-15.2.0-h39759b7_7/lib/:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

for system in neftel_gbm paul nestorowa sadefeldman; do
    echo "=== $system (no pretrain) === $(date)"
    python3 run_prism.py --system "configs/${system}.yaml" --stage train
    python3 run_prism.py --system "configs/${system}.yaml" --stage baselines
    echo "=== $system done === $(date)"
done
