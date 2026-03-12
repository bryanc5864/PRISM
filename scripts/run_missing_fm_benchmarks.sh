#!/bin/bash
set -e
cd /home/bcheng/PRISM

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(python -c "import site,os; print(os.path.join(site.getsitepackages()[0],'nvidia','cusparselt','lib'))"):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/bcheng/.conda/pkgs/libstdcxx-15.2.0-h39759b7_7/lib/:$LD_LIBRARY_PATH

echo "============================================================"
echo "  Foundation Model Benchmarks for Missing Systems"
echo "  $(date)"
echo "============================================================"

for sys in oligo paul nestorowa sadefeldman tirosh_melanoma neftel_gbm; do
    echo ""
    echo ">>> Starting $sys at $(date)"
    python -u benchmarks/run_foundation_benchmarks_cross_system.py --system $sys --gpu 1 2>&1
    echo ">>> Finished $sys at $(date)"
done

echo ""
echo "============================================================"
echo "  All FM benchmarks done. Now re-running full evaluation..."
echo "  $(date)"
echo "============================================================"

python -u scripts/run_all_evaluations.py 2>&1

echo ""
echo "============================================================"
echo "  ALL COMPLETE  $(date)"
echo "============================================================"
