#!/bin/bash
# After retrain_all_bs256.sh finishes, re-run PRISM-only evaluation
# to pick up the new embeddings, then regenerate UMAPs

set -e

export LD_LIBRARY_PATH=/home/bcheng/.conda/pkgs/libstdcxx-15.2.0-h39759b7_7/lib/:$(python -c "import site,os; print(os.path.join(site.getsitepackages()[0],'nvidia','cusparselt','lib'))"):$LD_LIBRARY_PATH

cd /home/bcheng/PRISM

echo "============================================================"
echo "  Waiting for retraining to finish..."
echo "============================================================"

# Wait for retrain to complete by checking for the completion marker
while ! grep -q "ALL RETRAINING COMPLETE" logs/retrain_all.log 2>/dev/null; do
    sleep 60
done

echo "Retraining complete! Starting post-retraining evaluation..."
echo "$(date)"

# Wait for initial evaluation to finish too
while ! grep -q "^Done!" logs/full_evaluation.log 2>/dev/null; do
    sleep 60
done

echo "Initial evaluation complete! Running PRISM re-evaluation..."
echo "$(date)"

# Re-run full evaluation with updated PRISM embeddings
python -u scripts/run_all_evaluations.py > logs/full_evaluation_final.log 2>&1

echo "Final evaluation complete!"
echo "$(date)"

# Regenerate UMAP figures with new embeddings
python -u scripts/regenerate_umap_labels.py > logs/regenerate_umaps_final.log 2>&1

echo "UMAP regeneration complete!"
echo "$(date)"

echo "============================================================"
echo "  ALL DONE  $(date)"
echo "============================================================"
