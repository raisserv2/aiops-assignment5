#!/usr/bin/env bash
# Builds Module4_<rollno>.zip in the layout of section 7 of the assignment (no data, models or databases).
#   bash make_submission.sh DA24B015
set -e
cd "$(dirname "$0")"
ROLL="${1:-DA24B015}"
[ -f REPORT.md ] || { echo "REPORT.md is missing"; exit 1; }
OUT="Module4_${ROLL}"
rm -rf "/tmp/$OUT" "$OUT.zip"
mkdir -p "/tmp/$OUT"/{partA,partB,partC,partD}
cp REPORT.md README.md requirements.txt eta_data.py load_test.py run_all.sh AI_DISCLOSURE.md "/tmp/$OUT/"
cp partA/eta_app.py partA/test_eta_app.py partA/loadtest_results.json partA/loadtest_results_2workers.json partA/loadtest_batch.json "/tmp/$OUT/partA/"
cp partB/eta_mlflow.py partB/test_gate.py partB/gate_output.txt partB/shadow_output.txt "/tmp/$OUT/partB/"
cp partC/eta_ray.py partC/verify_ray.py partC/verify_output.txt "/tmp/$OUT/partC/"
cp partD/smoke_eta.py partD/smoke_output.txt "/tmp/$OUT/partD/"
(cd /tmp && python -m zipfile -c "$OLDPWD/$OUT.zip" "$OUT/")
rm -rf "/tmp/$OUT"
python -m zipfile -l "$OUT.zip"
echo "-> $(pwd)/$OUT.zip"
