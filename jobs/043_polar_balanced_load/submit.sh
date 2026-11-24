#!/bin/bash
# Submit at the repository root. Each dependent array reads the preceding pilot round.
set -euo pipefail
job=jobs/043_polar_balanced_load/043_load.sh
test -f "$job"
id=$(qsub -J 1-48 -N ura043_p0 -v LOAD_PHASE=pilot,LOAD_ROUND=0 "$job")
echo "Pilot 0 $id"
for round in 1 2 3; do
    id=$(qsub -W "depend=afterok:$id" -J 1-48 -N "ura043_p$round" -v "LOAD_PHASE=pilot,LOAD_ROUND=$round" "$job")
    echo "Pilot $round $id"
done
id=$(qsub -W "depend=afterok:$id" -J 1-768 -N ura043_confirm -v LOAD_PHASE=confirm "$job")
echo "Confirmation $id"
id=$(qsub -W "depend=afterok:$id" -N ura043_plot -v LOAD_PHASE=merge "$job")
echo "Figure $id"
