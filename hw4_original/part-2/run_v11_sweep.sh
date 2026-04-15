#!/usr/bin/env bash
set -euo pipefail

rm -f sweep_summary.txt

configs=(
  "v11_lr1e5_linear 1e-5 linear"
  "v11_lr2e5_linear 2e-5 linear"
  "v11_lr2e5_cosine 2e-5 cosine"
  "v11_lr3e5_linear 3e-5 linear"
)

for cfg in "${configs[@]}"; do
  read -r exp lr sch <<< "$cfg"

  echo "==== RUN $exp (lr=$lr, scheduler=$sch) ===="
  python train_t5.py \
    --finetune \
    --experiment_name "$exp" \
    --optimizer_type AdamW \
    --learning_rate "$lr" \
    --scheduler_type "$sch" \
    --num_warmup_epochs 1 \
    --max_n_epochs 8 \
    --patience_epochs 2 \
    --batch_size 16 \
    --test_batch_size 16

  f1=$(python [evaluate.py](http://_vscodecontentref_/0) \
    -ps "results/t5_ft_${exp}_dev.sql" \
    -pr "records/t5_ft_${exp}_dev.pkl" \
    -ds "data/dev.sql" \
    -dr "records/dev_gt_records.pkl" | awk '{print $3}')

  echo "$exp $f1" | tee -a sweep_summary.txt
done

echo "==== DEV F1 SUMMARY ===="
sort -k2 -nr sweep_summary.txt

best_exp=$(sort -k2 -nr sweep_summary.txt | head -n1 | awk '{print $1}')
echo "BEST EXP: $best_exp"

cp "results/t5_ft_${best_exp}_test.sql" results/t5_ft_experiment_test.sql
cp "records/t5_ft_${best_exp}_test.pkl" records/t5_ft_experiment_test.pkl

python - <<'PY'
from pathlib import Path
t = [x for x in Path("data/test.nl").read_text().splitlines() if x.strip()]
s = [x for x in Path("results/t5_ft_experiment_test.sql").read_text().splitlines() if x.strip()]
unbal = sum(1 for l in s if l.count("(") != l.count(")"))
print("test.nl lines:", len(t))
print("test.sql lines:", len(s))
print("aligned:", len(t) == len(s))
print("unbalanced parens:", unbal)
PY

ls -lh results/t5_ft_experiment_test.sql records/t5_ft_experiment_test.pkl
