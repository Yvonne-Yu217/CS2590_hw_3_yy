# HW4 Submission Checklist

## 1) Required Programming Files

### Part 1
- `out_original.txt`
- `out_transformed.txt`
- `out_augmented_original.txt`
- `out_augmented_transformed.txt`

### Part 2
- `results/t5_ft_experiment_test.sql`
- `records/t5_ft_experiment_test.pkl`

## 2) Required Written PDF
- `hw4_report_template/hw4-report.pdf`

## 3) Required Links in PDF
- GitHub repository link (Part 1 + Part 2 code)
- Google Drive link to the model checkpoint used for Q7 outputs

## 4) Critical Part 2 Validation
Run in `hw4_original/part-2` on HPC:

```bash
wc -l data/test.nl
wc -l results/t5_ft_experiment_test.sql
```

These two counts must match exactly.

If SQL has one extra trailing blank line, fix with:

```bash
sed -i '${/^$/d;}' results/t5_ft_experiment_test.sql
```

## 5) Copy final enhanced model outputs to submission names

```bash
cp results/t5_ft_v6_enhanced_smoke_test.sql results/t5_ft_experiment_test.sql
cp records/t5_ft_v6_enhanced_smoke_test.pkl records/t5_ft_experiment_test.pkl
```
