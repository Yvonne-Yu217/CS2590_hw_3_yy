#!/usr/bin/env python3
"""Inference-only script: loads a checkpoint and runs eval_epoch + test_inference."""
import sys
import os

# Patch main() to skip training
from train_t5 import eval_epoch, test_inference, get_args, DEVICE
from t5_utils import load_model_from_checkpoint
from load_data import load_t5_data

def main():
    args = get_args()

    print(f"Loading data...")
    _, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)

    print(f"Loading checkpoint: {args.experiment_name} (best=True)")
    model = load_model_from_checkpoint(args, best=True)
    model.eval()

    model_type = 'ft' if args.finetune else 'scr'

    # Dev set
    gt_sql_path = 'data/dev.sql'
    gt_record_path = 'records/dev_gt_records.pkl'
    model_sql_path = f'results/t5_{model_type}_{args.experiment_name}_dev.sql'
    model_record_path = f'records/t5_{model_type}_{args.experiment_name}_dev.pkl'

    print("Running dev evaluation...")
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(
        args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path
    )
    print(f"Dev: F1={dev_record_f1:.4f}, EM={dev_record_em:.4f}, SQL_EM={dev_sql_em:.4f}, ErrRate={dev_error_rate*100:.2f}%")

    # Test set
    model_sql_path = f'results/t5_{model_type}_{args.experiment_name}_test.sql'
    model_record_path = f'records/t5_{model_type}_{args.experiment_name}_test.pkl'

    print("Running test inference...")
    test_inference(args, model, test_loader, model_sql_path, model_record_path)
    print("Done! Results saved.")

if __name__ == "__main__":
    main()
