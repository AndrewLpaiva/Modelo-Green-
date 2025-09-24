"""Continue training from a checkpoint stored under modelos_treinados.

Usage:
  python .\scripts\continue_from_modelos_treinados.py --dataset 2004 --ckpt .\modelos_treinados\modelo-treinado-Green-2003-2024-4\v02\checkpoint_full.pth --epochs 2

The script will:
  - read ./data/<dataset>_clean.csv to determine number of feature columns (excluding `date`)
  - build a `python run.py` command that sets `--enc_in/--dec_in/--c_out` accordingly
  - pass `--ckpt_path` so the experiment loads the provided checkpoint as warm start
  - run training for the requested number of epochs
"""

import argparse
import csv
import os
import subprocess
import sys


def detect_feature_count(data_file_path: str) -> int:
    # read header and count columns excluding 'date'
    with open(data_file_path, 'r', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        header = next(reader)
    # handle semicolon-separated files
    if len(header) == 1 and ';' in header[0]:
        header = header[0].split(';')
    header = [h.strip().lower() for h in header]
    if 'date' in header:
        return len(header) - 1
    # fallback: assume all columns are features
    return len(header)


def detect_target_column(data_file_path: str) -> str:
    # return a reasonable target name present in the CSV header
    with open(data_file_path, 'r', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        header = next(reader)
    if len(header) == 1 and ';' in header[0]:
        header = header[0].split(';')
    header_clean = [h.strip() for h in header]
    # prefer any column containing 'temp' (case-insensitive)
    for h in header_clean:
        if 'temp' in h.lower():
            return h
    # else, prefer 'target' or last column
    for h in header_clean:
        if h.lower() == 'target' or h.lower() == 'ot':
            return h
    return header_clean[-1]


def build_and_run(dataset: str, ckpt: str, epochs: int, use_gpu: bool, batch_size: int = 16):
    data_file = os.path.abspath(os.path.join('data', f"{dataset}_clean.csv"))
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return 2

    feat_count = detect_feature_count(data_file)
    target_col = detect_target_column(data_file)
    print(f"Detected {feat_count} feature columns in {data_file}; selected target: {target_col}")

    # construct run.py command
    cmd = [sys.executable, os.path.join('.', 'run.py'),
           '--is_training', '1',
           '--root_path', os.path.join('.', 'data'),
           '--data_path', os.path.basename(data_file),
           '--data', 'custom',
        '--features', 'M' if feat_count > 1 else 'S',
           '--enc_in', str(feat_count),
           '--dec_in', str(feat_count),
           '--c_out', str(feat_count),
        '--target', target_col,
        '--model', 'Pyraformer',
        '--train_epochs', str(epochs),
        '--batch_size', str(batch_size),
           '--ckpt_path', ckpt,
           '--use_gpu', 'True' if use_gpu else 'False']

    # keep other args conservative: reuse defaults for model, seq/label/pred lengths
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='dataset year label, e.g. 2004')
    parser.add_argument('--ckpt', required=True, help='path to checkpoint_full.pth to warm-start from')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16, help='batch size to use for training (default 16)')
    parser.add_argument('--use-gpu', action='store_true', dest='use_gpu', default=False)
    args = parser.parse_args(argv)

    return build_and_run(args.dataset, args.ckpt, args.epochs, args.use_gpu, args.batch_size)


if __name__ == '__main__':
    raise SystemExit(main())
"""Continue training from a checkpoint saved under modelos_treinados.

Usage:
  python .\scripts\continue_from_modelos_treinados.py --ckpt .\modelos_treinados\modelo-treinado-Green-2003-2024-4\v02\checkpoint_full.pth --data 2024_clean.csv --epochs 2 --use_simple_name

This script builds a `run.py` command that uses the provided checkpoint as an
initialization and continues training on the provided dataset.
"""
import argparse
import os
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='Path to checkpoint file (checkpoint_full.pth)')
    parser.add_argument('--data', required=True, help='Data filename under ./data/ (e.g. 2024_clean.csv)')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to continue training')
    # default to using simplified naming; the script will infer nice_prev/nice_new
    parser.add_argument('--use_simple_name', action='store_true', dest='use_simple_name', default=True, help='Enable simplified naming flags (default True)')
    parser.add_argument('--nice_prev', type=str, default='', help='(optional) Nice prev label for simplified naming; auto-detected from checkpoint if empty')
    parser.add_argument('--nice_new', type=str, default='', help='(optional) Nice new label for simplified naming; auto-detected from dataset if empty')
    args = parser.parse_args()

    # Resolve absolute paths
    ckpt = os.path.abspath(args.ckpt)
    if not os.path.exists(ckpt):
        print('Checkpoint not found:', ckpt)
        return 2

    data_path = args.data
    if not os.path.exists(os.path.join(os.getcwd(), 'data', data_path)):
        print('Data file not found in ./data/:', data_path)
        return 2

    # Build run.py command
    # attempt to infer feature count (number of columns minus the date column)
    enc_channels = 7
    try:
        csv_path = os.path.join(os.getcwd(), 'data', data_path)
        with open(csv_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split(',')
            # assume first column is `date`
            if len(header) > 1 and header[0].lower() == 'date':
                enc_channels = len(header) - 1
    except Exception:
        # fallback to 7 if anything goes wrong
        enc_channels = 7

    cmd = [sys.executable, os.path.join(os.getcwd(), 'run.py'),
        '--is_training', '1',
        '--root_path', os.path.join(os.getcwd(), 'data'),
        '--data_path', data_path,
        '--data', 'custom',
           '--task_name', 'long_term_forecast',
           '--model', 'Pyraformer',
           '--model_id', 'Green',
           '--features', 'M',
           '--target', 'temp_ar_bul_sec_hr',
           '--enc_in', str(enc_channels), '--dec_in', str(enc_channels), '--c_out', str(enc_channels),
           '--seq_len', '336', '--label_len', '168', '--pred_len', '168',
           '--train_epochs', str(args.epochs),
           '--ckpt_path', ckpt,
           '--use_gpu', 'False'
           ]

    # infer nice_new from dataset filename if not provided (e.g., 2004_clean.csv -> 2004)
    if args.nice_new:
        nice_new = args.nice_new
    else:
        try:
            nice_new = os.path.basename(data_path).split('_')[0]
        except Exception:
            nice_new = ''

    # infer nice_prev from checkpoint parent folder if not provided
    if args.nice_prev:
        nice_prev = args.nice_prev
    else:
        # attempt to extract a year-like token from the checkpoint path
        # e.g. '\\modelo-treinado-Green-2003-2024-4\\v02\\checkpoint_full.pth' -> '2003'
        ckpt_parts = os.path.normpath(ckpt).split(os.sep)
        nice_prev = ''
        for p in ckpt_parts[::-1]:
            if 'modelo-treinado' in p or 'modelo-treinado' in os.path.basename(p):
                # try to find tokens like 2003-2024 or 2003
                parts = p.replace('modelo-treinado-', '').split('-')
                for tok in parts:
                    if tok.isdigit() and len(tok) == 4:
                        nice_prev = tok
                        break
                if nice_prev:
                    break
        # fallback: try earlier path components for a 4-digit token
        if not nice_prev:
            for p in ckpt_parts[::-1]:
                if p.isdigit() and len(p) == 4:
                    nice_prev = p
                    break

    if args.use_simple_name:
        cmd += ['--simple_name', '--nice_prev', nice_prev, '--nice_new', nice_new]

    print('Running:', ' '.join(cmd))
    proc = subprocess.run(cmd)
    return proc.returncode

if __name__ == '__main__':
    raise SystemExit(main())
