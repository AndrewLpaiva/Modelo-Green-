"""Continue training from a checkpoint saved under modelos_treinados.

This script builds a `run.py` command that uses the provided checkpoint as an
initialization and continues training on the provided dataset.

Usage example:
  python .\scripts\continue_from_modelos_treinados.py --ckpt .\modelos_treinados\modelo-treinado-Green-2022-14\v02\checkpoint_full.pth --data 2012_clean.csv --epochs 2 --use_simple_name
"""
import argparse
import glob
import os
import subprocess
import sys
import csv


def detect_feature_count_and_target(data_path: str):
    csv_path = os.path.join(os.getcwd(), 'data', data_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    with open(csv_path, 'r', encoding='utf-8') as f:
        header_line = f.readline().strip()

    # detect delimiter
    if ';' in header_line and header_line.count(';') >= header_line.count(','):
        header = [h.strip() for h in header_line.split(';') if h.strip()]
    elif ',' in header_line:
        header = [h.strip() for h in header_line.split(',') if h.strip()]
    else:
        header = [h.strip() for h in header_line.split() if h.strip()]

    header_lower = [h.lower() for h in header]
    # drop date/time-like columns
    non_time_cols = [h for (h, lh) in zip(header, header_lower) if not any(k in lh for k in ('date', 'data', 'hora', 'time', 'utc'))]
    feat_count = len(non_time_cols) if non_time_cols else (len(header) - 1 if len(header) > 1 else 7)

    # detect target column: prefer any column containing 'temp'
    target = None
    for h in header:
        if 'temp' in h.lower():
            target = h
            break
    if target is None:
        for h in header:
            if h.lower() in ('target', 'ot'):
                target = h
                break
    if target is None:
        target = non_time_cols[-1] if non_time_cols else header[-1]

    return feat_count, target


def auto_detect_ckpt():
    roots = [os.path.join(os.getcwd(), 'modelos_treinados')]
    candidates = []
    for r in roots:
        pattern = os.path.join(r, '**', 'checkpoint_full.pth')
        candidates.extend(glob.glob(pattern, recursive=True))
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=False, default='', help='Path to checkpoint_full.pth to warm-start from; if omitted the script will auto-detect the most recent checkpoint in modelos_treinados/')
    parser.add_argument('--data', required=True, help='Data filename under ./data/ (e.g. 2012_clean.csv)')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to continue training')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--use_simple_name', action='store_true', dest='use_simple_name', default=False, help='Enable simplified naming flags')
    parser.add_argument('--nice_prev', type=str, default='', help='(optional) Nice prev label for simplified naming')
    parser.add_argument('--nice_new', type=str, default='', help='(optional) Nice new label for simplified naming')
    parser.add_argument('--use_gpu', action='store_true', dest='use_gpu', default=False, help='Pass --use_gpu True to run.py')
    args = parser.parse_args()

    ckpt = args.ckpt or auto_detect_ckpt()
    if not ckpt:
        print('No checkpoint_full.pth found under modelos_treinados/. Please provide --ckpt')
        return 2
    ckpt = os.path.abspath(ckpt)
    if not os.path.exists(ckpt):
        print('Checkpoint not found:', ckpt)
        return 2

    data_path = args.data
    if not os.path.exists(os.path.join(os.getcwd(), 'data', data_path)):
        print('Data file not found in ./data/:', data_path)
        return 2

    # If file appears semicolon-separated, create a comma-converted temp copy
    csv_full = os.path.join(os.getcwd(), 'data', data_path)
    try:
        with open(csv_full, 'r', encoding='utf-8') as fh:
            sample = fh.readline()
    except Exception:
        sample = ''
    if ';' in sample and sample.count(';') >= sample.count(','):
        base, ext = os.path.splitext(data_path)
        conv_name = f"{base}_comma{ext}"
        conv_full = os.path.join(os.getcwd(), 'data', conv_name)
        if not os.path.exists(conv_full):
            print(f"Converting semicolon-delimited {data_path} -> {conv_name} for compatibility...")
            with open(csv_full, 'r', encoding='utf-8') as fr, open(conv_full, 'w', encoding='utf-8') as fw:
                for line in fr:
                    fw.write(line.replace(';', ','))
        data_path = conv_name

    feat_count, target_col = detect_feature_count_and_target(data_path)
    print(f'Detected {feat_count} feature columns in data/{data_path}; selected target: {target_col}')

    cmd = [sys.executable, os.path.join(os.getcwd(), 'run.py'),
           '--is_training', '1',
           '--root_path', os.path.join(os.getcwd(), 'data'),
           '--data_path', data_path,
           '--data', 'custom',
           '--task_name', 'long_term_forecast',
           '--model', 'Pyraformer',
           '--model_id', 'Green',
           '--features', 'M' if feat_count > 1 else 'S',
           '--target', target_col,
           '--enc_in', str(feat_count), '--dec_in', str(feat_count), '--c_out', str(feat_count),
           '--seq_len', '336', '--label_len', '168', '--pred_len', '168',
           '--train_epochs', str(args.epochs),
           '--batch_size', str(args.batch_size),
           '--ckpt_path', ckpt,
           '--use_gpu', 'True' if args.use_gpu else 'False']

    # infer nice_new from dataset filename if not provided (e.g., 2012_clean.csv -> 2012)
    nice_new = args.nice_new or os.path.basename(data_path).split('_')[0]

    # infer nice_prev from checkpoint parent folder if not provided
    nice_prev = args.nice_prev
    if not nice_prev:
        ckpt_parts = os.path.normpath(ckpt).split(os.sep)
        for p in ckpt_parts[::-1]:
            if 'modelo-treinado' in p:
                parts = p.replace('modelo-treinado-', '').split('-')
                for tok in parts:
                    if tok.isdigit() and len(tok) == 4:
                        nice_prev = tok
                        break
                if nice_prev:
                    break
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
    parser.add_argument('--ckpt', required=False, default='', help='(optional) path to checkpoint_full.pth to warm-start from; if omitted the script will auto-detect the most recent checkpoint in modelos_treinados/')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16, help='batch size to use for training (default 16)')
    parser.add_argument('--use-gpu', action='store_true', dest='use_gpu', default=False)
    args = parser.parse_args(argv)

    # If ckpt not provided, attempt to auto-detect the most recently modified checkpoint_full.pth
    ckpt = args.ckpt
    if not ckpt:
        import glob
        import os
        roots = [os.path.join(os.getcwd(), 'modelos_treinados')]
        candidates = []
        for r in roots:
            pattern = os.path.join(r, '**', 'checkpoint_full.pth')
            candidates.extend(glob.glob(pattern, recursive=True))
        if not candidates:
            print('No checkpoint_full.pth found under modelos_treinados/. Please provide --ckpt')
            return 2
        # choose most recently modified
        candidates = sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)
        ckpt = candidates[0]
        print('Auto-detected latest checkpoint for warm-start:', ckpt)

    return build_and_run(args.dataset, ckpt, args.epochs, args.use_gpu, args.batch_size)


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
            header_line = f.readline().strip()
            # detect delimiter (handle semicolon-separated files)
            if ';' in header_line and header_line.count(';') >= header_line.count(','):
                header = [h.strip() for h in header_line.split(';') if h.strip()]
            elif ',' in header_line:
                header = [h.strip() for h in header_line.split(',') if h.strip()]
            else:
                header = [h.strip() for h in header_line.split() if h.strip()]
            # remove obvious date/time columns (names like 'date','data','hora','time','utc')
            header_lower = [h.lower() for h in header]
            non_time_cols = [h for (h, lh) in zip(header, header_lower) if not any(k in lh for k in ('date', 'data', 'hora', 'time', 'utc'))]
            if len(non_time_cols) > 0:
                enc_channels = len(non_time_cols)
            else:
                # fallback: assume first column is date
                if len(header) > 1:
                    enc_channels = len(header) - 1
                else:
                    enc_channels = 7
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
