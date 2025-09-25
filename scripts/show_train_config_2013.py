"""Build and print the training command/config that would continue the latest model
using data/2013_clean.csv. This script does NOT run training; it only shows all args.

Usage:
  python .\scripts\show_train_config_2013.py
"""
import os
import glob
import sys
import csv


def find_latest_ckpt():
    root = os.path.join(os.getcwd(), 'modelos_treinados')
    pattern = os.path.join(root, '**', 'checkpoint_full.pth')
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    files_sorted = sorted(files, key=lambda p: os.path.getmtime(p), reverse=True)
    return files_sorted[0]


def detect_feat_and_target(data_relpath):
    p = os.path.join(os.getcwd(), 'data', data_relpath)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    with open(p, 'r', encoding='utf-8') as fh:
        header_line = fh.readline().strip()
    # detect delimiter
    if ';' in header_line and header_line.count(';') >= header_line.count(','):
        header = [h.strip() for h in header_line.split(';') if h.strip()]
    elif ',' in header_line:
        header = [h.strip() for h in header_line.split(',') if h.strip()]
    else:
        header = [h.strip() for h in header_line.split() if h.strip()]
    header_lower = [h.lower() for h in header]
    non_time_cols = [h for (h, lh) in zip(header, header_lower) if not any(k in lh for k in ('date', 'data', 'hora', 'time', 'utc'))]
    feat_count = len(non_time_cols) if non_time_cols else (len(header) - 1 if len(header) > 1 else 7)
    # detect target
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


def infer_nice_prev(ckpt_path):
    if not ckpt_path:
        return ''
    parts = os.path.normpath(ckpt_path).split(os.sep)
    nice_prev = ''
    for p in parts[::-1]:
        if 'modelo-treinado' in p:
            toks = p.replace('modelo-treinado-', '').split('-')
            for tok in toks:
                if tok.isdigit() and len(tok) == 4:
                    nice_prev = tok
                    break
            if nice_prev:
                break
    if not nice_prev:
        for p in parts[::-1]:
            if p.isdigit() and len(p) == 4:
                nice_prev = p
                break
    return nice_prev


def build_cmd(ckpt, data_file, epochs=2, batch_size=16, use_gpu=False, use_simple_name=True):
    feat_count, target = detect_feat_and_target(data_file)
    features_flag = 'M' if feat_count > 1 else 'S'
    cmd = [sys.executable, os.path.join(os.getcwd(), 'run.py'),
           '--is_training', '1',
           '--root_path', os.path.join(os.getcwd(), 'data'),
           '--data_path', data_file,
           '--data', 'custom',
           '--task_name', 'long_term_forecast',
           '--model', 'Pyraformer',
           '--model_id', 'Green',
           '--features', features_flag,
           '--target', target,
           '--enc_in', str(feat_count), '--dec_in', str(feat_count), '--c_out', str(feat_count),
           '--seq_len', '336', '--label_len', '168', '--pred_len', '168',
           '--train_epochs', str(epochs),
           '--batch_size', str(batch_size),
           '--ckpt_path', ckpt,
           '--use_gpu', 'True' if use_gpu else 'False']

    nice_new = os.path.basename(data_file).split('_')[0]
    nice_prev = infer_nice_prev(ckpt)
    if use_simple_name:
        cmd += ['--simple_name', '--nice_prev', nice_prev, '--nice_new', nice_new]
    return cmd, {
        'model': 'Pyraformer',
        'model_id': 'Green',
        'features': features_flag,
        'target': target,
        'enc_in': feat_count,
        'dec_in': feat_count,
        'c_out': feat_count,
        'seq_len': 336,
        'label_len': 168,
        'pred_len': 168,
        'train_epochs': epochs,
        'batch_size': batch_size,
        'use_gpu': use_gpu,
        'ckpt_path': ckpt,
        'nice_prev': nice_prev,
        'nice_new': nice_new,
    }


def main():
    data_file = '2013_clean.csv'
    if not os.path.exists(os.path.join('data', data_file)):
        print('Data file not found:', os.path.join('data', data_file))
        return 2
    ckpt = find_latest_ckpt()
    if not ckpt:
        print('No checkpoint_full.pth found under modelos_treinados/')
        return 2
    cmd, cfg = build_cmd(ckpt, data_file, epochs=2, batch_size=16, use_gpu=False, use_simple_name=True)

    print('\n=== Training configuration that WILL be used (dry-run) ===\n')
    for k, v in cfg.items():
        print(f'{k:12}: {v}')
    print('\nFull command:')
    print(' '.join(cmd))
    print('\nNote: this script only prints the command. To actually run training, run the wrapper or run.py with the above args.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
