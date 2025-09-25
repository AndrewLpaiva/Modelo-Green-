"""Find the most recently modified checkpoint_full.pth under modelos_treinados/ and print details.

Usage:
    python .\scripts\find_latest_checkpoint_full.py
"""
import os
import glob
import datetime


def main():
    root = os.path.join(os.getcwd(), 'modelos_treinados')
    pattern = os.path.join(root, '**', 'checkpoint_full.pth')
    files = glob.glob(pattern, recursive=True)
    if not files:
        print('NO_FILES_FOUND')
        return 2
    files_sorted = sorted(files, key=lambda p: os.path.getmtime(p), reverse=True)
    latest = files_sorted[0]
    mtime = os.path.getmtime(latest)
    print('latest_checkpoint=' + latest)
    print('modified=' + datetime.datetime.fromtimestamp(mtime).isoformat())
    print('experiment_dir=' + os.path.dirname(latest))
    print('parent_experiment_dir=' + os.path.dirname(os.path.dirname(latest)))
    print('basename=' + os.path.basename(os.path.dirname(latest)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
