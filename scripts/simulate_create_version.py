import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from exp.exp_long_term_forecasting import _create_versioned_dirs_and_move_log

def main():
    setting = 'sim_versioning'
    base_model = os.path.join('modelos_treinados', setting)
    base_logs = os.path.join('logs', setting)

    # create mismatched existing versions: modelos has v01, logs has v02
    os.makedirs(os.path.join(base_model, 'v01'), exist_ok=True)
    os.makedirs(os.path.join(base_logs, 'v02'), exist_ok=True)

    tmp_dir = os.path.join(base_logs, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_log = os.path.join(tmp_dir, 'train_log.txt')
    with open(tmp_log, 'w', encoding='utf-8') as f:
        f.write('simulated temporary log line\n')

    model_dir, logs_dir, final_log = _create_versioned_dirs_and_move_log(base_model, base_logs, tmp_log)

    print('model_dir:', model_dir)
    print('logs_dir:', logs_dir)
    print('final_log:', final_log)
    if os.path.exists(final_log):
        with open(final_log, 'r', encoding='utf-8') as f:
            print('final_log content:')
            print(f.read())
    else:
        print('final_log not found')

if __name__ == '__main__':
    main()
