import os
import sys
# ensure project root is on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from exp.exp_long_term_forecasting import _create_versioned_dirs_and_move_log


def test_move_tmp_log(tmp_path):
    base_model = tmp_path / 'modelos' / 'setting'
    base_logs = tmp_path / 'logs' / 'setting'
    base_model.mkdir(parents=True)
    (base_logs / 'tmp').mkdir(parents=True)

    tmp_log = base_logs / 'tmp' / 'train_log.txt'
    tmp_log.write_text('temporary line\n')

    model_dir, logs_dir, final_log = _create_versioned_dirs_and_move_log(str(base_model), str(base_logs), str(tmp_log))

    assert os.path.exists(model_dir)
    assert os.path.exists(logs_dir)
    assert os.path.exists(final_log)
    assert 'temporary line' in open(final_log, encoding='utf-8').read()
