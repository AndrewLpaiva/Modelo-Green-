import os
import sys
import shutil
import tempfile

# ensure project root on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from exp.exp_long_term_forecasting import _create_versioned_dirs_and_move_log


def test_increment_from_existing_version(tmp_path):
    base_model = tmp_path / 'modelos' / 'setting'
    base_logs = tmp_path / 'logs' / 'setting'
    # create existing v01 and v02
    v1 = base_logs / 'v01'
    v2 = base_logs / 'v02'
    v1.mkdir(parents=True)
    v2.mkdir(parents=True)

    # tmp log
    tmp_dir = base_logs / 'tmp'
    tmp_dir.mkdir(parents=True)
    tmp_log = tmp_dir / 'train_log.txt'
    tmp_log.write_text('line')

    model_dir, logs_dir, final_log = _create_versioned_dirs_and_move_log(str(base_model), str(base_logs), str(tmp_log))

    # expect v03 created
    assert os.path.basename(logs_dir).startswith('v')
    assert int(os.path.basename(logs_dir)[1:]) == 3
    assert os.path.exists(final_log)
    assert 'line' in open(final_log, encoding='utf-8').read()
