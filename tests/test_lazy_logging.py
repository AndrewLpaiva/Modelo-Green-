import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from types import SimpleNamespace


def small_dataset_loader(seq_len=10, pred_len=2, batch_size=1):
    # create tiny tensors shaped like (batch, seq, features)
    x = torch.randn(4, seq_len, 1)
    y = torch.randn(4, pred_len, 1)
    x_mark = torch.zeros(4, seq_len, 1)
    y_mark = torch.zeros(4, pred_len, 1)
    ds = [(x[i:i+1], y[i:i+1], x_mark[i:i+1], y_mark[i:i+1]) for i in range(4)]
    # DataLoader that yields (batch_x, batch_y, batch_x_mark, batch_y_mark)
    return ds


def test_train_lazy_logs(tmp_path, monkeypatch):
    import sys
    import os
    # ensure project root is on sys.path for imports
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from exp.exp_basic import _load_available_models as real_loader
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    import exp.exp_basic as exp_basic
    import torch.nn as nn

    # stub out heavy model loading to avoid importing large model modules
    class DummyModel(nn.Module):
        def __init__(self, args=None):
            super().__init__()
            # simple learnable parameter to satisfy optimizer
            self.proj = nn.Linear(1, 1)
        def forward(self, x, x_mark, dec_inp, y_mark):
            # return zeros shaped like (batch, seq, features) using the proj layer
            b = x.shape[0]
            seq = x.shape[1]
            # apply proj to last feature of each timestep
            inp = x[..., :1]
            out = self.proj(inp.reshape(-1, 1)).reshape(b, seq, 1)
            return out

    # replace loader to register only Pyraformer -> DummyModel
    monkeypatch.setattr(exp_basic, '_load_available_models', lambda: {'Pyraformer': SimpleNamespace(Model=lambda a: DummyModel())})

    # minimal args
    args = SimpleNamespace()
    args.task_name = 'long_term_forecast'
    args.is_training = 1
    args.model_id = 'test'
    args.model = 'Pyraformer'
    args.data = 'custom'
    args.root_path = './'
    args.data_path = 'data/2001_clean.csv'
    args.features = 'S'
    args.target = 'temp_ar_bul_sec_hr'
    args.freq = 'h'
    args.checkpoints = './checkpoints'
    args.seq_len = 10
    args.label_len = 4
    args.pred_len = 2
    args.enc_in = 1
    args.dec_in = 1
    args.c_out = 1
    args.d_model = 16
    args.n_heads = 2
    args.e_layers = 1
    args.d_layers = 1
    args.d_ff = 32
    args.factor = 1
    args.embed = 'timeF'
    args.distil = False
    args.des = 'test'
    args.num_workers = 0
    args.train_epochs = 1
    args.train_steps = 1
    args.batch_size = 1
    args.patience = 2
    args.use_amp = False
    args.use_gpu = False
    args.use_multi_gpu = False
    args.gpu = 0
    args.devices = '0'
    args.learning_rate = 0.001
    args.lradj = 'type1'
    args.output_attention = False

    # instantiate experiment (uses our stubbed _load_available_models)
    exp = Exp_Long_Term_Forecast(args)

    # monkeypatch _get_data to return tiny datasets
    def fake_get_data(flag):
        ds = small_dataset_loader(seq_len=args.seq_len, pred_len=args.pred_len)
        return ds, ds

    monkeypatch.setattr(exp, '_get_data', lambda flag: fake_get_data(flag))

    setting = 'test_lazy'
    # run train (will create logs/tmp then move to v01)
    model = exp.train(setting)

    # assert that versioned dir was created
    logs_base = os.path.join('logs', setting)
    assert os.path.exists(logs_base)
    # there should be at least one vXX directory
    found = False
    for n in os.listdir(logs_base):
        if n.startswith('v'):
            found = True
            p = os.path.join(logs_base, n, 'train_log.txt')
            assert os.path.exists(p)
    assert found
