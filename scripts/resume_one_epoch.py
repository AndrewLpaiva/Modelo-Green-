from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import sys

# Ensure project root is on sys.path so `import exp` and other project modules work
# when running this script directly from the repository root.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import os
import torch
from argparse import Namespace
import multiprocessing
# Minimal args to match previous run; adjust if you used different flags
args = Namespace(
    task_name='long_term_forecast',
    is_training=1,
    model_id='test',
    model='Pyraformer',
    data='custom',
    root_path='./',
    data_path='data/2001_clean.csv',
    features='S',
    target='temp_ar_bul_sec_hr',
    freq='h',
    checkpoints='./checkpoints/',
    seq_len=336,
    label_len=168,
    pred_len=168,
    seasonal_patterns='Yearly',
    inverse=False,
    mask_rate=0.25,
    anomaly_ratio=0.25,
    top_k=5,
    num_kernels=6,
    enc_in=1,
    dec_in=1,
    c_out=1,
    d_model=512,
    n_heads=8,
    e_layers=2,
    d_layers=1,
    d_ff=2048,
    moving_avg=25,
    factor=1,
    distil=True,
    dropout=0.1,
    embed='timeF',
    activation='gelu',
    output_attention=False,
    channel_independence=0,
    num_workers=2,
    itr=1,
    train_epochs=1,
    train_steps=10,
    val_steps=10000,
    batch_size=16,
    patience=3,
    learning_rate=0.0001,
    des='test',
    loss='MSE',
    lradj='type1',
    use_amp=False,
    use_gpu=False,
    gpu=0,
    use_multi_gpu=False,
    devices='0,1,2,3',
    p_hidden_dims=[128,128],
    p_hidden_layers=2
)

# locate existing checkpoint (legacy)
setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
    args.task_name,
    args.model,
    args.model_id,
    args.data,
    args.features,
    args.seq_len,
    args.label_len,
    args.pred_len,
    args.d_model,
    args.n_heads,
    args.e_layers,
    args.d_layers,
    args.d_ff,
    args.factor,
    args.embed,
    args.distil,
    args.des, 0)

ckpt_path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
if not os.path.exists(ckpt_path):
    raise SystemExit(f'Checkpoint not found: {ckpt_path}')
 

def main():
    # On Windows the multiprocessing start method is 'spawn' and the main module
    # must be protected by this guard. Also freeze support helps when frozen.
    # Reduce num_workers to 0 on Windows to avoid spawn importing main prematurely.
    if sys.platform.startswith('win'):
        args.num_workers = 0

    # build experiment and load weights
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    exp = Exp_Long_Term_Forecast(args)
    # load weights
    ckpt = torch.load(ckpt_path, map_location=exp.device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    else:
        sd = ckpt
    exp.model.load_state_dict(sd)
    print('Loaded weights from', ckpt_path)

    # run one epoch
    exp.train(setting)
    print('Finished 1 additional epoch (setting=', setting, ')')


if __name__ == '__main__':
    # Required for Windows multiprocessing/freeze support
    try:
        multiprocessing.freeze_support()
    except Exception:
        pass
    main()
