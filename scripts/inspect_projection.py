import torch
import numpy as np
import os
p = r'C:\Users\supor\WEATHER-5K\checkpoints\long_term_forecast_Pyraformer_test_custom_ftS_sl336_ll168_pl168_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_test_0\checkpoint.pth'
if not os.path.exists(p):
    print('checkpoint not found:', p)
    raise SystemExit(1)
ckpt = torch.load(p, map_location='cpu')
# determine state_dict
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    sd = ckpt['model_state_dict']
elif isinstance(ckpt, dict) and all(hasattr(v, 'numel') for v in ckpt.values()):
    sd = ckpt
else:
    sd = ckpt
# look for projection weight
candidates = [k for k in sd.keys() if 'projection.weight' in k or k.endswith('projection.weight') or 'projection' in k and 'weight' in k]
# fallback: try exact 'projection.weight'
if 'projection.weight' in sd:
    key = 'projection.weight'
elif candidates:
    key = candidates[0]
else:
    # try to find any large 2D tensor as plausible final linear layer
    key = None
    for k,v in sd.items():
        try:
            if hasattr(v, 'dim') and v.dim()==2:
                key = k
                break
        except Exception:
            continue

if key is None:
    print('No suitable weight tensor found. Available keys sample:')
    for i,k in enumerate(list(sd.keys())[:20]):
        print(i+1, k)
    raise SystemExit(1)

w = sd[key].numpy()
print('key:', key)
print('shape:', w.shape)
# print first 5x5 (or smaller if dims <5)
r = min(5, w.shape[0])
c = min(5, w.shape[1])
np.set_printoptions(precision=6, suppress=True)
print('first {}x{} values:'.format(r,c))
print(w[:r, :c])
print('\nfirst row (first 20 vals):')
print(w[0, :min(20, w.shape[1])])
