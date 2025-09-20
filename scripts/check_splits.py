from pathlib import Path
import pandas as pd

p = Path('data/2001_clean.csv')
if not p.exists():
    print('Arquivo não encontrado:', p)
    raise SystemExit(1)

df = pd.read_csv(p)
N = len(df)
num_train = int(N * 0.7)
num_vali = int(N * 0.2)
num_test = N - num_train - num_vali
print('Total linhas:', N)
print('Train:', num_train, 'Val:', num_vali, 'Test:', num_test)
print('Indices:')
print('train 0..', num_train-1)
print('val', num_train, '..', num_train+num_vali-1)
print('test', num_train+num_vali, '..', N-1)

# quick sanity: lengths according to Dataset_Custom __len__ (len - seq_len - pred_len +1)
seq_len = 336
pred_len = 168
for flag, start, end in [('train', 0, num_train), ('val', num_train, num_train+num_vali), ('test', num_train+num_vali, N)]:
    length = max(0, (end - start) - seq_len - pred_len + 1)
    print(f"{flag} usable samples (len - seq_len - pred_len + 1):", length)
