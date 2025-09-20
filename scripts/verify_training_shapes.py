"""
Verificação rápida: carrega (ou gera) CSV com 8000 linhas x 7 colunas,
cria janelas (seq_len, label_len, pred_len), monta DataLoader e roda um
batch através de um modelo simples para validar shapes.

Uso:
  python scripts/verify_training_shapes.py --csv data/custom/mydata.csv

Se --csv não existir, o script gera dados sintéticos e salva em ./data/tmp_generated.csv
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


class WindowDataset(Dataset):
    def __init__(self, data, seq_len=96, label_len=48, pred_len=96):
        # data: numpy array shape (T, C)
        self.data = data.astype(np.float32)
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.T = data.shape[0]
        self.C = data.shape[1]

    def __len__(self):
        return max(0, self.T - self.seq_len - self.pred_len + 1)

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[s_begin:s_end]            # (seq_len, C)
        seq_y = self.data[r_begin:r_end]            # (label_len+pred_len, C)

        # convert to tensors
        return torch.from_numpy(seq_x), torch.from_numpy(seq_y)


class SimpleModel(nn.Module):
    def __init__(self, seq_len, pred_len, in_dim, hidden=128):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.in_dim = in_dim
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(seq_len * in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, pred_len * in_dim)
        )

    def forward(self, x):
        # x: (B, seq_len, C)
        B = x.shape[0]
        out = self.flatten(x)                # (B, seq_len*C)
        out = self.net(out)                  # (B, pred_len*C)
        out = out.view(B, self.pred_len, self.in_dim)
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='data/tmp_generated.csv')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV não encontrado em {args.csv} — gerando dados sintéticos com 8000x7")
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        T = 8000
        C = 7
        # gerar timestamps fictícios e 7 features
        dates = pd.date_range(start='2020-01-01', periods=T, freq='H')
        df = pd.DataFrame(np.random.randn(T, C), columns=[f'f{i}' for i in range(C)])
        df.insert(0, 'date', dates.astype(str))
        df.to_csv(args.csv, index=False)

    print('Carregando CSV:', args.csv)
    df = pd.read_csv(args.csv)
    # assumimos que a primeira coluna é 'date' e as demais são features
    feature_cols = [c for c in df.columns if c != 'date']
    data = df[feature_cols].values
    T, C = data.shape
    print(f'Dados: T={T}, C={C}, bytes aproximados={(data.nbytes/1024):.1f} KB')

    ds = WindowDataset(data, seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len)
    print('Número de amostras (janelas):', len(ds))

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    batch = next(iter(dl))
    batch_x, batch_y = batch
    print('batch_x shape (B, seq_len, C):', batch_x.shape)
    print('batch_y shape (B, label_len+pred_len, C):', batch_y.shape)

    model = SimpleModel(args.seq_len, args.pred_len, C)
    out = model(batch_x)
    print('Model output shape (B, pred_len, C):', out.shape)

    # exemplo de cálculo de perda rápido
    y_true = batch_y[:, -args.pred_len:, :]
    loss = nn.MSELoss()(out, y_true)
    print('Loss (exemplo):', loss.item())


if __name__ == '__main__':
    main()
