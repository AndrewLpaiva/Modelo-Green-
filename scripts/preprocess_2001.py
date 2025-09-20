"""Preprocess `2001.csv`:
- Read semicolon-delimited CSV
- Combine `DATA (YYYY-MM-DD)` + `HORA (UTC)` into `date` ISO datetime column
- Ensure hourly regular index, reindex and interpolate / ffill missing
- Save cleaned CSV to `data/2001_clean.csv` with `date` as header first column

Usage:
    python scripts/preprocess_2001.py --input 2001.csv --output data/2001_clean.csv --target temp_ar_bul_sec_hr

By default writes all numeric columns and the `date` column as first.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def preprocess(infile: Path, outfile: Path, date_col: str = "DATA (YYYY-MM-DD)",
               hour_col: str = "HORA (UTC)"):
    df = pd.read_csv(infile, sep=';', dtype=str)

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    if date_col not in df.columns or hour_col not in df.columns:
        raise ValueError(f"Esperado colunas '{date_col}' e '{hour_col}' no CSV. Encontradas: {df.columns.tolist()}")

    # Combine date + hour into datetime. Input format appears dd/mm/YYYY and HH:MM
    combined = df[date_col].str.strip() + ' ' + df[hour_col].str.strip()
    # try parsing with dayfirst
    dt = pd.to_datetime(combined, dayfirst=True, errors='coerce')

    if dt.isna().any():
        nbad = int(dt.isna().sum())
        print(f"Aviso: {nbad} linhas com timestamp inválido (serão removidas)")

    df = df.loc[~dt.isna()].copy()
    dt = dt.loc[~dt.isna()].copy()
    df['date'] = dt.dt.strftime('%Y-%m-%d %H:%M:%S')

    # Move date to first column and keep numeric cols
    # Convert numeric columns
    numeric_cols = []
    for c in df.columns:
        if c in (date_col, hour_col, 'date'):
            continue
        # try convert to numeric
        s = pd.to_numeric(df[c].str.replace(',', '.'), errors='coerce')
        if not s.isna().all():
            df[c] = s
            numeric_cols.append(c)
        else:
            # non-numeric columns are dropped
            df = df.drop(columns=[c])

    df = df[['date'] + numeric_cols]

    # Set datetime index
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Ensure hourly regular frequency from min to max
    idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='H')
    df = df.reindex(idx)

    # Count missing
    missing_before = df.isna().sum().sum()
    print(f"Tamanho após reindex: {df.shape}, missing totais: {missing_before}")

    # Interpolate numeric columns and fill remaining with forward-fill/backfill
    df = df.interpolate(method='time', limit=24)
    df = df.fillna(method='ffill').fillna(method='bfill')

    missing_after = df.isna().sum().sum()
    print(f"Missing após interpolação/ffill/bfill: {missing_after}")

    # Save to CSV with date as first column (ISO)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'date'}, inplace=True)
    df.to_csv(outfile, index=False)
    print(f"Salvo CSV limpo em: {outfile} (linhas: {len(df)})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='2001.csv')
    parser.add_argument('--output', type=str, default='data/2001_clean.csv')
    parser.add_argument('--date-col', type=str, default='DATA (YYYY-MM-DD)')
    parser.add_argument('--hour-col', type=str, default='HORA (UTC)')
    args = parser.parse_args()

    preprocess(Path(args.input), Path(args.output), date_col=args.date_col, hour_col=args.hour_col)
