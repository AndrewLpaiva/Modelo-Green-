import pandas as pd
import os
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/prepare_dataset.py <src_filename> [<dst_name> (without path)]')
        raise SystemExit(1)
    src = sys.argv[1]
    if not os.path.isabs(src):
        src = os.path.join(os.getcwd(), src)
    if not os.path.exists(src):
        print('Source file not found:', src)
        raise SystemExit(1)
    dst_name = None
    if len(sys.argv) >= 3:
        dst_name = sys.argv[2]
    else:
        # derive from filename, strip extension and any suffix
        base = os.path.basename(src)
        name = os.path.splitext(base)[0]
        dst_name = f'{name}_clean.csv'

    dst = os.path.join(os.path.dirname(__file__), '..', 'data', dst_name)
    dst = os.path.abspath(dst)

    print('Reading', src)
    # attempt semicolon first, then comma
    try:
        df = pd.read_csv(src, sep=';')
    except Exception:
        df = pd.read_csv(src)

    cols = list(df.columns)
    print('Detected columns:', cols[:6])
    # detect date/time
    date_col = None
    time_col = None
    if len(cols) >= 2 and (('DATE' in cols[0].upper()) or ('DATA' in cols[0].upper())) and (('HORA' in cols[1].upper()) or ('TIME' in cols[1].upper())):
        date_col = cols[0]
        time_col = cols[1]
        rest = cols[2:]
        # Normalize time strings like '0000 UTC', '0100 UTC', '00:00', '0:00' -> 'HH:MM:SS'
        def normalize_time(t):
            if pd.isna(t):
                return ''
            s = str(t).strip()
            # remove trailing 'UTC' or other text
            s = s.replace('UTC', '').strip()
            # if it's like '0000' or '0100' possibly with no colon
            import re
            m = re.match(r"^(\d{1,4})$", s)
            if m:
                hhmm = m.group(1).zfill(4)
                hh = hhmm[:2]
                mm = hhmm[2:]
                return f"{hh}:{mm}:00"
            # if like '00:00' or '0:00'
            m2 = re.match(r"^(\d{1,2}):(\d{2})(:?\d{0,2})$", s)
            if m2:
                hh = int(m2.group(1))
                mm = int(m2.group(2))
                if 0 <= hh < 24 and 0 <= mm < 60:
                    return f"{hh:02d}:{mm:02d}:00"
            # fallback: return original
            return s

        combined = df[date_col].astype(str).str.strip() + ' ' + df[time_col].astype(str).str.strip()
        # apply normalization to the time part
        def combine_row(r):
            date_part = str(r[0]).strip()
            time_part = normalize_time(r[1])
            if date_part == 'nan' or date_part == '' or time_part == '':
                return pd.NaT
            # replace '/' with '-' for consistent parsing
            date_part = date_part.replace('/', '-').strip()
            full = date_part + ' ' + time_part
            return pd.to_datetime(full, errors='coerce')

        df_out = df.copy()
        df_out['date'] = df[[date_col, time_col]].apply(combine_row, axis=1)
        out_cols = ['date'] + rest
        df_out = df_out[out_cols]
    else:
        # try to find a column named 'date' or similar
        lower_cols = [c.lower() for c in cols]
        if 'date' in lower_cols:
            date_col = cols[lower_cols.index('date')]
            rest = [c for c in cols if c != date_col]
            df_out = df.copy()
            df_out['date'] = pd.to_datetime(df_out[date_col], errors='coerce')
            out_cols = ['date'] + rest
            df_out = df_out[out_cols]
        else:
            raise SystemExit('Could not detect date/time columns in source CSV; columns: {}'.format(cols))

    df_out['date'] = df_out['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    df_out.to_csv(dst, index=False)
    print('Wrote', dst)
    print('Header of cleaned file:', list(df_out.columns)[:10])
