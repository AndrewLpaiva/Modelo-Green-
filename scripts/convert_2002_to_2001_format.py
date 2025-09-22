#!/usr/bin/env python3
import csv
import os

input_path = os.path.join(os.path.dirname(__file__), '..', 'data', '2002.csv')
output_tmp = os.path.join(os.path.dirname(__file__), '..', 'data', '2002_converted.csv')
output_final = os.path.join(os.path.dirname(__file__), '..', 'data', '2002.csv')

# Desired header to match 2001_clean.csv
header = ['date','prec_tot_hr','press_atm_niv_est_hr','temp_ar_bul_sec_hr','temp_pt_orv','temp_max_hr_ant','temp_min_hr_ant','umi_rel_ar_hr','ven_dir_hr','ven_vel_hr']

count = 0
with open(input_path, 'r', encoding='utf-8') as fin, open(output_tmp, 'w', newline='', encoding='utf-8') as fout:
    reader = csv.reader(fin, delimiter=';')
    writer = csv.writer(fout, delimiter=',')

    # read header
    try:
        in_header = next(reader)
    except StopIteration:
        raise SystemExit('Input file is empty: ' + input_path)

    # normalize header names (strip)
    in_header = [h.strip() for h in in_header]

    # find positions
    # expect first two columns are date and time (names may vary)
    # fallback: assume first column is date and second is time
    date_idx = 0
    time_idx = 1

    # remaining columns should match the rest of header order; map by name if possible
    # build a map of input column name -> index
    name2idx = {name: idx for idx, name in enumerate(in_header)}

    # try detect known names
    for k in name2idx.keys():
        lk = k.lower()
        if 'data' in lk and 'yyyy' in lk or 'data' == lk or 'date' == lk:
            date_idx = name2idx[k]
        if 'hora' in lk or 'time' in lk:
            time_idx = name2idx[k]

    # build input indices for the remaining fields in desired header order
    in_indices = []
    for col in header[1:]:
        if col in name2idx:
            in_indices.append(name2idx[col])
        else:
            # fallback: try case-insensitive match
            found = None
            for k, v in name2idx.items():
                if k.strip().lower() == col.lower():
                    found = v
                    break
            if found is not None:
                in_indices.append(found)
            else:
                raise SystemExit(f'Missing expected column "{col}" in input. Found: {in_header}')

    # write new header
    writer.writerow(header)

    for row in reader:
        if not row or len(row) < max(time_idx, max(in_indices))+1:
            continue
        # combine date and time into 'YYYY-MM-DD HH:MM:SS'
        date_val = row[date_idx].strip()
        time_val = row[time_idx].strip()
        # if time has format HH:MM, add :00 seconds
        if len(time_val.split(':')) == 2:
            time_val = time_val + ':00'
        timestamp = f"{date_val} {time_val}"

        out_row = [timestamp]
        for idx in in_indices:
            val = row[idx].strip()
            # normalize decimal comma to dot if any
            val = val.replace(',', '.')
            out_row.append(val)

        writer.writerow(out_row)
        count += 1

print(f'Converted {count} rows to {output_tmp}, moving to {output_final}')
# replace final
os.replace(output_tmp, output_final)
print('Done.')
