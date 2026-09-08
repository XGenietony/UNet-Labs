import argparse
import glob
import os

import numpy as np
import pandas as pd

METRICS = ['precision', 'recall', 'f1', 'ois_f1', 'ods_f1', 'boundary_iou', 'dice', 'iou']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv-dir', required=True,
                   help='Directory containing fold0.csv … fold{N}.csv from run_eval_best.sh')
    p.add_argument('--output',  default=None,
                   help='Output CSV path (default: <csv-dir>/kfold_summary.csv)')
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.csv_dir, 'fold*.csv')))
    if not files:
        print(f'[ERROR] No fold*.csv found in {args.csv_dir}')
        return

    rows = []
    for f in files:
        df = pd.read_csv(f)
        if df.empty:
            print(f'[WARN] {f} is empty, skipping')
            continue
        r = df.iloc[0]
        rows.append({'fold': int(r['fold']) if 'fold' in r else os.path.basename(f),
                     'epoch': int(r['epoch']) if 'epoch' in r else -1,
                     **{m: float(r[m]) for m in METRICS if m in r}})

    if not rows:
        print('[ERROR] No data loaded.')
        return

    df_all = pd.DataFrame(rows).set_index('fold')
    mean = df_all[METRICS].mean()
    std  = df_all[METRICS].std()

    print(df_all[['epoch'] + METRICS].to_string(float_format='%.4f'))
    print()
    print(pd.DataFrame({'Mean': mean, 'Std': std}).T.to_string(float_format='%.4f'))

    out = df_all[METRICS].copy()
    out.loc['Mean'] = mean
    out.loc['Std']  = std

    output_path = args.output or os.path.join(args.csv_dir, 'kfold_summary.csv')
    out.to_csv(output_path, float_format='%.4f')
    print(f'\nSaved → {output_path}')


if __name__ == '__main__':
    main()
