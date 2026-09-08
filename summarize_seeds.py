import argparse
import pandas as pd
import numpy as np

METRICS = ['dice', 'iou', 'precision', 'recall', 'f1', 'ois_f1', 'ods_f1', 'boundary_iou']

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds',  type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--epochs', type=int, nargs='+', required=True,
                   help='每个 seed 对应的最优 epoch，顺序与 --seeds 一致')
    p.add_argument('--exp-prefix', default='perlin_seed')
    p.add_argument('--output', default='results/seeds_stability.csv')
    args = p.parse_args()

    assert len(args.seeds) == len(args.epochs)

    rows = []
    for seed, epoch in zip(args.seeds, args.epochs):
        path = f'results/{args.exp_prefix}{seed}_eval.csv'
        df = pd.read_csv(path)
        row = df[df['epoch'] == epoch]
        if row.empty:
            print(f'[WARN] seed {seed} epoch {epoch} not found in {path}')
            continue
        r = row.iloc[0]
        rows.append({'seed': seed, 'epoch': epoch, **{m: r[m] for m in METRICS if m in r}})

    if not rows:
        print('[ERROR] No data found.')
        return

    df_all = pd.DataFrame(rows).set_index('seed')
    avail  = [m for m in METRICS if m in df_all.columns]
    mean   = df_all[avail].mean()
    std    = df_all[avail].std(ddof=1)

    print("\n=== Best epoch per seed ===")
    print(df_all[['epoch'] + avail].to_string(float_format='%.4f'))
    print("\n=== Mean ± Std across seeds ===")
    for m in avail:
        print(f"  {m:<14}: {mean[m]:.4f} ± {std[m]:.4f}")

    out = df_all[avail].copy()
    out.loc['Mean'] = mean
    out.loc['Std']  = std
    out.to_csv(args.output, float_format='%.4f')
    print(f'\nSaved → {args.output}')

if __name__ == '__main__':
    main()
