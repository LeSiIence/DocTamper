"""
为任意 LMDB 数据集生成 pks (随机 JPEG 压缩链) 文件。

用法:
    python generate_pks.py --lmdb_path DocTamperV1-TrainingSet --minq 75
    python generate_pks.py --lmdb_path DocTamperV1-TrainingSet --minq 75 80 85 90
"""
import argparse
import os
import pickle
import lmdb
import numpy as np


def generate_pks(num_samples: int, minq: int, seed: int = 42) -> dict:
    rng = np.random.RandomState(seed + minq)
    records = {}
    for i in range(num_samples):
        chain_len = rng.choice([1, 2, 3])
        qs = rng.randint(minq, 101, size=chain_len)
        records[i] = qs
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_path', type=str, required=True,
                        help='LMDB 目录路径，如 DocTamperV1-TrainingSet')
    parser.add_argument('--minq', type=int, nargs='+', default=[75],
                        help='要生成的 minq 值列表，如 75 80 85 90')
    parser.add_argument('--out_dir', type=str, default='pks',
                        help='输出目录 (默认 pks/)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    env = lmdb.open(args.lmdb_path, readonly=True, lock=False,
                    readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        num_samples = int(txn.get('num-samples'.encode('utf-8')))
    env.close()

    lmdb_name = os.path.basename(os.path.normpath(args.lmdb_path))
    os.makedirs(args.out_dir, exist_ok=True)

    for q in args.minq:
        records = generate_pks(num_samples, q, seed=args.seed)
        out_path = os.path.join(args.out_dir, f'{lmdb_name}_{q}.pk')
        with open(out_path, 'wb') as f:
            pickle.dump(records, f)
        print(f'Generated {out_path}: {num_samples} samples, minq={q}')


if __name__ == '__main__':
    main()
