import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect the structure of a preprocessed SER .npy file.")
    parser.add_argument("--path", required=True, help="Path to the .npy file.")
    parser.add_argument("--samples", type=int, default=2, help="Number of sample summaries to print.")
    return parser.parse_args()


def describe(name, value, samples):
    print(f"{name}: type={type(value).__name__}, dtype={getattr(value, 'dtype', None)}, shape={getattr(value, 'shape', None)}")
    if hasattr(value, "__len__"):
        for index in range(min(samples, len(value))):
            sample = value[index]
            print(
                f"  [{index}] type={type(sample).__name__}, "
                f"dtype={getattr(sample, 'dtype', None)}, shape={getattr(sample, 'shape', None)}"
            )


def main():
    args = parse_args()
    path = Path(args.path)
    raw = np.load(path, allow_pickle=True)
    root = raw.item() if raw.shape == () and raw.dtype == object else raw
    print(f"file: {path}")
    print(f"root type: {type(root).__name__}")
    if isinstance(root, dict):
        print(f"keys: {list(root.keys())}")
        for key, value in root.items():
            describe(key, value, args.samples)
    else:
        describe("root", root, args.samples)


if __name__ == "__main__":
    main()
