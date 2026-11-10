"""MWIS 题库生成入口。"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from instances.mwis import DEFAULT_OUTPUT, DEFAULT_P_VALUES, DEFAULT_SEED, DEFAULT_SIZES, generate


def main() -> None:
    """生成供 p=8/10/12 共享的 MWIS 题库（默认 n=10 为 4096/512/1024，n=6/8/12/14 为 0/0/1024）。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--p-values", type=int, nargs="+", default=list(DEFAULT_P_VALUES))
    parser.add_argument("--splits", type=int, nargs=3, default=None, metavar=("TRAIN", "VALIDATION", "TEST"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    generate(args.output, tuple(args.sizes), tuple(args.p_values),
             tuple(args.splits) if args.splits is not None else None, args.seed)


if __name__ == "__main__":
    main()
