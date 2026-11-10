"""冻结模型预测起点，后接能量优化并独立核验完整线路。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from learning.evaluation import evaluate


def main() -> None:
    """只读冻结 checkpoint，不按测试题结果挑选模型。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--grid-points", type=int, default=128)
    parser.add_argument("--decoder", choices=("1d", "2kd"), default="1d")
    evaluate(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
