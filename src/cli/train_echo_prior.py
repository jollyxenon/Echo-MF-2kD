"""训练逐比特概率编码器并执行 Echo 一维解码。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from learning.distance import ECHO_METHODS
from learning.echo_prior_pipeline import run


def main() -> None:
    """输入只有 J,h；目标态只用于冻结后的离线概率核验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=ECHO_METHODS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--grid-points", type=int, default=4097)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--threads", type=int, default=1)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
