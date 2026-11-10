"""对现有题目生成偶数层 Echo 参数，并在标准输出报告完整线路复算。"""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from echo.variants import METHOD_NAMES, generate
from compare.runtime import Objective
from compare.verify import verify
from records import emit


def main():
    """只读题库，输出配方、候选、模块计时与复算结果，不写实验目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--depth", type=int, required=True, help="正偶数层数 p=2k")
    parser.add_argument("--variant", choices=list(METHOD_NAMES), default="Flat-Top")
    args = parser.parse_args()
    with np.load(args.instance) as data:
        J, h, target, ground = (data[key].copy() for key in ("J", "h", "s_star", "ground_indices"))
    stream = io.StringIO()
    objective = Objective(J, h, target, args.depth, stream)
    formula, starts = generate(objective, args.variant)
    result = objective.result()
    verify(result, J, h, target, ground)
    result.update(instance=str(args.instance), method=args.variant,
                  method_name=METHOD_NAMES[args.variant], formula=formula,
                  start_batches=[len(batch) for batch in starts],
                  events=[json.loads(line) for line in stream.getvalue().splitlines()])
    emit(result)


if __name__ == "__main__":
    main()
