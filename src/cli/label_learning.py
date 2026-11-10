"""MWIS 题库、六方法目标概率标签采集和打包入口。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from instances.mwis import DEFAULT_OUTPUT, DEFAULT_P_VALUES, DEFAULT_SEED, DEFAULT_SIZES, generate
from learning.labels import DEFAULT_CAP, collect, pack, worker


def main() -> None:
    """不同阶段显式执行，正式采集只能恢复相同题库和协议。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    creation = commands.add_parser("generate", help="生成 MWIS 共享题库")
    creation.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    creation.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    creation.add_argument("--p-values", type=int, nargs="+", default=list(DEFAULT_P_VALUES))
    creation.add_argument("--splits", type=int, nargs=3, default=None,
                          metavar=("TRAIN", "VALIDATION", "TEST"))
    creation.add_argument("--seed", type=int, default=DEFAULT_SEED)
    collection = commands.add_parser("collect", help="采集一个固定深度的六方法角度标签")
    collection.add_argument("--dataset", type=Path, required=True)
    collection.add_argument("--output", type=Path, required=True)
    collection.add_argument("--depth", type=int, required=True)
    collection.add_argument("--cap", type=float, default=DEFAULT_CAP)
    collection.add_argument("--workers", type=int, default=16)
    collection.add_argument("--limit", type=int)
    collection.add_argument("--resume", action="store_true")
    worker_parser = commands.add_parser("worker", help="内部单核标签分片")
    worker_parser.add_argument("--dataset", type=Path, required=True)
    worker_parser.add_argument("--output", type=Path, required=True)
    worker_parser.add_argument("--depth", type=int, required=True)
    worker_parser.add_argument("--cap", type=float, default=DEFAULT_CAP)
    worker_parser.add_argument("--workers", type=int, required=True)
    worker_parser.add_argument("--shard", type=int, required=True)
    worker_parser.add_argument("--cpu", type=int, required=True)
    worker_parser.add_argument("--limit", type=int)
    packing = commands.add_parser("pack", help="打包完整标签")
    packing.add_argument("--dataset", type=Path, required=True)
    packing.add_argument("--labels", type=Path, required=True)
    packing.add_argument("--depth", type=int, required=True)
    packing.add_argument("--cap", type=float, default=DEFAULT_CAP)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "generate":
        splits = args.pop("splits")
        generate(args.pop("output"), tuple(args.pop("sizes")), tuple(args.pop("p_values")),
                 tuple(splits) if splits is not None else None, args.pop("seed"))
    elif command == "collect":
        collect(**args)
    elif command == "worker":
        worker(**args)
    else:
        pack(**args)


if __name__ == "__main__":
    main()
