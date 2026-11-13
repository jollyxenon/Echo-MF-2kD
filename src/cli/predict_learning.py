"""仅输入问题系数 (J, h) 执行免训练经典平均场回波初始化（1D 或 2kD 伴随精修）与 Stage-2 全角度能量 L-BFGS-B 求解。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from learning.baselines import MF_METHOD_MAP
from learning.echo_decoder import decode_mean_field, decode_mean_field_multiscalar
from learning.optimization import refine
from records import emit


def main() -> None:
    """在线部署不加载目标态，仅输入 (J, h) 输出优化前后角度、能量和实际耗时。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, required=True, help="单道测试实例 .npz 路径")
    parser.add_argument(
        "--mf-method",
        choices=("Echo-MF-Flat", "Echo-MF-End", "Echo-MF-Alt", "Echo-MF-2kD"),
        default="Echo-MF-2kD",
        help="免训练经典平均场回波初始化模式",
    )
    parser.add_argument(
        "--decoder",
        choices=("1d", "2kd"),
        default="1d",
        help="解码器类型：1d（128点重叠解码）或 2kd（分阶段伴随乘积流形精修）",
    )
    parser.add_argument("--depth", type=int, default=8, help="正偶数电路深度 p=2k")
    parser.add_argument("--beta", type=float, default=2.0, help="阻尼平均场逆温度")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--grid-points", type=int, default=128)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    with np.load(args.instance, allow_pickle=False) as archive:
        J, h = archive["J"].astype(np.float64), archive["h"].astype(np.float64)

    # 第一阶段：零量子测量选点的经典平均场流形初始化
    started = time.perf_counter()
    if args.mf_method == "Echo-MF-2kD" or args.decoder == "2kd":
        initial, meta, prior = decode_mean_field_multiscalar(args.depth, J, h, beta=args.beta)
        prediction = dict(
            seconds=time.perf_counter() - started,
            prior=prior.tolist(),
            decoder="2kd",
            variant=meta["variant"],
            product_energy=meta["e_prod"],
            quadrant_hit=meta["quadrant_hit"],
            beta=args.beta,
        )
        method_name = "Echo-MF-2kD"
    else:
        recipe = MF_METHOD_MAP[args.mf_method]
        scalar, initial, overlap, prior = decode_mean_field(
            recipe, args.depth, J, h, beta=args.beta, grid_points=args.grid_points
        )
        prediction = dict(
            seconds=time.perf_counter() - started,
            scalar=scalar,
            prior=prior.tolist(),
            decoder="1d",
            distribution_overlap=overlap,
            beta=args.beta,
            grid_points=args.grid_points,
        )
        method_name = args.mf_method

    # 第二阶段：释放全部 2p 角度执行有界 L-BFGS-B 能量重耦合优化
    result = refine(J, h, initial)
    stage1_seconds = prediction.pop("seconds")
    emit(dict(
        instance=str(args.instance), architecture="direct",
        method=method_name, depth=args.depth,
        initial_angles=initial.tolist(), angles=result["angles"],
        initial_energy=result["initial_energy"], energy=result["energy"],
        stage1_calls=0, stage2_calls=result["calls"],
        stage1_seconds=stage1_seconds, stage2_seconds=result["seconds"],
        total_seconds=stage1_seconds + result["seconds"],
        status=result["status"], message=result["message"], **prediction,
    ))


if __name__ == "__main__":
    main()
