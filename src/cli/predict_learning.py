"""仅凭 J,h 预测初始化（支持冻结神经权重或免训练 MF-Echo）并用能量驱动 L-BFGS-B 求解。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from learning.inference import load_predictor
from learning.echo_decoder import decode_mean_field, decode_mean_field_multiscalar
from learning.baselines import MF_METHOD_MAP
from learning.optimization import refine
from records import emit


def main() -> None:
    """部署不加载目标态；输出优化前后角度、能量和实际耗时。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None, help="冻结神经模型权重路径（与 --mf-method 二选一）")
    parser.add_argument(
        "--mf-method",
        choices=("Echo-MF-Flat", "Echo-MF-End", "Echo-MF-Alt", "Echo-MF-2kD"),
        default=None,
        help="免训练经典平均场回波初始化模式（Echo-MF-Flat / Echo-MF-End / Echo-MF-Alt / Echo-MF-2kD）",
    )
    parser.add_argument(
        "--decoder",
        choices=("1d", "2kd"),
        default="1d",
        help="逐比特边缘解码器类型：1d（128点重叠解码）或 2kd（分阶段伴随乘积流形精修）",
    )
    parser.add_argument("--depth", type=int, default=8, help="使用 --mf-method 时的偶电路深度 p=2k")
    parser.add_argument("--beta", type=float, default=2.0, help="使用 --mf-method 时的平均场逆温度")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--grid-points", type=int, default=128)
    args = parser.parse_args()
    if (args.checkpoint is None) == (args.mf_method is None):
        parser.error("必须且只能指定 --checkpoint 或 --mf-method 之一")

    torch.set_num_threads(args.threads)
    with np.load(args.instance, allow_pickle=False) as archive:
        J, h = archive["J"].astype(np.float64), archive["h"].astype(np.float64)

    if args.mf_method is not None:
        architecture = "direct"
        started = time.perf_counter()
        if args.mf_method == "Echo-MF-2kD" or args.decoder == "2kd":
            initial, meta, prior = decode_mean_field_multiscalar(
                args.depth, J, h, beta=args.beta
            )
            prediction = dict(
                seconds=time.perf_counter() - started,
                prior=prior.tolist(),
                decoder="2kd",
                variant=meta["variant"],
                product_energy=meta["e_prod"],
                quadrant_hit=meta["quadrant_hit"],
                beta=args.beta,
            )
            method_name, depth = "Echo-MF-2kD", args.depth
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
            method_name, depth = args.mf_method, args.depth
    else:
        predict, checkpoint = load_predictor(args.checkpoint, args.grid_points, decoder=args.decoder)
        architecture = checkpoint["architecture"]
        initial, prediction = predict(J, h)
        suffix = {"Flat-Top": "Flat", "Endpoint-Matched": "End",
                  "Alternating-Phase": "Alt"}.get(checkpoint["method"], checkpoint["method"])
        if checkpoint["architecture"] == "Echo-Prior":
            method_name = f"Echo-Prior-{suffix}" + ("-2kD" if args.decoder == "2kd" else "")
        else:
            method_name = checkpoint["method"]
        depth = checkpoint["depth"]

    result = refine(J, h, initial)
    stage1_seconds = prediction.pop("seconds")
    emit(dict(
        instance=str(args.instance), architecture=architecture,
        method=method_name, depth=depth,
        initial_angles=initial.tolist(), angles=result["angles"],
        initial_energy=result["initial_energy"], energy=result["energy"],
        stage1_calls=0, stage2_calls=result["calls"],
        stage1_seconds=stage1_seconds, stage2_seconds=result["seconds"],
        total_seconds=stage1_seconds + result["seconds"],
        status=result["status"], message=result["message"], **prediction,
    ))


if __name__ == "__main__":
    main()
