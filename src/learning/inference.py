"""冻结模型仅凭 J,h 生成初始化；逐比特模型使用分布重叠解码。"""
from __future__ import annotations

import time

import numpy as np
import torch

from learning.angle_models import to_angles
from learning.echo_decoder import decode, decode_multiscalar
from learning.echo_prior import build_features, echo_prior_probabilities
from learning.echo_prior_pipeline import restore as restore_prior
from learning.training import inputs_from_checkpoint, predict_raw, restore


def load_predictor(path, grid_points: int = 128, decoder: str = "1d",
                   variant: str = "Echo-Flat-2kD"):
    """加载一个冻结权重，返回只接受 J,h 的预测函数。"""
    header = torch.load(path, map_location="cpu", weights_only=True)
    if header["architecture"] == "Echo-Prior":
        model, checkpoint = restore_prior(path)
    else:
        model, checkpoint, _ = restore(path)

    def predict(J, h):
        """计入特征、网络、配方构造与扫描的逐题时间。"""
        started = time.perf_counter()
        J, h = np.asarray(J, dtype=float), np.asarray(h, dtype=float)
        extra = {}
        if checkpoint["architecture"] == "Echo-Prior":
            features = build_features(J[None], h[None]).astype(np.float32)
            prior = echo_prior_probabilities(model, features)[0]
            if decoder == "2kd":
                angles, meta = decode_multiscalar(
                    checkpoint["depth"], J, h, prior, variant=variant, is_online=True
                )
                extra = dict(prior=prior.tolist(), decoder=decoder, variant=meta["variant"],
                             product_energy=meta["e_prod"], quadrant_hit=meta["quadrant_hit"])
            else:
                scalar, angles, overlap = decode(checkpoint["method"], checkpoint["depth"], h,
                                                 prior, grid_points)
                extra = dict(scalar=scalar, prior=prior.tolist(), decoder=decoder,
                             distribution_overlap=overlap, grid_points=grid_points)
        else:
            with torch.no_grad():
                raw = predict_raw(model, checkpoint["architecture"],
                                  inputs_from_checkpoint(dict(J=J[None], h=h[None]), checkpoint))
                angles = to_angles(raw, checkpoint["depth"])[0].numpy()
        return angles, dict(seconds=time.perf_counter() - started, **extra)

    return predict, checkpoint
