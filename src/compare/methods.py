"""三种目标概率基线，共用有界候选评估与墙钟预算。"""
from __future__ import annotations

import numpy as np

METHODS = ("RI-LBFGSB", "TQA-LBFGSB", "INTERP-LBFGSB")
DEFAULT_CONFIG = {
    "lbfgsb": {"maxiter": 128, "maxfun": 2048, "maxls": 20,
                "ftol": 2.220446049250313e-9, "gtol": 1e-8},
    "tqa_seed_count": 32,
}


def random_start(rng, G, depth, multiscale):
    """交替使用全域与共同对数尺度，返回紧凑的 gamma、beta。"""
    scale = np.exp(rng.uniform(np.log(min(1.0, G)), np.log(G))) if multiscale else G
    return np.r_[rng.uniform(-scale, scale, depth),
                 rng.uniform(-np.pi / 2, np.pi / 2, depth)]


def bounds(G, depth):
    """给 gamma 和 beta 分别设置实际评估边界。"""
    return [(-G, G)] * depth + [(-np.pi / 2, np.pi / 2)] * depth


def interpolate(x):
    """使用 Zhou INTERP 相邻深度公式，分别插值两个角度向量。"""
    q = len(x) // 2
    weight = np.arange(q + 1) / q
    parts = []
    for part in (x[:q], x[q:]):
        padded = np.r_[0.0, part, 0.0]
        parts.append(weight * padded[:-1] + (1 - weight) * padded[1:])
    return np.concatenate(parts)
