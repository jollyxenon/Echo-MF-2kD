"""RI 全角度与 TQA/Echo 一维初始化的固定参数化。"""
from __future__ import annotations

from functools import lru_cache
import numpy as np
from echo.variants import load_formula
from compare.methods import random_start

G = 10 * np.pi


@lru_cache(maxsize=None)
def echo_formula(method, depth):
    """保留配方构造诊断，复用同一模型的固定配方。"""
    return load_formula(method, depth)


@lru_cache(maxsize=None)
def echo_basis(method, depth):
    """按配方构造线性 cost 方向与固定 mixer；缓存只供预测复用。"""
    formula = echo_formula(method, depth)
    ratios = np.asarray(formula["ratios"], dtype=float)
    ratios = ratios / np.max(np.abs(ratios))
    direction = np.column_stack((-ratios, ratios)).reshape(-1)
    beta = np.column_stack((np.full(depth // 2, np.pi / 2), formula["mixers"])).reshape(-1)
    return direction, beta


def expand(method, depth, scalar):
    """展开标量，保持全部 gamma 后接全部 beta 的顺序。"""
    scalar = np.atleast_1d(np.asarray(scalar, dtype=float))
    if method == "TQA-LBFGSB":
        ramp = (np.arange(depth) + .5) / depth
        gamma = scalar[:, None] * ramp
        beta = (-scalar[:, None] * (1 - ramp) + np.pi / 2) % np.pi - np.pi / 2
    else:
        direction, fixed = echo_basis(method, depth)
        gamma = scalar[:, None] * direction
        beta = np.broadcast_to(fixed, gamma.shape)
    return np.concatenate((gamma, beta), axis=-1)


def starts(method, depth, rng):
    """持续产生不依赖答案的起点；结构化方法逐批细化一维网格。"""
    if method == "RI-LBFGSB":
        restart = 0
        while True:
            yield random_start(rng, G, depth, restart % 2 == 1), None
            restart += 1
    elif method == "TQA-LBFGSB":
        ladder = np.geomspace(.05, 5., 32)
        fresh = ladder
        while True:
            for scalar in rng.permutation(fresh):
                yield expand(method, depth, scalar)[0], float(scalar)
            fresh = np.sqrt(ladder[:-1] * ladder[1:])
            ladder = np.sort(np.r_[ladder, fresh])
    else:
        echo_basis(method, depth)
        ladder = np.linspace(-G, G, 33)
        fresh = ladder
        while True:
            for scalar in rng.permutation(fresh):
                yield expand(method, depth, scalar)[0], float(scalar)
            fresh = (ladder[:-1] + ladder[1:]) / 2
            ladder = np.sort(np.r_[ladder, fresh])
