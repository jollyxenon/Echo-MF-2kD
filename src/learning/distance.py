"""标签方法名和角度等价距离。"""
from __future__ import annotations

import numpy as np

ECHO_METHODS = (
    "Flat-Top",
    "Alternating-Phase",
    "Endpoint-Matched",
)
BASELINE_METHODS = (
    "RI-LBFGSB",
    "TQA-LBFGSB",
    "INTERP-LBFGSB",
)
METHODS = ECHO_METHODS + BASELINE_METHODS


def squared_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """支持广播，整组 gamma/beta 同时反号且 beta 按 pi 周期等价。"""
    a, b = np.asarray(a), np.asarray(b)
    depth = a.shape[-1] // 2
    values = []
    for sign in (1.0, -1.0):
        gamma = (a[..., :depth] - sign * b[..., :depth]) / (10 * np.pi)
        beta = ((a[..., depth:] - sign * b[..., depth:] + np.pi / 2) % np.pi - np.pi / 2) / (np.pi / 2)
        values.append((gamma ** 2).sum(-1) + (beta ** 2).sum(-1))
    return np.minimum(*values) / (2 * depth)


def select(pool: list[dict], relative_floor: float = 0.9, limit: int = 4) -> list[dict]:
    """按概率降序保留质量接近且角度不重复的候选。"""
    if not pool:
        return []
    ordered = sorted(pool, key=lambda item: -item["probability"])
    chosen = []
    for item in ordered:
        if item["probability"] < relative_floor * ordered[0]["probability"]:
            continue
        if all(squared_distance(item["angles"], old["angles"]) > 0.01 ** 2 for old in chosen):
            chosen.append(item)
            if len(chosen) == limit:
                break
    return chosen
