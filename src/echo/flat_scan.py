"""平顶 Echo：偶数层完整 Echo 对与一维区间扫描。"""
from __future__ import annotations

import numpy as np
from scipy.special import betaln, xlogy


def log_success_terms(a, w, pairs):
    """计算 log(q) 与 log(S(1-q))；正项对数求和避免高阶幂的相消和下溢。"""
    phase = np.asarray(a)[...,None]*w-np.pi/2
    q = np.cos(phase/2)**2
    complement = np.sin(phase/2)**2
    log_series = np.zeros_like(q)
    log_coefficient = 0.
    for j in range(1,pairs):
        log_coefficient += np.log(pairs-1+j)-np.log(j)
        log_series = np.logaddexp(log_series,log_coefficient+xlogy(j,complement))
    return phase,q,complement,log_series


def flat_loss(a, w, pairs):
    """任意阶平顶的负对数成功概率，p(q)=q^k S(1-q)。"""
    _,q,_,log_series = log_success_terms(a,w,pairs)
    with np.errstate(divide="ignore"):
        return -np.sum(pairs*np.log(q)+log_series,axis=-1)


def flat_gradient(a, w, pairs):
    """由不完全 Beta 函数密度计算解析导数，供无零点区间二分使用。"""
    phase,_,complement,log_series = log_success_terms(a,w,pairs)
    weight = np.exp(xlogy(pairs-1,complement)-betaln(pairs,pairs)-log_series)
    return np.sum(w*np.tan(phase/2)*weight,axis=-1)


def flat_scan(h, target, pairs, maximum, points=4097):
    """枚举有限角度域内的全部无零点区间，以解析导数二分求驻点。

    返回带符号坐标 a（每对的第一个 cost）、对数概率与区间数；对数概率保序，
    下游按它排序，避免大 n 下真实概率下溢成 0 后丢失次序。驻点之后追加一整条
    均匀网格，供时间充裕时继续产起点。
    """
    w = 4*h*target
    w = w[w != 0]
    if not len(w):
        # 均匀场退化：全域概率相同，仍然交出一条完整网格供续产。
        return np.r_[0.,np.linspace(-maximum,maximum,points)], np.zeros(points+1), 1
    breaks = [-maximum, maximum]
    for frequency in w:
        lower, upper = sorted([-maximum*frequency, maximum*frequency])
        first = int(np.ceil((lower+np.pi/2)/(2*np.pi)))
        last = int(np.floor((upper+np.pi/2)/(2*np.pi)))
        breaks.extend(((-np.pi/2+2*np.pi*np.arange(first,last+1))/frequency).tolist())
    breaks = np.unique(breaks)
    lo, hi = breaks[:-1].copy(), breaks[1:].copy()
    count = len(lo)
    for _ in range(60):
        mid = (lo+hi)/2
        positive = flat_gradient(mid,w,pairs) >= 0
        lo, hi = np.where(positive,lo,mid), np.where(positive,mid,hi)
        if np.max(hi-lo) < 1e-9:
            break
    a = np.r_[(lo+hi)/2,-maximum,maximum,0.,np.linspace(-maximum,maximum,points)]
    return a, -flat_loss(a,w,pairs), count
