"""完整 Echo 对的精确直积态、固定配方的一维扫描与多起点分批。"""
from __future__ import annotations

import numpy as np

G = 10*np.pi

# 起点分批的口径。a 是带符号的 max|gamma|，相邻起点的间距按 a 计的弧度。
# 批内按分数降序，层与层之间只放宽最小间隔；批的组成只由候选池决定。
START_BATCH = 32
START_SPACINGS = (1., .5, .25)
# 进批之前先按这个宽度分箱，每箱只留分数最高的一个，避免近重复起点与网格堆积。
START_DEDUPE = .0625


def validate_depth(depth):
    """三种 Echo 只接受由完整 Echo 对组成的正偶数深度。"""
    if depth <= 0 or depth % 2:
        raise ValueError("Echo 层数必须为正偶数 p=2k，k>=1")
    return depth


def spinors(costs, mixers, h):
    """批量递推任意 g_j；最后两维分别为 Echo 对与比特。"""
    costs = np.asarray(costs)
    minus = np.ones(costs.shape[:-1]+(len(h),),dtype=complex)/np.sqrt(2)
    plus = minus.copy()
    mixers = np.asarray(mixers)
    for j in range(costs.shape[-1]):
        theta = mixers[...,j,None]
        phase = np.exp(2j*costs[...,j,None]*h)
        minus,plus = phase*plus,np.conj(phase)*minus
        minus,plus = (np.cos(theta)*minus-1j*np.sin(theta)*plus,
                      np.cos(theta)*plus-1j*np.sin(theta)*minus)
    return minus,plus


def probabilities(costs,mixers,h,target):
    """真实目标概率由精确单比特因子相乘，不使用代理响应。"""
    minus,plus = spinors(costs,mixers,h)
    return np.prod(np.where(target>0,abs(plus)**2,abs(minus)**2),axis=-1)


def angles(costs,mixers,depth):
    """把完整 Echo 对展开成标准 gamma 后接 beta 的线路参数。"""
    if depth <= 0 or depth % 2 or 2*len(costs) != depth or len(mixers) != len(costs):
        raise ValueError("Echo 层数必须为正偶数，且与完整 Echo 对数量一致")
    gamma,beta = [],[]
    for g,theta in zip(costs,mixers):
        gamma.extend([float(-g),float(g)])
        beta.extend([np.pi/2,float(theta)])
    return np.array(gamma+beta)


def start_batches(a,score,limit=START_BATCH,spacings=START_SPACINGS):
    """把候选池切成起点批次，批内按 score 降序。

    先按 START_DEDUPE 分箱去重（每箱留分数最高的一个），再逐层放宽最小间隔：
    有限时间下每层都取到互不重复的峰，时间充裕时用完后继续从剩余候选里成批产。
    批的组成只由候选池决定，与评价值无关，因此同一 seed 的短预算运行仍是长预算
    运行的严格前缀。
    """
    a = np.asarray(a,dtype=float)
    order = np.argsort(-np.asarray(score,dtype=float),kind="stable")
    bins = np.round(a[order]/START_DEDUPE).astype(np.int64)
    a = a[order][np.sort(np.unique(bins,return_index=True)[1])]
    taken = np.zeros(len(a),dtype=bool)
    for spacing in spacings:
        fresh = np.zeros(len(a),dtype=bool)
        batch = []
        for index in range(len(a)):
            if taken[index] or fresh[index]:
                continue
            batch.append(index)
            taken[index] = True
            fresh |= np.abs(a-a[index]) <= spacing
            if len(batch) == limit:
                break
        if batch:
            yield [float(a[index]) for index in batch]
    rest = np.flatnonzero(~taken)
    for first in range(0,len(rest),limit):
        yield [float(a[index]) for index in rest[first:first+limit]]


def fixed_scan(h,target,ratios,mixers,points=4097):
    """统一密网格后精修局部峰，返回按真实概率排序的候选池。"""
    ratios = np.asarray(ratios,dtype=float)
    normalized = ratios/max(abs(ratios))
    x = np.linspace(-1.,1.,points)
    p = probabilities(G*x[:,None]*normalized,mixers,h,target)
    peaks = np.flatnonzero((p[1:-1]>=p[:-2])&(p[1:-1]>=p[2:]))+1
    peaks_found = len(peaks)
    order = peaks[np.argsort(-p[peaks],kind="stable")]
    chosen = []
    # 只精修按第一批最小间隔就已经互不重复的峰，多下来的峰在后续批次里自然轮到。
    for index in order:
        if all(abs(x[index]-x[other])*G > START_SPACINGS[0] for other in chosen):
            chosen.append(index)
        if len(chosen) == 2*START_BATCH:
            break
    peaks = np.array(chosen,dtype=int)
    lo,hi = x[peaks-1],x[peaks+1]
    factor = (np.sqrt(5)-1)/2
    left,right = hi-factor*(hi-lo),lo+factor*(hi-lo)
    pl = probabilities(G*left[:,None]*normalized,mixers,h,target)
    pr = probabilities(G*right[:,None]*normalized,mixers,h,target)
    count = points+2*len(peaks)
    for _ in range(36):
        choose = pl>pr
        lo,hi = np.where(choose,lo,left),np.where(choose,right,hi)
        left,right = hi-factor*(hi-lo),lo+factor*(hi-lo)
        pl = probabilities(G*left[:,None]*normalized,mixers,h,target)
        pr = probabilities(G*right[:,None]*normalized,mixers,h,target)
        count += 2*len(peaks)
    refined = (lo+hi)/2*G
    # 精修峰在前、整条网格点在后；±G 与 0 已经包含在网格里。
    return dict(coordinates=np.r_[refined,G*x],
                score=np.r_[probabilities(refined[:,None]*normalized,mixers,h,target),p],
                evaluations=count,peaks_found=peaks_found,peaks_refined=len(peaks))
