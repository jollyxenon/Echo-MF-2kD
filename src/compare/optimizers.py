"""Echo 起点消费与三条基线的续产批次。"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from compare.methods import DEFAULT_CONFIG,random_start,interpolate,bounds


def local(objective,x):
    """L-BFGS-B保留本次所有已评估点中的最好点。"""
    best,best_p = x.copy(),-np.inf
    objective.start("LBFGSB",x)
    def loss(candidate):
        """返回共享自动微分得到的负概率与梯度。"""
        nonlocal best,best_p
        p,g = objective.evaluate(candidate,True)
        if p>best_p:
            best,best_p = candidate.copy(),p
        return -p,-g
    result = minimize(loss,x,jac=True,method="L-BFGS-B",bounds=bounds(objective.G,len(x)//2),options=DEFAULT_CONFIG["lbfgsb"])
    objective.event("local_exit",optimizer="LBFGSB",status=int(result.status),message=str(result.message),
                    best_probability=float(best_p),depth=len(x)//2)
    return best


def consume(objective,starts,run):
    """按给定批次顺序逐个起点跑一次本地优化，上限预算到点即停；无截断轨消费首批（Top-32）后终止。"""
    for batch in starts:
        for x in batch:
            objective.check()
            run(objective,x)
        if objective.budget is None:
            break


def tqa_batches(objective):
    """第一批是几何梯的 32 个步长，之后每批在相邻两点之间插入对数中点。

    每一批先按起点概率打分再按降序交出，批的组成只由梯本身决定，与评价值无关。
    """
    depth,G = objective.depth,objective.G
    ramp = (np.arange(depth)+.5)/depth
    def seed(dt):
        return np.r_[dt*ramp,(-dt*(1-ramp)+np.pi/2)%np.pi-np.pi/2]
    ladder = np.geomspace(.05*min(1.,G),min(5.,G),DEFAULT_CONFIG["tqa_seed_count"])
    fresh = ladder
    while True:
        scored = [(objective.evaluate(seed(dt)),seed(dt)) for dt in fresh]
        yield [x for _,x in sorted(scored,key=lambda item:-item[0])]
        fresh = np.sqrt(ladder[:-1]*ladder[1:])
        ladder = np.sort(np.r_[ladder,fresh])


def optimize(objective,method,seed,starts=None):
    """同一上限预算下消费起点：Echo 用扫描批次，TQA 用步长批次，随机基线一直取新起点。"""
    rng = np.random.default_rng(seed)
    depth,G = objective.depth,objective.G
    objective.stage(method)
    if starts is not None:
        consume(objective,starts,local)
        return
    if method == "TQA-LBFGSB":
        consume(objective,tqa_batches(objective),local)
        return
    restart = 0
    while True:
        objective.check()
        if method == "INTERP-LBFGSB":
            x = random_start(rng,G,1,restart%2==1)
            for p in range(1,depth+1):
                if p>1:
                    x = interpolate(x)
                x = local(objective,x)
                x[p:] = (x[p:]+np.pi/2)%np.pi-np.pi/2
        elif method == "RI-LBFGSB":
            local(objective,random_start(rng,G,depth,restart%2==1))
        else:
            raise ValueError(method)
        restart += 1
        if objective.budget is None and (method == "INTERP-LBFGSB" or restart >= DEFAULT_CONFIG["batch_size"]):
            break
