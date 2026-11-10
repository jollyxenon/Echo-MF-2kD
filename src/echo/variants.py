"""三种 Echo 的统一命名、固定公式与多起点批次。"""
import numpy as np

from echo.flat_scan import flat_scan
from echo.flat_synthesis import synthesize
from echo.core import fixed_scan, angles, probabilities, start_batches, validate_depth

METHOD_NAMES = {
    "Flat-Top": "平顶 Echo",
    "Alternating-Phase": "交替相位 Echo",
    "Endpoint-Matched": "端点匹配 Echo",
}


def load_formula(name, depth):
    """按任意正偶数深度构造配方，不依赖角度查表或实验产物。"""
    validate_depth(depth)
    if name not in METHOD_NAMES:
        raise ValueError(f"未知 Echo 方法：{name}")
    pairs = depth // 2
    if name == "Flat-Top":
        mixers,diagnostics = synthesize(pairs)
        return dict(name=name, method_name=METHOD_NAMES[name],
                    ratios=[1]+[2]*(pairs-1), mixers=mixers.tolist(),synthesis=diagnostics)
    if name == "Alternating-Phase":
        return dict(name=name, method_name=METHOD_NAMES[name], grid_points=4096*pairs+1,
                    ratios=[(-1)**(pairs-j) for j in range(1, pairs+1)],
                    mixers=[0.]*(pairs-1)+[np.pi/4])
    tau = 0.5
    length = 2*pairs-1
    if pairs == 1:
        # 单对只能得到 u*t；采用最大幅度响应，端点相等条件不适用于此退化情形。
        B = np.pi/2
    else:
        u = np.sin(np.pi/length)/np.sqrt(1+2*tau*np.cos(np.pi/length)+tau*tau)
        B = float(np.arcsin(u))
    return dict(name=name, method_name=METHOD_NAMES[name], grid_points=4097,
                ratios=[1]+[2]*(pairs-1),
                mixers=[(-1)**(pairs-j)*B for j in range(1, pairs)]+[B/2],
                tau=tau, B=B,endpoint_condition=pairs>=2)


def generate(objective, name):
    """生成所选偶数层 Echo，记录最好的扫描候选，返回配方与按批排好的起点。

    cost 向量统一写成 a*normalized，a 是带符号的 max|gamma|，所以三种变体的
    候选池、间距口径与角度展开只有一处实现。
    """
    objective.stage("formula_construction")
    formula = load_formula(name, objective.depth)
    objective.event("formula",formula=formula)
    objective.stage("scan")
    depth = objective.depth
    mixers = np.asarray(formula["mixers"])
    ratios = np.asarray(formula["ratios"],dtype=float)
    scale = max(abs(ratios))
    normalized = ratios/scale
    if name == "Flat-Top":
        a, score, cells = flat_scan(objective.h,objective.s_star,depth//2,objective.G/scale)
        a = a*scale
        objective.event("scan_candidates",cells=cells,candidates=len(a))
    else:
        pool = fixed_scan(objective.h,objective.s_star,formula["ratios"],mixers,formula["grid_points"])
        a, score = pool["coordinates"],pool["score"]
        objective.event("scan_candidates",probability_evaluations=pool["evaluations"],
                        peaks_found=pool["peaks_found"],peaks_refined=pool["peaks_refined"],
                        candidates=len(a))
    batches = list(start_batches(a,score))
    best = batches[0][0]*normalized
    objective.record(angles(best,mixers,depth),
                     float(probabilities(best,mixers,objective.h,objective.s_star)),
                     "scan",analytic=True)
    starts = [[angles(value*normalized,mixers,depth) for value in batch] for batch in batches]
    objective.finish_stage()
    objective.event("start_batches",batches=[len(batch) for batch in starts],pool=len(a))
    return formula, starts
