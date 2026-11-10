"""预算外完整线路复算：重算最终概率与各模块最好参数，不更改候选选择。"""
from __future__ import annotations

from time import perf_counter
import numpy as np

from basis import flips_for, spin_table, target_index
from compare.autodiff import evolve
from instances.benchmark import energy_table


def verify(result,J,h,target,ground):
    """预算外复算最终概率及各模块最好参数，不更改候选选择。"""
    start = perf_counter()
    energy = energy_table(spin_table(len(h)),J,h,0.)
    flips = flips_for(len(h))
    state = evolve(np.array(result["angles"]),energy,flips)
    index = target_index(target)
    residual = abs(float(abs(state[index])**2)-result["probability"])
    module_residuals = []
    for module in result["modules"]:
        candidate = evolve(np.array(module["best_angles"]),energy,flips)
        module_residuals.append(abs(float(abs(candidate[index])**2)-module["after_probability"]))
    result["verification"] = dict(outside_budget=True,probability_residual=residual,
        module_residual=max(module_residuals,default=0.),ground_probability=float(np.sum(abs(state[ground])**2)),
        norm=float(np.vdot(state,state).real),seconds=perf_counter()-start)
    if not np.all(np.isfinite(result["angles"])) or not np.isfinite(residual) or abs(float(np.vdot(state,state).real)-1)>1e-10:
        raise ValueError("非有限结果或态归一化错误")
    if residual>1e-10 or max(module_residuals,default=0.)>1e-10:
        raise ValueError("完整线路复算失败")
