"""在统一的上限预算下跑满六行方法，逐行保留完整改进轨迹。"""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import json
from pathlib import Path
import random
import resource
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
import numpy as np
from compare.optimizers import optimize
from compare.runtime import Objective,BudgetExpired
from compare.verify import verify
from echo.variants import generate, METHOD_NAMES
from echo.core import validate_depth
from records import save_json

# 六方法口径：三种 Echo 配方各占一行，先用该配方构造自己的起点批次再接 L-BFGS-B；
# 三条基线不接受 Echo 起点。每行是（行文件名，优化器，Echo 配方或 None）。
ECHO_ROWS = ("Flat-Top","Alternating-Phase","Endpoint-Matched")
BASELINE_ROWS = ("RI-LBFGSB","TQA-LBFGSB","INTERP-LBFGSB")
ROWS = ECHO_ROWS+BASELINE_ROWS


def solve(J,h,target,depth,variant,stream,cap,method,seed):
    """同一预算对象内先构造起点批次再接优化器，构造耗时计入 cap。"""
    objective = Objective(J,h,target,depth,stream,cap)
    construction,formula,starts = None,None,None
    try:
        if variant is not None:
            formula,starts = generate(objective,variant)
            construction = objective.elapsed()
        # Echo 行只消费自己的扫描批次；基线各自生成起点。
        optimize(objective,method,seed,starts)
        status = "completed"
    except BudgetExpired:
        # 构造本身不可中断，越界只能在第一个记录点发现，此时没有可用参数。
        status = "construction_over_cap" if variant is not None and construction is None else "budget_exhausted"
    result = objective.result()
    result.update(status=status,construction_seconds=construction)
    return result,formula


def run(instance,depth,seed,cap,output):
    """按固定种子打乱六行，全部使用同一个上限预算，逐行复算并落盘。"""
    validate_depth(depth)
    output.mkdir(parents=True,exist_ok=False)
    with np.load(instance) as data:
        J,h,target,ground = (data[k].copy() for k in ["J","h","s_star","ground_indices"])
    metadata = json.loads(instance.with_suffix(".json").read_text())
    results,statuses,constructions,formulas = {},{},{},{}
    tasks = [(row,"LBFGSB",row) for row in ECHO_ROWS]+[(row,row,None) for row in BASELINE_ROWS]
    random.Random(seed+depth*100).shuffle(tasks)
    for name,method,variant in tasks:
        with (output/(name+".jsonl")).open("w") as stream:
            result,formula = solve(J,h,target,depth,variant,stream,cap,method,seed)
        verify(result,J,h,target,ground)
        save_json(output/(name+".json"),result)
        results[name],statuses[name] = result,result["status"]
        constructions[name] = result["construction_seconds"]
        if variant is not None:
            formulas[variant] = formula
    save_json(output/"configuration.json",dict(instance=metadata,depth=depth,seed=seed,echo_rows=list(ECHO_ROWS),
         formulas=formulas,gamma_bounds=[-10*np.pi,10*np.pi],
         gradient_backend="HIPS autograd",known_target=True,budget_cap_seconds=cap))
    save_json(output/"result.json",dict(instance=metadata["id"],family=metadata["family"],n=len(h),depth=depth,
        seed=seed,budget_cap_seconds=cap,rows=list(ROWS),method_names={row:METHOD_NAMES.get(row,row) for row in ROWS},
        probabilities={k:v["probability"] for k,v in results.items()},statuses=statuses,
        construction_seconds=constructions,local_starts={k:v["local_starts"] for k,v in results.items()},
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        python=platform.python_version(),numpy=np.__version__))


def main():
    """单任务入口，未完成目录禁止静默覆盖。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance",type=Path,required=True)
    parser.add_argument("--depth",type=int,required=True,help="正偶数层数 p=2k")
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--cap",type=float,default=30.0,help="每行方法的墙钟上限秒数（传入 0 表示无截断自然收敛轨 budget=None）")
    parser.add_argument("--unconstrained",action="store_true",help="启用无截断自然收敛轨（budget = None）")
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.cap < 0:
        raise ValueError("上限预算不能为负")
    cap = None if (args.unconstrained or args.cap == 0) else args.cap
    run(args.instance,args.depth,args.seed,cap,args.output)


if __name__=="__main__":
    main()
