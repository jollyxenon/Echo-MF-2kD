"""读取比较目录里已有的改进轨迹，按任意时刻截断并打印对比表。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_runs(root):
    """单题目录直接使用，批量目录取其中含结果的子目录。"""
    if (root/"result.json").exists():
        return [root]
    runs = sorted(p for p in root.glob("*") if (p/"result.json").exists())
    if not runs:
        raise ValueError(f"未找到含 result.json 的比较目录：{root}")
    return runs


def sample(record,times):
    """返回每个时刻之前已完成的最好的真实候选，尚未产出时为 None。"""
    # 首条历史是均匀分布的合成起点，不是候选，必须跳过。
    history = [entry for entry in record["history"] if not entry["synthetic"]]
    values = []
    for time in times:
        done = [entry["probability"] for entry in history if entry["time"]<=time]
        values.append(done[-1] if done else None)
    return values


def median(values):
    """返回中位数，空集合返回 None。"""
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered)//2
    return ordered[middle] if len(ordered)%2 else (ordered[middle-1]+ordered[middle])/2


def main():
    """按 --times 截断轨迹，输出跨题中位数概率表。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,required=True,help="单题比较目录或其上层批量目录")
    parser.add_argument("--times",type=float,nargs="+",required=True,help="要读取的时刻，秒")
    args = parser.parse_args()
    times = sorted(args.times)
    if times[0] <= 0:
        raise ValueError("时刻必须为正")
    runs = load_runs(args.runs.resolve())
    summaries = [json.loads((run/"result.json").read_text()) for run in runs]
    cap = min(summary["budget_cap_seconds"] for summary in summaries)
    if times[-1] > cap:
        raise ValueError(f"时刻 {times[-1]} 超出目录的上限预算 {cap}，曲线无法向右读取")
    rows = summaries[0]["rows"]
    # 每行每时刻收集 (n, 概率)；尚未产出的时刻不参与统计，但要单独列出参与次数。
    samples = {row:[(summary["n"],sample(json.loads((run/(row+".json")).read_text()),times))
                    for run,summary in zip(runs,summaries)] for row in rows}
    cells = {row:[[p for _,values in samples[row] for p in [values[position]] if p is not None]
                  for position in range(len(times))] for row in rows}
    print(f"{len(runs)} 次运行，上限预算 {cap} s，时间网格 {times}")

    print("\n中位数最好概率（按 n 分组请在统计表中读取）")
    print(f"{'方法':16s}"+"".join(f"{t:>15}" for t in times)+f"{'产出率':>9}")
    for row in rows:
        line = f"{row:16s}"
        for values in cells[row]:
            value = median(values)
            line += f"{'未产出':>15}" if value is None else f"{value:>15.5f}"
        last = [values[-1] for _,values in samples[row]]
        print(line+f"{sum(p is not None for p in last)/len(last):>9.0%}")

    # 参与统计的运行数不等时，不同时刻的中位数来自不同样本集，曲线可能不单调。
    if any(len(values) < len(runs) for row in rows for values in cells[row]):
        print("\n参与统计的运行数（少于总数时各时刻样本集不同，中位数可能不单调）")
        print(f"{'方法':16s}"+"".join(f"{t:>15}" for t in times))
        for row in rows:
            print(f"{row:16s}"+"".join(f"{len(values):>15d}" for values in cells[row]))


if __name__=="__main__":
    main()
