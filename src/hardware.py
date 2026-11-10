"""物理核枚举与负载感知分配。"""
from __future__ import annotations

import os
from pathlib import Path
import time


def cpu_times() -> dict[int, tuple[int, int]]:
    """读取每个逻辑 CPU 的累计忙时间和空闲时间。"""
    values = {}
    for line in Path("/proc/stat").read_text().splitlines():
        fields = line.split()
        if not fields or not fields[0].startswith("cpu") or not fields[0][3:].isdigit():
            continue
        numbers = list(map(int, fields[1:9]))
        values[int(fields[0][3:])] = (sum(numbers), numbers[3] + numbers[4])
    return values


def topology(cpu: int) -> tuple[str, str]:
    """返回逻辑 CPU 所属的物理封装号与核编号。"""
    folder = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    return (folder / "physical_package_id").read_text().strip(), (folder / "core_id").read_text().strip()


def physical_cores() -> list[int]:
    """按 (封装号, 核编号) 去重，每个物理核保留一个逻辑 CPU。"""
    seen, cores = set(), []
    for cpu in sorted(os.sched_getaffinity(0)):
        key = topology(cpu)
        if key not in seen:
            seen.add(key)
            cores.append(cpu)
    return cores


def choose_cpus(workers: int) -> list[int]:
    """在物理核中挑选当前累计空闲比例最高的 workers 个。"""
    before, core_list = cpu_times(), physical_cores()
    time.sleep(0.25)
    after = cpu_times()
    ranked = sorted(
        core_list,
        key=lambda cpu: (after[cpu][1] - before[cpu][1]) / max(after[cpu][0] - before[cpu][0], 1),
        reverse=True,
    )
    if len(ranked) < workers:
        raise ValueError(f"可用物理核 {len(ranked)} 个，不足 {workers} 个")
    return ranked[:workers]
