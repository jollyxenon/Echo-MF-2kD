"""冻结模型后统一最小化能量，独立评分优化前后 P_gs。"""
from __future__ import annotations

import time

import torch

from learning.inference import load_predictor
from learning.optimization import OPTIONS, refine
from learning.scoring import score_pair, summarize_records
from records import append_jsonl, emit, load_npz, save_json, split_indices


def evaluate_one(path, data, indices, log_path, grid_points: int = 128, decoder: str = "1d") -> dict:
    """逐题只传入 J,h 预测与优化，返回后才允许答案进入评分器。"""
    load_started = time.perf_counter()
    predict, checkpoint = load_predictor(path / "model.pt", grid_points, decoder=decoder)
    load_seconds = time.perf_counter() - load_started
    log_path.write_text("", encoding="utf-8")
    records = []
    for index in indices:
        J, h = data["J"][index].astype(float), data["h"][index].astype(float)
        initial, prediction = predict(J, h)
        stage1_seconds = prediction.pop("seconds")
        optimized = refine(J, h, initial)
        score = score_pair(J, h, int(data["target_index"][index]), initial, optimized["angles"])
        if abs(score["energy"] - optimized["energy"]) > 1e-10:
            raise ValueError("优化器返回能量与独立评分不一致")
        if score["energy"] > score["initial_energy"] + 1e-10:
            raise ValueError("返回参数的能量高于起点")
        row = dict(
            id=str(data["ids"][index]),
            initial_angles=initial.tolist(),
            angles=optimized["angles"],
            initial_probability=score["initial_probability"],
            probability=score["probability"],
            initial_energy=score["initial_energy"],
            energy=score["energy"],
            stage1_calls=0,
            stage2_calls=optimized["calls"],
            stage1_seconds=stage1_seconds,
            stage2_seconds=optimized["seconds"],
            total_seconds=stage1_seconds + optimized["seconds"],
            status=optimized["status"],
            distribution_residual=score["distribution_residual"],
            energy_residual=score["energy_residual"],
            norm_error=score["norm_error"],
            **prediction,
        )
        append_jsonl(log_path, row)
        records.append(row)
        if len(records) % 32 == 0:
            emit(dict(job=path.name, evaluated=len(records), total=len(indices)))
    direct = checkpoint["architecture"] != "Echo-Prior"
    return dict(
        job=path.name, model=str(path / "model.pt"), architecture=checkpoint["architecture"],
        method=checkpoint["method"], n=checkpoint["n"], depth=checkpoint["depth"],
        seed=checkpoint["seed"], objective="energy",
        decoder=None if direct else decoder, grid_points=None if direct else grid_points,
        wall_clock_cap=None, optimizer=OPTIONS, model_load_seconds=load_seconds,
        records_path=str(log_path), **summarize_records(records),
    )


def evaluate(dataset, models, output, split: str = "test", threads: int = 1,
             grid_points: int = 128, decoder: str = "1d") -> list[dict]:
    """冻结后统一评价，测试概率不参与参数选择或停止；每个模型写一个扁平记录文件。"""
    torch.set_num_threads(threads)
    data = load_npz(dataset)
    indices = split_indices(data)[split]
    if not len(indices):
        raise ValueError(f"没有划分 {split}")
    output.mkdir(parents=True, exist_ok=True)
    model_dirs = sorted(models)
    for path in model_dirs:
        if not (path / "model.pt").exists():
            raise ValueError(f"模型目录缺少 model.pt：{path}")
    jobs = []
    for path in model_dirs:
        record = evaluate_one(path, data, indices, output / f"{path.name}.jsonl",
                              grid_points, decoder=decoder)
        save_json(output / f"{path.name}.json", record)
        jobs.append(record)
        emit(record)
    return jobs
