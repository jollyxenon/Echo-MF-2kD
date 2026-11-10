"""全项目标准实验执行与统计汇总脚本（严格遵循 docs/experiment-settings.md）。

涵盖标准子任务：
  1. n10_recompute          : 基于 runs/models_and_eval/ 评测记录，以 Bootstrap(4000, seed=0) 重算 n=10 汇总；
  2. n10_2kd_test1024       : 在 data/mwis/n10/dataset.npz 全部 1,024 道测试题（p=8,10,12）上评测 Echo-MF-2kD 与 Echo-Prior-Flat-2kD (seed=0)；
  3. cross_scale_mwis       : 在 data/mwis/ 跨规模测试集 n∈{6,8,10,12,14}（每规模 1,024 题，p=10）上统一评测离线与在线跨规模扩展、5 个流形变体消融及 12 种在线基线；
  4. five_families_n11_n12  : 在 data/benchmark/ 五家族 n=11,12（500 题，p=10）上评测无截断 Echo-Flat-1D、Echo-Flat-2kD 与 TQA；
  5. shot_noise             : 在 data/mwis/n10/dataset.npz 测试集 1,024 题（p=10）上评测 Echo-Prior-Flat、Echo-MF-2kD、Echo-MF-Flat 与 TQA 的散粒噪声模式 A 与模式 B；
  6. ising_online_benchmark : 在 MWIS、Grid Ising、Sparse Ising、Dense Ising 上评测全部 12 种在线无监督基线方法。
"""
from __future__ import annotations

import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[k] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import minimize
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from basis import flips_for, spin_table, target_index as spin_target_index
from compare.autodiff import evolve, probability_gradient
from compare.optimizers import DEFAULT_CONFIG, bounds
from echo.variants import load_formula
from echo.multiscalar import VARIANTS, generate_stage1_angles, resolve_prior_direction
from instances.benchmark import energy_table
from learning.aggregation import aggregate, bootstrap_ci
from learning.baselines import METHODS as BASELINE_METHODS, run_ma_qaoa, run_warm_start_qaoa, select_initial
from learning.echo_decoder import decode_mean_field_multiscalar, expand_scalar
from learning.echo_prior import build_features, echo_prior_probabilities, train_and_predict_manifold_ablations
from learning.echo_prior_pipeline import restore as restore_prior
from learning.features import mean_field
from learning.optimization import EnergyProblem, fast_dist, shot_noise_adam_optimize
from learning.scoring import score_angles
from records import append_jsonl, cache_path, read_jsonl, save_json, write_jsonl

PRIOR_FLAT_SEED0_WEIGHTS = {
    depth: ROOT / f"runs/models_and_eval/models/n10-p{depth:02d}__Echo-Prior__Flat-Top__seed0/model.pt"
    for depth in (8, 10, 12)
}

PRIOR_FLAT_SEED0_EVALS = {
    depth: ROOT / f"runs/models_and_eval/evaluation/n10-p{depth:02d}__Echo-Prior__Flat-Top__seed0.jsonl"
    for depth in (8, 10, 12)
}

SHOT_LEVELS_STAGE1 = [16, 32, 64, 128, 256, 512, 1024]
SHOT_LEVELS_PARAMETER_SHIFT = [64, 256, 1024, 4096]


def summarize_pct(values: np.ndarray, replicates: int = 4000, seed: int = 0) -> dict:
    """对百分比数组执行 Bootstrap(4000, seed=0)，同时返回绝对 95% 区间与相对均值的上下偏移量。"""
    arr = np.asarray(values, dtype=np.float64)
    mean_val, lo, hi = bootstrap_ci(arr, replicates=replicates, seed=seed)
    return dict(
        mean_pct=round(mean_val, 2),
        ci95=[round(lo, 2), round(hi, 2)],
        ci_offset=[round(mean_val - lo, 2), round(hi - mean_val, 2)],
    )


def summarize_paired_diff(target_pct: np.ndarray, base_pct: np.ndarray,
                          replicates: int = 4000, seed: int = 0) -> dict:
    """对两组同题百分比得分差 (target - base) 执行 Bootstrap(4000, seed=0)。"""
    diff = np.asarray(target_pct, dtype=np.float64) - np.asarray(base_pct, dtype=np.float64)
    mean_diff, lo, hi = bootstrap_ci(diff, replicates=replicates, seed=seed)
    return dict(
        mean_diff_pp=round(mean_diff, 2),
        ci95=[round(lo, 2), round(hi, 2)],
        ci_offset=[round(mean_diff - lo, 2), round(hi - mean_diff, 2)],
        win_rate_pct=round(float(np.mean(diff > 0.0) * 100.0), 1),
    )


def load_jsonl_sorted(path: Path) -> list[dict]:
    """读取 JSONL 文件并按题号 id 升序排列。"""
    return sorted(read_jsonl(path), key=lambda row: str(row["id"]))


def load_mwis_benchmark_instances(n: int, count: int) -> list[dict]:
    """加载跨规模测试题库（n∈{6,8,10,12,14} 来自 data/mwis/n{n:02d}/dataset.npz 的 test 划分，其余来自 data/benchmark/mwis）。"""
    mwis_npz = ROOT / f"data/mwis/n{n:02d}/dataset.npz"
    if mwis_npz.exists():
        with np.load(mwis_npz, allow_pickle=False) as raw:
            test_idx = np.flatnonzero(raw["split"] == "test")[:count]
            if len(test_idx) != count:
                raise ValueError(f"{mwis_npz} 中 test 划分题数 {len(test_idx)} 不足请求的 {count}")
            return [
                dict(
                    index=int(idx),
                    id=str(raw["ids"][idx]),
                    J=raw["J"][idx].astype(np.float64),
                    h=raw["h"][idx].astype(np.float64),
                    target_index=int(raw["target_index"][idx]),
                    s_star=raw["s_star"][idx].astype(np.float64),
                )
                for idx in test_idx
            ]

    files = sorted((ROOT / "data/benchmark/mwis").glob(f"mwis_n{n:02d}_*.npz"))[:count]
    if len(files) != count:
        raise FileNotFoundError(f"data/benchmark/mwis 下 n={n} 题数不足 {count}")
    instances = []
    for idx, f in enumerate(files):
        with np.load(f, allow_pickle=False) as d:
            instances.append(dict(
                index=idx,
                id=f"mwis_n{n:02d}_{idx:02d}",
                J=d["J"].astype(np.float64),
                h=d["h"].astype(np.float64),
                target_index=int(d["target_index"]),
                s_star=d["s_star"].astype(np.float64),
            ))
    return instances


def predict_echo_prior_flat_seed0(depth: int, J_batch: np.ndarray, h_batch: np.ndarray) -> np.ndarray:
    """使用 Echo-Prior-Flat (seed=0) 标准冻结权重对任意规模 (J_batch, h_batch) 进行单次零样本前向推理。"""
    model, _ = restore_prior(PRIOR_FLAT_SEED0_WEIGHTS[depth])
    features = build_features(
        np.asarray(J_batch, dtype=np.float64),
        np.asarray(h_batch, dtype=np.float64),
    ).astype(np.float32)
    return echo_prior_probabilities(model, features)


def run_stage2_prob_lbfgsb(init_ang: np.ndarray, energy: np.ndarray, flips: list[np.ndarray],
                           target_idx: int, G: float, depth: int) -> dict:
    """从给定起点运行无墙钟截断的标准 Stage-2 全角度 L-BFGS-B 概率最大化。"""
    started = time.perf_counter()
    init_p = float(np.abs(evolve(init_ang, energy, flips)[target_idx]) ** 2)
    best_x = [np.array(init_ang, dtype=np.float64)]
    best_p = [init_p]
    calls = [0]

    def loss_p(x: np.ndarray) -> tuple[float, np.ndarray]:
        calls[0] += 1
        p, g = probability_gradient(x, energy, flips, target_idx)
        if p > best_p[0]:
            best_x[0], best_p[0] = np.array(x, dtype=np.float64), float(p)
        return -float(p), -np.asarray(g, dtype=np.float64)

    res = minimize(loss_p, init_ang, jac=True, method="L-BFGS-B",
                   bounds=bounds(G, depth), options=DEFAULT_CONFIG["lbfgsb"])
    return dict(initial_angles=np.asarray(init_ang, dtype=np.float64).tolist(),
                angles=best_x[0].tolist(), initial_probability=init_p, probability=best_p[0],
                calls=calls[0], iterations=int(res.nit), seconds=time.perf_counter() - started,
                status="converged" if res.success else "optimizer_stopped")


def stage2_block(initial_angles, optimized: dict, score: dict,
                 stage1_calls: int, stage1_seconds: float, **extra) -> dict:
    """canonical 逐方法记录的唯一字段口径。"""
    return dict(
        initial_angles=initial_angles, angles=optimized["angles"],
        initial_probability=score["initial_probability"], probability=score["probability"],
        initial_energy=score["initial_energy"], energy=score["energy"],
        stage1_calls=int(stage1_calls), stage2_calls=int(optimized["calls"]),
        stage1_seconds=float(stage1_seconds), stage2_seconds=float(optimized["seconds"]),
        total_seconds=float(stage1_seconds) + float(optimized["seconds"]),
        status=optimized["status"], **extra,
    )


# =============================================================================
# 子任务 1: n10_recompute (B-01, B-02)
# =============================================================================
def run_task_n10_recompute(out_dir: Path, workers: int = 24) -> dict:
    """以 Bootstrap(4000, seed=0) 重算 n=10 神经模型、免训练基线与 3 组 1D 流形消融汇总。"""
    print("=== [Task: n10_recompute] Recomputing n=10 summaries with Bootstrap(4000, seed=0) ===", flush=True)
    eval_root = ROOT / "runs/models_and_eval"
    groups = aggregate(eval_root, replicates=4000, seed=0)

    ensure_baseline_records(eval_root, workers=workers)
    baselines_dir = eval_root / "baselines"
    baseline_specs = {
        "Echo-MF-Flat": "Echo-MF-Flat__p{p:02d}.jsonl",
        "Echo-MF-End": "Echo-MF-End__p{p:02d}.jsonl",
        "Echo-MF-Alt": "Echo-MF-Alt__p{p:02d}.jsonl",
        "Echo-Energy": "Echo-Energy__p{p:02d}.jsonl",
        "TQA": "TQA__p{p:02d}.jsonl",
        "RI (1)": "RI-1__p{p:02d}.jsonl",
        "RI (32)": "RI-32__p{p:02d}.jsonl",
    }
    manifold_specs = {
        "Echo-GNN-Scalar": "Echo-GNN-Scalar__p{p:02d}.jsonl",
        "Echo-GNN-Grid": "Echo-GNN-Grid__p{p:02d}.jsonl",
    }

    raw_mwis = np.load(ROOT / "data/mwis/n10/dataset.npz", allow_pickle=False)
    test_idx = np.flatnonzero(raw_mwis["split"] == "test")
    J_test = raw_mwis["J"][test_idx].astype(np.float64)
    h_test = raw_mwis["h"][test_idx].astype(np.float64)
    s_star_test = raw_mwis["s_star"][test_idx].astype(np.float64)

    baselines_summary = {}
    manifold_summary = {}
    product_state_decomposition = {}

    for p in (8, 10, 12):
        p_key = f"p{p:02d}"
        baselines_summary[p_key] = {}
        manifold_summary[p_key] = {}

        # 读取 Echo-Prior-Flat (seed=0) 1,024 题记录
        prior_rows = load_jsonl_sorted(PRIOR_FLAT_SEED0_EVALS[p])
        prior_init = np.array([r["initial_probability"] * 100.0 for r in prior_rows])
        prior_fin = np.array([r["probability"] * 100.0 for r in prior_rows])
        prior_calls = float(np.mean([r["stage2_calls"] for r in prior_rows]))

        # 计算直积态先验命中率（表 D.3）
        pred_pi = predict_echo_prior_flat_seed0(p, J_test, h_test)
        ideal_prod_prob = np.prod(np.where(s_star_test > 0, pred_pi, 1.0 - pred_pi), axis=1) * 100.0
        product_state_decomposition[p_key] = dict(
            ideal_prior_product=summarize_pct(ideal_prod_prob),
            echo_1d_initial=summarize_pct(prior_init),
            stage2_refined=summarize_pct(prior_fin),
            stage2_vs_initial=summarize_paired_diff(prior_fin, prior_init),
        )

        # 汇总免训练基线（表 4）
        tqa_rows = load_jsonl_sorted(baselines_dir / baseline_specs["TQA"].format(p=p))
        tqa_fin = np.array([r["probability"] * 100.0 for r in tqa_rows])
        for name, pattern in baseline_specs.items():
            rows = load_jsonl_sorted(baselines_dir / pattern.format(p=p))
            init_arr = np.array([r["initial_probability"] * 100.0 for r in rows])
            fin_arr = np.array([r["probability"] * 100.0 for r in rows])
            calls_mean = float(np.mean([r["stage2_calls"] for r in rows]))
            baselines_summary[p_key][name] = dict(
                initial=summarize_pct(init_arr),
                final=summarize_pct(fin_arr),
                mean_calls=round(calls_mean, 1),
                vs_tqa=summarize_paired_diff(fin_arr, tqa_fin),
            )

        # 汇总精简后的 3 组 1D 流形消融（原表 D.5：Echo-GNN-Scalar, Echo-GNN-Grid, Echo-Prior-Flat）
        manifold_summary[p_key]["Echo-Prior-Flat"] = dict(
            initial=summarize_pct(prior_init),
            final=summarize_pct(prior_fin),
            mean_calls=round(prior_calls, 1),
            vs_prior_flat=summarize_paired_diff(prior_fin, prior_fin),
        )
        for name, pattern in manifold_specs.items():
            spec_file = baselines_dir / pattern.format(p=p)
            if not spec_file.exists():
                manifold_summary[p_key][name] = None
                continue
            rows = load_jsonl_sorted(spec_file)
            init_arr = np.array([r["initial_probability"] * 100.0 for r in rows])
            fin_arr = np.array([r["probability"] * 100.0 for r in rows])
            calls_mean = float(np.mean([r["stage2_calls"] for r in rows]))
            manifold_summary[p_key][name] = dict(
                initial=summarize_pct(init_arr),
                final=summarize_pct(fin_arr),
                mean_calls=round(calls_mean, 1),
                vs_prior_flat=summarize_paired_diff(fin_arr, prior_fin),
                prior_flat_gain_vs_this=summarize_paired_diff(prior_fin, fin_arr),
            )

    report = dict(
        bootstrap_replicates=4000,
        seed=0,
        neural_groups=groups,
        baselines=baselines_summary,
        manifold_1d_ablation=manifold_summary,
        product_state_decomposition=product_state_decomposition,
    )
    save_json(out_dir / "n10_recompute_summary.json", report)
    print(f"  -> Saved recomputed n=10 summaries to {out_dir / 'n10_recompute_summary.json'}", flush=True)
    return report


# =============================================================================
# 子任务 2: n10_2kd_test1024 (B-03, C-15)
# =============================================================================
def _eval_n10_2kd_worker(task: tuple) -> dict:
    """在单道 n=10 测试题上评测 Echo-MF-2kD 与 Echo-Prior-Flat-2kD (Echo-Flat-2kD)。"""
    idx, inst_id, J, h, target_idx, s_star, depth, pi_mf, pi_prior, norm_r, mixers0, scale = task
    prob = EnergyProblem(J, h)
    G = 10.0 * np.pi
    row = dict(index=int(idx), id=str(inst_id), n=len(h), depth=int(depth))
    for label_name, pi_vec in (("Echo-MF-2kD", pi_mf), ("Echo-Prior-Flat-2kD", pi_prior)):
        stage1_started = time.perf_counter()
        ang, meta = generate_stage1_angles(
            "Echo-Flat-2kD",
            J, h, pi_vec, depth, norm_r=norm_r, mixers0=mixers0, scale=scale, G=G, is_online=True,
        )
        stage1_seconds = time.perf_counter() - stage1_started
        optimized = prob.optimize(ang)
        score = score_angles(prob.energy, prob.flips, target_idx, ang, optimized["angles"])
        row[label_name] = dict(
            top1=stage2_block(ang.tolist(), optimized, score, 0, stage1_seconds),
            stage1_calls=0, stage1_seconds=stage1_seconds,
            bit_acc_pct=float(np.mean(meta["s_target"] == s_star) * 100.0),
            exact_orientation_hit=int(np.array_equal(meta["s_target"], s_star)),
            quadrant_hit=int(meta["quadrant_hit"]),
        )
    return row


def run_task_n10_2kd_test1024(out_dir: Path, workers: int = 24, depths: tuple[int, ...] = (8, 10, 12)) -> dict:
    """在 data/mwis/n10/dataset.npz 全部 1,024 道测试题（p=8,10,12）上评测 Echo-MF-2kD 与 Echo-Prior-Flat-2kD (seed=0)。"""
    print(f"=== [Task: n10_2kd_test1024] Evaluating Staged-2kD-Flat on 1,024 test instances (depths={depths}) ===", flush=True)
    raw = np.load(ROOT / "data/mwis/n10/dataset.npz", allow_pickle=False)
    test_idx = np.flatnonzero(raw["split"] == "test")
    J_test = raw["J"][test_idx].astype(np.float64)
    h_test = raw["h"][test_idx].astype(np.float64)
    target_test = raw["target_index"][test_idx].astype(int)
    s_star_test = raw["s_star"][test_idx].astype(np.float64)
    ids_test = raw["ids"][test_idx].astype(str)

    pi_mf_all = (1.0 + mean_field(J_test, h_test, 2.0)) / 2.0
    summary = {"bootstrap_replicates": 4000, "seed": 0, "count": len(test_idx), "depths": {}}

    for depth in depths:
        t0 = time.perf_counter()
        cache_jsonl = cache_path(out_dir, f"n10_2kd_test1024_p{depth:02d}")
        records = read_jsonl(cache_jsonl) if cache_jsonl.exists() else []
        if len(records) != len(test_idx):
            pi_prior_all = predict_echo_prior_flat_seed0(depth, J_test, h_test)
            formula = load_formula("Flat-Top", depth)
            mixers0 = np.asarray(formula["mixers"], dtype=np.float64)
            ratios = np.asarray(formula["ratios"], dtype=np.float64)
            scale = float(np.max(np.abs(ratios)))
            norm_r = ratios / scale

            tasks = [
                (
                    int(test_idx[i]),
                    str(ids_test[i]),
                    J_test[i],
                    h_test[i],
                    int(target_test[i]),
                    s_star_test[i],
                    depth,
                    pi_mf_all[i],
                    pi_prior_all[i],
                    norm_r,
                    mixers0,
                    scale,
                )
                for i in range(len(test_idx))
            ]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                records = list(pool.map(_eval_n10_2kd_worker, tasks, chunksize=16))
            records.sort(key=lambda r: int(r["index"]))
            write_jsonl(cache_jsonl, records)

        # 读取对应的 1D 与 TQA 结果做同题配对比较
        mf_1d_rows = load_jsonl_sorted(
            ROOT / f"runs/models_and_eval/baselines/Echo-MF-Flat__p{depth:02d}.jsonl"
        )
        prior_1d_rows = load_jsonl_sorted(PRIOR_FLAT_SEED0_EVALS[depth])
        tqa_rows = load_jsonl_sorted(
            ROOT / f"runs/models_and_eval/baselines/TQA__p{depth:02d}.jsonl"
        )
        mf_1d_fin = np.array([r["probability"] * 100.0 for r in mf_1d_rows])
        prior_1d_fin = np.array([r["probability"] * 100.0 for r in prior_1d_rows])
        tqa_fin = np.array([r["probability"] * 100.0 for r in tqa_rows])

        records_sorted = sorted(records, key=lambda r: str(r["id"]))
        p_entry = {"depth": depth, "seconds": round(time.perf_counter() - t0, 1)}
        for m_key, base_1d in [("Echo-MF-2kD", mf_1d_fin), ("Echo-Prior-Flat-2kD", prior_1d_fin)]:
            init_arr = np.array([r[m_key]["top1"]["initial_probability"] for r in records_sorted]) * 100.0
            fin_arr = np.array([r[m_key]["top1"]["probability"] for r in records_sorted]) * 100.0
            bacc = float(np.mean([r[m_key]["bit_acc_pct"] for r in records_sorted]))
            qhit = float(np.mean([r[m_key]["quadrant_hit"] * 100.0 for r in records_sorted]))
            calls = float(np.mean([r[m_key]["top1"]["stage2_calls"] for r in records_sorted]))
            s1_ms = float(np.mean([r[m_key]["stage1_seconds"] * 1000.0 for r in records_sorted]))
            mis_fin = np.array([
                r[m_key]["top1"]["probability"] * 100.0
                for r in records_sorted
                if not r[m_key]["exact_orientation_hit"]
            ])
            p_entry[m_key] = dict(
                initial=summarize_pct(init_arr),
                final=summarize_pct(fin_arr),
                bit_acc_pct=round(bacc, 2),
                quadrant_hit_pct=round(qhit, 1),
                mean_calls=round(calls, 1),
                mean_stage1_ms=round(s1_ms, 2),
                misoriented_subset_count=int(len(mis_fin)),
                misoriented_subset_final=summarize_pct(mis_fin) if len(mis_fin) > 0 else None,
                vs_1d=summarize_paired_diff(fin_arr, base_1d),
                vs_tqa=summarize_paired_diff(fin_arr, tqa_fin),
            )
        summary["depths"][f"p{depth:02d}"] = p_entry
        print(
            f"  [p={depth:2d}] Echo-MF-2kD: {p_entry['Echo-MF-2kD']['initial']['mean_pct']:.2f}% -> {p_entry['Echo-MF-2kD']['final']['mean_pct']:.2f}% | "
            f"Echo-Prior-Flat-2kD: {p_entry['Echo-Prior-Flat-2kD']['initial']['mean_pct']:.2f}% -> {p_entry['Echo-Prior-Flat-2kD']['final']['mean_pct']:.2f}%",
            flush=True,
        )

    save_json(out_dir / "n10_2kd_test1024_summary.json", summary)
    return summary


# =============================================================================
# 子任务 3: cross_scale_mwis (B-04, B-05) & 子任务 4: five_families_n11_n12 (B-06)
# =============================================================================
def _eval_offline_scaling_worker(task: tuple) -> dict:
    """在单道实例上评测离线无截断 1D-Flat (Top-1)、Staged-2kD-Flat (V4 Top-1 & Top-2) 与 TQA (Top-1 & Top-2)。"""
    inst, depth, run_all_variants = task
    J, h, target_idx, s_star = inst["J"], inst["h"], inst["target_index"], inst["s_star"]
    n = len(h)
    G = 10.0 * np.pi
    energy = energy_table(spin_table(n), J, h, 0.0)
    flips = flips_for(n)
    true_pi = (s_star + 1.0) / 2.0

    formula = load_formula("Flat-Top", depth)
    mixers0 = np.asarray(formula["mixers"], dtype=np.float64)
    ratios = np.asarray(formula["ratios"], dtype=np.float64)
    scale = float(np.max(np.abs(ratios)))
    norm_r = ratios / scale

    def run_candidate(angles, stage1_calls: int, stage1_seconds: float) -> dict:
        """从给定起点精修一次并返回统一字段。"""
        optimized = run_stage2_prob_lbfgsb(angles, energy, flips, target_idx, G, depth)
        score = score_angles(energy, flips, target_idx, angles, optimized["angles"])
        return stage2_block(np.asarray(angles).tolist(), optimized, score,
                            stage1_calls, stage1_seconds)

    # 1. Echo-Flat-1D (flat_scan Top-1)
    stage1_started = time.perf_counter()
    ang_1d, meta_1d = generate_stage1_angles(
        "Echo-Flat-1D", J, h, true_pi, depth, norm_r=norm_r, mixers0=mixers0, scale=scale, G=G, is_online=False
    )
    stage1_1d = time.perf_counter() - stage1_started
    block_1d = dict(top1=run_candidate(ang_1d, 0, stage1_1d), stage1_calls=0,
                    stage1_seconds=stage1_1d, quadrant_hit=int(meta_1d["quadrant_hit"]))

    # 2. Echo-Flat-2kD (Top-1 & Top-2)
    stage1_started = time.perf_counter()
    ang_kd, meta_kd = generate_stage1_angles(
        "Echo-Flat-2kD", J, h, true_pi, depth, norm_r=norm_r, mixers0=mixers0, scale=scale, G=G, is_online=False
    )
    stage1_kd = time.perf_counter() - stage1_started
    block_kd = dict(
        top1=run_candidate(ang_kd, 0, stage1_kd),
        top2=run_candidate(meta_kd["top2_angles"], 0, 0.0),
        stage1_calls=0, stage1_seconds=stage1_kd, quadrant_hit=int(meta_kd["quadrant_hit"]),
    )

    # 3. TQA 32 点阶梯 (Top-1 & Top-2)
    stage1_started = time.perf_counter()
    u = (np.arange(depth) + 0.5) / depth
    tqa_cands = []
    for dt in np.geomspace(0.05, 5.0, 32):
        ang = np.r_[dt * u, (-dt * (1.0 - u) + np.pi / 2.0) % np.pi - np.pi / 2.0]
        tqa_cands.append((float(np.abs(evolve(ang, energy, flips)[target_idx]) ** 2), ang))
    tqa_cands.sort(key=lambda item: -item[0])
    stage1_tqa = time.perf_counter() - stage1_started
    block_tqa = dict(
        top1=run_candidate(tqa_cands[0][1], 32, stage1_tqa),
        top2=run_candidate(tqa_cands[1][1], 0, 0.0),
        stage1_calls=32, stage1_seconds=stage1_tqa,
    )

    out = dict(index=int(inst["index"]), id=str(inst["id"]), n=n,
               **{"Echo-Flat-1D": block_1d, "Echo-Flat-2kD": block_kd, "TQA": block_tqa})

    if run_all_variants:
        for var in ("Echo-Flat-kD (Cost-Only)", "Echo-Flat-kD (Guarded)", "Echo-Flat-2kD (Unconstrained)"):
            stage1_started = time.perf_counter()
            ang_v, meta_v = generate_stage1_angles(
                var, J, h, true_pi, depth, norm_r=norm_r, mixers0=mixers0, scale=scale, G=G, is_online=False
            )
            stage1_v = time.perf_counter() - stage1_started
            out[var] = dict(top1=run_candidate(ang_v, 0, stage1_v), stage1_calls=0,
                            stage1_seconds=stage1_v, quadrant_hit=int(meta_v["quadrant_hit"]))
    return out


def _eval_online_baseline_method(
    method: str,
    inst_idx: int,
    J: np.ndarray,
    h: np.ndarray,
    target_idx: int,
    s_star: np.ndarray,
    depth: int,
    prob: EnergyProblem,
    beta: float = 2.0,
) -> dict:
    """在单道实例上按 src/learning/baselines.py 的 12 种在线方法之一执行无监督求解与评分。"""
    if method in ("Classical-MF", "Classical-Cluster"):
        started = time.perf_counter()
        m_mf = mean_field(J[None, ...], h[None, ...], beta)[0]
        if method == "Classical-MF":
            s_pred = np.where(m_mf >= 0.0, 1.0, -1.0)
        else:
            s_pred = resolve_prior_direction(J, h, (1.0 + m_mf) / 2.0)
        e_val = float(0.5 * s_pred @ J @ s_pred + h @ s_pred)
        elapsed = time.perf_counter() - started
        hit = float(spin_target_index(s_pred) == target_idx)
        return dict(initial_angles=None, angles=None,
                    initial_probability=hit, probability=hit,
                    initial_energy=e_val, energy=e_val,
                    stage1_calls=0, stage2_calls=0, stage1_seconds=elapsed,
                    stage2_seconds=0.0, total_seconds=elapsed, status="discrete",
                    bit_acc_pct=float(np.mean(s_pred == s_star) * 100.0),
                    exact_orientation_hit=0)
    if method in ("Warm-Start QAOA", "ma-QAOA"):
        res = (run_warm_start_qaoa(J, h, depth, target_idx, beta_mf=beta)
               if method == "Warm-Start QAOA" else run_ma_qaoa(J, h, depth, target_idx, seed=0))
        return dict(initial_angles=res["initial_angles"], angles=res["angles"],
                    initial_probability=res["initial_probability"], probability=res["probability"],
                    initial_energy=res["initial_energy"], energy=res["energy"],
                    stage1_calls=res["stage1_calls"], stage2_calls=res["stage2_calls"],
                    stage1_seconds=res["stage1_seconds"], stage2_seconds=res["stage2_seconds"],
                    total_seconds=res["stage1_seconds"] + res["stage2_seconds"],
                    status=res["status"], exact_orientation_hit=0)
    method_idx = BASELINE_METHODS.index(method)
    rng = np.random.default_rng([0, int(inst_idx), depth, method_idx])
    stage1_started = time.perf_counter()
    init_ang, calls, extra = select_initial(method, depth, J, h, prob, rng, beta=beta, grid_points=128)
    stage1_seconds = time.perf_counter() - stage1_started
    optimized = prob.optimize(init_ang)
    score = score_angles(prob.energy, prob.flips, target_idx, init_ang, optimized["angles"])
    return stage2_block(init_ang.tolist(), optimized, score, calls, stage1_seconds,
                        exact_orientation_hit=0, **extra)


def _eval_single_baseline_task(task: tuple) -> dict:
    """在单道测试题上评测单一基线方法。"""
    method, inst_idx, inst_id, J, h, target_idx, s_star, depth = task
    prob = EnergyProblem(J, h)
    res = _eval_online_baseline_method(method, inst_idx, J, h, target_idx, s_star, depth, prob)
    return dict(id=str(inst_id), index=int(inst_idx), **res)


def _eval_ablation_scalar_task(task: tuple) -> dict:
    """在单道测试题上从给定 1D 平顶标量展开并运行 Stage-2 能量优化。"""
    inst_idx, inst_id, J, h, target_idx, depth, scalar_a = task
    prob = EnergyProblem(J, h)
    init_ang = expand_scalar("Flat-Top", depth, float(scalar_a))
    optimized = prob.optimize(init_ang)
    score = score_angles(prob.energy, prob.flips, target_idx, init_ang, optimized["angles"])
    return dict(
        id=str(inst_id),
        index=int(inst_idx),
        **stage2_block(init_ang.tolist(), optimized, score, 0, 0.0, scalar=float(scalar_a)),
    )


def ensure_baseline_records(eval_root: Path, workers: int = 24) -> None:
    """确保 n=10 测试集上的免训练基线与 1D 流形消融（p=8, 10, 12）评测记录完整生成。"""
    baselines_dir = eval_root / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)
    baseline_specs = {
        "Echo-MF-Flat": "Echo-MF-Flat__p{p:02d}.jsonl",
        "Echo-MF-End": "Echo-MF-End__p{p:02d}.jsonl",
        "Echo-MF-Alt": "Echo-MF-Alt__p{p:02d}.jsonl",
        "Echo-Energy": "Echo-Energy__p{p:02d}.jsonl",
        "TQA": "TQA__p{p:02d}.jsonl",
        "RI (1)": "RI-1__p{p:02d}.jsonl",
        "RI (32)": "RI-32__p{p:02d}.jsonl",
    }
    raw_mwis = np.load(ROOT / "data/mwis/n10/dataset.npz", allow_pickle=False)
    train_idx = np.flatnonzero(raw_mwis["split"] == "train")
    test_idx = np.flatnonzero(raw_mwis["split"] == "test")
    J_train = raw_mwis["J"][train_idx].astype(np.float64)
    h_train = raw_mwis["h"][train_idx].astype(np.float64)
    J_test = raw_mwis["J"][test_idx].astype(np.float64)
    h_test = raw_mwis["h"][test_idx].astype(np.float64)
    target_test = raw_mwis["target_index"][test_idx].astype(int)
    s_star_test = raw_mwis["s_star"][test_idx].astype(np.float64)
    ids_test = raw_mwis["ids"][test_idx].astype(str)

    X_train = build_features(J_train, h_train).astype(np.float32)
    X_test = build_features(J_test, h_test).astype(np.float32)

    for p in (8, 10, 12):
        for method, pat in baseline_specs.items():
            out_file = baselines_dir / pat.format(p=p)
            if out_file.exists():
                records = read_jsonl(out_file)
                if len(records) == len(test_idx):
                    continue
            print(f"  [Baseline] Generating {method} (p={p}) on n=10 test instances ({len(test_idx)} items)...", flush=True)
            tasks = [
                (method, int(test_idx[i]), ids_test[i], J_test[i], h_test[i], int(target_test[i]), s_star_test[i], p)
                for i in range(len(test_idx))
            ]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                records = list(pool.map(_eval_single_baseline_task, tasks, chunksize=16))
            records.sort(key=lambda r: int(r["index"]))
            write_jsonl(out_file, records)
            print(f"  [Baseline] Saved {len(records)} records to {out_file}", flush=True)

        f_scalar = baselines_dir / f"Echo-GNN-Scalar__p{p:02d}.jsonl"
        f_grid = baselines_dir / f"Echo-GNN-Grid__p{p:02d}.jsonl"
        if not (f_scalar.exists() and len(read_jsonl(f_scalar)) == len(test_idx)
                and f_grid.exists() and len(read_jsonl(f_grid)) == len(test_idx)):
            print(f"  [Ablation] Training & evaluating Echo-GNN-Scalar and Echo-GNN-Grid (p={p})...", flush=True)
            pi_train = predict_echo_prior_flat_seed0(p, J_train, h_train)
            pred_scalar_a, pred_grid_a = train_and_predict_manifold_ablations(
                X_train, h_train, pi_train, X_test, n_qubits=J_train.shape[1], depth=p, seed=0
            )
            for out_file, a_preds in ((f_scalar, pred_scalar_a), (f_grid, pred_grid_a)):
                tasks = [
                    (int(test_idx[i]), ids_test[i], J_test[i], h_test[i], int(target_test[i]), p, float(a_preds[i]))
                    for i in range(len(test_idx))
                ]
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    records = list(pool.map(_eval_ablation_scalar_task, tasks, chunksize=16))
                records.sort(key=lambda r: int(r["index"]))
                write_jsonl(out_file, records)
                print(f"  [Ablation] Saved {len(records)} records to {out_file}", flush=True)


def _eval_online_scaling_worker(task: tuple) -> dict:
    """在单道实例上评测在线无监督能量优化（MF 与 Echo-Prior-Flat seed=0 在 5 个流形变体下及全部 12 种在线基线）。"""
    inst, depth, pi_mf, pi_prior, variants_to_run, run_all_12_baselines = task
    J, h, target_idx, s_star = inst["J"], inst["h"], inst["target_index"], inst["s_star"]
    n = len(h)
    G = 10.0 * np.pi
    prob = EnergyProblem(J, h)

    formula = load_formula("Flat-Top", depth)
    mixers0 = np.asarray(formula["mixers"], dtype=np.float64)
    ratios = np.asarray(formula["ratios"], dtype=np.float64)
    scale = float(np.max(np.abs(ratios)))
    norm_r = ratios / scale

    out = dict(index=int(inst["index"]), id=str(inst["id"]), n=n)
    out["TQA"] = _eval_online_baseline_method("TQA", int(inst["index"]), J, h, target_idx, s_star, depth, prob)

    for prefix, pi_in in (("MF", pi_mf), ("Prior", pi_prior)):
        for var in variants_to_run:
            stage1_started = time.perf_counter()
            ang, meta = generate_stage1_angles(
                var, J, h, pi_in, depth, norm_r=norm_r, mixers0=mixers0, scale=scale, G=G, is_online=True
            )
            stage1_seconds = time.perf_counter() - stage1_started
            optimized = prob.optimize(ang)
            score = score_angles(prob.energy, prob.flips, target_idx, ang, optimized["angles"])
            out[f"{prefix}_{var}"] = stage2_block(
                ang.tolist(), optimized, score, 0, stage1_seconds,
                bit_acc_pct=float(np.mean(meta["s_target"] == s_star) * 100.0),
                quadrant_hit=int(meta["quadrant_hit"]),
                exact_orientation_hit=int(np.array_equal(meta["s_target"], s_star)),
            )

    if run_all_12_baselines:
        out["baselines_12"] = {}
        for b_method in BASELINE_METHODS:
            if b_method == "TQA":
                out["baselines_12"][b_method] = out["TQA"]
            elif b_method == "Echo-MF-2kD" and "MF_Echo-Flat-2kD" in out:
                out["baselines_12"][b_method] = out["MF_Echo-Flat-2kD"]
            else:
                out["baselines_12"][b_method] = _eval_online_baseline_method(
                    b_method, int(inst["index"]), J, h, target_idx, s_star, depth, prob
                )
    return out


def summarize_offline_records(recs: list[dict], n: int, count: int, elapsed: float) -> dict:
    """将离线跨规模单规模记录汇总为 Bootstrap(4000, seed=0) 统计字典。"""
    def series(method: str, top: str = "top1", field: str = "probability") -> np.ndarray:
        return np.array([r[method][top][field] for r in recs]) * 100.0

    i1d = series("Echo-Flat-1D", field="initial_probability")
    f1d = series("Echo-Flat-1D")
    ikd = series("Echo-Flat-2kD", field="initial_probability")
    fkd1 = series("Echo-Flat-2kD")
    fkd2 = np.maximum(fkd1, series("Echo-Flat-2kD", "top2"))
    qhit = np.array([r["Echo-Flat-2kD"]["quadrant_hit"] for r in recs]) * 100.0
    itqa = series("TQA", field="initial_probability")
    ftqa1 = series("TQA")
    ftqa2 = np.maximum(ftqa1, series("TQA", "top2"))

    s_1d = summarize_pct(f1d)
    s_kd1 = summarize_pct(fkd1)
    s_kd2 = summarize_pct(fkd2)
    s_tqa1 = summarize_pct(ftqa1)
    s_tqa2 = summarize_pct(ftqa2)
    diff_kd1_tqa1 = summarize_paired_diff(fkd1, ftqa1)
    diff_kd2_tqa2 = summarize_paired_diff(fkd2, ftqa2)
    diff_kd1_1d1 = summarize_paired_diff(fkd1, f1d)

    return dict(
        n=n,
        count=count,
        seconds=round(elapsed, 1),
        **{
            "Echo-Flat-1D_init_pct": round(float(np.mean(i1d)), 2),
            "Echo-Flat-1D_top1_pct": s_1d["mean_pct"],
            "Echo-Flat-1D_top1_ci95": s_1d["ci95"],
            "Echo-Flat-1D_top1_ci_offset": s_1d["ci_offset"],
            "Echo-Flat-2kD_init_pct": round(float(np.mean(ikd)), 2),
            "Echo-Flat-2kD_top1_pct": s_kd1["mean_pct"],
            "Echo-Flat-2kD_top1_ci95": s_kd1["ci95"],
            "Echo-Flat-2kD_top1_ci_offset": s_kd1["ci_offset"],
            "Echo-Flat-2kD_top2_pct": s_kd2["mean_pct"],
            "Echo-Flat-2kD_top2_ci95": s_kd2["ci95"],
            "Echo-Flat-2kD_top2_ci_offset": s_kd2["ci_offset"],
            "Echo-Flat-2kD_quadrant_hit_pct": round(float(np.mean(qhit)), 1),
            "TQA_init_pct": round(float(np.mean(itqa)), 2),
            "TQA_top1_pct": s_tqa1["mean_pct"],
            "TQA_top1_ci95": s_tqa1["ci95"],
            "TQA_top1_ci_offset": s_tqa1["ci_offset"],
            "TQA_top2_pct": s_tqa2["mean_pct"],
            "TQA_top2_ci95": s_tqa2["ci95"],
            "TQA_top2_ci_offset": s_tqa2["ci_offset"],
            "Echo-Flat-2kD_top1_vs_TQA_top1_pp": diff_kd1_tqa1["mean_diff_pp"],
            "Echo-Flat-2kD_top1_vs_TQA_top1_ci95": diff_kd1_tqa1["ci95"],
            "Echo-Flat-2kD_top2_vs_TQA_top2_pp": diff_kd2_tqa2["mean_diff_pp"],
            "Echo-Flat-2kD_top2_vs_TQA_top2_ci95": diff_kd2_tqa2["ci95"],
            "Echo-Flat-2kD_top1_vs_Echo-Flat-1D_top1_pp": diff_kd1_1d1["mean_diff_pp"],
            "Echo-Flat-2kD_top1_vs_Echo-Flat-1D_top1_ci95": diff_kd1_1d1["ci95"],
        },
    )


def run_task_cross_scale_mwis(out_dir: Path, workers: int = 24, depth: int = 10,
                              sizes: tuple[tuple[int, int], ...] = (
                                  (6, 1024), (8, 1024), (10, 1024), (12, 1024), (14, 1024),
                              )) -> dict:
    """运行离线与在线 MWIS 跨规模实验（n∈{6,8,10,12,14} 各 1,024 题, p=10，统一 V4 与 Prior-Flat seed=0，Bootstrap 4000 seed=0）。"""
    print(f"=== [Task: cross_scale_mwis] Running Offline & Online MWIS Scaling (p={depth}) ===", flush=True)
    mwis_offline_summary = {}
    online_kd_summary = {}
    ablation_report = {"bootstrap_replicates": 4000, "seed": 0, "offline": {}, "online": {}}

    for n, count in sizes:
        instances = load_mwis_benchmark_instances(n, count)
        run_all_variants = n <= 14
        variants_to_run = VARIANTS if n <= 14 else ("Echo-Flat-1D", "Echo-Flat-2kD")
        run_all_12_baselines = n in (6, 8, 10, 12, 14)

        # 1. 离线跨规模与 5 个流形变体消融
        t0 = time.perf_counter()
        off_cache = cache_path(out_dir, f"cache_offline_mwis_n{n:02d}_p{depth:02d}")
        off_recs = read_jsonl(off_cache) if off_cache.exists() else []
        if len(off_recs) != count:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                off_recs = list(pool.map(_eval_offline_scaling_worker, [(inst, depth, run_all_variants) for inst in instances]))
            off_recs.sort(key=lambda r: int(r["index"]))
            write_jsonl(off_cache, off_recs)

        s_off = summarize_offline_records(off_recs, n, count, time.perf_counter() - t0)
        mwis_offline_summary[f"n{n:02d}"] = s_off

        tqa_fin_off = np.array([r["TQA"]["top1"]["probability"] for r in off_recs]) * 100.0
        tqa_init_off = np.array([r["TQA"]["top1"]["initial_probability"] for r in off_recs]) * 100.0
        off_ablation_entry = {
            "n": n,
            "count": count,
            "TQA": dict(
                init=summarize_pct(tqa_init_off),
                final=summarize_pct(tqa_fin_off),
            ),
        }
        for var in variants_to_run:
            vi = np.array([r[var]["top1"]["initial_probability"] for r in off_recs]) * 100.0
            vf = np.array([r[var]["top1"]["probability"] for r in off_recs]) * 100.0
            qh = float(np.mean([r[var]["quadrant_hit"] * 100.0 for r in off_recs]))
            off_ablation_entry[var] = dict(
                init=summarize_pct(vi),
                final=summarize_pct(vf),
                vs_tqa=summarize_paired_diff(vf, tqa_fin_off),
                quadrant_hit_pct=round(qh, 1),
            )
        ablation_report["offline"][f"n{n:02d}"] = off_ablation_entry
        print(
            f"  [Offline n={n:2d}] Echo-Flat-1D={s_off['Echo-Flat-1D_top1_pct']:5.2f}% | "
            f"Echo-Flat-2kD={s_off['Echo-Flat-2kD_top1_pct']:5.2f}% (Top2={s_off['Echo-Flat-2kD_top2_pct']:5.2f}%) | "
            f"TQA={s_off['TQA_top1_pct']:5.2f}% (2kD-TQA={s_off['Echo-Flat-2kD_top1_vs_TQA_top1_pp']:+.2f} pp, CI={s_off['Echo-Flat-2kD_top1_vs_TQA_top1_ci95']})",
            flush=True,
        )

        # 2. 在线跨规模、5 个流形变体消融及 12 种在线基线
        t1 = time.perf_counter()
        on_cache = cache_path(out_dir, f"cache_online_mwis_n{n:02d}_p{depth:02d}")
        on_recs = read_jsonl(on_cache) if on_cache.exists() else []
        if len(on_recs) != count:
            done_dict = {int(r["index"]): r for r in on_recs}
            J_stack = np.stack([inst["J"] for inst in instances])
            h_stack = np.stack([inst["h"] for inst in instances])
            prior_list = predict_echo_prior_flat_seed0(depth, J_stack, h_stack)
            mf_list = (1.0 + mean_field(J_stack, h_stack, 2.0)) / 2.0
            remaining_tasks = [
                (instances[i], depth, mf_list[i], prior_list[i], variants_to_run, run_all_12_baselines)
                for i in range(count)
                if int(instances[i]["index"]) not in done_dict
            ]
            if remaining_tasks:
                print(f"  [Online n={n:2d}] Running remaining {len(remaining_tasks)} / {count} instances on {workers} workers...", flush=True)
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(_eval_online_scaling_worker, t) for t in remaining_tasks]
                    for fut in as_completed(futures):
                        r = fut.result()
                        done_dict[int(r["index"])] = r
                        append_jsonl(on_cache, r)
                        if len(done_dict) % 16 == 0 or len(done_dict) == count:
                            print(f"  [Online n={n:2d}] Progress: {len(done_dict)}/{count} instances completed", flush=True)
            on_recs = [done_dict[int(inst["index"])] for inst in instances]
            on_recs.sort(key=lambda r: int(r["index"]))
            write_jsonl(on_cache, on_recs)

        tqa_i_on = np.array([r["TQA"]["initial_probability"] for r in on_recs]) * 100.0
        tqa_f_on = np.array([r["TQA"]["probability"] for r in on_recs]) * 100.0
        tqa_s_init = summarize_pct(tqa_i_on)
        tqa_s_fin = summarize_pct(tqa_f_on)
        on_entry = {
            "n": n,
            "count": count,
            "seconds": round(time.perf_counter() - t1, 1),
            "TQA": dict(
                init_pct=tqa_s_init["mean_pct"],
                init_ci95=tqa_s_init["ci95"],
                init_ci=tqa_s_init["ci_offset"],
                final_pct=tqa_s_fin["mean_pct"],
                final_ci95=tqa_s_fin["ci95"],
                final_ci=tqa_s_fin["ci_offset"],
                mean_calls=round(float(np.mean([r["TQA"]["stage2_calls"] for r in on_recs])), 1),
            ),
        }
        for prefix in ("MF", "Prior"):
            for var in variants_to_run:
                key = f"{prefix}_{var}"
                vi = np.array([r[key]["initial_probability"] for r in on_recs]) * 100.0
                vf = np.array([r[key]["probability"] for r in on_recs]) * 100.0
                bacc = float(np.mean([r[key]["bit_acc_pct"] for r in on_recs]))
                qh = float(np.mean([r[key]["quadrant_hit"] * 100.0 for r in on_recs]))
                calls_m = float(np.mean([r[key]["stage2_calls"] for r in on_recs]))
                s1_ms = float(np.mean([r[key]["stage1_seconds"] * 1000.0 for r in on_recs]))
                mis_vf = np.array([r[key]["probability"] for r in on_recs
                                   if not r[key]["exact_orientation_hit"]]) * 100.0
                si = summarize_pct(vi)
                sf = summarize_pct(vf)
                sd = summarize_paired_diff(vf, tqa_f_on)
                on_entry[key] = dict(
                    init_pct=si["mean_pct"],
                    init_ci95=si["ci95"],
                    init_ci=si["ci_offset"],
                    final_pct=sf["mean_pct"],
                    final_ci95=sf["ci95"],
                    final_ci=sf["ci_offset"],
                    vs_tqa_diff_pp=sd["mean_diff_pp"],
                    vs_tqa_ci95=sd["ci95"],
                    win_rate_vs_tqa_pct=sd["win_rate_pct"],
                    bit_acc_pct=round(bacc, 2),
                    quadrant_hit_pct=round(qh, 1),
                    mean_calls=round(calls_m, 1),
                    mean_stage1_ms=round(s1_ms, 2),
                    misoriented_subset_count=int(len(mis_vf)),
                    misoriented_subset_final=summarize_pct(mis_vf) if len(mis_vf) > 0 else None,
                )
        if run_all_12_baselines:
            on_entry["baselines_12"] = {}
            for b_method in BASELINE_METHODS:
                bi = np.array([r["baselines_12"][b_method]["initial_probability"] for r in on_recs]) * 100.0
                bf = np.array([r["baselines_12"][b_method]["probability"] for r in on_recs]) * 100.0
                bcalls = float(np.mean([r["baselines_12"][b_method]["stage2_calls"] for r in on_recs]))
                si_b = summarize_pct(bi)
                sf_b = summarize_pct(bf)
                sd_b = summarize_paired_diff(bf, tqa_f_on)
                on_entry["baselines_12"][b_method] = dict(
                    init_pct=si_b["mean_pct"],
                    init_ci95=si_b["ci95"],
                    final_pct=sf_b["mean_pct"],
                    final_ci95=sf_b["ci95"],
                    vs_tqa_diff_pp=sd_b["mean_diff_pp"],
                    vs_tqa_ci95=sd_b["ci95"],
                    mean_calls=round(bcalls, 1),
                )
        ablation_report["online"][f"n{n:02d}"] = on_entry

        mf_v0 = on_entry["MF_Echo-Flat-1D"]
        pr_v0 = on_entry["Prior_Echo-Flat-1D"]
        mf_v4 = on_entry["MF_Echo-Flat-2kD"]
        pr_v4 = on_entry["Prior_Echo-Flat-2kD"]
        online_kd_summary[f"n{n:02d}"] = dict(
            n=n,
            count=count,
            tqa_init_pct=on_entry["TQA"]["init_pct"],
            tqa_final_pct=on_entry["TQA"]["final_pct"],
            tqa_final_ci95=on_entry["TQA"]["final_ci95"],
            tqa_final_ci=on_entry["TQA"]["final_ci"],
            tqa_calls=on_entry["TQA"]["mean_calls"],
            mf_1d_bit_acc_pct=mf_v0["bit_acc_pct"],
            mf_1d_init_pct=mf_v0["init_pct"],
            mf_1d_final_pct=mf_v0["final_pct"],
            mf_1d_final_ci95=mf_v0["final_ci95"],
            mf_1d_final_ci=mf_v0["final_ci"],
            prior_1d_bit_acc_pct=pr_v0["bit_acc_pct"],
            prior_1d_init_pct=pr_v0["init_pct"],
            prior_1d_final_pct=pr_v0["final_pct"],
            prior_1d_final_ci95=pr_v0["final_ci95"],
            prior_1d_final_ci=pr_v0["final_ci"],
            mf_kd_bit_acc_pct=mf_v4["bit_acc_pct"],
            mf_kd_init_pct=mf_v4["init_pct"],
            mf_kd_final_pct=mf_v4["final_pct"],
            mf_kd_final_ci95=mf_v4["final_ci95"],
            mf_kd_final_ci=mf_v4["final_ci"],
            mf_kd_vs_tqa_pp=mf_v4["vs_tqa_diff_pp"],
            mf_kd_vs_tqa_ci95=mf_v4["vs_tqa_ci95"],
            mf_kd_calls=mf_v4["mean_calls"],
            prior_kd_bit_acc_pct=pr_v4["bit_acc_pct"],
            prior_kd_init_pct=pr_v4["init_pct"],
            prior_kd_final_pct=pr_v4["final_pct"],
            prior_kd_final_ci95=pr_v4["final_ci95"],
            prior_kd_final_ci=pr_v4["final_ci"],
            prior_kd_vs_tqa_pp=pr_v4["vs_tqa_diff_pp"],
            prior_kd_vs_tqa_ci95=pr_v4["vs_tqa_ci95"],
            prior_kd_calls=pr_v4["mean_calls"],
        )
        print(
            f"  [Online  n={n:2d}] TQA={on_entry['TQA']['final_pct']:5.2f}% | "
            f"Echo-MF-2kD={mf_v4['final_pct']:5.2f}% ({mf_v4['vs_tqa_diff_pp']:+.2f} pp) | "
            f"Echo-Prior-Flat-2kD (seed=0)={pr_v4['final_pct']:5.2f}% ({pr_v4['vs_tqa_diff_pp']:+.2f} pp)",
            flush=True,
        )

    save_json(out_dir / "mwis_multiscalar_scaling_p10.json", mwis_offline_summary)
    save_json(out_dir / "online_kd_scaling_p10.json", online_kd_summary)
    save_json(out_dir / "kd_ablation_study_p10.json", ablation_report)
    return dict(offline=mwis_offline_summary, online=online_kd_summary, ablation=ablation_report)


def run_task_five_families_n11_n12(out_dir: Path, workers: int = 24, depth: int = 10) -> dict:
    """在 data/benchmark/ 五家族 n=11, 12（500 题，p=10）上评测无截断 1D-Flat、Staged-2kD-Flat 与 TQA（B-06）。"""
    print(f"=== [Task: five_families_n11_n12] Evaluating 5 benchmark families at n=11, 12 (p={depth}) ===", flush=True)
    fam_summary = {"bootstrap_replicates": 4000, "seed": 0, "families": {}}
    for fam in ("mwis", "dense_ising", "sparse_ising", "grid_ising", "pinned_maxcut"):
        fam_summary["families"][fam] = {}
        for n in (11, 12):
            t0 = time.perf_counter()
            cache_file = cache_path(out_dir, f"cache_offline_{fam}_n{n:02d}_p{depth:02d}")
            recs = read_jsonl(cache_file) if cache_file.exists() else []
            files = sorted((ROOT / f"data/benchmark/{fam}").glob(f"{fam}_n{n:02d}_*.npz"))
            if len(recs) != len(files):
                instances = []
                for idx, f in enumerate(files):
                    with np.load(f, allow_pickle=False) as d:
                        instances.append(dict(
                            index=idx,
                            id=f"{fam}_n{n:02d}_{idx:02d}",
                            J=d["J"].astype(np.float64),
                            h=d["h"].astype(np.float64),
                            target_index=int(d["target_index"]),
                            s_star=d["s_star"].astype(np.float64),
                        ))
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    recs = list(pool.map(_eval_offline_scaling_worker, [(inst, depth, fam == "mwis") for inst in instances]))
                recs.sort(key=lambda r: int(r["index"]))
                write_jsonl(cache_file, recs)

            s_fam = summarize_offline_records(recs, n, len(recs), time.perf_counter() - t0)
            fam_summary["families"][fam][f"n{n:02d}"] = s_fam
            print(
                f"  [{fam:13s} n={n:2d}] Init(1D->2kD): {s_fam['Echo-Flat-1D_init_pct']:5.2f}% -> {s_fam['Echo-Flat-2kD_init_pct']:5.2f}% | "
                f"Top1(1D / 2kD / TQA): {s_fam['Echo-Flat-1D_top1_pct']:5.2f}% / {s_fam['Echo-Flat-2kD_top1_pct']:5.2f}% / {s_fam['TQA_top1_pct']:5.2f}% | "
                f"Top2(2kD / TQA): {s_fam['Echo-Flat-2kD_top2_pct']:5.2f}% / {s_fam['TQA_top2_pct']:5.2f}%",
                flush=True,
            )

    save_json(out_dir / "five_families_n11_n12_multiscalar_p10.json", fam_summary)
    return fam_summary


# =============================================================================
# 子任务 5: shot_noise (B-07, C-08, C-10, C-11, C-12)
# =============================================================================
def run_stage1_shot_noise_instance(task: tuple) -> dict:
    """模式 A：单道测试题上 TQA（32 候选点）在有限采样 N_shots 下的选点压力测试（移除 2D-TQA）。"""
    idx, inst_id, J, h, target_idx, depth = task
    prob = EnergyProblem(J, h)
    energy = prob.energy
    flips = prob.flips
    u = (np.arange(depth) + 0.5) / depth

    ladder_1d = np.geomspace(0.05, 5.0, 32)
    cands_1d = [np.r_[dt * u, (-dt * (1.0 - u) + np.pi / 2.0) % np.pi - np.pi / 2.0] for dt in ladder_1d]
    dists_1d = np.array([np.abs(evolve(ang, energy, flips)) ** 2 for ang in cands_1d])
    exact_e_1d = dists_1d @ energy

    rng = np.random.default_rng(np.random.SeedSequence([0, 1, int(idx)]))
    n_trials = 5
    selected_1d = {"inf": [int(np.argmin(exact_e_1d))]}

    for shots in SHOT_LEVELS_STAGE1:
        s1_list = []
        for _ in range(n_trials):
            noisy_e_1d = [float((rng.multinomial(shots, p_dist) @ energy) / shots) for p_dist in dists_1d]
            s1_list.append(int(np.argmin(noisy_e_1d)))
        selected_1d[str(shots)] = s1_list

    unique_1d = set(m for m_list in selected_1d.values() for m in m_list)
    cache_1d = {}
    for m in unique_1d:
        init_p = float(dists_1d[m, target_idx])
        opt = prob.optimize(cands_1d[m])
        fin_st = evolve(np.asarray(opt["angles"], dtype=np.float64), energy, flips)
        fin_p = float(np.abs(fin_st[target_idx]) ** 2)
        cache_1d[m] = (init_p, fin_p, float(opt["energy"]), int(opt["calls"]))

    out_1d = {}
    exact_best_1d = selected_1d["inf"][0]
    for key, m_list in selected_1d.items():
        vals = [cache_1d[m] for m in m_list]
        out_1d[key] = dict(
            initial_probability=float(np.mean([v[0] for v in vals])),
            probability=float(np.mean([v[1] for v in vals])),
            energy=float(np.mean([v[2] for v in vals])),
            stage2_calls=float(np.mean([v[3] for v in vals])),
            exact_candidate_match=float(np.mean([m == exact_best_1d for m in m_list])),
        )
    return dict(index=int(idx), id=str(inst_id), n=len(h), TQA=out_1d)


def run_parameter_shift_instance(task: tuple) -> dict:
    """模式 B：单道测试题上调用 shot_noise_adam_optimize（多项分布抽样 + 参数移位梯度）评测 TQA, Echo-MF-Flat, Echo-MF-2kD, Echo-Prior-Flat。"""
    idx, inst_id, J, h, target_idx, depth, mf_flat_ang, mf_2kd_ang, prior_flat_ang = task
    energy = energy_table(spin_table(len(h)), J, h, 0.0)
    flips = flips_for(len(h))
    u = (np.arange(depth) + 0.5) / depth

    ladder_1d = np.geomspace(0.05, 5.0, 32)
    cands_1d = [np.r_[dt * u, (-dt * (1.0 - u) + np.pi / 2.0) % np.pi - np.pi / 2.0] for dt in ladder_1d]
    dists_1d = np.array([fast_dist(ang, energy, flips) for ang in cands_1d])

    opt_steps = 64
    res = {}
    for shots in SHOT_LEVELS_PARAMETER_SHIFT:
        rng = np.random.default_rng(np.random.SeedSequence([0, 2, int(shots), int(idx)]))
        noisy_e_1d = [float((rng.multinomial(shots, p) @ energy) / shots) for p in dists_1d]
        tqa_init = cands_1d[int(np.argmin(noisy_e_1d))]

        s_key = str(shots)
        res[s_key] = {}
        for m_name, init_ang in [
            ("TQA", tqa_init),
            ("Echo-MF-Flat", mf_flat_ang),
            ("Echo-MF-2kD", mf_2kd_ang),
            ("Echo-Prior-Flat", prior_flat_ang),
        ]:
            init_p = float(fast_dist(init_ang, energy, flips)[target_idx])
            snaps = shot_noise_adam_optimize(
                init_ang, energy, flips, shots=shots, steps=opt_steps, rng=rng
            )
            p16 = float(fast_dist(snaps[16], energy, flips)[target_idx])
            p32 = float(fast_dist(snaps[32], energy, flips)[target_idx])
            p64 = float(fast_dist(snaps[64], energy, flips)[target_idx])
            res[s_key][m_name] = dict(initial_probability=init_p, step16_probability=p16,
                                      step32_probability=p32, probability=p64)

    return dict(index=int(idx), id=str(inst_id), n=len(h), results=res)


def run_task_shot_noise(out_dir: Path, workers: int = 24, depth: int = 10) -> dict:
    """运行有限采样散粒噪声实验（在 data/mwis/n10/dataset.npz 1,024 题上同步评测 Echo-Prior-Flat, Echo-MF-2kD, Echo-MF-Flat 与 TQA，Bootstrap 4000 seed=0）。"""
    print(f"=== [Task: shot_noise] Running Shot-Noise Stress Tests on 1,024 test instances (p={depth}) ===", flush=True)
    raw = np.load(ROOT / "data/mwis/n10/dataset.npz", allow_pickle=False)
    test_idx = np.flatnonzero(raw["split"] == "test")
    J_all = raw["J"].astype(np.float64)
    h_all = raw["h"].astype(np.float64)
    target_all = raw["target_index"].astype(int)
    ids_all = raw["ids"].astype(str)

    def _rec_idx(r: dict) -> int:
        return int(r["index"]) if "index" in r else int(r["id"].split("_")[-1])

    mf_jsonl = ROOT / f"runs/models_and_eval/baselines/Echo-MF-Flat__p{depth:02d}.jsonl"
    mf_recs = load_jsonl_sorted(mf_jsonl)
    mf_init_arr = np.array([r["initial_probability"] * 100.0 for r in mf_recs])
    mf_final_arr = np.array([r["probability"] * 100.0 for r in mf_recs])
    mf_init_angles = {_rec_idx(r): np.array(r["initial_angles"], dtype=np.float64) for r in mf_recs}

    prior_jsonl = PRIOR_FLAT_SEED0_EVALS[depth]
    prior_recs = load_jsonl_sorted(prior_jsonl)
    prior_init_arr = np.array([r["initial_probability"] * 100.0 for r in prior_recs])
    prior_final_arr = np.array([r["probability"] * 100.0 for r in prior_recs])
    prior_init_angles = {_rec_idx(r): np.array(r["initial_angles"], dtype=np.float64) for r in prior_recs}

    mf2kd_init_angles = {}
    mf2kd_jsonl = cache_path(out_dir, f"n10_2kd_test1024_p{depth:02d}")
    if mf2kd_jsonl.exists():
        for r in load_jsonl_sorted(mf2kd_jsonl):
            entry = r["Echo-MF-2kD"]["top1"]
            mf2kd_init_angles[_rec_idx(r)] = np.array(entry["initial_angles"], dtype=np.float64)
    for idx in test_idx:
        if int(idx) not in mf2kd_init_angles:
            ang_2kd, _, _ = decode_mean_field_multiscalar(depth, J_all[idx], h_all[idx], beta=2.0)
            mf2kd_init_angles[int(idx)] = ang_2kd

    # 模式 A：Stage-1 选点散粒噪声
    stage1_cache = cache_path(out_dir, f"stage1_shot_noise_p{depth:02d}")
    s1_records = read_jsonl(stage1_cache) if stage1_cache.exists() else []
    if len(s1_records) != len(test_idx):
        t0 = time.perf_counter()
        tasks_s1 = [
            (int(idx), str(ids_all[idx]), J_all[idx], h_all[idx], int(target_all[idx]), depth)
            for idx in test_idx
        ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            s1_records = list(pool.map(run_stage1_shot_noise_instance, tasks_s1, chunksize=16))
        s1_records.sort(key=lambda r: int(r["index"]))
        write_jsonl(stage1_cache, s1_records)
        print(f"  -> Regime A completed in {time.perf_counter() - t0:.1f}s", flush=True)

    # 模式 B：端到端混合梯度含噪变分优化
    ps_cache = cache_path(out_dir, f"end_to_end_parameter_shift_p{depth:02d}")
    ps_records = read_jsonl(ps_cache) if ps_cache.exists() else []
    if len(ps_records) != len(test_idx):
        t0 = time.perf_counter()
        tasks_ps = [
            (
                int(idx),
                str(ids_all[idx]),
                J_all[idx],
                h_all[idx],
                int(target_all[idx]),
                depth,
                mf_init_angles[int(idx)],
                mf2kd_init_angles[int(idx)],
                prior_init_angles[int(idx)],
            )
            for idx in test_idx
        ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            ps_records = list(pool.map(run_parameter_shift_instance, tasks_ps, chunksize=16))
        ps_records.sort(key=lambda r: int(r["index"]))
        write_jsonl(ps_cache, ps_records)
        print(f"  -> Regime B completed in {time.perf_counter() - t0:.1f}s", flush=True)

    mf_i_s = summarize_pct(mf_init_arr)
    mf_f_s = summarize_pct(mf_final_arr)
    pr_i_s = summarize_pct(prior_init_arr)
    pr_f_s = summarize_pct(prior_final_arr)

    summary = dict(
        depth=depth,
        test_instances=len(test_idx),
        bootstrap_replicates=4000,
        seed=0,
        zero_shot_selection_references={
            "Echo-MF-Flat": dict(
                stage1_quantum_evals=0,
                mean_init_pct=mf_i_s["mean_pct"],
                init_ci95=mf_i_s["ci95"],
                mean_final_pct=mf_f_s["mean_pct"],
                final_ci95=mf_f_s["ci95"],
                final_ci_minus=mf_f_s["ci_offset"][0],
                final_ci_plus=mf_f_s["ci_offset"][1],
            ),
            "Echo-Prior-Flat": dict(
                stage1_quantum_evals=0,
                seed=0,
                mean_init_pct=pr_i_s["mean_pct"],
                init_ci95=pr_i_s["ci95"],
                mean_final_pct=pr_f_s["mean_pct"],
                final_ci95=pr_f_s["ci95"],
                final_ci_minus=pr_f_s["ci_offset"][0],
                final_ci_plus=pr_f_s["ci_offset"][1],
            ),
        },
        regime_a_stage1_shot_noise={"TQA": {}},
        regime_b_end_to_end_parameter_shift={},
    )

    s1_records.sort(key=lambda r: str(r["id"]))
    for shots_key in [str(s) for s in SHOT_LEVELS_STAGE1] + ["inf"]:
        init_v = np.array([r["TQA"][shots_key]["initial_probability"] * 100.0 for r in s1_records])
        fin_v = np.array([r["TQA"][shots_key]["probability"] * 100.0 for r in s1_records])
        match_v = np.array([r["TQA"][shots_key]["exact_candidate_match"] * 100.0 for r in s1_records])
        si = summarize_pct(init_v)
        sf = summarize_pct(fin_v)
        mf_diff = summarize_paired_diff(mf_final_arr, fin_v)
        pr_diff = summarize_paired_diff(prior_final_arr, fin_v)
        summary["regime_a_stage1_shot_noise"]["TQA"][shots_key] = dict(
            mean_init_pct=si["mean_pct"],
            init_ci95=si["ci95"],
            init_ci=si["ci_offset"],
            mean_final_pct=sf["mean_pct"],
            final_ci95=sf["ci95"],
            final_ci=sf["ci_offset"],
            exact_candidate_match_pct=round(float(np.mean(match_v)), 2),
            Echo_MF_Flat_net_gain_pp=mf_diff["mean_diff_pp"],
            Echo_MF_Flat_gain_ci95=mf_diff["ci95"],
            Echo_Prior_Flat_net_gain_pp=pr_diff["mean_diff_pp"],
            Echo_Prior_Flat_gain_ci95=pr_diff["ci95"],
        )

    ps_records.sort(key=lambda r: str(r["id"]))
    evals_per_step = 4 * depth  # 2 * 2p = 40 次电路评估/步 (p=10)
    for shots in SHOT_LEVELS_PARAMETER_SHIFT:
        s_key = str(shots)
        summary["regime_b_end_to_end_parameter_shift"][s_key] = {}
        for m_name, s1_evals in [("TQA", 32), ("Echo-MF-Flat", 0), ("Echo-MF-2kD", 0), ("Echo-Prior-Flat", 0)]:
            init_v = np.array([r["results"][s_key][m_name]["initial_probability"] * 100.0 for r in ps_records])
            p16_v = np.array([r["results"][s_key][m_name]["step16_probability"] * 100.0 for r in ps_records])
            p32_v = np.array([r["results"][s_key][m_name]["step32_probability"] * 100.0 for r in ps_records])
            fin_v = np.array([r["results"][s_key][m_name]["probability"] * 100.0 for r in ps_records])
            si = summarize_pct(init_v)
            s16 = summarize_pct(p16_v)
            s32 = summarize_pct(p32_v)
            sf = summarize_pct(fin_v)
            s2_evals_16 = 16 * evals_per_step
            s2_evals_32 = 32 * evals_per_step
            s2_evals_64 = 64 * evals_per_step
            summary["regime_b_end_to_end_parameter_shift"][s_key][m_name] = dict(
                stage1_evals=s1_evals,
                stage2_evals_per_step=evals_per_step,
                stage2_evals_16steps=s2_evals_16,
                stage2_evals_32steps=s2_evals_32,
                stage2_evals_64steps=s2_evals_64,
                total_quantum_shots_16steps=(s1_evals + s2_evals_16) * shots,
                total_quantum_shots_32steps=(s1_evals + s2_evals_32) * shots,
                total_quantum_shots_64steps=(s1_evals + s2_evals_64) * shots,
                mean_init_pct=si["mean_pct"],
                init_ci95=si["ci95"],
                init_ci=si["ci_offset"],
                step16_pct=s16["mean_pct"],
                step16_ci95=s16["ci95"],
                step16_ci=s16["ci_offset"],
                step32_pct=s32["mean_pct"],
                step32_ci95=s32["ci95"],
                step32_ci=s32["ci_offset"],
                mean_final_pct=sf["mean_pct"],
                final_ci95=sf["ci95"],
                final_ci=sf["ci_offset"],
            )

        tqa_64 = np.array([r["results"][s_key]["TQA"]["probability"] * 100.0 for r in ps_records])
        for target_m in ("Echo-MF-Flat", "Echo-MF-2kD", "Echo-Prior-Flat"):
            t64 = np.array([r["results"][s_key][target_m]["probability"] * 100.0 for r in ps_records])
            t32 = np.array([r["results"][s_key][target_m]["step32_probability"] * 100.0 for r in ps_records])
            t16 = np.array([r["results"][s_key][target_m]["step16_probability"] * 100.0 for r in ps_records])
            d64 = summarize_paired_diff(t64, tqa_64)
            d32 = summarize_paired_diff(t32, tqa_64)
            d16 = summarize_paired_diff(t16, tqa_64)
            summary["regime_b_end_to_end_parameter_shift"][s_key][f"{target_m}_vs_TQA_pp"] = dict(
                mean_diff_pp=d64["mean_diff_pp"],
                ci95=d64["ci95"],
                step32_vs_64step_base_pp=d32["mean_diff_pp"],
                step32_ci95=d32["ci95"],
                step16_vs_64step_base_pp=d16["mean_diff_pp"],
                step16_ci95=d16["ci95"],
            )

    save_json(out_dir / f"shot_noise_summary_p{depth:02d}.json", summary)
    print(f"  -> Saved shot-noise robustness summary to {out_dir / f'shot_noise_summary_p{depth:02d}.json'}", flush=True)
    return summary


# =============================================================================
# 子任务 6: ising_online_benchmark (吸收原 eval_ising_online_benchmark.py 全部功能 + 12 基线)
# =============================================================================
def _eval_ising_online_instance(task: tuple) -> dict:
    """在单道多家族实例上评测在线无监督方法（支持全部 12 种在线基线）。"""
    idx, fpath, depth, methods_to_run = task
    with np.load(fpath, allow_pickle=False) as d:
        J = d["J"].astype(np.float64)
        h = d["h"].astype(np.float64)
        s_star = d["s_star"].astype(np.float64)
        target_idx = int(d["target_index"])
    prob = EnergyProblem(J, h)
    m_mf = mean_field(J[None, ...], h[None, ...], 2.0)[0]
    raw_acc = float(np.mean(np.sign(m_mf) == s_star)) * 100.0
    pi_mf = (1.0 + m_mf) / 2.0
    s_dir = resolve_prior_direction(J, h, pi_mf)
    ord_acc = float(np.mean(s_dir == s_star)) * 100.0

    row = dict(index=int(idx), id=Path(fpath).stem, raw_acc=raw_acc, ord_acc=ord_acc, methods={})
    for m_name in methods_to_run:
        row["methods"][m_name] = _eval_online_baseline_method(
            m_name, int(idx), J, h, target_idx, s_star, depth, prob
        )
    return row


def run_task_ising_online_benchmark(
    out_dir: Path,
    workers: int = 16,
    n_inst: int = 50,
    configs: tuple[tuple[int, int], ...] = ((8, 8), (10, 10), (12, 10)),
) -> dict:
    """在 MWIS, Grid Ising, Sparse Ising, Dense Ising 上评测在线无监督基准（MWIS 评测全部 12 种基线）。"""
    print("=== [Task: ising_online_benchmark] Evaluating Online Benchmark Across Ising Families ===", flush=True)
    families = [
        ("mwis", "MWIS"),
        ("grid_ising", "Grid Ising"),
        ("sparse_ising", "Sparse Ising"),
        ("dense_ising", "Dense Ising"),
    ]
    report = {"bootstrap_replicates": 4000, "seed": 0, "suites": {}}
    for n_val, depth in configs:
        suite_key = f"n{n_val:02d}_p{depth:02d}"
        report["suites"][suite_key] = {}
        for fam_dir, fam_label in families:
            files = sorted((ROOT / f"data/benchmark/{fam_dir}").glob(f"{fam_dir}_n{n_val:02d}_*.npz"))[:n_inst]
            methods_to_run = BASELINE_METHODS if fam_dir == "mwis" else ("Echo-MF-2kD", "Echo-MF-Flat", "TQA")
            tasks = [(idx, f, depth, methods_to_run) for idx, f in enumerate(files)]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_eval_ising_online_instance, tasks))
            fam_entry = dict(
                family=fam_label,
                count=len(results),
                raw_mf_bit_acc_pct=round(float(np.mean([r["raw_acc"] for r in results])), 2),
                oriented_bit_acc_pct=round(float(np.mean([r["ord_acc"] for r in results])), 2),
                methods={},
            )
            tqa_fin = np.array([r["methods"]["TQA"]["probability"] for r in results]) * 100.0
            for m_name in methods_to_run:
                init_v = np.array([r["methods"][m_name]["initial_probability"] for r in results]) * 100.0
                fin_v = np.array([r["methods"][m_name]["probability"] for r in results]) * 100.0
                fam_entry["methods"][m_name] = dict(
                    initial=summarize_pct(init_v),
                    final=summarize_pct(fin_v),
                    vs_tqa=summarize_paired_diff(fin_v, tqa_fin),
                )
            report["suites"][suite_key][fam_dir] = fam_entry
            print(
                f"  [{suite_key} {fam_label:<12}] OrientAcc={fam_entry['oriented_bit_acc_pct']:5.1f}% | "
                f"Echo-MF-2kD={fam_entry['methods']['Echo-MF-2kD']['final']['mean_pct']:5.2f}% | "
                f"Echo-MF-Flat={fam_entry['methods']['Echo-MF-Flat']['final']['mean_pct']:5.2f}% | "
                f"TQA={fam_entry['methods']['TQA']['final']['mean_pct']:5.2f}%",
                flush=True,
            )
    save_json(out_dir / "ising_online_benchmark_summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=(
            "all",
            "n10_recompute",
            "n10_2kd_test1024",
            "cross_scale_mwis",
            "five_families_n11_n12",
            "shot_noise",
            "ising_online_benchmark",
        ),
        default="all",
        help="选择要执行的标准评测子任务",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/canonical")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--depth", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task in ("all", "n10_recompute"):
        run_task_n10_recompute(out_dir, workers=args.workers)
    if args.task in ("all", "n10_2kd_test1024"):
        run_task_n10_2kd_test1024(out_dir, workers=args.workers)
    if args.task in ("all", "cross_scale_mwis"):
        run_task_cross_scale_mwis(out_dir, workers=args.workers, depth=args.depth)
    if args.task in ("all", "five_families_n11_n12"):
        run_task_five_families_n11_n12(out_dir, workers=args.workers, depth=args.depth)
    if args.task in ("all", "shot_noise"):
        run_task_shot_noise(out_dir, workers=args.workers, depth=args.depth)
    if args.task in ("all", "ising_online_benchmark"):
        run_task_ising_online_benchmark(out_dir, workers=args.workers)


if __name__ == "__main__":
    main()
