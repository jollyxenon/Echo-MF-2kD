"""免训练在线初始化基线（Echo-MF、Echo-Energy、TQA、RI）与外部对照基线（经典启发式、Warm-Start QAOA、ma-QAOA）评测。"""
from __future__ import annotations

import time

import autograd.numpy as anp
from autograd import value_and_grad
import numpy as np
from scipy.optimize import minimize

from basis import flips_for, spin_table, target_index as spin_target_index
from compare.methods import bounds
from echo.multiscalar import resolve_prior_direction
from instances.benchmark import energy_table
from learning.echo_decoder import decode_mean_field, decode_mean_field_multiscalar, decode_product_energy
from learning.features import mean_field
from learning.optimization import EnergyProblem, OPTIONS
from learning.parameterization import expand, starts
from learning.scoring import score_pair, summarize_records
from records import append_jsonl, emit, load_npz, save_json, split_indices

MF_METHOD_MAP = {
    "Echo-MF-Flat": "Flat-Top",
    "Echo-MF-End": "Endpoint-Matched",
    "Echo-MF-Alt": "Alternating-Phase",
}

METHODS = (
    "Echo-MF-Flat",
    "Echo-MF-End",
    "Echo-MF-Alt",
    "Echo-MF-2kD",
    "Echo-Energy",
    "TQA",
    "RI (1)",
    "RI (32)",
    "Classical-MF",
    "Classical-Cluster",
    "Warm-Start QAOA",
    "ma-QAOA",
)


def run_warm_start_qaoa(J: np.ndarray, h: np.ndarray, depth: int, target: int,
                        eps_reg: float = 0.05, beta_mf: float = 2.0) -> dict:
    """运行极化初态与单比特旋转混合器的 Warm-Start QAOA（docs/module-settings.md §7.3）。"""
    started = time.perf_counter()
    J_arr = np.asarray(J, dtype=np.float64)
    h_arr = np.asarray(h, dtype=np.float64)
    n = len(h_arr)
    spins = spin_table(n)
    energy = energy_table(spins, J_arr, h_arr, 0.0)
    flips = flips_for(n)

    m_mf = mean_field(J_arr[None, ...], h_arr[None, ...], beta_mf)[0]
    pi_mf = (1.0 + m_mf) / 2.0
    c = np.clip(pi_mf, eps_reg, 1.0 - eps_reg)

    amp_table = np.where(spins > 0, np.sqrt(c)[None, :], np.sqrt(1.0 - c)[None, :])
    psi0 = np.prod(amp_table, axis=1).astype(np.complex128)
    diag_ws = [spins[:, i] * (2.0 * c[i] - 1.0) for i in range(n)]
    sin_th = [2.0 * float(np.sqrt(c[i] * (1.0 - c[i]))) for i in range(n)]

    def evolve_ws(angles):
        state = psi0
        for layer in range(depth):
            state = anp.exp(-1j * angles[layer] * energy) * state
            cb = anp.cos(angles[depth + layer])
            sb = anp.sin(angles[depth + layer])
            for i in range(n):
                state = (cb - 1j * sb * diag_ws[i]) * state - 1j * (sb * sin_th[i]) * state[flips[i]]
        return state

    def expectation_ws(angles):
        state = evolve_ws(angles)
        return anp.real(anp.sum(anp.conj(state) * energy * state))

    ws_val_and_grad = value_and_grad(expectation_ws)

    stage1_started = time.perf_counter()
    initial, initial_energy = None, np.inf
    for dt in np.geomspace(0.05, 5.0, 32):
        cand = expand("TQA-LBFGSB", depth, dt)[0]
        val = float(expectation_ws(cand))
        if val < initial_energy:
            initial, initial_energy = cand, val
    stage1_seconds = time.perf_counter() - stage1_started

    best = initial.copy()
    best_energy = initial_energy
    calls = 0

    def loss(angles):
        nonlocal best, best_energy, calls
        val, grad = ws_val_and_grad(angles)
        calls += 1
        if float(val) < best_energy:
            best, best_energy = np.array(angles, dtype=np.float64), float(val)
        return float(val), np.asarray(grad, dtype=np.float64)

    stage2_started = time.perf_counter()
    res = minimize(loss, initial, jac=True, method="L-BFGS-B",
                   bounds=bounds(10.0 * np.pi, depth), options=OPTIONS)
    stage2_seconds = time.perf_counter() - stage2_started
    init_dist = np.abs(np.asarray(evolve_ws(initial))) ** 2
    final_dist = np.abs(np.asarray(evolve_ws(best))) ** 2
    return dict(
        initial_angles=initial.tolist(),
        angles=best.tolist(),
        initial_probability=float(init_dist[target]),
        probability=float(final_dist[target]),
        initial_energy=float(init_dist @ energy),
        energy=float(final_dist @ energy),
        stage1_calls=32,
        stage2_calls=calls,
        stage1_seconds=stage1_seconds,
        stage2_seconds=stage2_seconds,
        iterations=int(res.nit),
        status="converged" if res.success else "optimizer_stopped",
        message=str(res.message),
        seconds=time.perf_counter() - started,
        eps_reg=eps_reg,
        beta_mf=beta_mf,
    )


def run_ma_qaoa(J: np.ndarray, h: np.ndarray, depth: int, target: int, seed: int = 0) -> dict:
    """运行逐节点外场角、逐边耦合角与逐节点混合角的多角 ma-QAOA（docs/module-settings.md §7.3）。"""
    started = time.perf_counter()
    J_arr = np.asarray(J, dtype=np.float64)
    h_arr = np.asarray(h, dtype=np.float64)
    n = len(h_arr)
    spins = spin_table(n)
    energy = energy_table(spins, J_arr, h_arr, 0.0)
    flips = flips_for(n)

    edge_u, edge_v = np.nonzero(np.triu(J_arr, 1) != 0.0)
    num_edges = len(edge_u)
    cost_dim_per_layer = n + num_edges
    total_cost_dim = depth * cost_dim_per_layer
    total_mixer_dim = depth * n
    total_dim = total_cost_dim + total_mixer_dim

    node_terms = h_arr[:, None] * spins.T
    if num_edges > 0:
        edge_weights = J_arr[edge_u, edge_v]
        edge_terms = edge_weights[:, None] * (spins[:, edge_u] * spins[:, edge_v]).T
        cost_terms = np.vstack((node_terms, edge_terms))
    else:
        cost_terms = node_terms

    def evolve_ma(params):
        state = np.ones(len(energy), dtype=np.complex128) / np.sqrt(len(energy))
        for layer in range(depth):
            c_start = layer * cost_dim_per_layer
            gamma_layer = params[c_start:c_start + cost_dim_per_layer]
            state = anp.exp(-1j * anp.dot(gamma_layer, cost_terms)) * state
            b_start = total_cost_dim + layer * n
            for i in range(n):
                beta_i = params[b_start + i]
                state = anp.cos(beta_i) * state - 1j * anp.sin(beta_i) * state[flips[i]]
        return state

    def expectation_ma(params):
        state = evolve_ma(params)
        return anp.real(anp.sum(anp.conj(state) * energy * state))

    ma_val_and_grad = value_and_grad(expectation_ma)

    stage1_started = time.perf_counter()
    rng = np.random.default_rng(seed)
    initial = rng.uniform(-0.1, 0.1, size=total_dim)
    stage1_seconds = time.perf_counter() - stage1_started

    best = initial.copy()
    best_energy = np.inf
    calls = 0

    def loss(params):
        nonlocal best, best_energy, calls
        val, grad = ma_val_and_grad(params)
        calls += 1
        if float(val) < best_energy:
            best, best_energy = np.array(params, dtype=np.float64), float(val)
        return float(val), np.asarray(grad, dtype=np.float64)

    ma_bounds = ([(-10.0 * np.pi, 10.0 * np.pi)] * total_cost_dim
                 + [(-np.pi / 2.0, np.pi / 2.0)] * total_mixer_dim)
    stage2_started = time.perf_counter()
    res = minimize(loss, initial, jac=True, method="L-BFGS-B",
                   bounds=ma_bounds, options=OPTIONS)
    stage2_seconds = time.perf_counter() - stage2_started
    init_dist = np.abs(np.asarray(evolve_ma(initial))) ** 2
    final_dist = np.abs(np.asarray(evolve_ma(best))) ** 2
    return dict(
        initial_angles=initial.tolist(),
        angles=best.tolist(),
        initial_probability=float(init_dist[target]),
        probability=float(final_dist[target]),
        initial_energy=float(init_dist @ energy),
        energy=float(final_dist @ energy),
        stage1_calls=0,
        stage2_calls=calls,
        stage1_seconds=stage1_seconds,
        stage2_seconds=stage2_seconds,
        iterations=int(res.nit),
        status="converged" if res.success else "optimizer_stopped",
        message=str(res.message),
        seconds=time.perf_counter() - started,
        num_parameters=total_dim,
    )


def select_initial(method: str, depth: int, J: np.ndarray, h: np.ndarray,
                   problem: EnergyProblem, rng: np.random.Generator,
                   beta: float = 2.0, grid_points: int = 128) -> tuple[np.ndarray, int, dict]:
    """按指定标准 QAOA 免训练方法生成初始角度，返回 (角度, Stage-1 量子评估数, 方法信息)。"""
    if method == "Echo-MF-2kD":
        angles, meta, _ = decode_mean_field_multiscalar(depth, J, h, beta=beta)
        return angles, 0, dict(variant=meta["variant"], product_energy=meta["e_prod"],
                               quadrant_hit=meta["quadrant_hit"], beta=beta)
    if method in MF_METHOD_MAP:
        scalar, angles, overlap, _ = decode_mean_field(MF_METHOD_MAP[method], depth, J, h,
                                                       beta=beta, grid_points=grid_points)
        return angles, 0, dict(scalar=scalar, distribution_overlap=overlap,
                               beta=beta, grid_points=grid_points)
    if method == "Echo-Energy":
        scalar, angles, prod_energy = decode_product_energy("Flat-Top", depth, J, h,
                                                            grid_points=grid_points)
        return angles, 0, dict(scalar=scalar, product_energy=prod_energy, grid_points=grid_points)
    if method == "RI (1)":
        angles, _ = next(starts("RI-LBFGSB", depth, rng))
        return angles, 0, {}
    if method in ("RI (32)", "TQA"):
        candidates = starts("TQA-LBFGSB" if method == "TQA" else "RI-LBFGSB", depth, rng)
        best, value = None, np.inf
        for _ in range(32):
            angles, _ = next(candidates)
            energy = problem.value(angles)
            if energy < value:
                best, value = angles, energy
        return best, 32, {}
    raise ValueError(f"未知初始化方法：{method}")


def run(dataset, output, depth, method, seed=0, split="test", beta=2.0, grid_points=128):
    """在统一能量 L-BFGS-B 停止条件下评测免训练与外部对照基线并记录全流程开销。"""
    data = load_npz(dataset)
    indices = split_indices(data)[split]
    output.mkdir(parents=True, exist_ok=False)
    records = []
    method_idx = METHODS.index(method)
    for index in indices:
        started = time.perf_counter()
        J, h = data["J"][index].astype(float), data["h"][index].astype(float)
        target = int(data["target_index"][index])
        if method in ("Classical-MF", "Classical-Cluster"):
            m_mf = mean_field(J[None, ...], h[None, ...], beta)[0]
            if method == "Classical-MF":
                s_pred = np.where(m_mf >= 0.0, 1.0, -1.0)
            else:
                s_pred = resolve_prior_direction(J, h, (1.0 + m_mf) / 2.0)
            value = float(0.5 * s_pred @ J @ s_pred + h @ s_pred)
            hit = float(spin_target_index(s_pred) == target)
            elapsed = time.perf_counter() - started
            row = dict(
                id=str(data["ids"][index]), initial_angles=None, angles=None,
                initial_probability=hit, probability=hit,
                initial_energy=value, energy=value,
                stage1_calls=0, stage2_calls=0,
                stage1_seconds=elapsed, stage2_seconds=0.0, total_seconds=elapsed,
                status="discrete",
                bit_acc_pct=float(np.mean(s_pred == data["s_star"][index]) * 100.0),
            )
        elif method in ("Warm-Start QAOA", "ma-QAOA"):
            res = (run_warm_start_qaoa(J, h, depth, target, beta_mf=beta)
                   if method == "Warm-Start QAOA" else run_ma_qaoa(J, h, depth, target, seed))
            extra = (dict(eps_reg=res["eps_reg"], beta_mf=res["beta_mf"])
                     if method == "Warm-Start QAOA" else dict(num_parameters=res["num_parameters"]))
            row = dict(
                id=str(data["ids"][index]),
                initial_angles=res["initial_angles"], angles=res["angles"],
                initial_probability=res["initial_probability"], probability=res["probability"],
                initial_energy=res["initial_energy"], energy=res["energy"],
                stage1_calls=res["stage1_calls"], stage2_calls=res["stage2_calls"],
                stage1_seconds=res["stage1_seconds"], stage2_seconds=res["stage2_seconds"],
                total_seconds=res["stage1_seconds"] + res["stage2_seconds"],
                status=res["status"],
                iterations=res["iterations"], message=res["message"], **extra,
            )
        else:
            problem = EnergyProblem(J, h)
            rng = np.random.default_rng([seed, int(index), depth, method_idx])
            stage1_started = time.perf_counter()
            best, calls, extra = select_initial(method, depth, J, h, problem, rng,
                                                beta=beta, grid_points=grid_points)
            stage1_seconds = time.perf_counter() - stage1_started
            optimized = problem.optimize(best)
            score = score_pair(J, h, target, best, optimized["angles"])
            row = dict(
                id=str(data["ids"][index]),
                initial_angles=best.tolist(), angles=optimized["angles"],
                initial_probability=score["initial_probability"], probability=score["probability"],
                initial_energy=score["initial_energy"], energy=score["energy"],
                stage1_calls=calls, stage2_calls=optimized["calls"],
                stage1_seconds=stage1_seconds, stage2_seconds=optimized["seconds"],
                total_seconds=stage1_seconds + optimized["seconds"],
                status=optimized["status"], iterations=optimized["iterations"],
                message=optimized["message"],
                distribution_residual=score["distribution_residual"],
                energy_residual=score["energy_residual"], norm_error=score["norm_error"],
                **extra,
            )
        append_jsonl(output / "records.jsonl", row)
        records.append(row)
        if len(records) % 32 == 0:
            emit(dict(job=method, seed=seed, evaluated=len(records), total=len(indices)))
    summary = dict(
        dataset=str(dataset), split=split, method=method, seed=seed, depth=depth,
        architecture="direct", wall_clock_cap=None, optimizer=OPTIONS,
        records_path=str(output / "records.jsonl"), **summarize_records(records),
    )
    save_json(output / "summary.json", summary)
    emit(summary)
    return summary
