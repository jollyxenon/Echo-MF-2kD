"""逐比特先验训练、Echo 一维解码和离线线路核验。"""
from __future__ import annotations

import time

import numpy as np
import torch

from basis import spin_table, target_indices
from learning.echo_decoder import decode, target_probability
from learning.echo_prior import (EchoPrior, build_features, echo_prior_probabilities,
                                 teacher_marginals, train_echo_prior)
from learning.quantum_circuit import energies, probability_distribution
from records import (SAVE_EVERY, append_jsonl, check_alignment, checkpoint_path, emit,
                     load_npz, save_json, split_indices, training_row, write_jsonl)


def restore(path) -> tuple[EchoPrior, dict]:
    """从权重及配置恢复编码器；不读取教师或题库。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = EchoPrior([0.0, 0.0], [1.0, 1.0], checkpoint["width"], checkpoint["rounds"])
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return model, checkpoint


def best_angles(labels: dict[str, np.ndarray], method: str) -> np.ndarray:
    """按每题已记录概率选择一条教师角度。"""
    names = [str(value) for value in labels["methods"]]
    if method not in names:
        raise ValueError(f"标签包没有方法 {method}")
    column = names.index(method)
    mask = labels["mask"][:, column]
    index = np.argmax(np.where(mask, labels["probabilities"][:, column], -np.inf), axis=1)
    return labels["angles"][np.arange(len(mask)), column, index]


def teacher_distributions(data: dict[str, np.ndarray], angles: np.ndarray,
                          batch_size: int = 32) -> np.ndarray:
    """由已选教师角度计算完整测量分布；目标态不参与此计算。"""
    J = torch.as_tensor(data["J"], dtype=torch.float64)
    h = torch.as_tensor(data["h"], dtype=torch.float64)
    distributions = []
    for start in range(0, len(angles), batch_size):
        stop = start + batch_size
        energy = energies(J[start:stop], h[start:stop])
        current = torch.as_tensor(angles[start:stop], dtype=torch.float64)
        distributions.append(probability_distribution(current, energy).numpy())
    return np.concatenate(distributions)


def run(dataset, labels, output, method: str, seed: int = 0, steps: int = 4000,
        width: int = 64, rounds: int = 4, grid_points: int = 4097,
        split: str = "validation", threads: int = 1) -> dict:
    """只用训练题教师边缘监督先验，冻结后在指定划分上评估。"""
    torch.set_num_threads(threads)
    data = load_npz(dataset)
    label = load_npz(labels)
    check_alignment(data, label)
    splits = split_indices(data)
    if not len(splits[split]):
        raise ValueError(f"没有题目划分 {split}")
    output.mkdir(parents=True, exist_ok=False)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    started = time.perf_counter()
    n = int(data["h"].shape[1])
    depth = int(label["angles"].shape[-1] // 2)

    # 仅训练划分的教师角度用于边缘监督，留出划分不参与监督标签构造。
    column = [str(value) for value in label["methods"]].index(method)
    train_indices = splits["train"][label["mask"][splits["train"], column].any(axis=1)]
    if not len(train_indices):
        raise ValueError("训练划分没有有效教师标签")
    training_labels = {key: label[key][train_indices] for key in ("angles", "mask", "probabilities")}
    training_labels["methods"] = label["methods"]
    teacher_angles = best_angles(training_labels, method)
    teacher = teacher_distributions({key: data[key][train_indices] for key in ("J", "h")},
                                    teacher_angles)
    margins = teacher_marginals(teacher, spin_table(n))
    X = build_features(data["J"].astype(np.float64), data["h"].astype(np.float64)).astype(np.float32)
    config = dict(
        dataset=str(dataset), labels=str(labels), architecture="Echo-Prior", method=method,
        n=n, depth=depth, seed=seed, width=width, rounds=rounds, epochs=None, steps=steps,
        batch_size=64, learning_rate=1e-3, threads=threads,
        training_count=len(train_indices), validation_count=0,
        excluded_train=len(splits["train"]) - len(train_indices), excluded_validation=0,
        selection="fixed_steps_last_checkpoint", dtype="float32",
        checkpoint_interval=SAVE_EVERY, split=split, grid_points=grid_points,
    )
    save_json(output / "configuration.json", config)
    history = output / "history.jsonl"

    def record(row: dict) -> None:
        """训练日志与角度模型保持同一列。"""
        entry = training_row(step=row["step"], loss=row["loss"],
                             learning_rate=row["learning_rate"], validation_loss=None,
                             seconds=time.perf_counter() - started)
        append_jsonl(history, entry)
        emit(dict(architecture="Echo-Prior", method=method, seed=seed, **entry))

    def save(step: int, model: EchoPrior) -> None:
        """每 save_every 步保存一份当前权重。"""
        torch.save(dict(step=step, state=model.state_dict(), **config),
                   checkpoint_path(checkpoints, step))

    model = train_echo_prior(X[train_indices], margins, n, seed, steps,
                             batch=64, width=width, rounds=rounds,
                             save_every=SAVE_EVERY, log=record, checkpoint=save)
    torch.save(dict(step=steps, state=model.state_dict(), **config), output / "model.pt")

    # 冻结模型后，为指定划分解码，并用完整 QAOA 线路独立复算。
    evaluation_started = time.perf_counter()
    prior = echo_prior_probabilities(model, X)
    decoded = [decode(method, depth, data["h"][index], prior[index], grid_points)
               for index in splits[split]]
    indices = np.asarray(splits[split], dtype=int)
    decoded_angles = np.stack([row[1] for row in decoded])
    energy = energies(torch.as_tensor(data["J"][indices], dtype=torch.float64),
                      torch.as_tensor(data["h"][indices], dtype=torch.float64))
    distribution = probability_distribution(torch.as_tensor(decoded_angles), energy).numpy()
    probability = distribution[np.arange(len(indices)), target_indices(data["s_star"][indices])]
    closed_probability = np.asarray([
        target_probability(method, depth, row[0], data["h"][index], data["s_star"][index])
        for index, row in zip(indices, decoded)
    ])
    residual = float(np.max(np.abs(probability - closed_probability)))
    if residual > 1e-10:
        raise ValueError(f"Echo 闭式概率与完整线路不一致：{residual}")
    records = [dict(id=str(data["ids"][index]), scalar=scalar, angles=angles.tolist(),
                    distribution_overlap=overlap, probability=float(prob))
               for (index, (scalar, angles, overlap)), prob in zip(zip(indices, decoded), probability)]
    write_jsonl(output / "decoded.jsonl", records)
    summary = dict(
        architecture="Echo-Prior", method=method, seed=seed, n=n, depth=depth,
        width=width, rounds=rounds, epochs=None, steps=steps, batch_size=64,
        learning_rate=1e-3, best_step=steps, best_epoch=None, best_validation_loss=None,
        parameter_count=sum(p.numel() for p in model.parameters()),
        seconds=time.perf_counter() - started, checkpoint_interval=SAVE_EVERY,
        evaluation=dict(dataset=str(dataset), labels=str(labels), split=split,
                        count=len(records), grid_points=grid_points,
                        mean_probability=float(np.mean(probability)),
                        independent_probability_residual=residual,
                        seconds=time.perf_counter() - evaluation_started),
    )
    save_json(output / "summary.json", summary)
    return summary
