"""结构化 MLP/GNN 的统一训练与模型恢复。"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from learning.angle_models import GNN, MLP, to_angles
from learning.features import build
from learning.quantum_circuit import nearest_squared
from records import (SAVE_EVERY, append_jsonl, checkpoint_path, check_alignment, emit,
                     load_npz, save_json, split_indices, training_row)


def standardize(values: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """用训练集统计量标准化输入，并返回均值、标准差和结果。"""
    mean = reference.mean(axis=0)
    std = np.where(reference.std(axis=0) > 0, reference.std(axis=0), 1.0)
    return mean, std, (values - mean) / std


def prepare_inputs(data: dict[str, np.ndarray], train_indices: np.ndarray) -> dict:
    """构造 MLP/GNN 输入，统计量只来自训练集。"""
    structured = build(data["J"].astype(np.float64), data["h"].astype(np.float64))
    flat_mean, flat_std, flat = standardize(structured["flat"], structured["flat"][train_indices])
    node_reference = structured["node"][train_indices].reshape(-1, structured["node"].shape[-1])
    node_mean, node_std, node_flat = standardize(
        structured["node"].reshape(-1, structured["node"].shape[-1]), node_reference
    )
    node = node_flat.reshape(structured["node"].shape)
    return dict(
        flat=torch.as_tensor(flat, dtype=torch.float64),
        node=torch.as_tensor(node, dtype=torch.float64),
        J=torch.as_tensor(data["J"], dtype=torch.float64),
        flat_mean=flat_mean,
        flat_std=flat_std,
        node_mean=node_mean,
        node_std=node_std,
        node_names=structured["node_names"],
        global_names=structured["global_names"],
    )


def inputs_from_checkpoint(data: dict[str, np.ndarray], checkpoint: dict) -> dict[str, torch.Tensor]:
    """按 checkpoint 中冻结的统计量重建输入。"""
    structured = build(data["J"].astype(np.float64), data["h"].astype(np.float64))
    return dict(
        flat=torch.as_tensor((structured["flat"] - np.asarray(checkpoint["flat_mean"]))
                             / np.asarray(checkpoint["flat_std"]), dtype=torch.float64),
        node=torch.as_tensor((structured["node"] - np.asarray(checkpoint["node_mean"]))
                             / np.asarray(checkpoint["node_std"]), dtype=torch.float64),
        J=torch.as_tensor(data["J"], dtype=torch.float64),
    )


def method_index(labels: dict[str, np.ndarray], method: str) -> int:
    """返回标签包中指定方法的列号。"""
    names = [str(value) for value in labels["methods"]]
    if method not in names:
        raise ValueError(f"标签包没有方法 {method}，可选：{names}")
    return names.index(method)


def make_model(architecture: str, n_qubits: int, node_dim: int, flat_dim: int,
               depth: int, width: int, rounds: int) -> torch.nn.Module:
    """按输入维度建立 MLP 或 GNN。"""
    if architecture == "MLP":
        return MLP(flat_dim, depth, width, rounds)
    if architecture == "GNN":
        return GNN(n_qubits, node_dim, depth, width, rounds)
    raise ValueError(f"未知网络结构：{architecture}")


def predict_raw(model: torch.nn.Module, architecture: str, inputs: dict,
                indices: np.ndarray | torch.Tensor | None = None) -> torch.Tensor:
    """用指定网络结构输出未转换角度。"""
    if indices is None:
        index = torch.arange(len(inputs["J"]), dtype=torch.long)
    else:
        index = torch.as_tensor(indices, dtype=torch.long)
    if architecture == "MLP":
        return model(inputs["flat"][index])
    return model(inputs["node"][index], inputs["J"][index])


def train(dataset, labels, output, architecture: str, method: str, seed: int = 0,
          epochs: int = 300, width: int | None = None, rounds: int | None = None,
          batch_size: int = 128, learning_rate: float = 1e-3, threads: int = 1,
          save_every: int = SAVE_EVERY) -> dict:
    """训练一个固定题目规模、深度、标签来源和网络结构的模型。"""
    width = (256 if architecture == "MLP" else 128) if width is None else width
    rounds = (3 if architecture == "MLP" else 4) if rounds is None else rounds
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    data = load_npz(dataset)
    label = load_npz(labels)
    check_alignment(data, label)
    splits = split_indices(data)
    train_indices_all, validation_indices_all = splits["train"], splits["validation"]
    depth = int(label["angles"].shape[-1] // 2)
    selected_method = method_index(label, method)
    target = torch.as_tensor(label["angles"][:, selected_method], dtype=torch.float64)
    mask = torch.as_tensor(label["mask"][:, selected_method], dtype=torch.bool)
    train_indices = train_indices_all[mask[train_indices_all].any(dim=1).numpy()]
    validation_indices = validation_indices_all[mask[validation_indices_all].any(dim=1).numpy()]
    if not len(train_indices) or not len(validation_indices):
        raise ValueError(f"方法 {method} 在训练或验证划分中没有有效标签")
    inputs = prepare_inputs(data, train_indices)
    model = make_model(architecture, inputs["node"].shape[1], inputs["node"].shape[-1],
                       inputs["flat"].shape[1], depth, width, rounds).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-5)
    generator = torch.Generator().manual_seed(100000 + seed)
    output.mkdir(parents=True, exist_ok=False)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    history = output / "history.jsonl"
    started = time.perf_counter()
    best = dict(validation_loss=float("inf"), step=0, epoch=0,
                state=copy.deepcopy(model.state_dict()))
    n = int(data["h"].shape[1])
    train_ids = data["ids"][train_indices].tolist()
    validation_ids = data["ids"][validation_indices].tolist()

    def payload(step: int, state: dict) -> dict:
        """两种训练器共用的检查点字段。"""
        return dict(architecture=architecture, method=method, n=n, depth=depth,
                    width=width, rounds=rounds, seed=seed, step=step, state=state,
                    flat_mean=torch.as_tensor(inputs["flat_mean"], dtype=torch.float64),
                    flat_std=torch.as_tensor(inputs["flat_std"], dtype=torch.float64),
                    node_mean=torch.as_tensor(inputs["node_mean"], dtype=torch.float64),
                    node_std=torch.as_tensor(inputs["node_std"], dtype=torch.float64),
                    node_names=inputs["node_names"], global_names=inputs["global_names"],
                    train_ids=train_ids, validation_ids=validation_ids)

    steps_per_epoch = (len(train_indices) + batch_size - 1) // batch_size
    total_steps = epochs * steps_per_epoch
    configuration = dict(
        dataset=str(dataset), labels=str(labels), architecture=architecture, method=method,
        n=n, depth=depth, seed=seed, width=width, rounds=rounds, epochs=epochs,
        steps=total_steps, batch_size=batch_size, learning_rate=learning_rate, threads=threads,
        training_count=len(train_indices), validation_count=len(validation_indices),
        excluded_train=len(train_indices_all) - len(train_indices),
        excluded_validation=len(validation_indices_all) - len(validation_indices),
        selection="minimum_validation_normalized_angle_mse", dtype="float64",
        checkpoint_interval=save_every,
    )
    save_json(output / "configuration.json", configuration)

    order_template = torch.as_tensor(train_indices, dtype=torch.long)
    recent_losses: list[float] = []
    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = order_template[torch.randperm(len(order_template), generator=generator)]
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            raw = predict_raw(model, architecture, inputs, index)
            angles = to_angles(raw, depth)
            loss = nearest_squared(angles, target[index], mask[index]).mean()
            if not torch.isfinite(loss):
                raise ValueError("训练损失非有限")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            recent_losses.append(float(loss.detach()))
            step += 1
            if step % save_every and step != total_steps:
                continue
            model.eval()
            with torch.no_grad():
                validation_angles = to_angles(
                    predict_raw(model, architecture, inputs, validation_indices), depth
                )
                validation_loss = float(nearest_squared(
                    validation_angles, target[validation_indices], mask[validation_indices]
                ).mean())
            if not np.isfinite(validation_loss):
                raise ValueError("验证损失非有限")
            row = training_row(step=step, loss=float(np.mean(recent_losses)),
                               learning_rate=float(scheduler.get_last_lr()[0]),
                               validation_loss=validation_loss,
                               seconds=time.perf_counter() - started)
            append_jsonl(history, row)
            emit(dict(architecture=architecture, method=method, seed=seed, **row))
            recent_losses = []
            if validation_loss < best["validation_loss"]:
                best = dict(validation_loss=validation_loss, step=step, epoch=epoch,
                            state=copy.deepcopy(model.state_dict()))
            torch.save(payload(step, model.state_dict()), checkpoint_path(checkpoints, step))
        scheduler.step()

    torch.save(payload(best["step"], best["state"]), output / "model.pt")
    summary = dict(
        architecture=architecture, method=method, seed=seed, n=n, depth=depth,
        width=width, rounds=rounds, epochs=epochs, steps=total_steps,
        batch_size=batch_size, learning_rate=learning_rate,
        best_step=best["step"], best_epoch=best["epoch"],
        best_validation_loss=best["validation_loss"],
        parameter_count=sum(p.numel() for p in model.parameters()),
        seconds=time.perf_counter() - started, checkpoint_interval=save_every,
    )
    save_json(output / "summary.json", summary)
    return summary


def restore(path) -> tuple[torch.nn.Module, dict, dict]:
    """恢复模型、checkpoint 和输入统计量。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    n = int(checkpoint["n"])
    depth = int(checkpoint["depth"])
    if checkpoint["architecture"] == "MLP":
        input_dim = len(checkpoint["node_names"]) * n + len(checkpoint["global_names"])
        model = MLP(input_dim, depth, int(checkpoint["width"]), int(checkpoint["rounds"])).double()
    elif checkpoint["architecture"] == "GNN":
        model = GNN(n, len(checkpoint["node_names"]), depth,
                    int(checkpoint["width"]), int(checkpoint["rounds"])).double()
    else:
        raise ValueError(f"未知 checkpoint 结构：{checkpoint['architecture']}")
    model.load_state_dict(checkpoint["state"])
    model.eval()
    stats = dict(
        flat_mean=np.asarray(checkpoint["flat_mean"]), flat_std=np.asarray(checkpoint["flat_std"]),
        node_mean=np.asarray(checkpoint["node_mean"]), node_std=np.asarray(checkpoint["node_std"]),
    )
    return model, checkpoint, stats
