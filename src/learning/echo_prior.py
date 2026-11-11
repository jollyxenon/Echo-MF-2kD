"""从 J,h 预测逐比特概率的 Echo 结构编码器。"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from records import SAVE_EVERY

GAMMA_MAX = 10 * np.pi
Z_AGG = 9.0


def upper_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    """返回上三角耦合索引。"""
    return np.triu_indices(n, 1)


def build_features(J: np.ndarray, h: np.ndarray) -> np.ndarray:
    """将对称 J 的上三角和 h 拼成编码器输入。"""
    i, j = upper_indices(h.shape[1])
    return np.concatenate((J[:, i, j], h), axis=1)


def teacher_marginals(distributions: np.ndarray, spins: np.ndarray) -> np.ndarray:
    """从教师完整分布 [题目, 状态] 计算逐比特正自旋边缘。"""
    positive = (np.asarray(spins) > 0).astype(np.float64)
    return np.einsum("qs,sk->qk", np.asarray(distributions, dtype=np.float64), positive)


class EchoPrior(nn.Module):
    """消息传递逐比特先验编码器，推理只需要 J,h。"""

    def __init__(self, mean, std, width: int = 64, rounds: int = 4):
        """建立边消息网络；比特数在每次前向时由特征维度确定。"""
        super().__init__()
        self.register_buffer("edge_mean", torch.as_tensor(mean[0], dtype=torch.float32))
        self.register_buffer("edge_std", torch.as_tensor(std[0], dtype=torch.float32))
        self.register_buffer("field_mean", torch.as_tensor(mean[1], dtype=torch.float32))
        self.register_buffer("field_std", torch.as_tensor(std[1], dtype=torch.float32))
        self.node = nn.Sequential(nn.Linear(1, width), nn.GELU())
        self.messages = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * width + 1, width), nn.GELU(),
                          nn.Linear(width, width), nn.GELU())
            for _ in range(rounds)
        ])
        self.updates = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * width, width), nn.GELU(),
                          nn.Linear(width, width), nn.GELU())
            for _ in range(rounds)
        ])
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """返回逐节点嵌入；比特数由 n(n+1)/2 维特征反推，支持任意规模零样本推理。"""
        n = int((np.sqrt(8 * x.shape[1] + 1) - 1) / 2)
        if n * (n + 1) // 2 != x.shape[1]:
            raise ValueError(f"输入特征维度 {x.shape[1]} 不是 n(n+1)/2")
        edge_count = n * (n - 1) // 2
        ui, uj = torch.triu_indices(n, n, offset=1, device=x.device)
        normalized = torch.cat(((x[:, :edge_count] - self.edge_mean) / self.edge_std,
                                (x[:, edge_count:] - self.field_mean) / self.field_std), dim=1)
        edges = x.new_zeros(len(x), n, n)
        raw = x.new_zeros(len(x), n, n)
        edges[:, ui, uj] = normalized[:, :edge_count]
        edges[:, uj, ui] = normalized[:, :edge_count]
        raw[:, ui, uj] = x[:, :edge_count]
        raw[:, uj, ui] = x[:, :edge_count]
        mask = (raw != 0).unsqueeze(-1)
        z = self.node(normalized[:, edge_count:, None])
        for message, update in zip(self.messages, self.updates):
            pair = torch.cat((z[:, :, None].expand(-1, -1, n, -1),
                              z[:, None].expand(-1, n, -1, -1), edges[..., None]), dim=-1)
            aggregate = (message(pair) * mask).sum(2) / Z_AGG
            z = z + update(torch.cat((z, aggregate), dim=-1))
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回每个比特取正自旋的 logit。"""
        return self.head(self.encode(x)).squeeze(-1)


class EchoGNNScalar(nn.Module):
    """图级均值池化 + 有向多分支标量回归的一维流形输出表示消融模型。"""

    def __init__(self, mean, std, width: int = 64, rounds: int = 4):
        super().__init__()
        self.backbone = EchoPrior(mean, std, width, rounds)
        self.head = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对节点嵌入做全局均值池化后输出标量 a_dec in [-10*pi, 10*pi]。"""
        pooled = self.backbone.encode(x).mean(dim=1)
        return GAMMA_MAX * torch.tanh(self.head(pooled).squeeze(-1))


class EchoGNNGrid(nn.Module):
    """四阶统计矩池化 + 128 维网格 logits 的一维流形输出表示消融模型。"""

    def __init__(self, mean, std, width: int = 64, rounds: int = 4, grid_points: int = 128):
        super().__init__()
        self.backbone = EchoPrior(mean, std, width, rounds)
        self.head = nn.Linear(4 * width, grid_points)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """提取节点嵌入的四阶统计矩，经线性头输出 128 维网格 logits。"""
        z = self.backbone.encode(x)
        mu = z.mean(dim=1)
        centered = z - mu.unsqueeze(1)
        var = centered.square().mean(dim=1)
        normed = centered / (var + 1e-8).sqrt().unsqueeze(1)
        skew = (normed ** 3).mean(dim=1)
        kurt = (normed ** 4).mean(dim=1)
        return self.head(torch.cat((mu, var, skew, kurt), dim=-1))


def train_echo_prior(X: np.ndarray, target: np.ndarray, n_qubits: int, seed: int,
                     steps: int = 4000, batch: int = 64, width: int = 64,
                     rounds: int = 4, learning_rate: float = 1e-3,
                     save_every: int = SAVE_EVERY, log=None, checkpoint=None) -> EchoPrior:
    """使用教师逐比特边缘概率训练先验编码器；日志与检查点由调用方落盘。"""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    edge_count = n_qubits * (n_qubits - 1) // 2
    mean = np.asarray([X[:, :edge_count].mean(), X[:, edge_count:].mean()])
    std = np.asarray([X[:, :edge_count].std(), X[:, edge_count:].std()])
    std = np.where(std > 0, std, 1.0)
    model = EchoPrior(mean, std, width, rounds)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=1e-5)
    features = torch.as_tensor(X, dtype=torch.float32)
    labels = torch.as_tensor(target, dtype=torch.float32)
    generator = torch.Generator().manual_seed(100000 + seed)
    model.train()
    for step in range(1, steps + 1):
        index = torch.randint(0, len(features), (batch,), generator=generator)
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(model(features[index]), labels[index])
        if not torch.isfinite(loss):
            raise ValueError("逐比特先验损失非有限")
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % save_every and step != steps:
            continue
        row = dict(step=step, loss=float(loss.detach()),
                   learning_rate=float(schedule.get_last_lr()[0]))
        if log is not None:
            log(row)
        if checkpoint is not None:
            checkpoint(step, model)
    model.eval()
    return model


def echo_prior_probabilities(model: EchoPrior, X: np.ndarray) -> np.ndarray:
    """把编码器输出转换为逐比特概率。"""
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(X, dtype=torch.float32))).cpu().numpy()


def train_and_predict_manifold_ablations(
    X_train: np.ndarray,
    h_train: np.ndarray,
    pi_train: np.ndarray,
    X_test: np.ndarray,
    n_qubits: int,
    depth: int,
    seed: int = 0,
    steps: int = 4000,
    batch: int = 64,
    width: int = 64,
    rounds: int = 4,
    learning_rate: float = 1e-3,
    grid_points: int = 128,
    run_grid: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """训练并预测 1D 平顶流形输出表示消融模型 Echo-GNN-Scalar 与 Echo-GNN-Grid，返回测试集标量 a (未乘 scale)。"""
    from echo.variants import load_formula
    from learning.echo_decoder import expected_success

    pairs = depth // 2
    formula = load_formula("Flat-Top", depth)
    ratios = np.asarray(formula["ratios"], dtype=np.float64)
    scale = float(np.max(np.abs(ratios)))
    grid_dec = np.linspace(-GAMMA_MAX, GAMMA_MAX, grid_points)
    grid_a = grid_dec / scale

    n_train = len(X_train)
    scalar_targets = np.zeros((n_train, 4), dtype=np.float32)
    scalar_mask = np.zeros((n_train, 4), dtype=bool)
    soft_grid_targets = np.zeros((n_train, grid_points), dtype=np.float32)

    for q in range(n_train):
        scores = expected_success("Flat-Top", pairs, grid_a, h_train[q], pi_train[q])
        s_sum = float(np.sum(scores))
        soft_grid_targets[q] = (scores / s_sum) if s_sum > 0 else (1.0 / grid_points)
        s_max = float(np.max(scores))
        order = np.argsort(-scores)
        chosen: list[float] = []
        for idx in order:
            if scores[idx] < 0.9 * s_max:
                break
            cand_dec = float(grid_dec[idx])
            if all(abs(cand_dec - c) / GAMMA_MAX > 0.01 for c in chosen):
                chosen.append(cand_dec)
                if len(chosen) == 4:
                    break
        if not chosen:
            chosen.append(float(grid_dec[int(order[0])]))
        for r_idx, val in enumerate(chosen):
            scalar_targets[q, r_idx] = val
            scalar_mask[q, r_idx] = True

    edge_count = n_qubits * (n_qubits - 1) // 2
    mean = np.asarray([X_train[:, :edge_count].mean(), X_train[:, edge_count:].mean()])
    std = np.asarray([X_train[:, :edge_count].std(), X_train[:, edge_count:].std()])
    std = np.where(std > 0, std, 1.0)
    features_tr = torch.as_tensor(X_train, dtype=torch.float32)
    features_te = torch.as_tensor(X_test, dtype=torch.float32)

    # 1. 训练 Echo-GNN-Scalar
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model_scalar = EchoGNNScalar(mean, std, width, rounds)
    opt_s = torch.optim.AdamW(model_scalar.parameters(), lr=learning_rate, weight_decay=1e-4)
    sch_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, steps, eta_min=1e-5)
    t_targets = torch.as_tensor(scalar_targets, dtype=torch.float32)
    t_mask = torch.as_tensor(scalar_mask, dtype=torch.bool)
    gen_s = torch.Generator().manual_seed(200000 + seed)
    model_scalar.train()
    for _ in range(steps):
        idx = torch.randint(0, n_train, (batch,), generator=gen_s)
        opt_s.zero_grad()
        pred_dec = model_scalar(features_tr[idx])[:, None]
        targ = t_targets[idx]
        d2 = ((pred_dec - targ) / GAMMA_MAX) ** 2
        d2 = d2.masked_fill(~t_mask[idx], float("inf"))
        loss_s = d2.min(dim=1).values.mean()
        loss_s.backward()
        opt_s.step()
        sch_s.step()
    model_scalar.eval()
    with torch.no_grad():
        pred_scalar_a = (model_scalar(features_te).cpu().numpy().astype(np.float64)) / scale

    if not run_grid:
        return pred_scalar_a, None

    # 2. 训练 Echo-GNN-Grid
    torch.manual_seed(seed)
    model_grid = EchoGNNGrid(mean, std, width, rounds, grid_points=grid_points)
    opt_g = torch.optim.AdamW(model_grid.parameters(), lr=learning_rate, weight_decay=1e-4)
    sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, steps, eta_min=1e-5)
    t_soft = torch.as_tensor(soft_grid_targets, dtype=torch.float32)
    gen_g = torch.Generator().manual_seed(300000 + seed)
    model_grid.train()
    for _ in range(steps):
        idx = torch.randint(0, n_train, (batch,), generator=gen_g)
        opt_g.zero_grad()
        logits = model_grid(features_tr[idx])
        loss_g = -(t_soft[idx] * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        loss_g.backward()
        opt_g.step()
        sch_g.step()
    model_grid.eval()
    with torch.no_grad():
        best_indices = torch.argmax(model_grid(features_te), dim=-1).cpu().numpy()
        pred_grid_a = grid_a[best_indices].astype(np.float64)

    return pred_scalar_a, pred_grid_a
