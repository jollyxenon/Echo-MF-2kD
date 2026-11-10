"""全项目统一的 JSON/JSONL 记录、题库切分、源码快照与检查点节奏。"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAVE_EVERY = 500
CACHE_VERSION = 2


def dumps(value) -> str:
    """项目的唯一 JSON 文本口径：UTF-8 原文、拒绝 NaN/Inf。"""
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def emit(value) -> None:
    """向标准输出写一行机器可读 JSON。"""
    print(dumps(value), flush=True)


def save_json(path: Path, value) -> None:
    """保存缩进 JSON，供人工审阅。"""
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def load_json(path: Path) -> dict:
    """读入一个 JSON 对象。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: Path, row: dict) -> None:
    """追加一行 JSON 记录。"""
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(dumps(row) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """整体写入 JSONL。"""
    Path(path).write_text("".join(dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    """读入 JSONL 的全部记录。"""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """保存压缩 NPZ 题库或标签包。"""
    np.savez_compressed(path, **arrays)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """完整读入一个题库或标签包。"""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def split_indices(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """按题库的 split 字段返回三个划分的行号。"""
    return {name: np.flatnonzero(data["split"] == name)
            for name in ("train", "validation", "test")}


def check_alignment(data: dict[str, np.ndarray], labels: dict[str, np.ndarray]) -> None:
    """核对题库与标签包的题目顺序。"""
    if not np.array_equal(data["ids"], labels["ids"]):
        raise ValueError("题库与标签题号不一致")


def checkpoint_path(directory: Path, step: int) -> Path:
    """训练检查点的唯一命名：checkpoints/step-000500.pt。"""
    return Path(directory) / f"step-{step:06d}.pt"


def cache_path(directory: Path, name: str) -> Path:
    """canonical 逐题缓存的唯一命名：<name>_v2.jsonl。"""
    return Path(directory) / f"{name}_v{CACHE_VERSION}.jsonl"


def training_row(step: int, loss: float, learning_rate: float,
                 validation_loss: float | None, seconds: float) -> dict:
    """两种训练器共用的历史行，列顺序固定。"""
    return dict(step=step, loss=loss, learning_rate=learning_rate,
                validation_loss=validation_loss, seconds=seconds)


def source_files() -> list[Path]:
    """标签与训练归档需要冻结的源码和依赖文件。"""
    return (sorted((ROOT / "src").rglob("*.py"))
            + [ROOT / "pixi.toml", ROOT / "pixi.lock"])


def snapshot_sources(destination: Path, files: list[Path]) -> None:
    """把源码按项目相对路径复制到 destination/source。"""
    for source in files:
        target = Path(destination) / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def check_snapshot(destination: Path, files: list[Path]) -> None:
    """续跑前核对冻结源码逐字节一致。"""
    for source in files:
        frozen = Path(destination) / "source" / source.relative_to(ROOT)
        if source.read_bytes() != frozen.read_bytes():
            raise ValueError(f"源码或题库变化，不能恢复：{source}")
