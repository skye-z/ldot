#!/usr/bin/env python3
"""Export LDOT training assets into a cloned LlamaFactory workspace for Kaggle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DATASET_PATH = REPO_ROOT / "dataset/ldot_train_clean_bilingual_train.json"
DATASET_INFO_PATH = REPO_ROOT / "dataset/dataset_info.json"
KAGGLE_TEMPLATE_PATH = REPO_ROOT / "setting.kaggle.t4x2.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy LDOT assets into LlamaFactory for Kaggle 2xT4 training."
    )
    parser.add_argument(
        "--llamafactory-dir",
        required=True,
        help="Path to the cloned LlamaFactory workspace.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help=(
            "Model path or model ID used by LlamaFactory. For Kaggle this is usually "
            "a local path under /kaggle/input/..."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        default="",
        help=(
            "Optional external dataset root. If provided, read "
            "ldot_train_clean_bilingual_train.json and dataset_info.json from there "
            "instead of from the local repository."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="/kaggle/working/lf_output/ldot_t4x2",
        help="Directory where Trainer checkpoints and logs should be written.",
    )
    return parser.parse_args()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    lf_dir = Path(args.llamafactory_dir).resolve()
    data_dir = lf_dir / "data"
    dataset_root = Path(args.dataset_root).resolve() if args.dataset_root else None

    if not lf_dir.exists():
        raise RuntimeError(f"LlamaFactory dir does not exist: {lf_dir}")

    if dataset_root is None:
        train_dataset_path = TRAIN_DATASET_PATH
        dataset_info_path = DATASET_INFO_PATH
    else:
        train_dataset_path = dataset_root / "ldot_train_clean_bilingual_train.json"
        dataset_info_path = dataset_root / "dataset_info.json"

    if not train_dataset_path.exists():
        raise RuntimeError(f"Training dataset not found: {train_dataset_path}")
    if not dataset_info_path.exists():
        raise RuntimeError(f"dataset_info.json not found: {dataset_info_path}")

    train_data = train_dataset_path.read_text(encoding="utf-8")
    dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    dataset_info = {
        "ldot_train_clean_bilingual_train": dataset_info["ldot_train_clean_bilingual_train"]
    }
    config_text = KAGGLE_TEMPLATE_PATH.read_text(encoding="utf-8")
    config_text = config_text.replace("__MODEL_PATH__", args.model_path)
    config_text = config_text.replace("__OUTPUT_DIR__", args.output_dir)

    write_text(data_dir / "ldot_train_clean_bilingual_train.json", train_data)
    write_text(
        data_dir / "dataset_info.json",
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
    )
    write_text(lf_dir / "setting.kaggle.t4x2.yaml", config_text)

    print(
        json.dumps(
            {
                "llamafactory_dir": str(lf_dir),
                "train_dataset": str(data_dir / "ldot_train_clean_bilingual_train.json"),
                "dataset_info": str(data_dir / "dataset_info.json"),
                "config_path": str(lf_dir / "setting.kaggle.t4x2.yaml"),
                "source_dataset_root": str(dataset_root) if dataset_root else "repo-default",
                "model_path": args.model_path,
                "output_dir": args.output_dir,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
