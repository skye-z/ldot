#!/usr/bin/env python3
"""Build evaluation datasets and remove them from the training corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DATASET_DIR = Path("dataset")
TEMP_DIR = Path("temp")
SOURCE_DIR = TEMP_DIR / "source"
EVAL_DIR = DATASET_DIR / "eval"

COMMUNITY_FORWARD_PATH = TEMP_DIR / "ldot_community_zh_en_clean.json"
SLANG_SOURCE_PATH = SOURCE_DIR / "ldot_slang_consistency.json"
FULL_TRAIN_PATH = TEMP_DIR / "ldot_train_clean_bilingual.json"
TRAIN_OUTPUT_PATH = DATASET_DIR / "ldot_train_clean_bilingual_train.json"

TECH_SOURCE_PATHS = {
    "gpt_base": SOURCE_DIR / "gpt_base.json",
    "gpt_mix": SOURCE_DIR / "gpt_mix.json",
    "gpt_high": SOURCE_DIR / "gpt_high.json",
    "gpt_term": SOURCE_DIR / "gpt_term.json",
}

COMMUNITY_EVAL_ZH_EN_PATH = EVAL_DIR / "ldot_eval_community_zh_en.json"
COMMUNITY_EVAL_EN_ZH_PATH = EVAL_DIR / "ldot_eval_community_en_zh.json"
TECH_EVAL_ZH_EN_PATH = EVAL_DIR / "ldot_eval_tech_zh_en.json"
TECH_EVAL_EN_ZH_PATH = EVAL_DIR / "ldot_eval_tech_en_zh.json"
SLANG_EVAL_ZH_EN_PATH = EVAL_DIR / "ldot_eval_slang_zh_en.json"
SLANG_EVAL_EN_ZH_PATH = EVAL_DIR / "ldot_eval_slang_en_zh.json"
EVAL_MANIFEST_PATH = EVAL_DIR / "ldot_eval_manifest.json"

INSTRUCTION_ZH_EN = (
    "将下面的中文内容翻译成自然、准确、完整的英文，保持原意，"
    "保留语气、术语、代码、命令、链接和格式，不漏译、不乱译、不过度意译。"
)
INSTRUCTION_EN_ZH = (
    "将下面的英文内容翻译成自然、准确、完整的中文，保持原意，"
    "保留语气、术语、代码、命令、链接和格式，不漏译、不乱译、不过度意译。"
)
TERM_INSTRUCTION_ZH_EN = (
    "将下面的技术术语准确翻译成英文，优先使用常见、标准、自然的技术表达。"
)
TERM_INSTRUCTION_EN_ZH = (
    "将下面的技术术语准确翻译成中文，优先使用常见、标准、自然的技术表达。"
)

TECH_EVAL_FORWARD_COUNTS = {
    "gpt_base": 8,
    "gpt_mix": 8,
    "gpt_high": 8,
    "gpt_term": 8,
}
SLANG_EVAL_SOURCE_INDICES = [
    0,
    1,
    4,
    5,
    8,
    9,
    12,
    13,
    16,
    19,
    21,
    23,
    24,
    26,
    28,
    30,
    33,
    35,
]
SLANG_MARKERS = [
    "佬",
    "大佬",
    "佬友",
    "神人",
    "吃瓜",
    "整活",
    "节目效果",
    "抽象",
    "魔怔",
    "回血",
    "机场",
    "纯水",
    "前排",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def build_sample(instruction: str, input_text: str, output_text: str) -> dict[str, str]:
    return {
        "instruction": normalize_text(instruction),
        "input": normalize_text(input_text),
        "output": normalize_text(output_text),
    }


def sample_key(sample: dict[str, Any]) -> str:
    core = {
        "instruction": normalize_text(sample.get("instruction")),
        "input": normalize_text(sample.get("input")),
        "output": normalize_text(sample.get("output")),
    }
    return hashlib.sha1(
        json.dumps(core, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def stable_sort(samples: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(samples, key=sample_key)


def reverse_sample(sample: dict[str, str], instruction: str) -> dict[str, str]:
    return build_sample(
        instruction,
        sample["output"],
        sample["input"],
    )


def attach_meta(
    sample: dict[str, str],
    split: str,
    source: str,
    bucket: str,
    index: int,
) -> dict[str, Any]:
    difficulty = difficulty_profile(sample, split)
    return {
        "instruction": sample["instruction"],
        "input": sample["input"],
        "output": sample["output"],
        "meta": {
            "eval_split": split,
            "source": source,
            "bucket": bucket,
            "sample_id": f"{split}-{index:03d}",
            "difficulty_score": difficulty["score"],
            "difficulty_band": difficulty["band"],
            "difficulty_signals": difficulty["signals"],
        },
    }


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def has_ascii_letters(text: str) -> bool:
    return any(("a" <= ch.lower() <= "z") for ch in text)


def difficulty_profile(sample: dict[str, str], split: str) -> dict[str, Any]:
    input_text = sample["input"]
    score = 20
    signals: list[str] = []

    if split.startswith("community"):
        score += 10
        signals.append("community_context")
    elif split.startswith("tech"):
        score += 12
        signals.append("technical_accuracy")
    elif split.startswith("slang"):
        score += 18
        signals.append("slang_register")

    length = len(input_text)
    if length >= 1200:
        score += 32
        signals.append("very_long_input")
    elif length >= 400:
        score += 24
        signals.append("long_input")
    elif length >= 120:
        score += 16
        signals.append("medium_input")
    elif length >= 30:
        score += 8
        signals.append("short_input")
    else:
        score += 3
        signals.append("very_short_input")

    newline_count = input_text.count("\n")
    if newline_count >= 8:
        score += 12
        signals.append("multi_section_structure")
    elif newline_count >= 2:
        score += 6
        signals.append("multi_line_structure")

    if any(marker in input_text for marker in ["```", "`", "{", "}", "://", "~/.", "CPU |", "image"]):
        score += 10
        signals.append("format_or_code_preservation")

    if has_cjk(input_text) and has_ascii_letters(input_text):
        score += 8
        signals.append("mixed_language")

    if any(marker in input_text for marker in SLANG_MARKERS):
        score += 12
        signals.append("slang_or_forum_register")

    if any(marker in input_text for marker in [":", "✓", "……", "…", "w", "～"]):
        score += 4
        signals.append("tone_or_symbol_preservation")

    score = min(100, score)
    if score >= 85:
        band = "very_hard"
    elif score >= 70:
        band = "hard"
    elif score >= 50:
        band = "medium"
    else:
        band = "easy"

    return {"score": score, "band": band, "signals": signals}


def select_from_bucket(
    samples: list[dict[str, str]],
    predicate,
    count: int,
    used_keys: set[str],
) -> list[dict[str, str]]:
    candidates = [
        sample for sample in stable_sort(samples) if predicate(sample) and sample_key(sample) not in used_keys
    ]
    selected = candidates[:count]
    for sample in selected:
        used_keys.add(sample_key(sample))
    return selected


def select_community_eval_forward() -> list[dict[str, str]]:
    payload = [build_sample(item["instruction"], item["input"], item["output"]) for item in read_json(COMMUNITY_FORWARD_PATH)]
    used_keys: set[str] = set()

    selected: list[dict[str, str]] = []
    selected.extend(
        select_from_bucket(payload, lambda item: len(item["input"]) <= 16, 6, used_keys)
    )
    selected.extend(
        select_from_bucket(
            payload,
            lambda item: 17 <= len(item["input"]) <= 60,
            5,
            used_keys,
        )
    )
    selected.extend(
        select_from_bucket(payload, lambda item: len(item["input"]) > 60, 5, used_keys)
    )
    return selected


def normalize_tech_forward(source_name: str, items: list[dict[str, Any]]) -> list[dict[str, str]]:
    if source_name == "gpt_term":
        instruction = TERM_INSTRUCTION_EN_ZH
    else:
        instruction = INSTRUCTION_EN_ZH
    normalized = [
        build_sample(instruction, item.get("input"), item.get("output"))
        for item in items
    ]
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for sample in normalized:
        key = sample_key(sample)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sample)
    return deduped


def select_tech_eval_forward() -> list[tuple[str, dict[str, str]]]:
    selected: list[tuple[str, dict[str, str]]] = []
    for source_name, path in TECH_SOURCE_PATHS.items():
        payload = read_json(path)
        normalized = normalize_tech_forward(source_name, payload)
        count = TECH_EVAL_FORWARD_COUNTS[source_name]
        for sample in stable_sort(normalized)[:count]:
            selected.append((source_name, sample))
    return selected


def select_slang_eval_forward() -> list[dict[str, str]]:
    payload = [
        build_sample(item["instruction"], item["input"], item["output"])
        for item in read_json(SLANG_SOURCE_PATH)
    ]
    selected: list[dict[str, str]] = []
    for index in SLANG_EVAL_SOURCE_INDICES:
        if index < len(payload):
            selected.append(payload[index])
    return selected


def build_eval_outputs() -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    community_forward = select_community_eval_forward()
    community_reverse = [reverse_sample(item, INSTRUCTION_EN_ZH) for item in community_forward]

    tech_forward_pairs = select_tech_eval_forward()
    tech_forward = [sample for _, sample in tech_forward_pairs]
    tech_reverse = [
        reverse_sample(
            sample,
            TERM_INSTRUCTION_ZH_EN if source_name == "gpt_term" else INSTRUCTION_ZH_EN,
        )
        for source_name, sample in tech_forward_pairs
    ]

    slang_forward = select_slang_eval_forward()
    slang_reverse = [reverse_sample(item, INSTRUCTION_EN_ZH) for item in slang_forward]

    outputs = {
        "community_zh_en": [
            attach_meta(item, "community_zh_en", "community", "forum", idx + 1)
            for idx, item in enumerate(community_forward)
        ],
        "community_en_zh": [
            attach_meta(item, "community_en_zh", "community", "forum", idx + 1)
            for idx, item in enumerate(community_reverse)
        ],
        "tech_en_zh": [
            attach_meta(item, "tech_en_zh", "tech", "technical", idx + 1)
            for idx, item in enumerate(tech_forward)
        ],
        "tech_zh_en": [
            attach_meta(item, "tech_zh_en", "tech", "technical", idx + 1)
            for idx, item in enumerate(tech_reverse)
        ],
        "slang_zh_en": [
            attach_meta(item, "slang_zh_en", "slang", "community_slang", idx + 1)
            for idx, item in enumerate(slang_forward)
        ],
        "slang_en_zh": [
            attach_meta(item, "slang_en_zh", "slang", "community_slang", idx + 1)
            for idx, item in enumerate(slang_reverse)
        ],
    }

    eval_keys: set[str] = set()
    for split_items in outputs.values():
        for item in split_items:
            eval_keys.add(sample_key(item))
    return outputs, eval_keys


def build_train_without_eval(eval_keys: set[str]) -> list[dict[str, str]]:
    payload = read_json(FULL_TRAIN_PATH)
    filtered: list[dict[str, str]] = []
    for item in payload:
        if sample_key(item) in eval_keys:
            continue
        filtered.append(
            build_sample(item.get("instruction"), item.get("input"), item.get("output"))
        )
    return filtered


def main() -> int:
    outputs, eval_keys = build_eval_outputs()

    write_json(COMMUNITY_EVAL_ZH_EN_PATH, outputs["community_zh_en"])
    write_json(COMMUNITY_EVAL_EN_ZH_PATH, outputs["community_en_zh"])
    write_json(TECH_EVAL_EN_ZH_PATH, outputs["tech_en_zh"])
    write_json(TECH_EVAL_ZH_EN_PATH, outputs["tech_zh_en"])
    write_json(SLANG_EVAL_ZH_EN_PATH, outputs["slang_zh_en"])
    write_json(SLANG_EVAL_EN_ZH_PATH, outputs["slang_en_zh"])

    train_without_eval = build_train_without_eval(eval_keys)
    write_json(TRAIN_OUTPUT_PATH, train_without_eval)

    manifest = {
        "community_zh_en_count": len(outputs["community_zh_en"]),
        "community_en_zh_count": len(outputs["community_en_zh"]),
        "tech_en_zh_count": len(outputs["tech_en_zh"]),
        "tech_zh_en_count": len(outputs["tech_zh_en"]),
        "slang_zh_en_count": len(outputs["slang_zh_en"]),
        "slang_en_zh_count": len(outputs["slang_en_zh"]),
        "eval_total_count": sum(len(items) for items in outputs.values()),
        "community_zh_en_avg_difficulty": round(sum(item["meta"]["difficulty_score"] for item in outputs["community_zh_en"]) / max(1, len(outputs["community_zh_en"])), 2),
        "community_en_zh_avg_difficulty": round(sum(item["meta"]["difficulty_score"] for item in outputs["community_en_zh"]) / max(1, len(outputs["community_en_zh"])), 2),
        "tech_en_zh_avg_difficulty": round(sum(item["meta"]["difficulty_score"] for item in outputs["tech_en_zh"]) / max(1, len(outputs["tech_en_zh"])), 2),
        "tech_zh_en_avg_difficulty": round(sum(item["meta"]["difficulty_score"] for item in outputs["tech_zh_en"]) / max(1, len(outputs["tech_zh_en"])), 2),
        "slang_zh_en_avg_difficulty": round(sum(item["meta"]["difficulty_score"] for item in outputs["slang_zh_en"]) / max(1, len(outputs["slang_zh_en"])), 2),
        "slang_en_zh_avg_difficulty": round(sum(item["meta"]["difficulty_score"] for item in outputs["slang_en_zh"]) / max(1, len(outputs["slang_en_zh"])), 2),
        "train_without_eval_count": len(train_without_eval),
        "train_without_eval_path": str(TRAIN_OUTPUT_PATH),
    }
    write_json(EVAL_MANIFEST_PATH, manifest)
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
