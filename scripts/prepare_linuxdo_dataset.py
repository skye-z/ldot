#!/usr/bin/env python3
"""Clean Linux Do topic archives, extract slang candidates, and merge train sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DATASET_INFO = Path("dataset/dataset_info.json")
DEFAULT_PENDING_OUTPUT = Path("temp/ldot_linuxdo_pending_clean.json")
DEFAULT_SLANG_OUTPUT = Path("temp/ldot_slang_candidates.json")
DEFAULT_MERGED_OUTPUT = Path("temp/ldot_linuxdo_merged_train.json")
DEFAULT_BASE_DATASETS = [
    "ldot_translation",
    "gpt_base",
    "gpt_mix",
    "gpt_term",
    "gpt_high",
]
PENDING_INSTRUCTION = (
    "将下面这段来自 Linux Do 社区讨论的中文内容翻译成英文，"
    "保留原有语气、梗、代码、命令、链接和格式。"
)
KNOWN_SLANG_TERMS = [
    "大佬",
    "佬友",
    "神人",
    "吃瓜",
    "整活",
    "节目效果",
    "抽象",
    "魔怔",
    "前排",
    "回血",
    "纯水",
    "机场",
    "白嫖",
    "上车",
    "发车",
    "冲塔",
    "车头",
    "水贴",
    "开摆",
    "破防",
    "逆天",
]
SHORT_REPLY_SIGNALS = [
    "前排",
    "吃瓜",
    "佬",
    "大佬",
    "佬友",
    "支持",
    "回血",
    "整活",
    "抽象",
    "魔怔",
    "神人",
    "节目效果",
    "纯水",
]
SLANG_PATTERNS = [
    re.compile(r"[A-Za-z0-9_\-\u4e00-\u9fff]{1,16}佬"),
]
SLANG_PREFIXES = [
    "感谢",
    "谢谢",
    "多谢",
    "请问",
    "麻烦",
    "求",
    "求问",
    "问下",
    "问问",
    "想问",
    "请教",
    "咨询",
    "找",
    "向",
    "跟",
    "给",
    "@",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Linux Do data for translation and model training."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    clean_parser = subparsers.add_parser(
        "clean",
        help="Clean a topic archive or pending dataset and extract slang candidates.",
    )
    clean_parser.add_argument(
        "--input",
        required=True,
        help="Input topics archive JSON or pending JSON produced by the collector.",
    )
    clean_parser.add_argument(
        "--output",
        default=str(DEFAULT_PENDING_OUTPUT),
        help="Output path for the cleaned pending translation dataset.",
    )
    clean_parser.add_argument(
        "--slang-output",
        default=str(DEFAULT_SLANG_OUTPUT),
        help="Output path for the extracted slang candidate lexicon.",
    )
    clean_parser.add_argument(
        "--min-chars",
        type=int,
        default=12,
        help="Drop items whose normalized content is shorter than this threshold.",
    )
    clean_parser.add_argument(
        "--min-slang-frequency",
        type=int,
        default=2,
        help="Only keep slang candidates with at least this many hits.",
    )
    clean_parser.add_argument(
        "--max-slang-examples",
        type=int,
        default=5,
        help="Max example snippets kept for each slang term.",
    )

    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge translated Linux Do data into the existing training corpus.",
    )
    merge_parser.add_argument(
        "--translated-input",
        action="append",
        default=[],
        help=(
            "Translated dataset JSON to append. Repeat the flag for multiple files. "
            "Items with empty output are ignored."
        ),
    )
    merge_parser.add_argument(
        "--dataset-info",
        default=str(DEFAULT_DATASET_INFO),
        help="dataset_info.json path used to resolve base dataset aliases.",
    )
    merge_parser.add_argument(
        "--base-datasets",
        default=",".join(DEFAULT_BASE_DATASETS),
        help="Comma-separated dataset aliases to merge before appending translated data.",
    )
    merge_parser.add_argument(
        "--output",
        default=str(DEFAULT_MERGED_OUTPUT),
        help="Output path for the merged training dataset.",
    )
    return parser.parse_args()


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


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
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    compact = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", compact)


def is_low_signal(text: str, min_chars: int) -> bool:
    stripped = re.sub(r"\s+", "", text)
    if any(signal in stripped for signal in SHORT_REPLY_SIGNALS) and len(stripped) >= 4:
        return False
    if len(stripped) < min_chars:
        return True

    content_chars = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", stripped)
    if len(content_chars) < max(4, min_chars // 2):
        return True

    if len(set(stripped)) <= 2 and len(stripped) < min_chars * 2:
        return True

    return False


def build_pending_item(
    source_payload: dict[str, Any],
    topic: dict[str, Any],
    post: dict[str, Any],
    cleaned_text: str,
) -> dict[str, Any]:
    return {
        "instruction": PENDING_INSTRUCTION,
        "input": cleaned_text,
        "output": "",
        "meta": {
            "source": "linux.do",
            "period": source_payload.get("period", "daily"),
            "capture_date": source_payload.get("capture_date"),
            "captured_at": source_payload.get("captured_at"),
            "topic_id": topic.get("topic_id") or topic.get("id"),
            "topic_title": topic.get("title"),
            "topic_url": topic.get("url"),
            "topic_rank": topic.get("rank"),
            "category_id": topic.get("category_id"),
            "tags": topic.get("tags", []),
            "post_id": post.get("post_id") or post.get("id"),
            "post_number": post.get("post_number"),
            "username": post.get("username"),
            "created_at": post.get("created_at"),
            "reply_to_post_number": post.get("reply_to_post_number"),
            "translation_status": "pending",
        },
    }


def snippet_for(text: str, term: str, width: int = 80) -> str:
    index = text.find(term)
    if index < 0:
        return text[:width]
    start = max(0, index - 20)
    end = min(len(text), index + len(term) + width)
    return text[start:end]


def extract_slang_terms(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for term in KNOWN_SLANG_TERMS:
        if term in text and term not in seen:
            found.append(term)
            seen.add(term)
    for pattern in SLANG_PATTERNS:
        for raw_term in pattern.findall(text):
            term = raw_term.strip(" ,.;:!?，。！？、")
            for prefix in SLANG_PREFIXES:
                if term.startswith(prefix) and len(term) > len(prefix) + 1:
                    term = term[len(prefix) :]
                    break
            if term and term not in seen:
                found.append(term)
                seen.add(term)
    return found


def clean_from_topics(
    payload: dict[str, Any],
    min_chars: int,
    min_slang_frequency: int,
    max_slang_examples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    pending_items: list[dict[str, Any]] = []
    term_counter: Counter[str] = Counter()
    term_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_posts: set[str] = set()
    skipped = {"empty_or_duplicate": 0, "low_signal": 0}

    for topic in payload.get("topics", []):
        for post in topic.get("posts", []):
            raw = normalize_text(post.get("raw"))
            if not raw:
                skipped["empty_or_duplicate"] += 1
                continue

            dedupe_key = hashlib.sha1(raw.encode("utf-8")).hexdigest()
            if dedupe_key in seen_posts:
                skipped["empty_or_duplicate"] += 1
                continue
            seen_posts.add(dedupe_key)

            if is_low_signal(raw, min_chars):
                skipped["low_signal"] += 1
                continue

            pending_items.append(build_pending_item(payload, topic, post, raw))

            for term in extract_slang_terms(raw):
                term_counter[term] += 1
                if len(term_examples[term]) >= max_slang_examples:
                    continue
                term_examples[term].append(
                    {
                        "topic_id": topic.get("topic_id") or topic.get("id"),
                        "post_id": post.get("post_id") or post.get("id"),
                        "username": post.get("username"),
                        "snippet": snippet_for(raw, term),
                    }
                )

    slang_payload = {
        "source": "linux.do",
        "capture_date": payload.get("capture_date"),
        "generated_at": now_utc_iso(),
        "min_frequency": min_slang_frequency,
        "terms": [
            {
                "term": term,
                "count": count,
                "examples": term_examples[term],
            }
            for term, count in sorted(
                term_counter.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if count >= min_slang_frequency
        ],
    }
    return pending_items, slang_payload, skipped


def clean_from_pending(
    payload: list[dict[str, Any]],
    min_chars: int,
    min_slang_frequency: int,
    max_slang_examples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    pending_items: list[dict[str, Any]] = []
    term_counter: Counter[str] = Counter()
    term_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_posts: set[str] = set()
    skipped = {"empty_or_duplicate": 0, "low_signal": 0}

    for item in payload:
        cleaned_input = normalize_text(item.get("input"))
        if not cleaned_input:
            skipped["empty_or_duplicate"] += 1
            continue

        dedupe_key = hashlib.sha1(cleaned_input.encode("utf-8")).hexdigest()
        if dedupe_key in seen_posts:
            skipped["empty_or_duplicate"] += 1
            continue
        seen_posts.add(dedupe_key)

        if is_low_signal(cleaned_input, min_chars):
            skipped["low_signal"] += 1
            continue

        normalized_item = {
            "instruction": item.get("instruction") or PENDING_INSTRUCTION,
            "input": cleaned_input,
            "output": "",
            "meta": item.get("meta", {}),
        }
        pending_items.append(normalized_item)

        for term in extract_slang_terms(cleaned_input):
            term_counter[term] += 1
            if len(term_examples[term]) >= max_slang_examples:
                continue
            meta = normalized_item["meta"]
            term_examples[term].append(
                {
                    "topic_id": meta.get("topic_id"),
                    "post_id": meta.get("post_id"),
                    "username": meta.get("username"),
                    "snippet": snippet_for(cleaned_input, term),
                }
            )

    slang_payload = {
        "source": "linux.do",
        "generated_at": now_utc_iso(),
        "min_frequency": min_slang_frequency,
        "terms": [
            {
                "term": term,
                "count": count,
                "examples": term_examples[term],
            }
            for term, count in sorted(
                term_counter.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if count >= min_slang_frequency
        ],
    }
    return pending_items, slang_payload, skipped


def command_clean(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    payload = read_json(input_path)

    if isinstance(payload, dict) and "topics" in payload:
        pending_items, slang_payload, skipped = clean_from_topics(
            payload,
            args.min_chars,
            args.min_slang_frequency,
            args.max_slang_examples,
        )
    elif isinstance(payload, list):
        pending_items, slang_payload, skipped = clean_from_pending(
            payload,
            args.min_chars,
            args.min_slang_frequency,
            args.max_slang_examples,
        )
    else:
        raise RuntimeError(
            f"Unsupported input format for {input_path}. Expected topics JSON or pending JSON."
        )

    write_json(Path(args.output), pending_items)
    write_json(Path(args.slang_output), slang_payload)

    print(
        json.dumps(
            {
                "command": "clean",
                "input_path": str(input_path),
                "output_path": str(Path(args.output)),
                "slang_output_path": str(Path(args.slang_output)),
                "pending_item_count": len(pending_items),
                "slang_term_count": len(slang_payload["terms"]),
                "skipped": skipped,
            },
            ensure_ascii=False,
        )
    )
    return 0


def normalize_sample(sample: dict[str, Any]) -> dict[str, str]:
    instruction = normalize_text(sample.get("instruction"))
    input_text = normalize_text(sample.get("input"))
    output_text = normalize_text(sample.get("output"))
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
    }


def sample_key(sample: dict[str, str]) -> str:
    return hashlib.sha1(
        json.dumps(sample, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_dataset_info(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid dataset info file: {path}")
    return payload


def load_base_samples(dataset_info_path: Path, aliases: list[str]) -> tuple[list[dict[str, str]], dict[str, int]]:
    dataset_info = load_dataset_info(dataset_info_path)
    samples: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    for alias in aliases:
        if alias not in dataset_info:
            raise RuntimeError(f"Dataset alias '{alias}' not found in {dataset_info_path}")
        dataset_path = dataset_info_path.parent / dataset_info[alias]["file_name"]
        payload = read_json(dataset_path)
        if not isinstance(payload, list):
            raise RuntimeError(f"Dataset '{alias}' must be a JSON array: {dataset_path}")
        normalized = [normalize_sample(item) for item in payload]
        samples.extend(normalized)
        counts[alias] = len(normalized)
    return samples, counts


def load_translated_samples(paths: list[str]) -> tuple[list[dict[str, str]], int]:
    translated_samples: list[dict[str, str]] = []
    skipped_pending = 0
    for raw_path in paths:
        payload = read_json(Path(raw_path))
        if not isinstance(payload, list):
            raise RuntimeError(f"Translated dataset must be a JSON array: {raw_path}")
        for item in payload:
            sample = normalize_sample(item)
            if not sample["output"]:
                skipped_pending += 1
                continue
            translated_samples.append(sample)
    return translated_samples, skipped_pending


def command_merge(args: argparse.Namespace) -> int:
    aliases = [alias.strip() for alias in args.base_datasets.split(",") if alias.strip()]
    base_samples, base_counts = load_base_samples(Path(args.dataset_info), aliases)
    translated_samples, skipped_pending = load_translated_samples(args.translated_input)

    merged: list[dict[str, str]] = []
    seen_keys: set[str] = set()
    duplicate_count = 0

    for sample in base_samples + translated_samples:
        if not sample["instruction"] or not sample["input"] or not sample["output"]:
            duplicate_count += 1
            continue
        key = sample_key(sample)
        if key in seen_keys:
            duplicate_count += 1
            continue
        seen_keys.add(key)
        merged.append(sample)

    write_json(Path(args.output), merged)

    print(
        json.dumps(
            {
                "command": "merge",
                "dataset_info": str(Path(args.dataset_info)),
                "base_datasets": base_counts,
                "translated_input_count": len(translated_samples),
                "skipped_pending_count": skipped_pending,
                "deduped_or_invalid_count": duplicate_count,
                "merged_count": len(merged),
                "output_path": str(Path(args.output)),
            },
            ensure_ascii=False,
        )
    )
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "clean":
        return command_clean(args)
    if args.command == "merge":
        return command_merge(args)
    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
