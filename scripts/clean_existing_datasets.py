#!/usr/bin/env python3
"""Clean current datasets, build bilingual corpora, and export review queues."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DATASET_DIR = Path("dataset")
TEMP_DIR = Path("temp")
SOURCE_DIR = TEMP_DIR / "source"
LDOT_TRANSLATION_PATH = SOURCE_DIR / "ldot_translation.json"
LINUXDO_TRANSLATED_PATH = TEMP_DIR / "ldot_linuxdo_v002.json"
GPT_DATASETS = {
    "gpt_base": SOURCE_DIR / "gpt_base.json",
    "gpt_mix": SOURCE_DIR / "gpt_mix.json",
    "gpt_term": SOURCE_DIR / "gpt_term.json",
    "gpt_high": SOURCE_DIR / "gpt_high.json",
}
SLANG_SOURCE_PATH = SOURCE_DIR / "ldot_slang_consistency.json"
SHORT_REPLIES_SOURCE_PATH = SOURCE_DIR / "ldot_forum_short_replies.json"

OUTPUT_IDENTITY = TEMP_DIR / "ldot_identity_clean.json"
OUTPUT_COMMUNITY_FORWARD = TEMP_DIR / "ldot_community_zh_en_clean.json"
OUTPUT_COMMUNITY_PARALLEL = TEMP_DIR / "ldot_community_parallel_clean.json"
OUTPUT_TECH_PARALLEL = TEMP_DIR / "ldot_tech_parallel_clean.json"
OUTPUT_SLANG_PARALLEL = TEMP_DIR / "ldot_slang_parallel.json"
OUTPUT_SHORT_REPLIES_PARALLEL = TEMP_DIR / "ldot_short_replies_parallel.json"
OUTPUT_LINUXDO_PARALLEL = TEMP_DIR / "ldot_linuxdo_parallel.json"
OUTPUT_REVIEW = TEMP_DIR / "ldot_translation_review.json"
OUTPUT_FINAL = TEMP_DIR / "ldot_train_clean_bilingual.json"

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

MANUAL_LDOT_OUTPUT_FIXES = {
    16: (
        "Ridiculous. The more people there are, the more weirdos you get; "
        "the fewer people there are, the weirder each weirdo becomes."
    ),
    17: "Hahaha, the petty villain has already been dealt with.",
    19: (
        "Princess, sorry to trouble you again. Next time, let me handle "
        "this sort of thing; Princess, I'll bring this insolent commoner "
        "back to you right away."
    ),
    23: (
        "TG doesn't allow screenshots over there, so I had to pull out my "
        "backup phone :distorted_face:"
    ),
    26: (
        "Because the page shows the referrer's profile, he thinks that "
        "basically makes the referrer his American sponsor. In his mind, "
        "only a recommendation letter written by someone important really "
        "counts and looks prestigious enough."
    ),
    29: (
        "If you're too lazy to do the pre-application, then just don't come in, meow. "
        "If it really doesn't work, you can always make your own QQ group and rename it "
        "to cosplay as the LinuxDo forum, mm."
    ),
    30: (
        "Effective immediately, Nebula Airport is officially shutting down and will no "
        "longer accept new user registrations. Going forward, it will no longer operate "
        "as an 'airport' service or provide related maintenance support. Only four home "
        "broadband nodes will be kept for personal or small-scale use. No new nodes will "
        "be added, no routes will be expanded, and no public-benefit service will be provided. "
        "This decision was made because the time, effort, and cost required for long-term "
        "maintenance have kept increasing, while stability and risk control have become "
        "harder and harder to balance. To avoid uncontrollable outages or unnecessary "
        "trouble later on, I chose to end operations proactively. Thank you all for your "
        "understanding and support along the way, and I sincerely apologize for the inconvenience."
    ),
    40: (
        "Maybe you could add paid subscriptions. After all, paying one provider or another "
        "is still paying. You could offer cheaper but less stable nodes, as well as more "
        "expensive stable nodes or clean-IP nodes."
    ),
    31: "Running a public service is no easy task :heart: Thanks, big cheese.",
    43: (
        "I redeemed it with points, I was just asking casually with no bad intentions "
        ":pleading_face: (and of course I'm very grateful for everything the big cheese "
        "has done. I've mostly kept it as a backup and barely used it. I'm also preparing my own "
        "public-benefit site.)"
    ),
    44: "Thanks for all your hard work, big cheese.",
    36: "Thank you for all you've done, big cheese!",
    39: "No wonder I couldn't connect today. Thanks for all your hard work, Nebula big cheese.",
    53: "A big cheese reminded me, so I deleted it...",
    45: (
        "I registered and bought a plan on March 7, went on a trip on March 8, only got "
        "back a couple of days ago, and when I opened L Station today it said the airport "
        "had shut down. My heart just went ice-cold. :pleading_face:"
    ),
    51: (
        "Sincerity, friendliness, unity, and professionalism: together, let's build a "
        "community we're all proud of."
    ),
    52: (
        "I've never done cosplay, never done the pink-haired fox thing, and never done "
        "cross-dressing either, so you'll just have to make do with this, okay? "
        ":distorted_face:"
    ),
    60: (
        "That's rough, my friend. Even though I can't help laughing :laughing:, let me "
        "give you a like to help you recover a bit."
    ),
    62: (
        "Mm, then people like it even more, it's a scruffy pink-haired fox after all."
    ),
    68: "Could one of the big wheels help take a look at this?",
    80: "Nowadays everyone uses AI for deployment. Credit cards are easy to get on Xianyu, but thanks for sharing, big cheese.",
}


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


def has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def is_identity_instruction(instruction: str) -> bool:
    return "身份信息" in instruction


def is_probably_truncated(source: str, target: str) -> bool:
    source = source.strip()
    target = target.strip()
    if not source or not target:
        return True
    if has_cjk(target):
        return True
    if len(source) >= 20 and len(target) < len(source) * 0.55:
        return True
    if len(source) >= 20 and len(target.split()) <= 2:
        return True
    return False


def dedupe_samples(samples: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for sample in samples:
        key = json.dumps(sample, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sample)
    return deduped


def build_sample(instruction: str, input_text: str, output_text: str) -> dict[str, str]:
    return {
        "instruction": normalize_text(instruction),
        "input": normalize_text(input_text),
        "output": normalize_text(output_text),
    }


def clean_ldot_translation() -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, Any]],
]:
    payload = read_json(LDOT_TRANSLATION_PATH)
    identity_samples: list[dict[str, str]] = []
    community_forward: list[dict[str, str]] = []
    review_samples: list[dict[str, Any]] = []

    for index, item in enumerate(payload):
        instruction = normalize_text(item.get("instruction"))
        input_text = normalize_text(item.get("input"))
        output_text = normalize_text(item.get("output"))

        if is_identity_instruction(instruction):
            identity_samples.append(build_sample(instruction, input_text, output_text))
            continue

        if index in MANUAL_LDOT_OUTPUT_FIXES:
            output_text = normalize_text(MANUAL_LDOT_OUTPUT_FIXES[index])

        if is_probably_truncated(input_text, output_text):
            review_samples.append(
                {
                    "source_dataset": "ldot_translation",
                    "index": index,
                    "reason": "suspicious_or_truncated",
                    "instruction": instruction,
                    "input": input_text,
                    "output": output_text,
                }
            )
            continue

        community_forward.append(build_sample(INSTRUCTION_ZH_EN, input_text, output_text))

    identity_samples = dedupe_samples(identity_samples)
    community_forward = dedupe_samples(community_forward)
    return identity_samples, community_forward, review_samples


def build_reverse_samples(
    samples: list[dict[str, str]],
    instruction: str,
) -> list[dict[str, str]]:
    return [
        build_sample(
            instruction,
            sample["output"],
            sample["input"],
        )
        for sample in samples
    ]


def clean_gpt_parallel() -> list[dict[str, str]]:
    en_zh_samples: list[dict[str, str]] = []
    zh_en_samples: list[dict[str, str]] = []

    for name, path in GPT_DATASETS.items():
        payload = read_json(path)
        if name == "gpt_term":
            forward_instruction = TERM_INSTRUCTION_EN_ZH
            reverse_instruction = TERM_INSTRUCTION_ZH_EN
        else:
            forward_instruction = INSTRUCTION_EN_ZH
            reverse_instruction = INSTRUCTION_ZH_EN

        normalized_forward = [
            build_sample(forward_instruction, item.get("input"), item.get("output"))
            for item in payload
        ]
        normalized_forward = dedupe_samples(normalized_forward)
        en_zh_samples.extend(normalized_forward)
        zh_en_samples.extend(build_reverse_samples(normalized_forward, reverse_instruction))

    return dedupe_samples(en_zh_samples + zh_en_samples)


def load_slang_parallel() -> list[dict[str, str]]:
    if not SLANG_SOURCE_PATH.exists():
        return []

    payload = read_json(SLANG_SOURCE_PATH)
    forward = [
        build_sample(
            item.get("instruction") or INSTRUCTION_ZH_EN,
            item.get("input"),
            item.get("output"),
        )
        for item in payload
    ]
    forward = dedupe_samples(forward)
    reverse = build_reverse_samples(forward, INSTRUCTION_EN_ZH)
    return dedupe_samples(forward + reverse)


def load_short_replies_parallel() -> list[dict[str, str]]:
    if not SHORT_REPLIES_SOURCE_PATH.exists():
        return []

    payload = read_json(SHORT_REPLIES_SOURCE_PATH)
    forward = [
        build_sample(
            item.get("instruction") or INSTRUCTION_ZH_EN,
            item.get("input"),
            item.get("output"),
        )
        for item in payload
    ]
    forward = dedupe_samples(forward)
    reverse = build_reverse_samples(forward, INSTRUCTION_EN_ZH)
    return dedupe_samples(forward + reverse)


def load_linuxdo_parallel() -> list[dict[str, str]]:
    if not LINUXDO_TRANSLATED_PATH.exists():
        return []

    payload = read_json(LINUXDO_TRANSLATED_PATH)
    forward = [
        build_sample(INSTRUCTION_ZH_EN, item.get("input"), item.get("output"))
        for item in payload
        if normalize_text(item.get("output"))
    ]
    forward = dedupe_samples(forward)
    reverse = build_reverse_samples(forward, INSTRUCTION_EN_ZH)
    return dedupe_samples(forward + reverse)


def main() -> int:
    identity_samples, community_forward, review_samples = clean_ldot_translation()
    community_reverse = build_reverse_samples(community_forward, INSTRUCTION_EN_ZH)
    community_parallel = dedupe_samples(community_forward + community_reverse)
    tech_parallel = clean_gpt_parallel()
    slang_parallel = load_slang_parallel()
    short_replies_parallel = load_short_replies_parallel()
    linuxdo_parallel = load_linuxdo_parallel()

    final_train = dedupe_samples(
        identity_samples
        + community_parallel
        + tech_parallel
        + slang_parallel
        + short_replies_parallel
        + linuxdo_parallel
    )

    write_json(OUTPUT_IDENTITY, identity_samples)
    write_json(OUTPUT_COMMUNITY_FORWARD, community_forward)
    write_json(OUTPUT_COMMUNITY_PARALLEL, community_parallel)
    write_json(OUTPUT_TECH_PARALLEL, tech_parallel)
    write_json(OUTPUT_SLANG_PARALLEL, slang_parallel)
    write_json(OUTPUT_SHORT_REPLIES_PARALLEL, short_replies_parallel)
    write_json(OUTPUT_LINUXDO_PARALLEL, linuxdo_parallel)
    write_json(OUTPUT_REVIEW, review_samples)
    write_json(OUTPUT_FINAL, final_train)

    print(
        json.dumps(
            {
                "identity_count": len(identity_samples),
                "community_forward_count": len(community_forward),
                "community_parallel_count": len(community_parallel),
                "tech_parallel_count": len(tech_parallel),
                "slang_parallel_count": len(slang_parallel),
                "short_replies_parallel_count": len(short_replies_parallel),
                "linuxdo_parallel_count": len(linuxdo_parallel),
                "review_count": len(review_samples),
                "final_train_count": len(final_train),
                "final_train_path": str(OUTPUT_FINAL),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
