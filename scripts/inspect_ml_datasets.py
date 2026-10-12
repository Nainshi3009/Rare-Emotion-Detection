from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preparation import (  # noqa: E402
    OVERSAMPLING_TARGET_COUNT,
    RANDOM_SEED,
    TARGET_EMOTIONS,
    check_split_integrity,
    create_gold_only,
    create_standard_oversampling,
    create_unfiltered_synthetic,
    create_vad_filtered_synthetic,
    get_label_mapping,
    get_target_training_counts,
    identify_target_emotions,
    load_goemotions,
    load_synthetic_candidates,
    validate_condition,
)


def _print_integrity(label, checks):
    print(f"{label} integrity checks:")
    for name, value in checks.items():
        print(f"  {name}: {value}")


def main() -> int:
    dataset = load_goemotions()
    gold = create_gold_only(dataset)
    label_mapping = get_label_mapping(dataset)
    target_mapping = identify_target_emotions(label_mapping)

    print("Split sizes:")
    for split in ("train", "validation", "test"):
        print(f"  {split}: {len(gold[split]):,}")

    print("\nLabel mapping:")
    for item in label_mapping:
        print(f"  {item['index']:2d}: {item['emotion']}")

    original_counts = get_target_training_counts(gold["train"])
    print("\nTarget emotion counts in original train:")
    for emotion in TARGET_EMOTIONS:
        print(f"  {emotion} ({next(x['index'] for x in target_mapping if x['emotion'] == emotion)}): {original_counts[emotion]}")

    oversampled_train = create_standard_oversampling(
        gold["train"],
        random_seed=RANDOM_SEED,
        target_count=OVERSAMPLING_TARGET_COUNT,
    )
    print(
        "\nStandard oversampling: "
        f"target={OVERSAMPLING_TARGET_COUNT} per target label, seed={RANDOM_SEED}"
    )
    for emotion, count in get_target_training_counts(oversampled_train).items():
        print(f"  {emotion}: {count}")
    print(
        "  rows_added="
        f"{int(oversampled_train['oversampled'].fillna(False).sum())}"
    )

    candidates = load_synthetic_candidates()
    availability = candidates.attrs["availability"]
    print("\nAvailable synthetic candidate counts (exact normalized duplicates removed):")
    for emotion in TARGET_EMOTIONS:
        print(f"  {emotion}: {availability['counts_per_emotion'][emotion]}")
    print(f"  available target emotions: {availability['available_emotions']}")
    print(f"  missing target emotions: {availability['missing_emotions']}")
    print(f"  duplicate rows removed: {availability['exact_duplicates_removed']}")

    unfiltered = create_unfiltered_synthetic(gold=gold, synthetic_candidates=candidates)
    filtered = create_vad_filtered_synthetic(gold=gold, synthetic_candidates=candidates)
    print("\nSynthetic condition completeness:")
    for name, condition in (
        ("C UNFILTERED_SYNTHETIC", unfiltered),
        ("D VAD_FILTERED_SYNTHETIC", filtered),
    ):
        print(
            f"  {name}: complete={condition['metadata']['complete']}, "
            f"blocked_for_final_experiment={condition['metadata']['blocked_for_final_experiment']}, "
            f"training_rows={len(condition['train'])}"
        )
        print(
            f"    available={condition['metadata']['synthetic_completeness']['available_emotions']}; "
            f"missing={condition['metadata']['synthetic_completeness']['missing_emotions']}"
        )

    print("\nLeakage/duplicate checks:")
    baseline_checks = check_split_integrity(gold)
    _print_integrity("Original gold splits", baseline_checks)
    oversampled_condition = {
        "train": oversampled_train,
        "validation": gold["validation"],
        "test": gold["test"],
    }
    _print_integrity(
        "Standard oversampling",
        check_split_integrity(oversampled_condition),
    )
    _print_integrity(
        "GOLD_ONLY",
        validate_condition(gold, gold),
    )
    for name, condition in (
        ("Unfiltered synthetic", unfiltered),
        ("VAD-filtered synthetic", filtered),
    ):
        checks = validate_condition(gold, condition)
        _print_integrity(name, checks)

    if not baseline_checks["no_split_leakage"]:
        print("WARNING: existing GoEmotions split ID/text overlap was detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
