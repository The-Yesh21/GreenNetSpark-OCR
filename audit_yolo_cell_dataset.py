import argparse
from collections import Counter
from pathlib import Path

from yolo_cells import YOLO_CELL_CLASSES


EXPECTED_CLASSES_PER_ROW = len(YOLO_CELL_CLASSES)


def count_labels(label_path: Path) -> Counter:
    counts = Counter()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls_id = int(line.split()[0])
        counts[YOLO_CELL_CLASSES[cls_id]] += 1
    return counts


def audit_split(root: Path, split: str, expected_rows: int | None) -> list[str]:
    issues: list[str] = []
    for label_path in sorted((root / split / "labels").glob("*.txt")):
        counts = count_labels(label_path)
        total = sum(counts.values())
        row_count = total / EXPECTED_CLASSES_PER_ROW if total else 0
        balanced = len(set(counts.get(name, 0) for name in YOLO_CELL_CLASSES)) == 1
        wrong_row_count = expected_rows is not None and row_count != expected_rows
        if not balanced or total % EXPECTED_CLASSES_PER_ROW != 0 or wrong_row_count:
            issues.append(
                f"{split}/{label_path.name}: total={total}, rows={row_count:.2f}, "
                + ", ".join(f"{name}={counts.get(name, 0)}" for name in YOLO_CELL_CLASSES)
            )
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit YOLO cell split labels for class/count problems.")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/yolo_cell_split"))
    parser.add_argument("--expected-rows", type=int, default=15, help="Expected table body rows per image")
    args = parser.parse_args()

    all_issues = audit_split(args.dataset, "train", args.expected_rows) + audit_split(args.dataset, "valid", args.expected_rows)
    if not all_issues:
        print("No label count issues found.")
        return
    print("Label files needing review:")
    for issue in all_issues:
        print(f"- {issue}")


if __name__ == "__main__":
    main()
