
import csv
import sys
from pathlib import Path
from collections import Counter

# Make the project root importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_engine import analyze_message


DATASET_FILE = Path(__file__).parent / "evaluation_dataset.csv"
RESULTS_FILE = Path(__file__).parent / "evaluation_results.csv"

SCAM_LABELS = {
    "phishing scam",
    "fake job scam",
    "investment scam",
    "impersonation scam",
    "payment fraud",
    "romance scam",
}


def normalize_label(label):
    """Normalize a label for comparison."""
    return label.strip().casefold()


def split_categories(predicted):
    """Convert a comma-separated prediction into normalized categories."""
    if normalize_label(predicted) in {"unknown", "", "error"}:
        return set()

    return {
        normalize_label(category)
        for category in predicted.split(",")
        if category.strip()
    }


def main():
    with open(DATASET_FILE, "r", encoding="utf-8-sig", newline="") as file:
        test_messages = list(csv.DictReader(file))

    if not test_messages:
        print("No test messages found in the dataset.")
        return

    results = []

    # Binary scam-detection metrics.
    tp = tn = fp = fn = 0

    # Category-level metrics for scam messages.
    scam_category_total = 0
    exact_category_matches = 0
    expected_category_found = 0
    extra_category_predictions = 0

    print(f"Running evaluation on {len(test_messages)} messages...\n")

    for index, row in enumerate(test_messages, start=1):
        message_id = row["id"]
        message = row["message"]
        expected = row["expected_label"].strip()
        expected_normalized = normalize_label(expected)
        expected_is_scam = expected_normalized in SCAM_LABELS

        try:
            prediction = analyze_message(message)

            predicted = str(
                prediction.get("scam_type", "Unknown")
            ).strip()
            risk_level = str(
                prediction.get("risk_level", "Unknown")
            )
            warning_signs = prediction.get("warning_signs", [])

            predicted_categories = split_categories(predicted)
            predicted_is_scam = bool(
                predicted_categories & SCAM_LABELS
            )

            # Binary scam detection.
            if expected_is_scam and predicted_is_scam:
                tp += 1
            elif not expected_is_scam and not predicted_is_scam:
                tn += 1
            elif not expected_is_scam and predicted_is_scam:
                fp += 1
            else:
                fn += 1

            # Strict category evaluation for scam messages.
            if expected_is_scam:
                scam_category_total += 1

                expected_found = (
                    expected_normalized in predicted_categories
                )
                if expected_found:
                    expected_category_found += 1

                if predicted_categories == {expected_normalized}:
                    exact_category_matches += 1

                if predicted_categories - {expected_normalized}:
                    extra_category_predictions += 1

            # A strict overall match requires exactly the expected
            # category for scams, or Unknown for legitimate messages.
            if expected_is_scam:
                correct = predicted_categories == {expected_normalized}
            else:
                correct = not predicted_is_scam

            results.append({
                "id": message_id,
                "message": message,
                "expected_label": expected,
                "predicted_label": predicted,
                "risk_level": risk_level,
                "expected_is_scam": expected_is_scam,
                "predicted_is_scam": predicted_is_scam,
                "expected_category_found": (
                    expected_normalized in predicted_categories
                    if expected_is_scam else ""
                ),
                "exact_category_match": (
                    predicted_categories == {expected_normalized}
                    if expected_is_scam else ""
                ),
                "correct": correct,
                "warning_sign_count": len(warning_signs),
                "error": "",
            })

            status = "PASS" if correct else "MISMATCH"
            print(
                f"[{index}/{len(test_messages)}] ID {message_id}: "
                f"{status} (expected={expected}; predicted={predicted})"
            )

        except Exception as error:
            results.append({
                "id": message_id,
                "message": message,
                "expected_label": expected,
                "predicted_label": "ERROR",
                "risk_level": "Unknown",
                "expected_is_scam": expected_is_scam,
                "predicted_is_scam": "",
                "expected_category_found": "",
                "exact_category_match": "",
                "correct": False,
                "warning_sign_count": 0,
                "error": str(error),
            })
            print(f"[{index}/{len(test_messages)}] ID {message_id}: ERROR")

    fieldnames = [
        "id",
        "message",
        "expected_label",
        "predicted_label",
        "risk_level",
        "expected_is_scam",
        "predicted_is_scam",
        "expected_category_found",
        "exact_category_match",
        "correct",
        "warning_sign_count",
        "error",
    ]

    with open(RESULTS_FILE, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    completed_results = [row for row in results if not row["error"]]
    errors = len(results) - len(completed_results)
    correct_count = sum(
        str(row["correct"]).casefold() == "true"
        for row in completed_results
    )

    binary_total = tp + tn + fp + fn
    binary_accuracy = (tp + tn) / binary_total if binary_total else 0
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    print("\n--- Evaluation Summary ---")
    print(f"Total messages: {len(results)}")
    print(f"Completed: {len(completed_results)}")
    print(f"Errors: {errors}")
    print(f"Strict overall matches: {correct_count}/{len(completed_results)}")

    print("\n--- Binary Scam Detection ---")
    print(f"True positives (scams detected): {tp}")
    print(f"True negatives (legitimate not flagged): {tn}")
    print(f"False positives (legitimate flagged): {fp}")
    print(f"False negatives (scams missed): {fn}")
    print(f"Accuracy: {binary_accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 score: {f1:.2%}")

    print("\n--- Scam Category Classification ---")
    print(f"Scam messages evaluated: {scam_category_total}")
    print(
        "Expected category present: "
        f"{expected_category_found}/{scam_category_total}"
    )
    print(
        "Exact category matches: "
        f"{exact_category_matches}/{scam_category_total}"
    )
    print(
        "Predictions with extra categories: "
        f"{extra_category_predictions}"
    )

    print(f"\nResults saved to: {RESULTS_FILE}")
    print("Note: These metrics describe this dataset, not real-world accuracy.")


if __name__ == "__main__":
    main()
