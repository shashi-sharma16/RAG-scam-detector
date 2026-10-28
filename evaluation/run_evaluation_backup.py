
import csv
import sys
from pathlib import Path

# Make the project root importable when this script is run.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_engine import analyze_message


DATASET_FILE = Path(__file__).parent / "evaluation_dataset.csv"
RESULTS_FILE = Path(__file__).parent / "evaluation_results.csv"


def normalize_label(label):
    """Standardize labels for comparison."""
    return label.strip().casefold()


def main():
    with open(DATASET_FILE, "r", encoding="utf-8-sig", newline="") as file:
        test_messages = list(csv.DictReader(file))

    if not test_messages:
        print("No test messages found in the dataset.")
        return

    results = []

    print(f"Running evaluation on {len(test_messages)} messages...\n")

    for index, row in enumerate(test_messages, start=1):
        message_id = row["id"]
        message = row["message"]
        expected = row["expected_label"].strip()

        try:
            prediction = analyze_message(message)

            predicted = prediction.get("scam_type", "Unknown").strip()
            risk_level = prediction.get("risk_level", "Unknown")
            warning_signs = prediction.get("warning_signs", [])

            # A multi-category prediction is correct if it includes
            # the expected category. Legitimate messages are handled
            # separately because the detector has no Legitimate label.
            if normalize_label(expected) == "legitimate":
                correct = normalize_label(predicted) == "unknown"
            else:
                predicted_categories = [
                    category.strip()
                    for category in predicted.split(",")
                ]
                correct = any(
                    normalize_label(category) == normalize_label(expected)
                    for category in predicted_categories
                )

            results.append({
                "id": message_id,
                "message": message,
                "expected_label": expected,
                "predicted_label": predicted,
                "risk_level": risk_level,
                "correct": correct,
                "warning_sign_count": len(warning_signs),
                "error": "",
            })

            status = "PASS" if correct else "MISMATCH"
            print(
                f"[{index}/{len(test_messages)}] "
                f"ID {message_id}: {status} "
                f"(expected={expected}; predicted={predicted})"
            )

        except Exception as error:
            results.append({
                "id": message_id,
                "message": message,
                "expected_label": expected,
                "predicted_label": "ERROR",
                "risk_level": "Unknown",
                "correct": False,
                "warning_sign_count": 0,
                "error": str(error),
            })

            print(f"[{index}/{len(test_messages)}] ID {message_id}: ERROR")

    with open(RESULTS_FILE, "w", encoding="utf-8-sig", newline="") as file:
        fieldnames = [
            "id",
            "message",
            "expected_label",
            "predicted_label",
            "risk_level",
            "correct",
            "warning_sign_count",
            "error",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    completed = sum(not row["error"] for row in results)
    correct_count = sum(row["correct"] for row in results)

    print("\n--- Evaluation Summary ---")
    print(f"Total messages: {len(results)}")
    print(f"Completed: {completed}")
    print(f"Errors: {len(results) - completed}")
    print(f"Category/legitimate matches: {correct_count}/{len(results)}")
    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
