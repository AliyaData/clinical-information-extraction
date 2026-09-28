import json
from pathlib import Path


INPUT_FILE = Path("results/evaluation_results.json")
OUTPUT_FILE = Path("results/accuracy_report.json")

FIELDS = [
    "conditions",
    "drugs",
    "dosages",
    "procedures"
]


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)


total = len(results)

successful = sum(
    1 for r in results
    if r["status"] == "success"
)

failed = total - successful


# --------------------------------------------------
# Exact match
# --------------------------------------------------

exact_matches = 0

for r in results:

    prediction = r["prediction"]
    ground_truth = r["ground_truth"]

    if prediction == ground_truth:
        exact_matches += 1


exact_match_accuracy = exact_matches / total


# --------------------------------------------------
# Field-level metrics
# --------------------------------------------------

true_positive = 0
false_positive = 0
false_negative = 0


field_metrics = {}


for field in FIELDS:

    tp = 0
    fp = 0
    fn = 0

    for r in results:

        prediction = set(r["prediction"].get(field, []))
        actual = set(r["ground_truth"].get(field, []))

        tp += len(prediction & actual)
        fp += len(prediction - actual)
        fn += len(actual - prediction)

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    field_metrics[field] = {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    true_positive += tp
    false_positive += fp
    false_negative += fn


# --------------------------------------------------
# Overall micro metrics
# --------------------------------------------------

overall_precision = (
    true_positive / (true_positive + false_positive)
    if (true_positive + false_positive) > 0
    else 0
)

overall_recall = (
    true_positive / (true_positive + false_negative)
    if (true_positive + false_negative) > 0
    else 0
)

overall_f1 = (
    2 * overall_precision * overall_recall /
    (overall_precision + overall_recall)
    if (overall_precision + overall_recall) > 0
    else 0
)


# --------------------------------------------------
# All-empty case performance
# --------------------------------------------------

empty_cases = [
    r for r in results
    if all(
        len(r["ground_truth"].get(field, [])) == 0
        for field in FIELDS
    )
]

empty_correct = 0

for r in empty_cases:

    prediction = r["prediction"]

    prediction_empty = all(
        len(prediction.get(field, [])) == 0
        for field in FIELDS
    )

    if prediction_empty:
        empty_correct += 1


empty_case_accuracy = (
    empty_correct / len(empty_cases)
    if empty_cases
    else 0
)


# --------------------------------------------------
# Final report
# --------------------------------------------------

report = {

    "dataset": {
        "total_samples": total,
        "successful_extractions": successful,
        "failed_extractions": failed,
        "validation_success_rate": successful / total
    },

    "exact_match": {
        "correct": exact_matches,
        "total": total,
        "accuracy": exact_match_accuracy
    },

    "overall_field_level_metrics": {
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1
    },

    "field_metrics": field_metrics,

    "all_empty_cases": {
        "total": len(empty_cases),
        "correctly_empty": empty_correct,
        "accuracy": empty_case_accuracy
    }
}


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        report,
        f,
        indent=2
    )


# --------------------------------------------------
# Print results
# --------------------------------------------------

print()
print("========================================")
print("CLINICAL EXTRACTION ACCURACY REPORT")
print("========================================")

print(f"Total samples: {total}")
print(f"Successful extractions: {successful}")
print(f"Failed extractions: {failed}")

print(
    f"Validation success rate: "
    f"{successful / total * 100:.2f}%"
)

print()

print(
    f"Exact-match accuracy: "
    f"{exact_match_accuracy * 100:.2f}%"
)

print()

print("Overall field-level metrics:")
print(
    f"Precision: {overall_precision * 100:.2f}%"
)
print(
    f"Recall:    {overall_recall * 100:.2f}%"
)
print(
    f"F1-score:  {overall_f1 * 100:.2f}%"
)

print()

print("Field-level results:")

for field, metrics in field_metrics.items():

    print(f"\n{field.upper()}")

    print(
        f"  Precision: "
        f"{metrics['precision'] * 100:.2f}%"
    )

    print(
        f"  Recall: "
        f"{metrics['recall'] * 100:.2f}%"
    )

    print(
        f"  F1: "
        f"{metrics['f1'] * 100:.2f}%"
    )


print()

print("All-empty cases:")
print(f"  Total: {len(empty_cases)}")
print(f"  Correctly empty: {empty_correct}")
print(
    f"  Accuracy: "
    f"{empty_case_accuracy * 100:.2f}%"
)

print()

print("Report saved to:")
print(OUTPUT_FILE)