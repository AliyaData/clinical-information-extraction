import json
from pathlib import Path

from extraction_pipeline import extract_clinical_information


# --------------------------------------------------
# Load evaluation dataset
# --------------------------------------------------

input_path = Path("data/evaluation_samples.json")
output_path = Path("results/evaluation_results.json")

with open(input_path, "r", encoding="utf-8") as f:
    samples = json.load(f)


print("========================================")
print("CLINICAL INFORMATION EXTRACTION EVALUATION")
print("========================================")
print(f"Total samples: {len(samples)}")
print()


results = []


# --------------------------------------------------
# Run extraction on every sample
# --------------------------------------------------

for i, sample in enumerate(samples, start=1):

    print(f"Processing sample {i}/{len(samples)}...")

    text = sample["text"]
    ground_truth = sample["ground_truth"]

    try:
        result = extract_clinical_information(text)

        prediction = result.get(
            "data",
            {
                "conditions": [],
                "drugs": [],
                "dosages": [],
                "procedures": []
            }
        )

        status = result.get("status")

        error = result.get("error")

    except Exception as e:

        prediction = {
            "conditions": [],
            "drugs": [],
            "dosages": [],
            "procedures": []
        }

        status = "pipeline_failed"
        error = str(e)


    # --------------------------------------------------
    # Store result
    # --------------------------------------------------

    results.append({
        "id": sample["id"],
        "text": text,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "status": status,
        "error": error
    })


    # Save progress after every sample
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )


# --------------------------------------------------
# Summary
# --------------------------------------------------

successful = sum(
    1 for r in results
    if r["status"] == "success"
)

failed = len(results) - successful


print()
print("========================================")
print("EVALUATION COMPLETE")
print("========================================")

print(f"Total samples: {len(results)}")
print(f"Successful extractions: {successful}")
print(f"Failed extractions: {failed}")

print()
print("Results saved to:")
print(output_path)