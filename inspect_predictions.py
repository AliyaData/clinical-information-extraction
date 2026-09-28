import json

with open("results/evaluation_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

successful = [
    r for r in results
    if r["status"] == "success"
]

print("========================================")
print("SUCCESSFUL EXTRACTION INSPECTION")
print("========================================")

for i, result in enumerate(successful[:15], start=1):

    print(f"\n========== SAMPLE {i} ==========")

    print("ID:", result["id"])

    print("\nGROUND TRUTH:")
    print(json.dumps(
        result["ground_truth"],
        indent=2,
        ensure_ascii=False
    ))

    print("\nMODEL PREDICTION:")
    print(json.dumps(
        result["prediction"],
        indent=2,
        ensure_ascii=False
    ))

    print("\nCLINICAL TEXT:")
    print(result["text"][:800])

    print("\n" + "-" * 60)