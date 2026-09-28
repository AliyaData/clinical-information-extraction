import json

with open("results/evaluation_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

failures = [
    r for r in results
    if r["status"] != "success"
]

print("========================================")
print("FAILED EXTRACTIONS")
print("========================================")

print("Total failures:", len(failures))
print()

for i, result in enumerate(failures, start=1):

    print(f"========== FAILURE {i} ==========")
    print("ID:", result["id"])
    print("Status:", result["status"])
    print("Error:", result["error"])

    print("\nClinical text:")
    print(result["text"][:500])

    print("\nGround truth:")
    print(json.dumps(
        result["ground_truth"],
        indent=2,
        ensure_ascii=False
    ))

    print("\n")