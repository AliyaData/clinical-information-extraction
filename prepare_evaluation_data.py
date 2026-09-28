import json
import random
import requests
from pathlib import Path

DATASET = "AmareshHebbar/medical-ner-sft"
SPLIT = "test"

TOTAL_ROWS = 1668
BATCH_SIZE = 100

random.seed(42)

all_rows = []

print("Downloading test dataset...")

for offset in range(0, TOTAL_ROWS, BATCH_SIZE):

    url = (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset={DATASET}"
        f"&config=default"
        f"&split={SPLIT}"
        f"&offset={offset}"
        f"&length={BATCH_SIZE}"
    )

    response = requests.get(url)
    response.raise_for_status()

    data = response.json()

    rows = data["rows"]

    all_rows.extend(rows)

    print(f"Downloaded {len(all_rows)} / {TOTAL_ROWS}")


print("\nProcessing records...")

non_empty = []
all_empty = []

for item in all_rows:

    row = item["row"]

    messages = row["messages"]

    clinical_text = messages[1]["content"]

    ground_truth_text = messages[2]["content"]

    ground_truth = json.loads(ground_truth_text)

    record = {
        "id": item.get("row_idx"),
        "text": clinical_text,
        "ground_truth": ground_truth
    }

    is_all_empty = all(
        len(ground_truth[field]) == 0
        for field in [
            "conditions",
            "drugs",
            "dosages",
            "procedures"
        ]
    )

    if is_all_empty:
        all_empty.append(record)
    else:
        non_empty.append(record)


print("\n========== DATASET SUMMARY ==========")
print("Total records:", len(all_rows))
print("Non-empty records:", len(non_empty))
print("All-empty records:", len(all_empty))


# Select 50 non-empty + 50 all-empty
selected_non_empty = random.sample(non_empty, 50)
selected_all_empty = random.sample(all_empty, 50)

evaluation_samples = selected_non_empty + selected_all_empty

random.shuffle(evaluation_samples)


output_path = Path("data/evaluation_samples.json")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(
        evaluation_samples,
        f,
        indent=2,
        ensure_ascii=False
    )


print("\n========== EVALUATION DATA ==========")
print("Selected non-empty:", len(selected_non_empty))
print("Selected all-empty:", len(selected_all_empty))
print("Total evaluation samples:", len(evaluation_samples))

print("\nSaved to:")
print(output_path)

print("\nDone!")