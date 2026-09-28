import json
import re

from transformers import AutoTokenizer, AutoModelForCausalLM
from schema import ClinicalExtraction


MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
print("Model loaded.")


def extract_json_object(text):
    """
    Extract the first complete JSON object from model output.
    """

    cleaned = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"```\s*",
        "",
        cleaned
    )

    start = cleaned.find("{")

    if start == -1:
        raise ValueError(
            "No JSON object found in model output."
        )

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(cleaned)):

        char = cleaned[i]

        if in_string:

            if escape:
                escape = False

            elif char == "\\":
                escape = True

            elif char == '"':
                in_string = False

        else:

            if char == '"':
                in_string = True

            elif char == "{":
                depth += 1

            elif char == "}":
                depth -= 1

                if depth == 0:
                    return cleaned[start:i + 1]

    raise ValueError(
        "Incomplete JSON object in model output."
    )


def normalize_output(parsed_json):
    """
    Normalize minor structural errors while keeping
    the final Pydantic schema strict.
    """

    required_fields = [
        "conditions",
        "drugs",
        "dosages",
        "procedures"
    ]

    if not isinstance(parsed_json, dict):
        raise ValueError(
            "Model output must be a JSON object."
        )

    normalized = {}
    repairs = []

    for field in required_fields:

        if field not in parsed_json:
            raise ValueError(
                f"Missing required field: {field}"
            )

        value = parsed_json[field]

        if not isinstance(value, list):
            raise ValueError(
                f"Field '{field}' must be a list."
            )

        cleaned_items = []

        for item in value:

            if isinstance(item, str):

                item = item.strip()

                if item:
                    cleaned_items.append(item)

            elif isinstance(item, dict):

                if (
                    "name" in item
                    and isinstance(item["name"], str)
                ):

                    name = item["name"].strip()

                    if name:
                        cleaned_items.append(name)

                        repairs.append(
                            f"Converted dictionary item to "
                            f"string in '{field}'."
                        )

                else:
                    raise ValueError(
                        f"Unsupported object inside "
                        f"'{field}' list."
                    )

            else:
                raise ValueError(
                    f"Invalid item type inside "
                    f"'{field}' list."
                )

        # Remove duplicates while preserving order
        unique_items = []

        for item in cleaned_items:

            if item not in unique_items:
                unique_items.append(item)

        normalized[field] = unique_items

    return normalized, repairs


def generate_extraction(clinical_text, retry=False):

    if not retry:

        prompt = f"""You are a clinical information extraction system.

Extract ONLY medical information explicitly stated in the
clinical text.

Extract exactly four categories:

1. conditions
   - diseases
   - diagnoses
   - medical conditions

2. drugs
   - medication names

3. dosages
   - dose
   - strength
   - frequency
   - administration instructions

4. procedures
   - medical procedures
   - operations
   - investigations
   - interventions

Return ONLY ONE valid JSON object.

The JSON MUST have exactly these four fields:

{{
  "conditions": [],
  "drugs": [],
  "dosages": [],
  "procedures": []
}}

IMPORTANT RULES:

- Every value MUST be a list.
- Every item inside every list MUST be a STRING.
- NEVER put dictionaries inside lists.
- NEVER put numbers inside lists.
- NEVER invent information.
- Extract only explicitly stated information.
- If a category is absent, return an empty list.
- Do not provide explanations.
- Do not use Markdown.
- Return ONLY JSON.

Example:

Clinical text:
Patient was prescribed Metformin 500 mg twice daily
for type 2 diabetes.

JSON:
{{
  "conditions": ["type 2 diabetes"],
  "drugs": ["Metformin"],
  "dosages": ["500 mg twice daily"],
  "procedures": []
}}

Now process this clinical text:

Clinical text:
{clinical_text}

JSON:
"""

    else:

        prompt = f"""Extract medical information from the
clinical text below.

Return ONLY this JSON structure:

{{
  "conditions": [],
  "drugs": [],
  "dosages": [],
  "procedures": []
}}

Rules:

- Use exactly these four fields.
- Every field must contain a list.
- Every list item must be a string.
- Do not use dictionaries.
- Do not add explanations.
- Do not use Markdown.
- If nothing is present, use [].
- Extract only information explicitly present.

Clinical text:
{clinical_text}

Return the complete JSON now:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1800
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=False
    )

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return response


def process_response(response):

    try:

        json_text = extract_json_object(response)

        parsed_json = json.loads(json_text)

        normalized_json, repairs = normalize_output(
            parsed_json
        )

        validated = ClinicalExtraction.model_validate(
            normalized_json
        )

        return {
            "status": "success",
            "data": validated.model_dump(),
            "repair_applied": len(repairs) > 0,
            "repairs": repairs,
            "raw_output": response
        }

    except json.JSONDecodeError as e:

        return {
            "status": "parsing_failed",
            "error": f"Invalid JSON: {str(e)}",
            "repair_applied": False,
            "repairs": [],
            "raw_output": response
        }

    except Exception as e:

        return {
            "status": "validation_failed",
            "error": str(e),
            "repair_applied": False,
            "repairs": [],
            "raw_output": response
        }


def extract_clinical_information(clinical_text):

    # First attempt
    response = generate_extraction(
        clinical_text,
        retry=False
    )

    result = process_response(response)

    if result["status"] == "success":

        result["retry_used"] = False

        return result

    # Controlled retry
    print("First extraction failed. Retrying...")

    retry_response = generate_extraction(
        clinical_text,
        retry=True
    )

    retry_result = process_response(
        retry_response
    )

    retry_result["retry_used"] = True

    return retry_result


if __name__ == "__main__":

    test_text = """
An 8-year-old boy with chronic encephalopathy secondary to hypoxic ischemic
syndrome, with cerebral palsy and symptomatic epilepsy, was admitted to the
emergency department of the children's hospital. He had been seizure-free for
the past year with an enteric-coated delayed release formulation of VPA
(375 mg every 8 hours). Thirty days prior to hospital admission, he was started
on LTG 25 mg/day along with VPA, since his seizures were no longer under
control with VPA. Two weeks later, the dose was changed.
"""
    result = extract_clinical_information(
        test_text
    )

    print(
        "\n========== EXTRACTION RESULT =========="
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )