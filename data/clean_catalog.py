"""
clean_catalog.py

Takes the raw scraped SHL catalog (shl_product_catalog.json) and converts it
into a clean, predictable format that the rest of our system (retrieval +
LLM prompt) can consume without surprises.

Why we need this step (not just use the raw file directly):
1. The raw file has stray literal newlines inside some string values
   (e.g. "name": "Microsoft \n365 (New)") which breaks strict JSON parsing.
2. The raw "keys" field uses long category names ("Personality & Behavior")
   but our API response schema needs short codes ("P") like in the sample
   conversations (test_type: "P", "K" etc.) -- SHL's own catalog uses these
   single-letter codes: A, B, C, D, E, K, P, S.
3. We only want the fields we actually need downstream (no scraped_at,
   no *_raw duplicate fields) to keep the retrieval index small and clean.
4. We build one flat "searchable_text" field per entry up front -- this is
   what we'll embed / keyword-match against later, so retrieval doesn't have
   to re-assemble it on every request.
"""

import json
import re

RAW_PATH = "data/shl_product_catalog.json"
CLEAN_PATH = "data/catalog_clean.json"

# Mapping from SHL's long category names to their official short codes.
# (These codes match what SHL itself uses on the live catalog pages, and
# what the sample conversations expect in `test_type`.)
CATEGORY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


def fix_stray_newlines(value: str) -> str:
    """Collapse accidental literal newlines/extra spaces inside a string
    (e.g. 'Microsoft \\n365 (New)' -> 'Microsoft 365 (New)')."""
    if not isinstance(value, str):
        return value
    value = value.replace("\n", " ").replace("\r", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def clean_entry(raw_entry: dict) -> dict:
    name = fix_stray_newlines(raw_entry.get("name", ""))
    description = fix_stray_newlines(raw_entry.get("description", ""))
    url = raw_entry.get("link", "").strip()

    # Convert long category names -> short codes, keep only known ones
    long_keys = raw_entry.get("keys", [])
    test_type_codes = [CATEGORY_TO_CODE[k] for k in long_keys if k in CATEGORY_TO_CODE]

    cleaned = {
        "id": raw_entry.get("entity_id"),
        "name": name,
        "url": url,
        "description": description,
        "test_type": test_type_codes,          # e.g. ["P"] or ["K"]
        "duration": fix_stray_newlines(raw_entry.get("duration", "")),
        "languages": raw_entry.get("languages", []),
        "job_levels": raw_entry.get("job_levels", []),
        "remote_testing": raw_entry.get("remote", "no") == "yes",
        "adaptive": raw_entry.get("adaptive", "no") == "yes",
    }

    # Build one flat text blob used later for retrieval (embeddings / keyword match)
    cleaned["searchable_text"] = " | ".join(filter(None, [
        cleaned["name"],
        cleaned["description"],
        ", ".join(cleaned["job_levels"]),
        ", ".join(long_keys),
    ]))

    return cleaned


def main():
    with open(RAW_PATH) as f:
        raw_data = json.load(f, strict=False)  # strict=False tolerates the stray newlines

    cleaned_data = [clean_entry(e) for e in raw_data]

    # Sanity checks before saving
    assert len(cleaned_data) == len(raw_data), "Lost some entries during cleaning!"
    missing_url = [e["name"] for e in cleaned_data if not e["url"]]
    if missing_url:
        print(f"WARNING: {len(missing_url)} entries have no URL: {missing_url[:5]}")

    with open(CLEAN_PATH, "w") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    print(f"Cleaned {len(cleaned_data)} entries -> {CLEAN_PATH}")
    print("\nSample cleaned entry:")
    print(json.dumps(cleaned_data[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
