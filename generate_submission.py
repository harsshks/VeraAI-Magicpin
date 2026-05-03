"""
Generate submission.jsonl — one composed message per test pair.

Usage:
    export GROQ_API_KEY=your_key_here
    python generate_submission.py

Reads: dataset/expanded/test_pairs.json + all context JSONs
Writes: submission.jsonl (30 lines)
"""

import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from composer import compose

DATASET_DIR = Path(__file__).parent / "dataset" / "expanded"


def load_json(path: Path) -> dict:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_contexts(dataset_dir: Path):
    """Load all expanded dataset contexts into dicts."""
    categories = {}
    merchants = {}
    customers = {}
    triggers = {}

    # Categories
    cat_dir = dataset_dir / "categories"
    if cat_dir.exists():
        for f in cat_dir.glob("*.json"):
            data = load_json(f)
            categories[data.get("slug", f.stem)] = data

    # Merchants
    merch_dir = dataset_dir / "merchants"
    if merch_dir.exists():
        for f in merch_dir.glob("*.json"):
            data = load_json(f)
            merchants[data.get("merchant_id", f.stem)] = data

    # Customers
    cust_dir = dataset_dir / "customers"
    if cust_dir.exists():
        for f in cust_dir.glob("*.json"):
            data = load_json(f)
            customers[data.get("customer_id", f.stem)] = data

    # Triggers
    trg_dir = dataset_dir / "triggers"
    if trg_dir.exists():
        for f in trg_dir.glob("*.json"):
            data = load_json(f)
            triggers[data.get("id", f.stem)] = data

    return categories, merchants, customers, triggers


def main():
    print("=" * 60)
    print("  Vera AI — Submission Generator")
    print("=" * 60)

    # Check API key
    if not os.environ.get("GROQ_API_KEY"):
        print("\n[ERROR] GROQ_API_KEY environment variable not set!")
        print("  Run: set GROQ_API_KEY=your_key_here")
        sys.exit(1)

    # Load test pairs
    test_pairs_path = DATASET_DIR / "test_pairs.json"
    if not test_pairs_path.exists():
        print(f"\n[ERROR] Test pairs not found at {test_pairs_path}")
        print("  Run: cd dataset && python generate_dataset.py --seed-dir . --out ./expanded")
        sys.exit(1)

    test_pairs = load_json(test_pairs_path)["pairs"]
    print(f"\n  Loaded {len(test_pairs)} test pairs")

    # Load all contexts
    categories, merchants, customers, triggers = load_all_contexts(DATASET_DIR)
    print(f"  Loaded {len(categories)} categories, {len(merchants)} merchants, "
          f"{len(customers)} customers, {len(triggers)} triggers")

    # Generate submissions
    output_path = Path(__file__).parent / "submission.jsonl"
    results = []
    errors = []

    for i, pair in enumerate(test_pairs):
        test_id = pair["test_id"]
        trigger_id = pair["trigger_id"]
        merchant_id = pair["merchant_id"]
        customer_id = pair.get("customer_id")

        print(f"\n  [{i+1}/{len(test_pairs)}] {test_id}: {trigger_id}")

        # Look up contexts
        trigger = triggers.get(trigger_id)
        if not trigger:
            print(f"    [WARN] Trigger not found: {trigger_id}")
            errors.append(test_id)
            continue

        merchant = merchants.get(merchant_id)
        if not merchant:
            print(f"    [WARN] Merchant not found: {merchant_id}")
            errors.append(test_id)
            continue

        cat_slug = merchant.get("category_slug", "")
        category = categories.get(cat_slug)
        if not category:
            print(f"    [WARN] Category not found: {cat_slug}")
            errors.append(test_id)
            continue

        customer = customers.get(customer_id) if customer_id else None

        # Compose
        try:
            result = compose(category, merchant, trigger, customer)
            result["test_id"] = test_id

            print(f"    Body: {result.get('body', '')[:80]}...")
            print(f"    CTA: {result.get('cta', '?')} | Send as: {result.get('send_as', '?')}")

            results.append(result)
        except Exception as e:
            print(f"    [ERROR] Composition failed: {e}")
            errors.append(test_id)
            # Add fallback entry
            results.append({
                "test_id": test_id,
                "body": f"Hi, Vera here. Quick check-in on your {trigger.get('kind', 'business')} update.",
                "cta": "open_ended",
                "send_as": "vera",
                "suppression_key": trigger.get("suppression_key", ""),
                "rationale": f"Fallback due to composition error: {str(e)[:50]}",
            })

        # Rate limit: 30 requests per minute for Groq free tier
        time.sleep(2.5)

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 60}")
    print(f"  Done! Wrote {len(results)} entries to {output_path}")
    if errors:
        print(f"  Errors: {len(errors)} — {errors}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
