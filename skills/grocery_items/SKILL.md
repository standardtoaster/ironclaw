---
name: grocery_items
version: 0.1.0
description: "Track items purchased and quantities for grocery shopping"
activation:
  keywords:
    - grocery
    - items
    - grocery items
    - track
    - purchased
    - quantities
    - shopping
  max_context_tokens: 800
  tools_prefix: grocery_items
---

# Grocery Items

Track items purchased and quantities for grocery shopping

## Tools

Call these tools to manage records. ALWAYS call the tool — never just acknowledge in text.

- **grocery_items_add** — Add a record. Fields:
  - `item_name` (text)
  - `quantity` (number)
- **grocery_items_query** — Search and filter records (eq, neq, gt, lt, gte, lte, between, in, contains).
- **grocery_items_summary** — Aggregations: sum, count, avg, min, max. Use group_by for breakdowns.
- **grocery_items_update** — Update a record by ID (partial update).
- **grocery_items_delete** — Delete a record by ID.

## Adding records

When the user mentions items to add, ALWAYS call grocery_items_add immediately:
- "Add X" → one grocery_items_add call
- "Add X and Y" → two grocery_items_add calls (one per item)
- "I also need X" / "and Y too" → one grocery_items_add call per item
- Do NOT just respond in text. Call the tool.

## Querying

- "What's on my list?" / "Show me everything" → grocery_items_query with no filters
- "Show me just [value]" → grocery_items_query with filter
- "How many?" / "Total?" / "Summary by [field]?" → grocery_items_summary
