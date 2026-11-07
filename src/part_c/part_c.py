"""
Part C - Support Ticket Triage with Guardrails
"""

import sys
import re
import json
import time
import random
from pathlib import Path

import pandas as pd

# Locate project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mock_llm


# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05

TOKEN_BUDGET_DEMO = 6000
TOKEN_BUDGET_FINAL = 100000

CATEGORIES = {"billing", "delivery", "technical", "fraud"}

RAW_FILE = ROOT / "data" / "raw" / "tickets.csv"
SEGMENT_FILE = ROOT / "outputs" / "customer_segments.csv"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. PROMPTS
# --------------------------------------------------

ZERO_SHOT_PROMPT = """
CLASSIFY the customer support ticket.

Allowed categories:
billing, delivery, technical, fraud.

Return JSON only in this exact format:
{"category": "...", "reason": "..."}

Do not include explanations outside the JSON.
"""

FEW_SHOT_PROMPT = """
CLASSIFY the customer support ticket.

Allowed categories:
billing, delivery, technical, fraud.

Return JSON only in this exact format:
{"category": "...", "reason": "..."}

EXAMPLES:
Example 1:
Ticket: I was charged twice for my order.
Answer: {"category": "billing", "reason": "Duplicate charge"}

Example 2:
Ticket: I received a suspicious message asking for my account details.
Answer: {"category": "fraud", "reason": "Suspicious activity"}

Example 3:
Ticket: My app keeps crashing when I open it.
Answer: {"category": "technical", "reason": "Application problem"}
"""


# --------------------------------------------------
# 3. PERSONAL INFORMATION REDACTION
# --------------------------------------------------

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_PATTERN = re.compile(r"\b[6-9]\d{9}\b")
ORDER_PATTERN = re.compile(r"ORD-\d{6}")

def redact_text(text):

    text = str(text)

    text, email_count = EMAIL_PATTERN.subn("[EMAIL]", text)
    text, phone_count = PHONE_PATTERN.subn("[PHONE]", text)
    text, order_count = ORDER_PATTERN.subn("[ORDER_ID]", text)

    counts = {
        "emails": email_count,
        "phones": phone_count,
        "order_ids": order_count
    }

    # Confirm no personal information remains
    assert not EMAIL_PATTERN.search(text)
    assert not PHONE_PATTERN.search(text)
    assert not ORDER_PATTERN.search(text)

    return text, counts


# --------------------------------------------------
# 4. OUTPUT VALIDATION
# --------------------------------------------------

def validate_answer(answer):

    try:
        result = json.loads(answer)

        category = result.get("category", "").lower()

        if category in CATEGORIES:
            return category, "classified_json"

        return "", "flagged_invalid_output"

    except (json.JSONDecodeError, AttributeError, TypeError):

        words = re.findall(r"[a-z]+", str(answer).lower())

        found = [word for word in words if word in CATEGORIES]

        if len(found) == 1:
            return found[0], "classified_keyword"

        return "", "flagged_invalid_output"


# --------------------------------------------------
# 5. MODEL CALL WITH RETRIES
# --------------------------------------------------

def call_model(ticket_id, system_prompt, redacted_text):

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": redacted_text}
    ]

    # Privacy assertion: only redacted ticket text is sent
    assert messages[1]["content"] == redacted_text
    assert not EMAIL_PATTERN.search(redacted_text)
    assert not PHONE_PATTERN.search(redacted_text)
    assert not ORDER_PATTERN.search(redacted_text)

    for retry_number in range(4):

        try:
            response = mock_llm.chat(messages)

            message = response.choices[0].message
            answer = message.content

            tokens = response.usage.total_tokens

            return answer, tokens

        except mock_llm.TransientLLMError:

            if retry_number == 3:
                raise

            wait = (
                BASE_DELAY
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"Retry: ticket={ticket_id}, "
                f"retry_number={retry_number}, "
                f"wait={wait:.4f}s"
            )

            time.sleep(wait)


# --------------------------------------------------
# 6. LOAD CUSTOMER SEGMENTS
# --------------------------------------------------

def load_segments():

    segments = pd.read_csv(SEGMENT_FILE)

    # Use the Premium flag if Part B already created it.
    if "is_premium" in segments.columns:
        segments["is_premium"] = (
            segments["is_premium"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )

    # Otherwise identify the highest-value segment.
    else:
        if "segment_name" in segments.columns:
            premium_name = "High-value active"

            segments["is_premium"] = (
                segments["segment_name"] == premium_name
            )
        else:
            segments["is_premium"] = False

    if "segment_name" not in segments.columns:
        segments["segment_name"] = ""

    return segments[
        ["customer_id", "segment_name", "is_premium"]
    ].drop_duplicates("customer_id")


# --------------------------------------------------
# 7. ROUTING
# --------------------------------------------------

def assign_route(status, category, is_premium):

    if status in {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted"
    }:
        return "human_review", "NORMAL"

    if category == "fraud":
        return "security_team", "HIGH"

    if is_premium:
        return "premium_desk", "HIGH"

    return "support_queue", "NORMAL"


# --------------------------------------------------
# 8. MAIN PIPELINE
# --------------------------------------------------

def run_pipeline(system_prompt, token_budget=None):

    tickets = pd.read_csv(
        RAW_FILE,
        keep_default_na=False
    )

    segments = load_segments()

    tickets = tickets.merge(
        segments,
        on="customer_id",
        how="left"
    )

    tickets["segment_name"] = tickets["segment_name"].fillna("")
    tickets["is_premium"] = tickets["is_premium"].fillna(False)

    mock_llm.reset()

    results = []

    total_tokens = 0
    total_emails = 0
    total_phones = 0
    total_orders = 0

    for _, ticket in tickets.iterrows():

        ticket_id = ticket["ticket_id"]
        original_text = str(ticket["text"])

        category = ""
        status = ""
        tokens_used = 0

        # Redact all tickets, including those later skipped.
        redacted_text, counts = redact_text(original_text)

        total_emails += counts["emails"]
        total_phones += counts["phones"]
        total_orders += counts["order_ids"]

        # Empty ticket guardrail
        if not original_text.strip():

            status = "skipped_empty"

        # Long ticket guardrail
        elif mock_llm.est_tokens(original_text) > MAX_INPUT_TOKENS:

            status = "flagged_long"

        # Token budget guardrail
        elif token_budget is not None and total_tokens >= token_budget:

            status = "budget_exhausted"

        else:

            try:

                answer, tokens_used = call_model(
                    ticket_id,
                    system_prompt,
                    redacted_text
                )

                total_tokens += tokens_used

                category, status = validate_answer(answer)

            except mock_llm.TransientLLMError:

                # This should not occur with the supplied mock.
                status = "flagged_invalid_output"
                category = ""

        is_premium = bool(ticket["is_premium"])

        route, priority = assign_route(
            status,
            category,
            is_premium
        )

        results.append({

            "ticket_id": ticket_id,
            "customer_id": ticket["customer_id"],
            "status": status,
            "category": category,
            "segment_name": ticket["segment_name"],
            "priority": priority,
            "route": route,
            "tokens_used": tokens_used

        })

    result_df = pd.DataFrame(results)

    print("\n--------------------------------")
    print("PIPELINE RESULTS")
    print("--------------------------------")

    print("\nStatus counts:")
    print(result_df["status"].value_counts())

    print("\nTotal tokens:", total_tokens)

    print("\nRedaction counts:")
    print("Emails:", total_emails)
    print("Phones:", total_phones)
    print("Order IDs:", total_orders)
    print("Total:", total_emails + total_phones + total_orders)

    return result_df, total_tokens


# --------------------------------------------------
# 9. ACCURACY AND CONFUSION TABLE
# --------------------------------------------------

def evaluate_results(result_df):

    original = pd.read_csv(
        RAW_FILE,
        keep_default_na=False
    )

    evaluated = result_df.merge(
        original[["ticket_id", "true_category"]],
        on="ticket_id",
        how="left"
    )

    classified = evaluated[
        evaluated["status"].isin([
            "classified_json",
            "classified_keyword"
        ])
    ]

    if len(classified) > 0:

        accuracy = (
            classified["category"] == classified["true_category"]
        ).mean() * 100

    else:
        accuracy = 0

    print("\nClassified tickets:", len(classified))
    print(f"Accuracy: {accuracy:.2f}%")

    confusion = pd.crosstab(
        classified["true_category"],
        classified["category"],
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False
    )

    print("\nConfusion table:")
    print(confusion)

    return len(classified), accuracy, confusion


# --------------------------------------------------
# 10. RUN ALL PART C EXPERIMENTS
# --------------------------------------------------

def main():

    print("\n========== ZERO-SHOT RUN ==========")

    zero_results, zero_tokens = run_pipeline(
        ZERO_SHOT_PROMPT
    )

    zero_results.to_csv(
        OUTPUT_DIR / "triage_zero_shot.csv",
        index=False
    )

    zero_count, zero_accuracy, confusion = evaluate_results(
        zero_results
    )

    confusion.to_csv(
        OUTPUT_DIR / "triage_confusion_matrix.csv"
    )

    print("\n========== FEW-SHOT RUN ==========")

    few_results, few_tokens = run_pipeline(
        FEW_SHOT_PROMPT
    )

    few_results.to_csv(
        OUTPUT_DIR / "triage_few_shot.csv",
        index=False
    )

    few_count, few_accuracy, _ = evaluate_results(
        few_results
    )

    print("\n========== TOKEN BUDGET DEMO ==========")

    budget_results, budget_tokens = run_pipeline(
        ZERO_SHOT_PROMPT,
        token_budget=TOKEN_BUDGET_DEMO
    )

    budget_results.to_csv(
        OUTPUT_DIR / "triage_budget_demo.csv",
        index=False
    )

    exhausted = (
        budget_results["status"] == "budget_exhausted"
    ).sum()

    print("Budget exhausted tickets:", exhausted)

    print("\n========== FINAL FEW-SHOT RUN ==========")

    final_results, final_tokens = run_pipeline(
        FEW_SHOT_PROMPT,
        token_budget=TOKEN_BUDGET_FINAL
    )

    final_results.to_csv(
        OUTPUT_DIR / "triage_results.csv",
        index=False
    )

    print("\n========== COMPARISON ==========")

    comparison = pd.DataFrame([

        {
            "prompt": "Zero-shot",
            "classified_tickets": zero_count,
            "accuracy_percent": round(zero_accuracy, 2),
            "total_tokens": zero_tokens,
            "tokens_per_ticket": round(
                zero_tokens / zero_count, 2
            ) if zero_count else 0
        },

        {
            "prompt": "Few-shot",
            "classified_tickets": few_count,
            "accuracy_percent": round(few_accuracy, 2),
            "total_tokens": few_tokens,
            "tokens_per_ticket": round(
                few_tokens / few_count, 2
            ) if few_count else 0
        }

    ])

    print(comparison)

    comparison.to_csv(
        OUTPUT_DIR / "triage_prompt_comparison.csv",
        index=False
    )

    print("\nPART C PIPELINE COMPLETED.")


if __name__ == "__main__":
    main()