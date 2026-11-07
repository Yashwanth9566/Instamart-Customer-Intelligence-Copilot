# ==========================================
# PART D: POLICY RAG
# ==========================================

import sys
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
POLICY_DIR = ROOT / "data" / "policies"
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"

sys.path.insert(0, str(ROOT))
import mock_llm


def load_and_chunk_policies(overlap=10):
    chunks = []

    for file_path in sorted(POLICY_DIR.glob("*.txt")):
        words = file_path.read_text(encoding="utf-8").split()

        step = 40 - overlap

        for start in range(0, len(words), step):
            chunk_words = words[start:start + 40]

            if not chunk_words:
                break

            chunks.append({
                "source": file_path.name,
                "text": " ".join(chunk_words)
            })

            if start + 40 >= len(words):
                break

    return chunks


class PolicyRetriever:

    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()

        self.matrix = self.vectorizer.fit_transform(
            [chunk["text"] for chunk in chunks]
        )

    def retrieve(self, query, k=3):
        query_vector = self.vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector, self.matrix
        ).flatten()

        ranked = scores.argsort()[::-1][:k]

        return [
            {
                "source": self.chunks[i]["source"],
                "score": float(scores[i]),
                "text": self.chunks[i]["text"]
            }
            for i in ranked
        ]# ==========================================
# PART D: REFUND AGENT - TOOL DEFINITIONS
# ==========================================

import json
import sqlite3
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DB_PATH = ROOT / "data" / "instamart.db"

LEDGER_PATH = ROOT / "outputs" / "refund_ledger.jsonl"

STATE_DIR = ROOT / "state"

ACTIVE_RETRIEVER = None


# TOOL 1: GET ORDER
def get_order(order_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    row = conn.execute(
        """
        SELECT order_id, status, order_amount, delivery_minutes
        FROM orders
        WHERE order_id = ?
        """,
        (order_id,)
    ).fetchone()

    conn.close()

    if row is None:
        return {"error": "order not found"}

    return dict(row)


# TOOL 2: SEARCH POLICY
def search_policy(query):
    if ACTIVE_RETRIEVER is None:
        return {"error": "Policy retriever not initialized"}

    results = ACTIVE_RETRIEVER.retrieve(query, k=3)

    return {"results": results}


# TOOL 3: CALCULATE REFUND
def calculate_refund(order_amount, delivery_minutes):
    if float(delivery_minutes) > 45:
        refund = float(order_amount)
    else:
        refund = 0.0

    return {"refund_inr": round(refund, 2)}


# HELPER: SAVE TRANSACTION
def save_to_ledger(tool_name, order_id, refund_inr):
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now().isoformat(),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2)
    }

    with open(LEDGER_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


# TOOL 4: ISSUE REFUND
def issue_refund(order_id, refund_inr):
    save_to_ledger("issue_refund", order_id, refund_inr)

    return {"refund_inr": round(float(refund_inr), 2)}


# TOOL 5: ESCALATE TO HUMAN
def escalate_to_human(order_id, refund_inr):
    save_to_ledger("escalate_to_human", order_id, refund_inr)

    return {"refund_inr": round(float(refund_inr), 2)}


# TOOL MAP
TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}


# TOOL SPECIFICATIONS
TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Retrieve order status, amount and delivery time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Search company policies for relevant refund rules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculate refund based on order amount and delivery time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {"type": "number"},
                    "delivery_minutes": {"type": "number"}
                },
                "required": ["order_amount", "delivery_minutes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Record an approved refund in the refund ledger.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Record a refund requiring human approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"}
                },
                "required": ["order_id", "refund_inr"]
            }
        }
    }
]
# CONNECT POLICY RETRIEVER
chunks = load_and_chunk_policies(overlap=10)
ACTIVE_RETRIEVER = PolicyRetriever(chunks)

print("\nPolicy retriever connected successfully!")
print("Total policy chunks:", len(chunks))

# ==========================================
# REFUND AGENT TEST
# ==========================================

# ==========================================
# CHECKPOINT AND RESUME FUNCTIONS
# ==========================================

import time


def save_checkpoint(state):

    STATE_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = STATE_DIR / f"{state['run_id']}.json"

    with open(checkpoint_path, "w") as file:
        json.dump(state, file, indent=4)

    return checkpoint_path


def load_checkpoint(run_id):

    checkpoint_path = STATE_DIR / f"{run_id}.json"

    with open(checkpoint_path, "r") as file:
        return json.load(file)


# ==========================================
# AGENT EXECUTION ENGINE
# ==========================================

def continue_agent(
    state,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    stop_after_tools=None
):

    start_time = time.monotonic()

    while state["llm_calls"] < max_iterations:

        if time.monotonic() - start_time >= time_limit_seconds:

            state["status"] = "FALLBACK_TIMEOUT"
            state["final_answer"] = (
                "FALLBACK_TIMEOUT: Request handed to a human agent."
            )

            save_checkpoint(state)
            return state

        response = mock_llm.chat(
            state["messages"],
            tools=TOOLS_SPEC
        )

        state["llm_calls"] += 1
        state["tokens_used"] += response.usage.total_tokens

        if state["tokens_used"] > token_limit:

            state["status"] = "FALLBACK_TOKEN_LIMIT"
            state["final_answer"] = (
                "FALLBACK_TOKEN_LIMIT: Request handed to a human agent."
            )

            save_checkpoint(state)
            return state

        message = response.choices[0].message
        tool_calls = message.tool_calls

        if not tool_calls:

            final_answer = message.content or ""

            state["final_answer"] = final_answer
            state["status"] = final_answer.split(":")[0].strip()

            save_checkpoint(state)
            return state

        state["messages"].append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments
                }
                for tc in tool_calls
            ]
        })

        for tc in tool_calls:

            arguments = tc.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            tool_function = TOOL_MAP[tc.name]

            observation = tool_function(**arguments)

            step = {
                "tool_name": tc.name,
                "arguments": arguments,
                "observation": observation
            }

            state["steps"].append(step)

            state["tools_executed"].append(tc.name)

            state["messages"].append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": json.dumps(observation)
            })

            state["step_number"] = len(state["steps"])

            # Save after EVERY tool observation
            save_checkpoint(state)

            # Simulate a crash after a specified number of tools
            if (
                stop_after_tools is not None
                and len(state["steps"]) >= stop_after_tools
            ):

                state["status"] = "PAUSED"
                state["final_answer"] = (
                    "PAUSED: Execution stopped for resume testing."
                )

                return state

    state["status"] = "FALLBACK_MAX_ITERATIONS"

    state["final_answer"] = (
        "FALLBACK_MAX_ITERATIONS: Request handed to a human agent."
    )

    save_checkpoint(state)

    return state


# ==========================================
# START NEW AGENT RUN
# ==========================================

def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    stop_after_tools=None,
    run_id=None
):

    if run_id is None:
        run_id = f"{scenario_id}_{time.time_ns()}"

    state = {
        "run_id": run_id,
        "scenario_id": scenario_id,

        "messages": [
            {
                "role": "system",
                "content": """
                AGENT

                You are an Instamart refund support agent.
                Follow the refund policy and use the supplied tools.
                Only completed orders are eligible.
                Delivery above 45 minutes qualifies for a full refund.
                Refunds above INR 1000 must be escalated to a human.
                Never issue a refund for an ineligible order.
                """
            },
            {
                "role": "user",
                "content": user_message
            }
        ],

        "tokens_used": 0,
        "llm_calls": 0,
        "step_number": 0,
        "steps": [],
        "tools_executed": [],
        "status": "RUNNING",
        "final_answer": ""
    }

    return continue_agent(
        state,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        stop_after_tools=stop_after_tools
    )


# ==========================================
# RESUME EXISTING AGENT RUN
# ==========================================

def resume_agent(
    run_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10
):

    state = load_checkpoint(run_id)

    return continue_agent(
        state,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds
    )
# ==========================================
# CHECKPOINT AND RESUME TEST
# ==========================================

# ==========================================
# SAFETY STOP TESTS
# ==========================================

def save_agent_run(result, scenario_id):
    log_path = OUTPUT_DIR / "agent_runs.jsonl"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "scenario_id": scenario_id,
        "status": result["status"],
        "final_answer": result["final_answer"],
        "llm_calls": result["llm_calls"],
        "tokens_used": result["tokens_used"],
        "tools_executed": result["tools_executed"],
        "steps": result["steps"]
    }

    with open(log_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, default=str) + "\n")

    print(f"Logged scenario {scenario_id}")

if __name__ == "__main__":

    import pandas as pd

    scenarios = pd.read_csv(RAW_DIR / "agent_scenarios.csv")

    scenario = scenarios[
        scenarios["scenario_id"] == "S1"
    ].iloc[0]

    # TEST 1: TOKEN LIMIT

    print("\n========== TOKEN LIMIT TEST ==========")

    mock_llm.reset()

    token_result = run_agent(
        user_message=scenario["message"],
        scenario_id="S1_TOKEN",
        token_limit=200
    )

    print("Status:", token_result["status"])
    print("Final Answer:", token_result["final_answer"])
    print("Tokens Used:", token_result["tokens_used"])

    # TEST 2: TIMEOUT

    print("\n========== TIMEOUT TEST ==========")

    mock_llm.reset()

    timeout_result = run_agent(
        user_message=scenario["message"],
        scenario_id="S1_TIMEOUT",
        time_limit_seconds=0
    )

    print("Status:", timeout_result["status"])
    print("Final Answer:", timeout_result["final_answer"])
    print("LLM Calls:", timeout_result["llm_calls"])

    # TEST 3: MAXIMUM ITERATIONS

    print("\n========== MAX ITERATIONS TEST ==========")

    mock_llm.reset()

    loop_result = run_agent(
        user_message="[LOOP] Refund order ORD-100048.",
        scenario_id="S7",
        max_iterations=6
    )

    print("Status:", loop_result["status"])
    print("Final Answer:", loop_result["final_answer"])
    print("LLM Calls:", loop_result["llm_calls"])

        # ==========================================
    # LOG ALL SEVEN AGENT SCENARIOS
    # ==========================================

    print("\n========== LOGGING ALL SCENARIOS ==========")

    for _, row in scenarios.iterrows():
        mock_llm.reset()

        result = run_agent(
            user_message=row["message"],
            scenario_id=row["scenario_id"]
        )

        save_agent_run(result, row["scenario_id"])

        print(
            row["scenario_id"],
            "Expected:", row["expected_outcome"],
            "Actual:", result["status"]
        )

    print("\nALL SCENARIOS LOGGED SUCCESSFULLY.")