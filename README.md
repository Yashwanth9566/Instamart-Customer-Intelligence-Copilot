# Instamart Customer Intelligence & Support Copilot

## 1. Project Overview

The Instamart Customer Intelligence & Support Copilot is an offline, AI-powered customer intelligence and support automation project.

The project combines data analytics, machine learning, natural language processing, Retrieval-Augmented Generation (RAG), and AI agents to improve customer support operations.

The project consists of four major components:

- **Part A – Data Foundation:** Data generation, cleaning, validation, and SQLite database creation.
- **Part B – Customer Segmentation and Lapse Prediction:** Customer segmentation, cohort analysis, and machine learning models to predict customer lapse.
- **Part C – Support Ticket Triage:** Automated ticket classification, sensitive information redaction, and output validation.
- **Part D – Policy RAG Assistant and Refund Agent:** Policy retrieval, automated refund decisions, human escalation, and checkpoint-based agent execution.

The project uses synthetic Instamart data and a supplied offline mock LLM. No external API key or internet connection is required to execute the AI workflows.

---

## 2. Project Structure

```text
Instamart_Capstone/
│
├── data/
│   ├── raw/
│   ├── clean/
│   └── policies/
│
├── sql/
│
├── src/
│   ├── part_a/
│   ├── part_b/
│   ├── part_c/
│   └── part_d/
│
├── outputs/
│
├── state/
│
├── generate_data.py
├── mock_llm.py
└── README.md
```

### Important Outputs

- Cleaned datasets
- SQLite database
- Customer segments
- Cohort analysis
- Lapse prediction model
- Retention list
- Ticket classification results
- Policy retrieval evaluation
- Refund transaction ledger
- Agent execution logs
- Agent checkpoint files

---

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SQLite
- TF-IDF Vectorization
- Cosine Similarity
- Logistic Regression
- Random Forest
- Natural Language Processing
- Retrieval-Augmented Generation (RAG)
- ReAct Agent Workflow
- JSON and JSONL

---

## 4. Part A – Data Foundation

The first part establishes the data foundation for the project.

### Activities Completed

- Generated synthetic customer, product, order, and ticket datasets.
- Cleaned and validated the datasets.
- Stored cleaned datasets in CSV format.
- Created a SQLite database.
- Created and executed SQL queries.
- Validated database tables and columns.

### Database Tables

- Customers
- Products
- Orders

The SQLite database provides a structured foundation for customer analytics and support automation.

---

## 5. Part B – Customer Segmentation and Lapse Prediction

This component focuses on identifying customer groups and predicting customers who may become inactive.

### Activities Completed

- Customer segmentation.
- Cohort analysis.
- Customer lapse prediction.
- Model comparison.
- Retention list generation.

### Machine Learning Models

Two classification algorithms were evaluated:

1. Logistic Regression
2. Random Forest

### Model Results

| Model | Test F1 Score |
|---|---:|
| Logistic Regression | 0.719 |
| Random Forest | 0.706 |

**Result:** Logistic Regression achieved the higher test F1 score of 0.719.

Both models exceeded the required minimum F1 score of 0.55.

### Key Outputs

- `customer_segments.csv`
- `cohort_analysis.csv`
- `lapse_model.joblib`
- `retention_list.csv`
- `lapse_model_comparison.csv`

---

## 6. Part C – Support Ticket Triage with Guardrails

This component automates customer support ticket classification while maintaining output validation and privacy safeguards.

### Activities Completed

- Ticket classification using zero-shot prompting.
- Ticket classification using few-shot prompting.
- Sensitive information redaction.
- Invalid output detection.
- Empty ticket handling.
- Long ticket detection.
- Classification accuracy evaluation.
- Token usage comparison.

### Classification Results

| Metric | Zero-Shot | Few-Shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Accuracy | 85.71% | 100.00% |
| Total tokens | 10,336 | 19,696 |
| Average tokens per ticket | 70.31 | 133.99 |

### Guardrail Results

| Status | Count |
|---|---:|
| classified_json | 135 |
| classified_keyword | 12 |
| flagged_invalid_output | 9 |
| skipped_empty | 4 |
| flagged_long | 4 |

### Sensitive Information Redaction

| Information Type | Redactions |
|---|---:|
| Emails | 55 |
| Phone numbers | 91 |
| Order IDs | 132 |
| **Total** | **278** |

**Result:** Few-shot classification achieved 100% accuracy on the evaluated synthetic dataset, while the guardrails identified invalid, empty, and excessively long inputs.

---

## 7. Part D – Policy RAG Assistant and Refund Agent

This component combines policy retrieval with an automated refund decision-making agent.

### 7.1 Policy RAG

The system retrieves relevant information from five policy documents:

- Refund Policy
- Delivery Policy
- Escalation Policy
- Payment Policy
- Privacy Policy

### RAG Configuration

| Parameter | Value |
|---|---|
| Chunk size | 40 words |
| Chunk overlap | 10 words |
| Total chunks | 15 |
| Retrieval method | TF-IDF |
| Similarity method | Cosine Similarity |
| Top results retrieved | 3 |

### RAG Evaluation

| Evaluation Metric | Result |
|---|---:|
| Correct document retrieval | 12/12 |
| Expected facts retrieved | 12/12 |
| Fallback checks | 2/2 |

### Chunk Overlap Comparison

| Configuration | Total Chunks | Correct Documents | Expected Facts |
|---|---:|---:|---:|
| With 10-word overlap | 15 | 12/12 | 12/12 |
| Without overlap | 13 | 12/12 | 10/12 |

The overlap configuration performed better because overlapping text helps preserve information that might otherwise be split across chunk boundaries.

### 7.2 Refund Agent

The refund agent uses a tool-based workflow to make decisions based on order information and retrieved policies.

The agent uses the following tools:

1. `get_order` – Retrieves order information.
2. `search_policy` – Retrieves relevant policy information.
3. `calculate_refund` – Calculates the eligible refund amount.
4. `issue_refund` – Records an automatically approved refund.
5. `escalate_to_human` – Records cases requiring human approval.

### Refund Decision Rules

- Only completed orders are eligible for refunds.
- Orders delivered in more than 45 minutes qualify for a full late-delivery refund.
- Orders delivered within 45 minutes do not qualify for a late-delivery refund.
- Refunds up to INR 1,000 are approved automatically.
- Refunds above INR 1,000 are escalated to a human support lead.
- Missing orders return a NOT_FOUND response.

### 7.3 Agent Scenario Evaluation

| Scenario | Expected Outcome | Actual Outcome |
|---|---|---|
| S1 | REFUND | REFUND |
| S2 | REFUND | REFUND |
| S3 | ESCALATE | ESCALATE |
| S4 | DENIED | DENIED |
| S5 | DENIED | DENIED |
| S6 | NOT_FOUND | NOT_FOUND |
| S7 | FALLBACK | FALLBACK_MAX_ITERATIONS |

All seven scenarios produced the intended outcomes.

S7 correctly triggered a fallback after reaching the maximum iteration limit.

### 7.4 Agent Execution Results

| Scenario | LLM Calls | Tokens Used |
|---|---:|---:|
| S1 | 5 | 1,152 |
| S2 | 5 | 1,152 |
| S3 | 5 | 1,139 |
| S4 | 4 | 809 |
| S5 | 3 | 490 |
| S6 | 2 | 174 |
| S7 | 6 | 3,306 |
| **Total** | **30** | **8,222** |

### 7.5 Safety and Reliability Tests

The agent was tested with three safety limits.

| Test | Expected Result | Actual Result |
|---|---|---|
| Token limit | FALLBACK_TOKEN_LIMIT | Passed |
| Time limit | FALLBACK_TIMEOUT | Passed |
| Maximum iterations | FALLBACK_MAX_ITERATIONS | Passed |

These tests demonstrate that the agent can stop execution safely when resource limits are reached.

### 7.6 Checkpoint and Resume

The agent supports checkpoint-based execution.

During testing:

- The agent executed `get_order` and `search_policy`.
- Its execution state was saved.
- Execution was paused to simulate a crash.
- The saved checkpoint was loaded.
- The agent resumed from its previous state.
- It completed the refund workflow successfully.

Final result:

`REFUND: INR 128.25 issued.`

This demonstrates that the agent can resume execution without repeating previously completed tool operations.

### 7.7 Agent Logging

Agent execution details are stored in:

`outputs/agent_runs.jsonl`

The log contains:

- Scenario ID
- Final status
- Final answer
- LLM calls
- Token usage
- Executed tools
- Tool arguments
- Tool observations

Refund and escalation transactions are recorded separately in:

`outputs/refund_ledger.jsonl`

---

## 8. ReAct vs Plan-and-Execute

ReAct and Plan-and-Execute are two approaches to building AI agents.

**ReAct** combines reasoning and tool execution step by step. The agent observes the result of each tool before deciding its next action.

**Plan-and-Execute** creates a complete plan first and then executes the planned steps.

For this project, ReAct is suitable because refund decisions depend on live order information, policy retrieval results, and calculated refund amounts. The next action may change depending on the observations returned by the tools.

For example, a completed order may qualify for an automatic refund, while a refund above INR 1,000 requires human escalation. A cancelled order or missing order requires a different response.

The ReAct workflow supports these conditional decisions and allows the agent to respond dynamically to tool observations.

The seven scenario tests, checkpoint-resume demonstration, and safety-stop tests show how this workflow operates in the project.

---

## 9. How to Reproduce the Project

### Requirements

- Python 3.14.10 or compatible Python version
- Visual Studio Code
- Required Python libraries
- Windows or another Python-supported operating system

### Installation

Open the project folder in VS Code and use the integrated terminal.

Install the required packages if they are not already installed:

```bash
pip install pandas numpy scikit-learn joblib
```

### Execution

Generate the synthetic datasets:

```bash
python generate_data.py
```

Run the individual Part A, Part B, Part C, and Part D Python scripts from their respective folders.

Part C:

```bash
python src/part_c/part_c.py
```

Part D:

```bash
python src/part_d/part_d.py
```

The project uses the supplied `mock_llm.py` file as an offline stand-in for a hosted language model. No API key is required.

---

## 10. Limitations and Future Improvements

### Limitations

1. The project uses synthetic datasets rather than actual Instamart customer records.
2. The supplied mock LLM is a deterministic offline simulation and does not represent the full capabilities of a hosted language model.
3. TF-IDF retrieval primarily relies on lexical similarity and may not identify all semantically related queries.
4. The refund ledger records simulated transactions rather than actual financial transactions.
5. The machine learning models require validation on larger real-world datasets before production use.

### Future Improvements

1. Integrate larger and more diverse datasets.
2. Evaluate additional machine learning algorithms.
3. Improve semantic policy retrieval using embedding-based methods.
4. Integrate a production-grade language model with appropriate security controls.
5. Develop a dashboard for customer segmentation and support analytics.
6. Integrate the system with a real customer support platform after proper testing and authorization.

---

## 11. Conclusion

The Instamart Customer Intelligence & Support Copilot demonstrates how data analytics, machine learning, NLP, RAG, and AI agents can be combined to improve customer intelligence and support automation.

The project successfully establishes a data foundation, identifies customer segments, predicts customer lapse, classifies support tickets, retrieves relevant policies, and automates refund decisions through a tool-based agent.

The evaluation results demonstrate the feasibility of combining analytical and AI-driven approaches in a simulated e-commerce customer support environment.