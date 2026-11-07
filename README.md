# Overview

## Instamart Customer Intelligence & Support Copilot

The Instamart Customer Intelligence & Support Copilot is an offline data analytics and AI-based customer support project. It combines data engineering, machine learning, natural language processing, Retrieval-Augmented Generation (RAG), and tool-based AI agents to improve customer understanding and support operations.

The project is divided into four parts:

- **Part A – Data Foundation:** Generate, clean, store, and validate customer, product, and order data.
- **Part B – Customer Segmentation and Lapse Prediction:** Identify customer segments and predict customers who may become inactive.
- **Part C – Support Ticket Triage with Guardrails:** Classify customer support tickets while applying validation and sensitive information redaction.
- **Part D – Policy RAG Assistant and Refund Agent:** Retrieve relevant company policies and simulate refund decisions using a tool-based agent.

The project uses synthetic Instamart-style data and runs locally without requiring an external LLM API or API key.

## Project Structure

```text
Instamart-Customer-Intelligence-Copilot/
│
├── data/
│   ├── clean/
│   │   ├── customers.csv
│   │   ├── products.csv
│   │   └── orders.csv
│   └── instamart.db
│
├── sql/
│   └── queries.sql
│
├── src/
│   ├── part_a/
│   ├── part_b/
│   ├── part_c/
│   └── part_d/
│
├── outputs/
│   ├── customer_segments.csv
│   ├── cohort_analysis.csv
│   ├── lapse_model.joblib
│   ├── retention_list.csv
│   ├── lapse_model_comparison.csv
│   ├── triage_results.csv
│   ├── triage_budget_demo.csv
│   ├── agent_runs.jsonl
│   └── refund_ledger.jsonl
│
├── state/
│   └── Agent checkpoint files
│
├── generate_data.py
├── mock_llm.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Technologies Used

- Python
- Pandas
- NumPy
- SQLite
- SQL
- Scikit-learn
- Logistic Regression
- Random Forest
- TF-IDF
- Cosine Similarity
- JSON and JSONL
- Git and GitHub
- Visual Studio Code

# Reproduce

Follow the commands below from the project root directory to reproduce the project on Windows.

1. Create a virtual environment:

   `python -m venv .venv`

2. Activate the virtual environment:

   `.\.venv\Scripts\Activate.ps1`

3. Upgrade pip:

   `python -m pip install --upgrade pip`

4. Install the required dependencies:

   `python -m pip install -r requirements.txt`

5. Generate the datasets:

   `python generate_data.py`

6. Execute Part A – Data Foundation:

   `python src/part_a/part_a.py`

7. Execute Part B – Customer Segmentation and Lapse Prediction:

   `python src/part_b/part_b.py`

8. Execute Part C – Support Ticket Triage:

   `python src/part_c/part_c.py`

9. Execute Part D – Policy RAG Assistant and Refund Agent:

   `python src/part_d/part_d.py`

The project uses the supplied `mock_llm.py` as an offline stand-in for a hosted language model. No API key or external LLM service is required.

# Part A

## Data Foundation

Part A establishes the data foundation for the project. Customer, product, and order datasets are generated, cleaned, stored, and validated.

### Activities Completed

- Generated synthetic customer, product, and order datasets.
- Cleaned and prepared the datasets.
- Stored the cleaned datasets as CSV files.
- Created a SQLite database.
- Created the required database tables.
- Prepared SQL queries for data analysis.
- Verified database records and data integrity.

### Database Record Counts

| Database Table | Number of Records |
|---|---:|
| Customers | 1,200 |
| Products | 40 |
| Orders | 12,309 |

### Database Integrity Results

| Validation Check | Result |
|---|---:|
| Orders without matching customers | 0 |
| Orders without matching products | 0 |
| Foreign key violations | 0 |

The database integrity check returned no foreign key violations.

### Key Files

- `data/clean/customers.csv`
- `data/clean/products.csv`
- `data/clean/orders.csv`
- `data/instamart.db`
- `sql/queries.sql`

# Part B

## Customer Segmentation and Lapse Prediction

Part B focuses on understanding customer behaviour, identifying customer segments, and predicting customers who may become inactive.

### Activities Completed

- Performed customer data analysis.
- Created customer segments.
- Conducted cohort analysis.
- Prepared features for lapse prediction.
- Trained and evaluated machine learning models.
- Generated a customer retention list.

### Machine Learning Models

Two machine learning algorithms were implemented:

1. Logistic Regression
2. Random Forest

### Model Evaluation Results

The models were evaluated using the F1-score on the test dataset.

| Model | Test F1-Score |
|---|---:|
| Logistic Regression | 0.719 |
| Random Forest | 0.706 |

**Required target F1-score:** 0.55 or above.

Both models achieved the required target. Logistic Regression produced the higher test F1-score of 0.719.

### Key Outputs

- `outputs/customer_segments.csv`
- `outputs/cohort_analysis.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`
- `outputs/lapse_model_comparison.csv`

# Part C

## Support Ticket Triage with Guardrails

Part C implements a support ticket classification system using the supplied offline language model simulator. The system classifies customer support tickets while applying input validation, output validation, token budgeting, and sensitive information redaction.

### Activities Completed

- Implemented support ticket classification.
- Used zero-shot and few-shot classification approaches.
- Implemented output validation.
- Added fallback handling for invalid responses.
- Added checks for empty and excessively long inputs.
- Implemented sensitive information redaction.
- Compared classification accuracy and token usage.

### Classification Results

| Metric | Zero-Shot | Few-Shot |
|---|---:|---:|
| Classified Tickets | 147 | 147 |
| Accuracy | 85.71% | 100.00% |
| Total Tokens | 10,336 | 19,696 |
| Average Tokens per Ticket | 70.31 | 133.99 |

The few-shot approach achieved higher classification accuracy but required more tokens than the zero-shot approach.

### Guardrail Results

The system processed 164 support ticket records.

| Processing Status | Number of Tickets |
|---|---:|
| Classified JSON | 135 |
| Classified Keyword | 12 |
| Flagged Invalid Output | 9 |
| Skipped Empty | 4 |
| Flagged Long | 4 |
| **Total** | **164** |

### Sensitive Information Redaction

| Information Type | Number of Redactions |
|---|---:|
| Email Addresses | 55 |
| Phone Numbers | 91 |
| Order IDs | 132 |
| **Total Redactions** | **278** |

### Key Outputs

- `outputs/triage_results.csv`
- `outputs/triage_budget_demo.csv`

# Part D

## Policy RAG Assistant and Refund Agent

Part D implements a Policy Retrieval-Augmented Generation (RAG) system and a tool-based refund agent. The system retrieves relevant policy information and uses order details and refund rules to determine appropriate customer support decisions.

## 1. Policy RAG Assistant

The policy retrieval system processes policy documents, divides them into smaller chunks, and retrieves relevant information based on customer queries.

### RAG Configuration

| Component | Configuration |
|---|---|
| Chunking Method | Word-based chunking |
| Chunk Size | 40 words |
| Chunk Overlap | 10 words |
| Total Chunks | 15 |
| Retrieval Method | TF-IDF |
| Similarity Measure | Cosine Similarity |
| Top Results Retrieved | 3 |

### Policy Documents

The system uses five policy documents:

- Refund Policy
- Delivery Policy
- Escalation Policy
- Payment Policy
- Privacy Policy

### RAG Evaluation Results

| Evaluation Metric | Result |
|---|---:|
| Correct Document Retrieval | 12/12 |
| Expected Facts Retrieved | 12/12 |
| Fallback Responses | 2/2 |

### Chunk Overlap Comparison

The effect of chunk overlap was evaluated by comparing retrieval performance.

| Metric | With Overlap | Without Overlap |
|---|---:|---:|
| Total Chunks | 15 | 13 |
| Correct Document Retrieval | 12/12 | 12/12 |
| Expected Facts Retrieved | 12/12 | 10/12 |

The overlapping chunk configuration retrieved all expected facts in the evaluation, while the non-overlapping configuration retrieved 10 out of 12 expected facts.

## 2. Refund Agent

The refund agent uses a tool-based approach to retrieve order information, search policies, calculate refund amounts, issue simulated refunds, and escalate cases when required.

### Tools Implemented

The agent uses the following tools:

1. `get_order`
2. `search_policy`
3. `calculate_refund`
4. `issue_refund`
5. `escalate_to_human`

### Refund Decision Rules

- Only completed orders are considered for refund processing.
- Orders delivered more than 45 minutes late qualify for a full refund under the implemented rules.
- Orders delivered within 45 minutes do not qualify for a late-delivery refund.
- Refunds up to INR 1,000 can be processed automatically.
- Refund amounts above INR 1,000 require human escalation.
- Missing orders return a `NOT_FOUND` status.

All refund operations are simulated. No real payments or customer transactions are processed.

## 3. Seven-Scenario Agent Evaluation

The agent was evaluated using seven scenarios.

| Scenario | Expected Result | Actual Result | Tool Calls | Tokens Used |
|---|---|---|---:|---:|
| S1 | REFUND | REFUND | 5 | 1,152 |
| S2 | REFUND | REFUND | 5 | 1,152 |
| S3 | ESCALATE | ESCALATE | 5 | 1,139 |
| S4 | DENIED | DENIED | 4 | 809 |
| S5 | DENIED | DENIED | 3 | 490 |
| S6 | NOT_FOUND | NOT_FOUND | 2 | 174 |
| S7 | FALLBACK | FALLBACK_MAX_ITERATIONS | 6 | 3,306 |
| **Total** | | | **30** | **8,222** |

The first six scenarios produced the expected decision statuses. Scenario S7 safely reached the maximum-iteration fallback.

## 4. Stop-Safely Demonstrations

The agent was tested to ensure that it stops safely when execution limits are reached.

| Demonstration | Actual Status | Result |
|---|---|---|
| Token Limit | FALLBACK_TOKEN_LIMIT | Passed |
| Timeout | FALLBACK_TIMEOUT | Passed |

The token-limit demonstration stopped when the configured token budget was reached. The timeout demonstration stopped safely when the execution time limit was reached.

The maximum-iteration fallback was also demonstrated in Scenario S7, which returned `FALLBACK_MAX_ITERATIONS` after 6 tool calls.

## 5. Crash-and-Resume Demonstration

A checkpoint and resume mechanism was implemented to preserve agent progress during execution.

### Execution Process

1. The agent started processing a refund request.
2. It executed `get_order`.
3. It executed `search_policy`.
4. The agent state was saved to a checkpoint.
5. Execution was paused to simulate an interruption.
6. The saved checkpoint was loaded.
7. The agent resumed from its saved state.
8. It executed `calculate_refund`.
9. It executed `issue_refund`.
10. The agent completed the refund workflow.

### Complete List of Tools Executed

`get_order → search_policy → calculate_refund → issue_refund`

### Crash-and-Resume Result

| Item | Result |
|---|---|
| Initial Status | PAUSED |
| Tools Executed Before Pause | get_order, search_policy |
| Resumed Status | REFUND |
| Final Answer | REFUND: INR 128.25 issued. |

The checkpoint and resume test confirmed that the agent continued from its saved state without repeating the completed operations.

## 6. Agent Logging

The project records agent execution details for tracking and evaluation.

### Log Files

- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`

The agent execution log stores scenario details, status, final answers, tool calls, arguments, observations, and token usage.

The refund ledger stores simulated refund transaction records.

## 7. ReAct vs Plan-and-Execute

### ReAct Approach

ReAct stands for Reasoning and Acting. In this approach, the agent alternates between reasoning about the current situation and executing tools to obtain information or perform an action.

### Plan-and-Execute Approach

In Plan-and-Execute, the system first creates a sequence of tasks and then executes the planned steps.

| ReAct | Plan-and-Execute |
|---|---|
| Reasons and acts step by step. | Creates a plan before execution. |
| Can adapt based on tool results. | Follows a predefined execution plan. |
| Suitable for dynamic support queries. | Suitable for structured multi-step workflows. |
| Decisions are made during execution. | Tasks are organised before execution. |

The project uses a tool-based agent workflow to retrieve order information, search policies, calculate refunds, and determine whether a case should be refunded, denied, or escalated.

# Limitations and Next Steps

## Limitations

- The project uses synthetic data rather than actual Instamart customer records.
- The language model is simulated using the supplied offline `mock_llm.py`.
- The refund agent does not connect to an actual payment gateway.
- The policy documents are limited to the sample policies included in the project.
- Machine learning predictions depend on the quality and characteristics of the generated dataset.
- The project is designed for local execution and demonstration rather than production deployment.

## Next Steps

- Test the system with larger and more diverse datasets.
- Improve customer segmentation and lapse prediction.
- Expand the policy knowledge base.
- Improve support ticket classification and validation.
- Integrate a real language model in an appropriately secured environment if permitted.
- Add monitoring and more extensive testing before production deployment.
- Evaluate the system using real-world operational requirements and authorised data.
