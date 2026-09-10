# W10D5 - Stateful Customer Support Agent

## Objective

Implement a stateful customer support agent using the approved AI/ML stack:

- CrewAI
- LangGraph
- MLflow
- Ragas
- MLOps

## Architecture

```text
User Query
    |
    v
LangGraph Stateful Workflow
    |
    v
Classify
    |
    v
Route
    |
    v
CrewAI Support Agent
    |
    v
Update Conversation Memory
    |
    v
Human Approval
    |
    v
Final Response
    |
    v
MemorySaver
    |
    v
MLflow Tracking
    |
    v
Ragas Evaluation
## Verified Execution

Customer support agent execution:

- 5 test cases executed successfully
- Stateful conversation memory verified
- Human-in-the-loop approval verified
- Local Ollama model used successfully
- MLflow tracking completed

Ragas evaluation:

- Total test cases: 5
- Successful test cases: 5
- Success rate: 100.00%
- Evaluation completed without an OpenAI API key

Generated evaluation artifact:

- output/ragas_evaluation.txt