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