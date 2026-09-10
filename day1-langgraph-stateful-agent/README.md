# W10D1 - LangGraph Stateful Agent Graphs

## Objective

Build a stateful LangGraph agent with classification, conditional routing, response generation, and human-in-the-loop interrupts.

## Graph Flow

```text
START
  |
  v
classify
  |
  v
route
  |
  +---- friendly ----+
  |                  |
  +---- support -----+--> respond --> human_review --> END
  |                  |
  +---- general -----+