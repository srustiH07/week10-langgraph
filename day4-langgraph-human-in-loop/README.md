# W10D4 - Human-in-the-Loop with LangGraph

## Objective

Build and test a LangGraph with conditional routing and human-in-the-loop interruption.

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