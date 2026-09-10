# W10D3 - LangGraph + Memory

## Objective

Build and test a LangGraph with conditional routing, human-in-the-loop interruption, and persistent conversation memory.

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