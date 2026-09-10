# W10D2 - LangGraph State Machines & Conditional Edges

## Objective

Build and test a LangGraph state machine with conditional edges, routing logic, and a human-in-the-loop interruption.

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
  +---- technical_support ----+
  |                           |
  +---- billing_support ------+--> respond --> human_review --> END
  |                           |
  +---- general_support ------+