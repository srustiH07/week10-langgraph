from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# ---------------------------------------------------------
# State definition
# ---------------------------------------------------------
class AgentState(TypedDict):
    user_input: str
    classification: str
    route: str
    response: str
    human_decision: str


# ---------------------------------------------------------
# Node 1: Classify the input
# ---------------------------------------------------------
def classify(state: AgentState):
    user_input = state["user_input"].lower()

    # Classify the request into one of three categories.
    if any(word in user_input for word in [
        "error", "bug", "technical", "login", "password"
    ]):
        classification = "technical"

    elif any(word in user_input for word in [
        "payment", "bill", "billing", "refund", "money"
    ]):
        classification = "billing"

    else:
        classification = "general"

    print(f"[CLASSIFY] Input: {state['user_input']}")
    print(f"[CLASSIFY] Classification: {classification}")

    return {
        "classification": classification
    }


# ---------------------------------------------------------
# Node 2: Route based on classification
# ---------------------------------------------------------
def route(state: AgentState):
    classification = state["classification"]

    if classification == "technical":
        selected_route = "technical_support"

    elif classification == "billing":
        selected_route = "billing_support"

    else:
        selected_route = "general_support"

    print(f"[ROUTE] Selected route: {selected_route}")

    return {
        "route": selected_route
    }


# ---------------------------------------------------------
# Conditional routing function
# ---------------------------------------------------------
def choose_route(state: AgentState):
    return state["route"]


# ---------------------------------------------------------
# Node 3: Respond according to route
# ---------------------------------------------------------
def respond(state: AgentState):
    route_name = state["route"]

    if route_name == "technical_support":
        response = (
            "Your request has been routed to technical support."
        )

    elif route_name == "billing_support":
        response = (
            "Your request has been routed to billing support."
        )

    else:
        response = (
            "Your request has been routed to general support."
        )

    print(f"[RESPOND] {response}")

    return {
        "response": response
    }


# ---------------------------------------------------------
# Human-in-the-loop review node
# ---------------------------------------------------------
def human_review(state: AgentState):
    print("\n[HUMAN REVIEW] Workflow paused.")
    print(f"Response waiting for approval: {state['response']}")

    # Pause the graph and wait for human input.
    decision = interrupt(
        {
            "message": "Human approval required.",
            "question": "Approve this response? Enter yes or no."
        }
    )

    print(f"[HUMAN REVIEW] Human decision: {decision}")

    return {
        "human_decision": str(decision)
    }


# ---------------------------------------------------------
# Build the state machine
# ---------------------------------------------------------
def build_graph():
    builder = StateGraph(AgentState)

    # Three required nodes.
    builder.add_node("classify", classify)
    builder.add_node("route", route)
    builder.add_node("respond", respond)

    # Additional node for human-in-the-loop approval.
    builder.add_node("human_review", human_review)

    # Start -> classify -> route.
    builder.add_edge(START, "classify")
    builder.add_edge("classify", "route")

    # Conditional edges based on the selected route.
    builder.add_conditional_edges(
        "route",
        choose_route,
        {
            "technical_support": "respond",
            "billing_support": "respond",
            "general_support": "respond",
        },
    )

    # Response -> human review -> end.
    builder.add_edge("respond", "human_review")
    builder.add_edge("human_review", END)

    # MemorySaver preserves graph state during interrupts.
    checkpointer = MemorySaver()

    return builder.compile(
        checkpointer=checkpointer
    )


# ---------------------------------------------------------
# Test five different inputs
# ---------------------------------------------------------
def run_five_tests(graph):
    print("\n" + "=" * 60)
    print("W10D2 - TESTING 5 INPUTS")
    print("=" * 60)

    test_inputs = [
        "I cannot login to my account",
        "There is a technical error in my application",
        "I need a refund for my payment",
        "I have a billing problem",
        "Tell me about your services",
    ]

    for number, user_input in enumerate(test_inputs, start=1):

        print(f"\n--- Test {number} ---")

        config = {
            "configurable": {
                "thread_id": f"w10d2-test-{number}"
            }
        }

        initial_state = {
            "user_input": user_input,
            "classification": "",
            "route": "",
            "response": "",
            "human_decision": "",
        }

        result = graph.invoke(
            initial_state,
            config=config
        )

        # Automatically approve the interrupt for
        # the five-input routing tests.
        if "__interrupt__" in result:
            result = graph.invoke(
                Command(resume="yes"),
                config=config
            )

        print(f"Final classification: {result['classification']}")
        print(f"Final route: {result['route']}")
        print(f"Human decision: {result['human_decision']}")
        print(f"Final response: {result['response']}")


# ---------------------------------------------------------
# Test explicit human interrupt and resume
# ---------------------------------------------------------
def run_human_interrupt_test(graph):
    print("\n" + "=" * 60)
    print("W10D2 - HUMAN-IN-THE-LOOP TEST")
    print("=" * 60)

    config = {
        "configurable": {
            "thread_id": "w10d2-human-demo"
        }
    }

    initial_state = {
        "user_input": "I have a problem with my payment",
        "classification": "",
        "route": "",
        "response": "",
        "human_decision": "",
    }

    print("\nStarting workflow...")

    paused_result = graph.invoke(
        initial_state,
        config=config
    )

    print("\nWorkflow paused successfully.")
    print("State returned by LangGraph:")
    print(paused_result)

    print("\nResuming workflow with human decision: yes")

    resumed_result = graph.invoke(
        Command(resume="yes"),
        config=config
    )

    print("\nWorkflow resumed successfully.")
    print(f"Human decision: {resumed_result['human_decision']}")
    print(f"Final route: {resumed_result['route']}")
    print(f"Final response: {resumed_result['response']}")


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------
if __name__ == "__main__":

    print("=" * 60)
    print("W10D2 - LANGGRAPH STATE MACHINES")
    print("=" * 60)

    graph = build_graph()

    run_five_tests(graph)

    run_human_interrupt_test(graph)

    print("\n" + "=" * 60)
    print("W10D2 COMPLETED SUCCESSFULLY")
    print("=" * 60)