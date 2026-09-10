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
# Node 1: Classify the user message
# ---------------------------------------------------------
def classify(state: AgentState):
    user_input = state["user_input"].lower()

    if any(word in user_input for word in [
        "hello", "hi", "hey"
    ]):
        classification = "greeting"

    elif any(word in user_input for word in [
        "error", "problem", "issue", "help"
    ]):
        classification = "support"

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

    if classification == "greeting":
        selected_route = "friendly"

    elif classification == "support":
        selected_route = "support"

    else:
        selected_route = "general"

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
# Node 3: Generate a response
# ---------------------------------------------------------
def respond(state: AgentState):
    route_name = state["route"]

    if route_name == "friendly":
        response = "Hello! How can I help you today?"

    elif route_name == "support":
        response = "I can help you investigate your problem."

    else:
        response = "I can provide information about your request."

    print(f"[RESPOND] {response}")

    return {
        "response": response
    }


# ---------------------------------------------------------
# Human-in-the-loop review
# ---------------------------------------------------------
def human_review(state: AgentState):
    print("\n[HUMAN REVIEW] Workflow paused.")
    print(f"Response waiting for approval: {state['response']}")

    decision = interrupt(
        {
            "message": "Human review required.",
            "question": "Approve this response? Enter yes or no."
        }
    )

    print(f"[HUMAN REVIEW] Human decision: {decision}")

    return {
        "human_decision": str(decision)
    }


# ---------------------------------------------------------
# Build the LangGraph
# ---------------------------------------------------------
def build_graph():

    builder = StateGraph(AgentState)

    # Add the three primary nodes.
    builder.add_node("classify", classify)
    builder.add_node("route", route)
    builder.add_node("respond", respond)

    # Add human review as the human-in-the-loop node.
    builder.add_node("human_review", human_review)

    # Start the workflow.
    builder.add_edge(START, "classify")

    # Classification flows into routing.
    builder.add_edge("classify", "route")

    # Conditional edges select the response path.
    builder.add_conditional_edges(
        "route",
        choose_route,
        {
            "friendly": "respond",
            "support": "respond",
            "general": "respond",
        },
    )

    # Response must be reviewed by a human.
    builder.add_edge("respond", "human_review")

    # End after human approval.
    builder.add_edge("human_review", END)

    # MemorySaver stores the interrupted workflow state.
    checkpointer = MemorySaver()

    return builder.compile(
        checkpointer=checkpointer
    )


# ---------------------------------------------------------
# Test five inputs
# ---------------------------------------------------------
def run_five_tests(graph):

    print("\n" + "=" * 60)
    print("W10D4 - TESTING 5 INPUTS")
    print("=" * 60)

    test_inputs = [
        "Hello there",
        "Hi, how are you?",
        "I have a problem with my account",
        "There is an error in my application",
        "Tell me about artificial intelligence",
    ]

    for number, user_input in enumerate(test_inputs, start=1):

        print(f"\n--- Test {number} ---")

        config = {
            "configurable": {
                "thread_id": f"w10d4-test-{number}"
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

        # Resume the workflow after the human interrupt.
        if "__interrupt__" in result:

            print("[TEST] Human review detected.")

            result = graph.invoke(
                Command(resume="yes"),
                config=config
            )

        print(f"Final classification: {result['classification']}")
        print(f"Final route: {result['route']}")
        print(f"Human decision: {result['human_decision']}")
        print(f"Final response: {result['response']}")


# ---------------------------------------------------------
# Explicit human-in-the-loop interrupt test
# ---------------------------------------------------------
def run_human_interrupt_test(graph):

    print("\n" + "=" * 60)
    print("W10D4 - HUMAN-IN-THE-LOOP INTERRUPT TEST")
    print("=" * 60)

    config = {
        "configurable": {
            "thread_id": "w10d4-human-demo"
        }
    }

    initial_state = {
        "user_input": "I have an issue with my account",
        "classification": "",
        "route": "",
        "response": "",
        "human_decision": "",
    }

    print("\nStarting workflow...")

    # First invocation pauses at interrupt().
    paused_result = graph.invoke(
        initial_state,
        config=config
    )

    print("\nWorkflow paused successfully.")
    print("State returned by LangGraph:")
    print(paused_result)

    # Resume using a human decision.
    print("\nResuming workflow with human decision: yes")

    resumed_result = graph.invoke(
        Command(resume="yes"),
        config=config
    )

    print("\nWorkflow resumed successfully.")
    print(f"Human decision: {resumed_result['human_decision']}")
    print(f"Final response: {resumed_result['response']}")


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------
if __name__ == "__main__":

    print("=" * 60)
    print("W10D4 - HUMAN-IN-THE-LOOP WITH LANGGRAPH")
    print("=" * 60)

    graph = build_graph()

    # Test the graph with five different inputs.
    run_five_tests(graph)

    # Demonstrate the interrupt and resume process explicitly.
    run_human_interrupt_test(graph)

    print("\n" + "=" * 60)
    print("W10D4 COMPLETED SUCCESSFULLY")
    print("=" * 60)