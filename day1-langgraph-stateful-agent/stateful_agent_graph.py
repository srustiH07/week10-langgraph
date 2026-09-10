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
# Node 1: Classify the user input
# ---------------------------------------------------------
def classify(state: AgentState):
    user_input = state["user_input"].lower()

    # Simple rule-based classification.
    # This keeps the demonstration local and deterministic.
    if any(word in user_input for word in ["hello", "hi", "hey"]):
        classification = "greeting"
    elif any(word in user_input for word in ["help", "problem", "issue", "error"]):
        classification = "support"
    else:
        classification = "general"

    print(f"[CLASSIFY] Input: {state['user_input']}")
    print(f"[CLASSIFY] Classification: {classification}")

    return {"classification": classification}


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

    return {"route": selected_route}


# ---------------------------------------------------------
# Conditional routing function
# ---------------------------------------------------------
def choose_response(state: AgentState):
    return state["route"]


# ---------------------------------------------------------
# Node 3: Respond
# ---------------------------------------------------------
def respond(state: AgentState):
    route_name = state["route"]

    if route_name == "friendly":
        response = "Hello! Welcome to the LangGraph agent."
    elif route_name == "support":
        response = "I can help you investigate your problem or error."
    else:
        response = "This is a general response from the LangGraph agent."

    print(f"[RESPOND] {response}")

    return {"response": response}


# ---------------------------------------------------------
# Human-in-the-loop node
# ---------------------------------------------------------
def human_review(state: AgentState):
    print("\n[HUMAN REVIEW] The workflow is paused.")
    print("Current response:")
    print(state["response"])

    # LangGraph pauses execution here.
    # The value supplied through Command(resume=...)
    # becomes the result of this interrupt.
    decision = interrupt(
        {
            "message": "Human review required.",
            "question": "Approve this response? Enter yes or no."
        }
    )

    print(f"[HUMAN REVIEW] Decision received: {decision}")

    return {"human_decision": str(decision)}


# ---------------------------------------------------------
# Build the LangGraph
# ---------------------------------------------------------
def build_graph():
    builder = StateGraph(AgentState)

    # Three primary nodes required by the task.
    builder.add_node("classify", classify)
    builder.add_node("route", route)
    builder.add_node("respond", respond)

    # Additional human-review node for HITL.
    builder.add_node("human_review", human_review)

    # Start -> classify -> route
    builder.add_edge(START, "classify")
    builder.add_edge("classify", "route")

    # Conditional routing based on the classification-derived route.
    builder.add_conditional_edges(
        "route",
        choose_response,
        {
            "friendly": "respond",
            "support": "respond",
            "general": "respond",
        },
    )

    # Respond -> human review -> end
    builder.add_edge("respond", "human_review")
    builder.add_edge("human_review", END)

    # MemorySaver allows LangGraph to pause and resume state.
    checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------
# Test 1: Five inputs for classification and routing
# ---------------------------------------------------------
def run_five_tests(graph):
    print("\n" + "=" * 60)
    print("TESTING 5 INPUTS")
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
                "thread_id": f"test-{number}"
            }
        }

        # For the five-input test, automatically approve the
        # human-review interrupt so every test can complete.
        result = graph.invoke(
            {
                "user_input": user_input,
                "classification": "",
                "route": "",
                "response": "",
                "human_decision": "",
            },
            config=config,
        )

        # If an interrupt occurs, resume the graph.
        if "__interrupt__" in result:
            result = graph.invoke(
                Command(resume="yes"),
                config=config,
            )

        print(f"Final classification: {result['classification']}")
        print(f"Final route: {result['route']}")
        print(f"Final response: {result['response']}")


# ---------------------------------------------------------
# Test 2: Human-in-the-loop pause and resume
# ---------------------------------------------------------
def run_human_interrupt_test(graph):
    print("\n" + "=" * 60)
    print("HUMAN-IN-THE-LOOP INTERRUPT TEST")
    print("=" * 60)

    config = {
        "configurable": {
            "thread_id": "human-review-demo"
        }
    }

    initial_state = {
        "user_input": "I need help with an application error",
        "classification": "",
        "route": "",
        "response": "",
        "human_decision": "",
    }

    print("\nStarting workflow...")
    paused_result = graph.invoke(initial_state, config=config)

    print("\nWorkflow state after interrupt:")
    print(paused_result)

    print("\nResuming workflow with human decision: yes")

    resumed_result = graph.invoke(
        Command(resume="yes"),
        config=config,
    )

    print("\nWorkflow resumed successfully.")
    print(f"Human decision: {resumed_result['human_decision']}")
    print(f"Final response: {resumed_result['response']}")


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("W10D1 - LANGGRAPH STATEFUL AGENT GRAPH")
    print("=" * 60)

    graph = build_graph()

    run_five_tests(graph)
    run_human_interrupt_test(graph)

    print("\n" + "=" * 60)
    print("W10D1 COMPLETED SUCCESSFULLY")
    print("=" * 60)