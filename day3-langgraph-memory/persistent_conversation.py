from typing import TypedDict, List

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# ---------------------------------------------------------
# State definition
# ---------------------------------------------------------
class ConversationState(TypedDict):
    user_input: str
    classification: str
    route: str
    response: str
    human_decision: str
    conversation_history: List[str]


# ---------------------------------------------------------
# Node 1: Classify the user message
# ---------------------------------------------------------
def classify(state: ConversationState):
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
def route(state: ConversationState):
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
def choose_route(state: ConversationState):
    return state["route"]


# ---------------------------------------------------------
# Node 3: Respond and update conversation history
# ---------------------------------------------------------
def respond(state: ConversationState):
    route_name = state["route"]

    if route_name == "friendly":
        response = "Hello! How can I help you today?"

    elif route_name == "support":
        response = "I can help you investigate your problem."

    else:
        response = "I can provide information about your request."

    # Retrieve the existing history from checkpointed state.
    previous_history = state.get("conversation_history", [])

    updated_history = previous_history + [
        f"User: {state['user_input']}",
        f"Agent: {response}"
    ]

    print(f"[RESPOND] {response}")
    print(f"[MEMORY] Conversation turns stored: {len(updated_history)}")

    return {
        "response": response,
        "conversation_history": updated_history
    }


# ---------------------------------------------------------
# Human-in-the-loop review
# ---------------------------------------------------------
def human_review(state: ConversationState):
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
    builder = StateGraph(ConversationState)

    builder.add_node("classify", classify)
    builder.add_node("route", route)
    builder.add_node("respond", respond)
    builder.add_node("human_review", human_review)

    builder.add_edge(START, "classify")
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

    builder.add_edge("respond", "human_review")
    builder.add_edge("human_review", END)

    # MemorySaver checkpoints the state.
    checkpointer = MemorySaver()

    return builder.compile(
        checkpointer=checkpointer
    )


# ---------------------------------------------------------
# Test five independent inputs
# ---------------------------------------------------------
def run_five_tests(graph):
    print("\n" + "=" * 60)
    print("W10D3 - TESTING 5 INPUTS")
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
                "thread_id": f"w10d3-test-{number}"
            }
        }

        initial_state = {
            "user_input": user_input,
            "classification": "",
            "route": "",
            "response": "",
            "human_decision": "",
            "conversation_history": [],
        }

        result = graph.invoke(
            initial_state,
            config=config
        )

        # Automatically approve the interrupt so all
        # five test cases can complete.
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
# Demonstrate persistent conversation memory
# ---------------------------------------------------------
def run_memory_test(graph):
    print("\n" + "=" * 60)
    print("W10D3 - PERSISTENT CONVERSATION MEMORY TEST")
    print("=" * 60)

    # The same thread_id is intentionally reused.
    # This allows LangGraph to retrieve previous state.
    config = {
        "configurable": {
            "thread_id": "persistent-conversation-demo"
        }
    }

    # -----------------------------------------------------
    # Conversation turn 1
    # -----------------------------------------------------
    print("\n--- Conversation Turn 1 ---")

    turn_one = {
        "user_input": "Hello, I need help",
        "classification": "",
        "route": "",
        "response": "",
        "human_decision": "",
        "conversation_history": [],
    }

    result_one = graph.invoke(
        turn_one,
        config=config
    )

    if "__interrupt__" in result_one:
        result_one = graph.invoke(
            Command(resume="yes"),
            config=config
        )

    print("Turn 1 completed.")
    print(f"Memory after Turn 1: {result_one['conversation_history']}")

    # -----------------------------------------------------
    # Conversation turn 2
    # -----------------------------------------------------
    print("\n--- Conversation Turn 2 ---")

    # Only the new user message is supplied.
    # Previous state is recovered from the same thread.
    turn_two = {
        "user_input": "There is an error in my application",
    }

    result_two = graph.invoke(
        turn_two,
        config=config
    )

    if "__interrupt__" in result_two:
        result_two = graph.invoke(
            Command(resume="yes"),
            config=config
        )

    print("Turn 2 completed.")
    print(f"Memory after Turn 2: {result_two['conversation_history']}")

    print("\nPersistent memory verified:")
    for item in result_two["conversation_history"]:
        print(f"  {item}")


# ---------------------------------------------------------
# Explicit human interrupt test
# ---------------------------------------------------------
def run_human_interrupt_test(graph):
    print("\n" + "=" * 60)
    print("W10D3 - HUMAN-IN-THE-LOOP TEST")
    print("=" * 60)

    config = {
        "configurable": {
            "thread_id": "w10d3-human-demo"
        }
    }

    initial_state = {
        "user_input": "I have an issue with my account",
        "classification": "",
        "route": "",
        "response": "",
        "human_decision": "",
        "conversation_history": [],
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
    print(f"Final response: {resumed_result['response']}")


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------
if __name__ == "__main__":

    print("=" * 60)
    print("W10D3 - LANGGRAPH + MEMORY")
    print("=" * 60)

    graph = build_graph()

    run_five_tests(graph)
    run_memory_test(graph)
    run_human_interrupt_test(graph)

    print("\n" + "=" * 60)
    print("W10D3 COMPLETED SUCCESSFULLY")
    print("=" * 60)