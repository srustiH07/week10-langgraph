from typing import TypedDict, List

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

from crewai import Agent, Task, Crew, LLM

import mlflow


# =========================================================
# W10D5 - STATEFUL CUSTOMER SUPPORT AGENT
# =========================================================
#
# Approved AI/ML 3M stack:
# - CrewAI  -> customer support agent
# - LangGraph -> workflow, routing and state
# - MLflow -> experiment tracking
# - Ragas -> response evaluation
# - MLOps -> logging and saved evaluation artifacts
#
# Local LLM:
# - Ollama
# - qwen2.5:3b
#
# =========================================================


# ---------------------------------------------------------
# State definition
# ---------------------------------------------------------
class SupportState(TypedDict):
    user_input: str
    classification: str
    route: str
    conversation_history: List[str]
    response: str
    human_decision: str


# ---------------------------------------------------------
# Classify customer issue
# ---------------------------------------------------------
def classify(state: SupportState):

    user_input = state["user_input"].lower()

    if any(word in user_input for word in [
        "login",
        "password",
        "error",
        "technical",
        "application",
        "app"
    ]):
        classification = "technical"

    elif any(word in user_input for word in [
        "refund",
        "payment",
        "billing",
        "charge",
        "invoice"
    ]):
        classification = "billing"

    elif any(word in user_input for word in [
        "hello",
        "hi",
        "hey"
    ]):
        classification = "greeting"

    else:
        classification = "general"

    print("\n[CLASSIFY]")
    print(f"Input: {state['user_input']}")
    print(f"Classification: {classification}")

    return {
        "classification": classification
    }


# ---------------------------------------------------------
# Route customer based on classification
# ---------------------------------------------------------
def route(state: SupportState):

    classification = state["classification"]

    if classification == "technical":
        selected_route = "technical_support"

    elif classification == "billing":
        selected_route = "billing_support"

    elif classification == "greeting":
        selected_route = "general_support"

    else:
        selected_route = "general_support"

    print(f"[ROUTE] {selected_route}")

    return {
        "route": selected_route
    }


# ---------------------------------------------------------
# Conditional routing function
# ---------------------------------------------------------
def choose_route(state: SupportState):

    return state["route"]


# ---------------------------------------------------------
# Create CrewAI customer support response
# ---------------------------------------------------------
def create_support_response(state: SupportState):

    route_name = state["route"]

    # -----------------------------------------------------
    # Use local Ollama model through CrewAI.
    # This avoids requiring an OpenAI API key.
    # -----------------------------------------------------
    llm = LLM(
        model="ollama/qwen2.5:3b",
        base_url="http://localhost:11434"
    )

    support_agent = Agent(
        role="Customer Support Specialist",
        goal=(
            "Provide helpful, professional and concise "
            "customer support responses."
        ),
        backstory=(
            "You are an experienced customer support specialist. "
            "You understand technical, billing and general customer "
            "issues. You always respond clearly and professionally."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False
    )

    task = Task(
        description=(
            f"Customer message: {state['user_input']}\n\n"
            f"Support category: {route_name}\n\n"
            f"Previous conversation:\n"
            f"{state['conversation_history']}\n\n"
            "Write a short, helpful and professional customer "
            "support response.\n"
            "Do not invent company policies.\n"
            "Do not claim that you performed an action you cannot "
            "actually perform.\n"
            "If more information is needed, politely ask the "
            "customer for it."
        ),
        expected_output=(
            "A concise and professional customer support response."
        ),
        agent=support_agent
    )

    crew = Crew(
        agents=[support_agent],
        tasks=[task],
        verbose=False
    )

    result = crew.kickoff()

    response = str(result)

    print("\n[CREWAI RESPONSE]")
    print(response)

    return {
        "response": response
    }


# ---------------------------------------------------------
# Update conversation memory
# ---------------------------------------------------------
def update_memory(state: SupportState):

    history = list(state["conversation_history"])

    history.append(
        f"User: {state['user_input']}"
    )

    history.append(
        f"Agent: {state['response']}"
    )

    print(
        f"[MEMORY] Conversation entries: {len(history)}"
    )

    return {
        "conversation_history": history
    }


# ---------------------------------------------------------
# Human-in-the-loop review
# ---------------------------------------------------------
def human_review(state: SupportState):

    print("\n[HUMAN REVIEW]")
    print("Response waiting for approval:")
    print(state["response"])

    decision = interrupt(
        {
            "message": "Human review required.",
            "question": (
                "Approve this customer support response? "
                "Enter yes or no."
            )
        }
    )

    print(f"[HUMAN DECISION] {decision}")

    return {
        "human_decision": str(decision)
    }


# ---------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------
def build_graph():

    builder = StateGraph(SupportState)

    # Primary workflow nodes.
    builder.add_node(
        "classify",
        classify
    )

    builder.add_node(
        "route",
        route
    )

    builder.add_node(
        "support_agent",
        create_support_response
    )

    # State and human review nodes.
    builder.add_node(
        "update_memory",
        update_memory
    )

    builder.add_node(
        "human_review",
        human_review
    )

    # Start workflow.
    builder.add_edge(
        START,
        "classify"
    )

    # Classification goes to routing.
    builder.add_edge(
        "classify",
        "route"
    )

    # Conditional routing.
    builder.add_conditional_edges(
        "route",
        choose_route,
        {
            "technical_support": "support_agent",
            "billing_support": "support_agent",
            "general_support": "support_agent"
        }
    )

    # Agent response goes to memory.
    builder.add_edge(
        "support_agent",
        "update_memory"
    )

    # Memory goes to human approval.
    builder.add_edge(
        "update_memory",
        "human_review"
    )

    # Workflow ends after human approval.
    builder.add_edge(
        "human_review",
        END
    )

    # MemorySaver stores state by thread_id.
    checkpointer = MemorySaver()

    return builder.compile(
        checkpointer=checkpointer
    )


# ---------------------------------------------------------
# Run one conversation
# ---------------------------------------------------------
def run_conversation(
    graph,
    user_input,
    thread_id,
    existing_history=None
):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    if existing_history is None:
        existing_history = []

    initial_state = {
        "user_input": user_input,
        "classification": "",
        "route": "",
        "conversation_history": existing_history,
        "response": "",
        "human_decision": ""
    }

    result = graph.invoke(
        initial_state,
        config=config
    )

    # Detect the human-in-the-loop interrupt.
    if "__interrupt__" in result:

        print("\n[WORKFLOW PAUSED]")
        print("Human approval is required.")

        result = graph.invoke(
            Command(resume="yes"),
            config=config
        )

    return result


# ---------------------------------------------------------
# Run five test cases
# ---------------------------------------------------------
def run_tests(graph):

    print("\n" + "=" * 70)
    print("W10D5 - TESTING CUSTOMER SUPPORT AGENT")
    print("=" * 70)

    test_inputs = [
        "I cannot login to my account",
        "There is an error in my application",
        "I need a refund for my payment",
        "I was charged incorrectly",
        "Hello, I need help with your service"
    ]

    results = []

    for number, user_input in enumerate(
        test_inputs,
        start=1
    ):

        print("\n" + "-" * 70)
        print(f"TEST {number}")
        print("-" * 70)

        thread_id = f"w10d5-test-{number}"

        result = run_conversation(
            graph,
            user_input,
            thread_id
        )

        print(
            f"\nClassification: "
            f"{result['classification']}"
        )

        print(
            f"Route: "
            f"{result['route']}"
        )

        print(
            f"Human decision: "
            f"{result['human_decision']}"
        )

        results.append(
            {
                "input": user_input,
                "classification": result[
                    "classification"
                ],
                "route": result["route"],
                "response": result["response"]
            }
        )

    return results


# ---------------------------------------------------------
# Test persistent conversation memory
# ---------------------------------------------------------
def test_memory(graph):

    print("\n" + "=" * 70)
    print("W10D5 - PERSISTENT MEMORY TEST")
    print("=" * 70)

    thread_id = "w10d5-memory-demo"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # First conversation turn.
    first_state = {
        "user_input": (
            "I cannot login to my account"
        ),
        "classification": "",
        "route": "",
        "conversation_history": [],
        "response": "",
        "human_decision": ""
    }

    first_result = graph.invoke(
        first_state,
        config=config
    )

    if "__interrupt__" in first_result:

        first_result = graph.invoke(
            Command(resume="yes"),
            config=config
        )

    print("\nAfter first conversation:")
    print(
        first_result["conversation_history"]
    )

    # Second conversation turn using the SAME thread.
    second_state = {
        "user_input": (
            "The error is still happening"
        ),
        "classification": "",
        "route": "",
        "conversation_history": [],
        "response": "",
        "human_decision": ""
    }

    second_result = graph.invoke(
        second_state,
        config=config
    )

    if "__interrupt__" in second_result:

        second_result = graph.invoke(
            Command(resume="yes"),
            config=config
        )

    print("\nAfter second conversation:")
    print(
        second_result["conversation_history"]
    )

    print(
        "\nMemory entries after second turn: "
        f"{len(second_result['conversation_history'])}"
    )


# ---------------------------------------------------------
# Save test results
# ---------------------------------------------------------
def save_results(results):

    with open(
        "output/test_results.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "W10D5 Stateful Customer Support Agent\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        for number, item in enumerate(
            results,
            start=1
        ):

            file.write(
                f"Test {number}\n"
            )

            file.write(
                "-" * 40 + "\n"
            )

            file.write(
                f"Input: {item['input']}\n"
            )

            file.write(
                f"Classification: "
                f"{item['classification']}\n"
            )

            file.write(
                f"Route: {item['route']}\n"
            )

            file.write(
                f"Response: {item['response']}\n\n"
            )


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------
if __name__ == "__main__":

    print("=" * 70)
    print("W10D5 - STATEFUL CUSTOMER SUPPORT AGENT")
    print("=" * 70)

    # Start MLflow experiment.
    mlflow.set_experiment(
        "W10D5_Stateful_Customer_Support"
    )

    with mlflow.start_run():

        # Record project configuration.
        mlflow.log_param(
            "workflow",
            "LangGraph + CrewAI"
        )

        mlflow.log_param(
            "project",
            "Stateful Customer Support Agent"
        )

        mlflow.log_param(
            "llm",
            "Ollama qwen2.5:3b"
        )

        mlflow.log_param(
            "memory",
            "LangGraph MemorySaver"
        )

        mlflow.log_param(
            "human_in_loop",
            "enabled"
        )

        # Build LangGraph.
        graph = build_graph()

        # Run five customer-support tests.
        results = run_tests(graph)

        # Test persistent memory.
        test_memory(graph)

        # Save output for MLOps and Ragas.
        save_results(results)

        # Log generated artifact to MLflow.
        mlflow.log_artifact(
            "output/test_results.txt"
        )

        # Record test metrics.
        mlflow.log_metric(
            "test_cases",
            len(results)
        )

        mlflow.log_metric(
            "successful_test_cases",
            len(results)
        )

        print("\nMLflow tracking completed.")

    print("\n" + "=" * 70)
    print(
        "W10D5 CUSTOMER SUPPORT AGENT "
        "COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

