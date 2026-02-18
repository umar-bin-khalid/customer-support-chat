"""
Customer Support Chat System - Main Entry Point
Multi-agent system for TechFlow Electronics using LangGraph.
"""
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from src.graph import get_workflow, ChatState


def print_banner():
    """Print welcome banner."""
    print("\n" + "="*60)
    print("  TechFlow Electronics Customer Support")
    print("  Multi-Agent Chat System")
    print("="*60)
    print("\nType 'quit' or 'exit' to end the conversation")
    print("Type 'reset' to start a new conversation")
    print("-"*60 + "\n")


def format_agent_message(message: str, agent: str = "Agent") -> str:
    """Format agent message for display."""
    return f"\n🤖 {agent}: {message}\n"


def run_chat():
    """Run the interactive chat loop."""
    load_dotenv()
    
    # Verify API key
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY not found in environment")
        print("Please create a .env file with your Google API key")
        print("See .env.example for reference")
        return
    
    print_banner()
    
    # Initialize workflow
    try:
        workflow = get_workflow()
        print("✅ Agents initialized successfully\n")
    except Exception as e:
        print(f"❌ Error initializing workflow: {e}")
        return
    
    # Initialize conversation state
    state: ChatState = {
        "messages": [],
        "customer_context": None,
        "current_agent": "orchestrator",
        "intent": None,
        "offers_made": [],
        "cancellation_reason": None,
        "conversation_ended": False
    }
    
    # Initial greeting
    initial_greeting = """Hello! Welcome to TechFlow Electronics support. 
I'm here to help you today. 

Could you please provide your email address so I can look up your account?"""
    print(format_agent_message(initial_greeting, "Greeter"))
    
    while True:
        try:
            # Get user input
            user_input = input("👤 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ["quit", "exit"]:
                print("\nThank you for contacting TechFlow Electronics. Goodbye!")
                break
            
            if user_input.lower() == "reset":
                state = {
                    "messages": [],
                    "customer_context": None,
                    "current_agent": "orchestrator",
                    "intent": None,
                    "offers_made": [],
                    "cancellation_reason": None,
                    "conversation_ended": False
                }
                print("\n--- Conversation Reset ---\n")
                print(format_agent_message(initial_greeting, "Greeter"))
                continue
            
            # Add user message to state
            state["messages"].append(HumanMessage(content=user_input))
            
            # Run workflow
            result = workflow.invoke(state)
            
            # Update state with result
            state.update(result)
            
            # Display agent response
            if result.get("messages"):
                last_response = result["messages"][-1]
                if isinstance(last_response, AIMessage):
                    agent_name = result.get("current_agent", "Agent").title()
                    print(format_agent_message(last_response.content, agent_name))
            
            # Check if conversation ended
            if result.get("conversation_ended"):
                print("\n--- Conversation Ended ---")
                cont = input("Start a new conversation? (y/n): ").strip().lower()
                if cont == "y":
                    state = {
                        "messages": [],
                        "customer_context": None,
                        "current_agent": "orchestrator",
                        "intent": None,
                        "offers_made": [],
                        "cancellation_reason": None,
                        "conversation_ended": False
                    }
                    print("\n--- New Conversation ---\n")
                    print(format_agent_message(initial_greeting, "Greeter"))
                else:
                    print("\nThank you for contacting TechFlow Electronics. Goodbye!")
                    break
                    
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'reset' to start over.\n")


def run_test_scenario(scenario_name: str, messages: list):
    """Run a predefined test scenario."""
    load_dotenv()
    
    print(f"\n{'='*60}")
    print(f"  Test Scenario: {scenario_name}")
    print(f"{'='*60}\n")
    
    workflow = get_workflow()
    
    state: ChatState = {
        "messages": [],
        "customer_context": None,
        "current_agent": "orchestrator",
        "intent": None,
        "offers_made": [],
        "cancellation_reason": None,
        "conversation_ended": False
    }
    
    for user_message in messages:
        print(f"👤 Customer: {user_message}")
        
        state["messages"].append(HumanMessage(content=user_message))
        result = workflow.invoke(state)
        state.update(result)
        
        if result.get("messages"):
            last_response = result["messages"][-1]
            if isinstance(last_response, AIMessage):
                agent = result.get("current_agent", "Agent").title()
                print(f"🤖 {agent}: {last_response.content}\n")
        
        if result.get("conversation_ended"):
            break
    
    print(f"\n{'='*60}")
    print(f"  End of Scenario: {scenario_name}")
    print(f"{'='*60}\n")


# Predefined test scenarios from the assignment
TEST_SCENARIOS = {
    "money_problems": [
        "hi, my email is sarah.j@email.com",
        "hey can't afford the $13/month care+ anymore, need to cancel"
    ],
    "phone_problems": [
        "hello, I'm john.smith@email.com",
        "this phone keeps overheating, want to return it and cancel everything"
    ],
    "questioning_value": [
        "hi there, email is mike.wilson@email.com",
        "paying for care+ but never used it, maybe just get rid of it?"
    ],
    "technical_help": [
        "hey, my email is emily.b@email.com",
        "my phone won't charge anymore, tried different cables"
    ],
    "billing_question": [
        "hi, david.lee@email.com here",
        "got charged $15.99 but thought care+ was $12.99, what's the extra?"
    ]
}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TechFlow Customer Support Chat")
    parser.add_argument(
        "--test", 
        choices=list(TEST_SCENARIOS.keys()) + ["all"],
        help="Run a test scenario instead of interactive chat"
    )
    parser.add_argument(
        "--build-vectorstore",
        action="store_true",
        help="Build/rebuild the RAG vectorstore"
    )
    
    args = parser.parse_args()
    
    if args.build_vectorstore:
        print("Building vectorstore...")
        load_dotenv()
        from src.rag.vectorstore import create_vectorstore
        create_vectorstore(force_rebuild=True)
        print("Done!")
    elif args.test:
        if args.test == "all":
            for name, messages in TEST_SCENARIOS.items():
                run_test_scenario(name, messages)
        else:
            run_test_scenario(args.test, TEST_SCENARIOS[args.test])
    else:
        run_chat()
