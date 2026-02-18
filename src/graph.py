"""
LangGraph workflow for the customer support chat system.
Orchestrates the three agents and manages conversation state.
"""
import os
import time
from typing import Annotated, Literal, TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agents.orchestrator import create_orchestrator_agent, IntentClassification
from src.agents.retention import create_retention_agent
from src.agents.processor import create_processor_agent
from src.tools.customer_tools import get_customer_data


# Define the state schema
class ChatState(TypedDict):
    """State maintained throughout the conversation."""
    messages: Annotated[List[BaseMessage], add_messages]
    customer_context: Optional[dict]
    current_agent: str
    intent: Optional[str]
    offers_made: List[dict]
    cancellation_reason: Optional[str]
    conversation_ended: bool


class RetryLLM:
    """Wrapper around ChatGoogleGenerativeAI that retries on rate limit errors."""
    
    def __init__(self, llm: ChatGoogleGenerativeAI, max_retries: int = 3):
        self._llm = llm
        self._max_retries = max_retries
    
    def __getattr__(self, name):
        attr = getattr(self._llm, name)
        if name == "invoke":
            return self._retry_invoke
        return attr
    
    def _retry_invoke(self, *args, **kwargs):
        for attempt in range(self._max_retries):
            try:
                return self._llm.invoke(*args, **kwargs)
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 15 * (attempt + 1)
                    print(f"\n⏳ Rate limited. Waiting {wait}s before retry ({attempt+1}/{self._max_retries})...")
                    time.sleep(wait)
                else:
                    raise
        return self._llm.invoke(*args, **kwargs)


def create_llm() -> ChatGoogleGenerativeAI:
    """Create the LLM instance based on environment configuration."""
    provider = os.getenv("LLM_PROVIDER", "gemini")
    
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7
        )
    # Add other providers as needed
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def create_workflow():
    """
    Create the LangGraph workflow that orchestrates the agents.
    
    Flow:
    1. Orchestrator receives message
    2. If customer not identified, try to get email and look up
    3. Classify intent
    4. Route to appropriate agent:
       - cancellation → Retention Agent
       - technical/billing → End with referral message
       - retention failed → Processor Agent
    5. Loop until conversation ends
    """
    raw_llm = create_llm()
    llm = RetryLLM(raw_llm)  # Wrap with retry logic for rate limits
    
    # Create agents
    orchestrator = create_orchestrator_agent(llm)
    retention = create_retention_agent(llm)
    processor = create_processor_agent(llm)
    
    # Define node functions
    def orchestrator_node(state: ChatState) -> ChatState:
        """Handle initial greeting and routing."""
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""
        
        # Build chat history string
        chat_history = "\n".join([
            f"{'Customer' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}"
            for m in messages[:-1]
        ])
        
        customer_context = state.get("customer_context")
        
        # Check if we need to identify customer (look for email in message)
        if not customer_context or not customer_context.get("found"):
            import re
            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', last_message)
            if email_match:
                email = email_match.group()
                customer_data = get_customer_data.invoke({"email": email})
                if customer_data.get("found"):
                    customer_context = customer_data
                    state["customer_context"] = customer_context
        
        # Classify intent
        intent_result = orchestrator.classify_intent(last_message, chat_history)
        state["intent"] = intent_result.intent
        
        # Generate response
        response = orchestrator.get_response(last_message, chat_history, customer_context)
        
        # Determine next agent
        next_agent = orchestrator.determine_routing(intent_result)
        state["current_agent"] = next_agent
        
        # Add response to messages
        state["messages"] = [AIMessage(content=response)]
        
        return state
    
    def retention_node(state: ChatState) -> ChatState:
        """Handle retention conversations."""
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""
        
        chat_history = "\n".join([
            f"{'Customer' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}"
            for m in messages[:-1]
        ])
        
        customer_context = state.get("customer_context", {})
        offers_made = state.get("offers_made", [])
        
        # Check if customer insists on cancellation
        if retention.should_escalate_to_processor(last_message, len(offers_made)):
            state["current_agent"] = "processor"
            # Let processor handle the response
            return state
        
        # Detect cancellation reason if not already known
        if not state.get("cancellation_reason"):
            reason = retention.detect_cancellation_reason(last_message, chat_history)
            state["cancellation_reason"] = reason
        
        # Generate retention response
        response, offer = retention.get_response(
            last_message, 
            chat_history, 
            customer_context,
            offers_made
        )
        
        # Track offer if made
        if offer:
            offers_made.append(offer)
            state["offers_made"] = offers_made
        
        state["messages"] = [AIMessage(content=response)]
        
        return state
    
    def processor_node(state: ChatState) -> ChatState:
        """Handle cancellation processing."""
        messages = state["messages"]
        last_message = messages[-1].content if messages else ""
        
        chat_history = "\n".join([
            f"{'Customer' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}"
            for m in messages[:-1]
        ])
        
        customer_context = state.get("customer_context", {})
        
        # Process the cancellation
        if customer_context.get("customer_id"):
            reason = state.get("cancellation_reason", "customer_requested")
            processor.process_cancellation(
                customer_context["customer_id"],
                reason
            )
        
        # Generate confirmation response
        response = processor.get_response(
            last_message,
            chat_history,
            customer_context,
            "cancel"
        )
        
        state["messages"] = [AIMessage(content=response)]
        state["conversation_ended"] = True
        
        return state
    
    def route_to_external(state: ChatState) -> ChatState:
        """Handle routing to external departments (tech support, billing)."""
        intent = state.get("intent", "general")
        
        routing_messages = {
            "technical": """I can see you're having a technical issue with your device. 
Let me connect you with our Technical Support team who can help you troubleshoot this.

You can reach them at:
• Phone: 1-800-TECHFLOW (option 2)
• Live Chat: techflow.com/support
• Email: techsupport@techflow.com

They're available 24/7 and can help resolve your device issues. Is there anything else I can help with before I transfer you?""",
            "billing": """I understand you have a billing question. 
Let me connect you with our Billing team who can assist you with this.

You can reach them at:
• Phone: 1-800-TECHFLOW (option 3)
• Email: billing@techflow.com
• Online: Log into your account at techflow.com/billing

Is there anything else I can help with?"""
        }
        
        response = routing_messages.get(intent, "Let me transfer you to the right department.")
        state["messages"] = [AIMessage(content=response)]
        state["conversation_ended"] = True
        
        return state
    
    # Define routing logic
    def route_next(state: ChatState) -> Literal["retention", "processor", "external", "end"]:
        """Determine which node to go to next."""
        if state.get("conversation_ended"):
            return "end"
        
        current = state.get("current_agent", "orchestrator")
        intent = state.get("intent")
        
        if current == "retention":
            return "retention"
        elif current == "processor":
            return "processor"
        elif intent in ["technical", "billing"]:
            return "external"
        else:
            return "end"  # Stay in orchestrator for general inquiries
    
    # Build the graph
    workflow = StateGraph(ChatState)
    
    # Add nodes
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("retention", retention_node)
    workflow.add_node("processor", processor_node)
    workflow.add_node("external", route_to_external)
    
    # Set entry point
    workflow.set_entry_point("orchestrator")
    
    # Add edges
    workflow.add_conditional_edges(
        "orchestrator",
        route_next,
        {
            "retention": "retention",
            "processor": "processor",
            "external": "external",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "retention",
        lambda s: "processor" if s.get("current_agent") == "processor" else "end",
        {
            "processor": "processor",
            "end": END
        }
    )
    
    workflow.add_edge("processor", END)
    workflow.add_edge("external", END)
    
    return workflow.compile()


# Singleton workflow instance
_workflow = None

def get_workflow():
    """Get or create the workflow instance."""
    global _workflow
    if _workflow is None:
        _workflow = create_workflow()
    return _workflow
