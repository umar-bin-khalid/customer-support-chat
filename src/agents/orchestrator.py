"""
Agent 1: The Greeter & Orchestrator
Responsibilities:
- Identify the customer
- Classify intent (cancellation, tech support, billing, general inquiry)
- Route to the appropriate specialist agent
"""
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.tools.customer_tools import get_customer_data
from src.rag.vectorstore import search_policies


class IntentClassification(BaseModel):
    """Structured output for intent classification."""
    intent: Literal["cancellation", "technical", "billing", "general"] = Field(
        description="The classified intent of the customer"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1"
    )
    reasoning: str = Field(
        description="Brief explanation of why this intent was chosen"
    )


ORCHESTRATOR_SYSTEM_PROMPT = """You are The Greeter, the first point of contact for TechFlow Electronics customer support.

Your responsibilities:
1. Warmly greet the customer
2. Identify who they are (get their email to look up their account)
3. Understand what they need help with
4. Route them to the right specialist

INTENT CLASSIFICATION:
- "cancellation": Customer wants to cancel, downgrade, or is unhappy with their service
  Keywords: cancel, stop, remove, too expensive, not worth it, don't need, get rid of
- "technical": Device issues, troubleshooting, repairs
  Keywords: broken, not working, won't charge, overheating, slow, freezing, crashing
- "billing": Payment questions, charges, invoices
  Keywords: charged, payment, invoice, bill, refund, price
- "general": All other inquiries

IMPORTANT RULES:
- Always be warm and helpful
- Get customer email early to look up their account
- Never make assumptions about intent - ask clarifying questions if unsure
- If someone mentions BOTH a technical issue AND wanting to cancel, route to technical first
- Billing questions should NOT go to retention unless they explicitly want to cancel

Available tools:
- get_customer_data: Look up customer by email

Current customer context:
{customer_context}

Conversation history:
{chat_history}
"""


class OrchestratorAgent:
    """
    The Greeter & Orchestrator agent.
    Handles initial customer contact and routes to appropriate specialist.
    """
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.tools = [get_customer_data]
        self.llm_with_tools = llm.bind_tools(self.tools)
        
    def classify_intent(self, message: str, context: str = "") -> IntentClassification:
        """
        Classify the customer's intent from their message.
        """
        classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """Classify the customer's intent into one of these categories:
- cancellation: Wants to cancel, stop, or remove service
- technical: Device issues, troubleshooting needed  
- billing: Payment or charge questions
- general: Other inquiries

Respond with JSON: {{"intent": "...", "confidence": 0.X, "reasoning": "..."}}"""),
            ("human", "Customer message: {message}\nContext: {context}")
        ])
        
        # Use structured output if available, otherwise parse manually
        chain = classification_prompt | self.llm | StrOutputParser()
        result = chain.invoke({"message": message, "context": context})
        
        # Parse the result (simplified - you may want more robust parsing)
        import json
        try:
            data = json.loads(result)
            return IntentClassification(**data)
        except:
            # Default to general if parsing fails
            return IntentClassification(
                intent="general",
                confidence=0.5,
                reasoning="Could not parse intent, defaulting to general"
            )
    
    def get_response(self, message: str, chat_history: str, customer_context: dict) -> str:
        """
        Generate a response from the orchestrator.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", ORCHESTRATOR_SYSTEM_PROMPT),
            ("human", "{message}")
        ])
        
        chain = prompt | self.llm_with_tools | StrOutputParser()
        
        response = chain.invoke({
            "message": message,
            "chat_history": chat_history,
            "customer_context": str(customer_context) if customer_context else "Not identified yet"
        })
        
        return response
    
    def should_identify_customer(self, customer_context: dict) -> bool:
        """Check if we still need to identify the customer."""
        return customer_context is None or not customer_context.get("found")
    
    def determine_routing(self, intent: IntentClassification) -> str:
        """
        Determine which agent should handle this conversation.
        
        Returns:
            Agent name: "retention", "processor", "technical", "billing", or "orchestrator"
        """
        if intent.intent == "cancellation":
            return "retention"
        elif intent.intent == "technical":
            return "technical"  # Could route to external system or end conversation with info
        elif intent.intent == "billing":
            return "billing"  # Could route to external system
        else:
            return "orchestrator"  # Keep handling general inquiries


def create_orchestrator_agent(llm: ChatGoogleGenerativeAI) -> OrchestratorAgent:
    """Factory function to create an orchestrator agent."""
    return OrchestratorAgent(llm)
