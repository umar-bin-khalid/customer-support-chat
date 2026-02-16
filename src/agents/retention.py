"""
Agent 2: The Problem Solver (Retention Specialist)
Responsibilities:
- Handle customers who want to cancel
- Understand their reasons
- Offer solutions (discounts, pauses, downgrades)
- Only give up if customer insists on cancellation
"""
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from src.tools.customer_tools import get_customer_data
from src.tools.retention_tools import calculate_retention_offer
from src.rag.vectorstore import search_policies


RETENTION_SYSTEM_PROMPT = """You are The Problem Solver, a skilled retention specialist for TechFlow Electronics.

Your goal is to RETAIN customers who want to cancel by genuinely helping them. You're not pushy - you're helpful.

YOUR APPROACH:
1. EMPATHIZE: Acknowledge their frustration or concern first
2. DISCOVER: Understand the real reason they want to cancel
3. SOLVE: Offer relevant solutions based on their specific situation
4. CONFIRM: Only proceed with cancellation if they insist after solutions offered

CANCELLATION REASONS & SOLUTIONS:
- Cost concerns → Offer discount, payment pause, or downgrade to cheaper plan
- Not using it → Educate on value, offer pause, remind of coverage benefits
- Phone problems → Offer replacement/repair FIRST, then discuss retention
- Competitor offer → Match or beat the offer if within guidelines
- Moving/temporary → Offer pause, promise easy reactivation

EMPATHY PHRASES:
- "I completely understand - let me see what I can do for you"
- "That's a valid concern, and I want to help find a solution"
- "I hear you, and I appreciate you giving us a chance to make things right"

RULES:
- Maximum 3 offers before accepting cancellation
- Never be pushy or guilt-trip
- If they say "just cancel" twice, respect their decision and route to processor
- Always use customer data to personalize offers
- Query policies when you need accurate information

Available tools:
- get_customer_data: Look up customer information
- calculate_retention_offer: Get personalized retention offers

Customer information:
{customer_context}

Offers already made this conversation:
{offers_made}

Conversation history:
{chat_history}
"""


class RetentionAgent:
    """
    The Problem Solver - handles retention conversations.
    """
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.tools = [get_customer_data, calculate_retention_offer]
        self.llm_with_tools = llm.bind_tools(self.tools)
        self.offers_made = []
        
    def detect_cancellation_reason(self, message: str, chat_history: str) -> str:
        """
        Detect the primary reason for cancellation from the conversation.
        """
        detection_prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze the customer message and conversation to identify the cancellation reason.
            
Categories:
- cost: Financial concerns, too expensive, can't afford
- not_using: Not using the service, don't need it
- phone_issues: Device problems, technical frustration
- competitor: Found better deal elsewhere
- value: Doesn't see the value, questioning worth
- temporary: Moving, traveling, temporary situation
- other: Unclear or other reason

Respond with just the category name."""),
            ("human", "Message: {message}\n\nHistory: {chat_history}")
        ])
        
        chain = detection_prompt | self.llm | StrOutputParser()
        reason = chain.invoke({"message": message, "chat_history": chat_history})
        return reason.strip().lower()
    
    def should_escalate_to_processor(self, message: str, offers_count: int) -> bool:
        """
        Determine if we should give up and route to cancellation processor.
        """
        cancellation_insistence_phrases = [
            "just cancel",
            "cancel it now",
            "want to cancel",
            "please cancel",
            "stop the service",
            "end my subscription",
            "i said cancel",
            "not interested",
            "no thanks just cancel"
        ]
        
        message_lower = message.lower()
        
        # Check for strong cancellation intent
        insists = any(phrase in message_lower for phrase in cancellation_insistence_phrases)
        
        # Route to processor if: insists AND (offers made >= 2 OR explicit refusal)
        if insists and offers_count >= 2:
            return True
        
        # Explicit refusal of all offers
        refusal_phrases = ["no", "not interested", "don't want", "already decided"]
        if insists and any(phrase in message_lower for phrase in refusal_phrases):
            return True
            
        return False
    
    def get_policy_context(self, query: str) -> str:
        """
        Get relevant policy information for the conversation.
        """
        try:
            docs = search_policies(query, k=2)
            context = "\n\n".join([doc.page_content for doc in docs])
            return context[:1000]  # Limit context size
        except Exception as e:
            return f"Policy lookup unavailable: {e}"
    
    def get_response(
        self, 
        message: str, 
        chat_history: str, 
        customer_context: dict,
        offers_made: list
    ) -> tuple[str, Optional[dict]]:
        """
        Generate a retention response.
        
        Returns:
            Tuple of (response_text, offer_made_or_none)
        """
        # Get relevant policy context
        policy_context = self.get_policy_context(message)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", RETENTION_SYSTEM_PROMPT + f"\n\nRelevant policy information:\n{policy_context}"),
            ("human", "{message}")
        ])
        
        chain = prompt | self.llm_with_tools | StrOutputParser()
        
        response = chain.invoke({
            "message": message,
            "chat_history": chat_history,
            "customer_context": str(customer_context),
            "offers_made": str(offers_made) if offers_made else "None yet"
        })
        
        # TODO: Parse response to extract any offers made
        # For now, return response without offer tracking
        return response, None


def create_retention_agent(llm: ChatGoogleGenerativeAI) -> RetentionAgent:
    """Factory function to create a retention agent."""
    return RetentionAgent(llm)
