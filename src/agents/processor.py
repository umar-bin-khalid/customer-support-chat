"""
Agent 3: The Processor
Responsibilities:
- Execute cancellations when retention has failed
- Update customer accounts
- Handle the "paperwork"
- Confirm cancellation details with customer
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from src.tools.customer_tools import update_customer_status


PROCESSOR_SYSTEM_PROMPT = """You are The Processor, handling service cancellations for TechFlow Electronics.

A customer has decided to cancel their service after speaking with our retention team. Your job is to:
1. Confirm their decision professionally
2. Process the cancellation
3. Explain what happens next
4. Leave a positive final impression

CANCELLATION PROCESS:
1. Confirm: "Just to confirm, you'd like to cancel your {plan_name}?"
2. Process: Use the update_customer_status tool with action="cancel"
3. Explain timeline:
   - Service continues until end of current billing cycle
   - No further charges after cancellation
   - They can reactivate anytime within 30 days
4. Close warmly: Thank them for being a customer, invite them back

KEY MESSAGES:
- "We're sorry to see you go, but we understand"
- "Your service will remain active until {end_date}"
- "You won't be charged going forward"
- "You're always welcome back - reactivation is easy"

IMPORTANT:
- Never try to retain at this stage - that ship has sailed
- Be professional and respectful
- Make the exit as smooth as the entry should have been
- Process pausees/downgrades the same way if that's the final decision

Available tools:
- update_customer_status: Process the cancellation/pause/downgrade

Customer information:
{customer_context}

Final decision: {final_decision}

Conversation history:
{chat_history}
"""


class ProcessorAgent:
    """
    The Processor - handles final cancellation processing.
    """
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.tools = [update_customer_status]
        self.llm_with_tools = llm.bind_tools(self.tools)
        
    def process_cancellation(self, customer_id: str, reason: str) -> dict:
        """
        Process a cancellation request.
        """
        result = update_customer_status.invoke({
            "customer_id": customer_id,
            "action": "cancel",
            "reason": reason
        })
        return result
    
    def process_pause(self, customer_id: str, reason: str) -> dict:
        """
        Process a pause request.
        """
        result = update_customer_status.invoke({
            "customer_id": customer_id,
            "action": "pause",
            "reason": reason
        })
        return result
    
    def process_downgrade(self, customer_id: str, reason: str) -> dict:
        """
        Process a downgrade request.
        """
        result = update_customer_status.invoke({
            "customer_id": customer_id,
            "action": "downgrade",
            "reason": reason
        })
        return result
    
    def get_response(
        self,
        message: str,
        chat_history: str,
        customer_context: dict,
        final_decision: str = "cancel"
    ) -> str:
        """
        Generate a processor response to finalize the cancellation.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", PROCESSOR_SYSTEM_PROMPT),
            ("human", "{message}")
        ])
        
        chain = prompt | self.llm_with_tools | StrOutputParser()
        
        response = chain.invoke({
            "message": message,
            "chat_history": chat_history,
            "customer_context": str(customer_context),
            "final_decision": final_decision
        })
        
        return response
    
    def generate_confirmation(self, customer_context: dict, action: str) -> str:
        """
        Generate a confirmation message after processing.
        """
        customer_name = customer_context.get("name", "Valued Customer")
        plan = customer_context.get("plan", "your plan")
        
        confirmations = {
            "cancel": f"""
Your cancellation has been processed, {customer_name}.

Here's what you need to know:
• Your {plan} coverage remains active until the end of your billing cycle
• You won't be charged for the next billing period
• You can reactivate anytime within 30 days at the same rate

Thank you for being a TechFlow customer. We hope to see you again!

Is there anything else I can help you with today?
""",
            "pause": f"""
Your subscription has been paused, {customer_name}.

Here's what you need to know:
• Your {plan} coverage is now on hold
• You won't be charged during the pause period
• Your benefits will automatically resume at the end of the pause
• You can resume early anytime by contacting us

Thank you for giving us a chance to earn your continued business!

Is there anything else I can help you with today?
""",
            "downgrade": f"""
Your plan has been downgraded, {customer_name}.

Here's what you need to know:
• Your new plan takes effect next billing cycle
• You'll see the reduced rate on your next bill
• Your current coverage continues until the changeover

Thank you for staying with TechFlow!

Is there anything else I can help you with today?
"""
        }
        
        return confirmations.get(action, confirmations["cancel"])


def create_processor_agent(llm: ChatGoogleGenerativeAI) -> ProcessorAgent:
    """Factory function to create a processor agent."""
    return ProcessorAgent(llm)
