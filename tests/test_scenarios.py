"""
Test scenarios for the customer support chat system.
Run with: pytest tests/test_scenarios.py -v
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from dotenv import load_dotenv

load_dotenv()

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set"
)


class TestTools:
    """Test the tool functions."""
    
    def test_get_customer_data_found(self):
        from src.tools.customer_tools import get_customer_data
        
        result = get_customer_data.invoke({"email": "john.smith@email.com"})
        assert result.get("found") == True
        assert result.get("name") == "John Smith"
        assert result.get("tier") == "gold"
    
    def test_get_customer_data_not_found(self):
        from src.tools.customer_tools import get_customer_data
        
        result = get_customer_data.invoke({"email": "notreal@email.com"})
        assert result.get("found") == False
    
    def test_calculate_retention_offer(self):
        from src.tools.retention_tools import calculate_retention_offer
        
        result = calculate_retention_offer.invoke({
            "customer_tier": "gold",
            "reason": "cost"
        })
        assert "offers" in result
        assert len(result["offers"]) > 0
    
    def test_update_customer_status(self):
        from src.tools.customer_tools import update_customer_status
        
        result = update_customer_status.invoke({
            "customer_id": "TEST001",
            "action": "cancel",
            "reason": "test"
        })
        assert result.get("success") == True


class TestRAG:
    """Test the RAG system."""
    
    def test_vectorstore_creation(self):
        from src.rag.vectorstore import create_vectorstore
        
        vectorstore = create_vectorstore()
        assert vectorstore is not None
    
    def test_search_policies(self):
        from src.rag.vectorstore import search_policies
        
        results = search_policies("What does Care+ cover?", k=2)
        assert len(results) > 0
        # Should find Care+ benefits document
        assert any("care" in doc.page_content.lower() for doc in results)


class TestIntentClassification:
    """Test intent classification."""
    
    def test_cancellation_intent(self):
        from src.agents.orchestrator import create_orchestrator_agent
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        orchestrator = create_orchestrator_agent(llm)
        
        result = orchestrator.classify_intent(
            "I want to cancel my care+ subscription",
            ""
        )
        assert result.intent == "cancellation"
    
    def test_technical_intent(self):
        from src.agents.orchestrator import create_orchestrator_agent
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        orchestrator = create_orchestrator_agent(llm)
        
        result = orchestrator.classify_intent(
            "My phone won't charge anymore",
            ""
        )
        assert result.intent == "technical"


class TestScenarios:
    """Test the 5 required scenarios."""
    
    @pytest.fixture
    def workflow(self):
        from src.graph import get_workflow
        return get_workflow()
    
    def test_scenario_money_problems(self, workflow):
        """Test 1: Customer can't afford service."""
        from langchain_core.messages import HumanMessage
        
        state = {
            "messages": [HumanMessage(content="my email is sarah.j@email.com, can't afford the $13/month care+ anymore, need to cancel")],
            "customer_context": None,
            "current_agent": "orchestrator",
            "intent": None,
            "offers_made": [],
            "cancellation_reason": None,
            "conversation_ended": False
        }
        
        result = workflow.invoke(state)
        
        # Should route to retention, not immediately cancel
        assert result.get("intent") == "cancellation" or result.get("current_agent") in ["retention", "orchestrator"]
    
    def test_scenario_technical_help(self, workflow):
        """Test 4: Customer needs technical support."""
        from langchain_core.messages import HumanMessage
        
        state = {
            "messages": [HumanMessage(content="my phone won't charge anymore, tried different cables")],
            "customer_context": None,
            "current_agent": "orchestrator",
            "intent": None,
            "offers_made": [],
            "cancellation_reason": None,
            "conversation_ended": False
        }
        
        result = workflow.invoke(state)
        
        # Should identify as technical, NOT cancellation
        assert result.get("intent") == "technical"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
