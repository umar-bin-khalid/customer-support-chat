"""
Retention offer calculation tools.
Uses retention_rules.json to generate personalized offers.
"""
import json
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _map_reason_to_category(reason: str) -> tuple[str, Optional[str]]:
    reason_lower = reason.lower().strip()

    # Financial/cost reasons
    if any(word in reason_lower for word in ["cost", "afford", "expensive", "money", "price", "financial"]):
        return ("financial_hardship", None)
    
    # Product/device issues
    if any(word in reason_lower for word in ["overheat", "hot", "heat"]):
        return ("product_issues", "overheating")
    if any(word in reason_lower for word in ["battery", "charge", "charging", "power"]):
        return ("product_issues", "battery_issues")
    if any(word in reason_lower for word in ["broken", "defect", "malfunction", "not working"]):
        return ("product_issues", "overheating")
    
    # Value questioning
    if any(word in reason_lower for word in ["value", "worth", "never used", "not using", "don't use"]):
        return ("service_value", None)

    # Default to financial hardhip
    return ("financial_harship", None)


def _map_tier_to_customer_type(tier: str) -> str:
    tier_lower = tier.lower().strip()

    tier_mapping = {
        "gold": "premium_customers",
        "premium": "premium_customers",
        "silver": "regular_customers",
        "regular": "regular_customers",
        "standard": "regular_customers",
        "basic": "new_customers",
        "new": "new_customers",
    }

    return tier_mapping.get(tier_lower, "regular_customers")


@tool("calculate_retention_offer", return_direct=False)
def calculate_retention_offer(customer_tier: str, reason: str) -> dict:
    """
    Generate personalized retention offers based on customer tier and cancellation reason.
    
    Args:
        customer_tier: Customer's tier level (e.g., "gold", "silver", "standard", "premium")
        reason: The reason for cancellation (e.g., "cost", "can't afford", "overheating", "not using")
        
    Returns:
        dict with available retention offers and recommendations
    """
    rules_path = DATA_DIR / "retention_rules.json"

    if not rules_path.exists():
        return _get_default_offers(customer_tier, reason)
    
    try:
        with open(rules_path, "r", encoding="utf-8") as f:
            rules = json.load(f)
        
        category, sub_category = _map_reason_to_category(reason)
        customer_type = _map_tier_to_customer_type(customer_tier)

        offers = []
        authorization_info = rules.get("authorization_levels", {})

        # Get offers based on category
        if category == "financial_hardship":
            # Financial hardship is organized by customer type
            category_data = rules.get("financial_hardship", {})
            raw_offers = category_data.get(customer_type, [])
            
            for offer in raw_offers:
                offers.append({
                    "type": offer.get("type"),
                    "description": offer.get("description"),
                    "details": _format_offer_details(offer),
                    "authorization": offer.get("authorization", "agent")
                })
        
        elif category == "product_issues":
            # Product issues organized by issue type
            category_data = rules.get("product_issues", {})
            raw_offers = category_data.get(sub_category, [])
            
            for offer in raw_offers:
                offers.append({
                    "type": offer.get("type"),
                    "description": offer.get("description"),
                    "details": _format_offer_details(offer),
                    "authorization": offer.get("authorization", "agent")
                })
        
        elif category == "service_value":
            # Service value - use care_plus_premium offers
            category_data = rules.get("service_value", {})
            raw_offers = category_data.get("care_plus_premium", [])
            
            for offer in raw_offers:
                if offer.get("type") == "explain_benefits":
                    # Special handling for benefits explanation
                    benefits = offer.get("benefits", [])
                    offers.append({
                        "type": "explain_benefits",
                        "description": "Explain the value of their current plan",
                        "details": "Benefits: " + "; ".join(benefits),
                        "authorization": "agent"
                    })
                else:
                    offers.append({
                        "type": offer.get("type"),
                        "description": offer.get("description", ""),
                        "details": _format_offer_details(offer),
                        "authorization": offer.get("authorization", "agent")
                    })
        
        agent_limits = authorization_info.get("agent", {})

        return {
            "customer_tier": customer_tier,
            "customer_type": customer_type,
            "reason": reason,
            "category": category,
            "sub_category": sub_category,
            "offers": offers,
            "recommendation": _get_recommendation(category, sub_category),
            "agent_can_authorize": {
                "max_discounts": agent_limits.get("max_discount_percentage", 25),
                "can_pause": agent_limits.get("can_pause", True),
                "can_downgrade": agent_limits.get("can_downgrade", True)
            }
        }

    except Exception as e:
        return {
            "error": f"",
            "customer_tier": customer_tier,
            "reason": reason,
            "offers": _get_default_offers(customer_tier, reason)["offers"]
        }
    

def _format_offer_details(offer: dict) -> str:
    """Format offer details into a readable string."""
    details = []
    
    if offer.get("percentage"):
        details.append(f"{offer['percentage']}% discount")
    if offer.get("duration_months"):
        details.append(f"for {offer['duration_months']} months")
    if offer.get("new_cost"):
        details.append(f"new cost: ${offer['new_cost']}/month")
    if offer.get("savings"):
        details.append(f"saves ${offer['savings']}/month")
    if offer.get("cost") == 0:
        details.append("free")
    if offer.get("new_plan"):
        details.append(f"switch to {offer['new_plan']}")
    if offer.get("refund_promise"):
        details.append(offer["refund_promise"])
    
    return " | ".join(details) if details else offer.get("description", "")


def _get_default_offers(tier: str, reason: str) -> dict:
    return {
        "customer_tier": tier,
        "reason": reason,
        "offers": [
            {
                "type": "discount",
                "description": "25% discount for 3 months",
                "details": "Reduced rate to help with costs",
                "authorization": "agent"
            },
            {
                "type": "pause",
                "description": "Pause subscription for upto 3 months",
                "details": "No charges during pause period",
                "authorization": "agent"
            },
            {
                "type": "downgrade",
                "description": "Switch to a lower-cost plan",
                "details": "Keep coverage at reduced price",
                "authorization": "agent"
            }
        ],
        "recommendation": "Start by understanding their concern. Offer pause for temporary issues, discount for cost concerns."
    }


def _get_recommendation(category: str, sub_category: Optional[str]) -> str:
    recommendations = {
        "financial_hardship": "Start with empathy about their financial situation. Offer the pause option first (no commitment), then discuss discounts if they prefer to keep service active.",
        "product_issues": {
            "overheating": "Apologize for the device issue. Offer free replacement immediately - this often resolves the cancellation. If they still want to cancel after replacement offered, don't push.",
            "battery_issues": "Offer free battery replacement first. This is usually a quick fix that saves the customer relationship."
        },
        "service_value": "Don't be defensive. Walk them through the specific benefits and their value. The explain_benefits offer lists concrete savings. If they are still unsure, offer the trial extension with refund promise."
    }

    if category == "product_issues" and sub_category:
        return recommendations.get("product_issues", {}).get(sub_category, recommendations["product_issues"]["overheating"])
    
    return recommendations.get(category, "Listen to their concerns first. Match the offer to their specific situation.")


# Export tols
__all__ = ["calculate_retention_offer"]