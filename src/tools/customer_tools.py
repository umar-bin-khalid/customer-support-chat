"""
Customer data tools for the chat support system.
These tools interact with customer data from CSV files.
"""
import csv
import json
from datatime import datatime
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool

# Path to data directory

DATA_DIR = Path(__file__).parent.parent.parent / "data"

@tool
def get_customer_data(email: str) -> dict:
    csv_path = DATA_DIR / "customers.csv"

    if not csv_path:
        return {"error": "Customer database not found", "email": email}
    
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("email", "").lower() == email.lower():
                    return {
                        "found": True,
                        "customer_id": row.get("customer_id", ""),
                        "email": row.get("email", ""),
                        "phone": row.get("phone", ""),
                        "name": row.get("name", ""),
                        "plan_type": row.get("plan_type", ""),
                        "monthly_charge": row.get("monthly_charge", ""),
                        "signup_date": row.get("signup_date", ""),
                        "status": row.get("status", ""),
                        "total_spent": row.get("total_spent", ""),
                        "support_tickets_count": row.get("support_tickets_count", ""),
                        "account_health_score": row.get("account_health_score", ""),
                        "tenure_months": row.get("tenure_months", ""),
                        "tier": row.get("tier", ""),
                        "device": row.get("device", ""),
                        "purchase_date": row.get("purchase_date", "")
                    }

        return {"found": False, "error": f"No customer found with email: {email}"}

    except Exception as e:
        return {"error": f"Error reading customer data: {str(e)}"}
    

@tool
def update_customer_status(customer_id: str, action: str, reason: Optional[str] = None) -> dict:
    valid_actions = ["cancel", "pause", "downgrade", "retain", "upgrade"]

    if action.lower() not in valid_actions:
        return {
            "success": False,
            "error": f"Invalid action. Must be one of: {valid_actions}"
        }
    
    log_path = DATA_DIR / "customer_actions.log"

    try:
        timestamp = datatime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "customer_id": customer_id,
            "action": action.lower(),
            "reason": reason or "No reason provided",
            "status": "completed"
        }

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry), "\n")
        
        action_messages = {
            "cancel": f"Cancellation processed for customer {customer_id}. Service will end at billing cycle.",
            "pause": f"Account paused for customer {customer_id}. No charges during paused period.",
            "downgrade": f"Plan downgraded for customer {customer_id}. New rate effective next billing cycle.",
            "retain": f"Customer {customer_id} retained. Retention offer applied.",
            "upgrade": f"Plan upgraded for customer {customer_id}. New benefits active immediately."
        }

        return {
            "success": True,
            "customer_id": customer_id,
            "action": action.lower(),
            "message": action_messages[action.lower()],
            "timestamp": timestamp
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed tp process action: {str(e)}"
        }
    

# Export tools for easy import
__all__ = ["get_customer_data", "update_customer_status"]