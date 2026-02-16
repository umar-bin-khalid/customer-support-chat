# TechFlow Customer Support Chat System

A multi-agent AI system for handling customer support conversations, built with LangChain and LangGraph.

## Overview

This system handles customer support for TechFlow Electronics' Care+ insurance product. It uses three specialized agents:

1. **Orchestrator (The Greeter)** - Identifies customers, classifies intent, routes to specialists
2. **Retention Agent (The Problem Solver)** - Handles cancellation requests, offers solutions
3. **Processor Agent** - Executes cancellations when retention fails

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Customer Message                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR AGENT                        │
│  • Identify customer (email lookup)                          │
│  • Classify intent                                           │
│  • Route to appropriate agent                                │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   RETENTION     │ │   TECHNICAL     │ │    BILLING      │
│    AGENT        │ │   (Route Out)   │ │   (Route Out)   │
│                 │ └─────────────────┘ └─────────────────┘
│ • Empathize     │
│ • Offer solutions│
│ • Use tools     │
└─────────────────┘
          │
          ▼ (if customer insists)
┌─────────────────────────────────────────────────────────────┐
│                    PROCESSOR AGENT                           │
│  • Confirm cancellation                                      │
│  • Update customer status                                    │
│  • Generate confirmation                                     │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Multi-Agent Orchestration**: LangGraph workflow with state management
- **Tool Integration**: Real data processing with customer CSV and retention rules JSON
- **RAG System**: Policy document retrieval using FAISS vector store
- **Intent Classification**: Automatic routing based on customer intent
- **Retention Logic**: Personalized offers based on customer tier and cancellation reason

## Quick Start

### 1. Setup

```bash
cd customer-support-chat

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 2. Get Google Gemini API Key (Free)

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create an API key
3. Add to your `.env` file:
   ```
   GOOGLE_API_KEY=your_key_here
   ```

### 3. Build Vectorstore (First time only)

```bash
python main.py --build-vectorstore
```

### 4. Run Interactive Chat

```bash
python main.py
```

### 5. Run Test Scenarios

```bash
# Run all 5 test scenarios
python main.py --test all

# Run specific scenario
python main.py --test money_problems
python main.py --test phone_problems
python main.py --test questioning_value
python main.py --test technical_help
python main.py --test billing_question
```

## Project Structure

```
customer-support-chat/
├── main.py                 # Entry point + CLI + test runner
├── requirements.txt        # Dependencies
├── .env.example           # Environment template
├── data/
│   ├── customers.csv      # Customer database
│   ├── retention_rules.json # Retention offer rules
│   └── policies/          # RAG documents
│       ├── return_policy.md
│       ├── care_plus_benefits.md
│       └── tech_support_guide.md
├── src/
│   ├── graph.py           # LangGraph workflow
│   ├── agents/
│   │   ├── orchestrator.py # Agent 1: Greeter
│   │   ├── retention.py    # Agent 2: Problem Solver
│   │   └── processor.py    # Agent 3: Cancellation handler
│   ├── tools/
│   │   ├── customer_tools.py  # get_customer_data, update_status
│   │   └── retention_tools.py # calculate_retention_offer
│   └── rag/
│       └── vectorstore.py  # FAISS setup + retrieval
└── tests/
    └── test_scenarios.py   # Automated tests
```

## Test Scenarios

| # | Scenario | Input | Expected Behavior |
|---|----------|-------|-------------------|
| 1 | Money Problems | "can't afford $13/month care+" | Offer discount/pause before canceling |
| 2 | Phone Problems | "phone overheating, want to cancel" | Offer replacement first |
| 3 | Questioning Value | "never used care+, maybe cancel" | Explain value, offer cheaper options |
| 4 | Technical Help | "phone won't charge" | Route to tech support (NOT retention) |
| 5 | Billing Question | "charged $15.99 instead of $12.99" | Route to billing (NOT retention) |

## Tools

### get_customer_data(email)
Looks up customer information from `customers.csv`:
- Customer ID, name, email, phone
- Tier (gold/silver/standard)
- Monthly payment, tenure
- Plan details, status

### calculate_retention_offer(customer_tier, reason)
Generates personalized retention offers from `retention_rules.json`:
- Discount offers based on tier
- Pause options
- Downgrade suggestions
- Special offers by cancellation reason

### update_customer_status(customer_id, action, reason)
Processes account changes:
- Actions: cancel, pause, downgrade, retain
- Logs to `customer_actions.log`

## Customization

### Replace Sample Data
Download the actual data files from the assignment and replace:
- `data/customers.csv`
- `data/retention_rules.json`
- `data/policies/*.md`

Then rebuild the vectorstore:
```bash
python main.py --build-vectorstore
```

### Switch LLM Provider
Edit `.env`:
```
LLM_PROVIDER=groq  # or openai, ollama
GROQ_API_KEY=your_key
```

## License

MIT
