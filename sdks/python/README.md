# Agentic Research Search Engine - Python SDK

Auto-generated Python client SDK for the Agentic Research Search Engine API.

## Installation

```bash
pip install agentic-research-sdk
```

## Usage

```python
from agentic_research_sdk import Client

client = Client(api_key="your-api-key")

# Search
results = client.search(query="quantum computing advances", mode="hybrid")

# Streaming answer
async for event in client.answer(query="Explain quantum entanglement", stream=True):
    print(event)

# Deep research
job = client.research(research_goal="Compare transformer architectures for code generation")
```
