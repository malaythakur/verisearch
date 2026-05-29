# Agentic Research Search Engine - TypeScript SDK

Auto-generated TypeScript client SDK for the Agentic Research Search Engine API.

## Installation

```bash
npm install @agentic-research/sdk
```

## Usage

```typescript
import { Client } from '@agentic-research/sdk';

const client = new Client({ apiKey: 'your-api-key' });

// Search
const results = await client.search({ query: 'quantum computing advances', mode: 'hybrid' });

// Streaming answer
for await (const event of client.answer({ query: 'Explain quantum entanglement', stream: true })) {
  console.log(event);
}

// Deep research
const job = await client.research({ researchGoal: 'Compare transformer architectures for code generation' });
```
