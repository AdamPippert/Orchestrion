# Orchestrion Agent Nodes

A comprehensive node package for building AI agent workflows in Orchestrion/Ryven.

## Overview

This package provides visual nodes for:
- **LLM Integration**: Connect to cloud and local LLM providers
- **Agent Definition**: Define agent roles, personas, and capabilities
- **Guardrails**: Safety, validation, and cost controls
- **Routing**: Intelligent model selection and load balancing
- **Memory**: RAG, vector stores, and context management
- **Observability**: Metrics, tracing, and debugging

## Supported Providers

### Cloud Providers
- **OpenAI**: GPT-4o, GPT-4o-mini, o1, o1-mini
- **Anthropic**: Claude Opus 4, Claude Sonnet 4, Claude 3.5 Haiku

### Local Runtimes
- **vLLM**: High-performance local inference server
- **RamaLama**: Container-based LLM runtime (Red Hat)
- **Ollama**: Easy local model deployment
- **llama.cpp**: CPU/GPU inference server

All local runtimes expose OpenAI-compatible APIs, enabling unified access.

## Node Categories

### 🤖 Model Nodes (`models/`)

| Node | Description |
|------|-------------|
| **Chat** | Send chat completion requests |
| **Chat Stream** | Stream responses in real-time |
| **System Prompt** | Create system messages |
| **User Message** | Create user messages |
| **Message List** | Combine messages into conversations |
| **Embed Text** | Generate embeddings |
| **Similarity** | Calculate cosine similarity |
| **Top-K Similar** | Find most similar items |

**Provider Nodes:**
- `OpenAI Provider` - Quick OpenAI setup
- `Anthropic Provider` - Quick Claude setup
- `Local Provider` - vLLM, RamaLama, Ollama, llama.cpp
- `Provider Config` - Full provider configuration

### 👤 Agent Nodes (`agents/`)

| Node | Description |
|------|-------------|
| **Agent** | Define complete agent with role, persona, tools |
| **Role** | Define agent responsibilities |
| **Persona** | Define personality and communication style |
| **Capability** | Define specific agent capabilities |
| **Tool Definition** | Create tools for agents |
| **Tool Registry** | Collect tools for an agent |
| **Tool Executor** | Execute tool calls from LLM |
| **Python Tool** | Create tools from Python code |
| **Agent State** | Manage persistent agent state |
| **Agent Memory** | Short-term conversation memory |
| **Conversation History** | Multi-turn chat history |

**Coordination Patterns:**
| Node | Description |
|------|-------------|
| **Supervisor** | Hierarchical agent coordination with delegation |
| **Team** | Team of agents with shared goals |
| **Debate** | Multi-agent debate/deliberation pattern |
| **Consensus** | Reach agreement among multiple agents |
| **Delegator** | Smart task delegation based on capabilities |
| **Handoff** | Clean handoff between agents |
| **Message Bus** | Pub/sub inter-agent communication |

**Reasoning Patterns:**
| Node | Description |
|------|-------------|
| **ReAct** | Reasoning + Acting interleaved loop |
| **Plan & Execute** | Create plan, then execute steps |
| **Chain of Thought** | Step-by-step reasoning |
| **Reflection** | Self-reflection and improvement |
| **Tree of Thoughts** | Explore multiple reasoning paths |
| **Self-Ask** | Decompose complex questions |

### 🛡️ Guardrail Nodes (`guardrails/`)

**Validation:**
| Node | Description |
|------|-------------|
| **Input Validator** | Validate inputs before LLM |
| **Output Validator** | Validate LLM outputs |
| **Content Filter** | Filter unsafe content |
| **PII Detector** | Detect and redact PII |

**Limits:**
| Node | Description |
|------|-------------|
| **Token Limit** | Enforce token usage limits |
| **Cost Limit** | Enforce spending limits |
| **Rate Limit** | Enforce request rate limits |
| **Timeout** | Enforce operation timeouts |

**Approval:**
| Node | Description |
|------|-------------|
| **Human Approval** | Require human approval |
| **Confidence Gate** | Route by confidence score |
| **Approval Queue** | Manage pending approvals |

**Audit:**
| Node | Description |
|------|-------------|
| **Audit Log** | Log all actions |
| **Compliance Check** | Check against policies |
| **Policy** | Define compliance policies |

### 🔀 Routing Nodes (`routing/`)

**Model Routing:**
| Node | Description |
|------|-------------|
| **Model Router** | Weighted cost/speed/quality routing |
| **Cost-Optimized Router** | Route to cheapest model |
| **Speed-Optimized Router** | Route to fastest model |
| **Quality Router** | Route to highest quality model |

**Semantic Routing:**
| Node | Description |
|------|-------------|
| **Semantic Router** | Route by content similarity |
| **Intent Classifier** | Classify user intent |
| **Topic Router** | Route by detected topic |

**Load Balancing:**
| Node | Description |
|------|-------------|
| **Load Balancer** | Distribute across providers |
| **Fallback Chain** | Try providers in order |
| **Retry** | Retry with exponential backoff |
| **Circuit Breaker** | Prevent cascading failures |

**Orchestration:**
| Node | Description |
|------|-------------|
| **Parallel** | Execute in parallel |
| **Sequence** | Execute in sequence |
| **Conditional** | Branch on condition |
| **Loop** | Iterate over collection |
| **Map** | Apply operation to each item |
| **Reduce** | Aggregate to single value |

**Workflow & Chains:**
| Node | Description |
|------|-------------|
| **LLM Chain** | Sequential LLM call chaining |
| **Transform Chain** | Transform data between LLM calls |
| **Router Chain** | Route to different chains by input |
| **Workflow** | Define reusable workflow templates |
| **Workflow Step** | Define individual workflow steps |
| **Workflow Runner** | Execute workflow definitions |
| **Pipeline** | Data processing pipeline with stages |
| **State Machine** | Finite state machine for complex flows |
| **Prompt Chain** | Chain prompts with variable substitution |

### 🧠 Memory Nodes (`memory/`)

**Vector Store:**
| Node | Description |
|------|-------------|
| **Vector Store** | In-memory vector database |
| **Document Chunker** | Split documents for embedding |
| **Retrieval** | Retrieve relevant context |
| **Hybrid Search** | Combine vector + keyword search |

**Context:**
| Node | Description |
|------|-------------|
| **Context Window** | Manage context limits |
| **Context Compressor** | Compress context |
| **Relevance Filter** | Filter by relevance |

**Knowledge:**
| Node | Description |
|------|-------------|
| **Knowledge Base** | Structured fact storage |
| **Fact Store** | Key-value fact storage |
| **Entity Memory** | Track mentioned entities |

### 📊 Observability Nodes (`observability/`)

**Metrics:**
| Node | Description |
|------|-------------|
| **Token Counter** | Track token usage |
| **Cost Tracker** | Track spending |
| **Latency Monitor** | Track response times |
| **Metrics Aggregator** | Combine all metrics |

**Tracing:**
| Node | Description |
|------|-------------|
| **Trace** | Create execution traces |
| **Span** | Add spans to traces |
| **Trace Viewer** | View trace data |

**Debugging:**
| Node | Description |
|------|-------------|
| **Debug** | Log debug output |
| **Inspector** | Inspect data structure |
| **Breakpoint** | Conditional breakpoints |

### 🔄 Shared State Nodes (`state/`)

The shared state layer enables "state as source of truth" coordination:

| Node | Description |
|------|-------------|
| **Shared State** | Central key-value store for all agents |
| **Plan Artifact** | First-class plans/tasks/results as objects |
| **State Watcher** | Reactive triggers on state changes |
| **Event Log** | Append-only log (chat as side effect) |
| **Agent Sync** | Sync agent state with shared store |

**Architecture:**
```
┌─────────────────────────────────────────┐
│         SHARED STATE STORE              │
│  Plans │ Artifacts │ Agent States       │
└────────────┬────────────────────────────┘
             │ read/write
    ┌────────┼────────┬────────┐
    │ Agent A│ Agent B│ Agent C│
    └────────┴────────┴────────┘
             │
    ┌────────▼────────┐
    │   Event Log     │  ← chat is just a log
    └─────────────────┘
```

## Example Workflows

### Basic Chat
```
[OpenAI Provider] → [System Prompt] ─┐
                                     ├→ [Message List] → [Chat] → [Output]
               [User Message] ───────┘
```

### RAG Pipeline
```
[Document] → [Chunker] → [Embed] → [Vector Store]
                                          ↓
[Query] → [Embed] → [Search] → [Retrieval] → [Context] → [Chat] → [Output]
```

### Multi-Model Routing
```
[Providers List] ─┬→ [Model Router] → [Chat] → [Output]
[Task Description]┘
```

### Guardrailed Agent
```
[Input] → [Input Validator] → [Content Filter] → [Token Limit]
    ↓
[Chat] → [Output Validator] → [Cost Tracker] → [Audit Log] → [Output]
```

## Installation

The agent_nodes package is included with Orchestrion. To use it:

1. Start Ryven/Orchestrion
2. Load the `agent_nodes` package from the packages menu
3. Start building your agent workflows!

## Dependencies

Required Python packages:
- `openai>=1.0.0` - For OpenAI and compatible APIs
- `anthropic` - For Anthropic Claude API

Optional:
- `numpy` - For optimized vector operations

## Configuration

### Environment Variables
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key

### Provider Configuration
Configure providers programmatically or through the Provider Config node:

```python
from agent_nodes import create_openai_provider, ProviderRegistry

provider = create_openai_provider(
    api_key="sk-...",
    model="gpt-4o"
)
ProviderRegistry.register("openai", provider, set_default=True)
```

## Best Practices

### Cost Control
1. Use the **Cost-Optimized Router** for non-critical tasks
2. Set **Token Limits** and **Cost Limits** as guardrails
3. Monitor with **Cost Tracker** and **Metrics Aggregator**

### Safety
1. Always use **Input Validator** and **Content Filter**
2. Add **Human Approval** for sensitive actions
3. Enable **Audit Log** for compliance

### Performance
1. Use **Speed-Optimized Router** for real-time applications
2. Prefer local models (vLLM, Ollama) for low latency
3. Implement **Circuit Breaker** for reliability

### Debugging
1. Add **Debug** nodes at key points
2. Use **Trace** and **Span** for full execution visibility
3. Check **Inspector** for data structure issues

## Contributing

Contributions welcome! Areas of interest:
- Additional provider integrations
- Custom GUI widgets
- Performance optimizations
- Documentation improvements

## License

MIT License - See LICENSE file for details.
