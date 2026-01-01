"""
Orchestrion Agent Nodes Package

A comprehensive collection of nodes for building AI agent workflows,
including LLM integration, agent orchestration, guardrails, routing,
memory management, and observability.

Supports:
- Cloud providers: OpenAI, Anthropic
- Local runtimes: vLLM, RamaLama, Ollama, llama.cpp
- Agent patterns: Role-based, tool-calling, multi-agent
- Guardrails: Validation, rate limiting, cost control, approval gates
- Routing: Cost/speed/quality optimization, semantic routing, load balancing
- Memory: Vector stores, context management, knowledge bases
- Observability: Metrics, tracing, debugging
"""

from ryven.node_env import on_gui_load, export_nodes

# Import all node modules
from .models.chat import chat_nodes
from .models.embeddings import embedding_nodes
from .models.providers import provider_nodes

from .agents.definition import definition_nodes
from .agents.tools import tool_nodes
from .agents.state import state_nodes

from .guardrails.validation import validation_nodes
from .guardrails.limits import limit_nodes
from .guardrails.approval import approval_nodes
from .guardrails.audit import audit_nodes

from .routing.model_router import model_router_nodes
from .routing.semantic_router import semantic_router_nodes
from .routing.load_balancer import load_balancer_nodes
from .routing.orchestration import orchestration_nodes

from .memory.vector_store import vector_store_nodes
from .memory.context import context_nodes
from .memory.knowledge import knowledge_nodes

from .observability.metrics import metrics_nodes
from .observability.tracing import tracing_nodes
from .observability.debugging import debugging_nodes


# Collect all node types
all_nodes = [
    # Model nodes
    *chat_nodes,
    *embedding_nodes,
    *provider_nodes,

    # Agent nodes
    *definition_nodes,
    *tool_nodes,
    *state_nodes,

    # Guardrail nodes
    *validation_nodes,
    *limit_nodes,
    *approval_nodes,
    *audit_nodes,

    # Routing nodes
    *model_router_nodes,
    *semantic_router_nodes,
    *load_balancer_nodes,
    *orchestration_nodes,

    # Memory nodes
    *vector_store_nodes,
    *context_nodes,
    *knowledge_nodes,

    # Observability nodes
    *metrics_nodes,
    *tracing_nodes,
    *debugging_nodes,
]

# Export all nodes
export_nodes(all_nodes)


# GUI loading (when available)
@on_gui_load
def load_gui():
    # GUI widgets would be loaded here if we had custom widgets
    # from . import gui
    pass
