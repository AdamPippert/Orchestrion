"""
Shared State Store module.

Provides the infrastructure for shared-state-based agent coordination:
- SharedStateStore: Central state singleton
- SharedStateNode: Read/write shared state
- PlanArtifactNode: First-class plan objects
- StateWatcherNode: Reactive state watching
- EventLogNode: Append-only event log
- AgentSyncNode: Agent state synchronization
"""

from .shared_state import (
    # Core store
    SharedStateStore,
    get_shared_store,

    # Data classes
    StateChange,
    StateChangeType,
    PlanArtifact,
    ArtifactType,
    EventLogEntry,

    # Nodes
    SharedStateNode,
    PlanArtifactNode,
    StateWatcherNode,
    EventLogNode,
    AgentSyncNode,

    # Node list for export
    shared_state_nodes,
)

__all__ = [
    'SharedStateStore',
    'get_shared_store',
    'StateChange',
    'StateChangeType',
    'PlanArtifact',
    'ArtifactType',
    'EventLogEntry',
    'SharedStateNode',
    'PlanArtifactNode',
    'StateWatcherNode',
    'EventLogNode',
    'AgentSyncNode',
    'shared_state_nodes',
]
