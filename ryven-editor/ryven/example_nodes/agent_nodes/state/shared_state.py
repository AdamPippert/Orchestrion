"""
Shared State Store for Multi-Agent Coordination.

Implements the "shared state as source of truth" pattern where:
- State store holds plans, artifacts, and agent states
- Chat/messages are just an event log on top
- Agents sync by reading/writing to shared state
- Reactive watchers trigger on state changes

This enables Slack-like agent coordination without message threads
being the source of truth.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Set
from enum import Enum
import time
import uuid
import threading
import json
from ryven.node_env import *


class StateChangeType(Enum):
    """Types of state changes."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    APPEND = "append"


class ArtifactType(Enum):
    """Types of plan artifacts."""
    PLAN = "plan"
    TASK = "task"
    RESULT = "result"
    DOCUMENT = "document"
    CODE = "code"
    DATA = "data"


@dataclass
class StateChange:
    """Record of a state change."""
    change_id: str
    change_type: StateChangeType
    key: str
    value: Any
    previous_value: Optional[Any]
    agent_id: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanArtifact:
    """A plan or artifact stored in shared state."""
    artifact_id: str
    artifact_type: ArtifactType
    name: str
    content: Any
    created_by: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version: int = 1
    parent_id: Optional[str] = None  # For hierarchical plans
    status: str = "active"  # active, completed, archived
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventLogEntry:
    """An entry in the append-only event log."""
    event_id: str
    event_type: str  # message, action, state_change, error
    source: str  # agent_id or system
    content: Any
    timestamp: float = field(default_factory=time.time)
    related_state_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SharedStateStore:
    """
    Central shared state store for multi-agent coordination.

    This is the source of truth - agents read/write here,
    and chat is just an event log reflecting state changes.
    """

    _instance: Optional['SharedStateStore'] = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern for global shared state."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._state: Dict[str, Any] = {}
        self._artifacts: Dict[str, PlanArtifact] = {}
        self._agent_states: Dict[str, Dict[str, Any]] = {}
        self._event_log: List[EventLogEntry] = []
        self._watchers: Dict[str, List[Callable]] = {}
        self._change_history: List[StateChange] = []
        self._initialized = True

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from shared state."""
        return self._state.get(key, default)

    def set(self, key: str, value: Any, agent_id: str = "system") -> StateChange:
        """Set value in shared state."""
        previous = self._state.get(key)
        change_type = StateChangeType.UPDATE if key in self._state else StateChangeType.CREATE

        self._state[key] = value

        change = StateChange(
            change_id=str(uuid.uuid4())[:8],
            change_type=change_type,
            key=key,
            value=value,
            previous_value=previous,
            agent_id=agent_id,
        )
        self._change_history.append(change)
        self._notify_watchers(key, change)

        return change

    def delete(self, key: str, agent_id: str = "system") -> Optional[StateChange]:
        """Delete key from shared state."""
        if key not in self._state:
            return None

        previous = self._state.pop(key)

        change = StateChange(
            change_id=str(uuid.uuid4())[:8],
            change_type=StateChangeType.DELETE,
            key=key,
            value=None,
            previous_value=previous,
            agent_id=agent_id,
        )
        self._change_history.append(change)
        self._notify_watchers(key, change)

        return change

    def keys(self, pattern: Optional[str] = None) -> List[str]:
        """Get all keys, optionally filtered by pattern."""
        if pattern is None:
            return list(self._state.keys())
        return [k for k in self._state.keys() if pattern in k]

    def watch(self, key_pattern: str, callback: Callable[[StateChange], None]):
        """Register a watcher for state changes."""
        if key_pattern not in self._watchers:
            self._watchers[key_pattern] = []
        self._watchers[key_pattern].append(callback)

    def unwatch(self, key_pattern: str, callback: Callable):
        """Remove a watcher."""
        if key_pattern in self._watchers:
            self._watchers[key_pattern] = [
                cb for cb in self._watchers[key_pattern] if cb != callback
            ]

    def _notify_watchers(self, key: str, change: StateChange):
        """Notify all matching watchers of a change."""
        for pattern, callbacks in self._watchers.items():
            if pattern == "*" or pattern in key or key.startswith(pattern):
                for callback in callbacks:
                    try:
                        callback(change)
                    except Exception:
                        pass  # Don't let watcher errors break state changes

    # Artifact management
    def store_artifact(self, artifact: PlanArtifact) -> PlanArtifact:
        """Store a plan artifact."""
        self._artifacts[artifact.artifact_id] = artifact
        self.set(f"artifact:{artifact.artifact_id}", artifact, artifact.created_by)
        return artifact

    def get_artifact(self, artifact_id: str) -> Optional[PlanArtifact]:
        """Get an artifact by ID."""
        return self._artifacts.get(artifact_id)

    def list_artifacts(
        self,
        artifact_type: Optional[ArtifactType] = None,
        status: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> List[PlanArtifact]:
        """List artifacts with optional filtering."""
        results = list(self._artifacts.values())

        if artifact_type:
            results = [a for a in results if a.artifact_type == artifact_type]
        if status:
            results = [a for a in results if a.status == status]
        if created_by:
            results = [a for a in results if a.created_by == created_by]

        return results

    # Agent state management
    def get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """Get an agent's state."""
        return self._agent_states.get(agent_id, {})

    def set_agent_state(self, agent_id: str, state: Dict[str, Any]):
        """Set an agent's state."""
        self._agent_states[agent_id] = state
        self.set(f"agent:{agent_id}", state, agent_id)

    def update_agent_state(self, agent_id: str, updates: Dict[str, Any]):
        """Partially update an agent's state."""
        current = self._agent_states.get(agent_id, {})
        current.update(updates)
        self.set_agent_state(agent_id, current)

    # Event log
    def log_event(self, entry: EventLogEntry):
        """Append to the event log."""
        self._event_log.append(entry)

    def get_events(
        self,
        since: Optional[float] = None,
        event_type: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
    ) -> List[EventLogEntry]:
        """Get events from the log with filtering."""
        results = self._event_log

        if since:
            results = [e for e in results if e.timestamp >= since]
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        if source:
            results = [e for e in results if e.source == source]

        return results[-limit:]

    def clear(self):
        """Clear all state (for testing)."""
        self._state.clear()
        self._artifacts.clear()
        self._agent_states.clear()
        self._event_log.clear()
        self._change_history.clear()


# Global store instance
_global_store = SharedStateStore()


def get_shared_store() -> SharedStateStore:
    """Get the global shared state store."""
    return _global_store


class SharedStateNode(Node):
    """
    Central shared state store node.

    All agents read/write to this shared state. This is the
    source of truth for coordination, not message passing.

    Inputs:
        - action: get, set, delete, keys, or watch
        - key: State key to operate on
        - value: Value to set (for set action)
        - agent_id: ID of agent performing action

    Outputs:
        - result: Operation result
        - change: StateChange record (for mutations)
        - all_keys: List of all keys (for keys action)
    """

    title = 'Shared State'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='action', default='get'),
        NodeInputType(label='key'),
        NodeInputType(label='value'),
        NodeInputType(label='agent_id', default='anonymous'),
    ]
    init_outputs = [
        NodeOutputType(label='result'),
        NodeOutputType(label='change'),
        NodeOutputType(label='all_keys'),
    ]

    def update_event(self, inp=-1):
        store = get_shared_store()

        action = "get"
        action_input = self.input(0)
        if action_input and action_input.payload:
            action = str(action_input.payload)

        key = None
        key_input = self.input(1)
        if key_input and key_input.payload:
            key = str(key_input.payload)

        value = None
        value_input = self.input(2)
        if value_input and value_input.payload:
            value = value_input.payload

        agent_id = "anonymous"
        agent_input = self.input(3)
        if agent_input and agent_input.payload:
            agent_id = str(agent_input.payload)

        result = None
        change = None

        if action == "get" and key:
            result = store.get(key)
        elif action == "set" and key:
            change = store.set(key, value, agent_id)
            result = value
        elif action == "delete" and key:
            change = store.delete(key, agent_id)
            result = change is not None
        elif action == "keys":
            result = store.keys(key)  # key is used as pattern

        self.set_output_val(0, Data(result))
        self.set_output_val(1, Data(change))
        self.set_output_val(2, Data(store.keys()))


class PlanArtifactNode(Node):
    """
    Create and manage plan artifacts as first-class objects.

    Plans are stored in shared state, not as messages.
    This enables proper versioning, hierarchy, and status tracking.

    Inputs:
        - action: create, get, update, list, or complete
        - artifact_type: plan, task, result, document, code, data
        - name: Artifact name
        - content: Artifact content
        - artifact_id: ID for get/update/complete actions
        - parent_id: Parent artifact ID (for hierarchy)
        - agent_id: Creating/updating agent

    Outputs:
        - artifact: The artifact object
        - artifact_id: ID of created/updated artifact
        - artifacts: List of artifacts (for list action)
    """

    title = 'Plan Artifact'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='action', default='create'),
        NodeInputType(label='artifact_type', default='plan'),
        NodeInputType(label='name'),
        NodeInputType(label='content'),
        NodeInputType(label='artifact_id'),
        NodeInputType(label='parent_id'),
        NodeInputType(label='agent_id', default='anonymous'),
    ]
    init_outputs = [
        NodeOutputType(label='artifact'),
        NodeOutputType(label='artifact_id'),
        NodeOutputType(label='artifacts'),
    ]

    def update_event(self, inp=-1):
        store = get_shared_store()

        action = "create"
        action_input = self.input(0)
        if action_input and action_input.payload:
            action = str(action_input.payload)

        artifact_type_str = "plan"
        type_input = self.input(1)
        if type_input and type_input.payload:
            artifact_type_str = str(type_input.payload)

        try:
            artifact_type = ArtifactType(artifact_type_str)
        except ValueError:
            artifact_type = ArtifactType.PLAN

        name = ""
        name_input = self.input(2)
        if name_input and name_input.payload:
            name = str(name_input.payload)

        content = None
        content_input = self.input(3)
        if content_input and content_input.payload:
            content = content_input.payload

        artifact_id = None
        id_input = self.input(4)
        if id_input and id_input.payload:
            artifact_id = str(id_input.payload)

        parent_id = None
        parent_input = self.input(5)
        if parent_input and parent_input.payload:
            parent_id = str(parent_input.payload)

        agent_id = "anonymous"
        agent_input = self.input(6)
        if agent_input and agent_input.payload:
            agent_id = str(agent_input.payload)

        artifact = None
        result_id = None
        artifacts_list = []

        if action == "create":
            artifact = PlanArtifact(
                artifact_id=str(uuid.uuid4())[:8],
                artifact_type=artifact_type,
                name=name,
                content=content,
                created_by=agent_id,
                parent_id=parent_id,
            )
            store.store_artifact(artifact)
            result_id = artifact.artifact_id

        elif action == "get" and artifact_id:
            artifact = store.get_artifact(artifact_id)
            if artifact:
                result_id = artifact.artifact_id

        elif action == "update" and artifact_id:
            artifact = store.get_artifact(artifact_id)
            if artifact:
                artifact.content = content if content else artifact.content
                artifact.name = name if name else artifact.name
                artifact.updated_at = time.time()
                artifact.version += 1
                store.store_artifact(artifact)
                result_id = artifact.artifact_id

        elif action == "complete" and artifact_id:
            artifact = store.get_artifact(artifact_id)
            if artifact:
                artifact.status = "completed"
                artifact.updated_at = time.time()
                store.store_artifact(artifact)
                result_id = artifact.artifact_id

        elif action == "list":
            artifacts_list = store.list_artifacts(
                artifact_type=artifact_type if artifact_type_str != "plan" else None,
            )

        self.set_output_val(0, Data(artifact))
        self.set_output_val(1, Data(result_id))
        self.set_output_val(2, Data(artifacts_list))


class StateWatcherNode(Node):
    """
    Watch for state changes and trigger on updates.

    Enables reactive agent coordination - agents respond to
    state changes rather than polling or message passing.

    Inputs:
        - key_pattern: Pattern to watch (supports * wildcard)
        - change_types: List of change types to watch
        - callback_id: Unique ID for this watcher
        - enabled: Whether watching is enabled

    Outputs:
        - triggered: Whether a change was detected
        - change: The StateChange that triggered
        - changes_since_last: All changes since last check
    """

    title = 'State Watcher'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='key_pattern', default='*'),
        NodeInputType(label='change_types'),
        NodeInputType(label='callback_id'),
        NodeInputType(label='enabled', default=True),
    ]
    init_outputs = [
        NodeOutputType(label='triggered'),
        NodeOutputType(label='change'),
        NodeOutputType(label='changes_since_last'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self._last_check_time = time.time()
        self._pending_changes: List[StateChange] = []
        self._registered = False

    def _on_change(self, change: StateChange):
        """Callback for state changes."""
        self._pending_changes.append(change)

    def update_event(self, inp=-1):
        store = get_shared_store()

        pattern = "*"
        pattern_input = self.input(0)
        if pattern_input and pattern_input.payload:
            pattern = str(pattern_input.payload)

        change_types = None
        types_input = self.input(1)
        if types_input and types_input.payload:
            if isinstance(types_input.payload, list):
                change_types = set(types_input.payload)

        enabled = True
        enabled_input = self.input(3)
        if enabled_input and enabled_input.payload is not None:
            enabled = bool(enabled_input.payload)

        # Register/unregister watcher
        if enabled and not self._registered:
            store.watch(pattern, self._on_change)
            self._registered = True
        elif not enabled and self._registered:
            store.unwatch(pattern, self._on_change)
            self._registered = False

        # Check for pending changes
        changes = self._pending_changes
        if change_types:
            changes = [c for c in changes if c.change_type.value in change_types]

        triggered = len(changes) > 0
        latest_change = changes[-1] if changes else None

        # Clear pending
        self._pending_changes = []
        self._last_check_time = time.time()

        self.set_output_val(0, Data(triggered))
        self.set_output_val(1, Data(latest_change))
        self.set_output_val(2, Data(changes))


class EventLogNode(Node):
    """
    Append-only event log - chat as a side effect of state.

    Messages and events are logged here, but they're not the
    source of truth - they're a record of what happened.

    Inputs:
        - action: log, query, or subscribe
        - event_type: message, action, state_change, error
        - content: Event content
        - source: Source agent/system
        - since: Timestamp for querying
        - limit: Max events to return

    Outputs:
        - logged: Whether event was logged
        - events: Query results
        - latest: Most recent event
    """

    title = 'Event Log'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='action', default='log'),
        NodeInputType(label='event_type', default='message'),
        NodeInputType(label='content'),
        NodeInputType(label='source', default='system'),
        NodeInputType(label='since'),
        NodeInputType(label='limit', default=100),
    ]
    init_outputs = [
        NodeOutputType(label='logged'),
        NodeOutputType(label='events'),
        NodeOutputType(label='latest'),
    ]

    def update_event(self, inp=-1):
        store = get_shared_store()

        action = "log"
        action_input = self.input(0)
        if action_input and action_input.payload:
            action = str(action_input.payload)

        event_type = "message"
        type_input = self.input(1)
        if type_input and type_input.payload:
            event_type = str(type_input.payload)

        content = None
        content_input = self.input(2)
        if content_input and content_input.payload:
            content = content_input.payload

        source = "system"
        source_input = self.input(3)
        if source_input and source_input.payload:
            source = str(source_input.payload)

        since = None
        since_input = self.input(4)
        if since_input and since_input.payload:
            since = float(since_input.payload)

        limit = 100
        limit_input = self.input(5)
        if limit_input and limit_input.payload is not None:
            limit = int(limit_input.payload)

        logged = False
        events = []
        latest = None

        if action == "log" and content is not None:
            entry = EventLogEntry(
                event_id=str(uuid.uuid4())[:8],
                event_type=event_type,
                source=source,
                content=content,
            )
            store.log_event(entry)
            logged = True
            latest = entry

        elif action == "query":
            events = store.get_events(
                since=since,
                event_type=event_type if event_type != "message" else None,
                source=source if source != "system" else None,
                limit=limit,
            )
            if events:
                latest = events[-1]

        self.set_output_val(0, Data(logged))
        self.set_output_val(1, Data(events))
        self.set_output_val(2, Data(latest))


class AgentSyncNode(Node):
    """
    Sync agent state with the shared store.

    Agents use this to publish their state and read others' state.
    Enables coordination without direct messaging.

    Inputs:
        - agent_id: This agent's ID
        - my_state: State to publish
        - watch_agents: List of agent IDs to watch
        - action: sync, publish, or read

    Outputs:
        - my_published_state: Current published state
        - watched_states: States of watched agents
        - all_agents: List of all agent IDs with state
    """

    title = 'Agent Sync'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='agent_id'),
        NodeInputType(label='my_state'),
        NodeInputType(label='watch_agents'),
        NodeInputType(label='action', default='sync'),
    ]
    init_outputs = [
        NodeOutputType(label='my_published_state'),
        NodeOutputType(label='watched_states'),
        NodeOutputType(label='all_agents'),
    ]

    def update_event(self, inp=-1):
        store = get_shared_store()

        agent_id = None
        id_input = self.input(0)
        if id_input and id_input.payload:
            agent_id = str(id_input.payload)

        my_state = None
        state_input = self.input(1)
        if state_input and state_input.payload:
            my_state = state_input.payload

        watch_agents = []
        watch_input = self.input(2)
        if watch_input and watch_input.payload:
            if isinstance(watch_input.payload, list):
                watch_agents = watch_input.payload
            else:
                watch_agents = [str(watch_input.payload)]

        action = "sync"
        action_input = self.input(3)
        if action_input and action_input.payload:
            action = str(action_input.payload)

        my_published = None
        watched = {}
        all_agents = []

        if agent_id:
            if action in ("sync", "publish") and my_state:
                store.set_agent_state(agent_id, my_state if isinstance(my_state, dict) else {"state": my_state})

            my_published = store.get_agent_state(agent_id)

            for other_id in watch_agents:
                watched[other_id] = store.get_agent_state(other_id)

        # Get all agent IDs
        all_agents = [k.replace("agent:", "") for k in store.keys("agent:")]

        self.set_output_val(0, Data(my_published))
        self.set_output_val(1, Data(watched))
        self.set_output_val(2, Data(all_agents))


# Export all shared state nodes
shared_state_nodes = [
    SharedStateNode,
    PlanArtifactNode,
    StateWatcherNode,
    EventLogNode,
    AgentSyncNode,
]
