"""
Tests for shared state store and related nodes.
"""

import pytest
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../ryven-editor/ryven/example_nodes'))


class TestSharedStateStore:
    """Tests for the SharedStateStore singleton."""

    def test_store_singleton(self):
        """Verify store is a singleton."""
        from agent_nodes.state.shared_state import SharedStateStore

        store1 = SharedStateStore()
        store2 = SharedStateStore()

        assert store1 is store2

    def test_get_set(self):
        """Test basic get/set operations."""
        from agent_nodes.state.shared_state import get_shared_store

        store = get_shared_store()
        store.clear()

        # Set value
        change = store.set("test_key", "test_value", "agent1")

        assert change.key == "test_key"
        assert change.value == "test_value"
        assert change.agent_id == "agent1"

        # Get value
        result = store.get("test_key")
        assert result == "test_value"

        # Get default
        result = store.get("nonexistent", "default")
        assert result == "default"

    def test_delete(self):
        """Test delete operation."""
        from agent_nodes.state.shared_state import get_shared_store

        store = get_shared_store()
        store.clear()

        store.set("to_delete", "value", "agent1")
        assert store.get("to_delete") == "value"

        change = store.delete("to_delete", "agent1")
        assert change is not None
        assert change.previous_value == "value"
        assert store.get("to_delete") is None

    def test_keys(self):
        """Test keys listing."""
        from agent_nodes.state.shared_state import get_shared_store

        store = get_shared_store()
        store.clear()

        store.set("prefix:a", 1, "agent1")
        store.set("prefix:b", 2, "agent1")
        store.set("other:c", 3, "agent1")

        all_keys = store.keys()
        assert len(all_keys) == 3

        prefix_keys = store.keys("prefix")
        assert len(prefix_keys) == 2

    def test_watchers(self):
        """Test state watchers."""
        from agent_nodes.state.shared_state import get_shared_store

        store = get_shared_store()
        store.clear()

        changes_received = []

        def on_change(change):
            changes_received.append(change)

        store.watch("watched_key", on_change)
        store.set("watched_key", "value1", "agent1")
        store.set("watched_key", "value2", "agent1")

        assert len(changes_received) == 2
        assert changes_received[0].value == "value1"
        assert changes_received[1].value == "value2"

        # Unwatch
        store.unwatch("watched_key", on_change)
        store.set("watched_key", "value3", "agent1")
        assert len(changes_received) == 2  # No new changes


class TestPlanArtifact:
    """Tests for PlanArtifact data class."""

    def test_artifact_creation(self):
        """Test creating a plan artifact."""
        from agent_nodes.state.shared_state import PlanArtifact, ArtifactType

        artifact = PlanArtifact(
            artifact_id="art-123",
            artifact_type=ArtifactType.PLAN,
            name="Test Plan",
            content={"steps": ["step1", "step2"]},
            created_by="agent1",
        )

        assert artifact.artifact_id == "art-123"
        assert artifact.artifact_type == ArtifactType.PLAN
        assert artifact.name == "Test Plan"
        assert artifact.version == 1
        assert artifact.status == "active"

    def test_artifact_storage(self):
        """Test storing and retrieving artifacts."""
        from agent_nodes.state.shared_state import (
            get_shared_store, PlanArtifact, ArtifactType
        )

        store = get_shared_store()
        store.clear()

        artifact = PlanArtifact(
            artifact_id="plan-001",
            artifact_type=ArtifactType.PLAN,
            name="My Plan",
            content="Plan content",
            created_by="planner",
        )

        store.store_artifact(artifact)

        retrieved = store.get_artifact("plan-001")
        assert retrieved is not None
        assert retrieved.name == "My Plan"

    def test_list_artifacts(self):
        """Test listing artifacts with filters."""
        from agent_nodes.state.shared_state import (
            get_shared_store, PlanArtifact, ArtifactType
        )

        store = get_shared_store()
        store.clear()

        # Create different artifact types
        store.store_artifact(PlanArtifact(
            artifact_id="p1", artifact_type=ArtifactType.PLAN,
            name="Plan 1", content="", created_by="agent1"
        ))
        store.store_artifact(PlanArtifact(
            artifact_id="t1", artifact_type=ArtifactType.TASK,
            name="Task 1", content="", created_by="agent2"
        ))
        store.store_artifact(PlanArtifact(
            artifact_id="p2", artifact_type=ArtifactType.PLAN,
            name="Plan 2", content="", created_by="agent1"
        ))

        # Filter by type
        plans = store.list_artifacts(artifact_type=ArtifactType.PLAN)
        assert len(plans) == 2

        # Filter by creator
        by_agent1 = store.list_artifacts(created_by="agent1")
        assert len(by_agent1) == 2


class TestEventLog:
    """Tests for the event log."""

    def test_log_event(self):
        """Test logging events."""
        from agent_nodes.state.shared_state import (
            get_shared_store, EventLogEntry
        )

        store = get_shared_store()
        store.clear()

        entry = EventLogEntry(
            event_id="evt-001",
            event_type="message",
            source="agent1",
            content="Hello world",
        )

        store.log_event(entry)

        events = store.get_events()
        assert len(events) == 1
        assert events[0].content == "Hello world"

    def test_query_events(self):
        """Test querying events with filters."""
        from agent_nodes.state.shared_state import (
            get_shared_store, EventLogEntry
        )

        store = get_shared_store()
        store.clear()

        # Add events with different types and sources
        store.log_event(EventLogEntry(
            event_id="e1", event_type="message",
            source="agent1", content="msg1"
        ))
        store.log_event(EventLogEntry(
            event_id="e2", event_type="action",
            source="agent2", content="action1"
        ))
        store.log_event(EventLogEntry(
            event_id="e3", event_type="message",
            source="agent1", content="msg2"
        ))

        # Filter by type
        messages = store.get_events(event_type="message")
        assert len(messages) == 2

        # Filter by source
        agent1_events = store.get_events(source="agent1")
        assert len(agent1_events) == 2


class TestAgentState:
    """Tests for agent state management."""

    def test_agent_state(self):
        """Test agent state get/set."""
        from agent_nodes.state.shared_state import get_shared_store

        store = get_shared_store()
        store.clear()

        store.set_agent_state("agent1", {"status": "active", "task": "research"})

        state = store.get_agent_state("agent1")
        assert state["status"] == "active"
        assert state["task"] == "research"

    def test_update_agent_state(self):
        """Test partial agent state update."""
        from agent_nodes.state.shared_state import get_shared_store

        store = get_shared_store()
        store.clear()

        store.set_agent_state("agent1", {"status": "active", "count": 0})
        store.update_agent_state("agent1", {"count": 5})

        state = store.get_agent_state("agent1")
        assert state["status"] == "active"  # Unchanged
        assert state["count"] == 5  # Updated


class TestSharedStateNode:
    """Tests for SharedStateNode."""

    def test_node_exists(self):
        """Verify node is properly defined."""
        from agent_nodes.state.shared_state import SharedStateNode

        assert hasattr(SharedStateNode, 'title')
        assert SharedStateNode.title == 'Shared State'


class TestPlanArtifactNode:
    """Tests for PlanArtifactNode."""

    def test_node_exists(self):
        """Verify node is properly defined."""
        from agent_nodes.state.shared_state import PlanArtifactNode

        assert hasattr(PlanArtifactNode, 'title')
        assert PlanArtifactNode.title == 'Plan Artifact'


class TestStateWatcherNode:
    """Tests for StateWatcherNode."""

    def test_node_exists(self):
        """Verify node is properly defined."""
        from agent_nodes.state.shared_state import StateWatcherNode

        assert hasattr(StateWatcherNode, 'title')
        assert StateWatcherNode.title == 'State Watcher'


class TestEventLogNode:
    """Tests for EventLogNode."""

    def test_node_exists(self):
        """Verify node is properly defined."""
        from agent_nodes.state.shared_state import EventLogNode

        assert hasattr(EventLogNode, 'title')
        assert EventLogNode.title == 'Event Log'


class TestAgentSyncNode:
    """Tests for AgentSyncNode."""

    def test_node_exists(self):
        """Verify node is properly defined."""
        from agent_nodes.state.shared_state import AgentSyncNode

        assert hasattr(AgentSyncNode, 'title')
        assert AgentSyncNode.title == 'Agent Sync'


class TestStateChangeType:
    """Tests for StateChangeType enum."""

    def test_change_types(self):
        """Test all change types exist."""
        from agent_nodes.state.shared_state import StateChangeType

        assert StateChangeType.CREATE.value == "create"
        assert StateChangeType.UPDATE.value == "update"
        assert StateChangeType.DELETE.value == "delete"
        assert StateChangeType.APPEND.value == "append"


class TestArtifactType:
    """Tests for ArtifactType enum."""

    def test_artifact_types(self):
        """Test all artifact types exist."""
        from agent_nodes.state.shared_state import ArtifactType

        assert ArtifactType.PLAN.value == "plan"
        assert ArtifactType.TASK.value == "task"
        assert ArtifactType.RESULT.value == "result"
        assert ArtifactType.DOCUMENT.value == "document"
        assert ArtifactType.CODE.value == "code"
        assert ArtifactType.DATA.value == "data"
