"""
Knowledge base and persistent memory nodes.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from datetime import datetime
from ryven.node_env import *


class KnowledgeBaseNode(Node):
    """
    Structured knowledge base for facts and relationships.

    Stores and retrieves structured knowledge that agents
    can reference during reasoning.

    Inputs:
        - action: 'add', 'query', 'update', 'delete'
        - subject: Subject of the fact
        - predicate: Relationship type
        - object: Object of the fact
        - query: Search query for retrieval
        - exec: Trigger

    Outputs:
        - results: Query results
        - count: Number of facts in KB
        - done: Exec
    """

    title = 'Knowledge Base'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='action'),
        NodeInputType(label='subject'),
        NodeInputType(label='predicate'),
        NodeInputType(label='object'),
        NodeInputType(label='query'),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='results'),
        NodeOutputType(label='count'),
        NodeOutputType(type_='exec', label='done'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self._facts: List[Dict[str, Any]] = []

    def update_event(self, inp=-1):
        if inp != 5:
            return

        action = 'query'
        action_input = self.input(0)
        if action_input and action_input.payload:
            action = str(action_input.payload).lower()

        subject = None
        subj_input = self.input(1)
        if subj_input and subj_input.payload:
            subject = str(subj_input.payload)

        predicate = None
        pred_input = self.input(2)
        if pred_input and pred_input.payload:
            predicate = str(pred_input.payload)

        obj = None
        obj_input = self.input(3)
        if obj_input and obj_input.payload:
            obj = str(obj_input.payload)

        query = None
        query_input = self.input(4)
        if query_input and query_input.payload:
            query = str(query_input.payload)

        results = []

        if action == 'add':
            if subject and predicate and obj:
                fact = {
                    'subject': subject,
                    'predicate': predicate,
                    'object': obj,
                    'created_at': datetime.now().isoformat(),
                }
                self._facts.append(fact)

        elif action == 'query':
            for fact in self._facts:
                match = True
                if subject and fact['subject'] != subject:
                    match = False
                if predicate and fact['predicate'] != predicate:
                    match = False
                if obj and fact['object'] != obj:
                    match = False
                if query:
                    query_lower = query.lower()
                    if (query_lower not in fact['subject'].lower() and
                        query_lower not in fact['predicate'].lower() and
                        query_lower not in fact['object'].lower()):
                        match = False
                if match:
                    results.append(fact)

        elif action == 'delete':
            self._facts = [
                f for f in self._facts
                if not (
                    (not subject or f['subject'] == subject) and
                    (not predicate or f['predicate'] == predicate) and
                    (not obj or f['object'] == obj)
                )
            ]

        self.set_output_val(0, Data(results))
        self.set_output_val(1, Data(len(self._facts)))
        self.exec_output(2)

    def get_state(self) -> dict:
        return {'facts': self._facts}

    def set_state(self, data: dict, version):
        self._facts = data.get('facts', [])


class FactStoreNode(Node):
    """
    Simple key-value fact storage.

    Stores individual facts that can be retrieved
    by key.

    Inputs:
        - action: 'set', 'get', 'delete', 'list'
        - key: Fact key
        - value: Fact value
        - exec: Trigger

    Outputs:
        - value: Retrieved value
        - all_facts: All stored facts
        - done: Exec
    """

    title = 'Fact Store'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='action'),
        NodeInputType(label='key'),
        NodeInputType(label='value'),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='value'),
        NodeOutputType(label='all_facts'),
        NodeOutputType(type_='exec', label='done'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self._facts: Dict[str, Any] = {}

    def update_event(self, inp=-1):
        if inp != 3:
            return

        action = 'get'
        action_input = self.input(0)
        if action_input and action_input.payload:
            action = str(action_input.payload).lower()

        key = None
        key_input = self.input(1)
        if key_input and key_input.payload:
            key = str(key_input.payload)

        value = None
        val_input = self.input(2)
        if val_input:
            value = val_input.payload

        result = None

        if action == 'set' and key:
            self._facts[key] = value
            result = value

        elif action == 'get' and key:
            result = self._facts.get(key)

        elif action == 'delete' and key:
            if key in self._facts:
                result = self._facts.pop(key)

        elif action == 'list':
            result = list(self._facts.keys())

        self.set_output_val(0, Data(result))
        self.set_output_val(1, Data(dict(self._facts)))
        self.exec_output(2)

    def get_state(self) -> dict:
        return {'facts': self._facts}

    def set_state(self, data: dict, version):
        self._facts = data.get('facts', {})


class EntityMemoryNode(Node):
    """
    Track entities mentioned in conversations.

    Maintains a memory of entities (people, places, things)
    and their attributes as they're discovered.

    Inputs:
        - action: 'add_entity', 'update_attribute', 'get', 'search'
        - entity_name: Name of entity
        - entity_type: Type (person, place, organization, etc.)
        - attribute: Attribute name
        - value: Attribute value
        - exec: Trigger

    Outputs:
        - entity: Retrieved entity data
        - all_entities: All known entities
        - done: Exec
    """

    title = 'Entity Memory'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='action'),
        NodeInputType(label='entity_name'),
        NodeInputType(label='entity_type'),
        NodeInputType(label='attribute'),
        NodeInputType(label='value'),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='entity'),
        NodeOutputType(label='all_entities'),
        NodeOutputType(type_='exec', label='done'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self._entities: Dict[str, Dict[str, Any]] = {}

    def update_event(self, inp=-1):
        if inp != 5:
            return

        action = 'get'
        action_input = self.input(0)
        if action_input and action_input.payload:
            action = str(action_input.payload).lower()

        name = None
        name_input = self.input(1)
        if name_input and name_input.payload:
            name = str(name_input.payload)

        entity_type = 'unknown'
        type_input = self.input(2)
        if type_input and type_input.payload:
            entity_type = str(type_input.payload)

        attribute = None
        attr_input = self.input(3)
        if attr_input and attr_input.payload:
            attribute = str(attr_input.payload)

        value = None
        val_input = self.input(4)
        if val_input:
            value = val_input.payload

        result = None

        if action == 'add_entity' and name:
            if name not in self._entities:
                self._entities[name] = {
                    'name': name,
                    'type': entity_type,
                    'attributes': {},
                    'first_seen': datetime.now().isoformat(),
                    'last_seen': datetime.now().isoformat(),
                }
            result = self._entities[name]

        elif action == 'update_attribute' and name and attribute:
            if name not in self._entities:
                self._entities[name] = {
                    'name': name,
                    'type': entity_type,
                    'attributes': {},
                    'first_seen': datetime.now().isoformat(),
                    'last_seen': datetime.now().isoformat(),
                }

            self._entities[name]['attributes'][attribute] = value
            self._entities[name]['last_seen'] = datetime.now().isoformat()
            result = self._entities[name]

        elif action == 'get' and name:
            result = self._entities.get(name)

        elif action == 'search':
            # Search by type or attribute value
            matches = []
            for entity in self._entities.values():
                if entity_type != 'unknown' and entity['type'] == entity_type:
                    matches.append(entity)
                elif value and str(value) in str(entity):
                    matches.append(entity)
            result = matches

        self.set_output_val(0, Data(result))
        self.set_output_val(1, Data(dict(self._entities)))
        self.exec_output(2)

    def get_state(self) -> dict:
        return {'entities': self._entities}

    def set_state(self, data: dict, version):
        self._entities = data.get('entities', {})


# Export knowledge nodes
knowledge_nodes = [
    KnowledgeBaseNode,
    FactStoreNode,
    EntityMemoryNode,
]
