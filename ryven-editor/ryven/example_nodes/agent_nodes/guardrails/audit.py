"""
Audit and compliance guardrails.

Nodes for logging actions, tracking decisions,
and ensuring compliance with policies.
"""

from __future__ import annotations
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
from ryven.node_env import *


class AuditLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditLogNode(Node):
    """
    Log actions for audit trail.

    Creates detailed audit logs of all agent actions
    for compliance and debugging.

    Inputs:
        - event: Event type/name
        - data: Event data to log
        - level: Log level (debug, info, warning, error, critical)
        - agent: Agent that triggered the event
        - exec: Trigger logging

    Outputs:
        - log_entry: The created log entry
        - log_id: Unique ID for this entry
        - done: Exec
    """

    title = 'Audit Log'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='event'),
        NodeInputType(label='data'),
        NodeInputType(label='level', default='info'),
        NodeInputType(label='agent'),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='log_entry'),
        NodeOutputType(label='log_id'),
        NodeOutputType(type_='exec', label='done'),
    ]

    # Class-level audit log storage
    _audit_logs: List[Dict[str, Any]] = []
    _log_counter = 0

    def __init__(self, params):
        super().__init__(params)

    @classmethod
    def get_logs(cls, limit: int = 100, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve audit logs, optionally filtered."""
        logs = cls._audit_logs
        if level:
            logs = [l for l in logs if l['level'] == level]
        return logs[-limit:]

    @classmethod
    def export_logs(cls, filepath: str) -> bool:
        """Export logs to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(cls._audit_logs, f, indent=2, default=str)
            return True
        except Exception:
            return False

    def update_event(self, inp=-1):
        if inp != 4:
            return

        event = "unknown"
        event_input = self.input(0)
        if event_input and event_input.payload:
            event = str(event_input.payload)

        data = None
        data_input = self.input(1)
        if data_input:
            data = data_input.payload

        level = AuditLevel.INFO
        level_input = self.input(2)
        if level_input and level_input.payload:
            level_str = str(level_input.payload).lower()
            try:
                level = AuditLevel(level_str)
            except ValueError:
                pass

        agent_name = None
        agent_input = self.input(3)
        if agent_input and agent_input.payload:
            agent = agent_input.payload
            if hasattr(agent, 'name'):
                agent_name = agent.name
            elif isinstance(agent, dict):
                agent_name = agent.get('name')
            else:
                agent_name = str(agent)

        # Create log entry
        AuditLogNode._log_counter += 1
        log_id = f"audit_{AuditLogNode._log_counter}"

        entry = {
            'id': log_id,
            'timestamp': datetime.now().isoformat(),
            'event': event,
            'level': level.value,
            'agent': agent_name,
            'data': data,
            'flow_id': id(self.flow) if hasattr(self, 'flow') else None,
        }

        self._audit_logs.append(entry)

        self.set_output_val(0, Data(entry))
        self.set_output_val(1, Data(log_id))
        self.exec_output(2)


class ComplianceCheckNode(Node):
    """
    Check actions against compliance policies.

    Validates that planned actions comply with
    defined policies and rules.

    Inputs:
        - action: Action to check
        - data: Data involved in the action
        - policies: List of policy rules to check
        - exec: Trigger

    Outputs:
        - compliant: Whether action is compliant
        - violations: List of policy violations
        - risk_score: Risk score (0-10)
        - passed: Exec if compliant
        - failed: Exec if non-compliant
    """

    title = 'Compliance Check'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='action'),
        NodeInputType(label='data'),
        NodeInputType(label='policies'),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='compliant'),
        NodeOutputType(label='violations'),
        NodeOutputType(label='risk_score'),
        NodeOutputType(type_='exec', label='passed'),
        NodeOutputType(type_='exec', label='failed'),
    ]

    # Built-in policy checks
    DEFAULT_POLICIES = {
        'no_pii_exposure': {
            'description': 'Prevent exposure of personally identifiable information',
            'check': lambda data: not any(
                term in str(data).lower()
                for term in ['ssn', 'social security', 'credit card', 'password']
            ),
            'risk_weight': 8,
        },
        'no_financial_actions': {
            'description': 'Block unauthorized financial transactions',
            'check': lambda data: 'transaction' not in str(data).lower(),
            'risk_weight': 10,
        },
        'no_external_api': {
            'description': 'Prevent calls to external APIs without approval',
            'check': lambda data: 'http' not in str(data).lower(),
            'risk_weight': 5,
        },
    }

    def update_event(self, inp=-1):
        if inp != 3:
            return

        action = ""
        action_input = self.input(0)
        if action_input and action_input.payload:
            action = str(action_input.payload)

        data = None
        data_input = self.input(1)
        if data_input:
            data = data_input.payload

        # Get policies to check
        policies = dict(self.DEFAULT_POLICIES)
        policies_input = self.input(2)
        if policies_input and policies_input.payload:
            custom_policies = policies_input.payload
            if isinstance(custom_policies, dict):
                policies.update(custom_policies)
            elif isinstance(custom_policies, list):
                # List of policy names to enable
                policies = {k: v for k, v in policies.items() if k in custom_policies}

        violations: List[Dict[str, Any]] = []
        total_risk = 0

        for policy_name, policy in policies.items():
            try:
                check_func = policy.get('check')
                if check_func and callable(check_func):
                    if not check_func(data):
                        violations.append({
                            'policy': policy_name,
                            'description': policy.get('description', ''),
                            'risk_weight': policy.get('risk_weight', 5),
                        })
                        total_risk += policy.get('risk_weight', 5)
            except Exception as e:
                violations.append({
                    'policy': policy_name,
                    'error': str(e),
                    'risk_weight': 5,
                })
                total_risk += 5

        # Cap risk score at 10
        risk_score = min(total_risk, 10)
        is_compliant = len(violations) == 0

        self.set_output_val(0, Data(is_compliant))
        self.set_output_val(1, Data(violations))
        self.set_output_val(2, Data(risk_score))

        if is_compliant:
            self.exec_output(3)
        else:
            self.exec_output(4)


class PolicyNode(Node):
    """
    Define a compliance policy rule.

    Creates a policy that can be used with the
    ComplianceCheckNode.

    Inputs:
        - name: Policy name
        - description: Human-readable description
        - pattern: Regex pattern to block
        - risk_weight: Risk score if violated (1-10)

    Outputs:
        - policy: Policy definition
    """

    title = 'Policy'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='name'),
        NodeInputType(label='description'),
        NodeInputType(label='pattern'),
        NodeInputType(label='risk_weight', default=5),
    ]
    init_outputs = [
        NodeOutputType(label='policy'),
    ]

    def update_event(self, inp=-1):
        import re

        name_input = self.input(0)
        if not name_input or not name_input.payload:
            return

        name = str(name_input.payload)

        description = ""
        desc_input = self.input(1)
        if desc_input and desc_input.payload:
            description = str(desc_input.payload)

        pattern = None
        pattern_input = self.input(2)
        if pattern_input and pattern_input.payload:
            pattern = str(pattern_input.payload)

        risk_weight = 5
        risk_input = self.input(3)
        if risk_input and risk_input.payload is not None:
            risk_weight = int(risk_input.payload)

        # Create check function
        if pattern:
            compiled = re.compile(pattern, re.IGNORECASE)
            check_func = lambda data, p=compiled: not p.search(str(data))
        else:
            check_func = lambda data: True

        policy = {
            name: {
                'description': description,
                'check': check_func,
                'risk_weight': risk_weight,
            }
        }

        self.set_output_val(0, Data(policy))


# Export audit nodes
audit_nodes = [
    AuditLogNode,
    ComplianceCheckNode,
    PolicyNode,
]
