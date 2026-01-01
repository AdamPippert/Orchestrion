"""
Human-in-the-loop approval guardrails.

Nodes for requiring human approval before certain actions
or when confidence is low.
"""

from __future__ import annotations
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
from ryven.node_env import *


class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class HumanApprovalNode(Node):
    """
    Require human approval before proceeding.

    Creates an approval request that must be approved
    before the flow continues.

    Inputs:
        - action: Description of action requiring approval
        - context: Additional context for reviewer
        - urgency: Urgency level (low, medium, high)
        - timeout_minutes: Auto-reject after timeout
        - exec: Trigger approval request

    Outputs:
        - request_id: Unique ID for this approval request
        - status: Current approval status
        - approved: Exec if approved
        - rejected: Exec if rejected
        - pending: Exec while waiting
    """

    title = 'Human Approval'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='action'),
        NodeInputType(label='context'),
        NodeInputType(label='urgency', default='medium'),
        NodeInputType(label='timeout_minutes', default=60),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='request_id'),
        NodeOutputType(label='status'),
        NodeOutputType(type_='exec', label='approved'),
        NodeOutputType(type_='exec', label='rejected'),
        NodeOutputType(type_='exec', label='pending'),
    ]

    # Class-level storage for approval requests
    _approval_requests: Dict[str, Dict[str, Any]] = {}

    def __init__(self, params):
        super().__init__(params)
        self._current_request_id: Optional[str] = None

    @classmethod
    def approve(cls, request_id: str) -> bool:
        """Approve a pending request."""
        if request_id in cls._approval_requests:
            cls._approval_requests[request_id]['status'] = ApprovalStatus.APPROVED
            cls._approval_requests[request_id]['resolved_at'] = datetime.now().isoformat()
            return True
        return False

    @classmethod
    def reject(cls, request_id: str, reason: str = "") -> bool:
        """Reject a pending request."""
        if request_id in cls._approval_requests:
            cls._approval_requests[request_id]['status'] = ApprovalStatus.REJECTED
            cls._approval_requests[request_id]['resolved_at'] = datetime.now().isoformat()
            cls._approval_requests[request_id]['rejection_reason'] = reason
            return True
        return False

    @classmethod
    def get_pending(cls) -> List[Dict[str, Any]]:
        """Get all pending approval requests."""
        return [
            req for req in cls._approval_requests.values()
            if req['status'] == ApprovalStatus.PENDING
        ]

    def update_event(self, inp=-1):
        if inp != 4:
            return

        action = ""
        action_input = self.input(0)
        if action_input and action_input.payload:
            action = str(action_input.payload)

        context = None
        context_input = self.input(1)
        if context_input and context_input.payload:
            context = context_input.payload

        urgency = "medium"
        urgency_input = self.input(2)
        if urgency_input and urgency_input.payload:
            urgency = str(urgency_input.payload)

        timeout = 60
        timeout_input = self.input(3)
        if timeout_input and timeout_input.payload:
            timeout = int(timeout_input.payload)

        # Create approval request
        request_id = str(uuid.uuid4())
        self._current_request_id = request_id

        request = {
            'id': request_id,
            'action': action,
            'context': context,
            'urgency': urgency,
            'timeout_minutes': timeout,
            'status': ApprovalStatus.PENDING,
            'created_at': datetime.now().isoformat(),
            'node_id': id(self),
        }
        self._approval_requests[request_id] = request

        self.set_output_val(0, Data(request_id))
        self.set_output_val(1, Data(ApprovalStatus.PENDING.value))
        self.exec_output(4)  # pending

    def check_status(self):
        """Check if approval has been resolved."""
        if self._current_request_id and self._current_request_id in self._approval_requests:
            request = self._approval_requests[self._current_request_id]
            status = request['status']

            self.set_output_val(1, Data(status.value))

            if status == ApprovalStatus.APPROVED:
                self.exec_output(2)
            elif status == ApprovalStatus.REJECTED:
                self.exec_output(3)


class ConfidenceGateNode(Node):
    """
    Route based on confidence score.

    Sends high-confidence results directly through,
    but low-confidence results to human review.

    Inputs:
        - value: The value to check
        - confidence: Confidence score (0-1)
        - threshold: Minimum confidence to auto-approve
        - exec: Trigger

    Outputs:
        - value: The original value
        - confidence: The confidence score
        - high_confidence: Exec if above threshold
        - low_confidence: Exec if below threshold
    """

    title = 'Confidence Gate'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='value'),
        NodeInputType(label='confidence'),
        NodeInputType(label='threshold', default=0.8),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='value'),
        NodeOutputType(label='confidence'),
        NodeOutputType(type_='exec', label='high_confidence'),
        NodeOutputType(type_='exec', label='low_confidence'),
    ]

    def update_event(self, inp=-1):
        if inp != 3:
            return

        value = None
        value_input = self.input(0)
        if value_input:
            value = value_input.payload

        confidence = 0.0
        conf_input = self.input(1)
        if conf_input and conf_input.payload is not None:
            confidence = float(conf_input.payload)

        threshold = 0.8
        thresh_input = self.input(2)
        if thresh_input and thresh_input.payload is not None:
            threshold = float(thresh_input.payload)

        self.set_output_val(0, Data(value))
        self.set_output_val(1, Data(confidence))

        if confidence >= threshold:
            self.exec_output(2)
        else:
            self.exec_output(3)


class ApprovalQueueNode(Node):
    """
    Display and manage pending approval requests.

    Provides a way to see all pending approvals and
    approve/reject them.

    Inputs:
        - action: 'list', 'approve', 'reject'
        - request_id: ID of request to approve/reject
        - reason: Rejection reason
        - exec: Trigger

    Outputs:
        - pending: List of pending requests
        - count: Number of pending requests
        - result: Result of approve/reject action
        - done: Exec
    """

    title = 'Approval Queue'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='action'),
        NodeInputType(label='request_id'),
        NodeInputType(label='reason'),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='pending'),
        NodeOutputType(label='count'),
        NodeOutputType(label='result'),
        NodeOutputType(type_='exec', label='done'),
    ]

    def update_event(self, inp=-1):
        if inp != 3:
            return

        action = 'list'
        action_input = self.input(0)
        if action_input and action_input.payload:
            action = str(action_input.payload).lower()

        request_id = None
        id_input = self.input(1)
        if id_input and id_input.payload:
            request_id = str(id_input.payload)

        reason = ""
        reason_input = self.input(2)
        if reason_input and reason_input.payload:
            reason = str(reason_input.payload)

        result = None

        if action == 'list':
            pending = HumanApprovalNode.get_pending()
            self.set_output_val(0, Data(pending))
            self.set_output_val(1, Data(len(pending)))

        elif action == 'approve' and request_id:
            success = HumanApprovalNode.approve(request_id)
            result = {'success': success, 'action': 'approved', 'request_id': request_id}
            pending = HumanApprovalNode.get_pending()
            self.set_output_val(0, Data(pending))
            self.set_output_val(1, Data(len(pending)))

        elif action == 'reject' and request_id:
            success = HumanApprovalNode.reject(request_id, reason)
            result = {'success': success, 'action': 'rejected', 'request_id': request_id, 'reason': reason}
            pending = HumanApprovalNode.get_pending()
            self.set_output_val(0, Data(pending))
            self.set_output_val(1, Data(len(pending)))

        self.set_output_val(2, Data(result))
        self.exec_output(3)


# Export approval nodes
approval_nodes = [
    HumanApprovalNode,
    ConfidenceGateNode,
    ApprovalQueueNode,
]
