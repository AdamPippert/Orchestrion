"""
Input and output validation guardrails.

These nodes help ensure that data flowing through the workflow
meets quality and safety standards.
"""

from __future__ import annotations
import re
from typing import List, Dict, Any, Optional, Pattern
from ryven.node_env import *


class InputValidatorNode(Node):
    """
    Validate input before sending to LLM.

    Checks for:
    - Empty or whitespace-only input
    - Input too long (token limit)
    - Required fields present
    - Format validation (regex)

    Inputs:
        - input: The input to validate
        - max_length: Maximum character length
        - required_fields: List of required field names (for dict input)
        - pattern: Regex pattern the input must match
        - exec: Trigger validation

    Outputs:
        - valid_input: The validated input (or None if invalid)
        - is_valid: Boolean indicating validity
        - errors: List of validation errors
        - passed: Exec if validation passed
        - failed: Exec if validation failed
    """

    title = 'Input Validator'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='input'),
        NodeInputType(label='max_length'),
        NodeInputType(label='required_fields'),
        NodeInputType(label='pattern'),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='valid_input'),
        NodeOutputType(label='is_valid'),
        NodeOutputType(label='errors'),
        NodeOutputType(type_='exec', label='passed'),
        NodeOutputType(type_='exec', label='failed'),
    ]

    def update_event(self, inp=-1):
        if inp != 4:
            return

        input_val = self.input(0)
        if not input_val:
            self.set_output_val(1, Data(False))
            self.set_output_val(2, Data(["No input provided"]))
            self.exec_output(4)
            return

        value = input_val.payload
        errors: List[str] = []

        # Check for empty input
        if value is None:
            errors.append("Input is None")
        elif isinstance(value, str) and not value.strip():
            errors.append("Input is empty or whitespace only")

        # Check max length
        max_length_input = self.input(1)
        if max_length_input and max_length_input.payload:
            max_len = int(max_length_input.payload)
            if isinstance(value, str) and len(value) > max_len:
                errors.append(f"Input exceeds max length ({len(value)} > {max_len})")

        # Check required fields (for dict input)
        required_input = self.input(2)
        if required_input and required_input.payload:
            required_fields = required_input.payload
            if isinstance(required_fields, str):
                required_fields = [required_fields]
            if isinstance(value, dict):
                for field in required_fields:
                    if field not in value or value[field] is None:
                        errors.append(f"Required field '{field}' is missing")

        # Check pattern
        pattern_input = self.input(3)
        if pattern_input and pattern_input.payload:
            pattern = str(pattern_input.payload)
            if isinstance(value, str) and not re.match(pattern, value):
                errors.append(f"Input does not match pattern: {pattern}")

        is_valid = len(errors) == 0
        self.set_output_val(0, Data(value if is_valid else None))
        self.set_output_val(1, Data(is_valid))
        self.set_output_val(2, Data(errors))

        if is_valid:
            self.exec_output(3)
        else:
            self.exec_output(4)


class OutputValidatorNode(Node):
    """
    Validate LLM output before using it.

    Checks for:
    - Empty responses
    - Expected format (JSON, specific structure)
    - Forbidden content
    - Length constraints

    Inputs:
        - output: LLM output to validate
        - min_length: Minimum character length
        - max_length: Maximum character length
        - must_contain: String/pattern that must be present
        - must_not_contain: String/pattern that must not be present
        - expect_json: Whether output should be valid JSON
        - exec: Trigger

    Outputs:
        - valid_output: Validated output
        - is_valid: Boolean
        - errors: Validation errors
        - passed: Exec if valid
        - failed: Exec if invalid
    """

    title = 'Output Validator'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='output'),
        NodeInputType(label='min_length'),
        NodeInputType(label='max_length'),
        NodeInputType(label='must_contain'),
        NodeInputType(label='must_not_contain'),
        NodeInputType(label='expect_json', default=False),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='valid_output'),
        NodeOutputType(label='is_valid'),
        NodeOutputType(label='errors'),
        NodeOutputType(type_='exec', label='passed'),
        NodeOutputType(type_='exec', label='failed'),
    ]

    def update_event(self, inp=-1):
        if inp != 6:
            return

        output_val = self.input(0)
        if not output_val:
            self.set_output_val(1, Data(False))
            self.set_output_val(2, Data(["No output provided"]))
            self.exec_output(4)
            return

        value = output_val.payload
        if isinstance(value, str):
            text = value
        else:
            text = str(value)

        errors: List[str] = []

        # Check min length
        min_len_input = self.input(1)
        if min_len_input and min_len_input.payload:
            min_len = int(min_len_input.payload)
            if len(text) < min_len:
                errors.append(f"Output too short ({len(text)} < {min_len})")

        # Check max length
        max_len_input = self.input(2)
        if max_len_input and max_len_input.payload:
            max_len = int(max_len_input.payload)
            if len(text) > max_len:
                errors.append(f"Output too long ({len(text)} > {max_len})")

        # Check must contain
        must_contain_input = self.input(3)
        if must_contain_input and must_contain_input.payload:
            pattern = str(must_contain_input.payload)
            if not re.search(pattern, text, re.IGNORECASE):
                errors.append(f"Output missing required content: {pattern}")

        # Check must not contain
        must_not_input = self.input(4)
        if must_not_input and must_not_input.payload:
            pattern = str(must_not_input.payload)
            if re.search(pattern, text, re.IGNORECASE):
                errors.append(f"Output contains forbidden content: {pattern}")

        # Check JSON format
        json_input = self.input(5)
        if json_input and json_input.payload:
            import json as json_module
            try:
                json_module.loads(text)
            except json_module.JSONDecodeError as e:
                errors.append(f"Output is not valid JSON: {e}")

        is_valid = len(errors) == 0
        self.set_output_val(0, Data(value if is_valid else None))
        self.set_output_val(1, Data(is_valid))
        self.set_output_val(2, Data(errors))

        if is_valid:
            self.exec_output(3)
        else:
            self.exec_output(4)


class ContentFilterNode(Node):
    """
    Filter content for safety and appropriateness.

    Detects and blocks:
    - Profanity and offensive language
    - Harmful content categories
    - Injection attempts
    - Custom blocklist terms

    Inputs:
        - content: Text to filter
        - blocklist: List of blocked terms/patterns
        - check_injection: Check for prompt injection attempts
        - severity_threshold: Minimum severity to block (1-5)
        - exec: Trigger

    Outputs:
        - filtered_content: Content with blocked terms removed/replaced
        - is_safe: Boolean indicating if content is safe
        - blocked_items: List of items that were blocked
        - severity: Overall severity score (0-5)
        - passed: Exec if safe
        - blocked: Exec if blocked
    """

    title = 'Content Filter'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='content'),
        NodeInputType(label='blocklist'),
        NodeInputType(label='check_injection', default=True),
        NodeInputType(label='severity_threshold', default=3),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='filtered_content'),
        NodeOutputType(label='is_safe'),
        NodeOutputType(label='blocked_items'),
        NodeOutputType(label='severity'),
        NodeOutputType(type_='exec', label='passed'),
        NodeOutputType(type_='exec', label='blocked'),
    ]

    # Common prompt injection patterns
    INJECTION_PATTERNS = [
        r'ignore\s+(all\s+)?previous\s+instructions',
        r'disregard\s+(all\s+)?previous',
        r'forget\s+(everything|all)',
        r'new\s+instructions?:',
        r'system\s*:\s*you\s+are',
        r'<\/?system>',
        r'\[INST\]',
        r'```system',
    ]

    def update_event(self, inp=-1):
        if inp != 4:
            return

        content_input = self.input(0)
        if not content_input or not content_input.payload:
            self.set_output_val(1, Data(True))
            self.set_output_val(2, Data([]))
            self.exec_output(4)
            return

        content = str(content_input.payload)
        blocked_items: List[Dict[str, Any]] = []
        filtered_content = content
        max_severity = 0

        # Check blocklist
        blocklist_input = self.input(1)
        if blocklist_input and blocklist_input.payload:
            blocklist = blocklist_input.payload
            if isinstance(blocklist, str):
                blocklist = [blocklist]

            for term in blocklist:
                if re.search(term, content, re.IGNORECASE):
                    blocked_items.append({
                        'type': 'blocklist',
                        'term': term,
                        'severity': 3,
                    })
                    max_severity = max(max_severity, 3)
                    filtered_content = re.sub(
                        term, '[BLOCKED]', filtered_content, flags=re.IGNORECASE
                    )

        # Check for injection attempts
        check_injection = True
        injection_input = self.input(2)
        if injection_input and injection_input.payload is not None:
            check_injection = bool(injection_input.payload)

        if check_injection:
            for pattern in self.INJECTION_PATTERNS:
                if re.search(pattern, content, re.IGNORECASE):
                    blocked_items.append({
                        'type': 'injection_attempt',
                        'pattern': pattern,
                        'severity': 5,
                    })
                    max_severity = 5
                    break

        # Get threshold
        threshold = 3
        threshold_input = self.input(3)
        if threshold_input and threshold_input.payload is not None:
            threshold = int(threshold_input.payload)

        is_safe = max_severity < threshold
        self.set_output_val(0, Data(filtered_content))
        self.set_output_val(1, Data(is_safe))
        self.set_output_val(2, Data(blocked_items))
        self.set_output_val(3, Data(max_severity))

        if is_safe:
            self.exec_output(4)
        else:
            self.exec_output(5)


class PIIDetectorNode(Node):
    """
    Detect and optionally redact personally identifiable information.

    Detects common PII types:
    - Email addresses
    - Phone numbers
    - Social Security Numbers
    - Credit card numbers
    - IP addresses
    - Custom patterns

    Inputs:
        - content: Text to scan
        - redact: Whether to redact found PII
        - redaction_char: Character to use for redaction
        - custom_patterns: Additional regex patterns to check
        - exec: Trigger

    Outputs:
        - processed_content: Content with PII optionally redacted
        - pii_found: List of PII items detected
        - pii_count: Number of PII items found
        - has_pii: Boolean
        - done: Exec
    """

    title = 'PII Detector'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='content'),
        NodeInputType(label='redact', default=True),
        NodeInputType(label='redaction_char', default='*'),
        NodeInputType(label='custom_patterns'),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='processed_content'),
        NodeOutputType(label='pii_found'),
        NodeOutputType(label='pii_count'),
        NodeOutputType(label='has_pii'),
        NodeOutputType(type_='exec', label='done'),
    ]

    # Common PII patterns
    PII_PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone_us': r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
        'ssn': r'\b[0-9]{3}[-\s]?[0-9]{2}[-\s]?[0-9]{4}\b',
        'credit_card': r'\b(?:[0-9]{4}[-\s]?){3}[0-9]{4}\b',
        'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
    }

    def update_event(self, inp=-1):
        if inp != 4:
            return

        content_input = self.input(0)
        if not content_input or not content_input.payload:
            self.set_output_val(0, Data(""))
            self.set_output_val(1, Data([]))
            self.set_output_val(2, Data(0))
            self.set_output_val(3, Data(False))
            self.exec_output(4)
            return

        content = str(content_input.payload)
        processed = content
        pii_found: List[Dict[str, Any]] = []

        should_redact = True
        redact_input = self.input(1)
        if redact_input and redact_input.payload is not None:
            should_redact = bool(redact_input.payload)

        redaction_char = '*'
        char_input = self.input(2)
        if char_input and char_input.payload:
            redaction_char = str(char_input.payload)[0]

        # Combine built-in and custom patterns
        patterns = dict(self.PII_PATTERNS)
        custom_input = self.input(3)
        if custom_input and custom_input.payload:
            custom = custom_input.payload
            if isinstance(custom, dict):
                patterns.update(custom)
            elif isinstance(custom, list):
                for i, p in enumerate(custom):
                    patterns[f'custom_{i}'] = p

        # Scan for PII
        for pii_type, pattern in patterns.items():
            matches = re.finditer(pattern, content)
            for match in matches:
                pii_found.append({
                    'type': pii_type,
                    'value': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                })

                if should_redact:
                    redacted = redaction_char * len(match.group())
                    processed = processed[:match.start()] + redacted + processed[match.end():]

        self.set_output_val(0, Data(processed))
        self.set_output_val(1, Data(pii_found))
        self.set_output_val(2, Data(len(pii_found)))
        self.set_output_val(3, Data(len(pii_found) > 0))
        self.exec_output(4)


# Export validation nodes
validation_nodes = [
    InputValidatorNode,
    OutputValidatorNode,
    ContentFilterNode,
    PIIDetectorNode,
]
