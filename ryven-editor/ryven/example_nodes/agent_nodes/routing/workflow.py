"""
Workflow composition and chain patterns.

Provides nodes for building reusable, composable workflows:
- LLM chains (sequential, branching, recursive)
- Workflow templates
- Pipeline patterns
- State machines
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Union
from enum import Enum
import time
import uuid
from ryven.node_env import *


class ChainType(Enum):
    """Types of LLM chains."""
    SEQUENTIAL = "sequential"     # A -> B -> C
    BRANCHING = "branching"       # A -> (B or C) based on condition
    PARALLEL = "parallel"         # A -> [B, C, D] -> combine
    RECURSIVE = "recursive"       # A -> B -> A (until condition)
    TRANSFORM = "transform"       # A -> transform -> B


@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    name: str
    node_type: str  # Type of node to execute
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: List[str] = field(default_factory=list)
    condition: Optional[str] = None  # Condition for execution
    on_success: Optional[str] = None  # Next step on success
    on_failure: Optional[str] = None  # Next step on failure
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowDefinition:
    """Complete workflow definition."""
    name: str
    description: str = ""
    steps: List[WorkflowStep] = field(default_factory=list)
    initial_step: str = ""
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: List[str] = field(default_factory=list)
    version: str = "1.0"
    created_at: float = field(default_factory=time.time)


class LLMChainNode(Node):
    """
    Sequential LLM chain - pass output of one LLM to another.

    Creates a chain where each LLM call's output becomes
    part of the next call's input.

    Inputs:
        - providers: List of providers to chain
        - initial_prompt: Starting prompt
        - chain_template: Template for connecting outputs to inputs
        - max_steps: Maximum chain length
        - stop_condition: Condition to stop early

    Outputs:
        - final_output: Final chain result
        - intermediate_outputs: All intermediate results
        - chain_metadata: Chain execution metadata
    """

    title = 'LLM Chain'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='providers'),
        NodeInputType(label='initial_prompt'),
        NodeInputType(label='chain_template', default='{previous_output}'),
        NodeInputType(label='max_steps', default=5),
        NodeInputType(label='stop_condition'),
    ]
    init_outputs = [
        NodeOutputType(label='final_output'),
        NodeOutputType(label='intermediate_outputs'),
        NodeOutputType(label='chain_metadata'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self.intermediate_outputs = []
        self.current_step = 0

    def _apply_template(self, template: str, outputs: List[str], step: int) -> str:
        """Apply chain template to generate next input."""
        if not outputs:
            return template

        result = template
        result = result.replace('{previous_output}', outputs[-1] if outputs else '')
        result = result.replace('{all_outputs}', '\n'.join(outputs))
        result = result.replace('{step}', str(step))
        return result

    def update_event(self, inp=-1):
        providers = []
        providers_input = self.input(0)
        if providers_input and providers_input.payload:
            if isinstance(providers_input.payload, list):
                providers = providers_input.payload
            else:
                providers = [providers_input.payload]

        initial_prompt = ""
        prompt_input = self.input(1)
        if prompt_input and prompt_input.payload:
            initial_prompt = str(prompt_input.payload)

        template = "{previous_output}"
        template_input = self.input(2)
        if template_input and template_input.payload:
            template = str(template_input.payload)

        max_steps = 5
        steps_input = self.input(3)
        if steps_input and steps_input.payload is not None:
            max_steps = int(steps_input.payload)

        stop_condition = None
        stop_input = self.input(4)
        if stop_input and stop_input.payload:
            stop_condition = stop_input.payload

        chain_config = {
            'type': 'sequential',
            'providers': providers,
            'initial_prompt': initial_prompt,
            'template': template,
            'max_steps': max_steps,
            'stop_condition': stop_condition,
            'current_step': self.current_step,
            'intermediate_outputs': self.intermediate_outputs,
        }

        metadata = {
            'chain_length': len(providers),
            'max_steps': max_steps,
            'current_step': self.current_step,
        }

        self.set_output_val(0, Data(chain_config))
        self.set_output_val(1, Data(self.intermediate_outputs))
        self.set_output_val(2, Data(metadata))


class TransformChainNode(Node):
    """
    Transform chain - apply transformations between LLM calls.

    Allows inserting transform functions (parsing, formatting,
    filtering) between LLM invocations.

    Inputs:
        - provider: LLM provider
        - input_data: Input to process
        - pre_transform: Transform before LLM call
        - post_transform: Transform after LLM call
        - prompt_template: Template for LLM prompt

    Outputs:
        - output: Transformed output
        - raw_output: Raw LLM output
    """

    title = 'Transform Chain'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='provider'),
        NodeInputType(label='input_data'),
        NodeInputType(label='pre_transform'),
        NodeInputType(label='post_transform'),
        NodeInputType(label='prompt_template'),
    ]
    init_outputs = [
        NodeOutputType(label='output'),
        NodeOutputType(label='raw_output'),
    ]

    def _apply_transform(self, data: Any, transform: Any) -> Any:
        """Apply a transform function to data."""
        if transform is None:
            return data
        if callable(transform):
            return transform(data)
        if isinstance(transform, str):
            # Simple template replacement
            if isinstance(data, str):
                return transform.replace('{input}', data)
            return transform
        return data

    def update_event(self, inp=-1):
        provider = None
        provider_input = self.input(0)
        if provider_input and provider_input.payload:
            provider = provider_input.payload

        input_data = None
        data_input = self.input(1)
        if data_input and data_input.payload:
            input_data = data_input.payload

        pre_transform = None
        pre_input = self.input(2)
        if pre_input and pre_input.payload:
            pre_transform = pre_input.payload

        post_transform = None
        post_input = self.input(3)
        if post_input and post_input.payload:
            post_transform = post_input.payload

        prompt_template = "{input}"
        template_input = self.input(4)
        if template_input and template_input.payload:
            prompt_template = str(template_input.payload)

        # Apply pre-transform
        transformed_input = self._apply_transform(input_data, pre_transform)

        config = {
            'provider': provider,
            'original_input': input_data,
            'transformed_input': transformed_input,
            'pre_transform': pre_transform,
            'post_transform': post_transform,
            'prompt_template': prompt_template,
        }

        self.set_output_val(0, Data(config))
        self.set_output_val(1, Data(None))  # Raw output after execution


class RouterChainNode(Node):
    """
    Router chain - route to different chains based on input.

    Analyzes input and routes to the appropriate processing
    chain based on classification.

    Inputs:
        - input_data: Data to route
        - routes: Dict mapping classifications to chains
        - classifier: Classification function or prompt
        - default_route: Default chain if no match

    Outputs:
        - selected_route: Which route was selected
        - chain: The chain to execute
        - classification: Classification result
    """

    title = 'Router Chain'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='input_data'),
        NodeInputType(label='routes'),
        NodeInputType(label='classifier'),
        NodeInputType(label='default_route'),
    ]
    init_outputs = [
        NodeOutputType(label='selected_route'),
        NodeOutputType(label='chain'),
        NodeOutputType(label='classification'),
    ]

    def _classify_input(self, data: Any, classifier: Any) -> str:
        """Classify input to determine routing."""
        if classifier is None:
            return "default"

        if callable(classifier):
            return str(classifier(data))

        # Simple keyword matching for demo
        if isinstance(data, str) and isinstance(classifier, dict):
            data_lower = data.lower()
            for keyword, classification in classifier.items():
                if keyword.lower() in data_lower:
                    return classification

        return "default"

    def update_event(self, inp=-1):
        input_data = None
        data_input = self.input(0)
        if data_input and data_input.payload:
            input_data = data_input.payload

        routes = {}
        routes_input = self.input(1)
        if routes_input and routes_input.payload:
            routes = routes_input.payload if isinstance(routes_input.payload, dict) else {}

        classifier = None
        classifier_input = self.input(2)
        if classifier_input and classifier_input.payload:
            classifier = classifier_input.payload

        default_route = None
        default_input = self.input(3)
        if default_input and default_input.payload:
            default_route = default_input.payload

        classification = self._classify_input(input_data, classifier)
        selected_chain = routes.get(classification, default_route)

        self.set_output_val(0, Data(classification))
        self.set_output_val(1, Data(selected_chain))
        self.set_output_val(2, Data({
            'input': input_data,
            'classification': classification,
            'available_routes': list(routes.keys()),
        }))


class WorkflowNode(Node):
    """
    Define a reusable workflow template.

    Creates a workflow definition that can be instantiated
    and executed multiple times.

    Inputs:
        - name: Workflow name
        - description: Workflow description
        - steps: List of workflow steps
        - initial_step: Name of first step

    Outputs:
        - workflow: Workflow definition
        - summary: Human-readable workflow summary
    """

    title = 'Workflow'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='name'),
        NodeInputType(label='description'),
        NodeInputType(label='steps'),
        NodeInputType(label='initial_step'),
    ]
    init_outputs = [
        NodeOutputType(label='workflow'),
        NodeOutputType(label='summary'),
    ]

    def update_event(self, inp=-1):
        name = "unnamed_workflow"
        name_input = self.input(0)
        if name_input and name_input.payload:
            name = str(name_input.payload)

        description = ""
        desc_input = self.input(1)
        if desc_input and desc_input.payload:
            description = str(desc_input.payload)

        steps = []
        steps_input = self.input(2)
        if steps_input and steps_input.payload:
            if isinstance(steps_input.payload, list):
                steps = steps_input.payload
            else:
                steps = [steps_input.payload]

        initial_step = ""
        initial_input = self.input(3)
        if initial_input and initial_input.payload:
            initial_step = str(initial_input.payload)

        workflow = WorkflowDefinition(
            name=name,
            description=description,
            steps=[],  # Convert raw steps to WorkflowStep objects
            initial_step=initial_step,
        )

        # Parse steps
        for step in steps:
            if isinstance(step, dict):
                workflow.steps.append(WorkflowStep(
                    name=step.get('name', 'unnamed'),
                    node_type=step.get('type', 'generic'),
                    inputs=step.get('inputs', {}),
                    outputs=step.get('outputs', []),
                    condition=step.get('condition'),
                    on_success=step.get('on_success'),
                    on_failure=step.get('on_failure'),
                ))
            elif isinstance(step, WorkflowStep):
                workflow.steps.append(step)

        # Generate summary
        summary_parts = [f"Workflow: {name}"]
        if description:
            summary_parts.append(f"Description: {description}")
        summary_parts.append(f"Steps: {len(workflow.steps)}")
        for s in workflow.steps:
            summary_parts.append(f"  - {s.name} ({s.node_type})")

        self.set_output_val(0, Data(workflow))
        self.set_output_val(1, Data("\n".join(summary_parts)))


class WorkflowStepNode(Node):
    """
    Define a single workflow step.

    Creates a step definition for use in workflows.

    Inputs:
        - name: Step name
        - type: Node type to execute
        - inputs: Input configuration
        - condition: Optional execution condition
        - on_success: Next step on success
        - on_failure: Next step on failure

    Outputs:
        - step: Step definition
    """

    title = 'Workflow Step'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='name'),
        NodeInputType(label='type'),
        NodeInputType(label='inputs'),
        NodeInputType(label='condition'),
        NodeInputType(label='on_success'),
        NodeInputType(label='on_failure'),
    ]
    init_outputs = [
        NodeOutputType(label='step'),
    ]

    def update_event(self, inp=-1):
        name = "step"
        name_input = self.input(0)
        if name_input and name_input.payload:
            name = str(name_input.payload)

        node_type = "generic"
        type_input = self.input(1)
        if type_input and type_input.payload:
            node_type = str(type_input.payload)

        inputs = {}
        inputs_input = self.input(2)
        if inputs_input and inputs_input.payload:
            inputs = inputs_input.payload if isinstance(inputs_input.payload, dict) else {}

        condition = None
        condition_input = self.input(3)
        if condition_input and condition_input.payload:
            condition = str(condition_input.payload)

        on_success = None
        success_input = self.input(4)
        if success_input and success_input.payload:
            on_success = str(success_input.payload)

        on_failure = None
        failure_input = self.input(5)
        if failure_input and failure_input.payload:
            on_failure = str(failure_input.payload)

        step = WorkflowStep(
            name=name,
            node_type=node_type,
            inputs=inputs,
            condition=condition,
            on_success=on_success,
            on_failure=on_failure,
        )

        self.set_output_val(0, Data(step))


class WorkflowRunnerNode(Node):
    """
    Execute a workflow definition.

    Takes a workflow definition and executes it step by step.

    Inputs:
        - workflow: Workflow definition
        - inputs: Workflow inputs
        - context: Execution context
        - dry_run: If true, validate but don't execute

    Outputs:
        - result: Execution result
        - execution_log: Step-by-step log
        - state: Final workflow state
    """

    title = 'Workflow Runner'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='workflow'),
        NodeInputType(label='inputs'),
        NodeInputType(label='context'),
        NodeInputType(label='dry_run', default=False),
    ]
    init_outputs = [
        NodeOutputType(label='result'),
        NodeOutputType(label='execution_log'),
        NodeOutputType(label='state'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self.execution_log = []
        self.state = {}

    def update_event(self, inp=-1):
        workflow = None
        workflow_input = self.input(0)
        if workflow_input and workflow_input.payload:
            workflow = workflow_input.payload

        inputs = {}
        inputs_input = self.input(1)
        if inputs_input and inputs_input.payload:
            inputs = inputs_input.payload if isinstance(inputs_input.payload, dict) else {}

        context = {}
        context_input = self.input(2)
        if context_input and context_input.payload:
            context = context_input.payload if isinstance(context_input.payload, dict) else {}

        dry_run = False
        dry_input = self.input(3)
        if dry_input and dry_input.payload is not None:
            dry_run = bool(dry_input.payload)

        # Prepare execution configuration
        run_config = {
            'workflow': workflow,
            'inputs': inputs,
            'context': context,
            'dry_run': dry_run,
            'execution_id': str(uuid.uuid4())[:8],
            'started_at': time.time(),
        }

        if workflow and hasattr(workflow, 'steps'):
            run_config['total_steps'] = len(workflow.steps)
            run_config['initial_step'] = workflow.initial_step

        self.set_output_val(0, Data(run_config))
        self.set_output_val(1, Data(self.execution_log))
        self.set_output_val(2, Data(self.state))


class PipelineNode(Node):
    """
    Data processing pipeline with stages.

    Creates a pipeline for processing data through multiple
    stages, with optional filtering and transformation.

    Inputs:
        - stages: List of processing stages
        - input_data: Data to process
        - continue_on_error: Whether to continue if a stage fails
        - collect_intermediate: Collect intermediate results

    Outputs:
        - output: Final pipeline output
        - intermediate: Intermediate stage outputs
        - errors: Any errors encountered
    """

    title = 'Pipeline'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='stages'),
        NodeInputType(label='input_data'),
        NodeInputType(label='continue_on_error', default=False),
        NodeInputType(label='collect_intermediate', default=True),
    ]
    init_outputs = [
        NodeOutputType(label='output'),
        NodeOutputType(label='intermediate'),
        NodeOutputType(label='errors'),
    ]

    def update_event(self, inp=-1):
        stages = []
        stages_input = self.input(0)
        if stages_input and stages_input.payload:
            if isinstance(stages_input.payload, list):
                stages = stages_input.payload
            else:
                stages = [stages_input.payload]

        input_data = None
        data_input = self.input(1)
        if data_input and data_input.payload:
            input_data = data_input.payload

        continue_on_error = False
        continue_input = self.input(2)
        if continue_input and continue_input.payload is not None:
            continue_on_error = bool(continue_input.payload)

        collect = True
        collect_input = self.input(3)
        if collect_input and collect_input.payload is not None:
            collect = bool(collect_input.payload)

        pipeline_config = {
            'stages': stages,
            'input_data': input_data,
            'continue_on_error': continue_on_error,
            'collect_intermediate': collect,
            'stage_count': len(stages),
        }

        self.set_output_val(0, Data(pipeline_config))
        self.set_output_val(1, Data([]))
        self.set_output_val(2, Data([]))


class StateMachineNode(Node):
    """
    State machine for complex workflow control.

    Implements a finite state machine for managing
    complex, stateful workflows.

    Inputs:
        - states: Dict of state definitions
        - transitions: List of valid transitions
        - initial_state: Starting state
        - context: Shared context

    Outputs:
        - machine: State machine object
        - current_state: Current state
        - available_transitions: Valid transitions from current state
    """

    title = 'State Machine'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='states'),
        NodeInputType(label='transitions'),
        NodeInputType(label='initial_state'),
        NodeInputType(label='context'),
    ]
    init_outputs = [
        NodeOutputType(label='machine'),
        NodeOutputType(label='current_state'),
        NodeOutputType(label='available_transitions'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self.current_state = None
        self.history = []

    def _get_available_transitions(
        self,
        current: str,
        transitions: List[Dict]
    ) -> List[Dict]:
        """Get transitions available from current state."""
        return [t for t in transitions if t.get('from') == current]

    def update_event(self, inp=-1):
        states = {}
        states_input = self.input(0)
        if states_input and states_input.payload:
            states = states_input.payload if isinstance(states_input.payload, dict) else {}

        transitions = []
        trans_input = self.input(1)
        if trans_input and trans_input.payload:
            if isinstance(trans_input.payload, list):
                transitions = trans_input.payload
            else:
                transitions = [trans_input.payload]

        initial_state = None
        initial_input = self.input(2)
        if initial_input and initial_input.payload:
            initial_state = str(initial_input.payload)

        context = {}
        context_input = self.input(3)
        if context_input and context_input.payload:
            context = context_input.payload if isinstance(context_input.payload, dict) else {}

        if self.current_state is None and initial_state:
            self.current_state = initial_state

        available = self._get_available_transitions(
            self.current_state, transitions
        ) if self.current_state else []

        machine = {
            'states': states,
            'transitions': transitions,
            'current_state': self.current_state,
            'initial_state': initial_state,
            'context': context,
            'history': self.history,
        }

        self.set_output_val(0, Data(machine))
        self.set_output_val(1, Data(self.current_state))
        self.set_output_val(2, Data(available))


class PromptChainNode(Node):
    """
    Chain prompts with variable substitution.

    Chains multiple prompt templates together, substituting
    variables from previous responses.

    Inputs:
        - prompts: List of prompt templates
        - variables: Initial variables
        - provider: LLM provider
        - extract_pattern: Regex to extract variables from responses

    Outputs:
        - final_response: Last response in chain
        - all_responses: All responses
        - final_variables: Variables after all substitutions
    """

    title = 'Prompt Chain'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='prompts'),
        NodeInputType(label='variables'),
        NodeInputType(label='provider'),
        NodeInputType(label='extract_pattern'),
    ]
    init_outputs = [
        NodeOutputType(label='final_response'),
        NodeOutputType(label='all_responses'),
        NodeOutputType(label='final_variables'),
    ]

    def _substitute_variables(self, template: str, variables: Dict) -> str:
        """Substitute variables in template."""
        result = template
        for key, value in variables.items():
            result = result.replace(f'{{{key}}}', str(value))
        return result

    def update_event(self, inp=-1):
        prompts = []
        prompts_input = self.input(0)
        if prompts_input and prompts_input.payload:
            if isinstance(prompts_input.payload, list):
                prompts = prompts_input.payload
            else:
                prompts = [prompts_input.payload]

        variables = {}
        vars_input = self.input(1)
        if vars_input and vars_input.payload:
            variables = vars_input.payload if isinstance(vars_input.payload, dict) else {}

        provider = None
        provider_input = self.input(2)
        if provider_input and provider_input.payload:
            provider = provider_input.payload

        extract_pattern = None
        pattern_input = self.input(3)
        if pattern_input and pattern_input.payload:
            extract_pattern = str(pattern_input.payload)

        # Prepare substituted prompts
        substituted = [self._substitute_variables(p, variables) for p in prompts if isinstance(p, str)]

        chain_config = {
            'prompts': prompts,
            'substituted_prompts': substituted,
            'variables': variables,
            'provider': provider,
            'extract_pattern': extract_pattern,
            'chain_length': len(prompts),
        }

        self.set_output_val(0, Data(chain_config))
        self.set_output_val(1, Data([]))
        self.set_output_val(2, Data(variables))


# Export all workflow nodes
workflow_nodes = [
    LLMChainNode,
    TransformChainNode,
    RouterChainNode,
    WorkflowNode,
    WorkflowStepNode,
    WorkflowRunnerNode,
    PipelineNode,
    StateMachineNode,
    PromptChainNode,
]
