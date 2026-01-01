"""
Agent reasoning patterns.

Implements structured reasoning approaches for AI agents:
- ReAct (Reasoning + Acting)
- Plan-and-Execute
- Chain of Thought
- Self-Reflection
- Tree of Thoughts
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import time
import uuid
from ryven.node_env import *


class ReasoningStepType(Enum):
    """Types of reasoning steps."""
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    REFLECTION = "reflection"
    PLAN = "plan"


@dataclass
class ReasoningStep:
    """A single step in the reasoning process."""
    step_type: ReasoningStepType
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    """A structured plan for task execution."""
    goal: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    current_step: int = 0
    status: str = "pending"  # pending, in_progress, completed, failed
    created_at: float = field(default_factory=time.time)


class ReActNode(Node):
    """
    ReAct (Reasoning + Acting) pattern implementation.

    Interleaves reasoning (thoughts) with actions to solve problems.
    Each iteration:
    1. Thought: Agent reasons about what to do
    2. Action: Agent takes an action
    3. Observation: Result of the action is observed

    Inputs:
        - agent: Agent definition
        - task: Task to complete
        - tools: Available tools for actions
        - max_iterations: Maximum ReAct loops
        - stop_conditions: Conditions to stop early

    Outputs:
        - result: Final result
        - trajectory: Full reasoning trajectory
        - actions_taken: List of actions performed
    """

    title = 'ReAct'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='agent'),
        NodeInputType(label='task'),
        NodeInputType(label='tools'),
        NodeInputType(label='max_iterations', default=10),
        NodeInputType(label='stop_conditions'),
    ]
    init_outputs = [
        NodeOutputType(label='result'),
        NodeOutputType(label='trajectory'),
        NodeOutputType(label='actions_taken'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self.trajectory: List[ReasoningStep] = []
        self.actions_taken: List[Dict] = []
        self.iteration = 0

    def _generate_react_prompt(
        self,
        task: str,
        tools: List[Any],
        trajectory: List[ReasoningStep]
    ) -> str:
        """Generate a ReAct-style prompt."""
        prompt = """You are solving a problem using the ReAct framework.
For each step:
1. Thought: Think about what you need to do
2. Action: Choose an action from available tools
3. Observation: Observe the result

"""
        prompt += f"Task: {task}\n\n"

        if tools:
            prompt += "Available Actions:\n"
            for tool in tools:
                name = tool.get('name', str(tool)) if isinstance(tool, dict) else str(tool)
                desc = tool.get('description', '') if isinstance(tool, dict) else ''
                prompt += f"- {name}: {desc}\n"
            prompt += "\n"

        if trajectory:
            prompt += "Trajectory so far:\n"
            for step in trajectory:
                prompt += f"{step.step_type.value.capitalize()}: {step.content}\n"
            prompt += "\n"

        prompt += "Continue with your next Thought, Action, or if done, provide Final Answer:"
        return prompt

    def update_event(self, inp=-1):
        agent = None
        agent_input = self.input(0)
        if agent_input and agent_input.payload:
            agent = agent_input.payload

        task = ""
        task_input = self.input(1)
        if task_input and task_input.payload:
            task = str(task_input.payload)

        tools = []
        tools_input = self.input(2)
        if tools_input and tools_input.payload:
            if isinstance(tools_input.payload, list):
                tools = tools_input.payload
            else:
                tools = [tools_input.payload]

        max_iter = 10
        iter_input = self.input(3)
        if iter_input and iter_input.payload is not None:
            max_iter = int(iter_input.payload)

        stop_conditions = []
        stop_input = self.input(4)
        if stop_input and stop_input.payload:
            if isinstance(stop_input.payload, list):
                stop_conditions = stop_input.payload
            else:
                stop_conditions = [stop_input.payload]

        # Generate ReAct configuration
        react_config = {
            'agent': getattr(agent, 'name', 'agent') if agent else 'agent',
            'agent_definition': agent,
            'task': task,
            'tools': tools,
            'max_iterations': max_iter,
            'stop_conditions': stop_conditions,
            'prompt': self._generate_react_prompt(task, tools, self.trajectory),
            'trajectory': self.trajectory,
            'iteration': self.iteration,
        }

        self.set_output_val(0, Data(react_config))
        self.set_output_val(1, Data([{
            'type': s.step_type.value,
            'content': s.content,
            'timestamp': s.timestamp
        } for s in self.trajectory]))
        self.set_output_val(2, Data(self.actions_taken))


class PlanExecuteNode(Node):
    """
    Plan-and-Execute pattern for structured task completion.

    First creates a plan, then executes steps sequentially.
    Supports re-planning when needed.

    Inputs:
        - agent: Agent definition
        - task: Task to complete
        - planner_prompt: Custom planning prompt
        - allow_replan: Whether to allow re-planning on failure
        - max_steps: Maximum plan steps

    Outputs:
        - plan: Generated plan
        - current_step: Current step being executed
        - execution_state: Current execution state
        - completed_steps: Steps that have been completed
    """

    title = 'Plan & Execute'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='agent'),
        NodeInputType(label='task'),
        NodeInputType(label='planner_prompt'),
        NodeInputType(label='allow_replan', default=True),
        NodeInputType(label='max_steps', default=10),
    ]
    init_outputs = [
        NodeOutputType(label='plan'),
        NodeOutputType(label='current_step'),
        NodeOutputType(label='execution_state'),
        NodeOutputType(label='completed_steps'),
    ]

    def __init__(self, params):
        super().__init__(params)
        self.plan: Optional[Plan] = None
        self.completed_steps: List[Dict] = []

    def _generate_planning_prompt(self, task: str, custom_prompt: Optional[str]) -> str:
        """Generate a planning prompt."""
        if custom_prompt:
            return custom_prompt.replace("{task}", task)

        return f"""Create a step-by-step plan to complete the following task.

Task: {task}

Break this down into clear, actionable steps. Each step should be:
1. Specific and concrete
2. Independently executable
3. Verifiable when complete

Format your plan as a numbered list of steps.
"""

    def update_event(self, inp=-1):
        agent = None
        agent_input = self.input(0)
        if agent_input and agent_input.payload:
            agent = agent_input.payload

        task = ""
        task_input = self.input(1)
        if task_input and task_input.payload:
            task = str(task_input.payload)

        custom_prompt = None
        prompt_input = self.input(2)
        if prompt_input and prompt_input.payload:
            custom_prompt = str(prompt_input.payload)

        allow_replan = True
        replan_input = self.input(3)
        if replan_input and replan_input.payload is not None:
            allow_replan = bool(replan_input.payload)

        max_steps = 10
        steps_input = self.input(4)
        if steps_input and steps_input.payload is not None:
            max_steps = int(steps_input.payload)

        # Initialize or continue plan
        if not self.plan:
            self.plan = Plan(
                goal=task,
                steps=[],
                current_step=0,
                status="pending",
            )

        plan_config = {
            'agent': getattr(agent, 'name', 'agent') if agent else 'agent',
            'agent_definition': agent,
            'task': task,
            'planning_prompt': self._generate_planning_prompt(task, custom_prompt),
            'allow_replan': allow_replan,
            'max_steps': max_steps,
            'plan': self.plan,
        }

        current = None
        if self.plan.steps and self.plan.current_step < len(self.plan.steps):
            current = self.plan.steps[self.plan.current_step]

        execution_state = {
            'status': self.plan.status,
            'current_step_index': self.plan.current_step,
            'total_steps': len(self.plan.steps),
            'completed': len(self.completed_steps),
        }

        self.set_output_val(0, Data(plan_config))
        self.set_output_val(1, Data(current))
        self.set_output_val(2, Data(execution_state))
        self.set_output_val(3, Data(self.completed_steps))


class ChainOfThoughtNode(Node):
    """
    Chain of Thought (CoT) reasoning pattern.

    Encourages step-by-step reasoning before providing answers.
    Supports few-shot examples for guidance.

    Inputs:
        - agent: Agent definition
        - problem: Problem to solve
        - examples: Few-shot examples (optional)
        - require_steps: Minimum reasoning steps
        - format: Output format (free, numbered, structured)

    Outputs:
        - reasoning: Step-by-step reasoning
        - answer: Final answer
        - confidence: Confidence in the answer
    """

    title = 'Chain of Thought'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='agent'),
        NodeInputType(label='problem'),
        NodeInputType(label='examples'),
        NodeInputType(label='require_steps', default=3),
        NodeInputType(label='format', default='numbered'),
    ]
    init_outputs = [
        NodeOutputType(label='reasoning'),
        NodeOutputType(label='answer'),
        NodeOutputType(label='confidence'),
    ]

    def _generate_cot_prompt(
        self,
        problem: str,
        examples: List[Dict],
        require_steps: int,
        format: str
    ) -> str:
        """Generate a Chain of Thought prompt."""
        prompt = "Think through this step by step.\n\n"

        if examples:
            prompt += "Here are some examples:\n\n"
            for i, ex in enumerate(examples, 1):
                prompt += f"Example {i}:\n"
                prompt += f"Problem: {ex.get('problem', '')}\n"
                if 'reasoning' in ex:
                    prompt += f"Reasoning: {ex['reasoning']}\n"
                if 'answer' in ex:
                    prompt += f"Answer: {ex['answer']}\n"
                prompt += "\n"

        prompt += f"Now solve this problem:\n{problem}\n\n"

        if format == "numbered":
            prompt += f"Show your reasoning in at least {require_steps} numbered steps, then provide your final answer.\n"
            prompt += "Format:\nStep 1: ...\nStep 2: ...\n...\nFinal Answer: ..."
        elif format == "structured":
            prompt += "Structure your response as:\n"
            prompt += "REASONING:\n[Your step-by-step thinking]\n\n"
            prompt += "ANSWER:\n[Your final answer]\n\n"
            prompt += "CONFIDENCE:\n[High/Medium/Low with brief explanation]"
        else:
            prompt += f"Think through at least {require_steps} steps before answering."

        return prompt

    def update_event(self, inp=-1):
        agent = None
        agent_input = self.input(0)
        if agent_input and agent_input.payload:
            agent = agent_input.payload

        problem = ""
        problem_input = self.input(1)
        if problem_input and problem_input.payload:
            problem = str(problem_input.payload)

        examples = []
        examples_input = self.input(2)
        if examples_input and examples_input.payload:
            if isinstance(examples_input.payload, list):
                examples = examples_input.payload
            else:
                examples = [examples_input.payload]

        require_steps = 3
        steps_input = self.input(3)
        if steps_input and steps_input.payload is not None:
            require_steps = int(steps_input.payload)

        format = "numbered"
        format_input = self.input(4)
        if format_input and format_input.payload:
            format = str(format_input.payload)

        cot_prompt = self._generate_cot_prompt(problem, examples, require_steps, format)

        cot_config = {
            'agent': getattr(agent, 'name', 'agent') if agent else 'agent',
            'agent_definition': agent,
            'problem': problem,
            'prompt': cot_prompt,
            'examples': examples,
            'require_steps': require_steps,
            'format': format,
        }

        self.set_output_val(0, Data(cot_config))
        self.set_output_val(1, Data(None))  # Answer comes after execution
        self.set_output_val(2, Data(None))  # Confidence comes after execution


class ReflectionNode(Node):
    """
    Self-reflection pattern for agent improvement.

    Agent reflects on its outputs to identify errors,
    improvements, and verify correctness.

    Inputs:
        - agent: Agent definition
        - output: Output to reflect on
        - original_task: Original task/prompt
        - reflection_prompt: Custom reflection prompt
        - criteria: Evaluation criteria

    Outputs:
        - reflection: Reflection analysis
        - improved_output: Improved version
        - issues_found: List of identified issues
        - quality_score: Self-assessed quality
    """

    title = 'Reflection'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='agent'),
        NodeInputType(label='output'),
        NodeInputType(label='original_task'),
        NodeInputType(label='reflection_prompt'),
        NodeInputType(label='criteria'),
    ]
    init_outputs = [
        NodeOutputType(label='reflection'),
        NodeOutputType(label='improved_output'),
        NodeOutputType(label='issues_found'),
        NodeOutputType(label='quality_score'),
    ]

    def _generate_reflection_prompt(
        self,
        output: str,
        original_task: str,
        custom_prompt: Optional[str],
        criteria: List[str]
    ) -> str:
        """Generate a reflection prompt."""
        if custom_prompt:
            return custom_prompt.replace("{output}", output).replace("{task}", original_task)

        prompt = f"""Reflect on the following output and identify any issues or improvements.

Original Task: {original_task}

Output to Review:
{output}

"""
        if criteria:
            prompt += "Evaluation Criteria:\n"
            for c in criteria:
                prompt += f"- {c}\n"
            prompt += "\n"

        prompt += """Please provide:
1. ISSUES: List any errors, inaccuracies, or problems
2. IMPROVEMENTS: Suggest specific improvements
3. QUALITY SCORE: Rate the output 1-10
4. IMPROVED VERSION: If there are issues, provide an improved version
"""
        return prompt

    def update_event(self, inp=-1):
        agent = None
        agent_input = self.input(0)
        if agent_input and agent_input.payload:
            agent = agent_input.payload

        output = ""
        output_input = self.input(1)
        if output_input and output_input.payload:
            output = str(output_input.payload)

        original_task = ""
        task_input = self.input(2)
        if task_input and task_input.payload:
            original_task = str(task_input.payload)

        custom_prompt = None
        prompt_input = self.input(3)
        if prompt_input and prompt_input.payload:
            custom_prompt = str(prompt_input.payload)

        criteria = []
        criteria_input = self.input(4)
        if criteria_input and criteria_input.payload:
            if isinstance(criteria_input.payload, list):
                criteria = criteria_input.payload
            else:
                criteria = [str(criteria_input.payload)]

        reflection_prompt = self._generate_reflection_prompt(
            output, original_task, custom_prompt, criteria
        )

        reflection_config = {
            'agent': getattr(agent, 'name', 'agent') if agent else 'agent',
            'agent_definition': agent,
            'output': output,
            'original_task': original_task,
            'prompt': reflection_prompt,
            'criteria': criteria,
        }

        self.set_output_val(0, Data(reflection_config))
        self.set_output_val(1, Data(None))  # Improved output after execution
        self.set_output_val(2, Data([]))  # Issues after analysis
        self.set_output_val(3, Data(None))  # Score after analysis


class TreeOfThoughtsNode(Node):
    """
    Tree of Thoughts (ToT) reasoning pattern.

    Explores multiple reasoning paths and evaluates them
    to find the best solution.

    Inputs:
        - agent: Agent definition
        - problem: Problem to solve
        - branching_factor: Number of paths to explore at each step
        - max_depth: Maximum depth of thought tree
        - evaluation_prompt: How to evaluate thoughts

    Outputs:
        - thought_tree: Full tree of explored thoughts
        - best_path: Best reasoning path
        - all_solutions: All viable solutions found
    """

    title = 'Tree of Thoughts'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='agent'),
        NodeInputType(label='problem'),
        NodeInputType(label='branching_factor', default=3),
        NodeInputType(label='max_depth', default=3),
        NodeInputType(label='evaluation_prompt'),
    ]
    init_outputs = [
        NodeOutputType(label='thought_tree'),
        NodeOutputType(label='best_path'),
        NodeOutputType(label='all_solutions'),
    ]

    def _generate_branching_prompt(
        self,
        problem: str,
        current_path: List[str],
        branching_factor: int
    ) -> str:
        """Generate a prompt for branching thoughts."""
        prompt = f"Problem: {problem}\n\n"

        if current_path:
            prompt += "Current reasoning path:\n"
            for i, thought in enumerate(current_path, 1):
                prompt += f"{i}. {thought}\n"
            prompt += "\n"

        prompt += f"Generate {branching_factor} different next steps in the reasoning. "
        prompt += "Each should be a distinct approach or direction.\n"
        prompt += "Format as:\nOption 1: ...\nOption 2: ...\n..."
        return prompt

    def _generate_evaluation_prompt(
        self,
        problem: str,
        paths: List[List[str]],
        custom_prompt: Optional[str]
    ) -> str:
        """Generate a prompt for evaluating thought paths."""
        if custom_prompt:
            return custom_prompt

        prompt = f"Problem: {problem}\n\n"
        prompt += "Evaluate these reasoning paths and rank them:\n\n"

        for i, path in enumerate(paths, 1):
            prompt += f"Path {i}:\n"
            for j, thought in enumerate(path, 1):
                prompt += f"  {j}. {thought}\n"
            prompt += "\n"

        prompt += "Rank the paths from best to worst, explaining why. "
        prompt += "Format: Path X > Path Y > Path Z because..."
        return prompt

    def update_event(self, inp=-1):
        agent = None
        agent_input = self.input(0)
        if agent_input and agent_input.payload:
            agent = agent_input.payload

        problem = ""
        problem_input = self.input(1)
        if problem_input and problem_input.payload:
            problem = str(problem_input.payload)

        branching = 3
        branch_input = self.input(2)
        if branch_input and branch_input.payload is not None:
            branching = int(branch_input.payload)

        max_depth = 3
        depth_input = self.input(3)
        if depth_input and depth_input.payload is not None:
            max_depth = int(depth_input.payload)

        eval_prompt = None
        eval_input = self.input(4)
        if eval_input and eval_input.payload:
            eval_prompt = str(eval_input.payload)

        tot_config = {
            'agent': getattr(agent, 'name', 'agent') if agent else 'agent',
            'agent_definition': agent,
            'problem': problem,
            'branching_factor': branching,
            'max_depth': max_depth,
            'branching_prompt': self._generate_branching_prompt(problem, [], branching),
            'evaluation_prompt': eval_prompt,
            'tree': {'root': problem, 'children': []},
            'explored_paths': [],
        }

        self.set_output_val(0, Data(tot_config))
        self.set_output_val(1, Data(None))  # Best path after exploration
        self.set_output_val(2, Data([]))  # Solutions after exploration


class SelfAskNode(Node):
    """
    Self-Ask pattern for decomposing complex questions.

    Agent asks and answers intermediate questions to
    build up to the final answer.

    Inputs:
        - agent: Agent definition
        - question: Main question to answer
        - max_subquestions: Maximum sub-questions to generate
        - search_tool: Tool for finding answers to sub-questions

    Outputs:
        - decomposition: Question decomposition
        - intermediate_answers: Answers to sub-questions
        - final_answer: Final composed answer
    """

    title = 'Self-Ask'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='agent'),
        NodeInputType(label='question'),
        NodeInputType(label='max_subquestions', default=5),
        NodeInputType(label='search_tool'),
    ]
    init_outputs = [
        NodeOutputType(label='decomposition'),
        NodeOutputType(label='intermediate_answers'),
        NodeOutputType(label='final_answer'),
    ]

    def _generate_self_ask_prompt(self, question: str) -> str:
        """Generate a Self-Ask prompt."""
        return f"""Answer this question by first asking and answering any needed follow-up questions.

Question: {question}

Use this format:
Are follow-up questions needed here: Yes/No
[If yes]
Follow-up: [sub-question]
Intermediate answer: [answer to sub-question]
...
[When ready]
So the final answer is: [answer]

Begin:
"""

    def update_event(self, inp=-1):
        agent = None
        agent_input = self.input(0)
        if agent_input and agent_input.payload:
            agent = agent_input.payload

        question = ""
        question_input = self.input(1)
        if question_input and question_input.payload:
            question = str(question_input.payload)

        max_sub = 5
        sub_input = self.input(2)
        if sub_input and sub_input.payload is not None:
            max_sub = int(sub_input.payload)

        search_tool = None
        tool_input = self.input(3)
        if tool_input and tool_input.payload:
            search_tool = tool_input.payload

        self_ask_config = {
            'agent': getattr(agent, 'name', 'agent') if agent else 'agent',
            'agent_definition': agent,
            'question': question,
            'prompt': self._generate_self_ask_prompt(question),
            'max_subquestions': max_sub,
            'search_tool': search_tool,
            'subquestions': [],
            'intermediate_answers': {},
        }

        self.set_output_val(0, Data(self_ask_config))
        self.set_output_val(1, Data({}))
        self.set_output_val(2, Data(None))


# Export all reasoning nodes
reasoning_nodes = [
    ReActNode,
    PlanExecuteNode,
    ChainOfThoughtNode,
    ReflectionNode,
    TreeOfThoughtsNode,
    SelfAskNode,
]
