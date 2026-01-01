"""
Multi-Agent Supervisor Pattern Example

Demonstrates a hierarchical multi-agent system with:
- Supervisor agent that delegates tasks
- Specialized worker agents
- Task coordination and result aggregation

This example shows how to build a team of agents that work together.
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class AgentProfile:
    """Definition of an agent's capabilities."""
    name: str
    role: str
    skills: List[str]
    tools: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 3


@dataclass
class Task:
    """A task to be completed by an agent."""
    id: str
    description: str
    assigned_to: Optional[str] = None
    status: str = "pending"
    result: Optional[str] = None


class SupervisorAgent:
    """
    A supervisor agent that coordinates worker agents.

    Implements the Supervisor coordination pattern from Orchestrion.

    Node Flow:
        [Task Input] --> [Supervisor] --> [Task Delegator]
                              │                   │
                              │    ┌──────────────┼──────────────┐
                              │    ▼              ▼              ▼
                              │  [Worker 1]  [Worker 2]    [Worker 3]
                              │    │              │              │
                              │    └──────────────┼──────────────┘
                              │                   ▼
                              └──────────────[Aggregator] --> [Final Result]
    """

    def __init__(
        self,
        name: str = "Supervisor",
        workers: List[AgentProfile] = None,
        provider=None,
    ):
        self.name = name
        self.workers = workers or []
        self.provider = provider
        self.task_history = []

    def add_worker(self, worker: AgentProfile):
        """Add a worker agent to the team."""
        self.workers.append(worker)
        print(f"Added worker: {worker.name} ({worker.role})")

    def select_worker(self, task: Task) -> Optional[AgentProfile]:
        """
        Select the best worker for a task.

        Simulates the DelegatorNode with skill_match strategy.
        """
        task_lower = task.description.lower()

        best_worker = None
        best_score = 0

        for worker in self.workers:
            score = 0
            # Check skill match
            for skill in worker.skills:
                if skill.lower() in task_lower:
                    score += 1

            # Check role match
            if worker.role.lower() in task_lower:
                score += 0.5

            if score > best_score:
                best_score = score
                best_worker = worker

        # Default to first available worker
        if best_worker is None and self.workers:
            best_worker = self.workers[0]

        return best_worker

    async def delegate_task(self, task: Task) -> Task:
        """
        Delegate a task to an appropriate worker.

        Simulates:
        - SupervisorNode (delegation)
        - Worker agent execution
        """
        # Select worker
        worker = self.select_worker(task)

        if worker is None:
            task.status = "failed"
            task.result = "No workers available"
            return task

        task.assigned_to = worker.name
        task.status = "in_progress"
        print(f"Delegating '{task.description}' to {worker.name}")

        # Simulate worker execution
        result = await self._execute_as_worker(worker, task)

        task.status = "completed"
        task.result = result
        self.task_history.append(task)

        return task

    async def _execute_as_worker(
        self,
        worker: AgentProfile,
        task: Task,
    ) -> str:
        """
        Execute a task as a specific worker.

        In a real implementation, this would:
        1. Build worker-specific system prompt
        2. Execute LLM call with worker persona
        3. Optionally use worker's tools
        """
        # Build worker prompt
        prompt = f"""You are {worker.name}, a {worker.role}.
Your skills: {', '.join(worker.skills)}.
Your tools: {', '.join(worker.tools) if worker.tools else 'None'}.

Task: {task.description}

Complete the task based on your role and skills."""

        if self.provider:
            from agent_nodes.providers import Message

            messages = [
                Message(role="system", content=prompt),
                Message(role="user", content=task.description),
            ]

            response = await self.provider.chat(messages=messages)
            return response.content
        else:
            # Mock response for demo
            return f"[{worker.name}] Completed: {task.description}"

    async def process_complex_task(self, description: str) -> Dict[str, Any]:
        """
        Process a complex task by breaking it down and delegating.

        Simulates:
        - PlanExecuteNode (breaking down task)
        - SupervisorNode (coordination)
        - Aggregation of results
        """
        print(f"\nProcessing complex task: {description}")
        print("-" * 50)

        # 1. Break down the task
        subtasks = await self._decompose_task(description)
        print(f"Decomposed into {len(subtasks)} subtasks")

        # 2. Delegate each subtask
        results = []
        for i, subtask_desc in enumerate(subtasks):
            task = Task(
                id=f"task_{i+1}",
                description=subtask_desc,
            )
            completed_task = await self.delegate_task(task)
            results.append(completed_task)

        # 3. Aggregate results
        aggregated = await self._aggregate_results(description, results)

        return {
            'original_task': description,
            'subtasks': len(subtasks),
            'results': [t.result for t in results],
            'final_answer': aggregated,
        }

    async def _decompose_task(self, description: str) -> List[str]:
        """
        Decompose a complex task into subtasks.

        Simulates PlanExecuteNode planning phase.
        """
        # Simple keyword-based decomposition for demo
        if "research" in description.lower() and "summarize" in description.lower():
            return [
                f"Research: {description}",
                "Analyze the research findings",
                "Create a summary of key points",
            ]
        elif "review" in description.lower():
            return [
                f"Initial review: {description}",
                "Identify issues and improvements",
                "Write final review report",
            ]
        else:
            return [description]

    async def _aggregate_results(
        self,
        original_task: str,
        results: List[Task],
    ) -> str:
        """
        Aggregate results from multiple workers.

        Simulates ReduceNode or final aggregation.
        """
        combined = "\n\n".join([
            f"## {task.assigned_to}'s Result:\n{task.result}"
            for task in results
        ])

        if self.provider:
            from agent_nodes.providers import Message

            messages = [
                Message(
                    role="system",
                    content="You are aggregating results from multiple workers. Synthesize their work into a coherent final answer."
                ),
                Message(
                    role="user",
                    content=f"Original task: {original_task}\n\nWorker results:\n{combined}"
                ),
            ]

            response = await self.provider.chat(messages=messages)
            return response.content
        else:
            return f"Aggregated Results:\n{combined}"


def visualize_supervisor_workflow():
    """Visualize the supervisor pattern workflow."""
    workflow = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║              Multi-Agent Supervisor Pattern                    ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║                                                               ║
    ║  ┌─────────────────────────────────────────────────────────┐  ║
    ║  │                    SUPERVISOR AGENT                      │  ║
    ║  │  ┌──────────┐    ┌────────────┐    ┌──────────────┐     │  ║
    ║  │  │  Planner │───▶│ Delegator  │───▶│  Aggregator  │     │  ║
    ║  │  └──────────┘    └─────┬──────┘    └──────▲───────┘     │  ║
    ║  └────────────────────────┼──────────────────┼─────────────┘  ║
    ║                           │                  │                 ║
    ║              ┌────────────┼────────────┐     │                 ║
    ║              ▼            ▼            ▼     │                 ║
    ║       ┌──────────┐ ┌──────────┐ ┌──────────┐ │                 ║
    ║       │ Researcher│ │  Writer  │ │ Reviewer │ │                 ║
    ║       │   Agent   │ │  Agent   │ │  Agent   │ │                 ║
    ║       └─────┬─────┘ └────┬─────┘ └────┬─────┘ │                 ║
    ║             │            │            │       │                 ║
    ║             └────────────┴────────────┴───────┘                 ║
    ║                                                               ║
    ║  Worker Agent Roles:                                          ║
    ║  ├─ Researcher: Data gathering, fact-checking, analysis       ║
    ║  ├─ Writer: Content creation, documentation, summaries        ║
    ║  └─ Reviewer: Quality control, editing, verification          ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(workflow)


async def demo_supervisor():
    """Demonstrate the supervisor pattern."""
    print("=" * 60)
    print("Multi-Agent Supervisor Demo")
    print("=" * 60)

    # Visualize the workflow
    visualize_supervisor_workflow()

    # Create supervisor
    supervisor = SupervisorAgent(name="ProjectManager")

    # Add workers
    supervisor.add_worker(AgentProfile(
        name="Alex",
        role="Research Analyst",
        skills=["research", "data analysis", "fact-checking"],
        tools=["web_search", "document_reader"],
    ))

    supervisor.add_worker(AgentProfile(
        name="Sam",
        role="Technical Writer",
        skills=["writing", "documentation", "summarization"],
        tools=["text_editor", "grammar_checker"],
    ))

    supervisor.add_worker(AgentProfile(
        name="Jordan",
        role="Code Reviewer",
        skills=["code review", "testing", "quality assurance"],
        tools=["linter", "test_runner"],
    ))

    # Process a complex task
    result = await supervisor.process_complex_task(
        "Research the latest trends in AI agents and summarize the key findings"
    )

    print("\n" + "=" * 60)
    print("Final Result:")
    print("=" * 60)
    print(result['final_answer'])


if __name__ == "__main__":
    asyncio.run(demo_supervisor())
