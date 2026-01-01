"""
ReAct (Reasoning + Acting) Agent Example

Demonstrates the ReAct pattern where an agent interleaves:
1. Thought: Reasoning about what to do
2. Action: Taking an action using tools
3. Observation: Observing the result

This is a powerful pattern for tool-using agents.
"""

import asyncio
import re
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class Tool:
    """A tool that the agent can use."""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, str] = field(default_factory=dict)


@dataclass
class ReActStep:
    """A single step in the ReAct loop."""
    step_type: str  # "thought", "action", "observation"
    content: str


class ReActAgent:
    """
    A ReAct agent that reasons and acts.

    Implements the ReAct reasoning pattern from Orchestrion.

    Node Flow:
        ┌──────────────────────────────────────────────────┐
        │                   ReAct Loop                      │
        │                                                   │
        │  [Task] --> [Thought] --> [Action] --> [Observe]  │
        │              ▲                            │       │
        │              └────────────────────────────┘       │
        │                    (until done)                   │
        └──────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        provider=None,
        tools: List[Tool] = None,
        max_iterations: int = 10,
    ):
        self.provider = provider
        self.tools = {t.name: t for t in (tools or [])}
        self.max_iterations = max_iterations
        self.trajectory: List[ReActStep] = []

    def add_tool(self, tool: Tool):
        """Add a tool to the agent."""
        self.tools[tool.name] = tool
        print(f"Added tool: {tool.name}")

    def _build_system_prompt(self) -> str:
        """Build the ReAct system prompt."""
        tools_desc = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])

        return f"""You are a helpful assistant that solves problems step by step.

You have access to the following tools:
{tools_desc}

To solve problems, use this format:

Thought: [Your reasoning about what to do next]
Action: [tool_name(arg1, arg2, ...)]

After each action, you will receive an observation. Continue thinking and acting until you have the final answer.

When you have the final answer, respond with:
Thought: I now have enough information to answer.
Final Answer: [Your final answer]

Important:
- Always think before acting
- Use tools when needed to gather information
- Be thorough but efficient
"""

    async def run(self, task: str) -> Dict[str, Any]:
        """
        Run the ReAct loop on a task.

        Returns the final answer and the full trajectory.
        """
        print(f"\nTask: {task}")
        print("=" * 50)

        self.trajectory = []
        current_prompt = task

        for iteration in range(self.max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")

            # Get agent response
            response = await self._get_agent_response(current_prompt)

            # Parse the response
            parsed = self._parse_response(response)

            if parsed.get('final_answer'):
                print(f"\nFinal Answer: {parsed['final_answer']}")
                return {
                    'answer': parsed['final_answer'],
                    'trajectory': self.trajectory,
                    'iterations': iteration + 1,
                }

            # Record thought
            if parsed.get('thought'):
                self.trajectory.append(ReActStep(
                    step_type="thought",
                    content=parsed['thought']
                ))
                print(f"Thought: {parsed['thought']}")

            # Execute action if present
            if parsed.get('action'):
                action = parsed['action']
                print(f"Action: {action}")

                observation = await self._execute_action(action)
                self.trajectory.append(ReActStep(
                    step_type="action",
                    content=action
                ))
                self.trajectory.append(ReActStep(
                    step_type="observation",
                    content=observation
                ))
                print(f"Observation: {observation}")

                # Build next prompt with observation
                current_prompt = f"""Previous trajectory:
{self._format_trajectory()}

Observation: {observation}

Continue solving the task. Remember to think step by step."""

        # Max iterations reached
        return {
            'answer': "Max iterations reached without final answer",
            'trajectory': self.trajectory,
            'iterations': self.max_iterations,
        }

    async def _get_agent_response(self, prompt: str) -> str:
        """Get response from the LLM."""
        if self.provider:
            from agent_nodes.providers import Message

            messages = [
                Message(role="system", content=self._build_system_prompt()),
                Message(role="user", content=prompt),
            ]

            response = await self.provider.chat(
                messages=messages,
                temperature=0.3,
            )
            return response.content
        else:
            # Mock response for demo
            return self._mock_response(prompt)

    def _mock_response(self, prompt: str) -> str:
        """Generate mock responses for demo."""
        if "weather" in prompt.lower():
            if "Observation:" not in prompt:
                return "Thought: I need to check the weather for this location.\nAction: get_weather(New York)"
            else:
                return "Thought: I now have the weather information.\nFinal Answer: The weather in New York is sunny with a temperature of 72°F (22°C)."
        elif "calculate" in prompt.lower() or "math" in prompt.lower():
            if "Observation:" not in prompt:
                return "Thought: I need to perform a calculation.\nAction: calculate(2 + 2)"
            else:
                return "Thought: I have the calculation result.\nFinal Answer: The answer is 4."
        else:
            return "Thought: Let me think about this.\nFinal Answer: Here is my response based on my knowledge."

    def _parse_response(self, response: str) -> Dict[str, str]:
        """Parse the agent's response into components."""
        result = {}

        # Extract thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|Final Answer:|$)', response, re.DOTALL)
        if thought_match:
            result['thought'] = thought_match.group(1).strip()

        # Extract action
        action_match = re.search(r'Action:\s*(.+?)(?=Observation:|Thought:|Final Answer:|$)', response, re.DOTALL)
        if action_match:
            result['action'] = action_match.group(1).strip()

        # Extract final answer
        final_match = re.search(r'Final Answer:\s*(.+?)$', response, re.DOTALL)
        if final_match:
            result['final_answer'] = final_match.group(1).strip()

        return result

    async def _execute_action(self, action: str) -> str:
        """Execute an action using tools."""
        # Parse action: tool_name(args)
        match = re.match(r'(\w+)\((.*)\)', action)
        if not match:
            return f"Error: Invalid action format. Use: tool_name(arg1, arg2, ...)"

        tool_name = match.group(1)
        args_str = match.group(2)

        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'. Available: {list(self.tools.keys())}"

        tool = self.tools[tool_name]

        try:
            # Parse and execute
            # Simple argument parsing for demo
            args = [a.strip().strip('"\'') for a in args_str.split(',') if a.strip()]
            result = tool.function(*args)
            if asyncio.iscoroutine(result):
                result = await result
            return str(result)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def _format_trajectory(self) -> str:
        """Format the trajectory for the prompt."""
        lines = []
        for step in self.trajectory:
            lines.append(f"{step.step_type.title()}: {step.content}")
        return "\n".join(lines)


def visualize_react_workflow():
    """Visualize the ReAct workflow."""
    workflow = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    ReAct Agent Pattern                         ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║                                                               ║
    ║  ┌─────────────┐                                              ║
    ║  │    Task     │                                              ║
    ║  └──────┬──────┘                                              ║
    ║         │                                                     ║
    ║         ▼                                                     ║
    ║  ┌─────────────────────────────────────────────┐              ║
    ║  │                 ReAct Loop                  │              ║
    ║  │  ┌──────────────────────────────────────┐   │              ║
    ║  │  │                                      │   │              ║
    ║  │  ▼                                      │   │              ║
    ║  │  ┌─────────────┐                        │   │              ║
    ║  │  │   THOUGHT   │ "I need to search..."  │   │              ║
    ║  │  └──────┬──────┘                        │   │              ║
    ║  │         │                               │   │              ║
    ║  │         ▼                               │   │              ║
    ║  │  ┌─────────────┐    ┌─────────────┐     │   │              ║
    ║  │  │   ACTION    │───▶│    TOOL     │     │   │              ║
    ║  │  │search("...")│    │  Executor   │     │   │              ║
    ║  │  └─────────────┘    └──────┬──────┘     │   │              ║
    ║  │                            │            │   │              ║
    ║  │                            ▼            │   │              ║
    ║  │                     ┌─────────────┐     │   │              ║
    ║  │                     │ OBSERVATION │─────┘   │              ║
    ║  │                     │  (results)  │         │              ║
    ║  │                     └─────────────┘         │              ║
    ║  │                                             │              ║
    ║  └─────────────────────────────────────────────┘              ║
    ║                          │                                    ║
    ║                          ▼                                    ║
    ║  ┌──────────────────────────────────────────┐                 ║
    ║  │   Check: Is task complete?               │                 ║
    ║  │   - No: Continue loop                    │                 ║
    ║  │   - Yes: Output Final Answer             │                 ║
    ║  └──────────────────────────────────────────┘                 ║
    ║                          │                                    ║
    ║                          ▼                                    ║
    ║                   ┌─────────────┐                             ║
    ║                   │Final Answer │                             ║
    ║                   └─────────────┘                             ║
    ║                                                               ║
    ║  Available Tools:                                             ║
    ║  ├─ search(query): Search the web                             ║
    ║  ├─ calculate(expr): Evaluate math expressions                ║
    ║  ├─ get_weather(city): Get current weather                    ║
    ║  └─ read_file(path): Read file contents                       ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(workflow)


# Example tools
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        # Safe eval for simple math
        allowed_chars = set('0123456789+-*/().^ ')
        if all(c in allowed_chars for c in expression):
            result = eval(expression.replace('^', '**'))
            return f"Result: {result}"
        return "Error: Invalid expression"
    except Exception as e:
        return f"Error: {str(e)}"


def get_weather(city: str) -> str:
    """Get weather for a city (mock)."""
    # Mock weather data
    weather_data = {
        "new york": "Sunny, 72°F (22°C)",
        "london": "Cloudy, 59°F (15°C)",
        "tokyo": "Rainy, 68°F (20°C)",
        "paris": "Clear, 64°F (18°C)",
    }
    city_lower = city.lower()
    if city_lower in weather_data:
        return f"Weather in {city}: {weather_data[city_lower]}"
    return f"Weather data not available for {city}"


def search(query: str) -> str:
    """Search for information (mock)."""
    # Mock search results
    return f"Search results for '{query}': Found 3 relevant articles about {query}."


async def demo_react_agent():
    """Demonstrate the ReAct agent."""
    print("=" * 60)
    print("ReAct Agent Demo")
    print("=" * 60)

    # Visualize the workflow
    visualize_react_workflow()

    # Create agent with tools
    agent = ReActAgent(max_iterations=5)

    agent.add_tool(Tool(
        name="calculate",
        description="Evaluate mathematical expressions",
        function=calculate,
    ))

    agent.add_tool(Tool(
        name="get_weather",
        description="Get current weather for a city",
        function=get_weather,
    ))

    agent.add_tool(Tool(
        name="search",
        description="Search for information",
        function=search,
    ))

    # Run the agent
    result = await agent.run("What is the weather in New York?")

    print("\n" + "=" * 60)
    print(f"Completed in {result['iterations']} iterations")
    print(f"Trajectory length: {len(result['trajectory'])} steps")


if __name__ == "__main__":
    asyncio.run(demo_react_agent())
