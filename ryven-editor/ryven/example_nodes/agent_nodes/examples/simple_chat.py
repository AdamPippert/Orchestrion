"""
Simple Chat Example

Demonstrates basic chat completion with various providers:
- OpenAI
- Anthropic Claude
- Local (vLLM, Ollama, etc.)

This example shows how to set up providers and make chat requests.
"""

import asyncio
from typing import Optional


async def openai_chat_example(api_key: str):
    """
    Simple chat with OpenAI.

    Workflow:
    1. Create OpenAI provider
    2. Build messages
    3. Get chat completion
    """
    from agent_nodes.providers import Message
    from agent_nodes.providers.openai_compat import create_openai_provider

    # Create provider
    provider = create_openai_provider(api_key=api_key)
    await provider.initialize()

    try:
        # Create messages
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="What is Python?"),
        ]

        # Get response
        response = await provider.chat(
            messages=messages,
            model="gpt-4o-mini",
            temperature=0.7,
        )

        print(f"Response: {response.content}")
        print(f"Tokens used: {response.total_tokens}")

        return response

    finally:
        await provider.close()


async def anthropic_chat_example(api_key: str):
    """
    Simple chat with Anthropic Claude.

    Workflow:
    1. Create Anthropic provider
    2. Build messages
    3. Get chat completion
    """
    from agent_nodes.providers import Message
    from agent_nodes.providers.anthropic import create_anthropic_provider

    # Create provider
    provider = create_anthropic_provider(api_key=api_key)
    await provider.initialize()

    try:
        # Create messages
        messages = [
            Message(role="user", content="Explain quantum computing in simple terms."),
        ]

        # Get response
        response = await provider.chat(
            messages=messages,
            model="claude-sonnet-4-20250514",
            system_prompt="You are a helpful science teacher.",
            temperature=0.7,
        )

        print(f"Response: {response.content}")
        print(f"Tokens used: {response.total_tokens}")

        return response

    finally:
        await provider.close()


async def local_chat_example(base_url: str = "http://localhost:8000/v1"):
    """
    Chat with local LLM (vLLM, Ollama, or llama.cpp).

    Workflow:
    1. Create local provider (OpenAI-compatible)
    2. Build messages
    3. Get chat completion
    """
    from agent_nodes.providers import Message
    from agent_nodes.providers.openai_compat import create_vllm_provider

    # Create provider for local vLLM instance
    provider = create_vllm_provider(base_url=base_url)
    await provider.initialize()

    try:
        # Create messages
        messages = [
            Message(role="system", content="You are a coding assistant."),
            Message(role="user", content="Write a Python function to reverse a string."),
        ]

        # Get response - model depends on what's loaded locally
        response = await provider.chat(
            messages=messages,
            model="meta-llama/Llama-3.1-8B-Instruct",
            temperature=0.3,
        )

        print(f"Response: {response.content}")

        return response

    finally:
        await provider.close()


async def streaming_chat_example(api_key: str):
    """
    Streaming chat response.

    Demonstrates real-time streaming of LLM responses.
    """
    from agent_nodes.providers import Message
    from agent_nodes.providers.openai_compat import create_openai_provider

    provider = create_openai_provider(api_key=api_key)
    await provider.initialize()

    try:
        messages = [
            Message(role="user", content="Tell me a short story about a robot."),
        ]

        print("Streaming response:")
        async for chunk in provider.chat_stream(
            messages=messages,
            model="gpt-4o-mini",
        ):
            print(chunk.content, end="", flush=True)
        print()  # Newline at end

    finally:
        await provider.close()


# Example node workflow (how it would work in Ryven)
def node_workflow_example():
    """
    Example of how nodes connect in a visual workflow.

    This shows the logical flow, not actual node execution.

    Workflow:
        [Provider Config] --> [System Prompt] -+
                                               |--> [Message List] --> [Chat] --> [Output]
              [User Input] --> [User Message] -+
    """
    workflow_description = """
    Visual Node Workflow:

    1. Provider Config Node:
       - Type: OpenAI
       - Model: gpt-4o
       - API Key: (from environment)

    2. System Prompt Node:
       - Content: "You are a helpful assistant."

    3. User Message Node:
       - Content: (from user input)

    4. Message List Node:
       - Combines system + user messages

    5. Chat Node:
       - Provider: (from step 1)
       - Messages: (from step 4)
       - Temperature: 0.7

    6. Output:
       - Displays response
    """
    print(workflow_description)


if __name__ == "__main__":
    import os

    print("=" * 50)
    print("Simple Chat Examples")
    print("=" * 50)

    # Show node workflow example
    print("\n1. Node Workflow Example:")
    node_workflow_example()

    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if openai_key:
        print("\n2. OpenAI Chat Example:")
        asyncio.run(openai_chat_example(openai_key))
    else:
        print("\n2. OpenAI: Set OPENAI_API_KEY to run this example")

    if anthropic_key:
        print("\n3. Anthropic Chat Example:")
        asyncio.run(anthropic_chat_example(anthropic_key))
    else:
        print("\n3. Anthropic: Set ANTHROPIC_API_KEY to run this example")

    print("\n4. Local Chat: Start vLLM/Ollama and update base_url to run")
