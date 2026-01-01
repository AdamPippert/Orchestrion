"""
Model routing nodes for intelligent provider selection.

Route requests to different models based on:
- Cost optimization
- Speed/latency requirements
- Quality/accuracy needs
- Task complexity
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from ryven.node_env import *

from ..providers.base import BaseProvider, ProviderConfig, ProviderType
from ..providers.model_info import get_model_info, list_models, ModelInfo


class ModelRouterNode(Node):
    """
    Route requests to optimal model based on criteria.

    Selects the best model from available options based on
    weighted criteria for cost, speed, and quality.

    Inputs:
        - providers: List of available providers
        - task: Task description (for complexity estimation)
        - cost_weight: Weight for cost optimization (0-1)
        - speed_weight: Weight for speed optimization (0-1)
        - quality_weight: Weight for quality optimization (0-1)
        - min_quality: Minimum acceptable quality threshold
        - max_cost: Maximum acceptable cost per request
        - exec: Trigger

    Outputs:
        - selected_provider: The chosen provider
        - selected_model: The chosen model name
        - reasoning: Explanation of the selection
        - all_scores: Scores for all considered models
        - done: Exec
    """

    title = 'Model Router'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='providers'),
        NodeInputType(label='task'),
        NodeInputType(label='cost_weight', default=0.3),
        NodeInputType(label='speed_weight', default=0.3),
        NodeInputType(label='quality_weight', default=0.4),
        NodeInputType(label='min_quality'),
        NodeInputType(label='max_cost'),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='selected_provider'),
        NodeOutputType(label='selected_model'),
        NodeOutputType(label='reasoning'),
        NodeOutputType(label='all_scores'),
        NodeOutputType(type_='exec', label='done'),
    ]

    # Quality tiers for models (rough estimates)
    MODEL_QUALITY_TIERS = {
        'gpt-4o': 0.95,
        'gpt-4o-mini': 0.85,
        'o1': 0.98,
        'o1-mini': 0.92,
        'claude-opus-4-20250514': 0.97,
        'claude-sonnet-4-20250514': 0.93,
        'claude-3-5-sonnet-20241022': 0.92,
        'claude-3-5-haiku-20241022': 0.85,
        'llama3.2': 0.80,
        'mistral': 0.78,
        'qwen2.5': 0.82,
    }

    def _estimate_complexity(self, task: str) -> float:
        """Estimate task complexity from description (0-1)."""
        complexity_indicators = {
            'simple': -0.2,
            'basic': -0.2,
            'quick': -0.1,
            'complex': 0.3,
            'detailed': 0.2,
            'analyze': 0.2,
            'reasoning': 0.3,
            'code': 0.2,
            'math': 0.3,
            'creative': 0.1,
        }

        base = 0.5
        task_lower = task.lower()
        for indicator, weight in complexity_indicators.items():
            if indicator in task_lower:
                base += weight

        return max(0.0, min(1.0, base))

    def _score_model(
        self,
        model_info: ModelInfo,
        cost_weight: float,
        speed_weight: float,
        quality_weight: float,
        complexity: float,
    ) -> Tuple[float, Dict[str, float]]:
        """Score a model based on weighted criteria."""
        cap = model_info.capabilities

        # Normalize cost (lower is better)
        # Use log scale for cost, max reasonable cost ~$100/M tokens
        max_cost = 100.0
        avg_cost = (cap.input_cost_per_million + cap.output_cost_per_million) / 2
        cost_score = 1.0 - min(avg_cost / max_cost, 1.0)

        # Speed score (based on typical latency, lower is better)
        # Assume local models are fastest, then smaller cloud models
        if cap.input_cost_per_million == 0:  # Local model
            speed_score = 0.95
        elif cap.input_cost_per_million < 1.0:
            speed_score = 0.85
        elif cap.input_cost_per_million < 5.0:
            speed_score = 0.70
        else:
            speed_score = 0.50

        # Quality score
        quality_score = self.MODEL_QUALITY_TIERS.get(model_info.name, 0.75)

        # Adjust quality based on task complexity
        # Complex tasks benefit more from high-quality models
        adjusted_quality = quality_score * (0.7 + 0.3 * complexity)

        # Calculate weighted score
        total_score = (
            cost_weight * cost_score +
            speed_weight * speed_score +
            quality_weight * adjusted_quality
        )

        breakdown = {
            'cost': cost_score,
            'speed': speed_score,
            'quality': quality_score,
            'adjusted_quality': adjusted_quality,
            'total': total_score,
        }

        return total_score, breakdown

    def update_event(self, inp=-1):
        if inp != 7:
            return

        providers_input = self.input(0)
        if not providers_input or not providers_input.payload:
            self.set_output_val(2, Data("Error: No providers specified"))
            self.exec_output(4)
            return

        providers = providers_input.payload
        if not isinstance(providers, list):
            providers = [providers]

        task = ""
        task_input = self.input(1)
        if task_input and task_input.payload:
            task = str(task_input.payload)

        cost_weight = 0.3
        cw_input = self.input(2)
        if cw_input and cw_input.payload is not None:
            cost_weight = float(cw_input.payload)

        speed_weight = 0.3
        sw_input = self.input(3)
        if sw_input and sw_input.payload is not None:
            speed_weight = float(sw_input.payload)

        quality_weight = 0.4
        qw_input = self.input(4)
        if qw_input and qw_input.payload is not None:
            quality_weight = float(qw_input.payload)

        min_quality = None
        mq_input = self.input(5)
        if mq_input and mq_input.payload is not None:
            min_quality = float(mq_input.payload)

        max_cost = None
        mc_input = self.input(6)
        if mc_input and mc_input.payload is not None:
            max_cost = float(mc_input.payload)

        complexity = self._estimate_complexity(task)

        # Score all models
        scored_models: List[Tuple[float, BaseProvider, str, Dict]] = []

        for provider in providers:
            model_name = ""
            if hasattr(provider, 'config'):
                model_name = provider.config.model
            elif isinstance(provider, dict):
                model_name = provider.get('model', '')

            model_info = get_model_info(model_name)
            if not model_info:
                continue

            # Check constraints
            cap = model_info.capabilities
            avg_cost = (cap.input_cost_per_million + cap.output_cost_per_million) / 2

            if min_quality:
                quality = self.MODEL_QUALITY_TIERS.get(model_name, 0.75)
                if quality < min_quality:
                    continue

            if max_cost and avg_cost > max_cost * 1000:  # Convert to per-M
                continue

            score, breakdown = self._score_model(
                model_info, cost_weight, speed_weight, quality_weight, complexity
            )
            scored_models.append((score, provider, model_name, breakdown))

        if not scored_models:
            self.set_output_val(2, Data("No models meet the specified constraints"))
            self.exec_output(4)
            return

        # Sort by score descending
        scored_models.sort(key=lambda x: x[0], reverse=True)
        best_score, best_provider, best_model, best_breakdown = scored_models[0]

        reasoning = (
            f"Selected {best_model} (score: {best_score:.2f}) based on:\n"
            f"  - Cost score: {best_breakdown['cost']:.2f}\n"
            f"  - Speed score: {best_breakdown['speed']:.2f}\n"
            f"  - Quality score: {best_breakdown['quality']:.2f}\n"
            f"  - Task complexity: {complexity:.2f}"
        )

        all_scores = [
            {'model': m, 'score': s, 'breakdown': b}
            for s, _, m, b in scored_models
        ]

        self.set_output_val(0, Data(best_provider))
        self.set_output_val(1, Data(best_model))
        self.set_output_val(2, Data(reasoning))
        self.set_output_val(3, Data(all_scores))
        self.exec_output(4)


class CostOptimizedRouterNode(Node):
    """
    Route to cheapest model that meets quality threshold.

    Simple router optimized for cost savings while
    maintaining minimum quality.

    Inputs:
        - providers: Available providers
        - min_quality: Minimum acceptable quality (0-1)
        - exec: Trigger

    Outputs:
        - provider: Selected provider
        - model: Selected model name
        - estimated_cost: Estimated cost per 1K tokens
        - done: Exec
    """

    title = 'Cost-Optimized Router'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='providers'),
        NodeInputType(label='min_quality', default=0.7),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='provider'),
        NodeOutputType(label='model'),
        NodeOutputType(label='estimated_cost'),
        NodeOutputType(type_='exec', label='done'),
    ]

    MODEL_QUALITY = {
        'gpt-4o': 0.95,
        'gpt-4o-mini': 0.85,
        'claude-sonnet-4-20250514': 0.93,
        'claude-3-5-haiku-20241022': 0.85,
        'llama3.2': 0.80,
        'mistral': 0.78,
    }

    def update_event(self, inp=-1):
        if inp != 2:
            return

        providers_input = self.input(0)
        if not providers_input or not providers_input.payload:
            return

        providers = providers_input.payload
        if not isinstance(providers, list):
            providers = [providers]

        min_quality = 0.7
        mq_input = self.input(1)
        if mq_input and mq_input.payload is not None:
            min_quality = float(mq_input.payload)

        # Find cheapest model meeting quality threshold
        candidates = []
        for provider in providers:
            model_name = ""
            if hasattr(provider, 'config'):
                model_name = provider.config.model

            quality = self.MODEL_QUALITY.get(model_name, 0.75)
            if quality < min_quality:
                continue

            model_info = get_model_info(model_name)
            if model_info:
                cost = (
                    model_info.capabilities.input_cost_per_million +
                    model_info.capabilities.output_cost_per_million
                ) / 2 / 1000  # Per 1K tokens
                candidates.append((cost, provider, model_name))

        if not candidates:
            self.set_output_val(0, Data(None))
            return

        candidates.sort(key=lambda x: x[0])
        cost, provider, model = candidates[0]

        self.set_output_val(0, Data(provider))
        self.set_output_val(1, Data(model))
        self.set_output_val(2, Data(cost))
        self.exec_output(3)


class SpeedOptimizedRouterNode(Node):
    """
    Route to fastest model that meets quality threshold.

    Prioritizes low latency for real-time applications.

    Inputs:
        - providers: Available providers
        - min_quality: Minimum acceptable quality (0-1)
        - prefer_local: Prefer local models when available
        - exec: Trigger

    Outputs:
        - provider: Selected provider
        - model: Selected model name
        - estimated_latency: Estimated latency category
        - done: Exec
    """

    title = 'Speed-Optimized Router'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='providers'),
        NodeInputType(label='min_quality', default=0.7),
        NodeInputType(label='prefer_local', default=True),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='provider'),
        NodeOutputType(label='model'),
        NodeOutputType(label='estimated_latency'),
        NodeOutputType(type_='exec', label='done'),
    ]

    # Latency estimates (lower is better)
    LATENCY_SCORES = {
        ProviderType.OLLAMA: 1,
        ProviderType.VLLM: 1,
        ProviderType.RAMALAMA: 1,
        ProviderType.LLAMACPP: 1,
        ProviderType.OPENAI: 3,
        ProviderType.ANTHROPIC: 3,
    }

    MODEL_QUALITY = {
        'gpt-4o-mini': 0.85,
        'claude-3-5-haiku-20241022': 0.85,
        'llama3.2': 0.80,
        'mistral': 0.78,
    }

    def update_event(self, inp=-1):
        if inp != 3:
            return

        providers_input = self.input(0)
        if not providers_input or not providers_input.payload:
            return

        providers = providers_input.payload
        if not isinstance(providers, list):
            providers = [providers]

        min_quality = 0.7
        mq_input = self.input(1)
        if mq_input and mq_input.payload is not None:
            min_quality = float(mq_input.payload)

        prefer_local = True
        pl_input = self.input(2)
        if pl_input and pl_input.payload is not None:
            prefer_local = bool(pl_input.payload)

        candidates = []
        for provider in providers:
            model_name = ""
            provider_type = ProviderType.CUSTOM

            if hasattr(provider, 'config'):
                model_name = provider.config.model
                provider_type = provider.config.provider_type

            quality = self.MODEL_QUALITY.get(model_name, 0.75)
            if quality < min_quality:
                continue

            latency_score = self.LATENCY_SCORES.get(provider_type, 5)

            # Boost local models if preferred
            if prefer_local and provider_type in (
                ProviderType.OLLAMA, ProviderType.VLLM,
                ProviderType.RAMALAMA, ProviderType.LLAMACPP
            ):
                latency_score -= 1

            candidates.append((latency_score, provider, model_name))

        if not candidates:
            self.set_output_val(0, Data(None))
            return

        candidates.sort(key=lambda x: x[0])
        latency, provider, model = candidates[0]

        latency_category = "fast" if latency <= 1 else "medium" if latency <= 3 else "slow"

        self.set_output_val(0, Data(provider))
        self.set_output_val(1, Data(model))
        self.set_output_val(2, Data(latency_category))
        self.exec_output(3)


class QualityRouterNode(Node):
    """
    Route to highest quality model within budget.

    Maximizes output quality while respecting cost limits.

    Inputs:
        - providers: Available providers
        - max_cost_per_1k: Maximum cost per 1K tokens
        - task_type: Type of task (reasoning, creative, code, etc.)
        - exec: Trigger

    Outputs:
        - provider: Selected provider
        - model: Selected model name
        - quality_score: Quality score of selected model
        - done: Exec
    """

    title = 'Quality Router'
    version = 'v0.1'
    init_inputs = [
        NodeInputType(label='providers'),
        NodeInputType(label='max_cost_per_1k'),
        NodeInputType(label='task_type'),
        NodeInputType(type_='exec', label='exec'),
    ]
    init_outputs = [
        NodeOutputType(label='provider'),
        NodeOutputType(label='model'),
        NodeOutputType(label='quality_score'),
        NodeOutputType(type_='exec', label='done'),
    ]

    # Quality by model and task type
    TASK_QUALITY = {
        'reasoning': {
            'o1': 0.99,
            'o1-mini': 0.95,
            'claude-opus-4-20250514': 0.96,
            'gpt-4o': 0.93,
        },
        'code': {
            'claude-sonnet-4-20250514': 0.95,
            'gpt-4o': 0.94,
            'claude-opus-4-20250514': 0.93,
        },
        'creative': {
            'claude-opus-4-20250514': 0.96,
            'gpt-4o': 0.94,
            'claude-sonnet-4-20250514': 0.92,
        },
        'general': {
            'gpt-4o': 0.95,
            'claude-sonnet-4-20250514': 0.93,
            'claude-opus-4-20250514': 0.97,
        },
    }

    def update_event(self, inp=-1):
        if inp != 3:
            return

        providers_input = self.input(0)
        if not providers_input or not providers_input.payload:
            return

        providers = providers_input.payload
        if not isinstance(providers, list):
            providers = [providers]

        max_cost = None
        mc_input = self.input(1)
        if mc_input and mc_input.payload is not None:
            max_cost = float(mc_input.payload)

        task_type = 'general'
        tt_input = self.input(2)
        if tt_input and tt_input.payload:
            task_type = str(tt_input.payload).lower()

        task_qualities = self.TASK_QUALITY.get(task_type, self.TASK_QUALITY['general'])

        candidates = []
        for provider in providers:
            model_name = ""
            if hasattr(provider, 'config'):
                model_name = provider.config.model

            model_info = get_model_info(model_name)
            if model_info and max_cost:
                cost = (
                    model_info.capabilities.input_cost_per_million +
                    model_info.capabilities.output_cost_per_million
                ) / 2 / 1000
                if cost > max_cost:
                    continue

            quality = task_qualities.get(model_name, 0.75)
            candidates.append((quality, provider, model_name))

        if not candidates:
            self.set_output_val(0, Data(None))
            return

        candidates.sort(key=lambda x: x[0], reverse=True)
        quality, provider, model = candidates[0]

        self.set_output_val(0, Data(provider))
        self.set_output_val(1, Data(model))
        self.set_output_val(2, Data(quality))
        self.exec_output(3)


# Export model router nodes
model_router_nodes = [
    ModelRouterNode,
    CostOptimizedRouterNode,
    SpeedOptimizedRouterNode,
    QualityRouterNode,
]
