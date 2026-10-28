"""
Unit tests for cross-model evaluation support.

These tests verify that solvers correctly instantiate separate models
for attacker and victim agents, enabling proper multi-model evaluation.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestCrossModelInstantiation:
    """Test that solvers instantiate separate models for each agent."""

    def test_gdpr_solver_instantiates_model_b(self):
        """GDPR solver should call get_model() for victim."""
        with patch('evals.gdpr.solver.get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_get_model.return_value = mock_model

            from evals.gdpr.solver import gdpr_solver

            # Create solver with different models
            solver = gdpr_solver(
                model_a="openai/gpt-4",
                model_b="anthropic/claude-3-5-sonnet-latest"
            )

            # Solver instantiates both sides; verify the victim (and exploiter)
            # models were each requested.
            mock_get_model.assert_any_call("anthropic/claude-3-5-sonnet-latest")
            mock_get_model.assert_any_call("openai/gpt-4")

class TestModelGenerateWithRetry:
    """Test the model_generate_with_retry function."""

    def test_model_generate_with_retry_exists(self):
        """Verify model_generate_with_retry is available."""
        from evals.common.retry_utils import model_generate_with_retry
        assert callable(model_generate_with_retry)

    @pytest.mark.asyncio
    async def test_model_generate_with_retry_success(self):
        """Test successful generation."""
        from evals.common.retry_utils import model_generate_with_retry
        from inspect_ai.model import GenerateConfig

        # Create mock model
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.completion = "Test response"
        mock_model.generate = MagicMock(return_value=mock_output)

        # Make generate return awaitable
        async def mock_generate(*args, **kwargs):
            return mock_output

        mock_model.generate = mock_generate

        # Create mock messages
        messages = [MagicMock()]
        config = GenerateConfig(max_tokens=100)

        # Call function
        result = await model_generate_with_retry(mock_model, messages, config)

        # Verify result
        assert result.completion == "Test response"


