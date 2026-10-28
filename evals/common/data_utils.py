"""
Data loading utilities for evaluation tasks.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser


def load_jsonl_dataset(
    file_path: Path,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load data from JSONL file.
    
    Args:
        file_path: Path to JSONL file
        limit: Maximum number of samples to load
        
    Returns:
        List of data dictionaries
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    
    if limit is not None:
        data = data[:limit]
    
    return data


def create_inspect_sample(
    sample_data: Dict[str, Any],
    sample_id: str,
    input_prompt: str,
    target_answer: str,
    metadata: Optional[Dict] = None
) -> Sample:
    """
    Create an Inspect AI Sample from data.
    
    Args:
        sample_data: Raw sample data
        sample_id: Unique identifier for sample
        input_prompt: Input message for the agent
        target_answer: Expected answer for scoring
        metadata: Additional metadata
        
    Returns:
        Inspect AI Sample object
    """
    return Sample(
        input=[ChatMessageUser(role="user", content=input_prompt)],
        target=[target_answer],
        metadata={
            **(metadata or {}),
            **sample_data
        },
        id=sample_id
    )