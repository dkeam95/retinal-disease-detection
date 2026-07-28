"""
Checkpoint management module.

This module provides utilities for saving and restoring training checkpoints,
allowing model states, optimizer configurations, and trainer statistics to be persisted.
"""

# Import the main checkpoint controller class
from checkpoint.manager import CheckpointManager

# Explicitly define public module API
__all__ = [
    "CheckpointManager",
]