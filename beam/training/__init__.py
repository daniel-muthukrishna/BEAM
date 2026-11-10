"""
Training functionality for BEAM.

This module provides training loops, distributed training utilities,
and monitoring components.
"""

from beam.training.trainer import SMTrainer
from beam.training.distributed import (
    setup_distributed,
    cleanup_distributed,
    run_distributed,
    all_reduce_dict,
    broadcast_value
)

__all__ = [
    'SMTrainer',
    'setup_distributed',
    'cleanup_distributed',
    'run_distributed',
    'all_reduce_dict',
    'broadcast_value'
]