from dataclasses import dataclass
from typing import Optional


@dataclass
class CheckpointState:
    """
    A data class which is used to hold important information about the checkpoint at the current training epoch.
    """
    callbacks_state: dict
    epoch: int
    epoch_state: dict
    model_state: Optional[dict]
    training_state: dict
