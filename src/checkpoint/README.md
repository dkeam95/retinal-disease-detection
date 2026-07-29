# Checkpoint Module

## Purpose

The `checkpoint` module provides fault-tolerant serialization and deserialization of training state checkpoints.

It enables training process interruption, recovery, and reproduction by storing model parameters, optimizer states, learning rate scheduler states, and trainer metadata to disk.

---

## Responsibilities

- Serializing model weights, optimizer states, scheduler states, and epoch counts to `.pt` files.
- Restoring training state to CPU or GPU devices seamlessly.
- Managing best checkpoint preservation (`save_best`) based on evaluation metrics (e.g. `val_loss`, `qwk`).
- Validating checkpoint file integrity before loading.

---

## Checkpoint Data Contract

Checkpoints serialize a dictionary structure containing:

- `model_state_dict`: Model parameters.
- `optimizer_state_dict`: Optimizer momentum and moment states.
- `scheduler_state_dict`: Learning rate schedule step states (if applicable).
- `trainer_state`: `TrainerState` dataclass snapshot (epoch, step, best metric score).
- `metadata`: Hardware info, timestamp, config snapshot.

---

## Public API Usage

```python
from checkpoint import CheckpointManager

manager = CheckpointManager(
    directory=Path("checkpoints/baseline_exp"),
    monitor="qwk",
    mode="max",
)

# Save checkpoint during training loop
manager.save(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    trainer_state=state,
)

# Restore checkpoint to resume training
state_dict = manager.load("checkpoints/baseline_exp/best_checkpoint.pt")
```

---

## Module Structure

```text
checkpoint/
├── __init__.py      # Public API exports
├── manager.py       # CheckpointManager implementation
├── types.py         # CheckpointMetadata, CheckpointData dataclasses
├── utils.py         # File path formatting and file validation helpers
├── exceptions.py     # CheckpointError hierarchy
└── README.md        # Technical documentation
```

---

## Design Principles

- **Single Responsibility Principle**: Only manages state serialization and file recovery.
- **Hardware Agnostic**: Restores checkpoints safely across CPU, single CUDA GPU, or distributed setups.
- **Fail-Fast Validation**: Verifies state dictionary keys before parameter assignment to prevent silent corruption.
