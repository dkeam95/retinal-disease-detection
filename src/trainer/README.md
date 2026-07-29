# Trainer Module

## Purpose

The `trainer` module implements the core training execution loop for the Retinal Disease Detection framework.

It orchestrates:

- Iterating over training and validation epochs;
- Forward/backward passes with Automatic Mixed Precision (AMP) support;
- Computing evaluation metrics across validation steps;
- Tracking mutable state (`TrainerState`);
- Integrating stateful checkpoint saving and loading;
- Returning final training summary statistics (`TrainingOutput`).

---

## Module Structure

```text
trainer/
├── __init__.py      # Public API exports
├── trainer.py       # Main Trainer class
├── step.py          # train_step & validation_step functions
├── state.py         # TrainerState tracking class
├── types.py         # StepOutput, EpochOutput, TrainingOutput dataclasses
├── exceptions.py     # TrainerError hierarchy
├── utils.py         # Helper utilities (moving tensors to device, loss accumulation)
└── README.md        # Technical documentation
```

---

## Core Types

### `StepOutput`
Stores statistics for a single batch iteration step.
- `loss`: `float`
- `batch_size`: `int`

### `EpochOutput`
Stores aggregated performance metrics for a completed epoch.
- `loss`: `float`
- `metrics`: `dict[str, float]`

### `TrainingOutput`
Stores the final summary of the complete training run.
- `best_epoch`: `int`
- `best_metric`: `float`
- `epochs_completed`: `int`

---

## Workflow Architecture

```text
               ┌────────────────────────────────┐
               │         Trainer.train()        │
               └───────────────┬────────────────┘
                               │
                For Epoch in 1..Total_Epochs
                               │
               ┌───────────────┴────────────────┐
               ▼                                ▼
       train_epoch()                    validate_epoch()
               │                                │
    For Batch in TrainLoader             For Batch in ValLoader
               │                                │
          train_step()                   validation_step()
               │                                │
    - Forward pass (AMP)             - Forward pass (eval)
    - Backward pass (scaler)         - Compute loss
    - Optimizer step                 - Collect predictions
               │                                │
               └───────────────┬────────────────┘
                               │
                               ▼
                    compute_metrics(QWK, F1)
                               │
                               ▼
                     CheckpointManager.save()
```

---

## Step Functions

### `train_step(...)`
Runs one optimization step for a batch:
1. Moves images and labels to the configured device (`cuda` or `cpu`);
2. Executes forward pass within `torch.cuda.amp.autocast` if AMP is enabled;
3. Calculates loss via the configured criterion;
4. Performs backward pass with gradient scaling;
5. Updates optimizer parameters and detaches loss.

### `validation_step(...)`
Runs evaluation on a single validation batch:
1. Disables gradient tracking (`torch.no_grad()`);
2. Executes forward pass;
3. Calculates validation loss and returns batch predictions/logits.

---

## Exceptions

The module defines dedicated, typed exceptions:

- `TrainerError`: Base exception for trainer failures.
- `InvalidTrainerStateError`: Raised on invalid state transitions.
- `TrainingStepError`: Raised during training batch execution failures.
- `ValidationStepError`: Raised during validation batch execution failures.

---

## Design Principles

- **Single Responsibility Principle**: Isolates training execution from dataset construction and model building.
- **Explicit State Tracking**: `TrainerState` immutably logs global steps, current epoch, and best metric scores.
- **Automatic Mixed Precision**: Native support for float16/bfloat16 training via PyTorch AMP.
- **Fail-Safe Checkpoints**: Seamless integration with `CheckpointManager` to guarantee state recoverability.
