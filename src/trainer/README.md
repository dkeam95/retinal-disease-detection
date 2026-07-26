# Trainer Module

## Purpose

The `trainer` module implements the main training loop for the project.

It is responsible for:

- running training epochs;
- running validation epochs;
- computing evaluation metrics;
- tracking training state;
- saving and loading checkpoints;
- returning final training summary statistics.

---

## Architecture

```text
trainer/
│
├── __init__.py
├── types.py
├── exceptions.py
├── state.py
├── utils.py
├── step.py
├── trainer.py
└── README.md
Core Types
StepOutput

Stores statistics for a single batch step.

Fields:

loss
batch_size
EpochOutput

Stores aggregated statistics for a completed epoch.

Fields:

loss
metrics
TrainingOutput

Stores the final training summary.

Fields:

best_epoch
best_metric
epochs_completed
Exceptions

The module defines trainer-specific exceptions:

TrainerError
InvalidTrainerStateError
TrainingStepError
ValidationStepError
CheckpointError
EarlyStoppingError
State Management

TrainerState stores mutable training progress, including:

current epoch;
global step;
best metric;
best epoch;
loss history.
Utility Functions

The module provides helper functions for common training tasks:

moving tensors to device;
switching model mode;
detaching loss tensors;
calculating average loss.
Step Functions
train_step(...)

Runs one optimization step:

forward pass;
loss computation;
backward pass;
optimizer update.
validation_step(...)

Runs one validation step:

forward pass;
loss computation;
returns logits for metric computation.
Main Trainer
Trainer

The main class orchestrating the full training process.

Responsibilities:

iterate over training batches;
iterate over validation batches;
compute metrics after validation;
track best validation score;
save and load checkpoints;
expose current trainer state.
Training Flow

The training loop follows this sequence:

set model to training mode;
run training batches;
set model to evaluation mode;
run validation batches;
compute metrics;
update best state;
step scheduler if available;
return final training summary.
Checkpointing

The trainer supports:

saving model state;
saving optimizer state;
saving scheduler state if present;
restoring trainer state from disk.
Testing

Covered by unit tests:

type definitions;
exceptions;
state transitions;
utility functions;
train step;
validation step;
trainer integration behavior.
Design Principles
Single Responsibility Principle
Modular design
Explicit state tracking
Reusable step functions
Clear separation between train and validation logic
Checkpoint support
```
