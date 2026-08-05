# Trainer Module (`src/trainer/`)

## Overview

The `trainer` module orchestrates the training, validation, and testing execution pipelines. It manages epoch loops, learning rate decay schedules, gradient backpropagation with Automatic Mixed Precision (AMP), early stopping, and metric checkpoint evaluations.

## Key Features

- **Automatic Mixed Precision (AMP)**: Optimizes GPU memory and computation speed using FP16 operations safely.
- **State Serialization**: Saves complete state (epochs, steps, best metric values, metrics history) to resume training runs cleanly.
- **Gradient Accumulation**: Supports virtual batch scaling by accumulating gradients over multiple sub-steps.

## API & Interfaces

### Classes

#### `Trainer`
Orchestrator of optimization loops.
- **`__init__(model, optimizer, scheduler, loss_fn, metrics, config, state=None)`**
- **`fit(train_loader: DataLoader, val_loader: DataLoader) -> TrainingOutput`**
  Executes training epochs, periodically validating, tracking performance metrics, and triggerring early stopping.
- **`test(test_loader: DataLoader) -> EpochOutput`**
  Evaluates model performance on the test split.

#### `TrainerState`
State tracking container:
- `epoch`: int
- `step`: int
- `best_metric_value`: float
- `metrics_history`: list

### Functions

- **`train_step(model, batch, optimizer, loss_fn, scaler, device) -> StepOutput`**
  Executes single forward pass, computes loss, scales gradients, and executes optimizer step.
- **`validation_step(model, batch, loss_fn, device) -> StepOutput`**
  Performs inference step on a validation batch without tracking gradients.

### Exceptions

- `InvalidTrainerStateError`: Raised if state histories contain NaN values or incorrect shapes.
- `TrainingStepError`: Raised if forward/backward passes raise runtime exceptions (e.g. out of memory).

## Dependencies

- `torch`: CUDA support, `autocast`, gradients scaling, and module parameters optimization.
