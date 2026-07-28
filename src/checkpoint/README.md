# Checkpoint Module

## Purpose

The checkpoint module is responsible for saving and restoring the
training process.

Its primary goal is to make training fault tolerant and reproducible.

---

## Responsibilities

- Save model parameters.
- Save optimizer state.
- Save learning rate scheduler state.
- Save trainer state.
- Restore complete training state.
- Validate checkpoint files.
- Generate checkpoint filenames.
- Manage checkpoint directory.

---

## Components

### manager.py

Implements the `CheckpointManager`.

Responsibilities:

- Save checkpoints.
- Load checkpoints.
- Restore model state.
- Restore optimizer state.
- Restore scheduler state.
- Restore trainer state.

---

### types.py

Contains immutable dataclasses.

- `CheckpointMetadata`
- `CheckpointData`

---

### exceptions.py

Defines checkpoint-specific exceptions.

- `CheckpointError`
- `CheckpointNotFoundError`
- `InvalidCheckpointError`
- `CheckpointSaveError`
- `CheckpointLoadError`

---

### utils.py

Contains helper utilities.

Responsibilities:

- Build checkpoint filename.
- Validate checkpoint file.
- Find latest checkpoint.
- Create checkpoint directory.

---

## Typical Workflow

Trainer
↓

CheckpointManager.save()

↓

checkpoint.pt

Later...

CheckpointManager.load()

↓

Restore

- model
- optimizer
- scheduler
- trainer state

↓

Resume training

---

## Design Principles

- Single Responsibility Principle (SRP)
- Immutable checkpoint metadata
- Hardware-independent checkpoint loading
- Portable across CPU and GPU systems
- Fully testable
- No training logic inside the module

---

## Dependencies

Internal

- trainer.state

External

- torch
- pathlib

---

## Public API

CheckpointManager

Methods

- save()
- load()
