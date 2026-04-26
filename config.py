"""Глобальные параметры эксперимента и путей."""

import torch

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data/raw"
CHECKPOINT_DIR = "data/processed/checkpoints"
RESULTS_DIR = "data/processed/results"
GRID_STEPS = 51
RADIUS_DEFAULT = 1.0
RADIUS_LIST = [0.1, 0.25, 0.5, 1.0, 2.0]
EPOCHS = 30
BATCH_SIZE = 128
LR = 0.1
