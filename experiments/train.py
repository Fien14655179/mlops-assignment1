import sys
from pathlib import Path
import argparse
import yaml

import torch
import torch.optim as optim

# Zorg dat `src/` gevonden wordt
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ml_core.data.loader import get_dataloaders
from src.ml_core.models.mlp import MLP
from src.ml_core.solver.trainer import Trainer


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    # FORCE CPU (stabiel)
    device = "cpu"
    print(f"Using device: {device}")

    # DataLoaders (Trainer.fit verwacht die)
    train_loader, val_loader = get_dataloaders(config)

    # Model
    mcfg = config["model"]
    model = MLP(
        input_shape=(3, 96, 96),
        hidden_units=mcfg["hidden_dims"],
        num_classes=mcfg["num_classes"],
    )

    optimizer = optim.AdamW(model.parameters(), lr=float(config["training"]["lr"]))

    trainer = Trainer(model=model, optimizer=optimizer, config=config, device=device)

    # Fit (epochs meestal uit config; als jouw fit epochs wil, zie note hieronder)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
