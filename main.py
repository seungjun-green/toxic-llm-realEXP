# main.py

import argparse
import yaml
from scripts.train import train_from_config

def parse_args():
    parser = argparse.ArgumentParser(description="Run PPO Trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to config file (YAML)")
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.total_steps is not None:
        config["training"]["total_steps"] = args.total_steps
    if args.lr is not None:
        config["training"]["lr"] = args.lr

    trainer, steps = train_from_config(config)
    trainer.train(steps)

if __name__ == "__main__":
    main()