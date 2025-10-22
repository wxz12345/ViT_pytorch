import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    parser = argparse.ArgumentParser(description='Run training: original or OvR')
    parser.add_argument('--ovr', action='store_true', help='Use One-vs-Rest training (per-class binary classifiers)')
    parser.add_argument('--train-dir', default='dataset/CIFAR10_imbalance')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--image-size', type=int, default=32)
    parser.add_argument('--patch-size', type=int, default=4)
    parser.add_argument('--dim', type=int, default=1024)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--heads', type=int, default=16)
    parser.add_argument('--mlp-dim', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--emb-dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    if args.ovr:
        from Agent.ovr_trainer import OvRTrainer, device_from_args
        dev = device_from_args(args)
        trainer = OvRTrainer(dev, args)
        trainer.train()
    else:
        # call original train.py
        from train import train
        train(args)


if __name__ == '__main__':
    main()
