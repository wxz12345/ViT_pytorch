import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from vit_pytorch import ViT
from vit_pytorch.dataloader import get_dataloaders


class OvRTrainer:
    """One-vs-Rest trainer wrapper.

    This trains one binary classifier per class using the same ViT architecture.
    It saves each classifier checkpoint to disk. It keeps the original multiclass
    training code untouched; this is a wrapper that, when used, will perform OvR.
    """

    def __init__(self, device: torch.device, args):
        self.device = device
        self.args = args

    def _make_model(self, num_classes: int = 2):
        # build a ViT with binary output (2) or a single output for binary
        model = ViT(
            image_size=self.args.image_size,
            patch_size=self.args.patch_size,
            num_classes=num_classes,
            dim=self.args.dim,
            depth=self.args.depth,
            heads=self.args.heads,
            mlp_dim=self.args.mlp_dim,
            dropout=self.args.dropout,
            emb_dropout=self.args.emb_dropout,
        ).to(self.device)
        return model

    def train(self):
        # load full multiclass dataset
        train_loader, _, num_classes = get_dataloaders(train_dir=self.args.train_dir, test_dir=None, batch_size=self.args.batch_size, image_size=self.args.image_size, num_workers=self.args.num_workers)

        if train_loader is None:
            raise RuntimeError('train loader could not be created')

        # we need dataset to map class indices; recreate ImageFolder to access targets
        # get_dataloaders uses ImageFolder internally; we re-create one here
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.Resize((self.args.image_size, self.args.image_size)),
            transforms.ToTensor(),
        ])
        full_ds = datasets.ImageFolder(self.args.train_dir, transform=transform)
        if len(full_ds.classes) == 0:
            raise RuntimeError('No classes found in dataset')

        # For each class, create binary labels and train a classifier
        for cls_idx, cls_name in enumerate(full_ds.classes):
            print(f"Starting OvR training for class {cls_idx}: {cls_name}")

            # Create a simple dataset wrapper that maps labels to {target, rest}
            targets = torch.tensor([1 if t == cls_idx else 0 for t in full_ds.targets], dtype=torch.long)

            # Create DataLoader that yields same images but binary labels
            from torch.utils.data import TensorDataset, DataLoader
            # load whole dataset into memory tensors (small datasets like CIFAR10 ok). If too large, could implement streaming wrapper.
            imgs = []
            for img, _ in full_ds:
                imgs.append(img.unsqueeze(0))
            imgs = torch.cat(imgs, dim=0)

            dataset = TensorDataset(imgs, targets)
            loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)

            model = self._make_model(num_classes=2)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.args.lr)

            start_epoch = 0
            for epoch in range(start_epoch, self.args.epochs):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                t0 = time.time()

                for imgs_b, labels_b in loader:
                    imgs_b = imgs_b.to(self.device)
                    labels_b = labels_b.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(imgs_b)
                    loss = criterion(outputs, labels_b)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * imgs_b.size(0)
                    _, preds = outputs.max(1)
                    correct += (preds == labels_b).sum().item()
                    total += labels_b.size(0)

                epoch_loss = running_loss / total if total > 0 else 0.0
                epoch_acc = correct / total if total > 0 else 0.0
                print(f"[Cls {cls_idx}] Epoch {epoch}/{self.args.epochs-1} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f} - time: {time.time()-t0:.1f}s")

            # save checkpoint per-class
            ckpt_dir = os.path.dirname(self.args.checkpoint) if self.args.checkpoint else '.'
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"ovr_cls_{cls_idx}.pth")
            state = {
                'epoch': self.args.epochs - 1,
                'model': model.state_dict(),
                'model_cfg': {
                    'image_size': self.args.image_size,
                    'patch_size': self.args.patch_size,
                    'dim': self.args.dim,
                    'depth': self.args.depth,
                    'heads': self.args.heads,
                    'mlp_dim': self.args.mlp_dim,
                    'dropout': self.args.dropout,
                    'emb_dropout': self.args.emb_dropout,
                    'pool': getattr(model, 'pool', 'cls'),
                },
                'num_classes': 2,
                'ovr_target': cls_idx,
            }
            torch.save(state, ckpt_path)
            print(f"Saved OvR checkpoint: {ckpt_path}")


def device_from_args(args) -> torch.device:
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device('cuda')
    return torch.device('cpu')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', default='dataset/CIFAR10_imbalance')
    parser.add_argument('--epochs', type=int, default=2)
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
    dev = device_from_args(args)
    trainer = OvRTrainer(dev, args)
    trainer.train()
