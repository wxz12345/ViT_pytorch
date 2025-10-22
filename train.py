import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from collections import Counter
from torchvision import datasets, transforms

from vit_pytorch import ViT
from vit_pytorch.dataloader import get_dataloaders


def train(args):
    # preferred device from args (default 'cuda') but fall back to cpu when unavailable
    if args.device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
        else:
            print('Warning: CUDA requested but not available, falling back to CPU')
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    print(f'Using device: {device}')

    # Load train loader (no validation loader)
    train_loader, _, num_classes = get_dataloaders(train_dir=args.train_dir, test_dir=None, batch_size=args.batch_size, image_size=args.image_size, num_workers=args.num_workers)

    # build class counts from the ImageFolder so we can compute class-balanced weights
    # Note: we recreate ImageFolder without heavy transforms to only inspect labels
    try:
        inspect_ds = datasets.ImageFolder(args.train_dir, transform=transforms.Compose([transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()]))
        counts = Counter(inspect_ds.targets)
        class_counts = [counts.get(i, 0) for i in range(num_classes)]
    except Exception:
        # fallback: uniform counts if dataset can't be read
        class_counts = [1 for _ in range(num_classes)]

    # TensorBoard writer
    writer = SummaryWriter(args.log_dir)
    global_step = 0

    model = ViT(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=num_classes,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout,
    ).to(device)

    # model summary
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model created. params: {param_count:,} (trainable: {trainable_params:,})')

    # Use class-balanced focal loss by default to mitigate class imbalance
    # Implementation included below; it will be moved to device by .to(device)
    class ClassBalancedFocalLoss(nn.Module):
        def __init__(self, class_counts, beta=0.9999, gamma=2.0, reduction='mean'):
            super().__init__()
            counts = torch.tensor(class_counts, dtype=torch.float32)
            effective_num = 1.0 - torch.pow(beta, counts)
            weights = (1.0 - beta) / (effective_num + 1e-12)
            # normalize to number of classes
            weights = weights / weights.sum() * len(class_counts)
            # register as buffer so .to(device) moves it
            self.register_buffer('weights', weights)
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, logits, targets):
            prob = F.softmax(logits, dim=1)
            pt = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
            alpha = self.weights[targets]
            focal_factor = (1.0 - pt) ** self.gamma
            logpt = torch.log(pt + 1e-12)
            loss = - alpha * focal_factor * logpt
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss

    # choose criterion based on flag: class-balanced focal loss or standard CrossEntropy
    if getattr(args, 'use_class_balanced', False):
        criterion = ClassBalancedFocalLoss(class_counts, beta=args.cb_beta, gamma=args.focal_gamma).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt.get('optimizer', {}))
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Loaded checkpoint {args.checkpoint}, starting from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        t_epoch_start = time.time()
        for batch_idx, (imgs, labels) in enumerate(train_loader, start=1):
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            # log per-batch metrics
            batch_acc = (preds == labels).float().mean().item()
            writer.add_scalar('train/batch_loss', loss.item(), global_step)
            writer.add_scalar('train/batch_acc', batch_acc, global_step)
            global_step += 1

            if args.verbose and (batch_idx % args.print_freq == 0 or batch_idx == len(train_loader)):
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] - batch_loss: {loss.item():.4f} - batch_acc: {batch_acc:.4f}")

        epoch_time = time.time() - t_epoch_start
        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0
        current_lr = optimizer.param_groups[0].get('lr', args.lr)

        print(f"Epoch {epoch}/{args.epochs-1} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f} - lr: {current_lr:.2e} - time: {epoch_time:.1f}s")

        # log epoch metrics
        writer.add_scalar('train/loss', epoch_loss, epoch)
        writer.add_scalar('train/acc', epoch_acc, epoch)
        writer.add_scalar('train/lr', current_lr, epoch)

        # save checkpoint
        if args.checkpoint:
            # include model configuration and num_classes so evaluation can reconstruct
            model_cfg = {
                'image_size': args.image_size,
                'patch_size': args.patch_size,
                'dim': args.dim,
                'depth': args.depth,
                'heads': args.heads,
                'mlp_dim': args.mlp_dim,
                'dropout': args.dropout,
                'emb_dropout': args.emb_dropout,
                'pool': getattr(model, 'pool', 'cls'),
            }
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_cfg': model_cfg,
                'num_classes': num_classes,
            }
            torch.save(state, args.checkpoint)
            print(f"Saved checkpoint: {args.checkpoint}")

        # optional validation was removed; train.py focuses on training and checkpointing


def validate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total if total > 0 else 0.0
    print(f"Validation accuracy: {acc:.4f}")
    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', default='dataset/CIFAR10_imbalance', help='Path to training dataset (ImageFolder format)')
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
    parser.add_argument('--cb-beta', type=float, default=0.9999, help='beta for effective number class-balanced weights')
    parser.add_argument('--use-class-balanced', action='store_true', help='Enable class-balanced focal loss to mitigate class imbalance')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='gamma parameter for focal loss')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--log-dir', type=str, default='runs', help='TensorBoard log directory')
    args = parser.parse_args()
    train(args)
