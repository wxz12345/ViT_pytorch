import os
import argparse
import time
import torch

from vit_pytorch import ViT
from vit_pytorch.dataloader import get_dataloaders


def evaluate(args):
    # preferred device from args (default 'cuda') but fall back to cpu when unavailable
    if args.device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print('Warning: CUDA requested but not available, falling back to CPU')
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    print(f'Using device: {device}')

    # Load checkpoint first to recover model configuration if saved
    if not args.checkpoint or not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f'Loading checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_cfg = ckpt.get('model_cfg', None)

    # Require model_cfg in checkpoint to ensure model can be reconstructed exactly
    if ckpt_cfg is None:
        raise RuntimeError(
            "Checkpoint does not contain model_cfg.\n"
            "Please use a checkpoint saved by train.py (which includes model_cfg),\n"
            "or re-run training to produce a compatible checkpoint."
        )

    # Build model args from checkpoint
    model_args = {
        'image_size': ckpt_cfg['image_size'],
        'patch_size': ckpt_cfg['patch_size'],
        'dim': ckpt_cfg['dim'],
        'depth': ckpt_cfg['depth'],
        'heads': ckpt_cfg['heads'],
        'mlp_dim': ckpt_cfg['mlp_dim'],
        'dropout': ckpt_cfg.get('dropout', 0.0),
        'emb_dropout': ckpt_cfg.get('emb_dropout', 0.0),
    }

    # Now load test loader to infer num_classes (but prefer ckpt num_classes if present)
    _, test_loader, ds_num_classes = get_dataloaders(train_dir=None, test_dir=args.test_dir, batch_size=args.batch_size, image_size=model_args['image_size'], num_workers=args.num_workers)
    num_classes = ckpt.get('num_classes', ds_num_classes)

    model = ViT(
        image_size=model_args.get('image_size', args.image_size),
        patch_size=model_args.get('patch_size', args.patch_size),
        num_classes=num_classes,
        dim=model_args.get('dim', args.dim),
        depth=model_args.get('depth', args.depth),
        heads=model_args.get('heads', args.heads),
        mlp_dim=model_args.get('mlp_dim', args.mlp_dim),
        dropout=model_args.get('dropout', args.dropout),
        emb_dropout=model_args.get('emb_dropout', args.emb_dropout),
    ).to(device)

    # model summary
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model params: {param_count:,} (trainable: {trainable_params:,})')

    model.load_state_dict(ckpt['model'])
    model.eval()

    correct = 0
    total = 0
    per_class_correct = [0] * num_classes
    per_class_total = [0] * num_classes

    t_start = time.time()
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(test_loader, start=1):
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            for p, t in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                per_class_total[t] += 1
                if p == t:
                    per_class_correct[t] += 1
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if args.verbose and (batch_idx % args.print_freq == 0 or batch_idx == len(test_loader)):
                print(f"Eval: batch {batch_idx}/{len(test_loader)} - cumulative_acc: {correct/total:.4f} ({correct}/{total})")

    elapsed = time.time() - t_start
    acc = correct / total if total > 0 else 0.0
    print(f"Test accuracy: {acc:.4f} ({correct}/{total}) - time: {elapsed:.1f}s")
    print("Per-class accuracy:")
    for i in range(num_classes):
        tot = per_class_total[i]
        corr = per_class_correct[i]
        acc_c = corr / tot if tot > 0 else 0.0
        print(f"  class {i}: {acc_c:.3f} ({corr}/{tot})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dir', default='dataset/CIFAR10_balance', help='Path to test dataset (ImageFolder format)')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda', help="Preferred device: 'cuda' or 'cpu'")
    parser.add_argument('--verbose', action='store_true', help='Print per-batch evaluation progress')
    parser.add_argument('--print-freq', type=int, default=10, help='How many batches between verbose prints')

    args = parser.parse_args()
    evaluate(args)
